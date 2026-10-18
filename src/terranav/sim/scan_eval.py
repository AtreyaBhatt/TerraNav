"""Scan-coverage evaluation jobs, runnable serially or in worker processes.

A job is (family, seed, strategy). Each job builds its scenario, scan poses and per-frame
generators np.random.default_rng((seed, frame_index)) itself, so results do not depend on
how jobs are distributed: a parallel run returns exactly the serial run's numbers.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np

from terranav.config import SimConfig
from terranav.perception.backproject import backproject
from terranav.sim.coverage import corridor_mask, coverage, reachable_mask
from terranav.sim.scenario import DEFAULT_SCAN_K, generate_scenario, scan_poses
from terranav.sim.sensor import capture
from terranav.types import SensorConfig

Job = tuple[str, int, str]  # (family, seed, strategy)
# Environment variables that cap BLAS/OpenMP thread pools in worker processes.
SINGLE_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def limit_threads() -> None:
    """Cap native thread pools at one thread (workers already run one per core).

    Takes effect for libraries that read these variables when they load, i.e. in processes
    that set them before importing numpy (as scripts/eval_scan.py does at the top). Also used
    as the pool initializer for start methods that import numpy afresh in each worker.
    """
    for name in SINGLE_THREAD_ENV:
        os.environ[name] = "1"


def evaluate_job(job: Job, k: int | None = None, sensor: SensorConfig | None = None) -> dict:
    """Capture every scan pose of one job and measure corridor / reachable coverage."""
    family, seed, strategy = job
    cfg = SimConfig()
    sensor = sensor if sensor is not None else cfg.sensor
    sc = generate_scenario(family, seed, cfg)
    k = DEFAULT_SCAN_K[strategy] if k is None else k
    poses = scan_poses(sc, k=k, strategy=strategy, sensor=sensor)
    points = []
    for i, T in enumerate(poses):
        frame, _ = capture(sc.world, T, sensor, np.random.default_rng((seed, i)))
        points.append(backproject(frame)[0])
    return {
        "family": family,
        "seed": seed,
        "strategy": strategy,
        "skipped": k - len(poses),
        "corridor": coverage(points, cfg.map_grid, corridor_mask(sc, cfg.map_grid)),
        "reachable": coverage(points, cfg.map_grid, reachable_mask(sc, cfg.map_grid)),
    }


def run_parallel(
    fn: Callable[[Job], dict],
    jobs: Sequence[Job],
    workers: int,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Apply a picklable fn to every job; results in input order. workers == 1 is serial.

    Jobs must be self-contained (own scenario, own per-frame generators), so the result does
    not depend on workers. progress(done, total) is called after each finished job, if given.
    """
    results: list[dict | None] = [None] * len(jobs)
    if workers == 1:
        for i, job in enumerate(jobs):
            results[i] = fn(job)
            if progress:
                progress(i + 1, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=limit_threads) as pool:
            futures = {pool.submit(fn, job): i for i, job in enumerate(jobs)}
            for done, fut in enumerate(as_completed(futures), start=1):
                results[futures[fut]] = fut.result()
                if progress:
                    progress(done, len(jobs))
    return results  # type: ignore[return-value]


def run_jobs(
    jobs: Sequence[Job],
    workers: int,
    k: int | None = None,
    sensor: SensorConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Scan-coverage evaluation of jobs (see evaluate_job) via run_parallel."""
    return run_parallel(partial(evaluate_job, k=k, sensor=sensor), jobs, workers, progress)


def git_revision() -> tuple[str, bool]:
    """(HEAD commit hash, dirty?) of the current directory; ("unknown", False) outside git.

    Dirty = any tracked change OR any untracked, non-ignored file (a new source file changes
    results too). Generated outputs live in the gitignored out/.
    """
    import subprocess

    def run(*cmd: str) -> str:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()

    try:
        rev = run("git", "rev-parse", "HEAD")
        return rev, bool(run("git", "status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def revision_label(allow_dirty: bool) -> str:
    """Commit hash for result headers. Exits with an error on a dirty tree unless allow_dirty,
    in which case it warns and returns "<hash>-dirty"."""
    rev, dirty = git_revision()
    print(f"git {rev}")
    if not dirty:
        return rev
    if not allow_dirty:
        raise SystemExit(
            "error: working tree has uncommitted or untracked changes; results would not be "
            "tied to a commit. Commit first, or pass --allow-dirty."
        )
    print("WARNING: --allow-dirty: results are labelled with a -dirty hash")
    return rev + "-dirty"
