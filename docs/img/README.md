# Presentation images

All generated; regenerate rather than edit. Figures read the committed benchmark run only
(no reruns); 3D renders use the showcase scenario, chosen from DEV seeds by a fixed rule.

| File | What it shows | Command | Data source |
|---|---|---|---|
| `headline.png` | Paired TerraNav − Baseline-Occ and TerraNav − Oracle differences (rough exposure [m²], max rise [cm], length [m]) per family at α = 4, 95% bootstrap CIs; every value checked against summary.md (converted back to its units) | `uv run python scripts/make_figures.py results/0cf7be5_96a4e7ffad40` | `results/0cf7be5_96a4e7ffad40/raw.jsonl` (held-out seeds 1000–1019) |
| `tradeoff_rough.png` | Paired extra length (%) vs reduction in rough exposure against Baseline-Occ over α, TerraNav (α-labelled, pre-registered α = 4 circled) and Oracle | same command | same run |
| `tradeoff_mixed.png` | As above, mixed family (rough exposure) | same command | same run |
| `tradeoff_slopes.png` | As above, slopes family (reduction in max rise) | same command | same run |
| `tradeoff_steps.png` | As above, steps family (reduction in max rise) | same command | same run |
| `showcase.json` | The showcase choice: rule, full ranking of dev seeds, evaluator metrics of the chosen scenario | `uv run python scripts/pick_showcase.py` | dev seeds 0–9, rough family (benchmark job at α = 4) |
| `hero.png` | 1920×1080 oblique 3D view: terrain coloured by GT traversability cost (0 to the scene's maximum), TerraNav / Baseline-Occ / Oracle (dashed) paths with rough-patch stretches brighter and thicker, green start, gold goal, evaluator length and rough exposure | `uv run python scripts/render_hero.py` | showcase: rough, dev seed 4 |
| `replay.gif` | 19.6 s replay (10 fps, 800 px): scanning, mapping, planning, executing with rough-exposure counters, final totals; a replay of the scan-once, plan-once pipeline, not a live simulation | `uv run python scripts/make_replay.py` | showcase: rough, dev seed 4 |

`scripts/make_replay.py` also writes `out/replay.mp4` (1280×720, H.264 via imageio-ffmpeg
0.6.0; `out/` is not committed).
