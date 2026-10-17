# Frame conventions

Authoritative reference for coordinate frames in TerraNav. Code: `src/terranav/frames.py`,
`src/terranav/types.py`. Tests: `tests/test_frames.py`. Angles are radians in every
function signature; degrees appear only in config defaults (converted once, in
`types.py`).

## World frame

- Right-handed: **x forward, y left, z up**, metres.
- Origin at the map corner; the default world covers x, y ∈ [0, 12] m.
- Heightfield: `(ny, nx)` float64 array indexed **`[row=iy, col=ix]`**, cell size 0.05 m.
  Cell `(iy, ix)` has its centre at `origin + ((ix+0.5)·res, (iy+0.5)·res)`;
  `GridSpec.world_to_cell(x, y) -> (iy, ix)` and `GridSpec.cell_to_world(iy, ix) -> (x, y)`.
  `scipy.ndimage.map_coordinates` takes `(row, col)` — convert metres↔cells explicitly.
- Map grid: 0.1 m (120×120); unknown cells are treated as blocked.

## Camera optical frame

- **x right, y down, z forward** (OpenCV convention). Pixel **u right, v down**.

```
  WORLD (top view, +z up out of page)     CAMERA optical (viewed from behind)

        +y (left)                            o------> +x (right)   pixel u →
         ^                                   |
         |                                   |
         |                                   v
         o------> +x (forward)              +y (down)              pixel v ↓

                                             +z (forward) points into the page
```

## Camera orientation

```
R_base = [[0,0,1],[-1,0,0],[0,-1,0]]          # optical → world at yaw=0, pitch=0
R_wc   = Rz(yaw) @ Ry(pitch_down) @ R_base
Ry(t)  = [[c,0,s],[0,1,0],[-s,0,c]]           Rz(t) = [[c,-s,0],[s,c,0],[0,0,1]]
```

The columns of `R_base` say where each optical axis points in world at rest:
x_c → −y_w (right), y_c → −z_w (down), z_c → +x_w (forward).
`yaw` rotates about world +z (yaw = π/2 looks along +y); positive `pitch_down` tilts the
optical axis toward −z. `T_world_cam = [[R_wc, p], [0,0,0,1]]` maps **camera-frame points
to world** (`p_w = R_wc p_c + p`), where `p` is the optical centre in world.

### Worked example: yaw = 0, pitch_down = 25°

Optical axis in camera is `z_c = [0,0,1]`.

1. `R_base z_c = [1, 0, 0]` (third column of `R_base`).
2. `Ry(25°) [1,0,0] = [cos 25°, 0, −sin 25°]` (first column of `Ry`).
3. `Rz(0)` is the identity.

Result: optical axis ≈ **[0.906, 0, −0.423]** in world — forward and 25° below the
horizon. In general the axis is `[cos p · cos y, cos p · sin y, −sin p]`.

## Intrinsics and pixel centres

```
fx = fy = (W/2) / tan(hfov/2)        cx = (W−1)/2        cy = (H−1)/2
d_cam(u, v) = [(u−cx)/fx, (v−cy)/fy, 1]
```

Default 320×240, hfov 87° → fx ≈ 168.6, cx = 159.5, cy = 119.5.

**Pixel centres are at integer indices.** Pixel `u` covers `[u−0.5, u+0.5]`, so the
image spans `[−0.5, W−0.5]` and its geometric centre is `(W−1)/2`. Using `W/2` instead
would shift every ray by half a pixel and make the image asymmetric about the optical
axis. With an even width the axis falls exactly between pixels 159 and 160, whose rays
are ±0.5/fx. The `fx` formula puts the *outer edges* of the edge pixels (not their
centres) at ±hfov/2.

`pixel_rays_cam` returns `(H, W, 3)` rays indexed `[v, u]`, normalised so **z = 1**
(not unit length).

## depth_z vs range_t

- `depth_z`: the point's **z coordinate in the camera frame** (distance along the optical
  axis). This is what `DepthFrame.depth_z` stores (float32, NaN = invalid, never 0).
- `range_t`: **Euclidean distance** from the optical centre along the ray (the ray
  parameter of a ray-caster using a unit direction).

Because `d_cam` has z = 1, the camera-frame point is `p_c = depth_z · d_cam`, so

```
range_t = depth_z · ||d_cam||        depth_z = range_t / ||d_cam||
```

They agree only on the optical axis; toward image corners `||d_cam|| > 1` and
`range_t > depth_z`. Variables are always named `depth_z` or `range_t`, never mixed.
