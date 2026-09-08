# Affine transform

`meshioplusplus.transform(mesh, ...)` applies a geometric transform to a mesh's point coordinates — translate, scale (per-axis or uniform), rotate (axis + angle), a general 4×4 affine matrix, or a unit-scale factor. Only the points change; connectivity, `cell_data`, `field_data`, and (by default) `point_data` are carried through. It is a mesh **operation** (like [reordering](/reorder) and [quality metrics](/mesh_quality)), not a file format, and uses only standard C++/numpy, so it runs under every mesh backend.

![A surface mesh rotated 45° about x, coloured by its original height](/images/transform_rotate.png)

```python
import meshioplusplus

mesh = meshioplusplus.read("part.vtu")

t = meshioplusplus.transform(mesh, translate=[1, 2, 3])
s = meshioplusplus.transform(mesh, scale=[2, 2, 1])          # per-axis
u = meshioplusplus.transform(mesh, scale=0.5)                # uniform
r = meshioplusplus.transform(mesh, rotate=("z", 90))         # 90° about z
m = meshioplusplus.transform(mesh, matrix=my_4x4)            # general affine
mm = meshioplusplus.transform(mesh, scale_units=0.001)       # mm -> m

meshioplusplus.write("part_moved.vtu", t)
```

## Parameters

| parameter | meaning |
|---|---|
| `translate` | a 3-vector `(dx, dy, dz)` (applied last when several are combined) |
| `scale` | a scalar (uniform) or 3-vector `(sx, sy, sz)` about the origin |
| `rotate` | `(axis, angle_deg)` where `axis` is `"x"`/`"y"`/`"z"` or a 3-vector; angle in **degrees** |
| `matrix` | a full 4×4 (or flat 16-element) row-major affine matrix; overrides the others |
| `scale_units` | a scalar unit-conversion factor |
| `rotate_vector_data` | when true, rotate vector (dim 3) / tensor (dim 9) `point_data` **and `cell_data`** by the linear part; off by default |

Several builders compose (in the fixed order units → scale → rotate → translate). An orientation-reversing transform (negative determinant of the 3×3 linear block) logs a warning that cell orientation may be flipped.

## What changes

- **Points** are transformed by `p' = M · [x, y, z, 1]`.
- **Connectivity**, **`cell_data`**, and **`field_data`** are unchanged.
- **`point_data` and `cell_data`** are copied unchanged unless `rotate_vector_data=True`, which rotates float arrays whose trailing dimension is 3 (`R·v`) or 9 (`R·A·Rᵀ`) at either location. A vector living on a cell rotates exactly as one living on a point does; before v10.33.0 the flag reached point data only, and cell data rode through in the old frame — silently, since nothing about an array says which frame it is in. An integer array is never rotated, so a material tag is safe.
- **`point_sets` / `cell_sets`** pass through unchanged (indices are stable).

## CLI

```bash
meshioplusplus transform in.vtu out.vtu --translate 1,2,3
meshioplusplus transform in.vtu out.vtu --scale 2,2,1
meshioplusplus transform in.vtu out.vtu --rotate z,90
meshioplusplus transform in.vtu out.vtu --matrix m00,...,m33
meshioplusplus transform in.vtu out.vtu --scale-units 0.001
```

Give exactly one transform source. Values that begin with `-` (e.g. a negative translation) need the `--translate=-1,0,0` form so argparse doesn't read them as a flag. See the [CLI reference](/cli).

## Other languages

- **C API** — `mio_transform(mesh, matrix[16], rotate_vector_data)`. See the [C API reference](/c_api).
- **Fortran** — `mesh%transform(matrix, rotate_vector_data=...)`. See the [Fortran reference](/fortran).
- **WebAssembly / JavaScript** — `transform(mesh, matrix, rotateVectorData)`. See the [WebAssembly reference](/wasm).
