# Cell conversion (linearize / simplexify / elevate)

`meshioplusplus.convert_cells(mesh, mode=...)` converts a mesh's **element representation** — which cell *types* it is built from — while leaving the object it describes intact. It is a mesh **operation** (like [crop](/crop) and [split](/split)), not a file format, and uses only standard C++/numpy, so it runs under every mesh backend.

![Each hexahedron split into 6 tetrahedra, coloured by child index](/images/convert_cells_simplexify.png)

```python
import meshioplusplus

mesh = meshioplusplus.read("domain.msh")

# drop the higher-order nodes: tetra10 -> tetra, hexahedron27 -> hexahedron, ...
linear = meshioplusplus.convert_cells(mesh, mode="linearize")

# decompose into simplices: hexahedron -> 6 tetra, quad -> 2 triangles, ...
tets = meshioplusplus.convert_cells(mesh, mode="simplexify")

# promote to serendipity quadratic: triangle -> triangle6, tetra -> tetra10, ...
quadratic = meshioplusplus.convert_cells(mesh, mode="elevate")

meshioplusplus.write("tets.vtu", tets)
```

## Modes

| mode | does | cells | points |
|---|---|---|---|
| `"linearize"` | every higher-order cell becomes its linear base, keeping the corner connectivity verbatim | unchanged | orphaned mid nodes **pruned** |
| `"simplexify"` | every cell is decomposed into simplices of the **same topological dimension** | **grows** | unchanged (children reuse the parent's corner nodes) |
| `"elevate"` | every linear cell is promoted to its **serendipity** (edge-only) quadratic counterpart | unchanged | **grows** (one new node per unique edge) |

Type mappings:

| linear | quadratic (`elevate` ⇄ `linearize`) | simplex decomposition (`simplexify`) |
|---|---|---|
| `line` | `line3` | — (already a simplex) |
| `triangle` | `triangle6` | — (already a simplex) |
| `quad` | `quad8` | 2 × `triangle` |
| `tetra` | `tetra10` | — (already a simplex) |
| `pyramid` | `pyramid13` | 2 × `tetra` |
| `wedge` | `wedge15` | 3 × `tetra` |
| `hexahedron` | `hexahedron20` | 6 × `tetra` |
| `polygon(n)` | — | (n − 2) × `triangle` (fan around node 0) |

`linearize` additionally accepts the full-Lagrange types (`quad9`, `hexahedron27`, `wedge18`, `pyramid14`, and the higher `*20`/`*64` families), mapping each to its linear base.

All three modes are **idempotent on cells they do not apply to** — a linear block passes through `linearize` unchanged, an already-simplex block through `simplexify`, an already-quadratic block through `elevate` — so they are safe on a mixed-order mesh. Under `simplexify`, higher-order input is linearized first.

Two constructs raise rather than guess:

- a **polyhedron** block under `simplexify` (a 2-level ragged block has no simplex template);
- the **full-Lagrange** targets `quad9`/`hexahedron27` under `elevate` — they need face- and body-centre nodes, which this version deliberately does not produce.

## What changes

- **Block structure is preserved 1:1** in every mode: the output has exactly as many cell blocks as the input, in the same order. That is what keeps the `cell_data` correspondence (one array per block) correct under every backend.
- **`linearize`**: corner connectivity verbatim; unreferenced points pruned and connectivity/`point_data` remapped. Cell count unchanged, so `cell_data` and `field_data` pass through.
- **`simplexify`**: points untouched; each parent's `cell_data` row is **replicated** to its children; `point_data`/`field_data` pass through.
- **`elevate`**: new nodes are appended after the originals, with coordinates at the edge midpoint and `point_data` set to the **mean of the edge's endpoints**. Cell count unchanged, so `cell_data`/`field_data` pass through.
- **`point_sets` / `cell_sets`** are remapped in the Python layer. Under `simplexify` a cell-set entry **expands to the parent's children**.
- `record_parent_ids=True` attaches an Int64 `convert:parent_cell` `cell_data` array recording, per output cell, the index of the input cell it came from within its own block.

Output is **deterministic**: the decomposition templates are fixed, and the `elevate` mid-edge numbering is assigned by a serial pass over a parallel-filled buffer, never a concurrent hash insert. Results are byte-identical across the MESHIO/NATIVE/KRATOS backends and any thread count.

Every emitted simplex is **positively oriented** for a well-oriented input, and the decomposition **conserves volume** — the hexahedron template is a canonical Freudenthal fan around the main diagonal 0–6. (It is conforming across a shared face only when neighbouring hexahedra agree on that diagonal, which is why the diagonal is fixed rather than chosen per cell.)

The numpy fallback handles rectangular cell blocks; ragged/polyhedron blocks are handled by the C++ core only.

`simplexify` and `elevate` linearize/decompose a curved cell's geometry — they keep only the corner nodes, and neither carries any provenance back from a synthetic simplex to the higher-order cell it came from. [`tessellate`](/tessellation) is the curvature-aware alternative for the five higher-order types it covers (`triangle6`/`quad8`/`quad9`/`tetra10`/`hexahedron27`): it subdivides onto a refinement lattice using each cell's own isoparametric shape functions (so a curved boundary tessellates onto the curve rather than being chopped straight), interpolates fields on the way in, and records a full point/cell provenance map that `Tessellation.gather`/`.scatter`/`.aggregate` use to move data back onto the original cell — the missing half of `convert_cells(simplexify)`'s own conformity guarantee.

## CLI

```bash
meshioplusplus convert-cells in.msh out.vtu --mode linearize
meshioplusplus convert-cells in.msh out.vtu --mode simplexify
meshioplusplus convert-cells in.msh out.vtu --mode elevate
meshioplusplus convert-cells in.msh out.vtu --mode simplexify --record-parent-ids
```

Note this is distinct from the format-conversion [`convert`](/cli) verb. See the [CLI reference](/cli).

## Other languages

- **C API** — `mio_convert_cells(mesh, mode, record_parent_ids)` returns an opaque `mio_convert_cells_result` (`_mesh` borrow, `_take_mesh`, zero-copy `_point_map` / `_cell_map`, `_free`). See the [C API reference](/c_api).
- **Fortran** — `mesh%convert_cells(mode, record_parent_ids=..., point_map=...)`. See the [Fortran reference](/fortran).
- **WebAssembly / JavaScript** — `convertCells(mesh, mode, recordParentIds)`. See the [WebAssembly reference](/wasm).
