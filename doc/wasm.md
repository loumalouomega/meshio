# WebAssembly / JavaScript

The C++ core also compiles to WebAssembly and ships as an npm package, [`@meshioplusplus/wasm`](https://www.npmjs.com/package/@meshioplusplus/wasm), for reading and writing meshes in the browser or Node.js. (It is one of two "flat" bindings over the same core and shared format-dispatch registry — the other is the [C API](/c_api).)

## Install

```sh
npm install @meshioplusplus/wasm
```

## Usage

```js
import { loadMeshioPlusPlus } from "@meshioplusplus/wasm";

const meshio = await loadMeshioPlusPlus();

// Write bytes into the Emscripten virtual filesystem, then read them as a mesh.
const response = await fetch("example.vtu");
meshio.FS.writeFile("/example.vtu", new Uint8Array(await response.arrayBuffer()));
const mesh = meshio.readMesh("/example.vtu");

console.log(mesh.points);          // Float64Array, flat (numPoints * dim)
console.log(mesh.cells[0].type);   // e.g. "triangle"
console.log(mesh.cells[0].data);   // Int32Array connectivity, flat (numCells * nodesPerCell)

// Convert directly (no intermediate JS object), or round-trip through one.
meshio.convert("/example.vtu", "/example.stl");
meshio.writeMesh("/example.msh", mesh, "gmsh");

// What this build can actually read and write, both sorted.
const { readers, writers } = meshio.availableFormats();
```

### Rendering a mesh

`convertSurface` produces something a surface renderer can draw: a mesh with 3D cells becomes its boundary, anything else passes through, and the result is linearized (a triangle renderer has no mid-side nodes, so `triangle6` connectivity drawn verbatim is visible garbage). Boundary facets inherit their owning cell's data, so a per-cell material or tag still colours correctly.

```js
meshio.convertSurface("/solid.msh", "/solid.vtp");
const vtp = meshio.FS.readFile("/solid.vtp");   // hand this to vtk.js
```

Prefer it over `readMesh` → `extractSkin` → `writeMesh` for anything headed to a renderer: it never materializes a JS mesh, so nothing is copied through the boundary at all, and multi-component arrays keep their exact dtype rather than being widened to `Float64` (they survive the object boundary too since v9.9.0, via `*_components` — see "The mesh object shape" — but at the cost of a copy).

### Applying operations first

`convertSurfaceOps` takes a pipeline and applies it before extracting the surface — all inside C++:

```js
const report = meshio.convertSurfaceOps('/part.msh', '/part.vtp', [
  { op: 'clean', weld: true },
  { op: 'smooth', method: 'taubin', iterations: 20 },
  { op: 'quality' },
]);
console.log(report.steps, report.warnings);
```

Prefer it over chaining the individual operation bindings (`clean`, `smooth`, …): each of those takes and returns a JS `Mesh`, so a pipeline built from them copies the whole mesh across the boundary once per step (and, before v9.9.0, flattened every multi-component array on the first one). An **empty** pipeline is byte-identical to `convertSurface`, which is what lets a viewer use one code path for the plain and the post-operation display — and makes undo a replay of a shortened pipeline rather than a set of inverse operations.

This is exactly what the [browser viewer](./viewer.md) does. It is built on this package and is worth reading as a worked example of the whole pipeline — worker, transferable buffers, and vtk.js — as well as being a live client-side format converter you can try at **<https://loumalouomega.github.io/meshioplusplus/viewer/>**.

### Running a settings pipeline

`runPipeline` (v9.11.0) runs a whole [settings document](./pipeline.md) — read → operation chain → write, against MEMFS paths — with **no** surface-extraction tail: what the pipeline produces is what is written.

```js
const report = meshio.runPipeline({
  Input: { Path: '/part.msh' },
  Operations: [
    { Op: 'ConvertCells', Mode: 'simplexify' },
    { Op: 'Gradient', Array: 'temperature' },
    { Op: 'Quality' },
  ],
  Output: { Path: '/part.vtu' },
});
```

It accepts the parsed object, the JSON text, or a MEMFS path ending in `.json` (the wrapper `JSON.parse`s the string forms — the wasm binary carries no JSON parser of its own, so this surface never needed the nlohmann submodule). Note the vocabulary is the settings.json **PascalCase** one (`Op`/`RemoveOrphans`), while `convertSurfaceOps` keeps its pre-existing camelCase op specs — the two dispatch through the same core engine (`apply_pipeline_step`), differing only in spelling, so they cannot drift. The returned report is `{steps: [{op, ...counters}], warnings: []}` with PascalCase counter keys; parsing is strict (unknown ops/keys error by name), and the multi-mesh ops (`Merge`, `Split`, ...) are rejected pointing at the CLI verbs.

## The mesh object shape

Unlike the Python bindings (which hand numpy a zero-copy view straight into the C++ buffer), WASM linear memory and the JS heap are different address spaces, so every value crossing the boundary is copied once. `readMesh` returns, and `writeMesh` accepts, a plain object:

```ts
{
  points: Float64Array,       // flat, row-major: numPoints * dim
  dim: number,                 // 2 or 3
  cells: [
    { type: string, data: Int32Array, nodesPerCell: number }
    // one entry per cell block, data flat row-major: numCells * nodesPerCell
  ],
  point_data?: { [name: string]: Float64Array },
  cell_data?: { [name: string]: Float64Array[] },   // one array per cell block
  field_data?: { [name: string]: Float64Array },

  // Per-array component count, for any array that is not a scalar (v9.9.0).
  point_data_components?: { [name: string]: number },
  cell_data_components?:  { [name: string]: number },   // per ARRAY, not per block
  field_data_components?: { [name: string]: number },
}
```

This deliberately mirrors the Python `Mesh`'s structure (points, a list of cell blocks, `cell_data` as one array per block). Cell connectivity is always `Int32Array` — the C++ core's connectivity dtype is Int64, but node/point counts for any mesh a browser can reasonably hold fit comfortably in 32 bits, and `Int32Array` is far more ergonomic in JS than `BigInt64Array`.

### Multi-component (vector / tensor) data arrays

Data arrays are flat, and **a flat typed array carries no shape** — the same problem `xdmfSeriesWriteDataArrays`' own `components` argument solves. Each of the three data maps therefore has a sibling `*_components` object giving the per-entity width of any array that is not a scalar:

```js
const mesh = {
  points, dim: 3, cells,
  point_data: { velocity: new Float64Array(n * 3) },   // interleaved
  point_data_components: { velocity: 3 },
};
meshio.writeMesh('/out.med', mesh, 'med');
const back = meshio.readMesh('/out.med', 'med');
back.point_data_components.velocity;   // 3
```

- **An absent name means one component.** A caller that never sets `*_components` gets exactly the pre-v9.9.0 scalar behaviour, and `readMesh` only writes an entry for genuinely multi-component arrays — so a scalar-only mesh comes back with three empty objects and no existing consumer sees a new key.
- **`cell_data_components` is one value per array, not per block**: every block of a named `cell_data` array must agree on its component count, which is the uniform mesh API's own invariant.
- A length that is not a multiple of the declared count, or a non-positive/non-integer count, is a catchable `Error` naming the array.

Before v9.9.0 there was no such metadata: an `(n,3)` array re-entered C++ as `(3n,1)`, so writing a vector field to MED produced a file the reader itself rejected (`"field data size does not match its declared shape"`), and every object-based operation silently passed the array through instead of gathering it, leaving stale values of the wrong length. The path-based `convert`/`convertSurface`/`convertSurfaceOps` calls were never affected, because they never materialize a JS mesh at all.

**Ragged cell blocks** (polygon/polyhedron blocks with a varying node count per cell, e.g. MED Voronoi polygons or OpenFOAM general polyhedra) cross the boundary as flat CSR arrays instead of a single rectangular `data`/`nodesPerCell` pair, since embind has no efficient representation for a nested array of arrays:

- **Polygon** (1-level ragged, jagged rows): `{type, data, rowOffsets}` — `data` is every row's node ids concatenated, `rowOffsets` is each cell's start index into `data` (length `numCells + 1`).
- **Polyhedron** (2-level ragged, cell → faces → node ids): `{type, data, faceOffsets, cellOffsets}` — `data` is every face's node ids concatenated, `faceOffsets` is each face's start index into `data` (length `totalFaces + 1`), `cellOffsets` is each cell's start index into the face list (length `numCells + 1`).

```js
// A triangle and a 4-gon in one polygon block.
const poly = {
  points: new Float64Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 1, 0]),
  dim: 3,
  cells: [{ type: 'polygon', data: new Int32Array([0, 1, 2, 1, 3, 4, 2]), rowOffsets: new Int32Array([0, 3, 7]) }],
};
m.writeMesh('/ragged.med', poly, 'med'); // MED is the ragged-polygon-capable writer (POG/POG2)
```

These are the same three arrays the [C API](/c_api) hands out through its `mio_poly_conn` snapshot and the Fortran/Julia/R bindings expose in each language's own shape — see [Polyhedra and ragged cells](/polyhedra) for the shared vocabulary and the winding rule. Malformed offsets (non-monotonic, or running past the end of `data`) throw naming the offending array rather than reading out of range.

`readMesh` reports whichever shape the source held; `writeMesh` accepts either, and the target format's own writer decides what it can represent — MED writes ragged polygons directly, but no C++ format writer accepts a polyhedron block yet (a documented, pre-existing gap distinct from the JS boundary itself, which does carry polyhedron blocks correctly through operations like `clean`/`merge`/`convert_cells`), so writing one throws naming the format rather than silently dropping data.

## Mesh operations

Besides read/write/convert, the module exposes the mesh operations on the same mesh-object shape: `extractSurface(mesh [, recordParentIds])`, `extractSkin(mesh [, linearize])`, `attachQuality(mesh)`, `sniffFormat(path)`, `computeBandwidth(mesh)`, and `reorder(mesh, method)` (method `"rcm"`, `"morton"`, or `"hilbert"`). `reorder` returns `{ mesh, nodePermutation, cellPermutations }` — the renumbered mesh plus the applied old→new permutations (`Int32Array`s). See [Reordering / renumbering](./reorder.md). Comparison is exposed as `meshesEqual(meshA, meshB, atol, rtol, unordered)` (boolean) and `diff(meshA, meshB, atol, rtol, unordered)` (a report object with `verdict`, `points`, `blocks`, `pointData`/`cellData`/`fieldData`); named point/cell sets are not compared. See [Mesh comparison (diff)](./diff.md). Merging is exposed as `merge([meshA, meshB, ...], weld, atol, sourceTag, dataPolicy, dropDuplicateCells)` (`dataPolicy` is `"intersection"` or `"fill"`), returning a new mesh object; named point/cell sets are not carried. See [Merge / combine](./merge.md). The editing/stats bundle is exposed too: `transform(mesh, matrix, rotateVectorData)` (a 16-element row-major matrix) → mesh; `clean(mesh, weld, atol, removeOrphans, dropDegenerate, dropDuplicateCells)` → `{ mesh, pointsWelded, ... }`; `cropBbox(mesh, lo, hi, mode, recordIds)` / `cropPlane(mesh, point, normal, mode, recordIds)` (`mode` `"all"`/`"any"`) / `cropPredicate(mesh, array, compare, value, recordIds)` (a scalar `cell_data` comparison, `"<"`/`"<="`/`">"`/`">="`/`"=="`/`"!="`; **no `mode`** — a per-cell value has nothing for an all/any rule to reduce) → mesh; `split(mesh, by, tagName)` → an array of `{ key, mesh }`; and `stats(mesh)` → an object of geometric measures (`bboxMin`/`bboxMax`/`extent`/`centroid`, `cellTypeCounts`, `totalArea`, `signedVolume`, `unsignedVolume`, `numInverted`). Element-representation conversion is exposed as `convertCells(mesh, mode, recordParentIds)` with `mode` `"linearize"`, `"simplexify"`, or `"elevate"`, returning a new mesh; a polyhedron block under `"simplexify"` and the full-Lagrange targets (`quad9`/`hexahedron27`) under `"elevate"` throw a catchable `Error`. **Polyhedral refinement** is exposed as `subdivide(mesh, recordParentIds)`: one polyhedral child per face of every eligible 3D cell, connected to a new interior point, with no per-type template table — tabulated types (reduced to corners for a quadratic variant) and existing polyhedron blocks are handled uniformly, and the result is automatically conforming, unlike `refine`. There is no point map, unlike `convertCells` (subdivide never prunes or renumbers a point); a cell whose faces are not a closed orientable surface throws a catchable `Error`. See [subdivide](/subdivide). **Polyhedral coarsening**, the many-to-one counterpart, is exposed as `agglomerate(mesh, targetGroupSize)`: greedy seed-and-grow over the mesh's shared-face dual, absorbing face-adjacent neighbours by accumulated shared-face area until each group reaches `targetGroupSize` (default 8) members, then emitting one polyhedron per group whose faces are exactly its external boundary — conserving volume exactly, since internal faces are simply dropped rather than re-triangulated. Like `subdivide` there is no point map (points are never pruned or renumbered — `clean(mesh, ..., removeOrphans: true)` is the follow-up for a minimal point set), and a non-manifold input (a face shared by three or more cells) throws a catchable `Error` naming the face. It is also a `convertSurfaceOps`/`runPipeline` pipeline step (`{op: 'agglomerate', targetGroupSize}`), reached through the same generic `pipe_op_table()` dispatch every other step goes through with no extra WASM code. See [agglomerate](/agglomerate). Uniform refinement is exposed as `refine(mesh, levels, recordParentIds, options)`, subdividing every cell into same-type children (`triangle`/`quad` into 4, `tetra`/`wedge`/`hexahedron` into 8) with shared mid-entity nodes, so the result has no hanging nodes; a higher-order cell, a `pyramid`, or a ragged block throws a catchable `Error`. The optional fourth argument selects a **subset** to refine — `{cells, region, array, compare, value, closure, recordLevels, recordHierarchy}`, at most one selector — in which case the hanging nodes that leaves are resolved by the closure and, for `'redgreen'` and `'propagate'`, the output is still conforming (`'balanced'` deliberately keeps them and reports each in `refine:hanging`); the `convertSurfaceOps` pipeline op `{op: 'refine', ...}` takes the same fields, where the comparison is spelled `compare` because `op` is the step's own discriminant. `recordHierarchy` attaches the persistent `refine:cell_id`/`refine:parent_id` cell_data — a link between the meshes a multigrid caller keeps across passes, not a tree inside one — and forces `refine:entity` (the prolongation stencil) even when the closure leaves no hanging node; see [refine](/refine#refinecell_id-and-refineparent_id). **Green-element undo**, the two-mesh op restoring `refine`'s transitional cells to their coarse parent, is exposed as `undoGreen(coarse, fine)` → `{ mesh, numGroupsUndone, numCellsRemoved }` — a lookup and substitution read verbatim from `coarse` (no template inversion, no winding repair), needing `fine` to carry `refine:cell_id`/`refine:parent_id`/`refine:level` (i.e. it must come from a `refine(coarse, ..., {recordHierarchy: true, recordLevels: true})` call). The six reserved `refine:*` arrays are dropped from the output. Being a two-mesh op it is deliberately **not** reachable as a `convertSurfaceOps` pipeline step — `{op: 'undoGreen'}` throws the same "unknown operation" catchable `Error` `Merge`/`Interpolate`/`Split`/`Diff` already throw there, since none of the two-mesh ops ever reach `pipeline_op_table()`. See [green-element undo](/undo_green). **Mass-preserving field transfer**, the sibling of `interpolate` that conserves `sum(value * measure)` over the region two meshes share rather than sampling pointwise, is exposed as `conservativeInterpolate(source, target, arrays, defaultValue, onConflict)` → mesh (always Float64 arrays). Both meshes are simplexified first, accepting ragged/polyhedron blocks for free; unlike `interpolate`, an empty `arrays` means every source point_data **and** cell_data array. Like `interpolate` it is a two-mesh op and so is **not** reachable as a `convertSurfaceOps` pipeline step (`{op: 'conservativeInterpolate'}` throws the same "unknown operation" catchable `Error`). See [conservative interpolation](/conservative_interpolate). Partitioning is exposed as `partition(mesh, nparts, method, imbalance, mode, seed, recordIds, ghostLayers, weightsKey)` → an array of `{ partId, mesh }` (exactly `nparts` entries, blocks kept 1:1 with the input, unlike `split`) and `partitionLabels(mesh, nparts, method, imbalance, mode, seed, weightsKey)` → one label array per cell block. **Only the SFC method exists in the WASM build** — KaHIP is never compiled in (no Emscripten port, and it would bloat the bundle), so `method: "kahip"` throws a catchable `Error` naming `MESHIOPLUSPLUS_WITH_KAHIP` and `"auto"` always resolves to SFC. Smoothing is exposed as `smooth(mesh, method, iterations, lambda, mu, fixBoundary, preserveFeatures, featureAngle, guardInversion)` with `method` `"taubin"` (the default, shrink-free) or `"laplacian"` (stronger per pass, but shrinking), returning `{ mesh, numNodesMoved, maxDisplacement, numSkippedInversion }`; only the point coordinates move, so connectivity and every data value come through unchanged. Boundary nodes, feature nodes and the nodes of blocks with unknown edge topology are pinned by default; the caller `frozen` mask of the C++ API is not exposed here. The two cutters are exposed as `slice(mesh, origin, normal, recordParentIds)` — the planar cross-section, one dimension below the cut cells — and `isosurface(mesh, array, isovalues, component, recordParentIds)`, its data-driven sibling: the level set of a scalar `point_data` array. `isovalues` accepts a number or an array (several contours land in one mesh, tagged per cell with `iso:value` and `iso:index`), `component` is negative for the row magnitude, and naming a `cell_data` array throws a catchable `Error` — cell data is piecewise constant and has no level set, so convert it with `dataCellToPoint` first.

`gradient(mesh, array, operator, method, location, output, component, overwrite)` returns `{ mesh, numSkipped, numFallback }` — the gradient, divergence or curl of a `point_data` field. Unlike the cutters it changes no geometry: it attaches one array and hands the mesh back. **Its result is an `(n, 3)` or `(n, 9)` array, so its width travels in the returned mesh's `point_data_components` / `cell_data_components` sibling maps** (see "The mesh object shape" above) — a JS caller that rebuilds a mesh by hand must carry them, or the array re-enters C++ flattened. Note that `component` is negative for **every** component here, deliberately the opposite of `isosurface`'s sentinel. It is also a `convertSurfaceOps` pipeline step (`{op: 'gradient', array, operator, method, location, output}`).

`hessian(mesh, array, method, location, output, overwrite)` returns `{ mesh, numSkipped, numFallback }` — the Hessian (second derivative) of a **scalar** `point_data` field, `gradient`'s companion one order further. A composition of TWO `gradient` calls, not a new numerical kernel: the field is differentiated once (point location, regardless of the caller's own `location`), then that `(n, 3)` gradient is differentiated again with the default gradient operator, producing `(n, 9)` — the flattened row-major 3x3 Hessian. `method` is forwarded to BOTH internal passes. Like `gradient`, the result's `(n, 9)` width travels in the returned mesh's `point_data_components` / `cell_data_components` sibling maps. A field that is at most linear has an exactly zero Hessian everywhere — the one mesh-shape-independent guarantee; a genuinely quadratic field's composition is exact on a structured/symmetric mesh away from its own boundary and approximate on an irregular mesh (see [second derivatives](./hessian.md)). Input must have exactly one component. It is also a `convertSurfaceOps` pipeline step (`{op: 'hessian', array, method, location, output}`).

`estimateError(mesh, array, method, marking, markingValue, output, marked, overwrite)` returns `{ mesh, globalError, numSkipped, numMarked }` — the Zienkiewicz-Zhu recovery-based error indicator of a `point_data` field, a composition of `gradient` with the measure-weighted point↔cell averaging round trip, not a new kernel. Like `gradient` it changes no geometry: `error:zz` (Float64) is always attached, and `error:marked` (Int64 0/1) too when `marking` is not `"none"` — so `refine`'s own `where` selector needs no change at all. Cells that cannot be evaluated read NaN in `error:zz` and `0` (never NaN) in `error:marked`, counted in `numSkipped`. It is also a `convertSurfaceOps` pipeline step (`{op: 'estimateError', array, method, marking, markingValue, output, marked}`). See [error estimation](./error.md), [field derivatives](./gradient.md), [second derivatives](./hessian.md), [transform](./transform.md), [clean](./clean.md), [crop](./crop.md), [split](./split.md), [stats](./stats.md), [cell conversion](./convert_cells.md), [polyhedral refinement](./subdivide.md), [polyhedral coarsening](./agglomerate.md), [refine](./refine.md), [green-element undo](./undo_green.md), [partitioning](./partition.md), [smoothing](./smooth.md), [slicing](./slice.md), and [isosurfaces](./isosurface.md).

::: tip Reachable from `loadMeshioPlusPlus()` since v7.4.0
Before v7.4.0 the geometry operations above were bound in the WASM module but **not forwarded by the package wrapper**, so they were unreachable through `loadMeshioPlusPlus()` (only file I/O and the `data_*` operations were). They are all forwarded now. The index maps the C++ core returns for `cropBbox`/`cropPlane`/`split`/`convertCells`/`subdivide`/`agglomerate`/`refine`/`partition` are still not carried across the JS boundary — use the `recordIds`/`recordParentIds` flags, which attach the same provenance as ordinary data arrays (or `partitionLabels` for the raw assignment). `smooth` needs none of that — it never adds, removes or renumbers a node or a cell. `undoGreen`'s per-block cell maps are likewise not exposed here — a documented flat-binding gap shared with its C API/Fortran counterparts.
:::

The [data operations](./data_operations.md) — which act on `point_data` / `cell_data` / `field_data` and never modify the geometry — are exposed as `dataDrop(mesh, location, names, ignoreMissing)`, `dataKeep(...)`, `dataRename(mesh, location, from, to)`, `dataPointToCell(mesh, names, suffix)`, `dataCellToPoint(mesh, names, weight, suffix)`, `dataCalc(mesh, expression, location, outputName, overwrite)`, `dataCondition(mesh, location, names, mode, lo, hi, scope, nanPolicy, nanReplacement, suffix)` — each returning a new mesh — and `dataInfo(mesh)`, which returns an array of per-array summary objects (`location`, `name`, `dtype`, `shape`, `numBlocks`, `numEntries`, `numComponents`, `numValues`, `min`, `max`, `mean`, `minPerComponent`/`maxPerComponent`/`meanPerComponent`, `numNan`, `numInf`, `numFinite`, `inconsistentBlocks`). Enumerations cross as strings: `location` is `"point"`/`"cell"`/`"field"`, `weight` is `"uniform"`/`"measure"`, `mode` is `"clamp"`/`"normalize"`/`"standardize"`, `scope` is `"component"`/`"magnitude"`, and `nanPolicy` is `"ignore"`/`"replace"`/`"fail"`. A malformed `dataCalc` expression throws a catchable `Error`. See [data operations](./data_operations.md), [array management](./data_manage.md), [averaging](./data_average.md), [expressions](./data_calc.md), [conditioning](./data_condition.md) and [data summary](./data_info.md).

`dataIntegrate(mesh, arrays)` returns an array of per-array field-integral objects (`name`, `numComponents`, `domain`, `regions`) — a cell-measure-weighted total/mean of one or more `cell_data` arrays, `gradient`'s integration counterpart. `domain` and each entry in `regions` share one shape (`numCells`, `numSkipped`, and parallel `totalPerComponent`/`meanPerComponent`/`domainMeasurePerComponent`/`numNanPerComponent` arrays — the `dataInfo` `*PerComponent` convention), `regions` carrying one entry per named Cell region present on the mesh (not a partition: a cell in two regions contributes fully to both). `arrays` empty/undefined means every `cell_data` array. Unlike `gradient`/`estimateError` it is **read-only** — like `dataInfo`, it changes no geometry and is not a `convertSurfaceOps`/pipeline step. A `point_data`-only name throws a catchable `Error` naming `dataPointToCell` as the fix. See [field integration](./field_integration.md).

## Transient (time-series) XDMF

`createXdmfTimeSeriesWriter(path, { dataFormat, gzipLevel })` is the **one stateful** thing in this API: every other binding is a pure function over a mesh object, but a time series writes the mesh **once** and then appends one cheap step per solve, and its `.xdmf` light data can only be written when the collection is complete. See [XDMF time series](./xdmf_time_series.md).

```javascript
const w = m.createXdmfTimeSeriesWriter('/series.xdmf');   // 'HDF' by default
w.writePointsCells(mesh);                                 // the static grid, once
for (let k = 0; k < nsteps; ++k) {
    w.writeData(k * dt, stepMesh(k));   // only point_data/cell_data are used
}
w.close();                              // the files appear HERE

const xdmf = m.FS.readFile('/series.xdmf');
const h5 = m.FS.readFile('/series.h5'); // 'HDF' writes TWO files — see below
```

The returned object has `writePointsCells(mesh)`, `writeData(time, mesh)`, `finalize()`, `numSteps()`, `finalized()` and `close()`. `close()` finalizes if needed and releases the handle; it is safe to call twice and safe to call from a `finally`. After it, every method throws a catchable `Error` — the underlying handle is an index into a module-local table, not a pointer, so a stale handle can never be a use-after-free.

::: warning `'HDF'` writes TWO files into the virtual filesystem
The `.xdmf` at the path you gave **and** its sibling heavy-data file `<path minus extension>.h5` (a sibling of the `.xdmf`, not a file in the current directory — this differs from the pure-Python `TimeSeriesWriter`). Copy both out of `FS`; an `.xdmf` without its `.h5` is unreadable. `'XML'` writes one self-contained file; `'Binary'` writes the `.xdmf` plus one `<path minus extension><n>.bin` per array. Nothing is on the filesystem until `finalize()`/`close()` runs.
:::

`'HDF'` works here: the shipped artifact links a wasm32 HDF5 (v8.0.0+). Read a series back with the ordinary `readMesh` (which resolves the temporal collection structurally and gives you the first step), `readMeshSelective(path, { timeStep: k })` for a particular one (negative counts from the end), or `readMetadata(path).timeValues` to see the steps without loading any payload.

::: tip Why a handle and not a class
The raw embind surface is an opaque integer handle plus seven free functions (`xdmfSeriesCreate`, `xdmfSeriesWritePointsCells`, `xdmfSeriesWriteData`, `xdmfSeriesFinalize`, `xdmfSeriesNumSteps`, `xdmfSeriesFinalized`, `xdmfSeriesFree`), deliberately **not** an embind `class_`, and the object above is the wrapper's ergonomic face of it. `@meshioplusplus/wasm` never hands JS a live C++ object — `NDArray`, `CellBlock` and `Mesh` are all internal — so an embind class instance, with the Emscripten-specific `.delete()` it comes with, would be the API's one exception. Free functions also all go through the same C++ error wrapper, so a `WriteError` arrives as a readable JS `Error`; a bound *member* function surfaces as a bare, message-less `WebAssembly.Exception`. It matches the [C API](/c_api)'s `mio_xdmf_series*` too, so the two flat bindings describe the same object the same way.
:::

This is not a registry format (there is no single `(path, mesh)` call for the registry to name), so it does **not** change the format counts below and does not appear in `availableFormats()`.

## Format support

**As of v8.0.0 the WASM build ships every format the C++ core has**, including the five that need HDF5 or netCDF — there is no longer a WASM-specific format gap. That is 43 readable and 46 writable format keys (`svg`, `tikz` and `gmsh22` are write-only). `gid` gained a reader in v10.19.0 and is now both. `openfoam` became writable in v9.20.0; `gid` (the GiD postprocess format, on a vendored gidpost) needed no WASM-specific binding at all — it reaches every registry-driven consumer, including this one, purely by being registered in the shared `registry.cpp` writer/extension tables, and `-sUSE_ZLIB=1` supplies the `<zlib.h>` its vendored C sources need at the compiler (not just the linker).

> **Four Python-only formats are absent here.** `pmsh`, `zarr`, `cae` and `usd` (v10.35.0, the physics-ML data path — see [formats](./formats.md)) are registered in the Python layer only, not in the shared C++ dispatch registry, so this surface does not see them. Two of the four are directory stores and one needs `usd-core`; all four are export targets for a training pipeline rather than interchange formats.

`abaqus`, `ansys`, `ansysInp` (read/write), `avsucd`, `cgns`, `dex`, `dolfin-xml`, `ensight` (EnSight Gold geometry, `.case`/`.geo`, ASCII + C-binary), `exodus`, `flac3d`, `flux`, `freefem`, `gid` (GiD postprocess; *writing* rides a vendored gidpost and needs zlib, *reading* is meshio++'s own code and needs nothing for the ascii flavour), `gmsh`, `h5m`, `hmf`, `ip`, `mdpa` (Kratos; mesh-level blocks only — see [MDPA](./formats/mdpa.md#c-core)), `med`, `medit`, `mff`, `mfm`, `mphtxt`, `nastran`, `netgen`, `obj`, `off`, `openfoam` (read/write; writing creates a `constant/polyMesh` directory in MEMFS), `permas`, `ply`, `stl`, `su2`, `svg` (**write-only**, 2D visualization), `tecplot`, `tetgen`, `tikz` (**write-only**, 2D LaTeX visualization), `triangle` (`.poly` by default; see the ambiguous-extensions table for `.node`/`.ele`), `ugrid`, `unv`, `vti` (VTK XML ImageData: a regular lattice, so writing needs one — see [VTI](./formats/vti.md)), `vtk`, `vtp`, `vtu` (zlib compression works via Emscripten's built-in port), `wkt`, `xdmf` (XML, Binary **and** HDF). The three field-only formats (`dex`, `ip`, `mff`) read/write geometry-less meshes (field values in `point_data`).

Ask the loaded module rather than trusting this list — it is generated from the same registry the build actually links:

```javascript
const { readers, writers } = m.availableFormats();
```

### What the HDF5/netCDF formats cost, and what changed

- **The `.wasm` is ~6.3 MB sequential / ~6.7 MB threaded, up from ~2.3 MB before HDF5/netCDF** (the published npm tarball is roughly a third of that, since the binary compresses about 3:1 and browsers fetch it compressed). libhdf5, libnetcdf and libcgns are statically linked, and they are real bytes — though cgnslib itself is the cheap one, adding about **290 KB** (measured, v9.22.0), because it is a thin layer over the HDF5 already there. If you do not need these formats, build your own artifact with `./build/configure-wasm.sh --without-hdf5 --build`, or keep HDF5 and drop only the CGNS MLL with `--without-cgnslib` (see below) — the JS API is identical either way, and `availableFormats()` reports the smaller set while `hasCgnslib()` reports whether the MLL is present.
- **Breaking: `.xdmf` now writes an HDF companion file.** The registry's XDMF writer default follows the build, exactly as it does natively: with HDF5 present it emits `Format="HDF"` heavy data into a sibling `<base>.h5` and leaves only the XML skeleton in the `.xdmf`. A caller that used to pull one file out of the virtual filesystem must now pull **two**. Reading is unaffected (all three data formats are read).
- **MED writes plain fields, but not the enhanced ones.** The C++ MED writer handles ordinary `point_data`/`cell_data` — the single-timestep `CHA` common case, including multi-component fields since v9.9.0 — so a field-carrying mesh writes fine here. What it cannot do is the constructs the Python reference writer alone implements (units, multi-timestep metadata, component names supplied via `med:nom`, named profiles, ELNO/ELGA), and this build has no Python to defer to, so those throw by name. MED geometry, `point_tags`/`cell_tags`, families, named regions and `med:num` global numbering are all written normally. See [MED quirks](./formats/med.md#quirks-limitations).
- **`sniffFormat` still cannot identify them.** A plain HDF5 magic number says nothing about which of `med`/`cgns`/`h5m`/`hmf`/XDMF-HDF a file is, so the sniffer deliberately never claims it. Use the extension, or pass `format` explicitly.

### Ambiguous extensions

Some extensions are shared by more than one format. `readMesh`/`writeMesh`/ `convert` all take an optional trailing `format` argument (or an `{inFormat, outFormat}` options object for `convert`) to disambiguate, mirroring Python's `file_format=` kwarg:

| Extension | Default format | Pass `format=` to select instead |
|-----------|-----------------|-----------------------------------|
| `.msh` | `gmsh` | `"ansys"`, `"freefem"` |
| `.inp` | `abaqus` | `"ansysinp"` |
| `.node` / `.ele` | `tetgen` | `"triangle"` (2D Triangle pairs) |

## Threads (OpenMP)

The package ships **two** native artifacts and picks one at load time:

- `meshioplusplus_wasm` — the **sequential** build. Always loadable, anywhere.
- `meshioplusplus_wasm_mt` — the **threaded** build, compiled with the OpenMP parallel backend over Emscripten's Wasm threads (pthreads + `SharedArrayBuffer`). The core's parallel loops — every [mesh operation](#mesh-operations) run through `convertSurfaceOps`, and VTU zlib compression — run multi-threaded here.

`loadMeshioPlusPlus()` **auto-selects**: the threaded build under Node (where `SharedArrayBuffer` is always available) and in a **cross-origin-isolated** browser context, the sequential build otherwise. Force one with the `variant` option:

```js
const meshio = await loadMeshioPlusPlus({}, { variant: "mt" });  // "auto" (default) | "mt" | "seq"
meshio.parallelBackend();  // "openmp" for the threaded build, "seq" for the sequential one
```

**A browser page must be [cross-origin isolated](https://developer.mozilla.org/en-US/docs/Web/API/crossOriginIsolated) to load the threaded build.** `SharedArrayBuffer` requires the document to be served with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

A threaded module **cannot instantiate** without this — and there is no in-artifact fallback — which is exactly why both variants ship and the loader chooses. Where you cannot set those headers (e.g. static hosts like GitHub Pages), a COOP/COEP [service worker](https://github.com/gzuidhof/coi-serviceworker) is the usual workaround; it is what the meshio++ browser viewer's Pages demo uses. If a page is not isolated, `loadMeshioPlusPlus()` transparently loads the sequential build instead, so correctness never depends on the headers — only speed does.

When you pass a `locateFile` override to relocate the `.wasm` (a bundler/CDN setup), return the URL matching the requested filename, since the two variants ask for different binaries:

```js
const meshio = await loadMeshioPlusPlus({
    locateFile: (path) => (path.includes("_mt") ? mtWasmUrl : wasmUrl),
});
```

Under Node no headers are needed — Wasm threads use `worker_threads`. The threaded artifact pre-spawns a worker pool of `navigator.hardwareConcurrency` (falling back to 8 where `navigator` is absent, e.g. Node < 21, then growing on demand).

## Known v1 limitations

- **No zero-copy.** Every array is copied once crossing the JS/WASM boundary (see above) — for very large meshes this has a real memory/time cost that the Python bindings' numpy views avoid.
- **No per-format write options.** Parameterized writers (binary vs ASCII, float format strings, gzip levels, VTK 4.2 vs 5.1) use a fixed default matching that format's own Python reference default (e.g. `vtu` writes binary+zlib, `stl` writes ASCII, `gmsh` writes the 4.1 binary format; `stl`/`ply` extract and write the boundary **skin** of a 3D volume mesh — the Python default, see [Skin extraction](./extract_skin.md) — and `svg`/`tikz` render 3D meshes with the default isometric camera). Per-call overrides may be added in a future release. The standalone `extractSkin` utility is not exposed to JS yet (documented follow-up).
- **Named regions are carried** (since v8.1.0). They ride on the mesh object itself — `mesh.regions` is an array of `{ name, kind, dim, tag, entries }` — so `readMesh` / `writeMesh` / `convert` carry them with no extra call, and nothing new had to be forwarded by the wrapper. `kind` is `'point'`, `'cell'` (global block-major cell indices) or `'side'` (`(cell, facet)` pairs). The Phase-1 formats (gmsh, abaqus, and MED since v9.6.0 — one region per `FAS`/`GRO` group name) map onto them fully; Exodus reads but does not yet write them. See [Named regions](./regions.md).
- **Remaining side-channel data isn't exposed.** `openfoam`'s cell-tag family names, and the `ansysInp`/`unv` set channels pending their Phase-2 region mapping (all carried through a C++ side-channel struct alongside the `Mesh`, mirroring the Python bindings' `AnsysInfo`/`OpenFoamInfo`), are not yet surfaced to JS.
- **Data arrays are always `Float64`.** `point_data`/`cell_data`/`field_data` cross the boundary widened to `Float64Array` whatever their dtype in the file, so an integer material id comes back as a double. Multi-component (vector/tensor) arrays *are* supported since v9.9.0 via the `*_components` objects (see "The mesh object shape" above); before that they were flattened to N unrelated scalars in both directions. The path-based `convert`/`convertSurface` calls still avoid the boundary entirely, and so preserve dtypes as well as shapes.

## Building from source

Requires the [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html):

```sh
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk && ./emsdk install latest && ./emsdk activate latest
source ./emsdk_env.sh
cd ../meshioplusplus  # this repo
./build/configure-wasm.sh --build
node tests/wasm/smoke.mjs
```

`build/configure-wasm.sh --build` builds **both** artifacts by default (`--seq-only` skips the threaded one). Each is configured with `-DMESHIOPLUSPLUS_BUILD_PYTHON=OFF` (no Python/pybind11 involved) and `-DMESHIOPLUSPLUS_MESH_BACKEND=NATIVE` (the fastest [in-memory mesh backend](cpp_backends.md) — canonical Float64/Int64 storage, so the embind typed-array boundary needs no dtype dispatch; the JS API shape is unchanged, and `meshBackend()` on the loaded module reports `"native"`). The two differ only in the parallel backend: the sequential variant uses `-DMESHIOPLUSPLUS_PARALLEL_BACKEND=SEQ`, the threaded one `-DMESHIOPLUSPLUS_PARALLEL_BACKEND=OPENMP -DMESHIOPLUSPLUS_WASM_THREADS=ON` (which adds `-pthread` and a pre-spawned worker pool — see [Threads (OpenMP)](#threads-openmp)). They build in separate trees (`build/wasm-<type>` and `build/wasm-<type>-mt`) because `-pthread` is a whole-translation-unit property. See `--help` for `--without-zlib`, `--without-hdf5`, `--without-netcdf`, `--deps-prefix` and `--build-type`. CI (`.github/workflows/wasm.yml`) builds and smoke-tests both on PRs touching the wasm surface, and publishes to npm on `v*` tags.

### The HDF5 and netCDF dependencies

Neither is a system library on this target, and meshio++'s CMake never downloads anything — so `build/build-wasm-deps.sh` produces them, and CMake only *finds* the result:

```sh
./build/build-wasm-deps.sh                 # HDF5 1.14.6 + netcdf-c 4.9.3 -> a prefix
./build/build-wasm-deps.sh --print-prefix   # where that prefix is
```

`configure-wasm.sh` runs it automatically the first time (several minutes) and passes the prefix to CMake via `CMAKE_FIND_ROOT_PATH` — the required knob, since the Emscripten toolchain re-roots every `find_package` at its sysroot, which makes `CMAKE_PREFIX_PATH` and `HINTS` alone ineffective. Pass `--deps-prefix` to reuse a prefix you already have.

Both libraries are built static, `-Oz`, against Emscripten's own zlib port, with everything that assumes an OS this target does not have switched off: HDF5 without threadsafe/MPI/plugins and without the ROS3, direct, mirror and subfiling VFDs; netCDF without DAP, DAP4, byterange, NCZarr, S3, libxml2 (it falls back to its bundled `ezxml`), the `dlopen` plugin loader and mmap. Two upstream-neutral fix-ups are applied and documented in the script: a one-hunk patch guarding HDF5's `feclearexcept(FE_INVALID)` (wasm defines no floating-point exception flags at all), and a rewrite of the exported CMake link interfaces, which name imported targets (`ZLIB::ZLIB`, `HDF5::HDF5`, `hdf5::hdf5_hl`) that no single `find_package` in a consumer defines.

**cgnslib is linked in too** (v9.22.0), cross-compiled by the same `build/build-wasm-deps.sh` as a third pinned + SHA256-checked source build. It is strictly an *addition*: meshio++ reads and writes CGNS itself over raw HDF5, including polyhedral `NGON_n`/`NFACE_n` sections since v9.21.0, so what the MLL buys here is **ADF-backed containers** — which are not HDF5 at all, and so are unreachable from the hand-rolled path by construction — and the **CGNS 3.x** section layout. `--without-cgnslib` drops it and keeps everything else.

**The wasm stack is sized for them.** `CMakeLists.txt` links the wasm target with `-sSTACK_SIZE=4MB`, well above Emscripten's 64 KiB default, because HDF5's and netCDF-4's frames overrun it. That overrun is silent — the stack grows down into the static data segment — and cost a real bug during development: one Exodus write clobbered libc++'s locale facets, after which every ASCII reader in the module trapped. `-sSTACK_OVERFLOW_CHECK=1` is on so a recurrence aborts loudly instead.

## Selective reads, metadata, and codecs

`readMeshSelective(path, { format, pointsOnly, arrays, timeStep })` and `readMetadata(path, format)` mirror the Python API; `readerSupportsOptions(format)` reports whether a format has a native selective path. `arrays: null` reads every data array, `arrays: []` reads none.

```javascript
const mesh = m.readMeshSelective('big.vtu', { arrays: ['u'] });
const meta = m.readMetadata('big.vtu');   // meta.fellBackToFullRead

// A multi-step file: 0 (default) is the first step, negative counts from the end.
const last = m.readMeshSelective('run.exo', { format: 'exodus', timeStep: -1 });
m.readMetadata('run.exo', 'exodus').timeValues;  // [0, 0.5, 1] -- always present
```

**Memory mapping is unavailable** here — the Emscripten virtual filesystem has nothing to map, so `FileSource` always uses buffered reads. The option is accepted and ignored.

**zstd and lz4 are compiled out.** Unlike HDF5 and netCDF, which this build now carries, neither has an Emscripten port and neither is worth a from-source dependency for an optional VTK block codec. zlib (`-sUSE_ZLIB=1`) is unchanged and remains the default codec, so every file the WASM build wrote before it still round-trips.

## v9.1.0 additions

- `readMeshSelective(path, { lenient: true })` — see [`doc/selective_read.md`](selective_read.md).
- XDMF series: `w.flush()`, `w.writeDataArrays(time, pointData, cellData, components)`, and `createXdmfTimeSeriesWriter(path, { mode: 'append', autoFlush: true })`. With `flush()` the `.xdmf` appears in MEMFS before `finalize()`, so a partially-written series can be copied out.

`MdpaInfo` (MDPA properties bodies and entity names) is not exposed, as for every flat binding.

## Sequences (transient / multi-file datasets)

A set of MEMFS files — or the steps inside one multi-step file — treated as one ordered dataset, the same surface `convert` and `runPipeline` already work on. See [sequences](sequences.md) for the ordering rule, the time-value precedence and the streaming guarantee.

```js
// Stage the steps in MEMFS, then treat them as one dataset.
for (let i = 0; i < 12; ++i) m.writeMesh(`/seq/out_${i}.vtu`, meshes[i]);

const plan = m.sequenceEntries('/seq/out_*.vtu');
// -> [{path, step, time, timeSource}, ...] in NATURAL-NUMERIC order, so
//    out_9.vtu precedes out_10.vtu (a plain sort gets that backwards).

m.sequenceToTimeseries('/seq/out_*.vtu', '/seq/series.xdmf');   // fan-in  -> 12
const bytes = m.FS.readFile('/seq/series.xdmf');

const paths = m.timeseriesToSequence('/seq/series.xdmf', '/seq/back_{step}.vtu');
// -> ['/seq/back_0000.vtu', ...] fan-out; read them back with Module.FS

// A whole transient post-processing run: the chain applies to EVERY step.
m.runPipeline({
  Version: 1,
  Input: { Pattern: '/seq/out_*.vtu' },
  Operations: [{ Op: 'Quality' }],
  Output: { Path: '/seq/post_{step}.vtu' },
});
```

`runPipeline` **routes** a transient document (a `Pattern`/`Paths` input, a `{step}`/`{index}` output, or `Mode`/`Parallel`/`Workers`) to the sequence driver automatically; a plain single-mesh document takes the unchanged path, so nobody has to know which kind of document they hold.

The pattern language is deliberately just `*` and `?` — **no** `**`, no `[set]`, and the directory component is literal — identical to the core's, so the browser and the CLIs accept exactly the same words.

Three things that fail **by name** rather than doing something surprising: a fan-in to a format that cannot hold a series (only XDMF can), a fan-out without a `{step}`/`{index}` token, and a multi-step input aimed at a single-step output. None of them silently keeps step 0.

**`Parallel` is accepted and ignored with a warning** in the report: it is a Python-driver feature (a process pool), and this build has no processes to pool. The steps run in order, which is what the streaming guarantee needs anyway.

## Regular grids and signed distance (v9.24.0, `computeSdf` v9.25.0)

`grid(dims, origin, spacing)`, `voxelize(mesh, resolution, ...)`, `surfaceWatertightCheck(mesh)`, `sampleDistance(surface, points)` and `distanceToSurface(query, surface)`. `grid` is the only binding in the package that takes no input mesh — it creates one. `sampleDistance` takes a **flat** `[x0,y0,z0, x1,y1,z1, …]` array and returns a `Float64Array`.

`computeSdf(surface, structure, resolution, ...)` does both halves in one call, returning `{ mesh, dims, origin, spacing, maxDepth, numBanded, quality }`. `structure: 'octree'` refines only near the surface, and sizes itself from `rootResolution`/`maxDepth` — passing `resolution` or `cellSize` with it is an error, since its finest cell is already determined. Its output is 1-irregular (it has hanging nodes).

`{ Op: 'Voxelize' }` and `{ Op: 'ComputeSdf' }` also work as pipeline / `convertSurfaceOps` steps, and are the only steps that *replace* their input's geometry rather than transforming it.

The generated grid carries an `sdf:*` numeric `field_data` header describing itself. No file format persists arbitrary `field_data`, so write it as [`.vti`](formats/vti.md), whose `Origin`/`Spacing`/`WholeExtent` attributes are the same information.

See [`doc/voxelize.md`](voxelize.md) and [`doc/sdf.md`](sdf.md).
