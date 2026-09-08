# MCP server

Expose every meshio++ operation to AI agents over the [Model Context Protocol](https://modelcontextprotocol.io/): reading and writing 40+ mesh formats, conversion, and the full mesh- and data-operation suite become **tools** any MCP client (Claude Code, Claude Desktop, the MCP inspector, …) can call.

```bash
pip install "meshioplusplus[mcp]"          # the mcp SDK needs Python >= 3.10
claude mcp add meshioplusplus -- meshioplusplus-mcp
```

Then ask the agent things like *"convert `bracket.msh` to VTU, report its quality, and slice it at z = 0.02"* — it drives `convert`, `quality` and `slice` itself.

Every tool is **stateless and file-path based**: input path(s) in, output path(s) out, a strict-JSON report back. That mirrors the CLI, keeps arbitrarily large meshes out of the protocol, and lets the agent work in its own filesystem workspace. Nothing here is part of the C++ core, which stays dependency-free.

## Installation

| Extra | Brings | For |
|---|---|---|
| `meshioplusplus[mcp]` | the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) (MIT) | the `meshioplusplus-mcp` server |
| `meshioplusplus[dashboard]` | the SDK plus [Starlette](https://www.starlette.io/) and [uvicorn](https://www.uvicorn.org/) (both BSD) | `meshioplusplus-mcp --http` — the [browser dataset manager](./dashboard)'s companion process and MCP over HTTP |

This extra is **not** in `meshioplusplus[all]` — `[all]` means "the optional dependencies the *formats* need", the same reasoning as `[interop]`/`[viewer]`. Note the `mcp` SDK itself requires **Python ≥ 3.10** while meshio++ supports 3.8: the package (and the server's pure tool layer) works everywhere, only *running* the server needs the newer Python.

Without the extra, everything degrades by name rather than by traceback:

```
$ meshioplusplus-mcp
meshio++: mcp server: mcp is not installed; install it with
`pip install meshioplusplus[mcp]` (requires Python >= 3.10)
```

`meshioplusplus.mcp.has_mcp()` answers the same question without raising.

## Running the server

The server speaks stdio (the standard local-agent transport):

```bash
meshioplusplus-mcp                          # unrestricted paths
meshioplusplus-mcp --root /path/to/work     # sandboxed (recommended)
python -m meshioplusplus.mcp                # equivalent
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "meshioplusplus": {
      "command": "meshioplusplus-mcp",
      "args": ["--root", "/path/to/workspace"]
    }
  }
}
```

Interactive browsing: `npx @modelcontextprotocol/inspector meshioplusplus-mcp`.

### Over HTTP (`--http`)

The same process can serve over HTTP instead (v10.23.0, the `[dashboard]` extra): MCP over streamable HTTP at `/mcp` for agents, plus a small JSON API the [dataset dashboard](./dashboard#the-companion-process) calls — `GET /api/health`, `POST /api/tools/<name>` (every tool below, dispatched through the same sandbox and sanitizer) and `GET /api/files` (a sandboxed download).

```bash
pip install "meshioplusplus[dashboard]"
meshioplusplus-mcp --http --root /path/to/cases        # 127.0.0.1:8765, a fresh token is printed
claude mcp add --transport http meshioplusplus http://127.0.0.1:8765/mcp \
  --header "Authorization: Bearer <token>"
```

Flags: `--host` (default `127.0.0.1`; binding elsewhere prints a warning), `--port` (default `8765`), `--token` to fix the bearer token instead of generating one, `--no-token` to require none (only on a machine you alone use), `--allow-origin ORIGIN` (repeatable; loopback origins on any port and the hosted docs site are always admitted), `--runs-dir` (where training runs land) and `--webhook URL` (POSTed once per job when it finishes, fails or is stopped — see [the dashboard](./dashboard#being-told-when-a-run-ends); server-side by design, since a client-supplied URL the server then fetches is server-side request forgery). Every `/api/*` and `/mcp*` request needs `Authorization: Bearer <token>`; a missing or wrong token is a `401` `{"error", "error_type": "PermissionError"}` payload. FastMCP's own DNS-rebinding protection stays on for `/mcp`; on an SDK older than streamable HTTP the endpoint is the SSE transport instead (`GET /api/health` reports which).

## Path sandbox

By default paths are unrestricted — the server runs locally under your own account and MCP clients gate filesystem access themselves. Pass `--root DIR` (or set `MESHIOPLUSPLUS_MCP_ROOT`) to confine every input **and** output path: each path is realpath-resolved (symlinks included) and must stay inside the root; relative paths resolve against it. Violations come back as clean `{"error": ..., "error_type": "ValueError"}` payloads the agent can act on — no tool ever surfaces a raw traceback.

## Tools

75 tools; the five marked *gated* need a further extra and return a named install error without it. Transforming tools take `input_path`/`output_path` (+ optional `input_format`/`output_format`, otherwise inferred from the extension) and return the written path plus a mesh summary and the operation's report.

### Inspection (read-only)

| Tool | Returns |
|---|---|
| `formats` | readable/writable format lists + extension map (also the `meshioplusplus://formats` resource) |
| `sniff` | format identified from leading bytes + extension |
| `info` | fast file summary via `read_metadata` — counts, cell blocks, data names, regions, time steps |
| `stats` | bbox, centroid, areas/volumes, per-type counts, inverted cells |
| `quality` | per-metric summaries + histograms; pass `output_path` to write the mesh with `quality:<metric>` cell data |
| `data_info` | every data array's dtype/shape/ranges/NaN counts |
| `data_integrate` | cell-measure-weighted total/mean of one or more `cell_data` arrays, whole-mesh and per named Cell region |
| `regions` | named point/cell/side groups with kind/dim/tag and an entries preview |
| `bandwidth` | node-numbering bandwidth |
| `data_preview` | a bounded window (`offset`/`limit`) of one data array's values |
| `diff` | verdict + `equal` boolean + per-section detail for two files |

### Conversion

| Tool | Notes |
|---|---|
| `convert` | any-to-any format conversion; `points_only`/`arrays`/`time_step` narrow the read; `mode: ascii\|binary` and `compression: zlib\|lz4\|zstd\|lzma\|gzip\|none` subsume the CLI's `ascii`/`binary`/`compress`/`decompress` verbs |
| `pipeline` | run a whole [settings pipeline](pipeline.md) (`settings_path`; read → operation chain → write, PascalCase ops/keys); `input_path`/`output_path` override the document's paths, and the sandbox root covers the paths **inside** the settings file, not just the file itself |
| `sequence` | run a multi-file / transient [sequence](sequences.md) (`input_pattern` **or** `input_paths`, `output_path`; optional `mode`/`times`/`time_from`). A `{step}`/`{index}` token in `output_path` writes one file per step (fan-out); a plain path writes one multi-step file (fan-in, XDMF only — anything else fails by name rather than keeping step 0). Ordering is natural-numeric, so `out_9` precedes `out_10`. A pattern's **directory** component is containment-checked against the sandbox root before it is expanded, and every matched file is re-checked individually |

### Mesh operations

`extract_surface`, `extract_skin`, `reorder` (reports bandwidth before/after), `clean`, `crop` (bbox, half-space, or a `where_array`/`where_compare`/`where_value` `cell_data` predicate), `slice`, `isosurface`, `compute_sdf` (a grid over a surface, filled — `structure` `voxel`/`octree`), `transform`, `convert_cells`, `subdivide` (polyhedral refinement: one polyhedral child per 3D cell face, connected to a new interior point — no per-type template table, unlike `refine`; see [subdivide](/subdivide)), `agglomerate` (polyhedral coarsening: merge groups of cells into single larger polyhedral cells via greedy seed-and-grow over the shared-face dual; see [agglomerate](/agglomerate)), `refine` (uniform, or a subset via `cells`/`region`/`where` with a conforming `closure`; `record_hierarchy` attaches the persistent `refine:cell_id`/`refine:parent_id` parent/child hierarchy a multigrid caller resolves across the sequence of meshes it keeps — see [refine](/refine#refinecell_id-and-refineparent_id)), `undo_green` (restores a `refine` transitional/green cell to its coarse parent, read verbatim from `coarse_path` — a two-mesh tool like `interpolate`; reports `num_groups_undone`/`num_cells_removed` — see [green-element undo](/undo_green)), `decimate`, `decimate_volume` (the volume-mesh sibling of `decimate`: quadric-error tet-edge collapse; `preserve_boundary` defaults `False`, the opposite of `decimate`'s own default, since boundary vertices participate by real quadric error here — see [volume decimation](/decimate_volume)), `remesh` (replaces a surface's triangulation with a new, well-shaped one at a chosen vertex count by ACVD clustering — the output has NO correspondence to the input, point_data/cell_data/named regions are dropped; see [remesh](/remesh)), `remesh_volume` (the volumetric sibling of `remesh`: retetrahedralizes a volume mesh or closed surface at a chosen resolution by isosurface stuffing over a BCC lattice — same no-correspondence output contract, plus `warp_fraction` trading boundary tet quality for a small, measured chance of non-manifold boundary edges reported as `num_non_manifold_edges`; see [remesh_volume](/remesh_volume)), `optimize_volume` (ODT-remeshes a tetrahedral mesh: raises its worst element quality by relocating vertices AND flipping connectivity (2-3/3-2, predicate-free) — the point set is invariant so point data and Point regions carry, cell data and Cell/Side regions are dropped; tet-only and C++-core-only; see [optimize_volume](/optimize_volume)), `smooth` (`method` `taubin`/`laplacian`/`odt` — `odt`, optimal-Delaunay-triangulation smoothing, is tet-only and C++-core-only with no pure-Python fallback), `merge` (N inputs), `split` (one file per piece, `name_template`), `partition` (one file per part), `interpolate` (source → target field transfer), `conservative_interpolate` (mass-preserving overlap-measure weighted transfer — unlike `interpolate`, an unset `arrays` covers every source point_data AND cell_data array; see [conservative interpolation](/conservative_interpolate)), `gradient` (the gradient, divergence or curl of a `point_data` field — see [field derivatives](/gradient); reports `num_skipped` and `num_fallback`), `hessian` (the Hessian, second derivative, of a **scalar** `point_data` field — a composition of two `gradient` calls, `gradient`'s companion one order further; see [second derivatives](/hessian); reports `num_skipped` and `num_fallback`), `estimate_error` (the Zienkiewicz-Zhu recovery-based error indicator of a `point_data` field, plus optional `absolute`/`fraction`/`dorfler` marking into `error:marked` for `refine`'s own `where` selector — see [error estimation](/error); reports `global_error`, `num_skipped` and `num_marked`). `grid_sample` (a mesh's point data onto a regular lattice, written as a `.vti` grid — the step a convolutional or superresolution model needs before it can see a mesh at all; reports the channel names and the `coverage`, the fraction of grid points inside the mesh, since a grid around a concave domain is mostly fill and a model would learn it), `grid_scatter` (the inverse, by trilinear interpolation — consecutive `name_0`/`name_1`/`name_2` channels rebuild one multi-component array), `grid_resample` (onto a finer or coarser lattice over the same box — the baseline a superresolution model has to beat) and `grid_power_spectrum` (azimuthally averaged; whether a super-resolved field carries the right small-scale content, which a pointwise error cannot see — see [mesh and regular grids](/grids)). `proximity_graph` (a radius or k-nearest-neighbour graph over a mesh's points or cell centroids, written as `line` cells with a `degree` array — the neighbourhoods a particle state needs, since it has positions and no connectivity; `box_size` turns on the periodic minimum-image convention and reports the degree spread and the isolated count — see [proximity graphs](/proximity_graphs)). `subsample` (a mesh reduced to a point cloud of exactly `count` points under a token budget — `method` `farthest` for exact farthest-point sampling, `grid` for the `O(N + count²)` lattice-then-farthest variant a million-point cloud needs, `random` for the baseline; `bounds` restricts the candidates to a box and `record_ids` attaches `budget:original_point_id`; the cells and `cell_data` are dropped, the point data and Point regions kept — see [point-cloud budgets](/point_budgets)). Parameters mirror the Python API / CLI one-to-one; operations that produce reports (`clean`, `decimate`, `decimate_volume`, `remesh`, `remesh_volume`, `optimize_volume`, `smooth`, `gradient`, `hessian`, `estimate_error`) include them in the response.

### Data operations

| Tool | Notes |
|---|---|
| `data_manage` | keep/drop/rename arrays: `keep`/`drop` are `[location, name]` pairs, `rename` is `[location, old, new]` triples |
| `data_convert` | average between locations (`direction: point_to_cell \| cell_to_point`) |
| `data_calc` | expression evaluator; accepts the CLI's `"NAME = EXPR"` spelling |
| `data_condition` | clamp / normalize / standardize |

### Dataset manifests

| Tool | Notes |
|---|---|
| `dataset_add` | add a case to a [dataset manifest](datasets.md) (created if absent): `input_pattern` **or** `input_paths` (the `sequence` tool's shape, same sandboxed glob handling), optional `entry_id`/`split`/`tags`/`group`/`notes`/`metadata`; the source is validated now and stored relative to the manifest's directory |
| `dataset_list` | a manifest's entries, filtered by `split`/`tags` (must carry all)/`group` (path or descendant); `resolve: true` also expands each entry's file/step/time plan — every resolved path is containment-checked, since a hand-edited manifest is client input too |
| `dataset_update` | curate: `split` on `entry_ids`/`all_entries`, `assign_splits` by fractions (deterministic via `seed`; `by_group` keeps groups together), `add_tags`/`remove_tags`, or one entry's `group`/`notes`/`metadata` |
| `dataset_find` | every `*.json` at most `max_depth` levels below `root_dir` that parses as a manifest, with its name, entry count, splits, modification time and SHA-256 content hash (the identity the [dashboard](./dashboard) binds a browser-side card to) |
| `dataset_health` | scan a manifest's entries (optionally one `split` / given `entry_ids`, step 0 or `all_steps`), one mesh alive at a time: per entry the step count, NaN/Inf counts over the data arrays (`quality:*` arrays excluded — their NaN means "N/A for this cell type"), inverted/degenerate cells and the worst scaled Jacobian from `compute_quality`, and the arrays present; per manifest the split balance, totals, fields missing across entries and the bad entries — the server-side producer of the dashboard's health summary |

### Training

Jobs on the machine the server runs on (see [the dashboard](./dashboard#launching-and-monitoring-a-run)); every run lives in its own directory under `--runs-dir` (default `<root>/runs`).

| Tool | Notes |
|---|---|
| `train_defaults` | what a launch needs: the data arrays the manifest's first entry carries, its splits, whether the frameworks are installed, and a complete default [training spec](physicsnemo.md#training-and-prediction) |
| `train_start` | *gated* — start a MeshGraphNet run (`fields` → `target_fields` over a split) as a subprocess; returns the job id. Needs `torch_geometric` + `nvidia-physicsnemo`, checked **before** spawning so a missing framework is a named error rather than a dead process |
| `train_status` / `train_metrics` / `train_log` | the three incremental polls: state + progress + ETA, the per-epoch rows since an epoch, and the log from a byte offset |
| `train_list` | every run (newest first) with its hyperparameters and final/best losses |
| `train_stop` | SIGTERM (the trainer finishes its epoch and writes `final.mdlus`), SIGKILL after `grace_seconds` |
| `train_checkpoints` / `train_mark_best` | the run's `.mdlus` files with epoch/validation loss/size, and which one is best |
| `train_predict` | *gated* — predict over a split with a job's (or an explicit) checkpoint, writing `<column>_pred`/`<column>_error` back into `output_dir/<entry_id>.vtu`; returns per-entry RMSE |

### Gated

| Tool | Extra | Notes |
|---|---|---|
| `data_export` | `[arrow]` | data arrays → Parquet table |
| `export_dataset` | `[arrow]` (`[zarr]`/h5py for those layouts) | a *set* of meshes → one `mesh_id`-keyed dataset (hive Parquet / zarr / hdf5; see [ML data handling](/ml)) |
| `screenshot` | `[viewer]` | off-screen PNG render, returned as MCP image content |
| `train_start`, `train_predict` | `torch_geometric` + `nvidia-physicsnemo` (no pip extra, [deliberately](physicsnemo.md#installation-deliberately-no-physicsnemo-extra)) | training and inference; the other `train_*` tools only read files and need neither |

## Reports are strict JSON

Every response survives `json.dumps(..., allow_nan=False)`: numpy scalars and arrays are converted, `NaN`/`±Inf` become `null` (with a `non_finite_replaced` count so the loss is visible), and any array longer than 1000 elements is replaced by a `{"truncated": true, "size", "shape", "dtype", "preview"}` wrapper — reports stay agent-sized no matter how large the mesh.

## Architecture: the pure payload layer

`src/python/meshioplusplus/mcp/` is split exactly the way [`_interop.py`](./interop) is, and for the same reason. Everything a tool *does* lives in the pure layer `_tools.py` — imports only meshioplusplus + numpy + stdlib, runs on every supported Python, tested in the default CI matrix with the `mcp` SDK absent. The FastMCP layer `_server.py` contributes only typed signatures (the JSON schemas) and docstrings (the tool descriptions), and is the one module importing the SDK.

`_tools.TOOL_REGISTRY` is the single source of truth: the server registers from it, and the **parity guard** in `tests/python/test_mcp.py` asserts every public operation in `meshioplusplus.__all__` is claimed by some tool — a new operation fails CI until it gets a tool (or a conscious exemption). That test is what keeps this page's tool table honest as the library grows.
