# Supported Formats

## Format table

Each format name links to a detailed reference page (structure, options, data mapping, and the C++ vs Python behaviour).

| Format name | Extensions | Read | Write | Extra dependencies |
|-------------|-----------|------|-------|--------------------|
| [`abaqus`](./formats/abaqus.md) | `.inp` | ✓ | ✓ | — |
| [`ansys`](./formats/ansys.md) | `.msh` | ✓ | ✓ | — |
| [`ansysInp`](./formats/ansysinp.md) | `.cdb`, `.inp` | ✓ | ✓ | — |
| [`avsucd`](./formats/avsucd.md) | `.avs` | ✓ | ✓ | — |
| [`cae`](./formats/cae.md) | `.npz` | ✓ | ✓ | — |
| [`cgns`](./formats/cgns.md) | `.cgns` | ✓ | ✓ | `h5py` |
| [`dex`](./formats/dex.md) | `.dex` | ✓ | ✓ | — |
| [`dolfin-xml`](./formats/dolfin.md) | `.xml` | ✓ | ✓ | — |
| [`ensight`](./formats/ensight.md) | `.case` / `.geo` | ✓ | ✓ | — |
| [`exodus`](./formats/exodus.md) | `.e`, `.exo`, `.ex2` | ✓ | ✓ | `netCDF4` |
| [`flac3d`](./formats/flac3d.md) | `.f3grid` | ✓ | ✓ | — |
| [`flux`](./formats/flux.md) | `.pf3` | ✓ | ✓ | — |
| [`freefem`](./formats/freefem.md) | `.msh` | ✓ | ✓ | — |
| [`gid`](./formats/gid.md) | `.post.msh` / `.post.res`, `.post.bin`, `.post.h5` | ✓ | ✓ | *writing* needs zlib (vendored gidpost); *reading* needs nothing for ascii, zlib for binary, HDF5 for hdf5 |
| [`gmsh` / `gmsh22`](./formats/gmsh.md) | `.msh` | ✓ | ✓ | — |
| [`h5m`](./formats/h5m.md) | `.h5m` | ✓ | ✓ | `h5py` |
| [`hmf`](./formats/hmf.md) | `.hmf` | ✓ | ✓ | `h5py` |
| [`ip`](./formats/ip.md) | `.ip` | ✓ | ✓ | — |
| [`mdpa`](./formats/mdpa.md) | `.mdpa` | ✓ | ✓ | — |
| [`med`](./formats/med.md) | `.med` | ✓ | ✓ | `h5py` |
| [`medit`](./formats/medit.md) | `.mesh`, `.meshb` | ✓ | ✓ | — |
| [`mff`](./formats/mff.md) | `.mff` | ✓ | ✓ | — |
| [`mfm`](./formats/mfm.md) | `.mfm` | ✓ | ✓ | — |
| [`mphtxt`](./formats/mphtxt.md) | `.mphtxt` | ✓ | ✓ | — |
| [`nastran`](./formats/nastran.md) | `.bdf`, `.fem`, `.nas` | ✓ | ✓ | — |
| [`netgen`](./formats/netgen.md) | `.vol`, `.vol.gz` | ✓ | ✓ | — |
| [`neuroglancer`](./formats/neuroglancer.md) | (no extension) | ✓ | ✓ | — |
| [`obj`](./formats/obj.md) | `.obj` | ✓ | ✓ | — |
| [`off`](./formats/off.md) | `.off` | ✓ | ✓ | — |
| [`openfoam`](./formats/openfoam.md) | `.foam` | ✓ | ✓ | — |
| [`permas`](./formats/permas.md) | `.post`, `.post.gz`, `.dato`, `.dato.gz` | ✓ | ✓ | — |
| [`ply`](./formats/ply.md) | `.ply` | ✓ | ✓ | — |
| [`pmsh`](./formats/pmsh.md) | `.pmsh` | ✓ | ✓ | — |
| [`stl`](./formats/stl.md) | `.stl` | ✓ | ✓ | — |
| [`su2`](./formats/su2.md) | `.su2` | ✓ | ✓ | — |
| [`svg`](./formats/svg.md) | `.svg` | — | ✓ | — |
| [`tecplot`](./formats/tecplot.md) | `.dat`, `.tec` | ✓ | ✓ | — |
| [`tetgen`](./formats/tetgen.md) | `.ele` / `.node` | ✓ | ✓ | — |
| [`tikz`](./formats/tikz.md) | `.tikz` | — | ✓ | — |
| [`triangle`](./formats/triangle.md) | `.node` / `.ele` / `.poly` | ✓ | ✓ | — |
| [`ugrid`](./formats/ugrid.md) | `.ugrid` | ✓ | ✓ | — |
| [`unv`](./formats/unv.md) | `.unv` | ✓ | ✓ | — |
| [`usd`](./formats/usd.md) | `.usd`, `.usda`, `.usdc` | ✓ | ✓ | `usd-core` |
| [`vti`](./formats/vti.md) | `.vti` | ✓ | ✓ | — |
| [`vtk` / `vtk42` / `vtk51`](./formats/vtk.md) | `.vtk` | ✓ | ✓ | — |
| [`vtp`](./formats/vtp.md) | `.vtp` | ✓ | ✓ | — |
| [`vtu`](./formats/vtu.md) | `.vtu` | ✓ | ✓ | — |
| [`wkt`](./formats/wkt.md) | `.wkt` | ✓ | ✓ | — |
| [`xdmf`](./formats/xdmf.md) | `.xdmf`, `.xmf` | ✓ | ✓ | `h5py` (for HDF data) |
| [`zarr`](./formats/zarr.md) | `.zarr` | ✓ | ✓ | `zarr` (writing needs 3.x) |

**Note on directory formats:** `openfoam`, [`pmsh`](./formats/pmsh.md) and [`zarr`](./formats/zarr.md) write a *directory* rather than a file. Extension dispatch still works (`case.pmsh` and `case.zarr` carry their suffix on the directory name), but a target with no extension needs an explicit `file_format=`, and none of the three can be read from or written to a buffer. A glob over such a set — `read_sequence("out_*.pmsh")` — matches them, which an ordinary file glob would not.

**Note on the physics-ML formats:** [`pmsh`](./formats/pmsh.md), [`zarr`](./formats/zarr.md), [`cae`](./formats/cae.md) and [`usd`](./formats/usd.md) are **Python-only**. They are not in the shared C++ dispatch registry, so they are absent from the WASM, C, Fortran, Julia, R and native-CLI surfaces; everything else in this table is reachable from all of them. `pmsh`, `zarr` and `cae` are also *lossy by design* — each reduces a mesh to what its consumer's data model holds (one simplex kind, or a triangulated skin plus node fields) — so they are export targets rather than interchange formats.

**Note on `.msh`:** `ansys`, `freefem`, and `gmsh` all use `.msh`. When writing without an explicit `file_format`, meshio++ picks `gmsh` if the mesh carries gmsh-native tags (`gmsh:physical`/`gmsh:geometrical`/`gmsh:dim_tags`) or MED-derived tags (`cell_tags`/`point_tags`/`med:*`), else falls back to the first registered candidate (`ansys`). When reading, meshio++ tries the registered formats in order and uses the first that parses the file. Specify `file_format` explicitly (e.g. `file_format="freefem"`) to avoid ambiguity either way.

**Note on `.post.*`:** extension resolution tries the *longest* matching suffix first, so a compound extension always wins over a shorter one that also matches — `.post.msh` resolves to `gid`, never falling through to `.msh`'s own `ansys`/`gmsh`/`freefem` candidates (even for a mesh carrying gmsh-native tags, which would otherwise steer a plain `.msh` write to `gmsh`); `.post` alone (no `.msh` suffix) still resolves to `permas`, unaffected.

**Note on `.inp`:** `abaqus` and `ansysInp` both use `.inp`. `abaqus` is registered first, so plain extension-based dispatch resolves to Abaqus by default; pass `file_format="ansysInp"` (or call `meshioplusplus.ansysInp.read`/`write` directly) to select the Ansys/APDL reader for a `.inp` file.

**Note on `tetgen`:** The format spans two files (`.node` + `.ele`). It cannot be read from or written to a buffer.

**Note on `triangle` vs `tetgen` (`.node`/`.ele`):** both formats use `.node`/`.ele`; `tetgen` is registered first, so plain extension dispatch resolves to it. When *reading*, a 2D pair makes tetgen raise and the dispatcher falls through to `triangle` automatically; when *writing*, pass `file_format="triangle"` (or use a `.poly` path, which defaults to `triangle`). Like tetgen, the format spans multiple files and cannot use buffers.

**Note on `ensight`:** EnSight Gold, geometry only (`.case` + `.geo` sibling pair, ASCII and C-binary with byte-order auto-detection; variables/time steps out of scope). Multi-part files concatenate into one point array with the owning part recorded as `cell_data["ensight:part"]`. Cannot use buffers. The `.geo` extension is also used by Gmsh *script* files, which meshio++ never claimed.

**Note on `vti`:** VTK XML ImageData is a **regular lattice**: its geometry is the `Origin`/`Spacing`/`WholeExtent` attributes rather than a point array. Reading expands the extent into explicit `hexahedron` cells; writing therefore *requires* a lattice, and a mesh that is not one — including a **partial** grid (`voxelize`'s `surface`/`inside` fills, or `compute_sdf`'s octree, whose holes ImageData cannot express) — raises `WriteError` by name. It is the only format that round-trips a generated grid's geometry, which is why [`compute_sdf`](./sdf.md) points at it.

**Note on `vtp`:** VTK XML PolyData holds surface cells only (`vertex`/`line`/`triangle`/`quad`/`polygon`); volume or quadratic cells raise `WriteError`. PolyData has no cell-type array, so 3-/4-noded `polygon` cells read back as `triangle`/`quad`. Triangle strips are not supported.

**Note on `stl` / `ply` and volume meshes:** both writers extract and write the boundary **skin** of a 3D volume mesh by default (`skin=True` — see [Skin extraction](./extract_skin.md); STL additionally triangulates quads, PLY compacts the vertex table). Pass `skin=False` for the legacy behavior (volume cells dropped with a warning).

**Note on `svg`:** Write-only. Flat 2D meshes draw directly; genuinely 3D meshes render their boundary skin (see [Skin extraction](./extract_skin.md)) through an orthographic camera (`azimuth`/`elevation`/`roll`, default the CAD isometric view) with painter's-algorithm depth ordering. C++ core with a Python fallback.

**Note on `tikz`:** Write-only; emits a standalone (directly `pdflatex`-compilable) LaTeX/TikZ document by default (`standalone=False` for a bare `tikzpicture` snippet). Flat 2D meshes draw directly; genuinely 3D meshes render their boundary skin like the SVG writer (same camera parameters). C++ core (byte-identical to the Python reference, including the 3D path) with a Python fallback.

**Note on `openfoam`:** A directory-based format (`points`/`faces`/`owner`/`neighbour`/`boundary` under `constant/polyMesh`), not a single file — so it is the only writer that *creates a directory*. Writing takes a `.foam` marker file, a `polyMesh` directory, or a case root; a case *directory* has no extension, so that form needs an explicit `file_format="openfoam"`. ASCII only on write (binary is a follow-up). Polyhedral cells are native here.

**Note on `mfm`:** Single element type per file (non-hybrid), linear elements only.

**Note on `exodus`:** One-node `SPHERE`/particle meshes (peridynamics solvers such as [PeriLab](https://github.com/PeriHub/PeriLab.jl) write these) read as `vertex` cells, and per-element **attributes** — `attrib{k}`, where a sphere's radius or a shell's thickness lives — round-trip as `cell_data` under the `exodus:attr:` prefix, always float64 and NaN for a block the file gives no such attribute. Since v9.9.0 ordinary (non-attribute) `cell_data` round-trips too, as element variables (`name_elem_var`/`elem_var_tab`/`vals_elem_var{j}eb{k}`); element-block names round-trip through `Cell` regions (`eb_names`), and `field_data["exodus:time"]` labels the written step. See [`exodus.md`](./formats/exodus.md#element-attributes).

**Note on FEconv-derived formats (`unv`, `mfm`, `freefem`, `mphtxt`, `flux`, `mff`, `dex`, `ip`):** These readers/writers were implemented against the [FEconv](https://github.com/victorsndvg/FEconv) format documentation and public format specs (FEconv is GPL; no FEconv code or data is used — MIT-clean, with fixtures generated by round-trip). `unv` handles the parabolic mid-node "sandwich" ordering, maps permanent groups (datasets 2467/2477/2452/2435/2432/2430) to `point_sets`/`cell_sets`, and reads/writes field datasets (2414, and legacy 55/57 in Code-Aster mode) as `point_data`/`cell_data`; `mphtxt` and `flux` round-trip per-element region references as `cell_data` (`mphtxt:geom`, `pf3:ref`). Node orderings for higher-order elements round-trip losslessly but may differ from the originating tool's internal ordering for some element types.

**Note on the field-only formats (`mff`, `dex`, `ip`):** These carry result data, not geometry. They read into a geometry-less `Mesh` (no cells) with the field(s) in `point_data`; `dex`/`ip` also populate `points` from the coordinates in the file, while `mff` carries no coordinates (its `points` has zero columns and only the field values round-trip). To attach a field to a mesh, read the field file and the mesh file separately and copy the field `Mesh`'s `point_data` onto the geometry `Mesh` — there is no fixed naming convention pairing a field file with its mesh (unlike TetGen's `.node`/`.ele`).

---

## Provenance

Since v10.15.0, every writer that carries a free-text header slot emits one canonical line, `Written by meshio++ v<release>` (`meshioplusplus._provenance.TAG` on the Python side, `detail::kProvenanceTag` on the C++ side — a single owner on each side, so the two engines emit character-identical bytes). Since v10.16.0 a caller can additionally opt into a richer block — source, target, the operation chain, conversion assumptions, a timestamp — rendered where a format's header slot can hold it; see [`doc/provenance.md`](./provenance.md) for the design and [`doc/roadmap.md`](./roadmap.md) section 1 for what remains open (reading the block back on read).

Before this, roughly two dozen writers hand-wrote their own version of the line and had drifted three ways: a stale `meshio` (not `meshio++`) name in a handful of Python writers, a `(C++ core)`-vs-`v{version}` split that made the C++/Python fallback boundary visible in the output bytes, and three writers (`obj`, `ply`, `exodus`) embedding a wall-clock timestamp that made writing the same mesh twice produce different bytes.

The table below is the audit this fixed: every format's comment syntax (if any), where a header may legally sit, and whether meshio++'s writer uses that slot today. A "—" in the last column means the format admits no free-text slot at all, or the slot it does admit (an ID/label field, a fixed keyword line) is a structural record rather than a place for a human-readable credit, and meshio++ does not write one there. The same classification, at the finer `SlotTier` granularity the opt-in block renders against (`None`/`Bounded`/`SingleLine`/`Block`), is [`doc/provenance.md`](./provenance.md#slot-tiers-and-the-moderequired-interaction)'s table -- this one stays the format-level audit.

| Format | Comment syntax admitted | Where a header may sit | Emits the tag? |
|---|---|---|---|
| `abaqus` | `**`-prefixed lines; free text inside `*HEADING`'s data lines | Anywhere; writer uses `*HEADING` | Yes |
| `ansys` | Section `(0 "text")` is the format's own comment section | Anywhere before `(0 ...)` sections | Yes (writer uses the `(1 "...")` program/version record, not a `(0 ...)` comment) |
| `ansysInp` | `!` or `/` prefix | Anywhere | — |
| `avsucd` | `#` prefix | Top of file | Yes |
| `cae` | A bytes array written as the archive's **first** zip member; NUL-padded rows | The file's first bytes, so the ordinary head scanner finds it | Yes |
| `cgns` | None (HDF5/CGNS binary container) — an HDF5 root attribute is the nearest equivalent | n/a | — |
| `dex` | None — a fixed two-line header, the second ending in `#` | n/a (structural, not free text) | — |
| `dolfin-xml` | XML `<!-- -->` | Anywhere in the document | — |
| `ensight` | None named, but the `.geo` header's description line 2 is free text | Fixed line 2 of the `.geo` header | Yes |
| `exodus` | netCDF root `title` attribute | n/a (one attribute, not a line) | Yes |
| `flac3d` | `*` prefix | Top of file | Yes |
| `flux` | None named — an unlabeled free-text line the reader skips (keyed lookup, not positional) | Top of file | Yes |
| `freefem` | None (fixed positional numeric header) | n/a | — |
| `gid` | `# Name: value` "user attribute" lines (ascii/binary); an HDF5 group attribute (`hdf5` flavour) | One per mesh/result "block", via `GiD_fWriteMeshUserAttribute`/`GiD_fWriteResultUserAttribute` | Yes (ascii confirmed; the `hdf5` flavour's result-side attribute placement is not independently verified — see `gid.md`) |
| `gmsh` / `gmsh22` | `$Comments`/`$EndComments` section (spec-legal; our reader skips it, neither writer emits one) | Anywhere between sections | — |
| `h5m` | None (HDF5 container) — an HDF5 root attribute is the nearest equivalent | n/a | — |
| `hmf` | None (HDF5 container) — an HDF5 root attribute is the nearest equivalent | n/a | — |
| `ip` | None (fixed positional numeric layout) | n/a | — |
| `mdpa` | `//` prefix (Kratos/C++ style) | Anywhere | — |
| `med` | HDF5 `DES` mesh-description field (user-overridable via `MedInfo.description`, not a provenance slot) | n/a | — (see the `med.md` note below) |
| `medit` | `#` prefix | Anywhere | — |
| `mff` | None (fixed positional numeric layout, no geometry) | n/a | — |
| `mfm` | None (fixed positional numeric layout) | n/a | — |
| `mphtxt` | `#` prefix | Top of file | Yes |
| `nastran` | `$` prefix | Top of file, before `BEGIN BULK` | Yes (see the sentinel note below) |
| `netgen` | `#` prefix | Top of file | Yes |
| `neuroglancer` | None (binary chunked octree format) | n/a | — |
| `obj` | `#` prefix | Top of file | Yes |
| `off` | `#` prefix | After the `OFF` magic line | Yes |
| `openfoam` | C-style `/* ... */`; the `FoamFile` banner's fixed-width credit cell is this writer's own convention | Top of file, inside the banner box | Yes (C++ writer only — no Python twin) |
| `permas` | `!` prefix | Top of file | Yes |
| `ply` | `comment ` prefix (the format's own keyword) | Anywhere in the header, before `end_header` | Yes |
| `pmsh` | None — a memmap directory has no free-text field, and a sidecar would add a file upstream's own loader never wrote | n/a | — |
| `stl` | Binary: an 80-byte free header slot. ASCII: none (the `solid` line names the object, not a comment) | Binary: the first 80 bytes. ASCII: none | Yes (binary only) |
| `su2` | `%` prefix | Anywhere | — |
| `usd` | The root layer's `documentation` metadata field | Stage metadata, at the top of a `.usda` | Yes (a binary crate layer needs `usd-core` to read it back) |
| `svg` | XML `<!-- -->` | Anywhere in the document | — |
| `tecplot` | None named — the `TITLE = "..."` record is the nearest free-text slot | Top of file | Yes |
| `tetgen` | `#` prefix | Top of each file (`.node`/`.ele`/`.poly`) | Yes |
| `tikz` | `%` prefix (LaTeX/TikZ convention) | Anywhere | — |
| `triangle` | `#` prefix | Top of each file (`.node`/`.ele`/`.poly`) | Yes |
| `ugrid` | None (fixed columnar numeric format) | n/a | — |
| `unv` | None general-purpose — dataset 2414's five ID-label records are a structural slot, not a comment | n/a | — |
| `vti` | XML `<!-- -->` | Anywhere in the document | Yes |
| `vtk` / `vtk42` / `vtk51` | The legacy format's title line (line 2, ≤256 chars) is the format's own free-text slot | Fixed line 2 | Yes |
| `vtp` | XML `<!-- -->` | Anywhere in the document | Yes |
| `vtu` | XML `<!-- -->` | Anywhere in the document | Yes |
| `wkt` | None (the OGC WKT grammar has no comment token) | n/a | — |
| `xdmf` | XML `<!-- -->` | Anywhere in the document | — |
| `zarr` | The root group's `meshioplusplus:provenance` attribute (plain JSON) | Store metadata; recovered without importing zarr | Yes |

Two formats carry a related but **structurally distinct** record that this table does not count as "the tag": `med`'s `DES` mesh-description field defaults to `"Mesh created with meshio++"` (both engines agree; user-overridable, so it is data, not a fixed credit) and `unv`'s dataset-2414 field-header records always read `meshioplusplus` on five fixed ID lines (a label field the format requires, not a comment).

`nastran` is the one documented exception to "both engines emit character-identical bytes": the C++ *reader* only accepts a file carrying its own literal sentinel comment (`meshioplusplus-cpp-nastran`), so it does not misparse a real-world `.bdf`/`.fem`/`.nas` file or a Python-written one — see [`nastran.md`](./formats/nastran.md). The C++ writer emits that sentinel line first and the provenance tag as a second `$` line after it; the Python writer emits only the tag, never the sentinel.

---

## Native acceleration and fallbacks

meshio++ ships a C++ core (`meshioplusplus._core`, built with pybind11 + scikit-build-core). Most formats read and write through the C++ core with zero-copy numpy at the I/O boundary; each has a pure-Python fallback that is used automatically when the C++ path can't handle a file or when the extension was built without an optional dependency:

- **HDF5** (`cgns`, `h5m`, `hmf`, `med`, and XDMF `data_format="HDF"`) — C++ when built with `MESHIOPLUSPLUS_WITH_HDF5`, otherwise `h5py`. For `med`, the C++ core covers the mesh-representation part (points, tags, families — including ones synthesized from named regions or `gmsh:physical`, same-type block consolidation — metadata, node orientation, `POG` ragged polygons) and defers the field/bitmask/multi-mesh constructs to the Python reference; see [`med.md`](./formats/med.md#quirks-limitations). `cgns` is a genuine CGNS/SIDS-compliant subset since v9.8.0 (readable by cgnslib/ParaView/VTK), covering the fixed-node-count element types and, since v9.21.0, polyhedral `NGON_n`/`NFACE_n` sections in both directions; see [`cgns.md`](./formats/cgns.md).
- **netCDF** (`exodus`) — C++ when built with `MESHIOPLUSPLUS_WITH_NETCDF`, otherwise `netCDF4`.
- **zlib** (VTU zlib compression) — C++ when built with `MESHIOPLUSPLUS_WITH_ZLIB`, otherwise the Python stdlib.

`mdpa` is the one format where the Python API deliberately does **not** prefer the C++ core for reading: only the pure-Python reference produces MDPA's `mesh.misc_data`, `mesh.geometries_block` and nested-by-cell-type `cell_data`. The C++ reader/writer exists (and is what the C API / Fortran / Julia / R / WebAssembly / native CLI use), and `mdpa.write` does use it for meshes carrying none of those extras — see [MDPA](./formats/mdpa.md#c-core).

Behaviour and file compatibility are identical either way; the native paths are only faster. Install the optional runtime deps with `pip install meshioplusplus[all]`.

---

## Format-specific write options

All writers are called as `meshioplusplus.write(filename, mesh, file_format=..., **kwargs)` or `mesh.write(filename, **kwargs)`. The `**kwargs` depend on the format.

### Gmsh (`.msh`)

```python
meshioplusplus.gmsh.write(filename, mesh,
    fmt_version="4.1",   # "2.2", "4.0", or "4.1"
    binary=True,
    float_fmt=".16e",
)
```

Use `file_format="gmsh22"` to write version 2.2 via the generic `meshioplusplus.write`.

### VTU (`.vtu`)

```python
meshioplusplus.vtu.write(filename, mesh,
    binary=True,
    compression="zlib",   # "zlib", "lzma", or None
    header_type=None,     # "UInt32" or "UInt64"
)
```

### VTI (`.vti`)

```python
meshioplusplus.vti.write(filename, mesh,   # mesh must be a dense lattice
    binary=True,
    compression="zlib",   # "zlib", "lz4", "zstd", or None
    header_type=None,     # "UInt32" or "UInt64"
)
```

### VTK (`.vtk`)

```python
meshioplusplus.vtk.write(filename, mesh,
    binary=True,
    # For version selection use file_format="vtk42" or "vtk51"
)
```

`file_format="vtk"` writes VTK 5.1. `file_format="vtk42"` or `"vtk51"` select specific versions.

### XDMF (`.xdmf`, `.xmf`)

```python
meshioplusplus.xdmf.write(filename, mesh,
    data_format="HDF",        # "HDF", "XML", or "Binary"
    compression="gzip",       # h5py compression filter (HDF only)
    compression_opts=4,       # compression level
)
```

With `data_format="HDF"`, meshio++ writes a companion `.h5` file alongside the `.xdmf`. With `"XML"`, all data is embedded in the XML. With `"Binary"`, data is written to separate `.bin` files.

### Medit (`.mesh`)

```python
meshioplusplus.medit.write(filename, mesh,
    float_fmt=".16e",
)
```

### PLY (`.ply`)

```python
meshioplusplus.ply.write(filename, mesh,
    binary=True,
)
```

### STL (`.stl`)

```python
meshioplusplus.stl.write(filename, mesh,
    binary=False,
)
```

### MED (`.med`)

```python
meshioplusplus.med.write(filename, mesh,
    med_version="4.1.0",   # MAJ.MIN.REL written to INFOS_GENERALES
)
```

MED does not support compression. `meshioplusplus.med.read_med_multi`/ `write_med_multi` read/write files containing several meshes — see [`med.md`](./formats/med.md). Since v9.6.0 MED is also a Phase-1 [named region](./regions.md) format (`FAS`/`GRO` group names ↔ `Point`/`Cell` regions, no side regions), carries the optional `NUM` global numbering as `point_data`/`cell_data["med:num"]`, and rejects a file written by a newer MED major version with a named error.

### AnsysInp (`.cdb`, `.inp`)

`meshioplusplus.ansysInp.read(filename)` / `meshioplusplus.ansysInp.write(filename, mesh)` — no extra options. See the [`.inp` note](#format-table) above for the Abaqus extension collision.

### OpenFOAM (`.foam`)

`meshioplusplus.openfoam.read(filename)` / `meshioplusplus.openfoam.write(filename, mesh)` — no extra options. `write` creates `<case>/constant/polyMesh/`; it is the only meshio++ writer that produces a directory, and needs the compiled core (there is no Python fallback writer).

### CGNS (`.cgns`)

```python
meshioplusplus.cgns.write(filename, mesh,
    compression="gzip",
    compression_opts=4,
)
```

### Nastran (`.bdf`)

```python
meshioplusplus.nastran.write(filename, mesh,
    point_format="fixed-large",   # or "fixed-small", "free"
    cell_format="fixed-small",
)
```

### FLAC3D (`.f3grid`)

```python
meshioplusplus.flac3d.write(filename, mesh,
    float_fmt=".16e",
    binary=False,
)
```

### SU2 (`.su2`)

`meshioplusplus.su2.write(filename, mesh)` — no extra options.

### AVS-UCD (`.avs`)

`meshioplusplus.avsucd.write(filename, mesh)` — no extra options.

### Abaqus (`.inp`)

Abaqus is one of the three Phase-1 [named region](./regions.md) formats (with gmsh and MED), and the only one that can express a **side set**: `*NSET` → point regions, `*ELSET` → cell regions and `*SURFACE, TYPE=ELEMENT` → side regions, in both the C++ core and the Python reference. Abaqus names its groups but has no integer id for them, so a region's `tag` is not preserved. Face identifiers (`S1`..`S6`) are remapped to meshio++'s own facet numbering — the two differ, and the per-type table is spelled out in [Named regions](./regions.md#abaqus-face-identifiers).

`meshioplusplus.abaqus.write(filename, mesh)` — no extra options.

### DOLFIN-XML (`.xml`)

`meshioplusplus.dolfin.write(filename, mesh)` — no extra options. Both `point_data` (since v9.9.0, as `dim="0"` mesh functions) and `cell_data` are written to sibling `<stem>_<name>.xml` files; see [`dolfin.md`](./formats/dolfin.md#file-structure).

### GiD (`.post.msh` / `.post.res`, `.post.bin`, `.post.h5`)

```python
meshioplusplus.gid.write(filename, mesh,
    mode="auto",             # "auto", "ascii", "binary", or "hdf5"
    analysis_name="meshio++",
    step=1.0,
)
```

See [`gid.md`](./formats/gid.md). Reading takes only `time_step` (selecting one step of a multi-step results file).

---

## CLI format names

When using `meshioplusplus convert -o <format>`, use one of the format names from the first column of the table above (e.g. `gmsh`, `gmsh22`, `vtk`, `vtk42`, `vtu`, `xdmf`, …).
