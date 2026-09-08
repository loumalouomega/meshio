// Ambient declarations for @meshioplusplus/wasm's hand-written wrapper
// (src/index.mjs). Mirrors the JS-facing mesh object shape produced/consumed
// by bindings/wasm/js_bindings.cpp's meshToVal/valToMesh (see doc/wasm.md for
// the full format-support table and known v1 limitations).

/**
 * A rectangular (uniform node count) group of cells, all the same meshio++
 * cell type.
 */
export interface RectangularCellBlock {
  /** meshio++ cell type name, e.g. "triangle", "tetra10", "hexahedron". */
  type: string;
  /** Flat, row-major connectivity: length === numCells * nodesPerCell. */
  data: Int32Array;
  nodesPerCell: number;
}

/**
 * A 1-level ragged (jagged polygon) group of cells: rows of varying node
 * count, so there is no single `nodesPerCell`. Two flat CSR arrays instead
 * of a nested array of arrays, which embind has no efficient representation
 * for: `data` is every row's node ids concatenated, and `rowOffsets` is each
 * cell's start index into `data` (length numCells + 1, so cell `c`'s row is
 * `data.slice(rowOffsets[c], rowOffsets[c + 1])`).
 */
export interface PolygonCellBlock {
  type: 'polygon' | 'polygon2';
  data: Int32Array;
  rowOffsets: Int32Array;
}

/**
 * A 2-level ragged (polyhedron) group of cells: each cell is a list of
 * faces, each face a list of node ids. Three flat CSR arrays: `data` is
 * every face's node ids concatenated; `faceOffsets` is each face's start
 * index into `data` (length totalFaces + 1); `cellOffsets` is each cell's
 * start index into the face list (length numCells + 1, so cell `c`'s faces
 * are `faceOffsets[cellOffsets[c]] .. faceOffsets[cellOffsets[c + 1]]`).
 *
 * No C++ format writer accepts a polyhedron block yet (a documented,
 * pre-existing gap) -- this shape crosses the JS boundary correctly (e.g.
 * through {@link MeshioPlusPlusModule.clean} or any other operation) but
 * `writeMesh` will throw naming the format when asked to write one.
 */
export interface PolyhedronCellBlock {
  type: string;
  data: Int32Array;
  faceOffsets: Int32Array;
  cellOffsets: Int32Array;
}

/** A single homogeneous group of cells, all the same meshio++ cell type. */
export type CellBlock = RectangularCellBlock | PolygonCellBlock | PolyhedronCellBlock;

/**
 * A mesh as exchanged with the WASM boundary: every array is copied (there
 * is no zero-copy path across the JS/WASM memory boundary, unlike the
 * Python bindings' numpy views) and cell connectivity is always Int32Array
 * (down-cast from the C++ core's Int64, which is safe for any mesh size a
 * browser can reasonably hold).
 */
export interface Mesh {
  /** Flat, row-major point coordinates: length === numPoints * dim. */
  points: Float64Array;
  /** 2 or 3. */
  dim: number;
  cells: CellBlock[];
  /**
   * name -> flat, row-major per-point data. A multi-component (vector/tensor)
   * array is stored interleaved, `numPoints * components` long, with its width
   * declared in {@link Mesh.point_data_components}.
   */
  point_data?: Record<string, Float64Array>;
  /**
   * Per-entity width of any `point_data` array that is not a scalar, since a
   * flat typed array carries no shape. A name absent here has one component.
   * Read back from `readMesh` for multi-component arrays only, so a
   * scalar-only mesh gets an empty object.
   */
  point_data_components?: Record<string, number>;
  /** name -> one flat array per cell block, same order as `cells`. */
  cell_data?: Record<string, Float64Array[]>;
  /**
   * Per-entity width of any `cell_data` array that is not a scalar. One value
   * per *array*, not per block: every block of a named cell_data array must
   * agree on its component count.
   */
  cell_data_components?: Record<string, number>;
  /** name -> scalar/small metadata arrays (e.g. material ids). */
  field_data?: Record<string, Float64Array>;
  /** Per-entity width of any `field_data` array that is not a scalar. */
  field_data_components?: Record<string, number>;
  /** Named groups of points / cells / cell facets (see {@link Region}). */
  regions?: Region[];
}

/**
 * A named group of mesh entities: a gmsh physical group, an Abaqus
 * `*NSET`/`*ELSET`/`*SURFACE`, an Exodus block or set, a MED family, a Kratos
 * SubModelPart. Regions travel on the {@link Mesh} object itself rather than
 * through a function of their own, so `readMesh` / `writeMesh` / `convert`
 * carry them with no extra call. See doc/regions.md.
 */
export interface Region {
  name: string;
  /**
   * What `entries` indexes:
   *  - `"point"` — point indices, one value per entry.
   *  - `"cell"`  — **global** cell indices, block-major (block 0's cells first,
   *    then block 1's, ...), one value per entry.
   *  - `"side"`  — `(global cell index, local facet index)` pairs, so `entries`
   *    holds two values per entry. Facets are numbered as meshio++ numbers the
   *    faces of a 3-D cell and the edges of a 2-D one.
   */
  kind: 'point' | 'cell' | 'side';
  /** Topological dimension the group was declared for, or -1 if unspecified. */
  dim: number;
  /** Format-native integer id (gmsh physical tag, MED family id), or -1. */
  tag: number;
  /** Flat entries, ascending and de-duplicated. */
  entries: Int32Array;
}

/**
 * One region's shape, without its entries -- the `readMetadata` counterpart of
 * {@link Region}. Cheap to enumerate (build a SubModelPart tree, say) without
 * the cost of loading every entry.
 */
export interface RegionSummary {
  name: string;
  kind: 'point' | 'cell' | 'side';
  /** Topological dimension the group was declared for, or -1 if unspecified. */
  dim: number;
  /** Format-native integer id (gmsh physical tag, MED family id), or -1. */
  tag: number;
  /** Number of grouped entities (not the entries themselves). */
  numEntries: number;
}

export interface ConvertOptions {
  /** Explicit input format key, or omit to infer from inPath's extension. */
  inFormat?: string;
  /** Explicit output format key, or omit to infer from outPath's extension. */
  outFormat?: string;
}

/** One cell block's shape, as reported by {@link MeshioPlusPlusModule.readMetadata}. */
export interface MeshMetadataCellBlock {
  /** meshio++ cell type name, e.g. `"triangle"`, `"tetra10"`. */
  type: string;
  numCells: number;
  /** 0 for a ragged block, whose rows have no single node count. */
  nodesPerCell: number;
  ragged: boolean;
}

/**
 * A file's shape without its heavy arrays -- the result of `readMetadata`.
 *
 * `bboxMin`/`bboxMax` are **omitted** rather than null when no bounding box was
 * computed, so "not computed" cannot be misread as a box at the origin.
 */
export interface MeshMetadata {
  numPoints: number;
  pointDim: number;
  /** Total across every block. */
  numCells: number;
  cellBlocks: MeshMetadataCellBlock[];
  pointDataNames: string[];
  cellDataNames: string[];
  fieldDataNames: string[];
  /** The format that was actually used, whether given or inferred/sniffed. */
  format: string;
  /**
   * True when the format has no header-only path and the file had to be read
   * whole. The summary is still correct, just not cheap.
   */
  fellBackToFullRead: boolean;
  /**
   * The file's recorded time-series values, empty for a format with no time
   * concept. This is the count `readMeshSelective`'s `timeStep` may name, so it
   * is what makes a step request checkable before issuing it.
   */
  timeValues: number[];
  /**
   * The file's named regions, without their entries. Always present -- empty
   * on a native metadata path (VTU/VTP/XDMF/Gmsh 4.1 today), since none of
   * those formats currently map regions at all, so this is never a wrong
   * answer, only cheap where a full read already happened anyway (every
   * fallback path, and Exodus, which always falls back).
   */
  regions: RegionSummary[];
  bboxMin?: number[];
  bboxMax?: number[];
}

/**
 * One operation in a {@link MeshioPlusPlusModule.convertSurfaceOps} pipeline.
 *
 * Parameters are optional and fall back to the same defaults the Python API
 * uses; only `op` is required.
 */
export type OpSpec =
  | { op: 'quality' }
  | {
      op: 'clean';
      weld?: boolean;
      atol?: number;
      removeOrphans?: boolean;
      dropDegenerate?: boolean;
      dropDuplicateCells?: boolean;
    }
  | {
      op: 'smooth';
      method?: SmoothMethod;
      iterations?: number;
      /** Negative means "this method's own default" (0.5 Laplacian, 0.33 Taubin). */
      lambda?: number;
      mu?: number;
      fixBoundary?: boolean;
    }
  | {
      op: 'refine';
      levels?: number;
      /** Global (block-major) indices of the cells to refine. */
      cells?: number[];
      /** Name of a cell or point region to refine. */
      region?: string;
      /**
       * Scalar `cell_data` array to threshold, with `compare` and `value`. The
       * comparison is spelled `compare` rather than `op` because `op` is this
       * union's own discriminant.
       */
      array?: string;
      compare?: RefineCompare;
      value?: number;
      closure?: RefineClosure;
      recordLevels?: boolean;
      /**
       * Attach `refine:cell_id`/`refine:parent_id` -- the persistent
       * parent/child hierarchy a multigrid caller resolves across the
       * sequence of meshes it keeps. Also forces `refine:entity` to be
       * attached even when the closure leaves no hanging node.
       */
      recordHierarchy?: boolean;
    }
  | {
      /**
       * QEM surface decimation. With no criterion given the pipeline chip
       * defaults to `ratio: 0.5`.
       */
      op: 'decimate';
      /** Fraction of the (triangulated) faces to KEEP, in (0, 1]. */
      ratio?: number;
      targetFaces?: number;
      maxError?: number;
      placement?: DecimatePlacement;
      preserveBoundary?: boolean;
      preserveFeatures?: boolean;
      featureAngle?: number;
    }
  | {
      /** Attaches the assignment as `partition:part` cell data. */
      op: 'partition';
      nparts?: number;
      method?: PartitionMethod;
    }
  | {
      /**
       * The planar cross-section through the mesh (slice), in **world**
       * coordinates: the volume is replaced by the surface where the plane
       * intersects it, one dimension lower — a genuine, flat, correctly-coloured
       * section. `mode` is accepted for backward compatibility but ignored (a
       * cross-section has no "keep side").
       */
      op: 'section';
      point: number[];
      normal: number[];
      mode?: CropMode;
    }
  | {
      /**
       * The level set of a scalar `point_data` field (isosurface) — section's
       * data-driven sibling, and like it a surface one dimension lower.
       * `component` is negative for the row magnitude.
       */
      op: 'isosurface';
      array: string;
      isovalue: number;
      component?: number;
    }
  | {
      /**
       * The gradient, divergence or curl of a `point_data` field. A pure data
       * step: geometry is untouched and one new array is attached.
       */
      op: 'gradient';
      array: string;
      operator?: GradientOperator;
      method?: GradientMethod;
      location?: 'point' | 'cell';
      output?: string;
      component?: number;
    }
  | {
      /**
       * The Hessian (second derivative) of a scalar `point_data` field --
       * `gradient`'s companion one order further, for curvature-based
       * adaptive refinement. A pure data step: geometry is untouched and
       * one new (n,9) array is attached.
       */
      op: 'hessian';
      array: string;
      method?: GradientMethod;
      location?: 'point' | 'cell';
      output?: string;
    }
  | {
      /**
       * The ZZ recovery-based error indicator of a `point_data` field, plus
       * optional marking. A pure data step: geometry is untouched and the
       * indicator (and, when `marking` is not `"none"`, the marking) array
       * is attached.
       */
      op: 'estimateError';
      array: string;
      method?: ErrorMethod;
      marking?: ErrorMarking;
      markingValue?: number;
      output?: string;
      marked?: string;
    }
  | {
      /**
       * A regular grid around the mesh. One of the two steps that replace their
       * input's geometry rather than transforming it: what comes out is a
       * lattice, not the mesh that went in. Give exactly one of `resolution` and
       * `cellSize`.
       */
      op: 'voxelize';
      resolution?: number[];
      cellSize?: number;
      bounds?: number[];
      padding?: number;
      paddingRelative?: number;
      fill?: VoxelFill;
      attachOccupancy?: boolean;
      maxCells?: number;
      sign?: SdfSign;
    }
  | {
      /**
       * A signed distance field: a grid over the mesh's surface, filled. Like
       * `voxelize` it replaces the geometry — what comes out is the grid,
       * carrying `sdf:distance`. `structure: 'octree'` refines only near the
       * surface and sizes itself from `rootResolution`/`maxDepth`, so passing
       * `resolution` or `cellSize` with it is an error; its output is
       * 1-irregular (it has hanging nodes).
       */
      op: 'computeSdf';
      structure?: SdfStructure;
      resolution?: number[];
      cellSize?: number;
      bounds?: number[];
      padding?: number;
      paddingRelative?: number;
      rootResolution?: number;
      maxDepth?: number;
      bandCells?: number;
      recordLevels?: boolean;
      maxCells?: number;
      sign?: SdfSign;
      location?: SdfLocation;
      band?: number;
    };

/** Per-operation counters and caveats from a pipeline run. */
export interface OpReport {
  steps: ({ op: OpSpec['op'] } & Record<string, number | string>)[];
  warnings: string[];
}

/** `gradient`'s differential operator. See doc/gradient.md. */
export type GradientOperator = 'gradient' | 'divergence' | 'curl';

/** Which cells `voxelize` keeps. See doc/voxelize.md. */
export type VoxelFill = 'all' | 'surface' | 'inside';

/** How a signed distance decides which side of the surface a point is on. */
export type SdfSign = 'unsigned' | 'pseudonormal' | 'winding-number';

/** Where a distance is evaluated on a mesh. */
export type SdfLocation = 'corner' | 'center';

/** What to do about a surface that is not watertight. */
export type SdfWatertightCheck = 'off' | 'warn' | 'error';

/** `gradient`'s reconstruction method. See doc/gradient.md. */
export type GradientMethod = 'green-gauss' | 'least-squares';

/** `estimateError`'s estimator family. Only `"zz"` exists today. */
export type ErrorMethod = 'zz';

/** `estimateError`'s marking policy. See doc/error.md. */
export type ErrorMarking = 'none' | 'absolute' | 'fraction' | 'dorfler';

/** `remesh`'s clustering objective. See doc/remesh.md. */
export type RemeshMetric = 'isotropic' | 'quadric' | 'anisotropic';

/** One data array's location: `point_data`, `cell_data`, or `field_data`. */
export type DataLocation = 'point' | 'cell' | 'field';

/** `data_condition`'s transform. See doc/data_condition.md. */
export type ConditionMode = 'clamp' | 'normalize' | 'standardize';

/** `data_condition`'s scope: independent components, or by row magnitude. */
export type ConditionScope = 'component' | 'magnitude';

/** Weighting for `dataCellToPoint`. See doc/data_average.md. */
export type CellPointWeight = 'uniform' | 'measure';

/** What reaches the output for non-finite (NaN/inf) values. They are always
 *  excluded from reductions regardless of this setting. */
export type NanPolicy = 'ignore' | 'replace' | 'fail';

/** Cell-keeping rule for the crop operations: every node inside, or any. */
export type CropMode = 'all' | 'any';

/** Comparison for `cropPredicate` — the same vocabulary `refine`'s selector uses. */
export type CropCompare = '<' | '<=' | '>' | '>=' | '==' | '!=';

/** What `computeSdf` generates to carry the field. */
export type SdfStructure = 'voxel' | 'octree';

/** Partitioning criterion for `split`. */
export type SplitBy = 'type' | 'component' | 'region' | 'tag';

/** Node-renumbering method for `reorder`. */
export type ReorderMethod = 'rcm' | 'morton' | 'hilbert';

/** How `merge` reconciles data arrays that are not present in every input. */
export type MergeDataPolicy = 'intersection' | 'fill';

/** Element-representation conversion performed by `convertCells`. */
export type ConvertCellsMode = 'linearize' | 'simplexify' | 'elevate';

/** Smoothing operator applied by `smooth`. `'taubin'` is shrink-free.
 *  `'odt'` (optimal-Delaunay-triangulation smoothing) is tet-only and
 *  moves each free interior vertex toward the volume-weighted average of
 *  its incident tets' circumcenters. See doc/smooth.md. */
export type SmoothMethod = 'laplacian' | 'taubin' | 'odt';

/**
 * How `refine` resolves the hanging nodes a partial refinement leaves behind.
 * `'redgreen'` promotes an affected cell's split-edge mask to the smallest
 * admissible superset, keeping the extra refinement local. `'propagate'`
 * promotes it straight to a full split: always conforming and defined for every
 * cell type, but it converges to uniform refinement of the whole edge-connected
 * component.
 */
export type RefineClosure = 'redgreen' | 'green' | 'propagate' | 'red' | 'balanced' | '2:1';

/** The comparison a `refine` predicate selector applies. */
export type RefineCompare = '<' | '<=' | '>' | '>=' | '==' | '!=';

/**
 * Which cells `refine` should split. At most ONE of `cells`, `region` and
 * `array` may be given; with none, every cell is refined.
 */
export interface RefineOptions {
  /** Global (block-major) indices of the cells to refine. */
  cells?: number[];
  /**
   * Name of a region to refine. A cell region selects its own cells, a point
   * region every cell with any node in it; a side region is an error.
   */
  region?: string;
  /** Name of a scalar `cell_data` array to threshold. */
  array?: string;
  /** The predicate's comparison (default `'<'`). */
  compare?: RefineCompare;
  /** The predicate's right-hand side. A non-finite cell value never matches. */
  value?: number;
  /** How to resolve hanging nodes (default `'redgreen'`). */
  closure?: RefineClosure;
  /** Attach the `refine:level` `cell_data` array. */
  recordLevels?: boolean;
  /**
   * Attach the `refine:cell_id`/`refine:parent_id` `cell_data` arrays -- the
   * persistent parent/child hierarchy a multigrid caller resolves across the
   * sequence of meshes it keeps ("a link between two meshes, not a tree
   * inside one"): an unsplit cell keeps its id and is its own parent; a
   * split cell's children each get a fresh id and carry the parent's id. An
   * input already carrying `refine:cell_id` is updated whatever this says.
   * Also forces `refine:entity` to be attached even when the closure leaves
   * no hanging node, since it already records the coarse corners each new
   * fine node is the mean of -- the multigrid prolongation weights, which
   * `'redgreen'`/`'propagate'` would otherwise never expose.
   */
  recordHierarchy?: boolean;
}

/** Where `decimate` places the surviving vertex of a collapsed edge. */
export type DecimatePlacement = 'optimal' | 'midpoint' | 'endpoint';

/** How `interpolate` draws a target sample's value from the source.
 *  `'barycentric'` simplexifies the source first (simplex-linear on quad/hex
 *  sources; triangles evaluated in the xy-plane) and is exact on a linear
 *  field; `'nearest'` copies the nearest source point's value bit-for-bit. */
export type InterpolateMethod = 'nearest' | 'barycentric';

/** What `interpolate` does when a transferred name already exists on the
 *  target: throw, replace, or write to `name + '_interp'`. */
export type InterpolateOnConflict = 'error' | 'overwrite' | 'suffix';

/** What `conservativeInterpolate` does when a transferred name already
 *  exists on the target: throw, replace, or write to `name + '_interp'`. */
export type ConservativeInterpolateOnConflict = 'error' | 'overwrite' | 'suffix';

/** Partitioning backend: SFC is always available; KaHIP is never compiled
 *  into the WASM build, so `'kahip'` always throws and `'auto'` = `'sfc'`. */
export type PartitionMethod = 'sfc' | 'kahip' | 'auto';

/** KaHIP preconfiguration (ignored by the SFC method). */
export type PartitionMode = 'fast' | 'eco' | 'strong';

/** Read-only per-array summary returned by `dataInfo`. See doc/data_info.md. */
export interface DataArrayInfo {
  location: 'point_data' | 'cell_data' | 'field_data';
  name: string;
  /** numpy-style dtype string, e.g. "f8", "i4". */
  dtype: string;
  shape: number[];
  /** cell_data: number of cell blocks; 1 otherwise. */
  numBlocks: number;
  numEntries: number;
  numComponents: number;
  numValues: number;
  /** Over finite values only; NaN when there are none. */
  min: number;
  max: number;
  mean: number;
  minPerComponent: number[];
  maxPerComponent: number[];
  meanPerComponent: number[];
  numNan: number;
  numInf: number;
  numFinite: number;
  /** cell_data whose blocks disagree in component count. */
  inconsistentBlocks: boolean;
}

/** Cell-measure-weighted reduction of one array over one set of cells (the
 *  whole mesh, or one named Cell region) -- see `dataIntegrate`. */
export interface FieldIntegralRegion {
  /** The region's name; absent on the whole-mesh `domain` entry. */
  name?: string;
  /** Cells with a computable measure. */
  numCells: number;
  /** Cells excluded: unmeasurable geometry (ragged, unsupported type, or
   *  degenerate). */
  numSkipped: number;
  /** `sum(value * |measure|)` over cells finite in component k. */
  totalPerComponent: number[];
  /** `totalPerComponent[k] / domainMeasurePerComponent[k]`, or NaN when that
   *  denominator is zero. */
  meanPerComponent: number[];
  /** `sum(|measure|)` over cells finite in component k. */
  domainMeasurePerComponent: number[];
  /** Measurable cells excluded from component k because its value was
   *  non-finite there. */
  numNanPerComponent: number[];
}

/** One array's field integral, returned by `dataIntegrate` --
 *  `gradient`'s integration counterpart. See doc/field_integration.md. */
export interface FieldIntegralArray {
  name: string;
  numComponents: number;
  /** The whole-mesh reduction. */
  domain: FieldIntegralRegion;
  /** One independent entry per named Cell region present on the mesh -- a
   *  cell in two regions contributes fully to both, one in none contributes
   *  to neither. */
  regions: FieldIntegralRegion[];
}

/** Heavy-data layout of a transient XDMF series. */
export type XdmfDataFormat = 'HDF' | 'XML' | 'Binary';

/**
 * A transient (time-series) XDMF writer: one static grid, then one `<Grid>`
 * per time step inside a temporal collection. Created by
 * {@link MeshioPlusPlusModule.createXdmfTimeSeriesWriter}.
 *
 * ## Why this one is a handle and not a function
 *
 * Every other binding in this package is a **stateless function** over a plain
 * mesh object: hand it values, get values back. A series cannot be that,
 * because its whole point is that the mesh is written **once** and each solve
 * appends a cheap step -- and because the `.xdmf` light data can only be
 * written when the collection is complete, since the collection element has to
 * enclose every step. So the object has to survive between calls.
 *
 * The raw embind surface for it is an **opaque integer handle plus free
 * functions** (`xdmfSeriesCreate`/`...WritePointsCells`/`...WriteData`/
 * `...Finalize`/`...NumSteps`/`...Finalized`/`...Free`), deliberately **not**
 * an embind `class_`, and this object is the wrapper's ergonomic face of it.
 * The reasons, in order of weight:
 *
 * 1. `@meshioplusplus/wasm` never hands JS a live C++ object -- `NDArray`,
 *    `CellBlock` and `Mesh` are all internal, and JS only ever sees plain
 *    objects of typed arrays. An embind `class_` instance would be the one
 *    exception, and would come with an Emscripten-specific `.delete()` that
 *    has no counterpart anywhere else in this API.
 * 2. Free functions all go through the same C++ error wrapper, so a
 *    `WriteError` arrives as a readable JS `Error`; a bound *member* function
 *    surfaces as a bare, message-less `WebAssembly.Exception`.
 * 3. It matches the C API (`mio_xdmf_series*`), so the two flat bindings
 *    describe the same object the same way.
 *
 * The handle is an index into a module-local table, not a raw pointer: a
 * stale or forged handle throws an `Error` instead of corrupting linear
 * memory.
 *
 * ## Getting the files out of MEMFS
 *
 * Nothing is on the virtual filesystem until `finalize()`/`close()` runs, and
 * **`'HDF'` produces TWO files**: the `.xdmf` at the path you gave, plus its
 * sibling heavy-data file `<path minus extension>.h5`. Copy both out, and read
 * them back together -- an `.xdmf` without its `.h5` is unreadable.
 *
 * ```js
 * const w = m.createXdmfTimeSeriesWriter('/series.xdmf');  // 'HDF' by default
 * w.writePointsCells(mesh);
 * for (let k = 0; k < 3; ++k) w.writeData(k * 0.5, stepMesh(k));
 * w.close();                                    // the .xdmf appears here
 * const xdmf = m.FS.readFile('/series.xdmf');   // Uint8Array
 * const h5   = m.FS.readFile('/series.h5');     // the companion -- do not skip
 * ```
 *
 * `'XML'` writes one self-contained file and needs no companion; `'Binary'`
 * writes the `.xdmf` plus one `<path minus extension><n>.bin` per array.
 */
export interface XdmfTimeSeriesWriter {
  /**
   * Write the static grid -- the points and cell blocks every step shares.
   * Only geometry and connectivity are used; any data on `mesh` is ignored,
   * because in a series data belongs to a step. Call exactly once, before the
   * first `writeData`.
   * @throws {Error} if called twice, on points of dimension > 3, or on a cell
   *   type with no XDMF spelling.
   */
  writePointsCells(mesh: Mesh): void;

  /**
   * Append one time step's `point_data` and `cell_data`. The mesh's geometry
   * is ignored, so a solver can pass the same object it updates in place --
   * but its cell-block structure must match the one given to
   * `writePointsCells`, since cell data is concatenated across blocks.
   * @param time emitted as the step's `<Time Value=...>`.
   * @throws {Error} if `writePointsCells` has not run, or after `finalize()`.
   */
  writeData(time: number, mesh: Mesh): void;

  /**
   * Append one step from raw arrays instead of a mesh -- the granularity a
   * solver has once `writePointsCells` has fixed the geometry. Arrays are
   * emitted in key order; `components` gives the per-entity width of any array
   * that is not a scalar, since a flat typed array carries no shape.
   * @throws {Error} if an array's length does not match the static grid.
   */
  writeDataArrays(
    time: number,
    pointData: Record<string, Float64Array | number[]>,
    cellData?: Record<string, Float64Array | number[]>,
    components?: Record<string, number>
  ): void;

  /**
   * Write the `.xdmf` as it currently stands, without finalizing, so a run that
   * is killed or still going leaves a readable file covering every flushed
   * step. Safe to call repeatedly; a no-op once finalized.
   * @throws {Error} if the `.xdmf` cannot be written.
   */
  flush(): void;

  /**
   * Write the `.xdmf` light data and close the heavy-data container. This is
   * when the files appear in MEMFS. Idempotent.
   * @throws {Error} if the `.xdmf` cannot be written.
   */
  finalize(): void;

  /** How many steps have been written so far. */
  numSteps(): number;

  /** Whether `finalize()` has already run. */
  finalized(): boolean;

  /**
   * Finalize (if not already) and release the handle. Safe to call twice and
   * safe to call from a `finally` block. After this the writer is unusable and
   * every other method throws.
   */
  close(): void;
}

/**
 * The instantiated module returned by `loadMeshioPlusPlus()`. `FS` is
 * Emscripten's virtual filesystem (MEMFS by default) -- write the bytes of a
 * mesh file there before calling `readMesh`, and read them back out after
 * `writeMesh`/`convert`. See https://emscripten.org/docs/api_reference/Filesystem-API.html
 */
export interface MeshioPlusPlusModule {
  FS: {
    writeFile(path: string, data: string | ArrayBufferView, opts?: object): void;
    readFile(path: string, opts?: { encoding?: 'binary' | 'utf8' }): Uint8Array | string;
    unlink(path: string): void;
    mkdir(path: string): void;
    [key: string]: unknown;
  };

  /**
   * Read a mesh file from the virtual filesystem.
   * @param path virtual FS path.
   * @param format explicit format key (see doc/wasm.md's table), or omit to
   *   infer from `path`'s extension. `.msh` defaults to gmsh, `.inp` to
   *   abaqus -- pass `format` explicitly to select ansys/freefem/ansysinp.
   * @throws {Error} on an unknown/unsupported format or a malformed file.
   */
  readMesh(path: string, format?: string): Mesh;

  /**
   * Read only part of a file: geometry alone, or a named subset of data arrays.
   *
   * `arrays` omitted/null reads every array; `arrays: []` reads none. Readers
   * with a native selective path (vtu/vtp/xdmf/gmsh) skip the unwanted arrays
   * outright; the rest are read whole and filtered, so the result is identical
   * either way and only the cost differs.
   *
   * `timeStep` selects a step of a multi-step file: 0 (the default) is the
   * first, preserving the historical behaviour; negative counts from the end.
   * Out of range throws naming the available count rather than clamping.
   * Honoured by formats carrying a time series (currently exodus); see
   * `readMetadata(...).timeValues` for how many there are.
   *
   * `lenient` downgrades "this reader cannot represent construct X" errors to a
   * warning plus a skip -- currently mdpa's `Table`, `Geometries`, `Mesh` and
   * `Constraints` blocks. It is *not* "ignore all errors": a malformed file, a
   * truncated block or a bad node reference still throws, because continuing
   * past those would return a mesh that is quietly wrong.
   *
   * @throws {Error} on an out-of-range `timeStep`.
   */
  readMeshSelective(
    path: string,
    options?: {
      format?: string;
      pointsOnly?: boolean;
      arrays?: string[] | null;
      timeStep?: number;
      lenient?: boolean;
    }
  ): Mesh;

  /**
   * Summarize a mesh file without loading its heavy arrays.
   *
   * `bboxMin`/`bboxMax` are present only when a bounding box was computed --
   * omitted rather than null, so "not computed" cannot read as a box at the
   * origin. `fellBackToFullRead` is true when the format has no header-only
   * path: the summary is still correct, just not cheap.
   */
  readMetadata(path: string, format?: string): MeshMetadata;

  /** Whether `format` has a native selective-read path. */
  readerSupportsOptions(format: string): boolean;

  /**
   * Write a mesh to the virtual filesystem.
   * @throws {Error} on an unknown/write-unsupported format or malformed input
   *   (e.g. a points/connectivity array length not divisible by its
   *   declared dim/nodesPerCell).
   */
  writeMesh(path: string, mesh: Mesh, format?: string): void;

  /**
   * Read `inPath` and write it to `outPath` directly (no intermediate JS
   * mesh object). Mirrors the CLI's `convert` subcommand.
   */
  convert(inPath: string, outPath: string, options?: ConvertOptions): void;

  /**
   * Like {@link convert}, but writes a *renderable surface*: a mesh with
   * skinnable 3D cells becomes its boundary, anything else passes through, and
   * the result is linearized (a triangle renderer has no concept of a mid-side
   * node, so `triangle6` connectivity drawn verbatim is visible garbage).
   *
   * Prefer this over `readMesh` -> `extractSkin` -> `writeMesh` for anything
   * headed to a renderer. It never materializes a JS {@link Mesh}, so
   * multi-component (vector/tensor) arrays survive -- the flat JS
   * representation cannot carry them.
   */
  convertSurface(inPath: string, outPath: string, options?: ConvertOptions): void;

  /**
   * Like {@link convertSurface}, but applies a pipeline of mesh operations
   * first — all inside C++.
   *
   * Chaining the individual operation bindings (`smooth`, `clean`, …) would
   * route the mesh through the flat JS {@link Mesh} on every step and so
   * destroy every multi-component array; nothing crosses the boundary here.
   *
   * An **empty** pipeline is exactly {@link convertSurface}. That is
   * deliberate: one call serves both the plain and the post-operation display,
   * so they cannot drift, and undo becomes a replay of a shortened pipeline
   * rather than a set of inverse operations.
   *
   * Operations run on the **full-dimensional** mesh, before the boundary is
   * extracted — smoothing a solid is not the same as smoothing its skin.
   *
   * @throws {Error} on an unknown operation name or an unreadable file.
   */
  convertSurfaceOps(
    inPath: string,
    outPath: string,
    ops?: OpSpec[],
    options?: ConvertOptions & {
      /**
       * Keep the `surface:parent_cell` provenance array in the output. A
       * picker needs it; a colour-by menu must filter it out.
       */
      keepProvenance?: boolean;
    }
  ): OpReport;

  /**
   * Run a whole settings pipeline: read `Input.Path`, apply `Operations` in
   * order, write `Output.Path` — the `settings.json` engine (see
   * `doc/pipeline.md`), against MEMFS paths.
   *
   * `settings` may be the parsed settings object, the JSON text itself, or a
   * MEMFS path ending in `.json`. The vocabulary is the canonical PascalCase
   * one (`{Op: "ConvertCells", Mode: "simplexify"}`), unlike
   * {@link convertSurfaceOps}' pre-existing camelCase op specs — the two
   * dispatch through the same core engine, differing only in spelling. The
   * returned report's counter keys are PascalCase too.
   *
   * Unlike {@link convertSurfaceOps} there is no surface-extraction tail:
   * what the pipeline produces is what is written.
   *
   * @throws {Error} on an unknown op or key (strict — never silently
   * ignored), an unreadable input, or an Output option the format cannot
   * honour.
   */
  runPipeline(settings: object | string): { steps: Array<Record<string, number | string>>; warnings: string[] };

  /**
   * One entry of a {@link MeshioPlusPlus.sequenceEntries} plan.
   */
  // (declared inline below rather than as a named export, matching the rest of
  // this file's plain-object style.)

  /**
   * The ordered plan for a **sequence** — a set of MEMFS files, or the steps
   * inside one multi-step file, treated as one transient dataset.
   *
   * `source` is a glob pattern (**`*` and `?` only** — no `**`, no `[set]`,
   * and the directory part is taken literally) or an array of paths, whose
   * order is yours and is kept unless `options.sort`.
   *
   * Ordering is **natural-numeric**, so `out_9.vtu` precedes `out_10.vtu`; a
   * plain sort gets that backwards. Each entry's `timeSource` reports where
   * its time came from, because "the file said 0.25" and "nothing said
   * anything, so this is position 3" are different facts.
   *
   * Reads no heavy data. See `doc/sequences.md`.
   */
  sequenceEntries(
    source: string | string[],
    options?: {
      format?: string;
      times?: number[];
      timeFrom?: 'auto' | 'file' | 'filename' | 'index';
      sort?: boolean;
    },
  ): Array<{
    path: string;
    step: number;
    time: number;
    timeSource: 'explicit' | 'file' | 'filename' | 'index';
  }>;

  /**
   * **Fan-in**: write every step of `source` into one multi-step file.
   *
   * Streams — at most one mesh is alive at a time, whatever the step count,
   * which is what makes a long dataset viable in a browser tab. A format that
   * cannot hold a series throws naming itself and pointing at `{step}`, never
   * a silent truncation to the first step (XDMF is the one that can today).
   *
   * @returns the number of steps written.
   */
  sequenceToTimeseries(
    source: string | string[],
    outPath: string,
    outFormat?: string,
    options?: {
      format?: string;
      times?: number[];
      timeFrom?: 'auto' | 'file' | 'filename' | 'index';
      sort?: boolean;
    },
  ): number;

  /**
   * **Fan-out**: write each step of the multi-step file `inPath` to
   * `outPattern`, which must contain `{step}` or `{index}`. Streams likewise.
   *
   * `{index}` is the plain index; `{step}` is zero-padded to at least four
   * digits, so a 12-step run writes `out_0000 … out_0011`.
   *
   * @returns the MEMFS paths written, so they can be read straight back with
   *   `Module.FS.readFile`.
   */
  timeseriesToSequence(
    inPath: string,
    outPattern: string,
    inFormat?: string,
    outFormat?: string,
  ): string[];

  /** The shared cell-type -> node-count table (e.g. `{triangle: 3, tetra: 4, ...}`). */
  numNodesPerCell(): Record<string, number>;

  /** The shared cell-type -> topological-dimension table (0-3). */
  topologicalDimension(): Record<string, number>;

  /** The mesh backend this build was compiled with ("meshio"/"native"/"kratos"). */
  meshBackend(): string;

  /**
   * The parallel backend this artifact was compiled with: "seq" for the
   * sequential `meshioplusplus_wasm`, "openmp" for the threaded
   * `meshioplusplus_wasm_mt` -- i.e. which of the two variants loaded.
   */
  parallelBackend(): string;

  /**
   * Whether the optional cgnslib (CGNS MLL) backend is linked in.
   *
   * CGNS works either way; this reports whether ADF-backed containers and the
   * CGNS 3.x `NGON_n` section layout are reachable.
   */
  hasCgnslib(): boolean;

  /**
   * The format names this build can actually read and write, both sorted.
   *
   * Prefer this over a hardcoded table when building a file-picker filter or a
   * "convert to" menu: the two lists genuinely differ (`openfoam` is
   * read-only, `svg` and `tikz` are write-only), and the set depends on how
   * the artifact was built -- an official build carries the HDF5/netCDF-backed
   * formats, one built with `--without-hdf5` does not.
   */
  availableFormats(): { readers: string[]; writers: string[] };

  /**
   * Mesh operations: computations ON a mesh (not file formats). The index maps
   * the C++ core returns for `cropBbox`/`cropPlane`/`split`/`convertCells` are
   * not carried across the JS boundary -- use the `recordIds`/`recordParentIds`
   * flags to get the same provenance as data arrays instead.
   */

  /** Extract the boundary of a mesh (faces of a volume, edges of a surface). */
  extractSurface(mesh: Mesh, recordParentIds?: boolean): Mesh;

  /** Extract the skin of a volume mesh (the volume-only special case). */
  extractSkin(mesh: Mesh, linearize?: boolean): Mesh;

  /** Return a clone with the quality metrics attached as `quality:*` cell_data. */
  attachQuality(mesh: Mesh): Mesh;

  /** Guess a file's format from its leading bytes, or `""` if ambiguous. */
  sniffFormat(path: string): string;

  /** Renumber nodes/cells to reduce bandwidth or improve locality. */
  reorder(mesh: Mesh, method?: ReorderMethod): { mesh: Mesh; nodePermutation: Int32Array; cellPermutations: Int32Array[] };

  /** Connectivity bandwidth: max over cells of (max node index - min node index). */
  computeBandwidth(mesh: Mesh): number;

  /** Structured comparison of two meshes. */
  diff(a: Mesh, b: Mesh, atol?: number, rtol?: number, unordered?: boolean): object;

  /** Whether two meshes are equal within tolerance. */
  meshesEqual(a: Mesh, b: Mesh, atol?: number, rtol?: number, unordered?: boolean): boolean;

  /** Combine several meshes into one, optionally welding coincident points. */
  merge(
    meshes: Mesh[],
    weld?: boolean,
    atol?: number,
    sourceTag?: boolean,
    dataPolicy?: MergeDataPolicy,
    dropDuplicateCells?: boolean,
  ): Mesh;

  /** Apply a row-major 4x4 affine transform to the point coordinates. */
  transform(mesh: Mesh, matrix: number[], rotateVectorData?: boolean): Mesh;

  /** Weld / prune / de-duplicate in one pass. */
  clean(
    mesh: Mesh,
    weld?: boolean,
    atol?: number,
    removeOrphans?: boolean,
    dropDegenerate?: boolean,
    dropDuplicateCells?: boolean,
  ): {
    mesh: Mesh;
    pointsWelded: number;
    pointsRemovedOrphan: number;
    cellsDroppedDegenerate: number;
    cellsDroppedDuplicate: number;
  };

  /**
   * Relax point coordinates toward their edge-neighbour centroids, leaving
   * connectivity and every data array untouched. `"taubin"` (the default)
   * alternates a `+lambda` and a `-mu` pass per iteration and is shrink-free;
   * `"laplacian"` is stronger per pass but contracts the mesh. A **negative**
   * `lambda` means "this method's own default" (0.5 Laplacian, 0.33 Taubin).
   * Boundary and feature nodes are pinned by default, and `guardInversion`
   * rejects any move that would flip an incident cell.
   * @throws {Error} on an unknown `method`, a non-negative `lambda` outside
   *   `(0, 1)`, or a `"taubin"` `mu` that does not satisfy `mu < -lambda < 0`.
   */
  smooth(
    mesh: Mesh,
    method?: SmoothMethod,
    iterations?: number,
    lambda?: number,
    mu?: number,
    fixBoundary?: boolean,
    preserveFeatures?: boolean,
    featureAngle?: number,
    guardInversion?: boolean,
  ): { mesh: Mesh; numNodesMoved: number; maxDisplacement: number; numSkippedInversion: number };

  /**
   * Sample data arrays from `source` onto `target` (cross-mesh field
   * transfer). Returns a copy of the target — geometry, connectivity and its
   * own data preserved exactly — with the requested source arrays sampled onto
   * it: source point_data at the target's points, source cell_data at the
   * target's cell centroids (always by nearest source-cell centroid, whatever
   * the method). An empty `arrays` transfers every source point_data array;
   * cell_data transfers only when named. Under `'barycentric'` a target point
   * outside the source domain receives `defaultValue` unless `extrapolate`.
   * @throws {Error} on an unknown method/onConflict, an unknown array name, a
   *   name collision under `'error'`, or a barycentric source with no
   *   triangle/tetrahedron cells after simplexification.
   */
  interpolate(
    source: Mesh,
    target: Mesh,
    method?: InterpolateMethod,
    arrays?: string[],
    extrapolate?: boolean,
    defaultValue?: number,
    onConflict?: InterpolateOnConflict,
  ): Mesh;

  /**
   * Mass-preserving cross-mesh field transfer: an exact overlap-measure
   * weighted remap, so that over the region the two meshes share,
   * sum(target value * target measure) equals sum(source value * source
   * measure) — the property `interpolate`'s `'barycentric'` mode does not
   * have. Both meshes are simplexified first (accepting ragged/polyhedron
   * blocks for free, unlike a restricted cell-type scope). Unlike
   * `interpolate`, an empty `arrays` transfers every source point_data AND
   * cell_data array — there is one algorithm regardless of location. Output
   * arrays are always Float64.
   * @throws {Error} on an unknown onConflict, an unknown array name, a name
   *   collision under `'error'`, mismatched maximum topological dimensions
   *   between the two meshes, or no triangle/tetrahedron cells on either
   *   side after simplexification.
   */
  conservativeInterpolate(
    source: Mesh,
    target: Mesh,
    arrays?: string[],
    defaultValue?: number,
    onConflict?: ConservativeInterpolateOnConflict,
  ): Mesh;

  /**
   * Green-element undo: restore `fine`'s transitional (closure-only) cells
   * back to their original parent, read verbatim from `coarse` — a lookup,
   * not a reconstruction, since `refine` never renumbers or prunes points.
   * `fine` must carry `refine:cell_id`/`refine:parent_id`/`refine:level`
   * (i.e. must come from `refine(coarse, ..., {recordHierarchy: true,
   * recordLevels: true})`); `coarse` must be the mesh that call was run on.
   * The six reserved `refine:*` arrays are dropped from the output. Only a
   * single-pass (`levels=1`) hierarchy is supported.
   * @throws {Error} when `fine` lacks the required hierarchy arrays, when a
   *   `refine:parent_id` value does not resolve against `coarse`'s id space,
   *   or when a sibling group's level matches neither the red nor the green
   *   relationship to its coarse parent's own level.
   */
  undoGreen(
    coarse: Mesh,
    fine: Mesh,
  ): { mesh: Mesh; numGroupsUndone: number; numCellsRemoved: number };

  /** Subset a mesh to an axis-aligned bounding box. */
  cropBbox(mesh: Mesh, lo: number[], hi: number[], mode?: CropMode, recordIds?: boolean): Mesh;

  /** Subset a mesh to the half-space `(p - point) . normal >= 0`. */
  cropPlane(
    mesh: Mesh,
    point: number[],
    normal: number[],
    mode?: CropMode,
    recordIds?: boolean,
  ): Mesh;

  /**
   * Subset a mesh to the cells whose value in a scalar `cell_data` array
   * satisfies a comparison.
   *
   * Deliberately general rather than inside/outside-a-surface specific:
   * inside/outside composes as `distanceToSurface(m, skin, 'pseudonormal',
   * 'center')` then `cropPredicate(field.mesh, 'sdf:distance', '<', 0)`, and the
   * same one mode also crops by `quality:*`, by a material id, or by anything
   * `dataCalc` can produce.
   *
   * There is **no `mode`**: `cropBbox`/`cropPlane` test *points* and then need
   * an all/any rule, whereas a `cell_data` predicate is already one value per
   * cell and has nothing to reduce.
   *
   * **A non-finite cell value never matches**, whatever the comparison —
   * `attachQuality` reports NaN where a metric does not apply.
   * @throws {Error} when `array` is not a scalar `cell_data` array covering
   *   every block, or the comparison is not one of `<`, `<=`, `>`, `>=`, `==`,
   *   `!=`.
   */
  cropPredicate(
    mesh: Mesh,
    array: string,
    compare?: CropCompare,
    value?: number,
    recordIds?: boolean,
  ): Mesh;

  /**
   * Planar cross-section of a mesh (marching tetrahedra on a simplexified
   * input): a volume mesh yields a triangle/quad surface, a 2D surface mesh a
   * line mesh. Crossing points on shared edges are deduped so the section is
   * watertight; each section cell inherits its parent's cell_data.
   */
  slice(mesh: Mesh, origin: number[], normal: number[], recordParentIds?: boolean): Mesh;

  /**
   * Isosurfaces / contours: the level set of a scalar `point_data` field, one
   * dimension below the cut cells — slice's data-driven sibling, sharing its
   * marching-tetrahedra cutter, so contours are watertight and faces are wound
   * toward increasing field. Several isovalues land in the one returned mesh,
   * tagged per cell with `iso:value` (Float64) and `iso:index` (Int64, the
   * ordinal — the integer tag `split(…, "tag")` needs). `component` is negative
   * for the row magnitude, in which case the contoured value is approximate
   * rather than exactly the isovalue.
   * @throws {Error} when `array` names a `cell_data` array (piecewise constant,
   *   so it has no level set — convert it with `dataCellToPoint` first) or no
   *   array at all.
   */
  isosurface(
    mesh: Mesh,
    array: string,
    isovalues: number | number[],
    component?: number,
    recordParentIds?: boolean,
  ): Mesh;

  /**
   * A regular hexahedron lattice from nothing — the only entry point here that
   * creates a mesh rather than transforming one. Points run x fastest, then y,
   * then z; every cell is a right parallelepiped.
   * @throws {Error} on a negative cell count, a non-positive spacing, or a grid
   *   above `maxCells`.
   */
  grid(dims: number[], origin?: number[] | null, spacing?: number[] | null,
       maxCells?: number): Mesh;

  /**
   * Build a regular grid around a mesh. Give exactly one of `resolution` and
   * `cellSize`; `fill` selects the whole bounding box, only the cells a triangle
   * passes through, or only the cells inside the surface.
   * @throws {Error} when neither or both size options are given, on an unknown
   *   `fill`/`sign`, or when the grid exceeds `maxCells`.
   */
  voxelize(
    mesh: Mesh,
    resolution?: number[] | null,
    cellSize?: number,
    bounds?: number[] | null,
    padding?: number,
    paddingRelative?: number,
    fill?: VoxelFill,
    sign?: SdfSign,
    attachOccupancy?: boolean,
    maxCells?: number,
    watertightCheck?: SdfWatertightCheck,
  ): {
    mesh: Mesh;
    dims: number[];
    origin: number[];
    spacing: number[];
    numOccupied: number;
  };

  /** What is wrong with a surface, in numbers rather than a bare flag. */
  surfaceWatertightCheck(mesh: Mesh): {
    boundaryEdges: number;
    nonManifoldEdges: number;
    inconsistentPairs: number;
    degenerateTriangles: number;
    watertight: boolean;
  };

  /**
   * Signed distances from a flat `[x0,y0,z0, x1,y1,z1, …]` array of query points
   * to a surface. Negative is inside.
   * @throws {Error} when the array length is not a multiple of three, or the
   *   surface has no triangles.
   */
  sampleDistance(surface: Mesh, points: number[], sign?: SdfSign, band?: number,
                 watertightCheck?: SdfWatertightCheck): Float64Array;

  /**
   * Attach the signed distance from a query mesh's points (or cell centres) to a
   * surface, as `sdf:distance`.
   */
  distanceToSurface(
    query: Mesh,
    surface: Mesh,
    sign?: SdfSign,
    location?: SdfLocation,
    band?: number,
    recordInside?: boolean,
    watertightCheck?: SdfWatertightCheck,
  ): { mesh: Mesh; numBanded: number; quality: object };

  /**
   * Generate a grid over a surface and fill it with signed distances — the one
   * call that turns a surface into a field.
   *
   * `structure` is `'voxel'` (a dense lattice) or `'octree'` (refined near the
   * surface, and therefore **1-irregular**: it has hanging nodes, like
   * `refine`'s `'balanced'` closure). `resolution`/`cellSize` size a voxel grid
   * and are an **error** with `'octree'`, whose finest cell is
   * `rootResolution / 2 ** maxDepth` and is therefore already determined.
   *
   * `dims` reports the ROOT cell counts and `spacing` the FINEST cell size. The
   * mesh also carries the numeric `sdf:*` `field_data` header describing itself;
   * no file format persists arbitrary `field_data`, so write it as `.vti`, whose
   * `Origin`/`Spacing`/`WholeExtent` attributes are the same information.
   * @throws {Error} when neither or both size options are given for a voxel
   *   grid, when either is given for an octree, on a non-positive
   *   `rootResolution`/`bandCells`, or when the grid exceeds `maxCells`.
   */
  computeSdf(
    surface: Mesh,
    structure?: SdfStructure,
    resolution?: number[] | null,
    cellSize?: number,
    bounds?: number[] | null,
    padding?: number,
    paddingRelative?: number,
    rootResolution?: number,
    maxDepth?: number,
    bandCells?: number,
    recordLevels?: boolean,
    maxCells?: number,
    sign?: SdfSign,
    location?: SdfLocation,
    band?: number,
    watertightCheck?: SdfWatertightCheck,
  ): {
    mesh: Mesh;
    dims: number[];
    origin: number[];
    spacing: number[];
    maxDepth: number;
    numBanded: number;
    quality: object;
  };

  /**
   * Field differential operators: the gradient, divergence or curl of a
   * `point_data` field. `"green-gauss"` applies the divergence theorem over the
   * cell's own faces and is exact for a linear field on any cell;
   * `"least-squares"` fits over the node-sharing neighbours and falls back to
   * Green-Gauss on a degenerate neighbourhood (counted in `numFallback`).
   *
   * An `nc`-component input yields `3 * nc` gradient components, flat and
   * row-major as `[component][derivative]`, so a scalar gives `(n, 3)` and a
   * 3-vector `(n, 9)`; divergence gives 1 and curl always 3, both needing a 2-
   * or 3-component field. The width travels with the array in the returned
   * mesh's `point_data_components` / `cell_data_components` maps.
   *
   * `component` is negative for EVERY component — the opposite of
   * `isosurface`'s sentinel, where negative means the row magnitude. Cells that
   * cannot be differentiated yield NaN and are counted in `numSkipped`.
   * @throws {Error} when `array` names a `cell_data` array (piecewise constant,
   *   so it has no derivative — convert it with `dataCellToPoint` first), an
   *   unknown array, an unknown operator/method, an out-of-range component, or
   *   a component count divergence/curl cannot use.
   */
  gradient(
    mesh: Mesh,
    array: string,
    operator?: GradientOperator,
    method?: GradientMethod,
    location?: 'point' | 'cell',
    output?: string,
    component?: number,
    overwrite?: boolean,
  ): { mesh: Mesh; numSkipped: number; numFallback: number };

  /**
   * The Hessian (second derivative) of a **scalar** `point_data` field --
   * `gradient`'s companion one order further, for curvature-based adaptive
   * refinement.
   *
   * A composition of TWO `gradient` calls, not a new numerical kernel: the
   * field is differentiated once (point location), then that `(n, 3)`
   * gradient is differentiated again with the default gradient operator,
   * producing `(n, 9)` -- the flattened row-major 3x3 Hessian, `H[i][j]` at
   * index `i*3+j`. `method` is forwarded to BOTH internal passes. The width
   * travels with the array in the returned mesh's `point_data_components` /
   * `cell_data_components` maps, exactly as `gradient`'s own output does.
   *
   * A field that is at most LINEAR has an exactly zero Hessian everywhere --
   * the one mesh-shape-independent guarantee. For a genuinely quadratic
   * field the composition is exact on a structured/symmetric mesh away from
   * its own boundary and a good, standard, but genuinely approximate
   * curvature estimate on an irregular mesh (see doc/hessian.md).
   *
   * A curvature-driven refinement indicator needs no new API: `norm(...)`
   * in `dataCalc` on the 9-component output is exactly its Frobenius norm,
   * ready for `refine`'s `where` selector.
   * @throws {Error} when `array` names a `cell_data` array (piecewise
   *   constant, so it has no derivative), an unknown array, or an array
   *   with more than one component (hessian is scalar-only).
   */
  hessian(
    mesh: Mesh,
    array: string,
    method?: GradientMethod,
    location?: 'point' | 'cell',
    output?: string,
    overwrite?: boolean,
  ): { mesh: Mesh; numSkipped: number; numFallback: number };

  /**
   * The Zienkiewicz-Zhu (ZZ) recovery-based error indicator of a `point_data`
   * field, plus optional marking. A composition of `gradient` (Green-Gauss,
   * cell location) with the measure-weighted point↔cell averaging round
   * trip: the indicator is `sqrt(|measure| * sum((recovered - raw)^2))` per
   * cell, attached as `output` (default `"error:zz"`, Float64).
   *
   * `marking` is `"none"` (default), `"absolute"`, `"fraction"`, or
   * `"dorfler"`; when not `"none"` a second Int64 0/1 array `marked`
   * (default `"error:marked"`) is attached too, so `refine`'s own `where`/
   * `--where` selector needs no change at all — the intended use is
   * `refine(mesh, {compare: '>', value: 0.5, array: 'error:marked'})`.
   * `markingValue`'s meaning depends on `marking`: an absolute indicator
   * threshold, a fraction in `(0, 1]` of cells, or the Doerfler bulk fraction
   * theta in `(0, 1]`.
   *
   * Cells that cannot be evaluated read NaN in the indicator array and 0
   * (never NaN) in the marking array, and are counted in `numSkipped`
   * (excluded from `globalError` and from `numMarked`).
   * @throws {Error} when `array` names a `cell_data` array (piecewise
   *   constant, so it has no derivative to recover), an unknown array, an
   *   unknown method/marking policy, or an out-of-range `markingValue` for
   *   `"fraction"`/`"dorfler"`.
   */
  estimateError(
    mesh: Mesh,
    array: string,
    method?: ErrorMethod,
    marking?: ErrorMarking,
    markingValue?: number,
    output?: string,
    marked?: string,
    overwrite?: boolean,
  ): { mesh: Mesh; globalError: number; numSkipped: number; numMarked: number };

  /**
   * Remesh a surface by approximated centroidal Voronoi diagram (ACVD)
   * clustering: replace its triangulation with a new, near-uniformly-sized,
   * well-shaped one at `numClusters` vertices. Unlike every other
   * resolution-changing operation, the output has NEW points and NEW
   * connectivity with no correspondence to the input — `point_data`,
   * `cell_data` and named regions are dropped, `field_data` is carried.
   *
   * `metric` is `"isotropic"` (default; area-weighted centroidal distance,
   * fast, rounds sharp features), `"quadric"` (Garland-Heckbert quadric
   * error, preserves sharp edges/corners at extra cost per candidate move),
   * or `"anisotropic"` (clusters shaped by a local curvature tensor —
   * elongated along low-curvature directions, compact across sharp ones —
   * see `maxAnisotropy`). `subdivide` defaults to automatic (`-1`): the
   * smallest count of uniform `refine` passes reaching `subsampleRatio`
   * items per cluster, capped at `maxSubdivide`; `0` disables subdivision.
   * `gradation` is the curvature-gradation exponent `gamma` in the item
   * weight `area * kappa^gamma` (`0.0`, the default, disables gradation
   * entirely and reproduces plain area weighting). `preserveBoundary`
   * (default `true`) detects the input's open boundary (if any), seeds it
   * before the interior, and emits a `line` dual cell along boundary edges
   * whose endpoints land in different clusters — a no-op on a closed mesh.
   * `maxAnisotropy` (default `4.0`, a measured value) is, under
   * `metric = "anisotropic"`, the maximum ratio between the two in-plane
   * target edge lengths a curvature tensor may request; `1.0` recovers the
   * isotropic shape exactly. Must be at least `1.0`, and an error to set
   * away from the default under any other metric.
   * `numIsolatedClusters` and `numNonManifoldVertices` are two distinct
   * causes of non-manifold output (disconnected clusters vs. "bowtie"
   * vertices) that repair could not fully fix; check both rather than
   * assuming.
   * @throws {Error} on a cluster count below 4 or above the subdivided
   *   input's vertex count, or on any block outside the surface scope (3D
   *   volume cells, higher-order cells, ragged polygon/polyhedron blocks).
   */
  remesh(
    mesh: Mesh,
    numClusters: number,
    subdivide?: number,
    subsampleRatio?: number,
    maxSubdivide?: number,
    maxIterations?: number,
    maxRepairPasses?: number,
    metric?: RemeshMetric,
    gradation?: number,
    preserveBoundary?: boolean,
    maxAnisotropy?: number,
  ): {
    mesh: Mesh;
    numClusters: number;
    numIterations: number;
    subdivideApplied: number;
    numIsolatedClusters: number;
    numNonManifoldVertices: number;
  };

  /**
   * Retetrahedralize a volume mesh (or a closed surface) at a caller-chosen
   * resolution by isosurface stuffing — the volumetric sibling of `remesh`.
   * Unlike `remesh`, this accepts a VOLUME mesh directly (its boundary is
   * extracted internally) as well as a closed surface. Same
   * no-correspondence-with-the-input output contract as `remesh`: new
   * points, new connectivity, `point_data`/`cell_data`/named regions
   * dropped, `field_data` carried.
   *
   * Exactly one of `resolution`/`cellSize` must be given, sizing a body-
   * centered cubic (BCC) lattice whose uncut tets have dihedral angles from
   * a fixed, mesh-size-independent set. `warpFraction` (default `0.35`)
   * moves lattice vertices near the surface onto it, trading a small,
   * measured chance of non-manifold boundary edges (reported as
   * `numNonManifoldEdges`) for substantially better boundary tet quality;
   * `0` disables warping and gives an exactly watertight but lower-quality
   * boundary. See doc/remesh_volume.md for the measured tradeoff.
   * @throws {Error} when neither or both of `resolution`/`cellSize` are
   *   set, on a non-positive resolution/cell size, on a negative
   *   `warpFraction`, when the root lattice would exceed `maxCells`, when
   *   the cut output would exceed `maxTets`, and on any input the boundary
   *   extraction itself refuses.
   */
  remeshVolume(
    mesh: Mesh,
    resolution?: number[] | null,
    cellSize?: number,
    bounds?: number[] | null,
    padding?: number,
    paddingRelative?: number,
    maxCells?: number,
    maxTets?: number,
    warpFraction?: number,
    sign?: SdfSign,
    watertightCheck?: SdfWatertightCheck,
  ): {
    mesh: Mesh;
    numTets: number;
    numVerticesWarped: number;
    numTetsRejected: number;
    numNonManifoldEdges: number;
  };

  /**
   * ODT-remesh a tetrahedral mesh: raise its worst element quality by
   * relocating vertices AND flipping connectivity (2-3/3-2, predicate-free).
   * The genuine "ODT remeshing" sibling of `remeshVolume` (which generates a
   * fresh lattice mesh) and of `smooth` method `"odt"` (which only moves
   * points on fixed connectivity). Tet-only. The point set is invariant, so
   * point_data/field_data and Point regions carry; cell_data + Cell/Side
   * regions are dropped. With `preserveBoundary` the boundary surface is
   * exactly preserved. See doc/optimize_volume.md.
   * @throws {Error} when the mesh contains a non-`tetra` block or no tetra.
   */
  optimizeVolume(
    mesh: Mesh,
    maxIterations?: number,
    relocate?: boolean,
    flip?: boolean,
    preserveBoundary?: boolean,
    minImprovement?: number,
  ): {
    mesh: Mesh;
    numFlips: number;
    num23Flips: number;
    num32Flips: number;
    numVerticesMoved: number;
    numTets: number;
    minQualityBefore: number;
    minQualityAfter: number;
  };

  /**
   * Per-vertex mean (H) and Gaussian (K) curvature of a surface mesh -- the
   * signed distance's natural companion as a node feature. K is the angle
   * defect, H the cotangent Laplace-Beltrami operator. Writes
   * `curvature:mean`/`curvature:gaussian` point data, optionally
   * `curvature:area` and `curvature:principal`. `totalAngleDefect` is the
   * Gauss-Bonnet oracle: `2*pi*chi` exactly for a CLOSED surface (`4*pi` for
   * anything sphere-like), whatever the tessellation and whichever
   * `dualArea`. H's sign is orientation-dependent and K's is not, so check
   * `quality.inconsistentPairs` before trusting a sign. Never repairs its
   * input. See doc/curvature.md.
   * @throws {Error} on a non-surface input or an unknown region.
   */
  computeCurvature(
    mesh: Mesh,
    mean?: boolean,
    gaussian?: boolean,
    dualArea?: 'mixed-voronoi' | 'barycentric',
    includeBoundary?: boolean,
    recordArea?: boolean,
    recordPrincipal?: boolean,
    region?: string,
  ): {
    mesh: Mesh;
    numBoundary: number;
    numIsolated: number;
    numDegenerate: number;
    totalAngleDefect: number;
    quality: {
      boundaryEdges: number;
      nonManifoldEdges: number;
      inconsistentPairs: number;
      degenerateTriangles: number;
      watertight: boolean;
    };
  };

  /** Partition a mesh into submeshes by type, connected component, or tag. */
  split(mesh: Mesh, by: SplitBy, tagName?: string): { key: string; mesh: Mesh }[];

  /**
   * Convert the element representation: drop higher-order nodes
   * (`"linearize"`), decompose into same-dimension simplices (`"simplexify"`),
   * or promote linear cells to serendipity quadratic (`"elevate"`).
   * @throws {Error} on a polyhedron block under `"simplexify"`, or a
   *   full-Lagrange target (quad9/hexahedron27) under `"elevate"`.
   */
  convertCells(mesh: Mesh, mode?: ConvertCellsMode, recordParentIds?: boolean): Mesh;

  /**
   * Polyhedrally refine a mesh: one polyhedral child per face of every
   * eligible 3D cell, connected to a new interior point. Needs no per-type
   * template table -- tabulated types (reduced to corners for a quadratic
   * variant) and existing polyhedron blocks are handled uniformly.
   * Automatically conforming, unlike `refine`. Non-3D blocks and the
   * full-Lagrange family (no face table) pass through unchanged. Unlike
   * `convertCells`, there is no point map -- subdivide never prunes or
   * renumbers an original point.
   * @throws {Error} when a cell's faces are not a closed orientable surface.
   */
  subdivide(mesh: Mesh, recordParentIds?: boolean): Mesh;

  /**
   * Polyhedrally coarsen a mesh: merge groups of cells into single larger
   * polyhedral cells via greedy seed-and-grow over the mesh's shared-face
   * dual, absorbing face-adjacent neighbours into a group until it reaches
   * `targetGroupSize` (default 8; a short group at a mesh boundary or
   * pocket is expected, not an error). `targetGroupSize=1` groups every
   * cell by itself. Non-volume blocks pass through unchanged; points are
   * never pruned or renumbered (`clean(mesh, ..., true)` is the follow-up
   * for a minimal point set).
   * @throws {Error} when targetGroupSize is 0, or the mesh contains a face
   *   shared by three or more cells (non-manifold).
   */
  agglomerate(mesh: Mesh, targetGroupSize?: number): Mesh;

  /**
   * Refine a mesh, subdividing cells into same-type children (`line` → 2,
   * `triangle` → 4, `quad` → 4, `tetra` → 8, `wedge` → 8, `hexahedron` → 8).
   * New nodes sit at edge / quad-face / body midpoints and are shared between
   * neighbouring cells, so the result has no hanging nodes. `levels` applies
   * the templates repeatedly; `0` returns an unchanged copy.
   *
   * With `options` naming a SUBSET of the cells — at most one of `cells`,
   * `region` and `array` — only those are refined and the hanging nodes that
   * leaves are resolved by `closure`, so the output is still conforming.
   * @throws {Error} on a higher-order cell, a `pyramid`, or a ragged
   *   polygon/polyhedron block — none has a same-type subdivision — and on more
   *   than one selector, an unknown region, or an unusable predicate array.
   */
  refine(
    mesh: Mesh,
    levels?: number,
    recordParentIds?: boolean,
    options?: RefineOptions
  ): Mesh;

  /**
   * Decimate a SURFACE mesh by quadric-error-metric (Garland-Heckbert) edge
   * collapse — the resolution-reducing inverse of `refine`. Exactly one of
   * `ratio` (fraction of the triangulated faces to KEEP, in (0, 1]),
   * `targetFaces` and `maxError` must be non-negative. The output is
   * all-triangle with the block structure kept 1:1; boundary vertices
   * (once-used-edge test) and feature vertices (face normals differing by
   * more than `featureAngle` degrees) are pinned by default, and the link
   * condition plus a normal-flip guard reject any collapse that would change
   * topology or fold the surface. The index maps and the frozen mask are not
   * carried across the JS boundary, as on the other flat bindings.
   * @throws {Error} on a 3D volume mesh (extract the surface first),
   *   higher-order or ragged blocks, `line`/`vertex` blocks, an unknown
   *   `placement`, or a criterion count other than one.
   */
  decimate(
    mesh: Mesh,
    ratio?: number,
    targetFaces?: number,
    maxError?: number,
    placement?: DecimatePlacement,
    preserveBoundary?: boolean,
    preserveFeatures?: boolean,
    featureAngle?: number,
  ): {
    mesh: Mesh;
    facesRemoved: number;
    pointsRemoved: number;
    collapsesRejected: number;
    maxErrorApplied: number;
  };

  /**
   * Decompose a mesh into exactly `nparts` balanced pieces for domain
   * decomposition (the count-driven complement to `split`). Pieces keep the
   * input's cell-block structure 1:1 (empty blocks included, unlike `split`),
   * so concatenating them reproduces the input. The index maps are not
   * carried across the JS boundary — use `recordIds` for the
   * `partition:original_*_id` arrays, or `partitionLabels` for the raw
   * assignment. `weightsKey` names a scalar `cell_data` array of per-cell
   * weights. `ghostLayers > 0` grows each piece by that many shared-node BFS
   * layers of other parts' cells (a halo), tagged `partition:ghost`.
   * @throws {Error} on `method: 'kahip'` (KaHIP is never part of the WASM
   *   build; the message names `MESHIOPLUSPLUS_WITH_KAHIP`), `nparts < 1`,
   *   `ghostLayers < 0`, or a bad weights array.
   */
  partition(
    mesh: Mesh,
    nparts: number,
    method?: PartitionMethod,
    imbalance?: number,
    mode?: PartitionMode,
    seed?: number,
    recordIds?: boolean,
    ghostLayers?: number,
    weightsKey?: string,
  ): { partId: number; mesh: Mesh }[];

  /**
   * The per-cell part assignment only: one array per cell block
   * (block-aligned, like a `cell_data` entry), values in `[0, nparts)`.
   */
  partitionLabels(
    mesh: Mesh,
    nparts: number,
    method?: PartitionMethod,
    imbalance?: number,
    mode?: PartitionMode,
    seed?: number,
    weightsKey?: string,
  ): number[][];

  /** Read-only geometric statistics (bbox, areas, volumes, inverted count). */
  stats(mesh: Mesh): object;

  /**
   * Run `fn` with a provenance scope open, so any write inside it records the
   * richer block (source, target, operation chain, conversion assumptions,
   * timestamp) into the target format's header slot. With no scope open a
   * writer emits only the one-line credit, which is the default everywhere.
   * The scope is closed even if `fn` throws. See doc/provenance.md.
   * @param mode 0 = off, 1 = best-effort (default), 2 = required.
   */
  withProvenance<T>(mode: number | null | undefined, fn: () => T): T;
  /** Open a provenance scope. Prefer `withProvenance`, which cannot leak one. */
  provenanceBegin(mode?: number): void;
  /** Close the most recently opened provenance scope. */
  provenanceEnd(): void;
  /** Record one conversion assumption. A no-op outside a scope. */
  provenanceNote(category: string, detail: string): void;
  /** Record where the mesh came from. A no-op outside a scope. */
  provenanceSetSource(path: string, format: string): void;
  /** Record what was actually written. A no-op outside a scope. */
  provenanceSetTarget(
    format: string,
    encoding?: string,
    codec?: string,
    floatFormat?: string,
  ): void;
  /** Recover the provenance block a file carries, if any. */
  readProvenance(path: string): { lines: string[]; recognised: boolean };

  /**
   * Data operations: act on `point_data`/`cell_data`/`field_data` only -- the
   * geometry is never modified. See doc/data_operations.md. Note the mesh
   * object's flat, shapeless array shape means these only support scalar
   * (1-component) arrays; a vector/tensor field cannot be round-tripped
   * through the WASM boundary at all.
   */

  /** Drop the named arrays at `location`. Empty `names` is a no-op. */
  dataDrop(mesh: Mesh, location: DataLocation, names?: string[], ignoreMissing?: boolean): Mesh;

  /** Keep only the named arrays at `location`; other locations are untouched. */
  dataKeep(mesh: Mesh, location: DataLocation, names?: string[], ignoreMissing?: boolean): Mesh;

  /** Rename one array, preserving its values and dtype. */
  dataRename(mesh: Mesh, location: DataLocation, from: string, to: string): Mesh;

  /** Average point_data onto the cells (mean over each cell's own nodes). */
  dataPointToCell(mesh: Mesh, names?: string[], suffix?: string): Mesh;

  /** Average cell_data onto the points, optionally measure-weighted. */
  dataCellToPoint(mesh: Mesh, names?: string[], weight?: CellPointWeight, suffix?: string): Mesh;

  /**
   * Evaluate an elementwise expression (`+ - * /`, parens, numbers, array
   * names, `abs`/`sqrt`/`min`/`max`/`norm`) and store the result as a new
   * array at `location`.
   * @throws {Error} on any lexical, syntactic, or name-resolution error.
   */
  dataCalc(
    mesh: Mesh,
    expression: string,
    location: DataLocation,
    outputName: string,
    overwrite?: boolean,
  ): Mesh;

  /** Clamp / normalize / standardize the selected arrays' values. */
  dataCondition(
    mesh: Mesh,
    location: DataLocation,
    names?: string[],
    mode?: ConditionMode,
    lo?: number,
    hi?: number,
    scope?: ConditionScope,
    nanPolicy?: NanPolicy,
    nanReplacement?: number,
    suffix?: string,
  ): Mesh;

  /** Read-only per-array summary of every data array the mesh carries. */
  dataInfo(mesh: Mesh): DataArrayInfo[];

  /**
   * Cell-measure-weighted total/mean of one or more cell_data arrays --
   * gradient's integration counterpart. `arrays` empty/undefined means
   * every cell_data array. A point_data-only name throws naming
   * dataPointToCell as the fix. See doc/field_integration.md.
   */
  dataIntegrate(mesh: Mesh, arrays?: string[]): FieldIntegralArray[];

  /**
   * Open a transient (time-series) XDMF writer on the virtual filesystem --
   * the one stateful object in this API. See {@link XdmfTimeSeriesWriter} for
   * the shape, why it is a handle, and how to get the resulting file(s) out of
   * MEMFS (`'HDF'` writes **two**: the `.xdmf` and a sibling `.h5`).
   *
   * @param path virtual FS path of the `.xdmf`/`.xmf` light-data file.
   * @param options.dataFormat `'HDF'` (default; the shipped artifact links a
   *   wasm32 HDF5, so this works here), `'XML'` or `'Binary'`.
   * @param options.gzipLevel gzip level for `'HDF'` datasets; negative (the
   *   default) means uncompressed. Ignored by the other formats.
   * @param options.mode `'append'` continues a series already at `path`
   *   instead of overwriting it; a path with no file yet is simply a fresh
   *   series, so a restartable caller can pass it unconditionally.
   * @param options.autoFlush flush the light data after every `writeData`
   *   (default false). Off by default because a flush re-serializes the whole
   *   document, making per-step flushing quadratic in the step count.
   * @throws {Error} on an unrecognized `dataFormat` or `mode`.
   */
  createXdmfTimeSeriesWriter(
    path: string,
    options?: {
      dataFormat?: XdmfDataFormat;
      gzipLevel?: number;
      mode?: 'truncate' | 'append';
      autoFlush?: boolean;
    },
  ): XdmfTimeSeriesWriter;
}

/**
 * Instantiate a fresh, independent meshio++ WASM module instance. Safe to
 * call more than once (e.g. one instance per Web Worker).
 *
 * The package ships two native artifacts: the sequential `meshioplusplus_wasm`
 * and the threaded (OpenMP over Wasm threads) `meshioplusplus_wasm_mt`. The
 * threaded build is faster for mesh operations but only loads where the page is
 * cross-origin isolated (COOP `same-origin` + COEP `require-corp`); it aborts on
 * instantiation otherwise. `variant: 'auto'` (default) picks the threaded build
 * under Node and in a cross-origin-isolated browser, else the sequential one.
 *
 * @param moduleOverrides forwarded as-is to the Emscripten module factory
 *   (e.g. `{ locateFile }` to relocate the `.wasm` binary for a bundler/CDN).
 *   `locateFile` receives the requested filename, so return the URL matching the
 *   loaded variant (`meshioplusplus_wasm.wasm` or `meshioplusplus_wasm_mt.wasm`).
 * @param options.variant which native artifact to load: `'auto'` (default),
 *   `'mt'` (force threaded), or `'seq'` (force sequential).
 */
export function loadMeshioPlusPlus(
    moduleOverrides?: object,
    options?: { variant?: 'auto' | 'mt' | 'seq' },
): Promise<MeshioPlusPlusModule>;

export default loadMeshioPlusPlus;
