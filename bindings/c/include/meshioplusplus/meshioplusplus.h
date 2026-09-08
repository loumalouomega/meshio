/*  ██████   ██████ ██████████  █████████  █████   █████ █████    ███████
 * ░░██████ ██████ ░░███░░░░░█ ███░░░░░███░░███   ░░███ ░░███   ███░░░░░███      ███         ███
 *  ░███░█████░███  ░███  █ ░ ░███    ░░░  ░███    ░███  ░███  ███     ░░███    ░███        ░███
 *  ░███░░███ ░███  ░██████   ░░█████████  ░███████████  ░███ ░███      ░███ ███████████ ███████████
 *  ░███ ░░░  ░███  ░███░░█    ░░░░░░░░███ ░███░░░░░███  ░███ ░███      ░███░░░░░███░░░ ░░░░░███░░░
 *  ░███      ░███  ░███ ░   █ ███    ░███ ░███    ░███  ░███ ░░███     ███     ░███        ░███
 *  █████     █████ ██████████░░█████████  █████   █████ █████ ░░░███████░      ░░░         ░░░
 * ░░░░░     ░░░░░ ░░░░░░░░░░  ░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░    ░░░░░░░
 *
 *
 *  License:         MIT License
 *                   meshio++ default license: LICENSE
 *
 *  Main authors:    Vicente Mataix Ferrandiz
 */

/**
 * @file meshioplusplus.h
 * @brief The meshio++ C API: a stable, pure-C99 interface to the C++ mesh
 *        I/O core, shipped as the `libmeshioplusplus` shared library. This is
 *        the only installed header -- the C++ headers behind it make no ABI
 *        promise.
 *
 * Conventions (the whole contract in five rules):
 *  1. Every fallible function returns a `mio_status` (`MIO_OK == 0`) or, for
 *     pointer/count returns, `NULL`/`-1`; the failure message is retrievable
 *     via mio_last_error() (thread-local, valid until the next mio_* call on
 *     the same thread). No C++ exception ever crosses this ABI.
 *  2. Setters COPY caller memory; the caller's buffers can be freed as soon
 *     as the call returns.
 *  3. Getters are ZERO-COPY: returned data pointers alias mesh-owned memory
 *     and stay valid until the next mutating `mio_mesh_*` call on that mesh
 *     or mio_mesh_free() -- read-only accessors never invalidate them.
 *  4. Arrays are row-major (C order). Points are `(num_points, dim)`,
 *     connectivity `(num_cells, nodes_per_cell)` with 0-based node indices.
 *  5. String getters use the snprintf convention: they copy at most
 *     `buflen - 1` bytes plus a NUL into `buf` (when `buflen > 0`) and
 *     return the full length excluding the NUL, or -1 on error.
 *
 * File I/O (`mio_read`/`mio_write`/`mio_convert`) infers the format from the
 * path's extension when `format` is NULL or ""; ambiguous extensions default
 * to `.msh` -> gmsh and `.inp` -> abaqus (pass "ansys"/"freefem"/"ansysinp"
 * explicitly instead). Formats backed by optional dependencies (cgns, h5m,
 * hmf, med need HDF5; exodus needs netCDF) exist only in builds configured
 * with them -- probe with mio_format_readable()/mio_format_writable().
 */

#ifndef MESHIOPLUSPLUS_H
#define MESHIOPLUSPLUS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(MIO_SHARED)
#ifdef MIO_BUILDING
#define MIO_API __declspec(dllexport)
#else
#define MIO_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#define MIO_API __attribute__((visibility("default")))
#else
#define MIO_API
#endif

/** Opaque mesh handle. Create with mio_mesh_create()/mio_read(); destroy
 *  with mio_mesh_free(). Not thread-safe per handle (distinct handles may be
 *  used from distinct threads freely). */
typedef struct mio_mesh mio_mesh;

/** Opaque result of mio_reorder(): owns the renumbered mesh plus the applied
 *  node/cell permutations. Destroy with mio_reorder_result_free(). */
typedef struct mio_reorder_result mio_reorder_result;

/** Opaque result of mio_diff(): the structured mesh comparison report. Destroy
 *  with mio_diff_result_free(). */
typedef struct mio_diff_result mio_diff_result;

/** Opaque result of mio_split(): the ordered submesh pieces. Destroy with
 *  mio_split_result_free(). */
typedef struct mio_split_result mio_split_result;
typedef struct mio_sequence mio_sequence;

/** Opaque result of mio_convert_cells(): the converted mesh plus the point/cell
 *  index maps. Destroy with mio_convert_cells_result_free(). */
typedef struct mio_convert_cells_result mio_convert_cells_result;

/** Opaque result of mio_subdivide(): the subdivided mesh plus the per-block
 *  cell index maps. Unlike mio_convert_cells_result, there is no point map --
 *  subdivide never prunes or renumbers an original point. Destroy with
 *  mio_subdivide_result_free(). */
typedef struct mio_subdivide_result mio_subdivide_result;

/** Opaque result of mio_agglomerate(): the coarsened mesh plus a single FLAT
 *  cell index map (unlike mio_subdivide_result's per-block one -- an output
 *  cell's index is a function of which group it joined, not which input
 *  block it came from). There is likewise no point map. Destroy with
 *  mio_agglomerate_result_free(). */
typedef struct mio_agglomerate_result mio_agglomerate_result;

/** Opaque result of mio_refine(): the refined mesh plus the point/cell index
 *  maps. Destroy with mio_refine_result_free(). */
typedef struct mio_refine_result mio_refine_result;

/** Opaque result of mio_decimate(): the decimated mesh plus the point/cell
 *  index maps and the collapse summary. Destroy with
 *  mio_decimate_result_free(). */
typedef struct mio_decimate_result mio_decimate_result;

/** Opaque result of mio_decimate_volume(): the decimated TET mesh plus the
 *  point/cell index maps and the collapse summary. A separate result type
 *  from mio_decimate_result -- decimate_volume is a separate operation, not a
 *  mode on decimate. Destroy with mio_decimate_volume_result_free(). */
typedef struct mio_decimate_volume_result mio_decimate_volume_result;

/** Opaque result of mio_partition(): the pieces plus their point/cell index
 *  maps. Destroy with mio_partition_result_free(). */
typedef struct mio_partition_result mio_partition_result;

typedef enum mio_status {
    MIO_OK = 0,           /**< success */
    MIO_ERR_READ = 1,     /**< file could not be parsed / read-side failure */
    MIO_ERR_WRITE = 2,    /**< mesh could not be serialized / write-side failure */
    MIO_ERR_INVALID_ARG = 3, /**< NULL handle/pointer, bad dtype, bad shape, ... */
    MIO_ERR_NOT_FOUND = 4,   /**< unknown format/data name/block index */
    MIO_ERR_UNSUPPORTED = 5, /**< valid but unsupported (e.g. ragged connectivity) */
    MIO_ERR_INTERNAL = 6     /**< unexpected failure; see mio_last_error() */
} mio_status;

/** Overall outcome of comparing two meshes (see mio_diff()), in increasing
 *  order of severity — mirrors meshioplusplus::DiffVerdict. */
typedef enum mio_diff_verdict {
    MIO_DIFF_IDENTICAL = 0,             /**< structurally equal, values bitwise-equal */
    MIO_DIFF_EQUAL_WITHIN_TOLERANCE = 1, /**< values drifted but within tolerance */
    MIO_DIFF_DIFFERENT = 2               /**< a structural or beyond-tolerance difference */
} mio_diff_verdict;

/** Element dtypes, matching numpy's fixed-width scalars. */
typedef enum mio_dtype {
    MIO_FLOAT32 = 0,
    MIO_FLOAT64 = 1,
    MIO_INT8 = 2,
    MIO_INT16 = 3,
    MIO_INT32 = 4,
    MIO_INT64 = 5,
    MIO_UINT8 = 6,
    MIO_UINT16 = 7,
    MIO_UINT32 = 8,
    MIO_UINT64 = 9
} mio_dtype;

/** Maximum rank of any data array crossing this API (out-param shape arrays
 *  must have at least this many elements). */
#define MIO_MAX_NDIM 8

/* X(EnumName) -- one entry per meshio++ cell type, in the exact order of the
 * C++ `meshioplusplus::CellType` enum (src/cpp/include/meshioplusplus/
 * cell_type.hpp). c_api.cpp static_asserts every entry and the terminal
 * MIO_CELL_Custom against the C++ enum, so any drift between the two lists
 * is a compile error, never a runtime mismatch. */
#define MIO_CELL_TYPES(X)                                                                        \
    X(Vertex) X(Line) X(Line3) X(Line4) X(Line5) X(Line6) X(Line7) X(Line8) X(Line9) X(Line10)  \
    X(Line11) X(Triangle) X(Triangle6) X(Triangle10) X(Triangle15) X(Triangle21) X(Triangle28)  \
    X(Triangle36) X(Triangle45) X(Triangle55) X(Triangle66) X(Quad) X(Quad8) X(Quad9) X(Quad16) \
    X(Quad25) X(Quad36) X(Quad49) X(Quad64) X(Quad81) X(Quad100) X(Quad121) X(Tetra) X(Tetra10) \
    X(Tetra20) X(Tetra35) X(Tetra56) X(Tetra84) X(Tetra120) X(Tetra165) X(Tetra220) X(Tetra286) \
    X(Hexahedron) X(Hexahedron20) X(Hexahedron24) X(Hexahedron27) X(Hexahedron64)               \
    X(Hexahedron125) X(Hexahedron216) X(Hexahedron343) X(Hexahedron512) X(Hexahedron729)        \
    X(Hexahedron1000) X(Hexahedron1331) X(Wedge) X(Wedge15) X(Wedge18) X(Wedge40) X(Wedge75)    \
    X(Wedge126) X(Wedge196) X(Wedge288) X(Wedge405) X(Wedge550) X(Pyramid) X(Pyramid13)         \
    X(Pyramid14) X(Polygon) X(Polyhedron) X(VtkLagrangeCurve) X(VtkLagrangeTriangle)            \
    X(VtkLagrangeQuadrilateral) X(VtkLagrangeTetrahedron) X(VtkLagrangeHexahedron)              \
    X(VtkLagrangeWedge) X(VtkLagrangePyramid)

/** Integer mirror of the meshio++ cell-type table. The string names (e.g.
 *  "tetra10") are the primary representation everywhere in this API; the
 *  enum exists for C-side switching and metadata lookup. `MIO_CELL_Custom`
 *  is the catch-all for names outside the table. */
typedef enum mio_cell_type {
#define MIO_CELL_TYPE_ENUM(Name) MIO_CELL_##Name,
    MIO_CELL_TYPES(MIO_CELL_TYPE_ENUM)
#undef MIO_CELL_TYPE_ENUM
        MIO_CELL_Custom
} mio_cell_type;

/**
 * What kind of entity a region groups. Mirrors meshioplusplus::RegionKind.
 * Declared here (rather than down with the rest of the Named regions section)
 * so mio_read_metadata's region accessors can use it too.
 */
typedef enum mio_region_kind {
    MIO_REGION_POINT = 0,
    MIO_REGION_CELL = 1,
    MIO_REGION_SIDE = 2
} mio_region_kind;

/** Fixed-size description of one region (see mio_regions_info). */
typedef struct mio_region_info {
    int32_t kind;         /**< a mio_region_kind */
    int32_t dim;          /**< topological dimension, or -1 if unspecified */
    int64_t tag;          /**< format-native integer id, or -1 if none */
    int64_t num_entries;  /**< number of grouped entities (rows) */
    int64_t stride;       /**< values per entry: 2 for SIDE, else 1 (0 where entries aren't
                            *   carried at all, as in mio_read_metadata_region_info) */
} mio_region_info;

/* ---------------------------------------------------------------------
 * Version / build introspection
 * --------------------------------------------------------------------- */

/*
 * The RELEASE version as compile-time macros, so a consumer can feature-detect
 * with the preprocessor:
 *
 *     #if MIO_VERSION_AT_LEAST(9, 5, 0)
 *         mio_refine_ex(mesh, &opts);      // new in 9.5.0
 *     #else
 *         mio_refine(mesh, 1, 0);
 *     #endif
 *
 * These describe the header you COMPILED against. mio_version() below reports
 * the library you actually LINKED against, and with a shared library the two can
 * genuinely differ -- which is why both exist. (A third question, "are my
 * headers binary-compatible with that library", is answered by the ABI version
 * and its link-time sentinel; see doc/abi.md.)
 *
 * MIO_VERSION is major*10000 + minor*100 + patch, so it orders exactly like the
 * release and can be compared directly when the macro is not expressive enough.
 * These are static_asserted against the C++ MESHIOPLUSPLUS_VERSION_* macros in
 * c_api.cpp, and CMake hard-fails at configure time if either disagrees with
 * project(... VERSION ...), so the copies cannot drift.
 */
#define MIO_VERSION_MAJOR 10
#define MIO_VERSION_MINOR 37
#define MIO_VERSION_PATCH 0
#define MIO_VERSION (MIO_VERSION_MAJOR * 10000 + MIO_VERSION_MINOR * 100 + MIO_VERSION_PATCH)

/** Whether the header being compiled against is at least major.minor.patch. */
#define MIO_VERSION_AT_LEAST(major, minor, patch) \
    (MIO_VERSION >= ((major) * 10000 + (minor) * 100 + (patch)))

/** Whether the header being compiled against is older than major.minor.patch. */
#define MIO_VERSION_BEFORE(major, minor, patch) (!MIO_VERSION_AT_LEAST(major, minor, patch))

/**
 * @return the meshio++ version string of the LINKED library, e.g. "9.6.0"
 *         (static storage). Compare with the MIO_VERSION_* macros above, which
 *         describe the header this call was compiled against.
 */
MIO_API const char* mio_version(void);

/** @return the compile-time mesh backend: "meshio", "native", or "kratos"
 *          (static storage). */
MIO_API const char* mio_mesh_backend(void);

/** @return 1 if `format` (e.g. "gmsh", "vtu", "med") is readable in this
 *          build, 0 otherwise. */
MIO_API int mio_format_readable(const char* format);

/** @return 1 if `format` is writable in this build, 0 otherwise (read-only
 *          formats like "openfoam" report 0). */
MIO_API int mio_format_writable(const char* format);

/** @return the failure message of the most recent failed mio_* call on this
 *          thread ("" if none). Valid until the next mio_* call on the same
 *          thread; never NULL. */
MIO_API const char* mio_last_error(void);

/* ---------------------------------------------------------------------
 * Cell-type metadata
 * --------------------------------------------------------------------- */

/** @return the meshio name of `t` (e.g. "tetra10"; "" for MIO_CELL_Custom or
 *          out-of-range values). Static storage. */
MIO_API const char* mio_cell_type_name(mio_cell_type t);

/** @return the enum value for a meshio cell-type name, or MIO_CELL_Custom if
 *          `name` is NULL or not in the table. */
MIO_API mio_cell_type mio_cell_type_from_name(const char* name);

/** @return the fixed nodes-per-cell of `t`, or -1 for variable-size types
 *          (polygon, polyhedron, VTK Lagrange) and MIO_CELL_Custom. */
MIO_API int mio_cell_type_num_nodes(mio_cell_type t);

/** @return the topological dimension (0-3) of `t`, or -1 for MIO_CELL_Custom. */
MIO_API int mio_cell_type_dimension(mio_cell_type t);

/* ---------------------------------------------------------------------
 * Lifecycle & file I/O
 * --------------------------------------------------------------------- */

/** @return a new empty mesh, or NULL on allocation failure. */
MIO_API mio_mesh* mio_mesh_create(void);

/** Destroy a mesh and every pointer previously handed out from it. NULL-safe. */
MIO_API void mio_mesh_free(mio_mesh* mesh);

/**
 * Read a mesh file.
 * @param path   filesystem path.
 * @param format explicit format name, or NULL/"" to infer from the extension.
 * @return the mesh, or NULL on failure (see mio_last_error()).
 */
MIO_API mio_mesh* mio_read(const char* path, const char* format);

/**
 * Options narrowing what a read materializes (selective / partial reads).
 *
 * ABI NOTE: this struct is part of the installed library's permanent ABI.
 * New fields may only be appended, replacing `reserved` capacity; never
 * reorder, resize or repurpose an existing field. Always zero-initialize
 * through mio_read_opts_init() rather than by hand, so fields added later
 * default sensibly in code compiled against an older header.
 */
typedef struct mio_read_opts {
    int points_only;  /**< read geometry only, skipping every data array */
    int metadata_only; /**< read the header/summary only (see mio_read_metadata) */
    /** NULL = every array; otherwise `num_arrays` names to keep. An array
     *  count of 0 with a non-NULL pointer means "no arrays" -- the distinction
     *  from NULL is deliberate and load-bearing. Names absent from the file
     *  are ignored. The pointed-to strings are copied during the call. */
    const char* const* arrays;
    int64_t num_arrays;
    int mmap_mode;      /**< 0 = auto, 1 = on, 2 = off */
    /** Which step of a multi-step file to materialize: 0 (the default) is the
     *  first, preserving the historical behaviour; negative counts from the end.
     *  Out of range fails the call rather than clamping. Honoured by formats
     *  carrying a time series (currently exodus); ignored by the rest.
     *  Takes one of the former `reserved` slots, keeping the struct's size and
     *  every preceding field's offset unchanged. */
    int64_t time_step;
    /** Nonzero downgrades "this reader cannot represent construct X" errors to
     *  a warning plus a skip, where the reader can do that and still return a
     *  correct mesh (currently mdpa's Table/Geometries/Mesh/Constraints blocks).
     *  It is NOT "ignore all errors": a malformed file, a truncated block or a
     *  bad node reference still fails the call. 0 (the default) preserves the
     *  historical behaviour. Takes one of the former `reserved` slots, keeping
     *  the struct's size and every preceding field's offset unchanged. */
    int64_t lenient;
    int64_t reserved[4]; /**< must be zero; room for additive growth */
} mio_read_opts;

/** Initialize `opts` to the defaults (read everything). Always use this. */
MIO_API void mio_read_opts_init(mio_read_opts* opts);

/**
 * Read a mesh file honouring `opts`. `mio_read` is exactly
 * `mio_read_ex(path, format, <defaults>)` and is unchanged.
 *
 * Formats without a native selective path are read in full; the options are
 * still honoured, just without the saving.
 * @return the mesh, or NULL on failure (see mio_last_error()).
 */
MIO_API mio_mesh* mio_read_ex(const char* path, const char* format, const mio_read_opts* opts);

/** Opaque file summary. Destroy with mio_read_metadata_free(). */
typedef struct mio_read_metadata mio_read_metadata;

/**
 * Summarize a mesh file without loading its heavy arrays.
 * @return a result handle (free with mio_read_metadata_free), or NULL on failure.
 */
MIO_API mio_read_metadata* mio_read_metadata_create(const char* path, const char* format);

/** @return number of points, or -1 on error. */
MIO_API int64_t mio_read_metadata_num_points(const mio_read_metadata* meta);
/** @return coordinate dimension, or -1 on error. */
MIO_API int64_t mio_read_metadata_point_dim(const mio_read_metadata* meta);
/** @return total cells over all blocks, or -1 on error. */
MIO_API int64_t mio_read_metadata_num_cells(const mio_read_metadata* meta);
/** @return number of cell blocks, or -1 on error. */
MIO_API int64_t mio_read_metadata_num_cell_blocks(const mio_read_metadata* meta);

/**
 * The `index`-th cell block's shape. Any out pointer may be NULL.
 * `num_nodes_per_cell` is 0 for a ragged block (`is_ragged` nonzero).
 */
MIO_API mio_status mio_read_metadata_cell_block(const mio_read_metadata* meta, int64_t index,
                                                int64_t* num_cells, int64_t* num_nodes_per_cell,
                                                int* is_ragged);

/**
 * Copy the `index`-th cell block's meshio type name into `buf`.
 * @return the required length excluding the NUL, or -1 on error.
 */
MIO_API int64_t mio_read_metadata_cell_block_type(const mio_read_metadata* meta, int64_t index,
                                                  char* buf, int64_t buflen);

/**
 * @return number of time steps the file records, or -1 on error.
 *
 * 0 for a format with no time concept. This is the count `mio_read_opts.time_step`
 * may name, so it is what makes a step request checkable before issuing it.
 */
MIO_API int64_t mio_read_metadata_num_time_values(const mio_read_metadata* meta);

/**
 * Copy the recorded time values into `out` (at most `count` of them).
 * @return the number written, or -1 on error.
 */
MIO_API int64_t mio_read_metadata_time_values(const mio_read_metadata* meta, double* out,
                                              int64_t count);

/**
 * @return number of named regions the file carries, or -1 on error.
 *
 * Populated whenever the summary came from an already-read mesh (every
 * fallback path, and any format -- like exodus -- that always falls back); 0
 * on a native metadata path that declined a full read, since none of the
 * formats with such a path (VTU/VTP/XDMF/Gmsh 4.1) currently map regions at
 * all, so this is never a wrong answer.
 */
MIO_API int64_t mio_read_metadata_num_regions(const mio_read_metadata* meta);

/**
 * Copy the `index`-th region's name into `buf`.
 * @return the required length excluding the NUL, or -1 on error.
 */
MIO_API int64_t mio_read_metadata_region_name(const mio_read_metadata* meta, int64_t index,
                                              char* buf, int64_t buflen);

/**
 * The `index`-th region's kind/dim/tag/entry count. Reuses `mio_region_info`'s
 * shape (see the Named regions section below); `stride` is meaningless here
 * (no entries are carried) and is always 0.
 */
MIO_API mio_status mio_read_metadata_region_info(const mio_read_metadata* meta, int64_t index,
                                                 mio_region_info* out);

/** @return number of data-array names at `location` (a mio_data_location), or -1. */
MIO_API int64_t mio_read_metadata_num_names(const mio_read_metadata* meta, int location);

/**
 * Copy the `index`-th data-array name at `location` into `buf`.
 * @return the required length excluding the NUL, or -1 on error.
 */
MIO_API int64_t mio_read_metadata_name(const mio_read_metadata* meta, int location, int64_t index,
                                       char* buf, int64_t buflen);

/**
 * The bounding box, when one was computed.
 * @return MIO_OK and fills `min`/`max` (3 doubles each, either may be NULL), or
 *         MIO_ERR_NOT_FOUND when no box is available -- which is the normal
 *         case for a native summary, since computing it would mean decoding the
 *         point coordinates and defeating the purpose.
 */
MIO_API mio_status mio_read_metadata_bbox(const mio_read_metadata* meta, double* min, double* max);

/**
 * Whether the whole file had to be read to produce this summary.
 * @return 1 when the format has no header-only path (the summary is still
 *         correct, just not cheap), 0 when a native path was used, -1 on error.
 */
MIO_API int mio_read_metadata_fell_back(const mio_read_metadata* meta);

/**
 * The provenance block the file carries (v10.17.0), one line per entry, comment
 * punctuation already stripped -- the read half of mio_provenance_*. Lines are
 * returned as found rather than re-parsed into fields: a block can have been
 * hand-edited, truncated by a fixed-width slot, or written by a later release
 * carrying fields this build has never heard of.
 * @return the number of lines, or -1 on error.
 */
MIO_API int64_t mio_read_metadata_num_provenance_lines(const mio_read_metadata* meta);

/**
 * Copy provenance line `index` into `out` (NUL-terminated, truncated to `cap`).
 * @return the required buffer size including the NUL, or -1 on error.
 */
MIO_API int64_t mio_read_metadata_provenance_line(const mio_read_metadata* meta, int64_t index,
                                                  char* out, int64_t cap);

/**
 * Whether the block's first line is meshio++'s own tag format.
 * @return 1 when it is, 0 when the file carries no block or one that does not
 *         start the way this library writes one, -1 on error.
 */
MIO_API int mio_read_metadata_provenance_recognised(const mio_read_metadata* meta);

/** Destroy a summary handle. Safe to call with NULL. */
MIO_API void mio_read_metadata_free(mio_read_metadata* meta);

/** @return 1 if `format` has a native selective-read path, 0 if not, -1 on error. */
MIO_API int mio_reader_supports_options(const char* format);

/** Write a mesh. `format` as in mio_read(). */
MIO_API mio_status mio_write(const char* path, const mio_mesh* mesh, const char* format);

/** ASCII/binary selection for mio_write_opts.encoding. */
typedef enum mio_write_encoding {
    MIO_ENCODING_DEFAULT = 0, /**< the format's own default (unchanged behaviour) */
    MIO_ENCODING_ASCII = 1,
    MIO_ENCODING_BINARY = 2
} mio_write_encoding;

/** Block compression codec for mio_write_opts.codec (vti/vtu/vtp only). */
typedef enum mio_write_codec {
    MIO_CODEC_DEFAULT = 0, /**< leave the format's default in place */
    MIO_CODEC_NONE = 1,
    MIO_CODEC_ZLIB = 2,
    MIO_CODEC_LZ4 = 3,  /**< vtkLZ4DataCompressor; ParaView-readable */
    MIO_CODEC_ZSTD = 4  /**< a meshio++ extension; ParaView cannot read it */
} mio_write_codec;

/**
 * Options for how a mesh is written -- the symmetric counterpart of
 * mio_read_opts.
 *
 * ABI NOTE: identical discipline to mio_read_opts. New fields may only be
 * appended, replacing `reserved` capacity; never reorder, resize or repurpose
 * an existing field. Always zero-initialize through mio_write_opts_init().
 *
 * An option the target format cannot honour FAILS the call rather than being
 * ignored -- asking for zstd in a Gmsh file is a mistake worth reporting.
 */
typedef struct mio_write_opts {
    int encoding;      /**< a mio_write_encoding value */
    int codec;         /**< a mio_write_codec value; vti/vtu/vtp only */
    /** printf-style float format for the ASCII writers that take one (e.g.
     *  ".16e"). NULL or empty keeps the writer's own default. Copied during
     *  the call. Currently honoured by flac3d. */
    const char* float_format;
    int64_t reserved[5]; /**< must be zero; room for additive growth */
} mio_write_opts;

/** Initialize `opts` to the defaults (exactly mio_write's behaviour). */
MIO_API void mio_write_opts_init(mio_write_opts* opts);

/**
 * Write a mesh honouring `opts`. `mio_write(path, mesh, format)` is exactly
 * `mio_write_ex(path, mesh, format, <defaults>)` and is unchanged.
 * @return MIO_OK, or an error code (see mio_last_error()).
 */
MIO_API mio_status mio_write_ex(const char* path, const mio_mesh* mesh, const char* format,
                                const mio_write_opts* opts);

/**
 * Provenance (v10.16.0, see detail/provenance.hpp for the full design): the
 * opt-in richer block a writer with a free-text header slot can render
 * (source, target, operation chain, conversion assumptions, timestamp)
 * alongside the unconditional one-line tag every writer has emitted since
 * v10.15.0. Mode is one of the mio_provenance_mode values below; Off (0)
 * reproduces that unconditional tag exactly, which is what every mio_write
 * call already does with no scope open.
 */
typedef enum mio_provenance_mode {
    MIO_PROVENANCE_OFF = 0,         /**< the default: just the one-line tag */
    MIO_PROVENANCE_BEST_EFFORT = 1, /**< the full block where the target slot allows it */
    MIO_PROVENANCE_REQUIRED = 2     /**< the full block, or MIO_ERR_UNSUPPORTED naming why not */
} mio_provenance_mode;

/**
 * Opens a provenance scope for the current thread. Nests: a later
 * mio_provenance_scope_end() restores whatever was active before this call
 * (including "nothing", the default). Every mio_write/mio_write_ex call made
 * while a scope is open renders under it.
 * @return MIO_OK, or MIO_ERR_INVALID_ARG if `mode` is not a mio_provenance_mode value.
 */
MIO_API mio_status mio_provenance_scope_begin(int mode);

/** Closes the most recently opened scope. Call exactly once per matching
 *  mio_provenance_scope_begin(); a stray call with none open is a no-op. */
MIO_API void mio_provenance_scope_end(void);

/** Records one conversion assumption against the active scope (a short
 *  stable `category` slug plus a human-readable `detail`). A no-op outside a
 *  scope, so it is safe to call unconditionally from any writer. Duplicate
 *  (category, detail) pairs are collapsed. */
MIO_API void mio_provenance_note(const char* category, const char* detail);

/** Sets the source path/format on the active record. A no-op outside a scope. */
MIO_API void mio_provenance_set_source(const char* path, const char* format);

/** Sets the target format/encoding/codec/float_format on the active record
 *  (pass "" for a field that does not apply). A no-op outside a scope. */
MIO_API void mio_provenance_set_target(const char* format, const char* encoding,
                                       const char* codec, const char* float_format);

/** Read `in_path` and immediately write it to `out_path` (the CLI's
 *  `convert`), without materializing a handle for the caller. */
MIO_API mio_status mio_convert(const char* in_path, const char* in_format, const char* out_path,
                               const char* out_format);

/* ---------------------------------------------------------------------
 * Mesh operations (computations on a mesh, not file formats)
 * --------------------------------------------------------------------- */

/**
 * Extract the boundary of a mesh's highest-dimension cells as a new mesh.
 * Volume cells -> boundary faces; a 2D surface mesh -> boundary edges.
 * @param mesh             input mesh.
 * @param record_parent_ids nonzero to attach an int64 cell_data
 *                          "surface:parent_cell" (owning input-cell index).
 * @return the boundary mesh, or NULL on failure (see mio_last_error()).
 */
MIO_API mio_mesh* mio_extract_surface(const mio_mesh* mesh, int record_parent_ids);

/**
 * Extract the boundary skin of a volume mesh as a new surface mesh.
 * @param mesh      input volume mesh.
 * @param linearize nonzero to emit only corner nodes (triangle/quad output).
 * @return the skin mesh, or NULL on failure (see mio_last_error()).
 */
MIO_API mio_mesh* mio_extract_skin(const mio_mesh* mesh, int linearize);

/**
 * Compute per-cell quality metrics and return a copy of the mesh with them
 * attached as cell_data (names "quality:<metric>"; read them back with the
 * cell-data accessors).
 * @param mesh input mesh.
 * @return the annotated mesh, or NULL on failure (see mio_last_error()).
 */
MIO_API mio_mesh* mio_attach_quality(const mio_mesh* mesh);

/**
 * Report aggregate quality counts. Any out-param may be NULL.
 * @param mesh            input mesh.
 * @param num_cells       total cells scored.
 * @param num_inverted    cells with negative signed volume/area.
 * @param num_degenerate  near-zero (degenerate) cells.
 */
MIO_API mio_status mio_quality_counts(const mio_mesh* mesh, int64_t* num_cells,
                                      int64_t* num_inverted, int64_t* num_degenerate);

/**
 * Guess a mesh file's format from its contents (magic-byte sniffing).
 * @param path   filesystem path to an existing, readable file.
 * @param buf    caller buffer for the format name (may be NULL to query length).
 * @param buflen size of `buf`.
 * @return the untruncated length of the format name (0 if undetermined), or -1
 *         on error; the name is written to `buf` NUL-terminated when it fits.
 */
MIO_API int64_t mio_sniff_format(const char* path, char* buf, int64_t buflen);

/* ---------------------------------------------------------------------
 * Mesh operations (computations on a mesh, not file formats)
 * --------------------------------------------------------------------- */

/**
 * Extract the boundary of a mesh's highest-dimension cells as a new mesh.
 * Volume cells -> boundary faces; a 2D surface mesh -> boundary edges.
 * @param mesh             input mesh.
 * @param record_parent_ids nonzero to attach an int64 cell_data
 *                          "surface:parent_cell" (owning input-cell index).
 * @return the boundary mesh, or NULL on failure (see mio_last_error()).
 */
MIO_API mio_mesh* mio_extract_surface(const mio_mesh* mesh, int record_parent_ids);

/**
 * Extract the boundary skin of a volume mesh as a new surface mesh.
 * @param mesh      input volume mesh.
 * @param linearize nonzero to emit only corner nodes (triangle/quad output).
 * @return the skin mesh, or NULL on failure (see mio_last_error()).
 */
MIO_API mio_mesh* mio_extract_skin(const mio_mesh* mesh, int linearize);

/**
 * Compute per-cell quality metrics and return a copy of the mesh with them
 * attached as cell_data (names "quality:<metric>"; read them back with the
 * cell-data accessors).
 * @param mesh input mesh.
 * @return the annotated mesh, or NULL on failure (see mio_last_error()).
 */
MIO_API mio_mesh* mio_attach_quality(const mio_mesh* mesh);

/**
 * Report aggregate quality counts. Any out-param may be NULL.
 * @param mesh            input mesh.
 * @param num_cells       total cells scored.
 * @param num_inverted    cells with negative signed volume/area.
 * @param num_degenerate  near-zero (degenerate) cells.
 */
MIO_API mio_status mio_quality_counts(const mio_mesh* mesh, int64_t* num_cells,
                                      int64_t* num_inverted, int64_t* num_degenerate);

/**
 * Guess a mesh file's format from its contents (magic-byte sniffing).
 * @param path   filesystem path to an existing, readable file.
 * @param buf    caller buffer for the format name (may be NULL to query length).
 * @param buflen size of `buf`.
 * @return the untruncated length of the format name (0 if undetermined), or -1
 *         on error; the name is written to `buf` NUL-terminated when it fits.
 */
MIO_API int64_t mio_sniff_format(const char* path, char* buf, int64_t buflen);

/**
 * Merge two or more meshes into one (concatenate, with optional welding of
 * coincident nodes). Cell blocks are merged by type; connectivity is offset so
 * indices stay valid; point_data/cell_data are combined per `data_policy`.
 * point_sets/cell_sets are not carried across the C ABI (as elsewhere).
 * @param meshes   array of `count` non-NULL input mesh handles.
 * @param count    number of input meshes (>= 1).
 * @param weld     nonzero to merge coincident nodes within `atol`.
 * @param atol     absolute coincidence tolerance for welding.
 * @param source_tag nonzero to add an int64 "source_mesh_id" cell_data.
 * @param data_policy 0 = intersection (drop partial keys), 1 = fill (NaN).
 * @param drop_duplicate_cells nonzero (with `weld`) to drop cells that become
 *                             identical after welding.
 * @return the merged mesh (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_merge(const mio_mesh* const* meshes, int64_t count, int weld, double atol,
                            int source_tag, int data_policy, int drop_duplicate_cells);

/**
 * Apply an affine transform to a mesh's point coordinates (translate / scale /
 * rotate / general matrix). Connectivity and data are carried through; only the
 * points change (plus optional vector/tensor point_data rotation).
 * @param mesh   input mesh.
 * @param matrix a row-major 4x4 affine matrix (16 doubles); point p maps to
 *               M * [p.x, p.y, p.z, 1].
 * @param rotate_vector_data nonzero to rotate vector (trailing dim 3) and tensor
 *               (trailing dim 9) point_data by the transform's 3x3 linear block.
 * @return the transformed mesh (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_transform(const mio_mesh* mesh, const double* matrix,
                                int rotate_vector_data);

/**
 * Clean a mesh in one pass: weld coincident points, drop degenerate/duplicate
 * cells, remove orphaned points. Each step is toggleable; the removal counts are
 * returned through the (optional) out-params. point_sets/cell_sets are not
 * carried across the C ABI.
 * @param mesh                    input mesh.
 * @param weld                    nonzero to fuse coincident points within `atol`.
 * @param atol                    absolute coincidence tolerance for welding.
 * @param remove_orphans          nonzero to drop points referenced by no cell.
 * @param drop_degenerate         nonzero to drop degenerate cells.
 * @param drop_duplicate_cells    nonzero to drop exact-duplicate cells.
 * @param points_welded           receives the number of points fused by welding.
 * @param points_removed_orphan   receives the number of orphaned points removed.
 * @param cells_dropped_degenerate receives the number of degenerate cells dropped.
 * @param cells_dropped_duplicate receives the number of duplicate cells dropped.
 * @return the cleaned mesh (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_clean(const mio_mesh* mesh, int weld, double atol, int remove_orphans,
                            int drop_degenerate, int drop_duplicate_cells, int64_t* points_welded,
                            int64_t* points_removed_orphan, int64_t* cells_dropped_degenerate,
                            int64_t* cells_dropped_duplicate);

/**
 * Smooth a mesh's point coordinates, leaving topology and data untouched. Only
 * the points move: connectivity, cell_data, field_data and point_data values are
 * carried through unchanged. The run summary is returned through the (optional)
 * out-params. The SmoothOptions `mFrozen` caller pin mask is not exposed across
 * the C ABI (documented flat-ABI gap).
 * @param mesh              input mesh.
 * @param method            "laplacian" or "taubin"; NULL means "taubin".
 * @param iterations        number of iterations (for Taubin one iteration is two
 *                          passes); zero or less returns an unchanged copy.
 * @param lambda            relaxation factor in (0, 1). **Negative means "this
 *                          method's own default"**: 0.5 for laplacian, 0.33 for
 *                          taubin.
 * @param mu                taubin only: the un-shrinking factor, which must
 *                          satisfy mu < -lambda < 0. Ignored by laplacian.
 * @param fix_boundary      nonzero to pin every node on a boundary facet.
 * @param preserve_features nonzero to additionally pin boundary nodes whose
 *                          incident boundary facets meet above feature_angle.
 * @param feature_angle     feature angle between boundary facet normals, in
 *                          degrees (0 = coplanar, 90 = the edge of a box).
 * @param guard_inversion   nonzero to reject any move that would flip the signed
 *                          measure of an incident cell, and count the rejection.
 * @param nodes_moved       receives the number of points whose net displacement
 *                          exceeded the relative move tolerance.
 * @param max_displacement  receives the largest net displacement, in mesh units.
 * @param skipped_inversion receives the number of (node, pass) events in which
 *                          the inversion guard rejected a move.
 * @return the smoothed mesh (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_smooth(const mio_mesh* mesh, const char* method, int iterations,
                             double lambda, double mu, int fix_boundary, int preserve_features,
                             double feature_angle, int guard_inversion, int64_t* nodes_moved,
                             double* max_displacement, int64_t* skipped_inversion);

/**
 * Green-element undo: restore `fine`'s transitional (closure-only) cells back
 * to their original parent, read verbatim from `coarse` -- a lookup, not a
 * reconstruction, since `mio_refine` never renumbers or prunes points. Undoes
 * the known quality-degradation issue of repeated selective `mio_refine`
 * passes over the same region ("restore the parent and re-split from
 * scratch").
 *
 * `fine` must carry one int64 scalar cell_data array per block for each of
 * "refine:cell_id", "refine:parent_id" and "refine:level" -- i.e. it must have
 * come from an `mio_refine_ex` call with `record_hierarchy` AND
 * `record_levels` both set; fails naming the missing array otherwise.
 * `coarse` must be the exact mesh that `mio_refine_ex` call was run on (or an
 * equally-shaped mesh sharing its point/id space); a `refine:parent_id` that
 * does not resolve against `coarse` fails by name rather than guessing.
 *
 * The output keeps `fine`'s own block structure (same types, same order);
 * some blocks' row counts shrink where a green sibling group collapsed to
 * one substituted row. The six reserved refine:* arrays (four cell_data:
 * parent_cell/level/cell_id/parent_id, two point_data: entity/hanging) are
 * always dropped from the output -- they describe a hierarchy relationship
 * that is now stale. Points are never pruned or renumbered. Only a
 * single-pass (levels=1) hierarchy is supported; a multi-level `mio_refine_ex`
 * call's deeper branches fail by name.
 *
 * Its per-block cell maps and the `mFrozen`-style extras are not exposed on
 * the flat ABI, a documented gap like `mio_smooth`'s.
 * @param coarse             the mesh a prior `mio_refine_ex` call was run on.
 * @param fine               that call's output.
 * @param num_groups_undone  out: number of green sibling groups substituted
 *                            (nullable).
 * @param num_cells_removed  out: total green child cells removed (nullable).
 * @return the undone mesh (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_undo_green(const mio_mesh* coarse, const mio_mesh* fine,
                                 int64_t* num_groups_undone, int64_t* num_cells_removed);

/**
 * Sample data arrays from a source mesh onto a target mesh (cross-mesh field
 * transfer). Returns a copy of the target — geometry, connectivity and its own
 * data preserved exactly — with the requested source arrays sampled onto it:
 * source point_data at the target's points, source cell_data at the target's
 * cell centroids (always by nearest source-cell centroid, whatever the method).
 * Under "barycentric" the source is simplexified first, so on a quad/hex source
 * the result is the simplex-linear interpolant (triangles are evaluated in the
 * xy-plane); "nearest" copies the nearest source point's value bit-for-bit.
 * @param source        the mesh whose data is sampled.
 * @param target        the mesh receiving the samples.
 * @param method        "nearest" or "barycentric"; NULL means "nearest".
 * @param arrays        source array names to transfer, as an array of C strings
 *                      with an explicit count (the mio_data_* convention). NULL
 *                      or arrays_count <= 0 means every source point_data array;
 *                      cell_data transfers only when named explicitly.
 * @param arrays_count  number of entries in arrays (see above).
 * @param extrapolate   barycentric only: nonzero to give a target point outside
 *                      the source domain the nearest source point's value
 *                      instead of default_value.
 * @param default_value barycentric only: the fill value (every component) for a
 *                      target point outside the source domain when extrapolate
 *                      is zero.
 * @param on_conflict   what to do when a transferred name already exists on the
 *                      target: "error", "overwrite" or "suffix" (writes to
 *                      name + "_interp"); NULL means "error".
 * @return the target copy with the sampled arrays (free with mio_mesh_free), or
 *         NULL on failure.
 */
MIO_API mio_mesh* mio_interpolate(const mio_mesh* source, const mio_mesh* target,
                                  const char* method, const char* const* arrays,
                                  int64_t arrays_count, int extrapolate, double default_value,
                                  const char* on_conflict);

/**
 * Mass-preserving cross-mesh field transfer: an exact overlap-measure
 * weighted remap, so that over the region the two meshes share,
 * sum(target value * target measure) equals sum(source value * source
 * measure) — the property mio_interpolate's "barycentric" mode does not
 * have. Both meshes are simplexified (accepting ragged/polyhedron blocks for
 * free); every overlapping simplex pair is measured with an exact geometric
 * clip. Unlike mio_interpolate, a NULL/empty arrays list means every source
 * point_data AND cell_data array (one algorithm regardless of location).
 * Output arrays are always Float64.
 * @param source        the mesh whose data is sampled.
 * @param target        the mesh receiving the samples.
 * @param arrays        source array names to transfer, as an array of C
 *                      strings with an explicit count. NULL or
 *                      arrays_count <= 0 means every source point_data and
 *                      cell_data array.
 * @param arrays_count  number of entries in arrays (see above).
 * @param default_value the fill value (every component) for a target cell
 *                      whose covered fraction of source overlap is below the
 *                      coverage tolerance.
 * @param on_conflict   what to do when a transferred name already exists on
 *                      the target: "error", "overwrite" or "suffix" (writes
 *                      to name + "_interp"); NULL means "error".
 * @return the target copy with the remapped arrays (free with
 *         mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_conservative_interpolate(const mio_mesh* source, const mio_mesh* target,
                                               const char* const* arrays, int64_t arrays_count,
                                               double default_value, const char* on_conflict);

/**
 * Planar cross-section of a mesh: the intersection with a plane, one topological
 * dimension below the cut cells (a volume mesh yields a triangle/quad surface, a
 * 2D surface mesh yields a line mesh). The input is simplexified first
 * (marching tetrahedra); crossing points on shared edges are deduped so the
 * section is watertight, and each section cell inherits its parent's cell_data.
 * @param origin            a point on the cutting plane (3 doubles).
 * @param normal            the plane normal (3 doubles, non-zero; need not be
 *                          unit length).
 * @param record_parent_ids nonzero to attach an Int64 "slice:parent_cell"
 *                          cell_data array (per section cell, the global input
 *                          cell it was cut from).
 * @return the cross-section mesh (free with mio_mesh_free), empty when the plane
 *         misses the geometry, or NULL on failure.
 */
MIO_API mio_mesh* mio_slice(const mio_mesh* mesh, const double* origin, const double* normal,
                            int record_parent_ids);

/**
 * Isosurfaces / contours: the level set of a scalar point_data field, one
 * topological dimension below the cut cells (a volume mesh yields a
 * triangle/quad surface, a 2D surface mesh a line mesh). The data-driven
 * sibling of mio_slice, sharing its marching-tetrahedra cutter: crossing points
 * on shared edges are deduped so contours are watertight, faces are wound toward
 * increasing field, and a node exactly at the isovalue counts as being on the
 * positive side (so a plateau emits its boundary once, not twice).
 *
 * The field must be point_data: cell_data is piecewise constant and has no level
 * set, so naming one fails -- convert it with mio_data_cell_to_point first.
 * All contours land in the one returned mesh, tagged per cell with a Float64
 * "iso:value" and an Int64 "iso:index" (the ordinal in the ascending isovalue
 * list, which is the integer tag mio_split's "tag" criterion needs).
 * @param array_name        the point_data array to contour.
 * @param isovalues         the level values (n_isovalues doubles); sorted
 *                          ascending and de-duplicated before cutting.
 * @param n_isovalues       how many, at least one.
 * @param component         component of a multi-component array to contour;
 *                          negative for the row magnitude (in which case the
 *                          contoured value is approximate, not exact).
 * @param record_parent_ids nonzero to attach an Int64 "iso:parent_cell"
 *                          cell_data array (per contour cell, the global input
 *                          cell it was cut from).
 * @return the contour mesh (free with mio_mesh_free), empty when no isovalue
 *         crosses the field, or NULL on failure.
 */
MIO_API mio_mesh* mio_isosurface(const mio_mesh* mesh, const char* array_name,
                                 const double* isovalues, int n_isovalues, int component,
                                 int record_parent_ids);

/**
 * Differentiate a point_data field: its gradient, divergence or curl.
 *
 * The result is a copy of the input carrying one new array; geometry,
 * connectivity, regions and every existing array come through unchanged.
 *
 * Shapes: an nc-component input yields a gradient of 3*nc components, flat and
 * row-major as [component i][derivative j] at index i*3 + j, so a scalar gives
 * 3 and a 3-vector gives 9. Divergence gives 1 component and curl always 3;
 * both need an input of 2 or 3 components (a 2-component field reads as
 * (u, v, 0)). Output is always Float64.
 *
 * Cells that cannot be differentiated -- below the mesh's own topological
 * dimension, ragged, or a 3-D Lagrange type with no face table -- yield NaN and
 * are counted in num_skipped, never approximated. With "least-squares", a cell
 * whose neighbourhood is degenerate falls back to Green-Gauss and is counted in
 * num_fallback.
 *
 * @param mesh         input mesh.
 * @param array_name   name of the point_data array to differentiate. A
 *                     cell_data name is an error naming the fix, since a
 *                     piecewise-constant field has no derivative.
 * @param op           "gradient" (default), "divergence" or "curl"; NULL or ""
 *                     selects the default.
 * @param method       "green-gauss" (default) or "least-squares"; NULL or ""
 *                     selects the default.
 * @param location     "cell" (default) or "point"; NULL or "" selects the
 *                     default.
 * @param output_name  name for the produced array; NULL or "" uses
 *                     "<array_name>:<op>".
 * @param component    differentiate only this component of a multi-component
 *                     input (gradient only); negative means EVERY component.
 *                     Note this is the opposite of mio_isosurface's sentinel,
 *                     where negative means the row magnitude.
 * @param overwrite    nonzero to replace an existing array of the output name
 *                     instead of failing.
 * @param num_skipped  optional out: cells that produced a NaN row.
 * @param num_fallback optional out: least-squares cells that fell back.
 * @return the mesh carrying the produced array (free with mio_mesh_free), or
 *         NULL on failure.
 */
MIO_API mio_mesh* mio_gradient(const mio_mesh* mesh, const char* array_name, const char* op,
                               const char* method, const char* location, const char* output_name,
                               int component, int overwrite, int64_t* num_skipped,
                               int64_t* num_fallback);

/**
 * The Hessian (second derivative) of a scalar point_data field --
 * mio_gradient's companion one order further, for curvature-based adaptive
 * refinement.
 *
 * A composition of TWO mio_gradient calls, not a new numerical kernel: the
 * field is differentiated once (Point location, so the result is a genuine
 * point_data array), then that (n,3) gradient is differentiated again with
 * the default "gradient" operator, which is generic over its input's own
 * component count and so produces (n,9) -- the flattened row-major 3x3
 * Hessian, H[i][j] at index i*3+j. The result is a copy of the input
 * carrying one new array; geometry, connectivity, regions and every
 * existing array come through unchanged.
 *
 * A field that is at most LINEAR has an exactly zero Hessian everywhere
 * (its gradient is a constant, and Green-Gauss of a constant field is
 * trivially exact) -- the one mesh-shape-independent guarantee. For a
 * genuinely QUADRATIC field the composition is exact on a
 * structured/symmetric mesh away from its own boundary (measured, not
 * assumed) and a good, standard, but genuinely approximate curvature
 * estimate on an irregular mesh, because the mandatory intermediate
 * Uniform-weighted point-averaging step between the two passes is only
 * exact for a field that is constant over the averaged neighbourhood.
 *
 * Input must have exactly one component -- a vector field's Hessian is a
 * separate quantity per component; call this once per component instead of
 * passing a multi-component array.
 *
 * Cells that cannot be evaluated (the same reasons mio_gradient's own
 * num_skipped covers, structurally identical between the two internal
 * passes) yield NaN in the Hessian and are counted in num_skipped. With
 * "least-squares", a cell whose neighbourhood is degenerate in either pass
 * falls back to Green-Gauss and is counted in num_fallback (summed over
 * both passes).
 *
 * A curvature-driven refinement indicator needs no new API here: a scalar
 * Frobenius-norm indicator over the 9-component output is
 * `mio_data_calc(mesh, "curv = norm(`<array>:hessian`)", ...)`, ready for
 * `mio_refine_ex`'s predicate selector.
 *
 * @param mesh         input mesh.
 * @param array_name   name of the scalar point_data array to differentiate
 *                     twice. A cell_data name is an error naming the fix; a
 *                     name with more than one component is an error naming
 *                     the per-component workaround.
 * @param method       "green-gauss" (default) or "least-squares"; NULL or ""
 *                     selects the default. Forwarded to BOTH internal
 *                     mio_gradient calls.
 * @param location     "cell" (default) or "point"; NULL or "" selects the
 *                     default.
 * @param output_name  name for the produced array; NULL or "" uses
 *                     "<array_name>:hessian".
 * @param overwrite    nonzero to replace an existing array of the output name
 *                     instead of failing.
 * @param num_skipped  optional out: cells that produced a NaN row.
 * @param num_fallback optional out: least-squares cells that fell back, in
 *                     either internal pass.
 * @return the mesh carrying the produced array (free with mio_mesh_free), or
 *         NULL on failure.
 */
MIO_API mio_mesh* mio_hessian(const mio_mesh* mesh, const char* array_name, const char* method,
                              const char* location, const char* output_name, int overwrite,
                              int64_t* num_skipped, int64_t* num_fallback);

/**
 * Estimate the per-cell recovered-gradient (Zienkiewicz-Zhu) error of a
 * point_data field, and optionally mark cells for refinement.
 *
 * A composition of mio_gradient (Green-Gauss, cell location) with the
 * measure-weighted point<->cell averaging round trip: the indicator is
 * sqrt(|measure| * sum((recovered - raw)^2)) per cell. The result is a copy
 * of the input carrying "<output_name>" (Float64, default "error:zz") and,
 * when marking is not "none", "<marked_name>" (Int64 0/1, default
 * "error:marked") -- so refine's own --where predicate needs no change at
 * all: the intended use is `mio_refine_ex` with a "error:marked > 0.5"
 * selector. Geometry, connectivity, regions and every existing array come
 * through unchanged.
 *
 * A cell that cannot be evaluated (an unsupported/degenerate gradient cell, a
 * recovery neighbourhood with no finite contribution, or an unmeasurable
 * cell) reads NaN in the indicator array and 0 (never NaN) in the marking
 * array, and is excluded from global_error and from every marking policy's
 * cell count.
 *
 * @param mesh          input mesh.
 * @param array_name    name of the point_data array to estimate the error
 *                      of. A cell_data name is an error naming the fix, since
 *                      a piecewise-constant field has no derivative to
 *                      recover.
 * @param method        estimator family; NULL, "" or "zz" selects the only
 *                      one that exists today.
 * @param marking       "none" (default; NULL or "" also selects it),
 *                      "absolute", "fraction", or "dorfler".
 * @param marking_value meaning depends on marking: ignored for "none"; an
 *                      absolute indicator threshold for "absolute"; a
 *                      fraction in (0, 1] of cells for "fraction"; the
 *                      Doerfler bulk fraction theta in (0, 1] for "dorfler".
 * @param output_name   indicator array name; NULL or "" selects "error:zz".
 * @param marked_name   marking array name; NULL or "" selects
 *                      "error:marked". Ignored when marking is "none".
 * @param overwrite     nonzero to replace an existing array of an output
 *                      name instead of failing.
 * @param global_error  optional out: sqrt(sum of eta_K^2) over evaluable
 *                      cells -- the estimated global error.
 * @param num_skipped   optional out: cells that could not be evaluated.
 * @param num_marked    optional out: cells marked; always 0 for "none".
 * @return the mesh carrying the produced array(s) (free with mio_mesh_free),
 *         or NULL on failure.
 */
MIO_API mio_mesh* mio_estimate_error(const mio_mesh* mesh, const char* array_name,
                                     const char* method, const char* marking,
                                     double marking_value, const char* output_name,
                                     const char* marked_name, int overwrite,
                                     double* global_error, int64_t* num_skipped,
                                     int64_t* num_marked);

/**
 * Remesh a surface by approximated centroidal Voronoi diagram (ACVD)
 * clustering: replace its triangulation with a new, near-uniformly-sized,
 * well-shaped one at a caller-chosen vertex count. Unlike every other
 * resolution-changing operation this does not work on the input's own
 * triangulation, so the output has NEW points and NEW connectivity with no
 * correspondence to the input -- point_data/cell_data/named regions are
 * dropped, field_data is carried. See doc/remesh.md.
 * @param mesh                input surface mesh.
 * @param num_clusters        number of clusters, i.e. output vertices aimed
 *                             for; must be >= 4 and <= the subdivided
 *                             input's vertex count.
 * @param subdivide            uniform refine passes applied before
 *                             clustering (each multiplies triangle count by
 *                             4); negative picks the smallest count reaching
 *                             subsample_ratio items/cluster, capped at
 *                             max_subdivide; 0 disables subdivision.
 * @param subsample_ratio      items per cluster targeted by automatic
 *                             subdivision.
 * @param max_subdivide        ceiling on automatic subdivision (ignored when
 *                             subdivide is set explicitly).
 * @param max_iterations       maximum energy-minimisation sweeps per pass.
 * @param max_repair_passes    "split disconnected clusters, minimise again"
 *                             passes; 0 skips repair.
 * @param metric               NULL, "" or "isotropic" (area-weighted
 *                             centroidal distance) selects the default;
 *                             "quadric" selects the Garland-Heckbert
 *                             quadric-error, feature-preserving metric.
 * @param gradation            curvature-gradation exponent gamma in the item
 *                             weight area * kappa^gamma; 0.0 disables
 *                             gradation (curvature is not even computed) and
 *                             reproduces plain area weighting.
 * @param preserve_boundary    nonzero (default behaviour) detects the
 *                             input's open boundary (if any), seeds it
 *                             before the interior, and emits a `line` dual
 *                             cell along every boundary edge whose endpoints
 *                             land in different clusters; a no-op on a
 *                             closed mesh either way.
 * @param num_clusters_out     optional out: clusters actually produced
 *                             (mesh->NumPoints(); may be lower than
 *                             num_clusters, never higher).
 * @param num_iterations       optional out: energy-minimisation sweeps
 *                             performed, summed over every repair pass.
 * @param subdivide_applied    optional out: subdivision passes actually
 *                             applied (the resolved value of subdivide).
 * @param num_isolated_clusters optional out: clusters still disconnected
 *                             after repair; non-zero means the output may be
 *                             non-manifold near them.
 * @param num_non_manifold_vertices optional out: output vertices whose
 *                             incident dual-triangle fan is still not a
 *                             single loop/chain after repair; a distinct
 *                             cause of non-manifold output from
 *                             num_isolated_clusters, reported separately.
 * @return the remeshed mesh (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_remesh(const mio_mesh* mesh, int64_t num_clusters, int subdivide,
                             double subsample_ratio, int max_subdivide, int max_iterations,
                             int max_repair_passes, const char* metric, double gradation,
                             int preserve_boundary, int64_t* num_clusters_out,
                             int64_t* num_iterations, int* subdivide_applied,
                             int64_t* num_isolated_clusters,
                             int64_t* num_non_manifold_vertices);

/**
 * Options for `mio_remesh_ex`; use `mio_remesh_opts_init` to get the same
 * defaults as `mio_remesh`. Append-only: new fields consume `reserved`
 * slots so `sizeof` never changes for an already-compiled consumer -- see
 * `mio_refine_opts`.
 */
typedef struct mio_remesh_opts {
    /** Number of clusters, i.e. output vertices aimed for. */
    int64_t num_clusters;
    /** Uniform refine passes before clustering; negative = automatic. */
    int32_t subdivide;
    /** Ceiling on automatic subdivision. */
    int32_t max_subdivide;
    /** Items per cluster targeted by automatic subdivision. */
    double subsample_ratio;
    /** Maximum energy-minimisation sweeps per pass. */
    int32_t max_iterations;
    /** "Split disconnected clusters, minimise again" passes. */
    int32_t max_repair_passes;
    /** NULL/"" = "isotropic"; "quadric" or "anisotropic" select the others. */
    const char* metric;
    /** Curvature-gradation exponent gamma; 0.0 disables gradation. */
    double gradation;
    /** Nonzero (default) detects/pins the input's open boundary. */
    int32_t preserve_boundary;
    int32_t reserved_pad; /**< must be zero; keeps the double tail aligned */
    /**
     * Under metric="anisotropic", the maximum ratio between the two in-plane
     * target edge lengths a per-vertex curvature tensor may request; `1.0`
     * recovers the isotropic shape exactly. Must be >= 1.0. An error to set
     * away from `mio_remesh_opts_init`'s default under any other metric.
     */
    double max_anisotropy;
    double reserved_d[3]; /**< must be zero; room for additive double growth */
    int64_t reserved[4];  /**< must be zero; room for additive int64 growth */
} mio_remesh_opts;

/** Zero-initialize remesh options to the same defaults as `mio_remesh`. */
MIO_API void mio_remesh_opts_init(mio_remesh_opts* opts);

/** Output counters from `mio_remesh_ex`; every field is always written. */
typedef struct mio_remesh_report {
    /** Clusters actually produced; may be lower than requested, never higher. */
    int64_t num_clusters;
    /** Energy-minimisation sweeps performed, summed over every pass. */
    int64_t num_iterations;
    /** Subdivision passes actually applied (the resolved value of subdivide). */
    int32_t subdivide_applied;
    int32_t reserved_pad; /**< must be zero; keeps the int64 tail aligned */
    /** Clusters still disconnected after repair. */
    int64_t num_isolated_clusters;
    /** Output vertices whose incident dual-triangle fan is still not a
     *  single loop/chain after repair -- distinct from num_isolated_clusters. */
    int64_t num_non_manifold_vertices;
    int64_t reserved[4]; /**< must be zero; room for additive growth */
} mio_remesh_report;

/**
 * Remesh a surface, `mio_remesh`'s options-struct growth path -- the
 * `mio_refine_ex`/`mio_refine_opts` shape, needed because `mio_remesh`
 * itself is a flat function with no room to grow (it already changed once).
 * @param mesh   input surface mesh.
 * @param opts   options; NULL means every field at its `mio_remesh_opts_init`
 *               default (which still requires `num_clusters` to be set, so a
 *               NULL `opts` is only useful after `mio_remesh_opts_init` sets
 *               it -- pass a real struct in practice).
 * @param report optional out: the run summary; NULL to ignore it.
 * @return the remeshed mesh (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_remesh_ex(const mio_mesh* mesh, const mio_remesh_opts* opts,
                                mio_remesh_report* report);

/**
 * Crop a mesh to an axis-aligned bounding box (keep cells inside the box).
 * @param mesh       input mesh.
 * @param lo         box lower corner (3 doubles).
 * @param hi         box upper corner (3 doubles).
 * @param mode       0 = keep cells with ALL nodes inside, 1 = ANY node inside.
 * @param record_ids nonzero to attach crop:original_point_id / _cell_id data.
 * @return the cropped mesh (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_crop_bbox(const mio_mesh* mesh, const double* lo, const double* hi, int mode,
                                int record_ids);

/**
 * Crop a mesh to the half-space (p - point) . normal >= 0.
 * @param mesh       input mesh.
 * @param point      a point on the plane (3 doubles).
 * @param normal     the plane normal (3 doubles); the kept side is non-negative.
 * @param mode       0 = keep cells with ALL nodes inside, 1 = ANY node inside.
 * @param record_ids nonzero to attach crop:original_point_id / _cell_id data.
 * @return the cropped mesh (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_crop_plane(const mio_mesh* mesh, const double* point, const double* normal,
                                 int mode, int record_ids);

/**
 * Crop a mesh to the cells whose value in a scalar cell_data array satisfies a
 * comparison.
 *
 * There is deliberately NO `mode` parameter. The bbox and half-space crops test
 * *points* and then need a rule for reducing a cell's several nodes to one
 * verdict; a cell_data predicate is already one value per cell and has nothing
 * to reduce, so a mode here would mean nothing.
 *
 * Inside/outside a surface composes rather than being its own entry point:
 * attach distances with mio_distance_to_surface at MIO_SDF_CENTER, then crop on
 * "sdf:distance" < 0. The same one mode also crops by quality:*, by a material
 * id, or by anything mio_data_calc can produce.
 *
 * @param mesh       the mesh to crop.
 * @param array      the name of a scalar cell_data array covering every block.
 * @param compare    a mio_refine_compare -- the SAME vocabulary refine uses,
 *                   evaluated by the same C++ function, so the two operations
 *                   cannot drift on the boundary cases.
 * @param value      the right-hand side. A NON-FINITE cell value never matches,
 *                   whatever the comparison -- compute_quality reports NaN where
 *                   a metric does not apply, and predicating over quality:* on a
 *                   mixed mesh is the headline use case.
 * @param record_ids nonzero to attach crop:original_point_id / _cell_id.
 * @return the cropped mesh (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_crop_predicate(const mio_mesh* mesh, const char* array, int compare,
                                     double value, int record_ids);

/**
 * Split a mesh into pieces by cell type, connected component, or integer
 * cell_data tag. point_sets/cell_sets are not carried across the C ABI.
 * @param mesh     input mesh.
 * @param by       "type", "component", or "region"/"tag".
 * @param tag_name for "region"/"tag", the integer cell_data name (NULL/"" =
 *                 auto-detect the first integer cell_data); ignored otherwise.
 * @return a result handle (free with mio_split_result_free), or NULL on failure.
 */
MIO_API mio_split_result* mio_split(const mio_mesh* mesh, const char* by, const char* tag_name);

/** Number of pieces in a split result, or -1 on error. */
MIO_API int64_t mio_split_result_count(const mio_split_result* result);

/**
 * Copy piece `index`'s key (type name / component index / tag value) into `buf`.
 * @return the untruncated key length, or -1 on error (string-getter protocol).
 */
MIO_API int64_t mio_split_result_key(const mio_split_result* result, int64_t index, char* buf,
                                     int64_t buflen);

/**
 * Borrow piece `index`'s mesh. Owned by the result: valid until
 * mio_split_result_free(result); do NOT pass it to mio_mesh_free().
 * @return the piece mesh, or NULL if out of range.
 */
MIO_API const mio_mesh* mio_split_result_mesh(const mio_split_result* result, int64_t index);

/**
 * Transfer ownership of piece `index`'s mesh out of the result.
 * @return a new owning mesh handle (free with mio_mesh_free), or NULL on error.
 */
MIO_API mio_mesh* mio_split_result_take_mesh(mio_split_result* result, int64_t index);

/** Free a split result (and any pieces it still owns). */
MIO_API void mio_split_result_free(mio_split_result* result);

/**
 * Convert the element representation of a mesh. point_sets/cell_sets are not
 * carried across the C ABI.
 * @param mode               "linearize" (higher-order cells -> their linear
 *                           base, pruning the orphaned nodes), "simplexify"
 *                           (decompose into same-dimension simplices), or
 *                           "elevate" (linear -> serendipity quadratic, adding
 *                           one node per unique edge).
 * @param record_parent_ids  nonzero to attach a convert:parent_cell cell_data
 *                           array of per-block source cell indices.
 * @return a result handle (free with mio_convert_cells_result_free), or NULL on
 *         failure — a polyhedron block under "simplexify" and the full-Lagrange
 *         targets (quad9/hexahedron27) under "elevate" both fail.
 */
MIO_API mio_convert_cells_result* mio_convert_cells(const mio_mesh* mesh, const char* mode,
                                                    int record_parent_ids);

/**
 * Borrow the converted mesh. Owned by the result: valid until
 * mio_convert_cells_result_free(result); do NOT pass it to mio_mesh_free().
 * @return the converted mesh, or NULL on error.
 */
MIO_API const mio_mesh* mio_convert_cells_result_mesh(const mio_convert_cells_result* result);

/**
 * Transfer ownership of the converted mesh out of the result.
 * @return a new owning mesh handle (free with mio_mesh_free), or NULL on error.
 */
MIO_API mio_mesh* mio_convert_cells_result_take_mesh(mio_convert_cells_result* result);

/**
 * Zero-copy borrow of the point map (int64, shape (num_points_in,), input point
 * index -> output index, -1 when the point was pruned). The pointer is valid
 * until the result is freed. Any out-param may be NULL.
 * @param result a convert-cells result.
 * @param data   receives a pointer into result-owned int64 memory.
 * @param dtype  receives MIO_INT64.
 * @param n      receives the input point count.
 */
MIO_API mio_status mio_convert_cells_result_point_map(const mio_convert_cells_result* result,
                                                      const void** data, mio_dtype* dtype,
                                                      int64_t* n);

/** Number of per-block cell maps in a convert-cells result, or -1 on error. */
MIO_API int64_t mio_convert_cells_result_num_cell_maps(const mio_convert_cells_result* result);

/**
 * Zero-copy borrow of cell block `block`'s cell map (int64, shape
 * (num_cells_in_block,), input cell -> the index of its FIRST child in the
 * corresponding output block). Under "linearize"/"elevate" there is one child
 * per parent, so the map is the identity; under "simplexify" a parent's
 * children occupy the contiguous range starting at this index. Any out-param
 * may be NULL.
 * @param result a convert-cells result.
 * @param block  cell-block index in [0, mio_convert_cells_result_num_cell_maps).
 * @param data   receives a pointer into result-owned int64 memory.
 * @param dtype  receives MIO_INT64.
 * @param n      receives the block's input cell count.
 */
MIO_API mio_status mio_convert_cells_result_cell_map(const mio_convert_cells_result* result,
                                                     int64_t block, const void** data,
                                                     mio_dtype* dtype, int64_t* n);

/** Free a convert-cells result (and the mesh it still owns). */
MIO_API void mio_convert_cells_result_free(mio_convert_cells_result* result);

/**
 * Polyhedrally refine a mesh: one polyhedral child per face of every eligible
 * 3D cell, connected to a new interior point. Needs no per-type template
 * table -- tabulated types (reduced to corners for a quadratic variant) and
 * existing polyhedron blocks are handled uniformly. Automatically conforming:
 * a shared face between two input cells is never touched. Non-3D blocks and
 * the full-Lagrange family (no face table) pass through unchanged. Point/Cell
 * regions survive; named Side regions do not (no facet correspondence).
 * point_sets/cell_sets are not carried across the C ABI.
 * @param record_parent_ids  nonzero to attach a subdivide:parent_cell
 *                           cell_data array of per-block source cell indices.
 * @return a result handle (free with mio_subdivide_result_free), or NULL on
 *         failure -- a cell whose faces are not a closed orientable surface.
 */
MIO_API mio_subdivide_result* mio_subdivide(const mio_mesh* mesh, int record_parent_ids);

/**
 * Borrow the subdivided mesh. Owned by the result: valid until
 * mio_subdivide_result_free(result); do NOT pass it to mio_mesh_free().
 * @return the subdivided mesh, or NULL on error.
 */
MIO_API const mio_mesh* mio_subdivide_result_mesh(const mio_subdivide_result* result);

/**
 * Transfer ownership of the subdivided mesh out of the result.
 * @return a new owning mesh handle (free with mio_mesh_free), or NULL on error.
 */
MIO_API mio_mesh* mio_subdivide_result_take_mesh(mio_subdivide_result* result);

/** Number of per-block cell maps in a subdivide result, or -1 on error. */
MIO_API int64_t mio_subdivide_result_num_cell_maps(const mio_subdivide_result* result);

/**
 * Zero-copy borrow of cell block `block`'s cell map (int64, shape
 * (num_cells_in_block,), input cell -> the index of its FIRST child (one per
 * face) in the corresponding output block). Any out-param may be NULL.
 * @param result a subdivide result.
 * @param block  cell-block index in [0, mio_subdivide_result_num_cell_maps).
 * @param data   receives a pointer into result-owned int64 memory.
 * @param dtype  receives MIO_INT64.
 * @param n      receives the block's input cell count.
 */
MIO_API mio_status mio_subdivide_result_cell_map(const mio_subdivide_result* result, int64_t block,
                                                 const void** data, mio_dtype* dtype, int64_t* n);

/** Free a subdivide result (and the mesh it still owns). */
MIO_API void mio_subdivide_result_free(mio_subdivide_result* result);

/**
 * Polyhedrally coarsen a mesh: merge groups of cells into single larger
 * polyhedral cells via greedy seed-and-grow over the mesh's shared-face
 * dual, absorbing face-adjacent neighbours into a group until it reaches
 * target_group_size (or no unclaimed neighbour remains -- a short group at a
 * mesh boundary or pocket is expected, not an error). Non-volume blocks
 * (2D/1D, or any 3D block with no face table) pass through unchanged.
 * Points are never pruned or renumbered; mio_clean(..., remove_orphans=1) is
 * the follow-up for a caller wanting a minimal point set. Point/Cell regions
 * survive; named Side regions cannot (a many-to-one collapse has no facet
 * correspondence to preserve). point_sets/cell_sets are not carried across
 * the C ABI.
 * @param target_group_size  approximate member cells per output group; must
 *                           be at least 1 (1 means every cell is its own
 *                           group).
 * @return a result handle (free with mio_agglomerate_result_free), or NULL
 *         on failure -- target_group_size == 0, or a face shared by three or
 *         more cells (non-manifold), which the merge refuses rather than
 *         guessing a boundary classification for.
 */
MIO_API mio_agglomerate_result* mio_agglomerate(const mio_mesh* mesh, int64_t target_group_size);

/**
 * Borrow the coarsened mesh. Owned by the result: valid until
 * mio_agglomerate_result_free(result); do NOT pass it to mio_mesh_free().
 * @return the coarsened mesh, or NULL on error.
 */
MIO_API const mio_mesh* mio_agglomerate_result_mesh(const mio_agglomerate_result* result);

/**
 * Transfer ownership of the coarsened mesh out of the result.
 * @return a new owning mesh handle (free with mio_mesh_free), or NULL on error.
 */
MIO_API mio_mesh* mio_agglomerate_result_take_mesh(mio_agglomerate_result* result);

/**
 * Zero-copy borrow of the FLAT cell map (int64, shape (total input cell
 * count,), input global cell -> output global cell). Unlike
 * mio_subdivide_result_cell_map, this takes no block argument -- an output
 * cell's index is a function of which group it joined, not which input
 * block it came from -- and there is correspondingly no
 * mio_agglomerate_result_num_cell_maps (there is exactly one array). Any
 * out-param may be NULL.
 * @param result an agglomerate result.
 * @param data   receives a pointer into result-owned int64 memory.
 * @param dtype  receives MIO_INT64.
 * @param n      receives the input mesh's total cell count.
 */
MIO_API mio_status mio_agglomerate_result_cell_map(const mio_agglomerate_result* result,
                                                   const void** data, mio_dtype* dtype,
                                                   int64_t* n);

/** Free an agglomerate result (and the mesh it still owns). */
MIO_API void mio_agglomerate_result_free(mio_agglomerate_result* result);

/** How mio_refine_ex resolves the hanging nodes a partial refinement leaves. */
typedef enum mio_refine_closure {
    MIO_REFINE_CLOSURE_REDGREEN = 0,  /**< promote to the smallest admissible mask (local) */
    MIO_REFINE_CLOSURE_PROPAGATE = 1, /**< promote straight to a full split (spreads) */
    /**
     * Do not close at all: KEEP the hanging nodes and only enforce 2:1 balance.
     * The output is 1-irregular and NOT conforming; the constrained nodes are
     * reported in the refine:hanging point_data array.
     */
    MIO_REFINE_CLOSURE_BALANCED = 2
} mio_refine_closure;

/** The comparison a mio_refine_opts predicate selector applies. */
typedef enum mio_refine_compare {
    MIO_REFINE_LT = 0,
    MIO_REFINE_LE = 1,
    MIO_REFINE_GT = 2,
    MIO_REFINE_GE = 3,
    MIO_REFINE_EQ = 4,
    MIO_REFINE_NE = 5
} mio_refine_compare;

/**
 * Options for mio_refine_ex: which cells to refine and how to close up after.
 *
 * ABI NOTE: this struct is part of the installed library's permanent ABI. New
 * fields may only be appended, replacing `reserved` capacity; never reorder,
 * resize or repurpose an existing field. Always zero-initialize through
 * mio_refine_opts_init() rather than by hand, so fields added later default
 * sensibly in code compiled against an older header.
 *
 * All zero means "refine every cell once", i.e. exactly mio_refine(mesh, 1, 0).
 * At most ONE of `cells`, `region` and `predicate_array` may be given; two is an
 * error rather than a precedence rule.
 */
typedef struct mio_refine_opts {
    /** Global (block-major) indices of the cells to refine; NULL for none. */
    const int64_t* cells;
    /** Length of `cells`; ignored when `cells` is NULL. */
    int64_t num_cells;
    /**
     * Name of a region to refine, or NULL/"" for none. A cell region selects its
     * own cells; a point region selects every cell with ANY node in it; a side
     * region is an error, since a facet is not a cell.
     */
    const char* region;
    /**
     * Name of a scalar numeric cell_data array to threshold, or NULL/"" for
     * none. A non-finite cell value never matches.
     */
    const char* predicate_array;
    /** The predicate's right-hand side. */
    double predicate_value;
    /** How many times to apply the templates; 0 or less is an unchanged clone. */
    int32_t levels;
    /** Nonzero to attach the refine:parent_cell cell_data array. */
    int32_t record_parent_ids;
    /** Nonzero to attach the refine:level cell_data array. */
    int32_t record_levels;
    /** A mio_refine_closure. */
    int32_t closure;
    /** A mio_refine_compare. */
    int32_t predicate_op;
    int32_t reserved_pad; /**< must be zero; keeps the int64 tail aligned */
    /**
     * Nonzero to attach the refine:cell_id/refine:parent_id cell_data arrays --
     * the persistent parent/child hierarchy a multigrid caller resolves across
     * the sequence of meshes it keeps ("a link between two meshes, not a tree
     * inside one"). An input already carrying refine:cell_id is updated
     * whatever this says. Also forces refine:entity to be attached even when
     * the closure leaves no hanging node, since it already records the coarse
     * corners each new fine node is the mean of -- the multigrid prolongation
     * weights. Takes one of the former `reserved` slots, keeping the struct's
     * size and every other field's offset unchanged -- the mio_read_opts
     * `lenient` precedent.
     */
    int64_t record_hierarchy;
    int64_t reserved[5]; /**< must be zero; room for additive growth */
} mio_refine_opts;

/** Zero-initialize refine options (every field its default). */
MIO_API void mio_refine_opts_init(mio_refine_opts* opts);

/**
 * Refine a mesh, subdividing every cell into same-type children (line -> 2,
 * triangle -> 4, quad -> 4, tetra -> 8, wedge -> 8, hexahedron -> 8). New nodes
 * sit at edge / quad-face / body midpoints and are shared between neighbouring
 * cells, so the refined mesh has no hanging nodes.
 * point_sets/cell_sets are not carried across the C ABI.
 *
 * This is mio_refine_ex with every option at its default; use that to refine a
 * SUBSET of the cells instead.
 * @param levels             how many times to apply the templates; 0 or less
 *                           returns an unchanged clone.
 * @param record_parent_ids  nonzero to attach a refine:parent_cell cell_data
 *                           array naming each output cell's ORIGINAL ancestor
 *                           within its own block.
 * @return a result handle (free with mio_refine_result_free), or NULL on
 *         failure — higher-order cells, pyramids and ragged polygon/polyhedron
 *         blocks have no same-type subdivision and all fail.
 */
MIO_API mio_refine_result* mio_refine(const mio_mesh* mesh, int levels, int record_parent_ids);

/**
 * Refine a mesh with a cell selection and a conforming closure.
 *
 * With no selector set this is mio_refine. With one, only the selected cells get
 * the full template and the hanging nodes that leaves are resolved by the
 * closure, so the output is still a conforming mesh — see doc/refine.md for the
 * per-cell-type support matrix.
 * @param mesh  the mesh to refine (unchanged).
 * @param opts  the options, or NULL for every default.
 * @return a result handle (free with mio_refine_result_free), or NULL on
 *         failure — including more than one selector, an out-of-range cell
 *         index, an unknown region, a side region used as a selector, and a
 *         predicate array that is missing, does not cover every block or is not
 *         scalar.
 */
MIO_API mio_refine_result* mio_refine_ex(const mio_mesh* mesh, const mio_refine_opts* opts);

/**
 * Borrow the refined mesh. Owned by the result: valid until
 * mio_refine_result_free(result); do NOT pass it to mio_mesh_free().
 * @return the refined mesh, or NULL on error.
 */
MIO_API const mio_mesh* mio_refine_result_mesh(const mio_refine_result* result);

/**
 * Transfer ownership of the refined mesh out of the result.
 * @return a new owning mesh handle (free with mio_mesh_free), or NULL on error.
 */
MIO_API mio_mesh* mio_refine_result_take_mesh(mio_refine_result* result);

/**
 * Zero-copy borrow of the point map (int64, shape (num_points_in,), input point
 * index -> output index). Refinement never prunes, so this is the identity; it
 * is exposed in full so callers need not depend on that. The pointer is valid
 * until the result is freed. Any out-param may be NULL.
 * @param result a refine result.
 * @param data   receives a pointer into result-owned int64 memory.
 * @param dtype  receives MIO_INT64.
 * @param n      receives the input point count.
 */
MIO_API mio_status mio_refine_result_point_map(const mio_refine_result* result, const void** data,
                                               mio_dtype* dtype, int64_t* n);

/** Number of per-block cell maps in a refine result, or -1 on error. */
MIO_API int64_t mio_refine_result_num_cell_maps(const mio_refine_result* result);

/**
 * Zero-copy borrow of cell block `block`'s cell map (int64, shape
 * (num_cells_in_block,), input cell -> the index of its FIRST child in the
 * corresponding output block). A cell's children are contiguous, so parent c
 * owns [map[c], map[c+1]) (or the block's end for the last parent). Any
 * out-param may be NULL.
 * @param result a refine result.
 * @param block  cell-block index in [0, mio_refine_result_num_cell_maps).
 * @param data   receives a pointer into result-owned int64 memory.
 * @param dtype  receives MIO_INT64.
 * @param n      receives the block's input cell count.
 */
MIO_API mio_status mio_refine_result_cell_map(const mio_refine_result* result, int64_t block,
                                              const void** data, mio_dtype* dtype, int64_t* n);

/** Free a refine result (and the mesh it still owns). */
MIO_API void mio_refine_result_free(mio_refine_result* result);

/**
 * Decimate a SURFACE mesh by greedy quadric-error-metric (Garland-Heckbert)
 * edge collapse — the resolution-reducing inverse of mio_refine. The output is
 * all-triangle (quad/polygon blocks are triangulated first) with the block
 * structure kept 1:1. Boundary vertices (once-used-edge test) and feature
 * vertices (incident face normals differing by more than feature_angle
 * degrees) are pinned by default; the link condition and a normal-flip guard
 * reject any collapse that would change topology or fold the surface. The
 * caller frozen mask is not exposed across the C ABI (a documented flat-ABI
 * gap, like mio_smooth's).
 * @param target_ratio      fraction of the (triangulated) faces to KEEP, in
 *                          (0, 1]; negative = unset.
 * @param target_faces      absolute face count to stop at (the result lands
 *                          within one collapse of it); negative = unset.
 * @param max_error         collapse only while the cheapest candidate's
 *                          quadric error is at most this; negative = unset.
 *                          Exactly one of the three criteria must be set.
 * @param placement         "optimal" (quadric minimizer, midpoint when
 *                          ill-conditioned), "midpoint" or "endpoint"; NULL =
 *                          "optimal".
 * @param preserve_boundary nonzero to pin boundary vertices (default-on
 *                          behaviour: pass 1).
 * @param preserve_features nonzero to pin feature vertices.
 * @param feature_angle     the feature angle in degrees (30 is the
 *                          vtkFeatureEdges convention).
 * @return a result handle (free with mio_decimate_result_free), or NULL on
 *         failure — a 3D volume mesh (extract the surface first), higher-order
 *         or ragged blocks, line/vertex blocks, and a criterion count != 1 all
 *         fail by name.
 */
MIO_API mio_decimate_result* mio_decimate(const mio_mesh* mesh, double target_ratio,
                                          int64_t target_faces, double max_error,
                                          const char* placement, int preserve_boundary,
                                          int preserve_features, double feature_angle);

/**
 * Borrow the decimated mesh. Owned by the result: valid until
 * mio_decimate_result_free(result); do NOT pass it to mio_mesh_free().
 * @return the decimated mesh, or NULL on error.
 */
MIO_API const mio_mesh* mio_decimate_result_mesh(const mio_decimate_result* result);

/**
 * Transfer ownership of the decimated mesh out of the result.
 * @return a new owning mesh handle (free with mio_mesh_free), or NULL on error.
 */
MIO_API mio_mesh* mio_decimate_result_take_mesh(mio_decimate_result* result);

/**
 * Zero-copy borrow of the point map (int64, shape (num_points_in,), input
 * point index -> output index). A collapsed point maps to its SURVIVOR's
 * output index — which is what makes it usable for remapping external
 * per-point arrays — and is -1 only when the survivor itself ended up
 * unreferenced. The pointer is valid until the result is freed. Any out-param
 * may be NULL.
 * @param result a decimate result.
 * @param data   receives a pointer into result-owned int64 memory.
 * @param dtype  receives MIO_INT64.
 * @param n      receives the input point count.
 */
MIO_API mio_status mio_decimate_result_point_map(const mio_decimate_result* result,
                                                 const void** data, mio_dtype* dtype, int64_t* n);

/** Number of per-block cell maps in a decimate result, or -1 on error. */
MIO_API int64_t mio_decimate_result_num_cell_maps(const mio_decimate_result* result);

/**
 * Zero-copy borrow of INPUT cell block `block`'s cell map (int64, shape
 * (num_cells_in_block,), input cell -> the output index, within the
 * corresponding output block, of its first surviving triangle; -1 when none
 * survived). Any out-param may be NULL.
 * @param result a decimate result.
 * @param block  cell-block index in [0, mio_decimate_result_num_cell_maps).
 * @param data   receives a pointer into result-owned int64 memory.
 * @param dtype  receives MIO_INT64.
 * @param n      receives the block's input cell count.
 */
MIO_API mio_status mio_decimate_result_cell_map(const mio_decimate_result* result, int64_t block,
                                                const void** data, mio_dtype* dtype, int64_t* n);

/** Triangles removed (counted on the triangulated mesh), or -1 on error. */
MIO_API int64_t mio_decimate_result_faces_removed(const mio_decimate_result* result);

/** Points removed (collapsed plus pruned-unreferenced), or -1 on error. */
MIO_API int64_t mio_decimate_result_points_removed(const mio_decimate_result* result);

/** Guard-rejection events during the run, or -1 on error. */
MIO_API int64_t mio_decimate_result_collapses_rejected(const mio_decimate_result* result);

/** The largest committed collapse error (0.0 when nothing collapsed), or a
 *  negative value on error. */
MIO_API double mio_decimate_result_max_error_applied(const mio_decimate_result* result);

/** Free a decimate result (and the mesh it still owns). */
MIO_API void mio_decimate_result_free(mio_decimate_result* result);

/**
 * Decimate a TET mesh by greedy quadric-error-metric tet-edge collapse — the
 * volume-mesh sibling of mio_decimate, a SEPARATE operation (mio_decimate
 * keeps failing by name on any 3D volume block, pointing here). Tet-only:
 * hexahedron/wedge/pyramid/polyhedron/ragged/higher-order 3D blocks, or any
 * non-3D block mixed in, fail by name pointing at simplexify
 * (mio_convert_cells). Unlike mio_decimate, boundary vertices PARTICIPATE by
 * default (preserve_boundary defaults to pass 0) with a real quadric-error
 * objective built from the mesh's own outer-skin triangles; purely interior
 * edges (whose combined quadric is exactly zero) are scored by squared
 * length instead and always rank behind boundary-touching ones. The link
 * condition, a duplicate-tet guard and a tet-inversion guard reject any
 * collapse that would change topology or invert a tet; boundary-touching
 * collapses additionally run mio_decimate's own ring/shared-face link
 * condition and normal-flip check over the mesh's own skin. The caller
 * frozen mask is not exposed across the C ABI (a documented flat-ABI gap,
 * like mio_smooth's).
 * @param target_ratio      fraction of the tets to KEEP, in (0, 1];
 *                          negative = unset.
 * @param target_cells      absolute tet count to stop at (the result lands
 *                          within one collapse of it); negative = unset.
 * @param max_error         collapse only while the cheapest boundary-touching
 *                          candidate's quadric error is at most this;
 *                          negative = unset. Exactly one of the three
 *                          criteria must be set.
 * @param placement         "optimal" (quadric minimizer, midpoint when
 *                          ill-conditioned or purely interior), "midpoint" or
 *                          "endpoint"; NULL = "optimal".
 * @param preserve_boundary nonzero to pin every boundary vertex outright,
 *                          reproducing mio_decimate's own default instead of
 *                          letting boundary vertices participate.
 * @param preserve_features nonzero to pin boundary feature vertices.
 * @param feature_angle     the feature angle in degrees (30 is the
 *                          vtkFeatureEdges convention).
 * @return a result handle (free with mio_decimate_volume_result_free), or
 *         NULL on failure — a non-tet-only mesh, a non-manifold boundary
 *         face, and a criterion count != 1 all fail by name.
 */
MIO_API mio_decimate_volume_result* mio_decimate_volume(const mio_mesh* mesh, double target_ratio,
                                                         int64_t target_cells, double max_error,
                                                         const char* placement,
                                                         int preserve_boundary,
                                                         int preserve_features,
                                                         double feature_angle);

/**
 * Borrow the decimated mesh. Owned by the result: valid until
 * mio_decimate_volume_result_free(result); do NOT pass it to mio_mesh_free().
 * @return the decimated mesh, or NULL on error.
 */
MIO_API const mio_mesh* mio_decimate_volume_result_mesh(const mio_decimate_volume_result* result);

/**
 * Transfer ownership of the decimated mesh out of the result.
 * @return a new owning mesh handle (free with mio_mesh_free), or NULL on error.
 */
MIO_API mio_mesh* mio_decimate_volume_result_take_mesh(mio_decimate_volume_result* result);

/**
 * Zero-copy borrow of the point map (int64, shape (num_points_in,), input
 * point index -> output index). A collapsed point maps to its SURVIVOR's
 * output index, and is -1 only when the survivor itself ended up
 * unreferenced. The pointer is valid until the result is freed. Any out-param
 * may be NULL.
 */
MIO_API mio_status mio_decimate_volume_result_point_map(const mio_decimate_volume_result* result,
                                                         const void** data, mio_dtype* dtype,
                                                         int64_t* n);

/** Number of per-block cell maps in a decimate_volume result, or -1 on error. */
MIO_API int64_t mio_decimate_volume_result_num_cell_maps(const mio_decimate_volume_result* result);

/**
 * Zero-copy borrow of INPUT tet block `block`'s cell map (int64, shape
 * (num_cells_in_block,), input cell -> its own output index, or -1 when it
 * did not survive). Any out-param may be NULL.
 * @param result a decimate_volume result.
 * @param block  cell-block index in [0, mio_decimate_volume_result_num_cell_maps).
 * @param data   receives a pointer into result-owned int64 memory.
 * @param dtype  receives MIO_INT64.
 * @param n      receives the block's input cell count.
 */
MIO_API mio_status mio_decimate_volume_result_cell_map(const mio_decimate_volume_result* result,
                                                        int64_t block, const void** data,
                                                        mio_dtype* dtype, int64_t* n);

/** Tets removed, or -1 on error. */
MIO_API int64_t
mio_decimate_volume_result_tets_removed(const mio_decimate_volume_result* result);

/** Points removed (collapsed plus pruned-unreferenced), or -1 on error. */
MIO_API int64_t
mio_decimate_volume_result_points_removed(const mio_decimate_volume_result* result);

/** Guard-rejection events during the run, or -1 on error. */
MIO_API int64_t
mio_decimate_volume_result_collapses_rejected(const mio_decimate_volume_result* result);

/** The largest committed boundary-touching (regime 0) collapse error (0.0
 *  when nothing collapsed, or every collapse was purely interior), or a
 *  negative value on error. */
MIO_API double
mio_decimate_volume_result_max_error_applied(const mio_decimate_volume_result* result);

/** Free a decimate_volume result (and the mesh it still owns). */
MIO_API void mio_decimate_volume_result_free(mio_decimate_volume_result* result);

/**
 * Decompose a mesh into `nparts` balanced pieces for domain decomposition (the
 * count-driven complement to mio_split). Methods: "sfc" (Hilbert space-filling
 * curve cut of the cell centroids — always available, deterministic), "kahip"
 * (KaHIP kaffpa() on the shared-face dual graph — only in builds with
 * -DMESHIOPLUSPLUS_WITH_KAHIP=ON; requesting it elsewhere fails with an error
 * naming that option), or "auto"/NULL/"" (kahip when built, else sfc). Every
 * piece keeps the input's cell-block structure 1:1 (empty blocks included,
 * unlike mio_split), so the cell maps index by input block and the pieces
 * recombine into the input. point_sets/cell_sets are not carried across the
 * C ABI.
 * @param nparts       number of pieces (>= 1); the result always has exactly
 *                     nparts pieces, possibly with empty meshes.
 * @param method       "sfc", "kahip", "auto" (NULL/"" = "auto").
 * @param imbalance    kahip only: allowed imbalance fraction in (0, 1)
 *                     (0.03 = 3%).
 * @param mode         kahip only: "fast", "eco", "strong" (NULL/"" = "eco").
 * @param seed         kahip only: random seed (deterministic per seed).
 * @param record_ids   nonzero to attach Int64 partition:original_point_id
 *                     point_data and partition:original_cell_id cell_data
 *                     (original input indices) to every piece.
 * @param ghost_layers grow each piece by this many shared-node BFS layers of
 *        other parts' cells (a halo), tagged with an Int64 `partition:ghost`
 *        cell_data (0 = owned, L = reached at layer L). 0 keeps the pieces
 *        disjoint; negative fails the call.
 * @param weights_key  name of a scalar numeric cell_data array of per-cell
 *                     weights, or NULL/"" for the unweighted rule.
 * @return a result handle (free with mio_partition_result_free), or NULL on
 *         failure (see mio_last_error()).
 */
MIO_API mio_partition_result* mio_partition(const mio_mesh* mesh, int nparts, const char* method,
                                            double imbalance, const char* mode, int seed,
                                            int record_ids, int ghost_layers,
                                            const char* weights_key);

/** Number of pieces in a partition result (== nparts), or -1 on error. */
MIO_API int64_t mio_partition_result_num_pieces(const mio_partition_result* result);

/** The part id of piece `index` (== index, in [0, nparts)), or -1 on error. */
MIO_API int mio_partition_result_part_id(const mio_partition_result* result, int64_t index);

/**
 * Borrow piece `index`'s mesh. Owned by the result: valid until
 * mio_partition_result_free(result); do NOT pass it to mio_mesh_free().
 * @return the piece mesh, or NULL on error.
 */
MIO_API const mio_mesh* mio_partition_result_mesh(const mio_partition_result* result,
                                                  int64_t index);

/**
 * Transfer ownership of piece `index`'s mesh out of the result.
 * @return a new owning mesh handle (free with mio_mesh_free), or NULL on error.
 */
MIO_API mio_mesh* mio_partition_result_take_mesh(mio_partition_result* result, int64_t index);

/**
 * Zero-copy borrow of piece `index`'s point map (int64, shape
 * (num_points_in,), input point index -> piece point index, -1 where the point
 * is not in the piece). The pointer is valid until the result is freed. Any
 * out-param may be NULL.
 * @param result a partition result.
 * @param index  piece index in [0, mio_partition_result_num_pieces).
 * @param data   receives a pointer into result-owned int64 memory.
 * @param dtype  receives MIO_INT64.
 * @param n      receives the input point count.
 */
MIO_API mio_status mio_partition_result_point_map(const mio_partition_result* result,
                                                  int64_t index, const void** data,
                                                  mio_dtype* dtype, int64_t* n);

/** Number of per-block cell maps in each piece (== the input's cell-block
 *  count), or -1 on error. */
MIO_API int64_t mio_partition_result_num_cell_maps(const mio_partition_result* result);

/**
 * Zero-copy borrow of piece `index`'s cell map for input block `block` (int64,
 * shape (num_cells_in_block,), input cell -> piece cell index within that
 * block, -1 where the cell is not in the piece). Any out-param may be NULL.
 * @param result a partition result.
 * @param index  piece index in [0, mio_partition_result_num_pieces).
 * @param block  cell-block index in [0, mio_partition_result_num_cell_maps).
 * @param data   receives a pointer into result-owned int64 memory.
 * @param dtype  receives MIO_INT64.
 * @param n      receives the block's input cell count.
 */
MIO_API mio_status mio_partition_result_cell_map(const mio_partition_result* result, int64_t index,
                                                 int64_t block, const void** data,
                                                 mio_dtype* dtype, int64_t* n);

/** Free a partition result (and every piece mesh it still owns). */
MIO_API void mio_partition_result_free(mio_partition_result* result);

/**
 * Compute the per-cell part assignment only, without building the pieces.
 * Fills a caller-allocated flat buffer in block-major global cell order (the
 * mesh's blocks concatenated in order — recover the block alignment from the
 * mesh's own per-block cell counts). Parameters as mio_partition (record_ids /
 * ghost_layers do not apply).
 * @param labels      caller buffer of labels_size int64 entries; receives
 *                    values in [0, nparts).
 * @param labels_size must equal the mesh's total cell count.
 * @return MIO_OK, or an error status (see mio_last_error()).
 */
MIO_API mio_status mio_partition_labels(const mio_mesh* mesh, int nparts, const char* method,
                                        double imbalance, const char* mode, int seed,
                                        const char* weights_key, int64_t* labels,
                                        int64_t labels_size);

/** Geometric statistics of a mesh (see mio_stats). Per-cell-type counts are not
 *  carried across the C ABI (use mio_num_cell_blocks / mio_cell_block_type). */
typedef struct mio_stats_report {
    int64_t num_points;     /**< number of points */
    int64_t num_cells;      /**< total number of cells */
    double bbox_min[3];     /**< bounding-box lower corner */
    double bbox_max[3];     /**< bounding-box upper corner */
    double extent[3];       /**< bounding-box extent (max - min) */
    double centroid[3];     /**< mean of the point coordinates */
    double total_area;      /**< area of 2D cells + boundary of 3D cells */
    double signed_volume;   /**< sum of signed volumes of 3D cells */
    double unsigned_volume; /**< sum of |volume| of 3D cells */
    int64_t num_inverted;   /**< 3D cells with negative signed volume */
} mio_stats_report;

/**
 * Compute geometric statistics of a mesh (read-only).
 * @param mesh input mesh.
 * @param out  receives the statistics (must be non-NULL).
 * @return MIO_OK, or an error status (see mio_last_error()).
 */
MIO_API mio_status mio_stats(const mio_mesh* mesh, mio_stats_report* out);

/* ------------------------------------------------------------------------- */
/* Regular grids and signed distance                                         */
/*                                                                           */
/* mio_grid builds a regular hexahedron lattice from nothing -- the only      */
/* entry point here that takes no input mesh. mio_voxelize builds one around  */
/* a mesh and optionally keeps only the cells its surface passes through or   */
/* encloses. The rest measure distance to a triangle surface.                 */
/*                                                                           */
/* Everything comes back as an ordinary mio_mesh with one hexahedron block,   */
/* so every writer, every operation and every getter already works on it.     */
/* ------------------------------------------------------------------------- */

/** Which cells mio_voxelize keeps. */
typedef enum mio_voxel_fill {
    MIO_VOXEL_ALL = 0,     /**< every cell of the bounding box */
    MIO_VOXEL_SURFACE = 1, /**< only cells a surface triangle passes through */
    MIO_VOXEL_INSIDE = 2   /**< only cells whose centre is inside the surface */
} mio_voxel_fill;

/** How a signed distance decides which side of the surface a point is on. */
typedef enum mio_sdf_sign {
    MIO_SDF_UNSIGNED = 0,       /**< no sign; the only mode valid on an open sheet */
    MIO_SDF_PSEUDONORMAL = 1,   /**< angle-weighted pseudonormal of the nearest feature */
    MIO_SDF_WINDING_NUMBER = 2  /**< generalized winding number; O(triangles) per query */
} mio_sdf_sign;

/** How incident faces are weighted when building a vertex pseudonormal. */
typedef enum mio_sdf_weight {
    MIO_SDF_WEIGHT_ANGLE = 0, /**< the geometrically correct choice */
    MIO_SDF_WEIGHT_AREA = 1   /**< free of acos, and therefore bit-reproducible */
} mio_sdf_weight;

/** Where a distance is evaluated on a mesh. */
typedef enum mio_sdf_location {
    MIO_SDF_CORNER = 0, /**< at the points, as point_data */
    MIO_SDF_CENTER = 1  /**< at the cell centroids, as cell_data */
} mio_sdf_location;

/** What to do when the surface turns out not to be watertight. */
typedef enum mio_sdf_watertight_check {
    MIO_SDF_WATERTIGHT_OFF = 0,   /**< do not look */
    MIO_SDF_WATERTIGHT_WARN = 1,  /**< log the counts and carry on */
    MIO_SDF_WATERTIGHT_ERROR = 2  /**< fail, naming the counts */
} mio_sdf_watertight_check;

/**
 * What is wrong with a surface, in numbers rather than a bare flag.
 *
 * The four counts are reported separately because they need different fixes:
 * "12 boundary edges" is actionable, "not watertight" is not.
 *
 * ABI NOTE: as with every struct here, new fields may only be appended,
 * replacing `reserved` capacity.
 */
typedef struct mio_surface_quality {
    int64_t boundary_edges;       /**< edges used by exactly one triangle */
    int64_t non_manifold_edges;   /**< edges used by three or more */
    int64_t inconsistent_pairs;   /**< two triangles winding an edge the same way */
    int64_t degenerate_triangles; /**< zero-area triangles */
    int32_t watertight;           /**< nonzero when all four counts are zero */
    int32_t reserved_pad;         /**< must be zero; keeps the int64 tail aligned */
    int64_t reserved[4];          /**< must be zero; room for additive growth */
} mio_surface_quality;

/**
 * Options for the distance entry points.
 *
 * ABI NOTE: this struct is part of the installed library's permanent ABI. New
 * fields may only be appended, replacing `reserved` capacity; never reorder,
 * resize or repurpose an existing field. Always zero-initialize through
 * mio_sdf_opts_init() rather than by hand, so fields added later default
 * sensibly in code compiled against an older header.
 */
typedef struct mio_sdf_opts {
    /** Restrict the surface to a named cell region; NULL/"" is all of it. */
    const char* surface_region;
    /** Clamp distances beyond this and mark them in sdf:band; <= 0 is the full field. */
    double band;
    /**
     * Bucket size of the search grid; 0 derives one from the triangle sizes.
     * A tuning knob only: every candidate comparison is totally ordered on
     * (squared distance, triangle id), so the bucket size provably cannot change
     * the answer -- which is also what makes this a usable test knob.
     */
    double grid_cell_size;
    /** Refuse a winding-number query whose n_queries * n_triangles exceeds this. */
    double max_winding_work;
    int32_t sign;                /**< a mio_sdf_sign */
    int32_t weight;              /**< a mio_sdf_weight */
    int32_t location;            /**< a mio_sdf_location */
    int32_t watertight_check;    /**< a mio_sdf_watertight_check */
    int32_t record_closest_cell; /**< nonzero to attach sdf:closest_cell */
    int32_t record_inside;       /**< nonzero to attach sdf:inside */
    int64_t reserved[6];         /**< must be zero; room for additive growth */
} mio_sdf_opts;

/** Zero-initialize distance options (every field its default). */
MIO_API void mio_sdf_opts_init(mio_sdf_opts* opts);

/**
 * Options for mio_voxelize.
 *
 * ABI NOTE: the same append-only discipline as mio_sdf_opts. Always
 * zero-initialize through mio_voxel_opts_init().
 *
 * Exactly ONE of `resolution` and `cell_size` must be given; giving neither or
 * both is an error rather than a precedence rule, because the cost of the
 * choice is cubic in it.
 */
typedef struct mio_voxel_opts {
    /** Cell counts (three entries), or NULL when sizing by cell_size. */
    const int64_t* resolution;
    /** Explicit bounds {xlo, ylo, zlo, xhi, yhi, zhi}, or NULL for the mesh's own. */
    const double* bounds;
    /** Cubic cell size, or <= 0 when sizing by resolution. */
    double cell_size;
    /** Grow the box by this on every side, in world units. */
    double padding;
    /** Grow the box by this fraction of its diagonal on every side. */
    double padding_relative;
    /** Refuse above this many cells; 0 or less lifts the limit. */
    int64_t max_cells;
    int32_t fill;              /**< a mio_voxel_fill */
    int32_t attach_occupancy;  /**< nonzero to attach voxel:occupancy */
    int32_t sign;              /**< a mio_sdf_sign, used by MIO_VOXEL_INSIDE */
    int32_t watertight_check;  /**< a mio_sdf_watertight_check */
    int64_t reserved[6];       /**< must be zero; room for additive growth */
} mio_voxel_opts;

/** Zero-initialize voxelization options (every field its default). */
MIO_API void mio_voxel_opts_init(mio_voxel_opts* opts);

/**
 * Build a regular hexahedron lattice from nothing.
 *
 * Points run x fastest, then y, then z; each cell's eight nodes are in the
 * meshio/VTK hexahedron winding, so every cell of a positively spaced lattice
 * has unit scaled Jacobian.
 *
 * @param dims      cell counts (three entries); a zero on any axis gives an
 *                  empty mesh, which is a legal request rather than an error.
 * @param origin    the lo corner (three entries), or NULL for the origin.
 * @param spacing   cell size per axis (three entries), or NULL for unit cells.
 * @param max_cells refuse above this many cells; 0 or less lifts the limit.
 * @return the lattice (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_grid(const int64_t dims[3], const double origin[3],
                           const double spacing[3], int64_t max_cells);

/**
 * Build a regular grid around a mesh and keep the cells its fill rule selects.
 *
 * @param mesh          the mesh to voxelize; used for its bounding box, and for
 *                      its surface when the fill is not MIO_VOXEL_ALL.
 * @param opts          options; NULL is an error, since a resolution or cell
 *                      size must be given.
 * @param dims_out      receives the cell counts (three entries), or NULL.
 * @param origin_out    receives the lattice lo corner (three entries), or NULL.
 * @param spacing_out   receives the cell size per axis (three entries), or NULL.
 * @param num_occupied  receives how many cells the fill kept, or NULL.
 * @return the grid (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_voxelize(const mio_mesh* mesh, const mio_voxel_opts* opts,
                               int64_t dims_out[3], double origin_out[3],
                               double spacing_out[3], int64_t* num_occupied);

/**
 * Report what is wrong with a surface, without computing any distances.
 * @param surface a triangle (or triangulatable) surface mesh.
 * @param out     receives the counts (must be non-NULL).
 * @return MIO_OK, or an error status (see mio_last_error()).
 */
MIO_API mio_status mio_surface_watertight_check(const mio_mesh* surface,
                                                mio_surface_quality* out);

/**
 * Signed distances from arbitrary points to a surface.
 *
 * The output buffer is supplied by the caller -- the mio_partition_labels
 * convention -- so nothing here needs freeing and no opaque handle is involved.
 *
 * @param surface  the surface to measure against.
 * @param points   query coordinates, `n_points` rows of three doubles.
 * @param n_points how many query points.
 * @param opts     options, or NULL for the defaults.
 * @param out      receives `n_points` signed distances (negative inside).
 * @return MIO_OK, or an error status (see mio_last_error()).
 */
MIO_API mio_status mio_sample_distance(const mio_mesh* surface, const double* points,
                                       int64_t n_points, const mio_sdf_opts* opts, double* out);

/**
 * Attach distances from a query mesh's points (or cell centres) to a surface.
 *
 * @param query       the mesh to annotate; its geometry is copied unchanged.
 * @param surface     the surface to measure against.
 * @param opts        options, or NULL for the defaults.
 * @param num_banded  receives how many queries were clamped to the band, or NULL.
 * @param quality     receives the surface verdict, or NULL.
 * @return the annotated mesh (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_distance_to_surface(const mio_mesh* query, const mio_mesh* surface,
                                          const mio_sdf_opts* opts, int64_t* num_banded,
                                          mio_surface_quality* quality);

/** Which structure mio_compute_sdf generates. */
typedef enum mio_sdf_structure {
    MIO_SDF_VOXEL = 0,  /**< a dense uniform lattice over the padded box */
    MIO_SDF_OCTREE = 1  /**< adaptive, refined near the surface (1-irregular) */
} mio_sdf_structure;

/**
 * Options for mio_compute_sdf.
 *
 * ABI NOTE: the same append-only discipline as mio_sdf_opts. Always
 * zero-initialize through mio_compute_sdf_opts_init().
 *
 * For MIO_SDF_VOXEL exactly ONE of `resolution` and `cell_size` must be given.
 * For MIO_SDF_OCTREE **neither** may be: its finest cell is
 * `root cell / 2^depth` and is therefore already determined by
 * `root_resolution` and `max_depth`, so accepting either would silently ignore
 * one of the two.
 */
typedef struct mio_compute_sdf_opts {
    /** Cell counts (three entries), or NULL when sizing by cell_size. Voxel only. */
    const int64_t* resolution;
    /** Explicit bounds {xlo, ylo, zlo, xhi, yhi, zhi}, or NULL for the surface's own. */
    const double* bounds;
    /** Cubic cell size, or <= 0 when sizing by resolution. Voxel only. */
    double cell_size;
    /** Grow the box by this on every side, in world units. */
    double padding;
    /** Grow the box by this fraction of its diagonal; the default is 0.1. */
    double padding_relative;
    /** Octree: refine while |distance| <= this * the cell's own diagonal. */
    double band_cells;
    /** Refuse above this many cells; re-checked after every octree pass. */
    int64_t max_cells;
    /** Octree: cell count per axis of the root lattice. */
    int64_t root_resolution;
    /** Octree: how many refinement passes. */
    int64_t max_depth;
    int32_t structure;      /**< a mio_sdf_structure */
    int32_t record_levels;  /**< octree: nonzero to attach refine:level */
    int64_t reserved[6];    /**< must be zero; room for additive growth */
    /** How the distances themselves are computed. Embedded by value, as in C++. */
    mio_sdf_opts distance;
} mio_compute_sdf_opts;

/** Zero-initialize compute_sdf options (every field its default). */
MIO_API void mio_compute_sdf_opts_init(mio_compute_sdf_opts* opts);

/**
 * Generate a grid over a surface and fill it with signed distances.
 *
 * The field is computed ONCE, on the final mesh: an octree pass interpolates
 * point_data, so a field attached mid-way would come out a smooth, plausible
 * interpolation of the coarse values rather than the distance.
 *
 * The generated mesh also carries the numeric `sdf:*` field_data header
 * describing itself. No file format persists arbitrary field_data, so that
 * header survives in memory only -- write the grid as `.vti`, whose
 * Origin/Spacing/WholeExtent attributes ARE the same information.
 *
 * @param surface      the surface to measure against.
 * @param opts         options; NULL is an error, since a sizing must be given.
 * @param dims_out     receives the ROOT cell counts (three entries), or NULL.
 * @param origin_out   receives the lattice lo corner (three entries), or NULL.
 * @param spacing_out  receives the FINEST cell size (three entries), or NULL.
 * @param max_depth_out receives how many octree passes ran (0 for voxel), or NULL.
 * @param num_banded   receives how many queries were clamped to the band, or NULL.
 * @param quality      receives the surface verdict, or NULL.
 * @return the grid (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_compute_sdf(const mio_mesh* surface, const mio_compute_sdf_opts* opts,
                                  int64_t dims_out[3], double origin_out[3],
                                  double spacing_out[3], int64_t* max_depth_out,
                                  int64_t* num_banded, mio_surface_quality* quality);

/**
 * Options for mio_remesh_volume_ex.
 *
 * ABI NOTE: the same append-only discipline as mio_compute_sdf_opts, which
 * this mirrors: `distance` is embedded BY VALUE (not flattened, unlike
 * mio_voxel_opts's sign/watertight_check-only fields) because
 * RemeshVolumeOptions, like SdfOptions, needs the FULL SurfaceDistanceOptions
 * surface, not just two of its fields -- and mio_sdf_opts carries its own
 * `reserved` tail, so growing it never shifts anything in THIS struct.
 * Always zero-initialize through mio_remesh_volume_opts_init().
 *
 * Unlike mio_remesh (which has both a flat function and this options-struct
 * growth path, because mio_remesh shipped flat first and grew once), this
 * operation ships ONLY the options-struct form: a brand-new entry point has
 * no back-compat flat signature to keep, so there is nothing to repeat.
 *
 * Exactly ONE of `resolution` and `cell_size` must be given, `mio_voxel_opts`'s
 * own rule.
 */
typedef struct mio_remesh_volume_opts {
    /** Cell counts (three entries), or NULL when sizing by cell_size. */
    const int64_t* resolution;
    /** Explicit bounds {xlo, ylo, zlo, xhi, yhi, zhi}, or NULL for the mesh's own. */
    const double* bounds;
    /** Cubic cell size of the root lattice, or <= 0 when sizing by resolution. */
    double cell_size;
    /** Grow the box by this on every side, in world units. */
    double padding;
    /** Grow the box by this fraction of its diagonal; the default is 0.1. */
    double padding_relative;
    /** Refuse to generate a root lattice above this many cells; <= 0 lifts the limit. */
    int64_t max_cells;
    /** Refuse an output with more tets than this. UNLIKE max_cells, this has
     *  no "<= 0 lifts the limit" escape hatch -- the check is unconditional,
     *  so 0 or a negative value refuses any non-empty output. Leave it at
     *  mio_remesh_volume_opts_init()'s default (20000000) unless a smaller
     *  budget is genuinely wanted. */
    int64_t max_tets;
    /** Fraction of a lattice vertex's shortest incident edge within which it
     *  may be warped onto the surface; 0 disables warping. */
    double warp_fraction;
    int64_t reserved[6]; /**< must be zero; room for additive growth */
    /** Sign, region restriction and watertight-check policy for every
     *  surface query. Embedded by value, as in C++ -- see the ABI note above. */
    mio_sdf_opts distance;
} mio_remesh_volume_opts;

/** Zero-initialize remesh_volume options to their default values. */
MIO_API void mio_remesh_volume_opts_init(mio_remesh_volume_opts* opts);

/**
 * Output counters and the surface verdict from mio_remesh_volume_ex; every
 * field is always written.
 */
typedef struct mio_remesh_volume_report {
    /** The verdict for the INPUT surface (or the extracted boundary of an
     *  input volume mesh), not the output. */
    mio_surface_quality input_quality;
    /** Tets emitted. */
    int64_t num_tets;
    /** Lattice vertices moved onto the surface. */
    int64_t num_vertices_warped;
    /** Cut sub-tets discarded as degenerate. */
    int64_t num_tets_rejected;
    /**
     * Non-manifold edges of the OUTPUT tet mesh's own boundary. `warp_fraction
     * == 0` gives a mathematically watertight cut and this is always 0 there.
     * A nonzero `warp_fraction` reuses a warped lattice vertex's position for
     * every cut edge it touches, which on some inputs can leave a small
     * number of coincident-but-topologically-distinct boundary edges near the
     * warped region -- MEASURED and reported here rather than hidden. See
     * doc/remesh_volume.md for the measured magnitude on this repo's own test
     * fixtures. A caller needing an exactly watertight result should check
     * this counter, or fall back to `warp_fraction = 0` / post-process with
     * mio_clean(weld=true).
     */
    int64_t num_non_manifold_edges;
    int64_t reserved[4]; /**< must be zero; room for additive growth */
} mio_remesh_volume_report;

/**
 * Retetrahedralize a volume (or a closed surface) at a caller-chosen
 * resolution by isosurface stuffing -- the volumetric sibling of mio_remesh,
 * generating an entirely new tet mesh rather than working on the input's own
 * cells. See doc/remesh_volume.md for the algorithm and its measured
 * warp/quality tradeoff.
 *
 * @param mesh   a volume mesh (its boundary is extracted internally) or a
 *               closed surface (triangle, quad, rectangular polygon).
 * @param opts   options; NULL is an error, since a resolution or cell size
 *               must be given -- mio_voxelize's own rule.
 * @param report optional out: the run's counters and input surface verdict;
 *               NULL to ignore it.
 * @return the retetrahedralized mesh, a single tetra block (free with
 *         mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_remesh_volume_ex(const mio_mesh* mesh, const mio_remesh_volume_opts* opts,
                                       mio_remesh_volume_report* report);

/**
 * ODT-remesh a tetrahedral mesh: raise its worst element quality by relocating
 * vertices AND flipping connectivity (2-3/3-2, predicate-free). The genuine
 * "ODT remeshing" sibling of mio_remesh_volume_ex (which generates a fresh
 * lattice mesh) and of mio_smooth method "odt" (which only moves points on
 * fixed connectivity). Tet-only. The point set is invariant, so point_data and
 * named Point regions carry; cell_data and Cell/Side regions are dropped (a
 * flip has no cell correspondence). With preserve_boundary the boundary surface
 * is exactly preserved. See doc/optimize_volume.md.
 *
 * Like mio_smooth this is a plain function with scalar counter out-params
 * rather than an opts/result struct; the pin mask (`mFrozen`) is not exposed on
 * the flat ABI, a documented gap like mio_smooth's own `frozen`.
 *
 * @param mesh              a tetra-only volume mesh.
 * @param max_iterations    optimisation sweeps; stops early at a fixed point.
 * @param relocate          nonzero to run the ODT vertex-relocation half.
 * @param flip              nonzero to run the topological-flip half.
 * @param preserve_boundary nonzero to pin boundary vertices during relocation.
 * @param min_improvement   the strict scaled-Jacobian gain a flip must deliver.
 * @param num_flips         receives the total flips accepted (2-3 + 3-2).
 * @param num_23_flips      receives the 2-3 flips accepted.
 * @param num_32_flips      receives the 3-2 flips accepted.
 * @param num_vertices_moved receives the vertices the relocation half moved.
 * @param num_tets          receives the output tet count.
 * @param min_quality_before receives the input min scaled Jacobian.
 * @param min_quality_after  receives the output min scaled Jacobian.
 * @return the optimised mesh, a single tetra block (free with mio_mesh_free),
 *         or NULL on failure. Every out-param is optional (pass NULL to skip).
 */
MIO_API mio_mesh* mio_optimize_volume(const mio_mesh* mesh, int max_iterations, int relocate,
                                      int flip, int preserve_boundary, double min_improvement,
                                      int64_t* num_flips, int64_t* num_23_flips,
                                      int64_t* num_32_flips, int64_t* num_vertices_moved,
                                      int64_t* num_tets, double* min_quality_before,
                                      double* min_quality_after);

/** Which dual area a per-vertex curvature is divided by. */
typedef enum mio_curvature_dual_area {
    /** Meyer et al.'s mixed Voronoi area. The default; converges better on an
     *  irregular tessellation. */
    MIO_CURVATURE_MIXED_VORONOI = 0,
    /** A third of each incident triangle's area. Cruder, but branch-free. */
    MIO_CURVATURE_BARYCENTRIC = 1
} mio_curvature_dual_area;

/**
 * Options for mio_compute_curvature.
 *
 * ABI NOTE: this struct is part of the installed library's permanent ABI. New
 * fields may only be appended, replacing `reserved` capacity; never reorder,
 * resize or repurpose an existing field. Always zero-initialize through
 * mio_curvature_opts_init() rather than by hand -- `mean` and `gaussian`
 * default to ON, so an all-zero struct is NOT the default here.
 *
 * Ships only this options-struct form: like mio_compute_sdf, a brand-new entry
 * point has no back-compat flat signature to keep, so there is nothing to grow
 * out of later the way mio_remesh had to.
 */
typedef struct mio_curvature_opts {
    /** Restrict to this named cell region; NULL or "" takes every surface cell. */
    const char* region;
    int32_t mean;             /**< nonzero (the default) attaches curvature:mean */
    int32_t gaussian;         /**< nonzero (the default) attaches curvature:gaussian */
    int32_t dual_area;        /**< a mio_curvature_dual_area */
    int32_t include_boundary; /**< nonzero computes a biased value at boundary vertices */
    int32_t record_area;      /**< nonzero attaches curvature:area */
    int32_t record_principal; /**< nonzero attaches curvature:principal, (n, 2) */
    int64_t reserved[6];      /**< must be zero; room for additive growth */
} mio_curvature_opts;

/**
 * Zero-initialize curvature options to their defaults: mean and gaussian ON,
 * the mixed-Voronoi dual area, every opt-in off and no region restriction.
 */
MIO_API void mio_curvature_opts_init(mio_curvature_opts* opts);

/** What mio_compute_curvature computed, and what it found on the way. */
typedef struct mio_curvature_report {
    /** The verdict for the INPUT surface. A nonzero `inconsistent_pairs` means
     *  the SIGN of curvature:mean is not trustworthy -- H is
     *  orientation-dependent and K is not. This never repairs its input. */
    mio_surface_quality quality;
    int64_t num_boundary;   /**< vertices left NaN because they sit on a boundary */
    int64_t num_isolated;   /**< vertices left NaN because no triangle references them */
    int64_t num_degenerate; /**< triangles skipped for zero area */
    /** The sum of every vertex's angle defect. For a CLOSED surface this is
     *  2*pi*chi exactly -- 4*pi for a sphere -- whatever the tessellation and
     *  whichever dual area was chosen. The cheapest check that a result is
     *  sane, and the reason it is reported rather than kept internal. */
    double total_angle_defect;
    int64_t reserved[4]; /**< must be zero; room for additive growth */
} mio_curvature_report;

/**
 * Per-vertex mean and Gaussian curvature of a surface mesh, by the angle
 * defect (K) and the cotangent Laplace-Beltrami operator (H).
 *
 * Triangles come from the same fan convert_cells(simplexify) uses; a volume or
 * polyhedron block is refused by name pointing at extract_surface, a
 * higher-order one pointing at linearize. See doc/curvature.md.
 *
 * @param mesh   a surface mesh.
 * @param opts   options; NULL means every mio_curvature_opts_init() default.
 * @param report optional out: the counters and the input surface's verdict.
 * @return the mesh with the requested point_data attached (free with
 *         mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_compute_curvature(const mio_mesh* mesh, const mio_curvature_opts* opts,
                                        mio_curvature_report* report);

/* ------------------------------------------------------------------------- */
/* Data operations                                                           */
/*                                                                           */
/* These act on a mesh's point_data / cell_data / field_data arrays rather    */
/* than on its geometry, which they never modify. Each returns a NEW mesh     */
/* (free it with mio_mesh_free) or NULL on failure; see mio_last_error().     */
/*                                                                           */
/* Name lists cross as an array of C strings plus an explicit count (the same */
/* convention as mio_merge), not NULL-terminated -- easier for Fortran to     */
/* build. Passing count == 0 means "every array at that location" for the     */
/* averaging and conditioning entry points.                                   */
/*                                                                           */
/* Documented flat-ABI gap: the combined data_manage (keep + drop + rename in */
/* one call) is not exposed, since an array-of-triples-of-enum-and-two-string */
/* parameter is unwieldy and the three primitives below compose to the same   */
/* effect. Likewise point/cell SETS never reach the C++ core at all.           */
/* ------------------------------------------------------------------------- */

/** Which of a mesh's three data maps an array lives in. */
typedef enum mio_data_location {
    MIO_DATA_POINT = 0, /**< point_data -- one row per mesh point */
    MIO_DATA_CELL = 1,  /**< cell_data -- one array per cell block */
    MIO_DATA_FIELD = 2  /**< field_data -- mesh-global */
} mio_data_location;

/** Weighting for mio_data_cell_to_point. */
typedef enum mio_cell_point_weight {
    MIO_WEIGHT_UNIFORM = 0, /**< every incident cell contributes equally */
    MIO_WEIGHT_MEASURE = 1  /**< weight by |cell measure| (length/area/volume) */
} mio_cell_point_weight;

/** Conditioning transform for mio_data_condition. */
typedef enum mio_condition_mode {
    MIO_COND_CLAMP = 0,      /**< x -> min(max(x, lo), hi) */
    MIO_COND_NORMALIZE = 1,  /**< map the array's [min,max] onto [lo,hi] */
    MIO_COND_STANDARDIZE = 2 /**< (x - mean) / stddev */
} mio_condition_mode;

/** Whether conditioning treats components independently or by row magnitude. */
typedef enum mio_condition_scope {
    MIO_SCOPE_COMPONENT = 0, /**< each trailing component independently */
    MIO_SCOPE_MAGNITUDE = 1  /**< by row magnitude; direction preserved */
} mio_condition_scope;

/** What reaches the output for non-finite (NaN/inf) values. They are ALWAYS
 *  excluded from reductions regardless of this setting. */
typedef enum mio_nan_policy {
    MIO_NAN_IGNORE = 0,  /**< pass through unchanged */
    MIO_NAN_REPLACE = 1, /**< substitute the caller's replacement value */
    MIO_NAN_FAIL = 2     /**< fail with MIO_ERR_INVALID_ARG */
} mio_nan_policy;

/**
 * Drop the named data arrays at one location.
 * @param mesh           input mesh (unmodified).
 * @param location       which data map to drop from.
 * @param names          array of `count` names.
 * @param count          number of entries in `names`.
 * @param ignore_missing non-zero to skip names that do not exist.
 * @return a new mesh (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_data_drop(const mio_mesh* mesh, mio_data_location location,
                                const char* const* names, int64_t count, int ignore_missing);

/**
 * Keep only the named data arrays at one location, dropping the rest there.
 * The other two locations are left untouched; `count == 0` drops everything at
 * `location`.
 * @return a new mesh (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_data_keep(const mio_mesh* mesh, mio_data_location location,
                                const char* const* names, int64_t count, int ignore_missing);

/**
 * Rename one data array, preserving its values, dtype and shape.
 * @return a new mesh (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_data_rename(const mio_mesh* mesh, mio_data_location location,
                                  const char* from_name, const char* to_name);

/**
 * Average point_data onto the cells: each cell's value is the mean over its own
 * nodes. The output is always Float64. `count == 0` converts every point_data
 * array.
 * @return a new mesh (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_data_point_to_cell(const mio_mesh* mesh, const char* const* names,
                                         int64_t count, const char* suffix);

/**
 * Average cell_data onto the points: each point's value is the (optionally
 * measure-weighted) mean over the incident cells. A point touched by no cell
 * gets NaN. The output is always Float64.
 * @return a new mesh (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_data_cell_to_point(const mio_mesh* mesh, const char* const* names,
                                         int64_t count, mio_cell_point_weight weight,
                                         const char* suffix);

/**
 * Evaluate an elementwise expression over the arrays at `location` and store
 * the result there as `output_name`.
 *
 * The grammar accepts + - * / unary minus, parentheses, numeric literals,
 * array names, and the functions abs, sqrt, min, max and norm. Nothing else is
 * evaluated -- there is no arbitrary-code path.
 * @param mesh        input mesh (unmodified).
 * @param expression  the expression text.
 * @param location    where operands are looked up and the result is stored.
 * @param output_name name of the new array (required).
 * @param overwrite   non-zero to allow replacing an existing array.
 * @return a new mesh (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_data_calc(const mio_mesh* mesh, const char* expression,
                                mio_data_location location, const char* output_name,
                                int overwrite);

/**
 * Condition the values of the selected data arrays (clamp / normalize /
 * standardize). For cell_data the statistics are computed jointly across all
 * cell blocks. `count == 0` conditions every array at `location`.
 * @param lo,hi           clamp bounds, or the normalize target range.
 * @param nan_replacement used when `nan_policy` is MIO_NAN_REPLACE.
 * @param suffix          NULL or "" replaces in place; otherwise the result is
 *                        stored as name+suffix.
 * @return a new mesh (free with mio_mesh_free), or NULL on failure.
 */
MIO_API mio_mesh* mio_data_condition(const mio_mesh* mesh, mio_data_location location,
                                     const char* const* names, int64_t count,
                                     mio_condition_mode mode, double lo, double hi,
                                     mio_condition_scope scope, mio_nan_policy nan_policy,
                                     double nan_replacement, const char* suffix);

/** Opaque per-array data summary. Destroy with mio_data_info_free(). */
typedef struct mio_data_info mio_data_info;

/** Fixed-size summary of one data array (see mio_data_info_entry). Per-component
 *  statistics are retrieved separately with mio_data_info_component(). */
typedef struct mio_data_array_info {
    int location;           /**< a mio_data_location */
    int dtype;              /**< a mio_dtype, as stored */
    int64_t num_blocks;     /**< cell_data: number of cell blocks; else 1 */
    int64_t num_entries;    /**< rows: points / cells over all blocks / length */
    int64_t num_components; /**< product of the trailing dimensions */
    int64_t num_values;     /**< num_entries * num_components */
    double min;             /**< over finite values; NaN when there are none */
    double max;             /**< over finite values; NaN when there are none */
    double mean;            /**< over finite values; NaN when there are none */
    int64_t num_nan;        /**< count of NaN values */
    int64_t num_inf;        /**< count of +/-inf values */
    int64_t num_finite;     /**< count of finite values */
    int inconsistent_blocks; /**< cell_data blocks disagree in component count */
} mio_data_array_info;

/**
 * Summarize every data array a mesh carries (read-only).
 * @return a result handle (free with mio_data_info_free), or NULL on failure.
 */
MIO_API mio_data_info* mio_data_info_create(const mio_mesh* mesh);

/** @return the number of arrays described, or -1 on error. */
MIO_API int64_t mio_data_info_count(const mio_data_info* info);

/**
 * Copy the `index`-th array's name into `buf`.
 * @return the required length excluding the NUL (so a value >= buflen means the
 *         name was truncated), or -1 on error.
 */
MIO_API int64_t mio_data_info_name(const mio_data_info* info, int64_t index, char* buf,
                                   int64_t buflen);

/** Fill `out` with the `index`-th array's summary. */
MIO_API mio_status mio_data_info_entry(const mio_data_info* info, int64_t index,
                                       mio_data_array_info* out);

/**
 * Per-component statistics of the `index`-th array. Any out pointer may be NULL.
 * @param comp component index, in [0, num_components).
 */
MIO_API mio_status mio_data_info_component(const mio_data_info* info, int64_t index, int64_t comp,
                                           double* min, double* max, double* mean);

/** Destroy a summary handle. Safe to call with NULL. */
MIO_API void mio_data_info_free(mio_data_info* info);

/** Opaque cell-measure-weighted field-integration report. Destroy with
 *  mio_data_integrate_free(). See meshioplusplus::data_integrate. */
typedef struct mio_data_integrate mio_data_integrate;

/** Fixed-size summary of one array's or one region's reduction (see
 *  mio_data_integrate_entry / mio_data_integrate_region_entry). Per-component
 *  totals are retrieved separately with mio_data_integrate_component /
 *  mio_data_integrate_region_component. */
typedef struct mio_field_integral_info {
    int64_t num_components; /**< product of the trailing dimensions */
    int64_t num_cells;      /**< cells with a computable measure */
    int64_t num_skipped;    /**< cells excluded: unmeasurable geometry */
} mio_field_integral_info;

/**
 * Cell-measure-weighted total/mean of one or more cell_data arrays, for the
 * whole mesh and independently for every named Cell region -- gradient's
 * integration counterpart. A point_data-only name fails, naming
 * point_data_to_cell_data (CLI `data to-cell`) as the fix.
 * @param names array names to integrate, or NULL for every cell_data array.
 * @param count length of names; ignored when names is NULL.
 * @return a result handle (free with mio_data_integrate_free), or NULL on
 *         failure.
 */
MIO_API mio_data_integrate* mio_data_integrate_create(const mio_mesh* mesh,
                                                       const char* const* names, int64_t count);

/** @return the number of arrays described, or -1 on error. */
MIO_API int64_t mio_data_integrate_count(const mio_data_integrate* result);

/**
 * Copy the `index`-th array's name into `buf`.
 * @return the required length excluding the NUL (so a value >= buflen means the
 *         name was truncated), or -1 on error.
 */
MIO_API int64_t mio_data_integrate_name(const mio_data_integrate* result, int64_t index,
                                        char* buf, int64_t buflen);

/** Fill `out` with the `index`-th array's whole-mesh summary. */
MIO_API mio_status mio_data_integrate_entry(const mio_data_integrate* result, int64_t index,
                                            mio_field_integral_info* out);

/**
 * Per-component whole-mesh totals of the `index`-th array. Any out pointer
 * may be NULL.
 * @param comp component index, in [0, num_components).
 */
MIO_API mio_status mio_data_integrate_component(const mio_data_integrate* result, int64_t index,
                                                int64_t comp, double* total, double* mean,
                                                double* domain_measure, int64_t* num_nan);

/** @return the number of named Cell regions reported for the `index`-th
 *          array, or -1 on error. */
MIO_API int64_t mio_data_integrate_region_count(const mio_data_integrate* result, int64_t index);

/** Copy the `index`-th array's `region`-th region name into `buf`.
 * @return the required length excluding the NUL, or -1 on error. */
MIO_API int64_t mio_data_integrate_region_name(const mio_data_integrate* result, int64_t index,
                                               int64_t region, char* buf, int64_t buflen);

/** Fill `out` with the `index`-th array's `region`-th region summary. */
MIO_API mio_status mio_data_integrate_region_entry(const mio_data_integrate* result, int64_t index,
                                                   int64_t region, mio_field_integral_info* out);

/** Per-component totals of the `index`-th array's `region`-th region. Any out
 *  pointer may be NULL. */
MIO_API mio_status mio_data_integrate_region_component(const mio_data_integrate* result,
                                                        int64_t index, int64_t region,
                                                        int64_t comp, double* total, double* mean,
                                                        double* domain_measure, int64_t* num_nan);

/** Destroy a result handle. Safe to call with NULL. */
MIO_API void mio_data_integrate_free(mio_data_integrate* result);

/**
 * Renumber a mesh (RCM / Morton / Hilbert) as a pure permutation of node and
 * element indices, to reduce sparse-matrix bandwidth / improve cache locality.
 * @param mesh   input mesh.
 * @param method "rcm", "morton", or "hilbert".
 * @return a result handle (free with mio_reorder_result_free), or NULL on
 *         failure (see mio_last_error()).
 */
MIO_API mio_reorder_result* mio_reorder(const mio_mesh* mesh, const char* method);

/**
 * Borrow the renumbered mesh from a reorder result. The returned handle is
 * owned by the result: valid until mio_reorder_result_free(result); do NOT
 * pass it to mio_mesh_free().
 * @param result a reorder result.
 * @return the renumbered mesh, or NULL if `result` is NULL.
 */
MIO_API const mio_mesh* mio_reorder_result_mesh(const mio_reorder_result* result);

/**
 * Zero-copy borrow of the node permutation (int64, shape (num_points,),
 * old index -> new index). The pointer is valid until the result is freed.
 * Any out-param may be NULL.
 * @param result a reorder result.
 * @param data   receives a pointer into result-owned int64 memory.
 * @param dtype  receives MIO_INT64.
 * @param n      receives num_points.
 */
MIO_API mio_status mio_reorder_result_node_perm(const mio_reorder_result* result, const void** data,
                                                mio_dtype* dtype, int64_t* n);

/**
 * Number of per-block cell permutations in a reorder result (equal to the
 * number of cell blocks of the mesh).
 * @param result a reorder result.
 * @return the block count, or -1 on error (see mio_last_error()).
 */
MIO_API int64_t mio_reorder_result_num_cell_perms(const mio_reorder_result* result);

/**
 * Zero-copy borrow of cell block `block`'s cell permutation (int64, shape
 * (num_cells_in_block,), old index -> new index). Any out-param may be NULL.
 * @param result a reorder result.
 * @param block  cell-block index in [0, mio_reorder_result_num_cell_perms).
 * @param data   receives a pointer into result-owned int64 memory.
 * @param dtype  receives MIO_INT64.
 * @param n      receives the block's cell count.
 */
MIO_API mio_status mio_reorder_result_cell_perm(const mio_reorder_result* result, int64_t block,
                                                const void** data, mio_dtype* dtype, int64_t* n);

/**
 * Transfer ownership of the renumbered mesh out of the result. Afterwards the
 * result no longer owns a mesh (its borrowed accessor yields an empty mesh).
 * @param result a reorder result.
 * @return a new owning mesh handle (free it with mio_mesh_free), or NULL on error.
 */
MIO_API mio_mesh* mio_reorder_result_take_mesh(mio_reorder_result* result);

/** Free a reorder result (and its owned mesh and permutation arrays). */
MIO_API void mio_reorder_result_free(mio_reorder_result* result);

/**
 * Connectivity bandwidth: the maximum over all cells of (max node index - min
 * node index) within a cell. A before/after measure of a reordering.
 * @param mesh input mesh.
 * @return the bandwidth (0 for a node-less mesh), or -1 on error.
 */
MIO_API int64_t mio_compute_bandwidth(const mio_mesh* mesh);

/**
 * Compare two meshes and produce a structured report. Compares points, cell
 * blocks (types + counts + connectivity), and point/cell/field data within the
 * tolerance abs_err <= atol + rtol*|expected| (expected taken from mesh `a`).
 * Named point_sets/cell_sets are NOT compared (they never reach the C++ core).
 * @param a         first ("expected") mesh.
 * @param b         second mesh.
 * @param atol      absolute tolerance.
 * @param rtol      relative tolerance.
 * @param unordered nonzero to match points by spatial proximity (tolerant to a
 *                  shuffled node order) instead of by index.
 * @param out       receives a new result handle (free with mio_diff_result_free).
 * @return MIO_OK, or an error status (see mio_last_error()).
 */
MIO_API mio_status mio_diff(const mio_mesh* a, const mio_mesh* b, double atol, double rtol,
                            int unordered, mio_diff_result** out);

/**
 * Convenience boolean: are two meshes equal within tolerance? (verdict is not
 * MIO_DIFF_DIFFERENT). Named sets are not considered.
 * @param a,b       the meshes to compare.
 * @param atol,rtol tolerances.
 * @param unordered nonzero for proximity node matching.
 * @param out_equal receives 1 (equal) or 0 (different).
 * @return MIO_OK, or an error status (see mio_last_error()).
 */
MIO_API mio_status mio_meshes_equal(const mio_mesh* a, const mio_mesh* b, double atol, double rtol,
                                    int unordered, int* out_equal);

/** Overall verdict of a diff result (MIO_DIFF_DIFFERENT on a NULL result). */
MIO_API mio_diff_verdict mio_diff_result_verdict(const mio_diff_result* result);

/**
 * Points comparison summary from a diff result. Any out-param may be NULL.
 * @param result        a diff result.
 * @param max_abs_error largest absolute coordinate error.
 * @param max_rel_error largest relative coordinate error.
 * @param worst_index   flat index of the worst-offending coordinate, or -1.
 * @param num_exceeding number of coordinates beyond tolerance.
 * @return MIO_OK, or an error status.
 */
MIO_API mio_status mio_diff_result_point_summary(const mio_diff_result* result,
                                                 double* max_abs_error, double* max_rel_error,
                                                 int64_t* worst_index, int64_t* num_exceeding);

/** Number of per-block comparisons in a diff result (-1 on a NULL result). */
MIO_API int64_t mio_diff_result_num_block_diffs(const mio_diff_result* result);

/**
 * One block comparison from a diff result. Any out-param may be NULL.
 * @param result             a diff result.
 * @param block              block index in [0, mio_diff_result_num_block_diffs).
 * @param type_mismatch      set to 1 if the two blocks have different cell types.
 * @param count_mismatch     set to 1 if they have different cell counts.
 * @param conn_mismatch_count number of cells whose connectivity differs.
 * @return MIO_OK, MIO_ERR_NOT_FOUND if `block` is out of range, else an error.
 */
MIO_API mio_status mio_diff_result_block(const mio_diff_result* result, int64_t block,
                                         int* type_mismatch, int* count_mismatch,
                                         int64_t* conn_mismatch_count);

/** Free a diff result. */
MIO_API void mio_diff_result_free(mio_diff_result* result);

/* ---------------------------------------------------------------------
 * Building a mesh (setters -- all COPY caller memory)
 * --------------------------------------------------------------------- */

/**
 * Assign the point coordinates, replacing any previous ones.
 * @param dtype      MIO_FLOAT32 or MIO_FLOAT64; stored as given.
 * @param num_points number of points.
 * @param dim        coordinates per point (usually 2 or 3).
 * @param xyz        row-major `(num_points, dim)` buffer.
 */
MIO_API mio_status mio_mesh_set_points(mio_mesh* mesh, mio_dtype dtype, int64_t num_points,
                                       int64_t dim, const void* xyz);

/**
 * Append one homogeneous cell block.
 * @param cell_type      meshio type name (e.g. "triangle", "tetra10").
 * @param num_cells      number of cells in the block.
 * @param nodes_per_cell nodes per cell (must match the type's fixed count
 *                       when it has one).
 * @param dtype          MIO_INT32 or MIO_INT64; widened to int64 internally
 *                       (the core's connectivity type).
 * @param connectivity   row-major `(num_cells, nodes_per_cell)` buffer of
 *                       0-based point indices.
 */
MIO_API mio_status mio_mesh_add_cell_block(mio_mesh* mesh, const char* cell_type,
                                           int64_t num_cells, int64_t nodes_per_cell,
                                           mio_dtype dtype, const void* connectivity);

/**
 * Append one 1-level ragged (jagged polygon) cell block, in flat CSR form.
 *
 * @param cell_type   meshio type name, e.g. "polygon" or "polygon2".
 * @param num_cells   number of cells in the block.
 * @param row_offsets `num_cells + 1` entries: each cell's start index into
 *                    `nodes`. Must be non-decreasing, start at 0 and end at
 *                    `num_nodes`, so cell `c`'s node ids are
 *                    `nodes[row_offsets[c] .. row_offsets[c + 1])`.
 * @param nodes       every row's 0-based node ids concatenated.
 * @param num_nodes   length of `nodes`.
 *
 * Both arrays are int64 -- unlike mio_mesh_add_cell_block there is no `dtype`
 * parameter. A CSR pair is two arrays that must agree, one dtype covering both
 * invites the wrong pairing and two doubles the arity for no real gain; offsets
 * are inherently int64-ranged; and the int32 path on the rectangular setter
 * exists because numpy/Fortran meshes commonly *are* int32 rectangular, which a
 * hand-built CSR is not. Widen int32 ragged data once into a scratch buffer.
 */
MIO_API mio_status mio_mesh_add_polygon_block(mio_mesh* mesh, const char* cell_type,
                                              int64_t num_cells, const int64_t* row_offsets,
                                              const int64_t* nodes, int64_t num_nodes);

/**
 * Append one 2-level ragged (polyhedron) cell block, in flat CSR form: each
 * cell is a list of faces, each face a list of node ids.
 *
 * @param cell_type    meshio type name, e.g. "polyhedron" or "polyhedron12".
 * @param num_cells    number of cells in the block.
 * @param cell_offsets `num_cells + 1` entries: each cell's start index into the
 *                     FACE list (not into `nodes`), ending at `num_faces`.
 * @param num_faces    total number of faces across every cell.
 * @param face_offsets `num_faces + 1` entries: each face's start index into
 *                     `nodes`, ending at `num_nodes`.
 * @param nodes        every face's 0-based node ids concatenated.
 * @param num_nodes    length of `nodes`.
 *
 * So face `f` of cell `c` spans
 * `nodes[face_offsets[cell_offsets[c] + f] .. face_offsets[cell_offsets[c] + f + 1])`.
 * This is exactly the shape the JS API calls `{data, faceOffsets, cellOffsets}`.
 *
 * Faces should be wound so their right-hand normal points out of the cell, but
 * meshio++ does not require it: the geometric kernels repair an inconsistent
 * winding rather than rejecting it. See doc/polyhedra.md.
 */
MIO_API mio_status mio_mesh_add_polyhedron_block(mio_mesh* mesh, const char* cell_type,
                                                 int64_t num_cells, const int64_t* cell_offsets,
                                                 int64_t num_faces, const int64_t* face_offsets,
                                                 const int64_t* nodes, int64_t num_nodes);

/**
 * Attach a named per-point data array. `shape[0]` must equal the number of
 * points (e.g. `{num_points}` for a scalar field, `{num_points, 3}` for a
 * vector field).
 */
MIO_API mio_status mio_mesh_add_point_data(mio_mesh* mesh, const char* name, mio_dtype dtype,
                                           int32_t ndim, const int64_t* shape, const void* data);

/**
 * Append one per-cell-block array to the named cell-data field: call once per
 * cell block, in block order, after adding the blocks (`shape[0]` must equal
 * that block's cell count).
 */
MIO_API mio_status mio_mesh_append_cell_data(mio_mesh* mesh, const char* name, mio_dtype dtype,
                                             int32_t ndim, const int64_t* shape, const void* data);

/** Attach a named mesh-level (field) data array of arbitrary shape. */
MIO_API mio_status mio_mesh_add_field_data(mio_mesh* mesh, const char* name, mio_dtype dtype,
                                           int32_t ndim, const int64_t* shape, const void* data);

/* ---------------------------------------------------------------------
 * Reading a mesh back (getters -- zero-copy, see rule 3)
 * --------------------------------------------------------------------- */

/** @return the number of points, or -1 if `mesh` is NULL. */
MIO_API int64_t mio_mesh_num_points(const mio_mesh* mesh);

/** @return coordinates per point (usually 2 or 3), or -1 if `mesh` is NULL. */
MIO_API int64_t mio_mesh_point_dim(const mio_mesh* mesh);

/** Borrow the `(num_points, point_dim)` row-major coordinate buffer. */
MIO_API mio_status mio_mesh_get_points(const mio_mesh* mesh, const void** data, mio_dtype* dtype);

/** @return the number of cell blocks, or -1 if `mesh` is NULL. */
MIO_API int64_t mio_mesh_num_cell_blocks(const mio_mesh* mesh);

/**
 * Describe cell block `block`. Any out-param may be NULL.
 * A ragged block (`is_ragged == 1`: polygons/polyhedra of varying size)
 * reports `nodes_per_cell == 0`; read its connectivity with
 * mio_poly_conn_create(), and use mio_mesh_cell_block_info_ex() to learn
 * whether it is 1-level (polygon) or 2-level (polyhedron).
 */
MIO_API mio_status mio_mesh_cell_block_info(const mio_mesh* mesh, int64_t block,
                                            int64_t* num_cells, int64_t* nodes_per_cell,
                                            int32_t* is_ragged);

/**
 * One cell block's full shape -- the growable successor to the five-argument
 * mio_mesh_cell_block_info(), which cannot gain an out-param without breaking
 * ABI and so stays supported unchanged. Grows only into `reserved`, exactly
 * like mio_read_opts / mio_write_opts.
 */
typedef struct mio_cell_block_info {
    int64_t num_cells;
    int64_t nodes_per_cell; /**< 0 for a ragged block. */
    int32_t is_ragged;      /**< 1 for a polygon or polyhedron block. */
    int32_t is_polyhedron;  /**< 1 for a 2-level block; implies `is_ragged`. */
    int64_t num_faces;      /**< total faces; `num_cells` unless `is_polyhedron`. */
    int64_t num_nodes;      /**< total node ids the block stores. */
    int64_t reserved[6];
} mio_cell_block_info;

/**
 * Fill `out` with cell block `block`'s shape.
 *
 * Note `num_faces` / `num_nodes` are an O(cells) walk on the MESHIO backend,
 * which stores ragged blocks as nested vectors rather than CSR -- fine to size
 * a buffer with, not something to call per cell in a loop.
 */
MIO_API mio_status mio_mesh_cell_block_info_ex(const mio_mesh* mesh, int64_t block,
                                               mio_cell_block_info* out);

/** Copy cell block `block`'s meshio type name into `buf` (string rule 5). */
MIO_API int64_t mio_mesh_cell_block_type(const mio_mesh* mesh, int64_t block, char* buf,
                                         int64_t buflen);

/** Borrow cell block `block`'s row-major `(num_cells, nodes_per_cell)`
 *  0-based connectivity. Fails with MIO_ERR_UNSUPPORTED on a ragged block,
 *  which has no rectangular buffer to borrow -- use mio_poly_conn_create(). */
MIO_API mio_status mio_mesh_cell_block_conn(const mio_mesh* mesh, int64_t block, const void** conn,
                                            mio_dtype* dtype);

/* ---------------------------------------------------------------------
 * Ragged (polygon / polyhedron) connectivity
 *
 * A ragged block's connectivity has no single rectangular buffer to borrow, so
 * it is read through an owning flat-CSR *snapshot* rather than a rule-3 borrow:
 * the MESHIO backend stores nested vectors and the KRATOS backend rebuilds its
 * blocks lazily, so there is no offsets array inside the mesh to point at. The
 * snapshot is therefore independent of the mesh once created -- it stays valid
 * across mutating calls, unlike every other getter here -- and must be released
 * with mio_poly_conn_free().
 *
 * Layout, identical to the JS boundary's and to the setters above:
 *   nodes         concatenated 0-based node ids of every face
 *   face_offsets  num_faces + 1 entries, each face's start in `nodes`
 *   cell_offsets  num_cells + 1 entries, each cell's start in the FACE list
 *
 * A 1-level (polygon) block has exactly one face per cell: `face_offsets` is
 * then what the JS API calls `rowOffsets`, and `cell_offsets` is NULL with a
 * count of 0 -- a NULL nobody can mistake for data, rather than a synthesized
 * 0,1,2,... identity that looks like information.
 *
 * See doc/polyhedra.md.
 * --------------------------------------------------------------------- */

/** Opaque snapshot of one ragged block's connectivity. Free with
 *  mio_poly_conn_free(). */
typedef struct mio_poly_conn mio_poly_conn;

/** Fixed-size description of a snapshot's CSR shape. */
typedef struct mio_poly_conn_shape {
    int32_t is_polyhedron; /**< 1 = 2-level (cell -> faces -> nodes), 0 = 1-level. */
    int32_t reserved0;
    int64_t num_cells;
    int64_t num_faces; /**< 2-level: total faces. 1-level: equal to `num_cells`. */
    int64_t num_nodes; /**< length of the node array. */
    int64_t reserved[4];
} mio_poly_conn_shape;

/**
 * Snapshot cell block `block`'s ragged connectivity.
 * Fails with MIO_ERR_UNSUPPORTED on a rectangular block -- use
 * mio_mesh_cell_block_conn(), which borrows and does not copy.
 * @return a handle (free with mio_poly_conn_free), or NULL on failure.
 */
MIO_API mio_poly_conn* mio_poly_conn_create(const mio_mesh* mesh, int64_t block);

/** Fill `out` with the snapshot's shape. */
MIO_API mio_status mio_poly_conn_get_shape(const mio_poly_conn* poly, mio_poly_conn_shape* out);

/** Borrow the concatenated node ids. `count` may be NULL. */
MIO_API const int64_t* mio_poly_conn_nodes(const mio_poly_conn* poly, int64_t* count);

/** Borrow the `num_faces + 1` face offsets into the node array. */
MIO_API const int64_t* mio_poly_conn_face_offsets(const mio_poly_conn* poly, int64_t* count);

/** Borrow the `num_cells + 1` cell offsets into the face list.
 *  Returns NULL with `*count = 0` for a 1-level (polygon) block. */
MIO_API const int64_t* mio_poly_conn_cell_offsets(const mio_poly_conn* poly, int64_t* count);

/** Destroy a snapshot. Safe to call with NULL. */
MIO_API void mio_poly_conn_free(mio_poly_conn* poly);

/** @return the number of named point-data arrays, or -1 if `mesh` is NULL. */
MIO_API int64_t mio_mesh_num_point_data(const mio_mesh* mesh);

/** Copy the `index`-th point-data name (ascending lexicographic order --
 *  identical on every mesh backend) into `buf` (string rule 5). */
MIO_API int64_t mio_mesh_point_data_name(const mio_mesh* mesh, int64_t index, char* buf,
                                         int64_t buflen);

/** Borrow the named point-data array. `shape` (when non-NULL) must hold
 *  MIO_MAX_NDIM elements; any out-param may be NULL. */
MIO_API mio_status mio_mesh_get_point_data(const mio_mesh* mesh, const char* name,
                                           const void** data, mio_dtype* dtype, int32_t* ndim,
                                           int64_t* shape);

/** @return the number of named cell-data fields, or -1 if `mesh` is NULL. */
MIO_API int64_t mio_mesh_num_cell_data(const mio_mesh* mesh);

/** Copy the `index`-th cell-data name (sorted, as above) into `buf`. */
MIO_API int64_t mio_mesh_cell_data_name(const mio_mesh* mesh, int64_t index, char* buf,
                                        int64_t buflen);

/** @return how many per-block arrays the named cell-data field has (normally
 *          the number of cell blocks), or -1 on error. */
MIO_API int64_t mio_mesh_cell_data_num_blocks(const mio_mesh* mesh, const char* name);

/** Borrow the named cell-data field's array for cell block `block`. */
MIO_API mio_status mio_mesh_get_cell_data(const mio_mesh* mesh, const char* name, int64_t block,
                                          const void** data, mio_dtype* dtype, int32_t* ndim,
                                          int64_t* shape);

/** @return the number of named field-data arrays, or -1 if `mesh` is NULL. */
MIO_API int64_t mio_mesh_num_field_data(const mio_mesh* mesh);

/** Copy the `index`-th field-data name (sorted, as above) into `buf`. */
MIO_API int64_t mio_mesh_field_data_name(const mio_mesh* mesh, int64_t index, char* buf,
                                         int64_t buflen);

/** Borrow the named field-data array. */
MIO_API mio_status mio_mesh_get_field_data(const mio_mesh* mesh, const char* name,
                                           const void** data, mio_dtype* dtype, int32_t* ndim,
                                           int64_t* shape);

/* ---------------------------------------------------------------------
 * Named regions
 *
 * A region is a named group of mesh entities: a gmsh physical group, an Exodus
 * block / node set / side set, an Abaqus *NSET / *ELSET / *SURFACE, a MED
 * family, a Kratos SubModelPart. Before meshio++ 8.1 these never left the
 * Python layer at all -- this section is what closed that gap. See
 * doc/regions.md.
 *
 * Entry conventions, by kind:
 *   MIO_REGION_POINT  point indices,                     stride 1
 *   MIO_REGION_CELL   GLOBAL cell indices, block-major   stride 1
 *                     (block 0's cells first, then block 1's, ...)
 *   MIO_REGION_SIDE   (global cell index, local facet)   stride 2
 *
 * Facets are numbered as meshio++ numbers the faces of a 3-D cell and the edges
 * of a 2-D one. Entries are stored ascending and de-duplicated.
 * --------------------------------------------------------------------- */

/** Opaque snapshot of a mesh's regions. Destroy with mio_regions_free(). */
typedef struct mio_regions mio_regions;

/**
 * Snapshot every named region a mesh carries (read-only).
 * @return a handle (free with mio_regions_free), or NULL on failure.
 */
MIO_API mio_regions* mio_regions_create(const mio_mesh* mesh);

/** @return the number of regions described, or -1 on error. */
MIO_API int64_t mio_regions_count(const mio_regions* regions);

/**
 * Copy the `index`-th region's name into `buf`.
 * @return the required length excluding the NUL (so a value >= buflen means the
 *         name was truncated), or -1 on error.
 */
MIO_API int64_t mio_regions_name(const mio_regions* regions, int64_t index, char* buf,
                                 int64_t buflen);

/** Fill `out` with the `index`-th region's kind / dim / tag / sizes. */
MIO_API mio_status mio_regions_info(const mio_regions* regions, int64_t index,
                                    mio_region_info* out);

/**
 * Borrow the `index`-th region's entries: `num_entries * stride` int64 values,
 * row-major. Valid until mio_regions_free().
 * @param count receives `num_entries * stride` (may be NULL).
 * @return the entry buffer, or NULL on error / when the region is empty.
 */
MIO_API const int64_t* mio_regions_entries(const mio_regions* regions, int64_t index,
                                           int64_t* count);

/** Destroy a regions handle. Safe to call with NULL. */
MIO_API void mio_regions_free(mio_regions* regions);

/**
 * Add a region to a mesh, replacing any existing one with the same
 * (kind, name, dim, tag). Entries are copied and canonicalized (sorted,
 * de-duplicated).
 * @param entries `count` int64 values: `count` indices for POINT/CELL, or
 *                `count / 2` (cell, facet) pairs for SIDE.
 */
MIO_API mio_status mio_mesh_add_region(mio_mesh* mesh, const char* name, mio_region_kind kind,
                                       int32_t dim, int64_t tag, const int64_t* entries,
                                       int64_t count);

/* ---------------------------------------------------------------------
 * Transient (time-series) XDMF writing
 *
 * The write half of what mio_read_opts.time_step / mio_read_metadata_time_value
 * expose on the read side. A solver writes the mesh ONCE and then one cheap step
 * per solve, so this is a stateful handle rather than a (path, mesh) call -- the
 * one writer here that mio_write() cannot express.
 *
 *   mio_xdmf_series* s = mio_xdmf_series_create("out.xdmf", "HDF", -1);
 *   mio_xdmf_series_write_points_cells(s, mesh);
 *   for (k = 0; k < nsteps; ++k) { solve(); mio_xdmf_series_write_data(s, k*dt, mesh); }
 *   mio_xdmf_series_free(s);          // finalizes if mio_xdmf_series_finalize wasn't called
 *
 * The .xdmf light data is buffered and written once, at finalize; a series is
 * therefore only readable after the handle is finalized or freed. Heavy data for
 * "HDF" goes to a <path minus extension>.h5 SIBLING of the .xdmf.
 * --------------------------------------------------------------------- */

/** Opaque transient XDMF writer. Destroy with mio_xdmf_series_free(). */
typedef struct mio_xdmf_series mio_xdmf_series;

/**
 * Open a transient XDMF series for writing. Nothing is written yet.
 * @param path        the .xdmf/.xmf light-data file to write.
 * @param data_format "HDF" (NULL means "HDF"; needs an HDF5-enabled build),
 *                    "XML" or "Binary".
 * @param gzip_level  gzip level for "HDF" datasets, negative for none.
 * @return a handle (free with mio_xdmf_series_free), or NULL on failure -- an
 *         unknown data_format, or "HDF" without HDF5 support.
 */
MIO_API mio_xdmf_series* mio_xdmf_series_create(const char* path, const char* data_format,
                                                int32_t gzip_level);

/** How a series relates to a file already at its path (see mio_xdmf_series_opts). */
typedef enum mio_xdmf_series_mode {
    MIO_XDMF_SERIES_TRUNCATE = 0, /**< start a fresh series (the default) */
    MIO_XDMF_SERIES_APPEND = 1    /**< continue the existing one, if any */
} mio_xdmf_series_mode;

/**
 * Options for mio_xdmf_series_create_ex().
 *
 * ABI NOTE: as for mio_read_opts/mio_write_opts, new fields may only be
 * appended, replacing `reserved` capacity; never reorder, resize or repurpose
 * an existing field. Always zero-initialize through mio_xdmf_series_opts_init().
 */
typedef struct mio_xdmf_series_opts {
    /** "HDF" (NULL means "HDF"), "XML" or "Binary". */
    const char* data_format;
    int32_t gzip_level; /**< gzip level for "HDF" datasets, negative for none */
    int32_t mode;       /**< a mio_xdmf_series_mode */
    /** Nonzero flushes the light data after every write_data. Off by default:
     *  a flush re-serializes the whole document, so per-step flushing is
     *  quadratic in the step count. Drive mio_xdmf_series_flush() yourself for
     *  any other cadence. */
    int32_t auto_flush;
    int32_t reserved_pad; /**< must be zero; keeps the int64 tail aligned */
    int64_t reserved[6];  /**< must be zero; room for additive growth */
} mio_xdmf_series_opts;

/** Initialize `opts` to the defaults (HDF, no gzip, truncate, no auto-flush). */
MIO_API void mio_xdmf_series_opts_init(mio_xdmf_series_opts* opts);

/**
 * Open a transient XDMF series honouring `opts`. mio_xdmf_series_create() is
 * exactly mio_xdmf_series_create_ex(path, <defaults with that data_format and
 * gzip_level>) and is unchanged.
 * @return a handle (free with mio_xdmf_series_free), or NULL on failure.
 */
MIO_API mio_xdmf_series* mio_xdmf_series_create_ex(const char* path,
                                                   const mio_xdmf_series_opts* opts);

/**
 * Write the static grid every step shares. Call once, before the first
 * mio_xdmf_series_write_data(). Only the mesh's points and cells are used.
 */
MIO_API mio_status mio_xdmf_series_write_points_cells(mio_xdmf_series* series,
                                                      const mio_mesh* mesh);

/**
 * Write one time step's point_data and cell_data; the mesh's geometry is
 * ignored, and its cell blocks must match those of the static grid.
 */
MIO_API mio_status mio_xdmf_series_write_data(mio_xdmf_series* series, double time,
                                              const mio_mesh* mesh);

/**
 * Write the .xdmf and close the heavy-data container. Idempotent, and done by
 * mio_xdmf_series_free() too -- call it explicitly to see a write failure.
 */
MIO_API mio_status mio_xdmf_series_finalize(mio_xdmf_series* series);

/** One solver array for mio_xdmf_series_write_data_arrays(). Nothing is owned:
 *  `values` is read during the call and may be freed afterwards. */
typedef struct mio_named_array {
    const char* name;
    int64_t num_components; /**< values per entity; 1 for a scalar field */
    const double* values;   /**< row-major, entities * num_components doubles */
    int64_t num_values;     /**< length of `values`, validated against the grid */
} mio_named_array;

/**
 * Write one step from raw solver arrays instead of a whole mesh -- the
 * granularity a solver has once mio_xdmf_series_write_points_cells() has fixed
 * the geometry. Arrays are emitted in the order given.
 * @param point_arrays nodal arrays (may be NULL when num_point_arrays is 0).
 * @param cell_arrays  cell arrays (may be NULL when num_cell_arrays is 0).
 * @return MIO_OK, or an error if an array's length does not match the grid.
 */
MIO_API mio_status mio_xdmf_series_write_data_arrays(mio_xdmf_series* series, double time,
                                                     const mio_named_array* point_arrays,
                                                     int64_t num_point_arrays,
                                                     const mio_named_array* cell_arrays,
                                                     int64_t num_cell_arrays);

/**
 * Write the .xdmf as it currently stands, without finalizing, so a run that is
 * killed or still going leaves a readable file covering every flushed step.
 * The document is written to a temp file and renamed, so a crash during a flush
 * cannot truncate the previous one. Safe to call repeatedly; a no-op once
 * finalized.
 */
MIO_API mio_status mio_xdmf_series_flush(mio_xdmf_series* series);

/** @return the number of steps written so far, or -1 on error. */
MIO_API int64_t mio_xdmf_series_num_steps(const mio_xdmf_series* series);

/** @return 1 if the series has been finalized, 0 if not, -1 on error. */
MIO_API int32_t mio_xdmf_series_finalized(const mio_xdmf_series* series);

/** Finalize (if needed) and destroy a series handle. Safe to call with NULL. */
MIO_API void mio_xdmf_series_free(mio_xdmf_series* series);

/* --------------------------------------------------------------------------
 * Settings pipeline (v9.11.0). A settings.json document describes read ->
 * operation chain -> write (PascalCase ops/keys; see doc/pipeline.md), and
 * these run it whole -- the flat ABI carries JSON text only, no step struct.
 *
 * Both need a build with the JSON parser (-DMESHIOPLUSPLUS_WITH_JSON=ON, the
 * default when the src/cpp/third_party/json submodule is checked out);
 * otherwise they return MIO_ERR_INTERNAL and mio_last_error() names the flag.
 * -------------------------------------------------------------------------- */

/**
 * Run the settings pipeline stored in the file at `settings_path`.
 * @return MIO_OK, or an error (schema errors name the offending op/key).
 */
MIO_API mio_status mio_pipeline_run_file(const char* settings_path);

/**
 * Run the settings pipeline given as JSON text.
 * @return MIO_OK, or an error (schema errors name the offending op/key).
 */
MIO_API mio_status mio_pipeline_run_json(const char* json_text);


/** @return 1 when this build carries the JSON pipeline parser, else 0. */
MIO_API int32_t mio_pipeline_has_json(void);

/* ---------------------------------------------------------------------------
 * Sequences: multi-file / transient datasets (doc/sequences.md).
 *
 * A sequence is an ordered plan -- files, the step inside each, and a time
 * value -- built once by mio_sequence_open*, then read ONE STEP AT A TIME.
 * That is the whole point: a 500-step dataset must be traversable without
 * materializing it, so this handle stores the plan and never a mesh.
 *
 * Ordering is natural-numeric, so out_9.vtu precedes out_10.vtu.
 * -------------------------------------------------------------------------- */

/**
 * Options for opening a sequence.
 *
 * Same append-only rule as mio_read_opts / mio_write_opts: new fields may only
 * replace `reserved` capacity, never reorder or resize an existing one. Always
 * zero-initialize through mio_sequence_opts_init().
 */
typedef struct mio_sequence_opts {
    const char* format;  /**< NULL/"" resolves per file (with the sniff fallback). */
    const double* times; /**< explicit per-entry times, or NULL to derive them. */
    int64_t num_times;   /**< length of `times`; 0 when it is NULL. */
    int32_t time_from;   /**< 0 auto, 1 file, 2 filename, 3 index. */
    int32_t sort;        /**< nonzero also natural-sorts an explicit path LIST. */
    int64_t reserved[6]; /**< must be zero; room for additive growth. */
} mio_sequence_opts;

/** Zero-initialize `opts` to the documented defaults. */
MIO_API void mio_sequence_opts_init(mio_sequence_opts* opts);

/**
 * Open a sequence from a glob pattern (`*` and `?` only; the directory part is
 * taken literally). A pattern matching nothing is an error, never an empty
 * sequence.
 * @return a handle (free with mio_sequence_free), or NULL on failure.
 */
MIO_API mio_sequence* mio_sequence_open(const char* pattern);

/** mio_sequence_open with options; `opts` may be NULL for the defaults. */
MIO_API mio_sequence* mio_sequence_open_ex(const char* pattern,
                                           const mio_sequence_opts* opts);

/**
 * Open a sequence from an explicit, ordered path list. The order is the
 * caller's and is kept unless `opts->sort` asks otherwise.
 * @return a handle (free with mio_sequence_free), or NULL on failure.
 */
MIO_API mio_sequence* mio_sequence_open_list(const char* const* paths, int64_t num_paths,
                                             const mio_sequence_opts* opts);

/** Number of steps in the sequence, or -1 on error. */
MIO_API int64_t mio_sequence_count(const mio_sequence* seq);

/**
 * Copy entry `index`'s file path into `buf`.
 * @return the untruncated length, or -1 on error (string-getter protocol).
 */
MIO_API int64_t mio_sequence_path(const mio_sequence* seq, int64_t index, char* buf,
                                  int64_t buflen);

/** Entry `index`'s step index WITHIN its file (0 for a single-step file), or -1. */
MIO_API int64_t mio_sequence_step(const mio_sequence* seq, int64_t index);

/** Write entry `index`'s time value to `out_time`. */
MIO_API mio_status mio_sequence_time(const mio_sequence* seq, int64_t index, double* out_time);

/**
 * Where entry `index`'s time came from: 0 explicit, 1 file, 2 filename,
 * 3 index (the fallback). -1 on error.
 *
 * Reported rather than left to be guessed: "this time is 0.25 because the file
 * said so" and "this time is 3 because nothing said anything" are different
 * facts, and a caller plotting against it needs to know which it has.
 */
MIO_API int32_t mio_sequence_time_source(const mio_sequence* seq, int64_t index);

/**
 * Read entry `index`.
 *
 * The returned mesh is OWNED -- free it with mio_mesh_free(). Deliberately not
 * the borrow shape mio_split_result_mesh() uses: a borrow would force this
 * handle to cache every mesh it handed out, which is exactly the accumulation
 * the streaming guarantee forbids. The C ABI expresses the rule.
 *
 * @return a new owning mesh handle, or NULL on failure.
 */
MIO_API mio_mesh* mio_sequence_read(mio_sequence* seq, int64_t index);

/** Free a sequence handle. The meshes it produced are unaffected (it owns none). */
MIO_API void mio_sequence_free(mio_sequence* seq);

/**
 * Fan-in: write every step of `seq` into one multi-step file at `out_path`.
 *
 * Streams -- one mesh alive at a time. A format that cannot hold a series
 * fails naming itself and pointing at '{step}', never a silent truncation to
 * the first step.
 */
MIO_API mio_status mio_sequence_to_timeseries(const mio_sequence* seq, const char* out_path,
                                              const char* out_format);

/**
 * mio_sequence_to_timeseries with write options (currently only `encoding` is
 * meaningful here: MIO_ENCODING_ASCII selects the XDMF "XML" data format --
 * everything inline, no HDF5 needed -- while the default/MIO_ENCODING_BINARY
 * selects "HDF", which needs an HDF5-enabled build). `opts` may be NULL for
 * mio_sequence_to_timeseries's defaults. `codec`/`float_format` are accepted
 * but XDMF honours neither, so non-default values there fail by name exactly
 * as mio_write_ex's do.
 */
MIO_API mio_status mio_sequence_to_timeseries_ex(const mio_sequence* seq, const char* out_path,
                                                 const char* out_format,
                                                 const mio_write_opts* opts);

/**
 * Fan-out: write each step of the multi-step file `in_path` to `out_pattern`,
 * which must contain '{step}' or '{index}'. Streams likewise.
 */
MIO_API mio_status mio_timeseries_to_sequence(const char* in_path, const char* in_format,
                                              const char* out_pattern, const char* out_format);

/**
 * Run a sequence settings document (the pipeline schema plus Mode /
 * Input.Pattern / Input.Paths / Input.Times / Input.TimeFrom). A document
 * using none of those behaves exactly as mio_pipeline_run_file would.
 *
 * Needs the JSON parser, like the pipeline entry points above.
 */
MIO_API mio_status mio_sequence_pipeline_run_file(const char* settings_path);

/** mio_sequence_pipeline_run_file over JSON text. */
MIO_API mio_status mio_sequence_pipeline_run_json(const char* json_text);

#ifdef __cplusplus
}
#endif

#endif /* MESHIOPLUSPLUS_H */
