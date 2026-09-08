//  ██████   ██████ ██████████  █████████  █████   █████ █████    ███████
// ░░██████ ██████ ░░███░░░░░█ ███░░░░░███░░███   ░░███ ░░███   ███░░░░░███      ███         ███
//  ░███░█████░███  ░███  █ ░ ░███    ░░░  ░███    ░███  ░███  ███     ░░███    ░███        ░███
//  ░███░░███ ░███  ░██████   ░░█████████  ░███████████  ░███ ░███      ░███ ███████████ ███████████
//  ░███ ░░░  ░███  ░███░░█    ░░░░░░░░███ ░███░░░░░███  ░███ ░███      ░███░░░░░███░░░ ░░░░░███░░░
//  ░███      ░███  ░███ ░   █ ███    ░███ ░███    ░███  ░███ ░░███     ███     ░███        ░███
//  █████     █████ ██████████░░█████████  █████   █████ █████ ░░░███████░      ░░░         ░░░
// ░░░░░     ░░░░░ ░░░░░░░░░░  ░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░    ░░░░░░░
//
//
//  License:         MIT License
//                   meshio++ default license: LICENSE
//
//  Main authors:    Vicente Mataix Ferrandiz
//
//

/**
 * @file c_api.cpp
 * @brief Implementation of the meshio++ C API (bindings/c/include/
 *        meshioplusplus/meshioplusplus.h), compiled into the installable
 *        `libmeshioplusplus` shared library.
 *
 * Like the WASM binding (and unlike `bindings/_core.cpp`), this is a flat,
 * whole-mesh interface built exclusively on the uniform mesh API
 * (mesh_api.hpp) -- it therefore compiles unchanged under every mesh backend
 * (MESHIO/NATIVE/KRATOS) -- and dispatches formats through the shared
 * registry (registry.hpp). The three ABI rules the implementation enforces:
 *
 *  - No C++ exception ever crosses `extern "C"`: every public function body
 *    runs inside guarded()/guarded_ptr(), which map ReadError/WriteError/
 *    anything else to a `mio_status` plus a thread-local message retrievable
 *    via mio_last_error().
 *  - Setters copy caller memory into owning NDArrays (validated first);
 *    getters hand out pointers into mesh-owned storage, which every backend
 *    keeps stable until the next mutating call (KRATOS serves accessors from
 *    its persistent staging mesh).
 *  - Strings cross via the caller-buffer/required-length protocol -- never a
 *    `c_str()` of a possibly-temporary (CellView::Type()'s return category
 *    differs per backend).
 */

// System includes
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Project includes
#include "meshioplusplus/meshioplusplus.h"

#include "meshioplusplus/cell_type.hpp"
#include "meshioplusplus/exceptions.hpp"
#include "meshioplusplus/formats/xdmf_time_series.hpp"
#include "meshioplusplus/ndarray.hpp"
#include "meshioplusplus/operations/clean.hpp"
#include "meshioplusplus/operations/convert_cells.hpp"
#include "meshioplusplus/operations/curvature.hpp"
#include "meshioplusplus/operations/crop.hpp"
#include "meshioplusplus/operations/data_average.hpp"
#include "meshioplusplus/operations/data_calc.hpp"
#include "meshioplusplus/operations/data_common.hpp"
#include "meshioplusplus/operations/data_condition.hpp"
#include "meshioplusplus/operations/data_info.hpp"
#include "meshioplusplus/operations/data_integrate.hpp"
#include "meshioplusplus/operations/data_manage.hpp"
#include "meshioplusplus/operations/decimate.hpp"
#include "meshioplusplus/operations/decimate_volume.hpp"
#include "meshioplusplus/operations/conservative_interpolate.hpp"
#include "meshioplusplus/operations/diff.hpp"
#include "meshioplusplus/operations/interpolate.hpp"
#include "meshioplusplus/operations/merge.hpp"
#include "meshioplusplus/operations/partition.hpp"
#include "meshioplusplus/operations/pipeline.hpp"
#include "meshioplusplus/operations/sequence.hpp"
#include "meshioplusplus/operations/quality.hpp"
#include "meshioplusplus/operations/refine.hpp"
#include "meshioplusplus/operations/reorder.hpp"
#include "meshioplusplus/operations/isosurface.hpp"
#include "meshioplusplus/operations/error.hpp"
#include "meshioplusplus/operations/gradient.hpp"
#include "meshioplusplus/operations/hessian.hpp"
#include "meshioplusplus/operations/slice.hpp"
#include "meshioplusplus/operations/smooth.hpp"
#include "meshioplusplus/operations/sniff.hpp"
#include "meshioplusplus/operations/split.hpp"
#include "meshioplusplus/operations/sdf.hpp"
#include "meshioplusplus/operations/stats.hpp"
#include "meshioplusplus/operations/agglomerate.hpp"
#include "meshioplusplus/operations/subdivide.hpp"
#include "meshioplusplus/operations/voxelize.hpp"
#include "meshioplusplus/operations/surface.hpp"
#include "meshioplusplus/operations/transform.hpp"
#include "meshioplusplus/operations/undo_green.hpp"
#include "meshioplusplus/operations/remesh.hpp"
#include "meshioplusplus/operations/remesh_volume.hpp"
#include "meshioplusplus/operations/optimize_volume.hpp"
#include "meshioplusplus/read_options.hpp"
#include "meshioplusplus/registry.hpp"
#include "meshioplusplus/detail/provenance.hpp"
#include "meshioplusplus/version.hpp"
#include "meshioplusplus/write_options.hpp"
#include "meshioplusplus/skin.hpp"

struct mio_mesh {
    meshioplusplus::Mesh mMesh;
};

struct mio_regions {
    // A snapshot, not a borrow: the KRATOS backend serves Region(i) from lazily
    // rebuilt staging, so a pointer into the mesh would not stay valid.
    std::vector<meshioplusplus::Region> mRegions;
};

struct mio_poly_conn {
    // Likewise a snapshot rather than a rule-3 borrow, and for a stronger
    // reason: the MESHIO backend stores ragged blocks as nested vectors, so no
    // offsets array exists inside the mesh to point at, and only NativeMesh
    // keeps real CSR -- which c_api.cpp cannot reach, since it compiles under
    // all three backends. Built from the uniform API's RowSize/Row/NumFaces/
    // Face exactly as bindings/wasm/js_bindings.cpp does.
    bool mIsPolyhedron = false;
    std::int64_t mNumCells = 0;
    std::vector<std::int64_t> mNodes;
    std::vector<std::int64_t> mFaceOffsets;
    std::vector<std::int64_t> mCellOffsets;  // empty for a 1-level block
};

struct mio_reorder_result {
    mio_mesh mMesh;  // owns the renumbered mesh; borrowed via mio_reorder_result_mesh
    meshioplusplus::NDArray mNodePerm;
    std::vector<meshioplusplus::NDArray> mCellPerms;
};

struct mio_diff_result {
    meshioplusplus::DiffReport mReport;
};

struct mio_xdmf_series {
    // Held by value: the writer is already move-only and owns its open
    // heavy-data container, so the handle is just the C-side name for it.
    meshioplusplus::XdmfTimeSeriesWriter mWriter;
};

/// The PLAN for a sequence -- paths, per-file step indices and times. It owns
/// no mesh, deliberately: caching what it handed out is exactly the
/// accumulation the streaming guarantee forbids, which is why
/// mio_sequence_read returns an OWNED mesh rather than a borrow.
struct mio_sequence {
    std::vector<meshioplusplus::SequenceEntry> mEntries;
    std::string mFormat;
    meshioplusplus::ReadOptions mOptions;
};

struct mio_split_result {
    std::vector<mio_mesh> mMeshes;   // owns each piece; borrowed via mio_split_result_mesh
    std::vector<std::string> mKeys;  // per-piece key (type / component / tag value)
};

struct mio_convert_cells_result {
    mio_mesh mMesh;  // owns the converted mesh; borrowed via _result_mesh
    meshioplusplus::NDArray mPointMap;
    std::vector<meshioplusplus::NDArray> mCellMaps;
};

struct mio_subdivide_result {
    mio_mesh mMesh;  // owns the subdivided mesh; borrowed via _result_mesh
    std::vector<meshioplusplus::NDArray> mCellMaps;
};

struct mio_agglomerate_result {
    mio_mesh mMesh;                    // owns the coarsened mesh; borrowed via _result_mesh
    meshioplusplus::NDArray mCellMap;  // flat, unlike mio_subdivide_result's per-block vector
};

struct mio_refine_result {
    mio_mesh mMesh;  // owns the refined mesh; borrowed via _result_mesh
    meshioplusplus::NDArray mPointMap;
    std::vector<meshioplusplus::NDArray> mCellMaps;
};

struct mio_decimate_result {
    mio_mesh mMesh;  // owns the decimated mesh; borrowed via _result_mesh
    meshioplusplus::NDArray mPointMap;
    std::vector<meshioplusplus::NDArray> mCellMaps;
    int64_t mFacesRemoved = 0;
    int64_t mPointsRemoved = 0;
    int64_t mCollapsesRejected = 0;
    double mMaxErrorApplied = 0.0;
};

struct mio_decimate_volume_result {
    mio_mesh mMesh;  // owns the decimated mesh; borrowed via _result_mesh
    meshioplusplus::NDArray mPointMap;
    std::vector<meshioplusplus::NDArray> mCellMaps;
    int64_t mTetsRemoved = 0;
    int64_t mPointsRemoved = 0;
    int64_t mCollapsesRejected = 0;
    double mMaxErrorApplied = 0.0;
};

struct mio_partition_result {
    struct Piece {
        mio_mesh mMesh;  // owns the piece; borrowed via _result_mesh
        meshioplusplus::NDArray mPointMap;
        std::vector<meshioplusplus::NDArray> mCellMaps;
    };
    std::vector<Piece> mPieces;  // exactly nparts entries, part id == index
};

struct mio_data_info {
    meshioplusplus::DataInfoReport mReport;
};

struct mio_data_integrate {
    meshioplusplus::DataIntegrateReport mReport;
};

/// Opaque handle behind mio_read_metadata_*; modelled on mio_data_info above,
/// since the summary is variable-length (cell blocks, name lists).
struct mio_read_metadata {
    meshioplusplus::MeshMetadata mMeta;
};

namespace {

using meshioplusplus::DType;
using meshioplusplus::Mesh;
using meshioplusplus::NDArray;

// The C header's MIO_CELL_TYPES list must mirror the C++ enum exactly; any
// added/removed/reordered entry on either side fails to compile right here.
#define MIO_CELL_TYPE_CHECK(Name)                                                              \
    static_assert(                                                                             \
        static_cast<int>(meshioplusplus::CellType::Name) == static_cast<int>(MIO_CELL_##Name), \
        "meshioplusplus.h cell-type list drifted from cell_type.hpp");
MIO_CELL_TYPES(MIO_CELL_TYPE_CHECK)
#undef MIO_CELL_TYPE_CHECK
static_assert(static_cast<int>(meshioplusplus::CellType::Custom) ==
                  static_cast<int>(MIO_CELL_Custom),
              "meshioplusplus.h cell-type list drifted from cell_type.hpp");

#ifndef MIO_VERSION_STRING
#define MIO_VERSION_STRING "unknown"
#endif

thread_local std::string g_last_error;

mio_status fail(mio_status code, std::string msg) {
    g_last_error = std::move(msg);
    return code;
}

// Every fallible extern "C" body runs inside one of these two: no exception
// crosses the ABI, and the thread-local message is always set on failure.
template <class F>
mio_status guarded(F&& f) {
    try {
        return f();
    } catch (const meshioplusplus::ReadError& e) {
        return fail(MIO_ERR_READ, e.what());
    } catch (const meshioplusplus::WriteError& e) {
        return fail(MIO_ERR_WRITE, e.what());
    } catch (const std::bad_alloc&) {
        return fail(MIO_ERR_INTERNAL, "meshio++: out of memory");
    } catch (const std::exception& e) {
        return fail(MIO_ERR_INTERNAL, e.what());
    } catch (...) {
        return fail(MIO_ERR_INTERNAL, "meshio++: unknown error");
    }
}

// Variant for functions returning a pointer/count instead of a status.
template <class T, class F>
T guarded_ptr(T on_error, F&& f) {
    T result = on_error;
    guarded([&]() -> mio_status {
        result = f();
        return MIO_OK;
    });
    return result;
}

bool to_dtype(mio_dtype in, DType& rOut) {
    switch (in) {
        case MIO_FLOAT32:
            rOut = DType::Float32;
            return true;
        case MIO_FLOAT64:
            rOut = DType::Float64;
            return true;
        case MIO_INT8:
            rOut = DType::Int8;
            return true;
        case MIO_INT16:
            rOut = DType::Int16;
            return true;
        case MIO_INT32:
            rOut = DType::Int32;
            return true;
        case MIO_INT64:
            rOut = DType::Int64;
            return true;
        case MIO_UINT8:
            rOut = DType::UInt8;
            return true;
        case MIO_UINT16:
            rOut = DType::UInt16;
            return true;
        case MIO_UINT32:
            rOut = DType::UInt32;
            return true;
        case MIO_UINT64:
            rOut = DType::UInt64;
            return true;
    }
    return false;
}

mio_dtype from_dtype(DType in) {
    switch (in) {
        case DType::Float32:
            return MIO_FLOAT32;
        case DType::Float64:
            return MIO_FLOAT64;
        case DType::Int8:
            return MIO_INT8;
        case DType::Int16:
            return MIO_INT16;
        case DType::Int32:
            return MIO_INT32;
        case DType::Int64:
            return MIO_INT64;
        case DType::UInt8:
            return MIO_UINT8;
        case DType::UInt16:
            return MIO_UINT16;
        case DType::UInt32:
            return MIO_UINT32;
        case DType::UInt64:
            return MIO_UINT64;
    }
    return MIO_FLOAT64;  // unreachable
}

// Validate an (ndim, shape) pair from the caller and return it as the vector
// NDArray wants; MIO_MAX_NDIM bounds the rank in both directions of the API.
bool to_shape(int32_t ndim, const int64_t* pShape, std::vector<std::size_t>& rOut) {
    if (ndim < 1 || ndim > MIO_MAX_NDIM || !pShape)
        return false;
    rOut.assign(static_cast<std::size_t>(ndim), 0);
    for (int32_t i = 0; i < ndim; ++i) {
        if (pShape[i] < 0)
            return false;
        rOut[static_cast<std::size_t>(i)] = static_cast<std::size_t>(pShape[i]);
    }
    return true;
}

// Owning NDArray copied from a caller buffer (setter rule: setters copy).
NDArray copy_in(DType dt, std::vector<std::size_t> shape, const void* pData) {
    NDArray out = NDArray::Uninit(dt, std::move(shape));
    if (out.Nbytes() > 0)
        std::memcpy(out.Data(), pData, out.Nbytes());
    return out;
}

// String-getter protocol (header rule 5): copy what fits, NUL-terminate,
// return the untruncated length.
int64_t copy_string(const std::string& rStr, char* pBuf, int64_t buflen) {
    if (pBuf && buflen > 0) {
        std::size_t n = std::min(rStr.size(), static_cast<std::size_t>(buflen - 1));
        std::memcpy(pBuf, rStr.data(), n);
        pBuf[n] = '\0';
    }
    return static_cast<int64_t>(rStr.size());
}

std::string format_or_empty(const char* pFormat) {
    return pFormat ? std::string(pFormat) : std::string();
}

std::string unknown_format_message(const std::string& rFormat, bool for_write) {
    std::string msg = for_write
                          ? "meshio++: unknown, read-only, or unsupported format '" + rFormat + "'"
                          : "meshio++: unknown or unsupported format '" + rFormat + "'";
    if (const char* dep = meshioplusplus::registry_compiled_out(rFormat))
        msg += " (this build has no " + std::string(dep) + " support)";
    return msg;
}

// Fill the (data, dtype, ndim, shape) out-params from a mesh-owned NDArray;
// any out-param may be NULL.
mio_status array_out(const NDArray& rArr, const void** ppData, mio_dtype* pDtype, int32_t* pNdim,
                     int64_t* pShape) {
    const auto& shape = rArr.Shape();
    if (shape.size() > MIO_MAX_NDIM)
        return fail(MIO_ERR_INTERNAL, "meshio++: array rank exceeds MIO_MAX_NDIM");
    if (ppData)
        *ppData = rArr.Data();
    if (pDtype)
        *pDtype = from_dtype(rArr.Dtype());
    if (pNdim)
        *pNdim = static_cast<int32_t>(shape.size());
    if (pShape) {
        for (std::size_t i = 0; i < shape.size(); ++i)
            pShape[i] = static_cast<int64_t>(shape[i]);
    }
    return MIO_OK;
}

// Shared body of the three named-data setters (they differ only in the
// leading-dimension check and the uniform-API call).
template <class AddFn>
mio_status add_named_array(mio_mesh* pMesh, const char* pName, mio_dtype dtype, int32_t ndim,
                           const int64_t* pShape, const void* pData, const char* pWhat,
                           std::int64_t required_dim0, AddFn&& add) {
    DType dt;
    std::vector<std::size_t> shape;
    if (!pMesh || !pName)
        return fail(MIO_ERR_INVALID_ARG,
                    std::string("meshio++: bad ") + pWhat + " argument (NULL mesh/name)");
    if (!to_dtype(dtype, dt))
        return fail(MIO_ERR_INVALID_ARG, std::string("meshio++: bad ") + pWhat + " dtype");
    if (!to_shape(ndim, pShape, shape))
        return fail(MIO_ERR_INVALID_ARG, std::string("meshio++: bad ") + pWhat +
                                             " shape (rank 1.." + std::to_string(MIO_MAX_NDIM) +
                                             ", non-negative extents)");
    if (required_dim0 >= 0 && shape[0] != static_cast<std::size_t>(required_dim0))
        return fail(MIO_ERR_INVALID_ARG, std::string("meshio++: ") + pWhat + " '" + pName +
                                             "' shape[0] is " + std::to_string(shape[0]) +
                                             ", expected " + std::to_string(required_dim0));
    NDArray arr = NDArray::Uninit(dt, std::move(shape));
    if (arr.Nbytes() > 0) {
        if (!pData)
            return fail(MIO_ERR_INVALID_ARG,
                        std::string("meshio++: bad ") + pWhat + " argument (NULL data)");
        std::memcpy(arr.Data(), pData, arr.Nbytes());
    }
    add(std::move(arr));
    return MIO_OK;
}

bool block_in_range(const mio_mesh* pMesh, int64_t block) {
    return pMesh && block >= 0 && static_cast<std::size_t>(block) < pMesh->mMesh.NumCellBlocks();
}

}  // namespace

extern "C" {

/* ------------------------------------------------------------------ */
/* Version / build introspection                                       */
/* ------------------------------------------------------------------ */

// The C header repeats the release version as preprocessor macros so consumers
// can feature-detect; these pin them to the C++ ones, so the two copies cannot
// drift. (CMake separately hard-fails if either disagrees with the project
// version.) Same technique as the mio_cell_type / CellType static_asserts.
static_assert(MIO_VERSION_MAJOR == MESHIOPLUSPLUS_VERSION_MAJOR &&
                  MIO_VERSION_MINOR == MESHIOPLUSPLUS_VERSION_MINOR &&
                  MIO_VERSION_PATCH == MESHIOPLUSPLUS_VERSION_PATCH,
              "MIO_VERSION_* drifted from MESHIOPLUSPLUS_VERSION_*");

const char* mio_version(void) {
    return MIO_VERSION_STRING;
}

const char* mio_mesh_backend(void) {
    return meshioplusplus::mesh_backend_name();
}

int mio_format_readable(const char* format) {
    return guarded_ptr(0, [&]() -> int {
        return format && meshioplusplus::registry_readers().count(format) ? 1 : 0;
    });
}

int mio_format_writable(const char* format) {
    return guarded_ptr(0, [&]() -> int {
        return format && meshioplusplus::registry_writers().count(format) ? 1 : 0;
    });
}

const char* mio_last_error(void) {
    return g_last_error.c_str();
}

/* ------------------------------------------------------------------ */
/* Cell-type metadata                                                  */
/* ------------------------------------------------------------------ */

const char* mio_cell_type_name(mio_cell_type t) {
    if (t < 0 || t > MIO_CELL_Custom)
        return "";
    return meshioplusplus::cell_type_name(static_cast<meshioplusplus::CellType>(t)).c_str();
}

mio_cell_type mio_cell_type_from_name(const char* name) {
    if (!name)
        return MIO_CELL_Custom;
    return static_cast<mio_cell_type>(meshioplusplus::cell_type_from_name(name));
}

int mio_cell_type_num_nodes(mio_cell_type t) {
    if (t < 0 || t > MIO_CELL_Custom)
        return -1;
    return meshioplusplus::cell_type_num_nodes(static_cast<meshioplusplus::CellType>(t));
}

int mio_cell_type_dimension(mio_cell_type t) {
    if (t < 0 || t > MIO_CELL_Custom)
        return -1;
    return meshioplusplus::cell_type_dimension(static_cast<meshioplusplus::CellType>(t));
}

/* ------------------------------------------------------------------ */
/* Lifecycle & file I/O                                                */
/* ------------------------------------------------------------------ */

mio_mesh* mio_mesh_create(void) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), []() { return new mio_mesh(); });
}

void mio_mesh_free(mio_mesh* mesh) {
    delete mesh;
}

mio_mesh* mio_read(const char* path, const char* format) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!path)
            throw meshioplusplus::ReadError("meshio++: path is NULL");
        std::string fmt;
        try {
            fmt = meshioplusplus::resolve_format(path, format_or_empty(format));
        } catch (const meshioplusplus::ReadError&) {
            // Extension gave nothing: fall back to a conservative content sniff.
            fmt = meshioplusplus::sniff_format(path);
            if (fmt.empty())
                throw;
        }
        auto it = meshioplusplus::registry_readers().find(fmt);
        if (it == meshioplusplus::registry_readers().end())
            throw meshioplusplus::ReadError(unknown_format_message(fmt, /*for_write=*/false));
        return new mio_mesh{it->second(path)};
    });
}

namespace {

/** @brief Resolve a read format, falling back to a content sniff. */
std::string capi_resolve_read_format(const char* pPath, const char* pFormat) {
    try {
        return meshioplusplus::resolve_format(pPath, format_or_empty(pFormat));
    } catch (const meshioplusplus::ReadError&) {
        // Extension gave nothing: fall back to a conservative content sniff.
        std::string sniffed = meshioplusplus::sniff_format(pPath);
        if (sniffed.empty())
            throw;
        return sniffed;
    }
}

/** @brief `mio_read_opts` -> `ReadOptions`; NULL means the defaults. */
meshioplusplus::ReadOptions capi_read_options(const mio_read_opts* pOpts) {
    meshioplusplus::ReadOptions out;
    if (!pOpts)
        return out;
    out.mPointsOnly = pOpts->points_only != 0;
    out.mMetadataOnly = pOpts->metadata_only != 0;
    // A NULL array pointer means "every array"; a non-NULL pointer with count 0
    // means "no arrays". Collapsing the two here would silently turn an
    // explicit "none" into "everything".
    if (pOpts->arrays) {
        std::vector<std::string> names;
        names.reserve(static_cast<std::size_t>(pOpts->num_arrays));
        for (std::int64_t i = 0; i < pOpts->num_arrays; ++i)
            names.emplace_back(pOpts->arrays[i] ? pOpts->arrays[i] : "");
        out.mDataArrays = std::move(names);  // setters copy, per the ABI contract
    }
    switch (pOpts->mmap_mode) {
        case 1:
            out.mMmap = meshioplusplus::MmapMode::On;
            break;
        case 2:
            out.mMmap = meshioplusplus::MmapMode::Off;
            break;
        default:
            out.mMmap = meshioplusplus::MmapMode::Auto;
            break;
    }
    // int64 on the ABI (the reserved slots are int64) but int in the core: a
    // step index beyond int range cannot name a real step, so reject it here
    // rather than let the narrowing wrap into a valid-looking one.
    if (pOpts->time_step > INT_MAX || pOpts->time_step < INT_MIN)
        throw meshioplusplus::ReadError("meshio++: time_step is out of range");
    out.mTimeStep = static_cast<int>(pOpts->time_step);
    out.mLenient = pOpts->lenient != 0;
    return out;
}

/** @brief The name list for a `mio_data_location`. */
const std::vector<std::string>& capi_metadata_names(const meshioplusplus::MeshMetadata& rMeta,
                                                    int location) {
    switch (location) {
        case MIO_DATA_POINT:
            return rMeta.mPointDataNames;
        case MIO_DATA_CELL:
            return rMeta.mCellDataNames;
        case MIO_DATA_FIELD:
            return rMeta.mFieldDataNames;
        default:
            throw meshioplusplus::ReadError("meshio++: unknown data location");
    }
}

}  // namespace

void mio_read_opts_init(mio_read_opts* opts) {
    if (!opts)
        return;
    *opts = mio_read_opts{};  // value-initialized: all zero == read everything
}

mio_mesh* mio_read_ex(const char* path, const char* format, const mio_read_opts* opts) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!path)
            throw meshioplusplus::ReadError("meshio++: path is NULL");
        const std::string fmt = capi_resolve_read_format(path, format);
        if (!meshioplusplus::registry_readers().count(fmt) &&
            !meshioplusplus::registry_reader_supports_options(fmt))
            throw meshioplusplus::ReadError(unknown_format_message(fmt, /*for_write=*/false));
        return new mio_mesh{meshioplusplus::registry_read(path, fmt, capi_read_options(opts))};
    });
}

mio_read_metadata* mio_read_metadata_create(const char* path, const char* format) {
    return guarded_ptr(static_cast<mio_read_metadata*>(nullptr), [&]() -> mio_read_metadata* {
        if (!path)
            throw meshioplusplus::ReadError("meshio++: path is NULL");
        const std::string fmt = capi_resolve_read_format(path, format);
        return new mio_read_metadata{
            meshioplusplus::registry_read_metadata(path, fmt, meshioplusplus::ReadOptions{})};
    });
}

int64_t mio_read_metadata_num_points(const mio_read_metadata* meta) {
    return guarded_ptr(std::int64_t(-1), [&]() -> std::int64_t {
        if (!meta)
            throw meshioplusplus::ReadError("meshio++: metadata handle is NULL");
        return static_cast<std::int64_t>(meta->mMeta.mNumPoints);
    });
}

int64_t mio_read_metadata_point_dim(const mio_read_metadata* meta) {
    return guarded_ptr(std::int64_t(-1), [&]() -> std::int64_t {
        if (!meta)
            throw meshioplusplus::ReadError("meshio++: metadata handle is NULL");
        return static_cast<std::int64_t>(meta->mMeta.mPointDim);
    });
}

int64_t mio_read_metadata_num_cells(const mio_read_metadata* meta) {
    return guarded_ptr(std::int64_t(-1), [&]() -> std::int64_t {
        if (!meta)
            throw meshioplusplus::ReadError("meshio++: metadata handle is NULL");
        return static_cast<std::int64_t>(meta->mMeta.NumCells());
    });
}

int64_t mio_read_metadata_num_cell_blocks(const mio_read_metadata* meta) {
    return guarded_ptr(std::int64_t(-1), [&]() -> std::int64_t {
        if (!meta)
            throw meshioplusplus::ReadError("meshio++: metadata handle is NULL");
        return static_cast<std::int64_t>(meta->mMeta.mCellBlocks.size());
    });
}

mio_status mio_read_metadata_cell_block(const mio_read_metadata* meta, int64_t index,
                                        int64_t* num_cells, int64_t* num_nodes_per_cell,
                                        int* is_ragged) {
    return guarded([&]() -> mio_status {
        if (!meta)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: metadata handle is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= meta->mMeta.mCellBlocks.size())
            return fail(MIO_ERR_INVALID_ARG, "meshio++: cell block index out of range");
        const meshioplusplus::CellBlockInfo& block =
            meta->mMeta.mCellBlocks[static_cast<std::size_t>(index)];
        if (num_cells)
            *num_cells = static_cast<std::int64_t>(block.mNumCells);
        if (num_nodes_per_cell)
            *num_nodes_per_cell = static_cast<std::int64_t>(block.mNodesPerCell);
        if (is_ragged)
            *is_ragged = block.mRagged ? 1 : 0;
        return MIO_OK;
    });
}

int64_t mio_read_metadata_cell_block_type(const mio_read_metadata* meta, int64_t index, char* buf,
                                          int64_t buflen) {
    return guarded_ptr(std::int64_t(-1), [&]() -> std::int64_t {
        if (!meta)
            throw meshioplusplus::ReadError("meshio++: metadata handle is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= meta->mMeta.mCellBlocks.size())
            throw meshioplusplus::ReadError("meshio++: cell block index out of range");
        return copy_string(meta->mMeta.mCellBlocks[static_cast<std::size_t>(index)].mType, buf,
                           buflen);
    });
}

int64_t mio_read_metadata_num_time_values(const mio_read_metadata* meta) {
    return guarded_ptr(std::int64_t(-1), [&]() -> std::int64_t {
        if (!meta)
            throw meshioplusplus::ReadError("meshio++: metadata handle is NULL");
        return static_cast<std::int64_t>(meta->mMeta.mTimeValues.size());
    });
}

int64_t mio_read_metadata_time_values(const mio_read_metadata* meta, double* out, int64_t count) {
    return guarded_ptr(std::int64_t(-1), [&]() -> std::int64_t {
        if (!meta)
            throw meshioplusplus::ReadError("meshio++: metadata handle is NULL");
        const std::vector<double>& values = meta->mMeta.mTimeValues;
        const std::int64_t n =
            std::min<std::int64_t>(count < 0 ? 0 : count, static_cast<std::int64_t>(values.size()));
        if (out)
            for (std::int64_t i = 0; i < n; ++i)
                out[i] = values[static_cast<std::size_t>(i)];
        return n;
    });
}

int64_t mio_read_metadata_num_regions(const mio_read_metadata* meta) {
    return guarded_ptr(std::int64_t(-1), [&]() -> std::int64_t {
        if (!meta)
            throw meshioplusplus::ReadError("meshio++: metadata handle is NULL");
        return static_cast<std::int64_t>(meta->mMeta.mRegions.size());
    });
}

int64_t mio_read_metadata_region_name(const mio_read_metadata* meta, int64_t index, char* buf,
                                      int64_t buflen) {
    return guarded_ptr(std::int64_t(-1), [&]() -> std::int64_t {
        if (!meta)
            throw meshioplusplus::ReadError("meshio++: metadata handle is NULL");
        const std::vector<meshioplusplus::RegionSummary>& regions = meta->mMeta.mRegions;
        if (index < 0 || static_cast<std::size_t>(index) >= regions.size())
            throw meshioplusplus::ReadError("meshio++: region index " + std::to_string(index) +
                                            " out of range");
        return copy_string(regions[static_cast<std::size_t>(index)].mName, buf, buflen);
    });
}

mio_status mio_read_metadata_region_info(const mio_read_metadata* meta, int64_t index,
                                         mio_region_info* out) {
    return guarded([&]() -> mio_status {
        if (!meta || !out)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: metadata/out is NULL");
        const std::vector<meshioplusplus::RegionSummary>& regions = meta->mMeta.mRegions;
        if (index < 0 || static_cast<std::size_t>(index) >= regions.size())
            return fail(MIO_ERR_INVALID_ARG,
                        "meshio++: region index " + std::to_string(index) + " out of range");
        const meshioplusplus::RegionSummary& r = regions[static_cast<std::size_t>(index)];
        out->kind = static_cast<int32_t>(r.mKind);
        out->dim = static_cast<int32_t>(r.mDim);
        out->tag = r.mTag;
        out->num_entries = static_cast<int64_t>(r.mNumEntries);
        out->stride = 0;  // no entries are carried by a summary
        return MIO_OK;
    });
}

int64_t mio_read_metadata_num_names(const mio_read_metadata* meta, int location) {
    return guarded_ptr(std::int64_t(-1), [&]() -> std::int64_t {
        if (!meta)
            throw meshioplusplus::ReadError("meshio++: metadata handle is NULL");
        return static_cast<std::int64_t>(capi_metadata_names(meta->mMeta, location).size());
    });
}

int64_t mio_read_metadata_name(const mio_read_metadata* meta, int location, int64_t index,
                               char* buf, int64_t buflen) {
    return guarded_ptr(std::int64_t(-1), [&]() -> std::int64_t {
        if (!meta)
            throw meshioplusplus::ReadError("meshio++: metadata handle is NULL");
        const std::vector<std::string>& names = capi_metadata_names(meta->mMeta, location);
        if (index < 0 || static_cast<std::size_t>(index) >= names.size())
            throw meshioplusplus::ReadError("meshio++: name index out of range");
        return copy_string(names[static_cast<std::size_t>(index)], buf, buflen);
    });
}

mio_status mio_read_metadata_bbox(const mio_read_metadata* meta, double* min, double* max) {
    return guarded([&]() -> mio_status {
        if (!meta)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: metadata handle is NULL");
        if (!meta->mMeta.mHasBBox)
            return fail(MIO_ERR_NOT_FOUND,
                        "meshio++: no bounding box in this summary (a native metadata path "
                        "does not decode the point coordinates)");
        for (int d = 0; d < 3; ++d) {
            if (min)
                min[d] = meta->mMeta.mBBoxMin[d];
            if (max)
                max[d] = meta->mMeta.mBBoxMax[d];
        }
        return MIO_OK;
    });
}

int mio_read_metadata_fell_back(const mio_read_metadata* meta) {
    return guarded_ptr(-1, [&]() -> int {
        if (!meta)
            throw meshioplusplus::ReadError("meshio++: metadata handle is NULL");
        return meta->mMeta.mFellBackToFullRead ? 1 : 0;
    });
}

int64_t mio_read_metadata_num_provenance_lines(const mio_read_metadata* meta) {
    return guarded_ptr(std::int64_t(-1), [&]() -> std::int64_t {
        if (!meta)
            throw meshioplusplus::ReadError("meshio++: metadata handle is NULL");
        return static_cast<std::int64_t>(meta->mMeta.mProvenance.size());
    });
}

int64_t mio_read_metadata_provenance_line(const mio_read_metadata* meta, int64_t index, char* out,
                                          int64_t cap) {
    return guarded_ptr(std::int64_t(-1), [&]() -> std::int64_t {
        if (!meta)
            throw meshioplusplus::ReadError("meshio++: metadata handle is NULL");
        const std::vector<std::string>& lines = meta->mMeta.mProvenance;
        if (index < 0 || static_cast<std::size_t>(index) >= lines.size())
            throw meshioplusplus::ReadError("meshio++: provenance line index " +
                                            std::to_string(index) + " out of range");
        return copy_string(lines[static_cast<std::size_t>(index)], out, cap);
    });
}

int mio_read_metadata_provenance_recognised(const mio_read_metadata* meta) {
    return guarded_ptr(-1, [&]() -> int {
        if (!meta)
            throw meshioplusplus::ReadError("meshio++: metadata handle is NULL");
        return meta->mMeta.mProvenanceRecognised ? 1 : 0;
    });
}

void mio_read_metadata_free(mio_read_metadata* meta) {
    delete meta;
}

int mio_reader_supports_options(const char* format) {
    return guarded_ptr(-1, [&]() -> int {
        if (!format)
            throw meshioplusplus::ReadError("meshio++: format is NULL");
        return meshioplusplus::registry_reader_supports_options(format) ? 1 : 0;
    });
}

mio_status mio_write(const char* path, const mio_mesh* mesh, const char* format) {
    return guarded([&]() -> mio_status {
        if (!path || !mesh)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: path/mesh is NULL");
        std::string fmt = meshioplusplus::resolve_format(path, format_or_empty(format));
        meshioplusplus::detail::provenance_begin_write();
        auto it = meshioplusplus::registry_writers().find(fmt);
        if (it == meshioplusplus::registry_writers().end())
            return fail(MIO_ERR_NOT_FOUND, unknown_format_message(fmt, /*for_write=*/true));
        it->second(path, mesh->mMesh);
        return MIO_OK;
    });
}

void mio_write_opts_init(mio_write_opts* opts) {
    if (!opts)
        return;
    *opts = mio_write_opts{};  // value-initialized: all zero == mio_write()
}

namespace {

/// mio_write_opts -> WriteOptions, shared by mio_write_ex and
/// mio_sequence_to_timeseries_ex. Returns MIO_ERR_INVALID_ARG (via fail(),
/// which sets the thread-local message) on a bad enum value; MIO_OK otherwise.
mio_status write_opts_to_cxx(const mio_write_opts& rOpts, meshioplusplus::WriteOptions& rOut) {
    switch (rOpts.encoding) {
        case MIO_ENCODING_DEFAULT:
            break;
        case MIO_ENCODING_ASCII:
            rOut.mEncoding = meshioplusplus::WriteEncoding::Ascii;
            break;
        case MIO_ENCODING_BINARY:
            rOut.mEncoding = meshioplusplus::WriteEncoding::Binary;
            break;
        default:
            return fail(MIO_ERR_INVALID_ARG, "meshio++: bad mio_write_opts.encoding");
    }
    switch (rOpts.codec) {
        case MIO_CODEC_DEFAULT:
            break;
        case MIO_CODEC_NONE:
            rOut.mCodec = meshioplusplus::detail::VtkCodec::None;
            rOut.mCodecSet = true;
            break;
        case MIO_CODEC_ZLIB:
            rOut.mCodec = meshioplusplus::detail::VtkCodec::Zlib;
            rOut.mCodecSet = true;
            break;
        case MIO_CODEC_LZ4:
            rOut.mCodec = meshioplusplus::detail::VtkCodec::LZ4;
            rOut.mCodecSet = true;
            break;
        case MIO_CODEC_ZSTD:
            rOut.mCodec = meshioplusplus::detail::VtkCodec::ZSTD;
            rOut.mCodecSet = true;
            break;
        default:
            return fail(MIO_ERR_INVALID_ARG, "meshio++: bad mio_write_opts.codec");
    }
    if (rOpts.float_format)
        rOut.mFloatFormat = rOpts.float_format;
    return MIO_OK;
}

}  // namespace

mio_status mio_write_ex(const char* path, const mio_mesh* mesh, const char* format,
                        const mio_write_opts* opts) {
    return guarded([&]() -> mio_status {
        if (!path || !mesh)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: path/mesh is NULL");
        if (!opts)
            return mio_write(path, mesh, format);

        meshioplusplus::WriteOptions w;
        const mio_status opt_status = write_opts_to_cxx(*opts, w);
        if (opt_status != MIO_OK)
            return opt_status;

        meshioplusplus::registry_write_ex(path, mesh->mMesh, format_or_empty(format), w);
        return MIO_OK;
    });
}

namespace {

/// The thread-local stack of open provenance scopes -- the exact twin of the
/// pybind bridge's `provenance_scope_stack()` (bindings/python/_core.cpp),
/// needed for the identical reason: a C caller's begin/end calls are two
/// separate ABI crossings, so something has to keep the scope alive between
/// them rather than relying on C++ RAII across the boundary.
std::vector<std::unique_ptr<meshioplusplus::detail::ProvenanceScope>>& provenance_scope_stack() {
    thread_local std::vector<std::unique_ptr<meshioplusplus::detail::ProvenanceScope>> stack;
    return stack;
}

}  // namespace

mio_status mio_provenance_scope_begin(int mode) {
    return guarded([&]() -> mio_status {
        if (mode < 0 || mode > 2)
            return fail(MIO_ERR_INVALID_ARG,
                        "meshio++: unknown mio_provenance_mode " + std::to_string(mode));
        provenance_scope_stack().push_back(
            std::make_unique<meshioplusplus::detail::ProvenanceScope>(
                static_cast<meshioplusplus::detail::ProvenanceMode>(mode)));
        return MIO_OK;
    });
}

void mio_provenance_scope_end(void) {
    auto& stack = provenance_scope_stack();
    if (!stack.empty())
        stack.pop_back();
}

void mio_provenance_note(const char* category, const char* detail) {
    if (!category || !detail)
        return;
    meshioplusplus::detail::provenance_note(category, detail);
}

void mio_provenance_set_source(const char* path, const char* format) {
    meshioplusplus::detail::provenance_set_source(path ? path : "", format ? format : "");
}

void mio_provenance_set_target(const char* format, const char* encoding, const char* codec,
                               const char* float_format) {
    meshioplusplus::detail::provenance_set_target(format ? format : "", encoding ? encoding : "",
                                                  codec ? codec : "",
                                                  float_format ? float_format : "");
}

mio_status mio_convert(const char* in_path, const char* in_format, const char* out_path,
                       const char* out_format) {
    return guarded([&]() -> mio_status {
        if (!in_path || !out_path)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: path is NULL");
        std::string rfmt = meshioplusplus::resolve_format(in_path, format_or_empty(in_format));
        std::string wfmt = meshioplusplus::resolve_format(out_path, format_or_empty(out_format));
        auto rit = meshioplusplus::registry_readers().find(rfmt);
        if (rit == meshioplusplus::registry_readers().end())
            return fail(MIO_ERR_NOT_FOUND, unknown_format_message(rfmt, /*for_write=*/false));
        auto wit = meshioplusplus::registry_writers().find(wfmt);
        if (wit == meshioplusplus::registry_writers().end())
            return fail(MIO_ERR_NOT_FOUND, unknown_format_message(wfmt, /*for_write=*/true));
        wit->second(out_path, rit->second(in_path));
        return MIO_OK;
    });
}

/* ------------------------------------------------------------------ */
/* Mesh operations                                                     */
/* ------------------------------------------------------------------ */

mio_mesh* mio_extract_surface(const mio_mesh* mesh, int record_parent_ids) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        return new mio_mesh{meshioplusplus::extract_surface(mesh->mMesh, record_parent_ids != 0)};
    });
}

mio_mesh* mio_extract_skin(const mio_mesh* mesh, int linearize) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        return new mio_mesh{meshioplusplus::extract_skin(mesh->mMesh, linearize != 0)};
    });
}

mio_mesh* mio_attach_quality(const mio_mesh* mesh) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        return new mio_mesh{meshioplusplus::attach_quality(mesh->mMesh)};
    });
}

mio_mesh* mio_merge(const mio_mesh* const* meshes, int64_t count, int weld, double atol,
                    int source_tag, int data_policy, int drop_duplicate_cells) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!meshes || count < 1)
            throw meshioplusplus::ReadError("meshio++: need at least one input mesh");
        std::vector<const meshioplusplus::Mesh*> ptrs;
        ptrs.reserve(static_cast<std::size_t>(count));
        for (int64_t i = 0; i < count; ++i) {
            if (!meshes[i])
                throw meshioplusplus::ReadError("meshio++: input mesh is NULL");
            ptrs.push_back(&meshes[i]->mMesh);
        }
        meshioplusplus::MergeOptions opts;
        opts.weld = weld != 0;
        opts.atol = atol;
        opts.source_tag = source_tag != 0;
        opts.drop_duplicate_cells = drop_duplicate_cells != 0;
        opts.data_policy = (data_policy == 1) ? meshioplusplus::MergeDataPolicy::Fill
                                              : meshioplusplus::MergeDataPolicy::Intersection;
        return new mio_mesh{meshioplusplus::merge(ptrs, opts).mMesh};
    });
}

mio_mesh* mio_transform(const mio_mesh* mesh, const double* matrix, int rotate_vector_data) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        if (!matrix)
            throw meshioplusplus::ReadError("meshio++: matrix is NULL");
        meshioplusplus::AffineTransform xf = meshioplusplus::transform_from_matrix(matrix);
        return new mio_mesh{meshioplusplus::transform(mesh->mMesh, xf, rotate_vector_data != 0)};
    });
}

mio_mesh* mio_clean(const mio_mesh* mesh, int weld, double atol, int remove_orphans,
                    int drop_degenerate, int drop_duplicate_cells, int64_t* points_welded,
                    int64_t* points_removed_orphan, int64_t* cells_dropped_degenerate,
                    int64_t* cells_dropped_duplicate) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        meshioplusplus::CleanOptions opts;
        opts.weld = weld != 0;
        opts.atol = atol;
        opts.remove_orphans = remove_orphans != 0;
        opts.drop_degenerate = drop_degenerate != 0;
        opts.drop_duplicate_cells = drop_duplicate_cells != 0;
        meshioplusplus::CleanResult r = meshioplusplus::clean(mesh->mMesh, opts);
        if (points_welded)
            *points_welded = r.mPointsWelded;
        if (points_removed_orphan)
            *points_removed_orphan = r.mPointsRemovedOrphan;
        if (cells_dropped_degenerate)
            *cells_dropped_degenerate = r.mCellsDroppedDegenerate;
        if (cells_dropped_duplicate)
            *cells_dropped_duplicate = r.mCellsDroppedDuplicate;
        return new mio_mesh{std::move(r.mMesh)};
    });
}

mio_mesh* mio_smooth(const mio_mesh* mesh, const char* method, int iterations, double lambda,
                     double mu, int fix_boundary, int preserve_features, double feature_angle,
                     int guard_inversion, int64_t* nodes_moved, double* max_displacement,
                     int64_t* skipped_inversion) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        meshioplusplus::SmoothOptions opts;
        // A negative lambda is the "this method's own default" sentinel and is
        // passed through unchanged; mFrozen is deliberately not exposed here.
        opts.mMethod = meshioplusplus::smooth_method_from_name(method ? method : "taubin");
        opts.mIterations = iterations;
        opts.mLambda = lambda;
        opts.mMu = mu;
        opts.mFixBoundary = fix_boundary != 0;
        opts.mPreserveFeatures = preserve_features != 0;
        opts.mFeatureAngleDeg = feature_angle;
        opts.mGuardInversion = guard_inversion != 0;
        meshioplusplus::SmoothResult r = meshioplusplus::smooth(mesh->mMesh, opts);
        if (nodes_moved)
            *nodes_moved = r.mNumNodesMoved;
        if (max_displacement)
            *max_displacement = r.mMaxDisplacement;
        if (skipped_inversion)
            *skipped_inversion = r.mNumSkippedInversion;
        return new mio_mesh{std::move(r.mMesh)};
    });
}

mio_mesh* mio_optimize_volume(const mio_mesh* mesh, int max_iterations, int relocate, int flip,
                              int preserve_boundary, double min_improvement, int64_t* num_flips,
                              int64_t* num_23_flips, int64_t* num_32_flips,
                              int64_t* num_vertices_moved, int64_t* num_tets,
                              double* min_quality_before, double* min_quality_after) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        meshioplusplus::OptimizeVolumeOptions opts;
        opts.mMaxIterations = max_iterations;
        opts.mRelocate = relocate != 0;
        opts.mFlip = flip != 0;
        opts.mPreserveBoundary = preserve_boundary != 0;
        opts.mMinImprovement = min_improvement;
        meshioplusplus::OptimizeVolumeResult r = meshioplusplus::optimize_volume(mesh->mMesh, opts);
        if (num_flips)
            *num_flips = r.mNumFlips;
        if (num_23_flips)
            *num_23_flips = r.mNum23Flips;
        if (num_32_flips)
            *num_32_flips = r.mNum32Flips;
        if (num_vertices_moved)
            *num_vertices_moved = r.mNumVerticesMoved;
        if (num_tets)
            *num_tets = r.mNumTets;
        if (min_quality_before)
            *min_quality_before = r.mMinQualityBefore;
        if (min_quality_after)
            *min_quality_after = r.mMinQualityAfter;
        return new mio_mesh{std::move(r.mMesh)};
    });
}

mio_mesh* mio_undo_green(const mio_mesh* coarse, const mio_mesh* fine, int64_t* num_groups_undone,
                         int64_t* num_cells_removed) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!coarse)
            throw meshioplusplus::ReadError("meshio++: coarse mesh is NULL");
        if (!fine)
            throw meshioplusplus::ReadError("meshio++: fine mesh is NULL");
        meshioplusplus::UndoGreenResult r = meshioplusplus::undo_green(coarse->mMesh, fine->mMesh);
        if (num_groups_undone)
            *num_groups_undone = r.mNumGroupsUndone;
        if (num_cells_removed)
            *num_cells_removed = r.mNumCellsRemoved;
        return new mio_mesh{std::move(r.mMesh)};
    });
}

mio_mesh* mio_interpolate(const mio_mesh* source, const mio_mesh* target, const char* method,
                          const char* const* arrays, int64_t arrays_count, int extrapolate,
                          double default_value, const char* on_conflict) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!source)
            throw meshioplusplus::ReadError("meshio++: source mesh is NULL");
        if (!target)
            throw meshioplusplus::ReadError("meshio++: target mesh is NULL");
        meshioplusplus::InterpolateOptions opts;
        opts.mMethod = meshioplusplus::interpolate_method_from_name(method ? method : "nearest");
        // NULL or a non-positive count means "every source point_data array"
        // (deliberately laxer than data_name_list — empty-means-all is the
        // core's own convention for this option).
        if (arrays && arrays_count > 0) {
            opts.mArrays.reserve(static_cast<std::size_t>(arrays_count));
            for (int64_t i = 0; i < arrays_count; ++i) {
                if (!arrays[i])
                    throw meshioplusplus::ReadError("meshio++: arrays[" + std::to_string(i) +
                                                    "] is NULL");
                opts.mArrays.emplace_back(arrays[i]);
            }
        }
        opts.mExtrapolate = extrapolate != 0;
        opts.mDefaultValue = default_value;
        opts.mOnConflict =
            meshioplusplus::interpolate_conflict_from_name(on_conflict ? on_conflict : "error");
        return new mio_mesh{meshioplusplus::interpolate(source->mMesh, target->mMesh, opts)};
    });
}

mio_mesh* mio_conservative_interpolate(const mio_mesh* source, const mio_mesh* target,
                                       const char* const* arrays, int64_t arrays_count,
                                       double default_value, const char* on_conflict) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!source)
            throw meshioplusplus::ReadError("meshio++: source mesh is NULL");
        if (!target)
            throw meshioplusplus::ReadError("meshio++: target mesh is NULL");
        meshioplusplus::ConservativeInterpolateOptions opts;
        // NULL or a non-positive count means "every source point_data and
        // cell_data array" (the interpolate convention, extended: there is
        // one algorithm regardless of location here).
        if (arrays && arrays_count > 0) {
            opts.mArrays.reserve(static_cast<std::size_t>(arrays_count));
            for (int64_t i = 0; i < arrays_count; ++i) {
                if (!arrays[i])
                    throw meshioplusplus::ReadError("meshio++: arrays[" + std::to_string(i) +
                                                    "] is NULL");
                opts.mArrays.emplace_back(arrays[i]);
            }
        }
        opts.mDefaultValue = default_value;
        opts.mOnConflict = meshioplusplus::conservative_interpolate_conflict_from_name(
            on_conflict ? on_conflict : "error");
        return new mio_mesh{
            meshioplusplus::conservative_interpolate(source->mMesh, target->mMesh, opts)};
    });
}

mio_mesh* mio_slice(const mio_mesh* mesh, const double* origin, const double* normal,
                    int record_parent_ids) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh || !origin || !normal)
            throw meshioplusplus::ReadError("meshio++: mesh/origin/normal is NULL");
        meshioplusplus::SliceOptions opts;
        opts.mOrigin = {origin[0], origin[1], origin[2]};
        opts.mNormal = {normal[0], normal[1], normal[2]};
        opts.mRecordParentIds = record_parent_ids != 0;
        return new mio_mesh{meshioplusplus::slice(mesh->mMesh, opts)};
    });
}

mio_mesh* mio_isosurface(const mio_mesh* mesh, const char* array_name, const double* isovalues,
                         int n_isovalues, int component, int record_parent_ids) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh || !array_name || !isovalues)
            throw meshioplusplus::ReadError("meshio++: mesh/array_name/isovalues is NULL");
        if (n_isovalues <= 0)
            throw meshioplusplus::ReadError("meshio++: at least one isovalue is required");
        meshioplusplus::IsosurfaceOptions opts;
        opts.mArrayName = array_name;
        opts.mIsovalues.assign(isovalues, isovalues + n_isovalues);
        if (component >= 0)
            opts.mComponent = component;
        opts.mRecordParentIds = record_parent_ids != 0;
        return new mio_mesh{meshioplusplus::isosurface(mesh->mMesh, opts)};
    });
}

mio_mesh* mio_gradient(const mio_mesh* mesh, const char* array_name, const char* op,
                       const char* method, const char* location, const char* output_name,
                       int component, int overwrite, int64_t* num_skipped, int64_t* num_fallback) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh || !array_name)
            throw meshioplusplus::ReadError("meshio++: mesh/array_name is NULL");
        meshioplusplus::GradientOptions opts;
        opts.mArrayName = array_name;
        opts.mOperator = meshioplusplus::gradient_operator_from_name(op ? op : "");
        opts.mMethod = meshioplusplus::gradient_method_from_name(method ? method : "");
        opts.mLocation =
            meshioplusplus::data_location_from_name((location && *location) ? location : "cell");
        if (output_name)
            opts.mOutputName = output_name;
        // A negative component means EVERY component here -- deliberately the
        // opposite of mio_isosurface, where it means the row magnitude.
        if (component >= 0)
            opts.mComponent = component;
        opts.mOverwrite = overwrite != 0;
        meshioplusplus::GradientResult r = meshioplusplus::gradient(mesh->mMesh, opts);
        if (num_skipped)
            *num_skipped = r.mNumSkipped;
        if (num_fallback)
            *num_fallback = r.mNumFallback;
        return new mio_mesh{std::move(r.mMesh)};
    });
}

mio_mesh* mio_hessian(const mio_mesh* mesh, const char* array_name, const char* method,
                      const char* location, const char* output_name, int overwrite,
                      int64_t* num_skipped, int64_t* num_fallback) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh || !array_name)
            throw meshioplusplus::ReadError("meshio++: mesh/array_name is NULL");
        meshioplusplus::HessianOptions opts;
        opts.mArrayName = array_name;
        opts.mMethod = meshioplusplus::gradient_method_from_name(method ? method : "");
        opts.mLocation =
            meshioplusplus::data_location_from_name((location && *location) ? location : "cell");
        if (output_name)
            opts.mOutputName = output_name;
        opts.mOverwrite = overwrite != 0;
        meshioplusplus::HessianResult r = meshioplusplus::hessian(mesh->mMesh, opts);
        if (num_skipped)
            *num_skipped = r.mNumSkipped;
        if (num_fallback)
            *num_fallback = r.mNumFallback;
        return new mio_mesh{std::move(r.mMesh)};
    });
}

mio_mesh* mio_estimate_error(const mio_mesh* mesh, const char* array_name, const char* method,
                             const char* marking, double marking_value, const char* output_name,
                             const char* marked_name, int overwrite, double* global_error,
                             int64_t* num_skipped, int64_t* num_marked) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh || !array_name)
            throw meshioplusplus::ReadError("meshio++: mesh/array_name is NULL");
        meshioplusplus::ErrorOptions opts;
        opts.mArrayName = array_name;
        opts.mMethod = meshioplusplus::error_method_from_name(method ? method : "");
        opts.mMarking = meshioplusplus::error_marking_from_name(marking ? marking : "");
        opts.mMarkingValue = marking_value;
        if (output_name)
            opts.mOutputName = output_name;
        if (marked_name)
            opts.mMarkedName = marked_name;
        opts.mOverwrite = overwrite != 0;
        meshioplusplus::ErrorResult r = meshioplusplus::estimate_error(mesh->mMesh, opts);
        if (global_error)
            *global_error = r.mGlobalError;
        if (num_skipped)
            *num_skipped = r.mNumSkipped;
        if (num_marked)
            *num_marked = r.mNumMarked;
        return new mio_mesh{std::move(r.mMesh)};
    });
}

namespace {

/// Translate the flat option struct into the core's RemeshOptions -- the
/// mio_refine_opts precedent (capi_refine_options).
meshioplusplus::RemeshOptions capi_remesh_options(const mio_remesh_opts& rOpts) {
    meshioplusplus::RemeshOptions options;
    options.mNumClusters = rOpts.num_clusters;
    options.mSubdivide = rOpts.subdivide;
    options.mSubsampleRatio = rOpts.subsample_ratio;
    options.mMaxSubdivide = rOpts.max_subdivide;
    options.mMaxIterations = rOpts.max_iterations;
    options.mMaxRepairPasses = rOpts.max_repair_passes;
    options.mMetric =
        meshioplusplus::remesh_metric_from_name(rOpts.metric ? rOpts.metric : "isotropic");
    options.mGradation = rOpts.gradation;
    options.mPreserveBoundary = rOpts.preserve_boundary != 0;
    options.mMaxAnisotropy = rOpts.max_anisotropy;
    return options;
}

mio_mesh* capi_remesh(const mio_mesh* pMesh, const meshioplusplus::RemeshOptions& rOptions,
                      mio_remesh_report* pReport) {
    if (!pMesh)
        throw meshioplusplus::ReadError("meshio++: mesh is NULL");
    meshioplusplus::RemeshResult r = meshioplusplus::remesh(pMesh->mMesh, rOptions);
    if (pReport) {
        *pReport = mio_remesh_report{};
        pReport->num_clusters = r.mNumClusters;
        pReport->num_iterations = r.mNumIterations;
        pReport->subdivide_applied = r.mSubdivideApplied;
        pReport->num_isolated_clusters = r.mNumIsolatedClusters;
        pReport->num_non_manifold_vertices = r.mNumNonManifoldVertices;
    }
    return new mio_mesh{std::move(r.mMesh)};
}

}  // namespace

static_assert(sizeof(mio_remesh_opts) == 120, "mio_remesh_opts grew outside its reserved tail");
static_assert(sizeof(mio_remesh_report) == 72, "mio_remesh_report grew outside its reserved tail");

void mio_remesh_opts_init(mio_remesh_opts* opts) {
    if (!opts)
        return;
    *opts = mio_remesh_opts{};  // value-initialized: zero every field first
    opts->subdivide = -1;
    opts->max_subdivide = 4;
    opts->subsample_ratio = 10.0;
    opts->max_iterations = 100;
    opts->max_repair_passes = 10;
    opts->preserve_boundary = 1;
    opts->max_anisotropy = meshioplusplus::kRemeshDefaultMaxAnisotropy;
}

mio_mesh* mio_remesh(const mio_mesh* mesh, int64_t num_clusters, int subdivide,
                     double subsample_ratio, int max_subdivide, int max_iterations,
                     int max_repair_passes, const char* metric, double gradation,
                     int preserve_boundary, int64_t* num_clusters_out, int64_t* num_iterations,
                     int* subdivide_applied, int64_t* num_isolated_clusters,
                     int64_t* num_non_manifold_vertices) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        meshioplusplus::RemeshOptions opts;
        opts.mNumClusters = num_clusters;
        opts.mSubdivide = subdivide;
        opts.mSubsampleRatio = subsample_ratio;
        opts.mMaxSubdivide = max_subdivide;
        opts.mMaxIterations = max_iterations;
        opts.mMaxRepairPasses = max_repair_passes;
        opts.mMetric = meshioplusplus::remesh_metric_from_name(metric ? metric : "isotropic");
        opts.mGradation = gradation;
        opts.mPreserveBoundary = preserve_boundary != 0;
        // max_anisotropy has no flat parameter here -- mio_remesh_ex is
        // required to move it away from the default (RemeshOptions's own
        // member-initializer default, inherited unchanged above).
        mio_remesh_report report{};
        mio_mesh* out = capi_remesh(mesh, opts, &report);
        if (num_clusters_out)
            *num_clusters_out = report.num_clusters;
        if (num_iterations)
            *num_iterations = report.num_iterations;
        if (subdivide_applied)
            *subdivide_applied = report.subdivide_applied;
        if (num_isolated_clusters)
            *num_isolated_clusters = report.num_isolated_clusters;
        if (num_non_manifold_vertices)
            *num_non_manifold_vertices = report.num_non_manifold_vertices;
        return out;
    });
}

mio_mesh* mio_remesh_ex(const mio_mesh* mesh, const mio_remesh_opts* opts,
                        mio_remesh_report* report) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (opts == nullptr) {
            mio_remesh_opts defaults;
            mio_remesh_opts_init(&defaults);
            return capi_remesh(mesh, capi_remesh_options(defaults), report);
        }
        return capi_remesh(mesh, capi_remesh_options(*opts), report);
    });
}

mio_mesh* mio_crop_bbox(const mio_mesh* mesh, const double* lo, const double* hi, int mode,
                        int record_ids) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh || !lo || !hi)
            throw meshioplusplus::ReadError("meshio++: mesh/lo/hi is NULL");
        meshioplusplus::CropMode m =
            (mode == 1) ? meshioplusplus::CropMode::Any : meshioplusplus::CropMode::All;
        return new mio_mesh{
            meshioplusplus::crop_bbox(mesh->mMesh, lo, hi, m, record_ids != 0).mMesh};
    });
}

mio_mesh* mio_crop_plane(const mio_mesh* mesh, const double* point, const double* normal, int mode,
                         int record_ids) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh || !point || !normal)
            throw meshioplusplus::ReadError("meshio++: mesh/point/normal is NULL");
        meshioplusplus::CropMode m =
            (mode == 1) ? meshioplusplus::CropMode::Any : meshioplusplus::CropMode::All;
        return new mio_mesh{
            meshioplusplus::crop_halfspace(mesh->mMesh, point, normal, m, record_ids != 0).mMesh};
    });
}

mio_mesh* mio_crop_predicate(const mio_mesh* mesh, const char* array, int compare, double value,
                             int record_ids) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh || !array)
            throw meshioplusplus::ReadError("meshio++: mesh/array is NULL");
        if (compare < 0 || compare > MIO_REFINE_NE)
            throw meshioplusplus::ReadError("meshio++: crop: unknown comparison " +
                                            std::to_string(compare));
        return new mio_mesh{
            meshioplusplus::crop_predicate(mesh->mMesh, array,
                                           static_cast<meshioplusplus::RefineCompare>(compare),
                                           value, record_ids != 0)
                .mMesh};
    });
}

mio_split_result* mio_split(const mio_mesh* mesh, const char* by, const char* tag_name) {
    return guarded_ptr(static_cast<mio_split_result*>(nullptr), [&]() -> mio_split_result* {
        if (!mesh || !by)
            throw meshioplusplus::ReadError("meshio++: mesh/by is NULL");
        meshioplusplus::SplitResult r = meshioplusplus::split(
            mesh->mMesh, meshioplusplus::split_by_from_name(by), tag_name ? tag_name : "");
        auto* out = new mio_split_result{};
        out->mMeshes.reserve(r.mPieces.size());
        out->mKeys.reserve(r.mPieces.size());
        for (meshioplusplus::SplitPiece& p : r.mPieces) {
            out->mMeshes.push_back(mio_mesh{std::move(p.mMesh)});
            out->mKeys.push_back(std::move(p.mKey));
        }
        return out;
    });
}

int64_t mio_split_result_count(const mio_split_result* result) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return static_cast<int64_t>(result->mMeshes.size());
    });
}

int64_t mio_split_result_key(const mio_split_result* result, int64_t index, char* buf,
                             int64_t buflen) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= result->mKeys.size())
            throw meshioplusplus::ReadError("meshio++: split piece index out of range");
        return copy_string(result->mKeys[static_cast<std::size_t>(index)], buf, buflen);
    });
}

const mio_mesh* mio_split_result_mesh(const mio_split_result* result, int64_t index) {
    return guarded_ptr(static_cast<const mio_mesh*>(nullptr), [&]() -> const mio_mesh* {
        if (!result || index < 0 || static_cast<std::size_t>(index) >= result->mMeshes.size())
            return nullptr;
        return &result->mMeshes[static_cast<std::size_t>(index)];
    });
}

mio_mesh* mio_split_result_take_mesh(mio_split_result* result, int64_t index) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!result || index < 0 || static_cast<std::size_t>(index) >= result->mMeshes.size())
            throw meshioplusplus::ReadError("meshio++: split piece index out of range");
        return new mio_mesh{std::move(result->mMeshes[static_cast<std::size_t>(index)].mMesh)};
    });
}

void mio_split_result_free(mio_split_result* result) {
    delete result;
}

mio_convert_cells_result* mio_convert_cells(const mio_mesh* mesh, const char* mode,
                                            int record_parent_ids) {
    return guarded_ptr(static_cast<mio_convert_cells_result*>(nullptr),
                       [&]() -> mio_convert_cells_result* {
                           if (!mesh || !mode)
                               throw meshioplusplus::ReadError("meshio++: mesh/mode is NULL");
                           meshioplusplus::ConvertCellsOptions options;
                           options.mMode = meshioplusplus::convert_cells_mode_from_name(mode);
                           options.mRecordParentIds = record_parent_ids != 0;
                           meshioplusplus::ConvertCellsResult r =
                               meshioplusplus::convert_cells(mesh->mMesh, options);
                           auto* out = new mio_convert_cells_result{};
                           out->mMesh = mio_mesh{std::move(r.mMesh)};
                           out->mPointMap = std::move(r.mPointMap);
                           out->mCellMaps = std::move(r.mCellMaps);
                           return out;
                       });
}

const mio_mesh* mio_convert_cells_result_mesh(const mio_convert_cells_result* result) {
    return guarded_ptr(static_cast<const mio_mesh*>(nullptr), [&]() -> const mio_mesh* {
        if (!result)
            return nullptr;
        return &result->mMesh;
    });
}

mio_mesh* mio_convert_cells_result_take_mesh(mio_convert_cells_result* result) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return new mio_mesh{std::move(result->mMesh.mMesh)};
    });
}

mio_status mio_convert_cells_result_point_map(const mio_convert_cells_result* result,
                                              const void** data, mio_dtype* dtype, int64_t* n) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        const NDArray& a = result->mPointMap;
        if (data)
            *data = a.Data();
        if (dtype)
            *dtype = from_dtype(a.Dtype());
        if (n)
            *n = a.Shape().empty() ? 0 : static_cast<int64_t>(a.Shape()[0]);
        return MIO_OK;
    });
}

int64_t mio_convert_cells_result_num_cell_maps(const mio_convert_cells_result* result) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return static_cast<int64_t>(result->mCellMaps.size());
    });
}

mio_status mio_convert_cells_result_cell_map(const mio_convert_cells_result* result, int64_t block,
                                             const void** data, mio_dtype* dtype, int64_t* n) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        if (block < 0 || static_cast<std::size_t>(block) >= result->mCellMaps.size())
            return fail(MIO_ERR_NOT_FOUND, "meshio++: cell-map block index out of range");
        const NDArray& a = result->mCellMaps[static_cast<std::size_t>(block)];
        if (data)
            *data = a.Data();
        if (dtype)
            *dtype = from_dtype(a.Dtype());
        if (n)
            *n = a.Shape().empty() ? 0 : static_cast<int64_t>(a.Shape()[0]);
        return MIO_OK;
    });
}

void mio_convert_cells_result_free(mio_convert_cells_result* result) {
    delete result;
}

mio_subdivide_result* mio_subdivide(const mio_mesh* mesh, int record_parent_ids) {
    return guarded_ptr(static_cast<mio_subdivide_result*>(nullptr), [&]() -> mio_subdivide_result* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        meshioplusplus::SubdivideOptions options;
        options.mRecordParentIds = record_parent_ids != 0;
        meshioplusplus::SubdivideResult r = meshioplusplus::subdivide(mesh->mMesh, options);
        auto* out = new mio_subdivide_result{};
        out->mMesh = mio_mesh{std::move(r.mMesh)};
        out->mCellMaps = std::move(r.mCellMaps);
        return out;
    });
}

const mio_mesh* mio_subdivide_result_mesh(const mio_subdivide_result* result) {
    return guarded_ptr(static_cast<const mio_mesh*>(nullptr), [&]() -> const mio_mesh* {
        if (!result)
            return nullptr;
        return &result->mMesh;
    });
}

mio_mesh* mio_subdivide_result_take_mesh(mio_subdivide_result* result) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return new mio_mesh{std::move(result->mMesh.mMesh)};
    });
}

int64_t mio_subdivide_result_num_cell_maps(const mio_subdivide_result* result) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return static_cast<int64_t>(result->mCellMaps.size());
    });
}

mio_status mio_subdivide_result_cell_map(const mio_subdivide_result* result, int64_t block,
                                         const void** data, mio_dtype* dtype, int64_t* n) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        if (block < 0 || static_cast<std::size_t>(block) >= result->mCellMaps.size())
            return fail(MIO_ERR_NOT_FOUND, "meshio++: cell-map block index out of range");
        const NDArray& a = result->mCellMaps[static_cast<std::size_t>(block)];
        if (data)
            *data = a.Data();
        if (dtype)
            *dtype = from_dtype(a.Dtype());
        if (n)
            *n = a.Shape().empty() ? 0 : static_cast<int64_t>(a.Shape()[0]);
        return MIO_OK;
    });
}

void mio_subdivide_result_free(mio_subdivide_result* result) {
    delete result;
}

mio_agglomerate_result* mio_agglomerate(const mio_mesh* mesh, int64_t target_group_size) {
    return guarded_ptr(
        static_cast<mio_agglomerate_result*>(nullptr), [&]() -> mio_agglomerate_result* {
            if (!mesh)
                throw meshioplusplus::ReadError("meshio++: mesh is NULL");
            if (target_group_size < 0)
                throw std::invalid_argument(
                    "meshio++: agglomerate: target_group_size must be >= 1");
            meshioplusplus::AgglomerateOptions options;
            options.mTargetGroupSize = static_cast<std::size_t>(target_group_size);
            meshioplusplus::AgglomerateResult r = meshioplusplus::agglomerate(mesh->mMesh, options);
            auto* out = new mio_agglomerate_result{};
            out->mMesh = mio_mesh{std::move(r.mMesh)};
            out->mCellMap = std::move(r.mCellMap);
            return out;
        });
}

const mio_mesh* mio_agglomerate_result_mesh(const mio_agglomerate_result* result) {
    return guarded_ptr(static_cast<const mio_mesh*>(nullptr), [&]() -> const mio_mesh* {
        if (!result)
            return nullptr;
        return &result->mMesh;
    });
}

mio_mesh* mio_agglomerate_result_take_mesh(mio_agglomerate_result* result) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return new mio_mesh{std::move(result->mMesh.mMesh)};
    });
}

mio_status mio_agglomerate_result_cell_map(const mio_agglomerate_result* result, const void** data,
                                           mio_dtype* dtype, int64_t* n) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        const NDArray& a = result->mCellMap;
        if (data)
            *data = a.Data();
        if (dtype)
            *dtype = from_dtype(a.Dtype());
        if (n)
            *n = a.Shape().empty() ? 0 : static_cast<int64_t>(a.Shape()[0]);
        return MIO_OK;
    });
}

void mio_agglomerate_result_free(mio_agglomerate_result* result) {
    delete result;
}

// The two flat enums duplicate the C++ ones across the ABI, so drift is a
// compile error rather than a wrong answer -- the mio_region_kind pattern.
static_assert(static_cast<int>(meshioplusplus::RefineClosure::RedGreen) ==
                  MIO_REFINE_CLOSURE_REDGREEN,
              "mio_refine_closure drifted from RefineClosure");
static_assert(static_cast<int>(meshioplusplus::RefineClosure::Propagate) ==
                  MIO_REFINE_CLOSURE_PROPAGATE,
              "mio_refine_closure drifted from RefineClosure");
static_assert(static_cast<int>(meshioplusplus::RefineClosure::Balanced) ==
                  MIO_REFINE_CLOSURE_BALANCED,
              "mio_refine_closure drifted from RefineClosure");
static_assert(static_cast<int>(meshioplusplus::RefineCompare::Less) == MIO_REFINE_LT &&
                  static_cast<int>(meshioplusplus::RefineCompare::LessEqual) == MIO_REFINE_LE &&
                  static_cast<int>(meshioplusplus::RefineCompare::Greater) == MIO_REFINE_GT &&
                  static_cast<int>(meshioplusplus::RefineCompare::GreaterEqual) == MIO_REFINE_GE &&
                  static_cast<int>(meshioplusplus::RefineCompare::Equal) == MIO_REFINE_EQ &&
                  static_cast<int>(meshioplusplus::RefineCompare::NotEqual) == MIO_REFINE_NE,
              "mio_refine_compare drifted from RefineCompare");

// The size is ABI: the Fortran and Julia mirrors of this struct hard-code it,
// and the Julia binding checks it at load. Pinning it here makes a field added
// outside the `reserved` tail a compile error rather than silent corruption.
static_assert(sizeof(mio_refine_opts) == 112, "mio_refine_opts grew outside its reserved tail");

void mio_refine_opts_init(mio_refine_opts* opts) {
    if (!opts)
        return;
    *opts = mio_refine_opts{};  // value-initialized: all zero == refine everything once
    opts->levels = 1;
}

namespace {

/// Translate the flat option struct into the core's RefineOptions.
meshioplusplus::RefineOptions capi_refine_options(const mio_refine_opts& rOpts) {
    meshioplusplus::RefineOptions options;
    options.mLevels = rOpts.levels;
    options.mRecordParentIds = rOpts.record_parent_ids != 0;
    options.mRecordLevels = rOpts.record_levels != 0;
    options.mRecordHierarchy = rOpts.record_hierarchy != 0;
    if (rOpts.cells != nullptr && rOpts.num_cells > 0)
        options.mCells.assign(rOpts.cells, rOpts.cells + rOpts.num_cells);
    if (rOpts.region != nullptr)
        options.mRegion = rOpts.region;
    if (rOpts.predicate_array != nullptr)
        options.mPredicateArray = rOpts.predicate_array;
    options.mPredicateValue = rOpts.predicate_value;
    if (rOpts.closure < 0 || rOpts.closure > MIO_REFINE_CLOSURE_BALANCED)
        throw meshioplusplus::ReadError("meshio++: refine: unknown closure " +
                                        std::to_string(rOpts.closure));
    options.mClosure = static_cast<meshioplusplus::RefineClosure>(rOpts.closure);
    if (rOpts.predicate_op < 0 || rOpts.predicate_op > MIO_REFINE_NE)
        throw meshioplusplus::ReadError("meshio++: refine: unknown comparison " +
                                        std::to_string(rOpts.predicate_op));
    options.mPredicateOp = static_cast<meshioplusplus::RefineCompare>(rOpts.predicate_op);
    return options;
}

mio_refine_result* capi_refine(const mio_mesh* pMesh,
                               const meshioplusplus::RefineOptions& rOptions) {
    if (!pMesh)
        throw meshioplusplus::ReadError("meshio++: mesh is NULL");
    meshioplusplus::RefineResult r = meshioplusplus::refine(pMesh->mMesh, rOptions);
    auto* out = new mio_refine_result{};
    out->mMesh = mio_mesh{std::move(r.mMesh)};
    out->mPointMap = std::move(r.mPointMap);
    out->mCellMaps = std::move(r.mCellMaps);
    return out;
}

}  // namespace

mio_refine_result* mio_refine(const mio_mesh* mesh, int levels, int record_parent_ids) {
    return guarded_ptr(static_cast<mio_refine_result*>(nullptr), [&]() -> mio_refine_result* {
        meshioplusplus::RefineOptions options;
        options.mLevels = levels;
        options.mRecordParentIds = record_parent_ids != 0;
        return capi_refine(mesh, options);
    });
}

mio_refine_result* mio_refine_ex(const mio_mesh* mesh, const mio_refine_opts* opts) {
    return guarded_ptr(static_cast<mio_refine_result*>(nullptr), [&]() -> mio_refine_result* {
        if (opts == nullptr)
            return capi_refine(mesh, meshioplusplus::RefineOptions{});
        return capi_refine(mesh, capi_refine_options(*opts));
    });
}

const mio_mesh* mio_refine_result_mesh(const mio_refine_result* result) {
    return guarded_ptr(static_cast<const mio_mesh*>(nullptr), [&]() -> const mio_mesh* {
        if (!result)
            return nullptr;
        return &result->mMesh;
    });
}

mio_mesh* mio_refine_result_take_mesh(mio_refine_result* result) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return new mio_mesh{std::move(result->mMesh.mMesh)};
    });
}

mio_status mio_refine_result_point_map(const mio_refine_result* result, const void** data,
                                       mio_dtype* dtype, int64_t* n) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        const NDArray& a = result->mPointMap;
        if (data)
            *data = a.Data();
        if (dtype)
            *dtype = from_dtype(a.Dtype());
        if (n)
            *n = a.Shape().empty() ? 0 : static_cast<int64_t>(a.Shape()[0]);
        return MIO_OK;
    });
}

int64_t mio_refine_result_num_cell_maps(const mio_refine_result* result) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return static_cast<int64_t>(result->mCellMaps.size());
    });
}

mio_status mio_refine_result_cell_map(const mio_refine_result* result, int64_t block,
                                      const void** data, mio_dtype* dtype, int64_t* n) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        if (block < 0 || static_cast<std::size_t>(block) >= result->mCellMaps.size())
            return fail(MIO_ERR_NOT_FOUND, "meshio++: cell-map block index out of range");
        const NDArray& a = result->mCellMaps[static_cast<std::size_t>(block)];
        if (data)
            *data = a.Data();
        if (dtype)
            *dtype = from_dtype(a.Dtype());
        if (n)
            *n = a.Shape().empty() ? 0 : static_cast<int64_t>(a.Shape()[0]);
        return MIO_OK;
    });
}

void mio_refine_result_free(mio_refine_result* result) {
    delete result;
}

mio_decimate_result* mio_decimate(const mio_mesh* mesh, double target_ratio, int64_t target_faces,
                                  double max_error, const char* placement, int preserve_boundary,
                                  int preserve_features, double feature_angle) {
    return guarded_ptr(static_cast<mio_decimate_result*>(nullptr), [&]() -> mio_decimate_result* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        meshioplusplus::DecimateOptions options;
        options.mTargetRatio = target_ratio;
        options.mTargetFaces = target_faces;
        options.mMaxError = max_error;
        options.mPlacement =
            meshioplusplus::decimate_placement_from_name(placement ? placement : "optimal");
        options.mPreserveBoundary = preserve_boundary != 0;
        options.mPreserveFeatures = preserve_features != 0;
        options.mFeatureAngleDeg = feature_angle;
        meshioplusplus::DecimateResult r = meshioplusplus::decimate(mesh->mMesh, options);
        auto* out = new mio_decimate_result{};
        out->mMesh = mio_mesh{std::move(r.mMesh)};
        out->mPointMap = std::move(r.mPointMap);
        out->mCellMaps = std::move(r.mCellMaps);
        out->mFacesRemoved = r.mFacesRemoved;
        out->mPointsRemoved = r.mPointsRemoved;
        out->mCollapsesRejected = r.mCollapsesRejected;
        out->mMaxErrorApplied = r.mMaxErrorApplied;
        return out;
    });
}

const mio_mesh* mio_decimate_result_mesh(const mio_decimate_result* result) {
    return guarded_ptr(static_cast<const mio_mesh*>(nullptr), [&]() -> const mio_mesh* {
        if (!result)
            return nullptr;
        return &result->mMesh;
    });
}

mio_mesh* mio_decimate_result_take_mesh(mio_decimate_result* result) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return new mio_mesh{std::move(result->mMesh.mMesh)};
    });
}

mio_status mio_decimate_result_point_map(const mio_decimate_result* result, const void** data,
                                         mio_dtype* dtype, int64_t* n) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        const NDArray& a = result->mPointMap;
        if (data)
            *data = a.Data();
        if (dtype)
            *dtype = from_dtype(a.Dtype());
        if (n)
            *n = a.Shape().empty() ? 0 : static_cast<int64_t>(a.Shape()[0]);
        return MIO_OK;
    });
}

int64_t mio_decimate_result_num_cell_maps(const mio_decimate_result* result) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return static_cast<int64_t>(result->mCellMaps.size());
    });
}

mio_status mio_decimate_result_cell_map(const mio_decimate_result* result, int64_t block,
                                        const void** data, mio_dtype* dtype, int64_t* n) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        if (block < 0 || static_cast<std::size_t>(block) >= result->mCellMaps.size())
            return fail(MIO_ERR_NOT_FOUND, "meshio++: cell-map block index out of range");
        const NDArray& a = result->mCellMaps[static_cast<std::size_t>(block)];
        if (data)
            *data = a.Data();
        if (dtype)
            *dtype = from_dtype(a.Dtype());
        if (n)
            *n = a.Shape().empty() ? 0 : static_cast<int64_t>(a.Shape()[0]);
        return MIO_OK;
    });
}

int64_t mio_decimate_result_faces_removed(const mio_decimate_result* result) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return result->mFacesRemoved;
    });
}

int64_t mio_decimate_result_points_removed(const mio_decimate_result* result) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return result->mPointsRemoved;
    });
}

int64_t mio_decimate_result_collapses_rejected(const mio_decimate_result* result) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return result->mCollapsesRejected;
    });
}

double mio_decimate_result_max_error_applied(const mio_decimate_result* result) {
    return guarded_ptr(-1.0, [&]() -> double {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return result->mMaxErrorApplied;
    });
}

void mio_decimate_result_free(mio_decimate_result* result) {
    delete result;
}

mio_decimate_volume_result* mio_decimate_volume(const mio_mesh* mesh, double target_ratio,
                                                int64_t target_cells, double max_error,
                                                const char* placement, int preserve_boundary,
                                                int preserve_features, double feature_angle) {
    return guarded_ptr(
        static_cast<mio_decimate_volume_result*>(nullptr), [&]() -> mio_decimate_volume_result* {
            if (!mesh)
                throw meshioplusplus::ReadError("meshio++: mesh is NULL");
            meshioplusplus::DecimateVolumeOptions options;
            options.mTargetRatio = target_ratio;
            options.mTargetCells = target_cells;
            options.mMaxError = max_error;
            options.mPlacement =
                meshioplusplus::decimate_placement_from_name(placement ? placement : "optimal");
            options.mPreserveBoundary = preserve_boundary != 0;
            options.mPreserveFeatures = preserve_features != 0;
            options.mFeatureAngleDeg = feature_angle;
            meshioplusplus::DecimateVolumeResult r =
                meshioplusplus::decimate_volume(mesh->mMesh, options);
            auto* out = new mio_decimate_volume_result{};
            out->mMesh = mio_mesh{std::move(r.mMesh)};
            out->mPointMap = std::move(r.mPointMap);
            out->mCellMaps = std::move(r.mCellMaps);
            out->mTetsRemoved = r.mTetsRemoved;
            out->mPointsRemoved = r.mPointsRemoved;
            out->mCollapsesRejected = r.mCollapsesRejected;
            out->mMaxErrorApplied = r.mMaxErrorApplied;
            return out;
        });
}

const mio_mesh* mio_decimate_volume_result_mesh(const mio_decimate_volume_result* result) {
    return guarded_ptr(static_cast<const mio_mesh*>(nullptr), [&]() -> const mio_mesh* {
        if (!result)
            return nullptr;
        return &result->mMesh;
    });
}

mio_mesh* mio_decimate_volume_result_take_mesh(mio_decimate_volume_result* result) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return new mio_mesh{std::move(result->mMesh.mMesh)};
    });
}

mio_status mio_decimate_volume_result_point_map(const mio_decimate_volume_result* result,
                                                const void** data, mio_dtype* dtype, int64_t* n) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        const NDArray& a = result->mPointMap;
        if (data)
            *data = a.Data();
        if (dtype)
            *dtype = from_dtype(a.Dtype());
        if (n)
            *n = a.Shape().empty() ? 0 : static_cast<int64_t>(a.Shape()[0]);
        return MIO_OK;
    });
}

int64_t mio_decimate_volume_result_num_cell_maps(const mio_decimate_volume_result* result) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return static_cast<int64_t>(result->mCellMaps.size());
    });
}

mio_status mio_decimate_volume_result_cell_map(const mio_decimate_volume_result* result,
                                               int64_t block, const void** data, mio_dtype* dtype,
                                               int64_t* n) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        if (block < 0 || static_cast<std::size_t>(block) >= result->mCellMaps.size())
            return fail(MIO_ERR_NOT_FOUND, "meshio++: cell-map block index out of range");
        const NDArray& a = result->mCellMaps[static_cast<std::size_t>(block)];
        if (data)
            *data = a.Data();
        if (dtype)
            *dtype = from_dtype(a.Dtype());
        if (n)
            *n = a.Shape().empty() ? 0 : static_cast<int64_t>(a.Shape()[0]);
        return MIO_OK;
    });
}

int64_t mio_decimate_volume_result_tets_removed(const mio_decimate_volume_result* result) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return result->mTetsRemoved;
    });
}

int64_t mio_decimate_volume_result_points_removed(const mio_decimate_volume_result* result) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return result->mPointsRemoved;
    });
}

int64_t mio_decimate_volume_result_collapses_rejected(const mio_decimate_volume_result* result) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return result->mCollapsesRejected;
    });
}

double mio_decimate_volume_result_max_error_applied(const mio_decimate_volume_result* result) {
    return guarded_ptr(-1.0, [&]() -> double {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return result->mMaxErrorApplied;
    });
}

void mio_decimate_volume_result_free(mio_decimate_volume_result* result) {
    delete result;
}

namespace {

// Shared option assembly for mio_partition / mio_partition_labels. NULL strings
// mean "auto" / "eco" / unweighted (the *_from_name parsers accept "").
meshioplusplus::PartitionOptions partition_options_from_c(int nparts, const char* method,
                                                          double imbalance, const char* mode,
                                                          int seed, int record_ids,
                                                          int ghost_layers,
                                                          const char* weights_key) {
    meshioplusplus::PartitionOptions options;
    options.mNParts = nparts;
    options.mMethod = meshioplusplus::partition_method_from_name(method ? method : "");
    options.mImbalance = imbalance;
    options.mMode = meshioplusplus::partition_mode_from_name(mode ? mode : "");
    options.mSeed = seed;
    options.mRecordIds = record_ids != 0;
    options.mGhostLayers = ghost_layers;
    options.mWeightsKey = weights_key ? weights_key : "";
    return options;
}

}  // namespace

mio_partition_result* mio_partition(const mio_mesh* mesh, int nparts, const char* method,
                                    double imbalance, const char* mode, int seed, int record_ids,
                                    int ghost_layers, const char* weights_key) {
    return guarded_ptr(static_cast<mio_partition_result*>(nullptr), [&]() -> mio_partition_result* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        meshioplusplus::PartitionOptions options = partition_options_from_c(
            nparts, method, imbalance, mode, seed, record_ids, ghost_layers, weights_key);
        meshioplusplus::PartitionResult r = meshioplusplus::partition(mesh->mMesh, options);
        auto* out = new mio_partition_result{};
        out->mPieces.reserve(r.mPieces.size());
        for (meshioplusplus::PartitionPiece& p : r.mPieces) {
            mio_partition_result::Piece piece;
            piece.mMesh = mio_mesh{std::move(p.mMesh)};
            piece.mPointMap = std::move(p.mPointMap);
            piece.mCellMaps = std::move(p.mCellMaps);
            out->mPieces.push_back(std::move(piece));
        }
        return out;
    });
}

int64_t mio_partition_result_num_pieces(const mio_partition_result* result) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return static_cast<int64_t>(result->mPieces.size());
    });
}

int mio_partition_result_part_id(const mio_partition_result* result, int64_t index) {
    return guarded_ptr(-1, [&]() -> int {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= result->mPieces.size())
            throw meshioplusplus::ReadError("meshio++: piece index out of range");
        return static_cast<int>(index);
    });
}

const mio_mesh* mio_partition_result_mesh(const mio_partition_result* result, int64_t index) {
    return guarded_ptr(static_cast<const mio_mesh*>(nullptr), [&]() -> const mio_mesh* {
        if (!result)
            return nullptr;
        if (index < 0 || static_cast<std::size_t>(index) >= result->mPieces.size())
            return nullptr;
        return &result->mPieces[static_cast<std::size_t>(index)].mMesh;
    });
}

mio_mesh* mio_partition_result_take_mesh(mio_partition_result* result, int64_t index) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= result->mPieces.size())
            throw meshioplusplus::ReadError("meshio++: piece index out of range");
        return new mio_mesh{
            std::move(result->mPieces[static_cast<std::size_t>(index)].mMesh.mMesh)};
    });
}

mio_status mio_partition_result_point_map(const mio_partition_result* result, int64_t index,
                                          const void** data, mio_dtype* dtype, int64_t* n) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= result->mPieces.size())
            return fail(MIO_ERR_NOT_FOUND, "meshio++: piece index out of range");
        const NDArray& a = result->mPieces[static_cast<std::size_t>(index)].mPointMap;
        if (data)
            *data = a.Data();
        if (dtype)
            *dtype = from_dtype(a.Dtype());
        if (n)
            *n = a.Shape().empty() ? 0 : static_cast<int64_t>(a.Shape()[0]);
        return MIO_OK;
    });
}

int64_t mio_partition_result_num_cell_maps(const mio_partition_result* result) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        if (result->mPieces.empty())
            return 0;
        return static_cast<int64_t>(result->mPieces.front().mCellMaps.size());
    });
}

mio_status mio_partition_result_cell_map(const mio_partition_result* result, int64_t index,
                                         int64_t block, const void** data, mio_dtype* dtype,
                                         int64_t* n) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= result->mPieces.size())
            return fail(MIO_ERR_NOT_FOUND, "meshio++: piece index out of range");
        const mio_partition_result::Piece& piece = result->mPieces[static_cast<std::size_t>(index)];
        if (block < 0 || static_cast<std::size_t>(block) >= piece.mCellMaps.size())
            return fail(MIO_ERR_NOT_FOUND, "meshio++: cell-map block index out of range");
        const NDArray& a = piece.mCellMaps[static_cast<std::size_t>(block)];
        if (data)
            *data = a.Data();
        if (dtype)
            *dtype = from_dtype(a.Dtype());
        if (n)
            *n = a.Shape().empty() ? 0 : static_cast<int64_t>(a.Shape()[0]);
        return MIO_OK;
    });
}

void mio_partition_result_free(mio_partition_result* result) {
    delete result;
}

mio_status mio_partition_labels(const mio_mesh* mesh, int nparts, const char* method,
                                double imbalance, const char* mode, int seed,
                                const char* weights_key, int64_t* labels, int64_t labels_size) {
    return guarded([&]() -> mio_status {
        if (!mesh)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: mesh is NULL");
        if (!labels)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: labels is NULL");
        meshioplusplus::PartitionOptions options =
            partition_options_from_c(nparts, method, imbalance, mode, seed, /*record_ids=*/0,
                                     /*ghost_layers=*/0, weights_key);
        std::vector<NDArray> per_block = meshioplusplus::partition_labels(mesh->mMesh, options);
        int64_t total = 0;
        for (const NDArray& a : per_block)
            total += a.Shape().empty() ? 0 : static_cast<int64_t>(a.Shape()[0]);
        if (labels_size != total)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: labels_size (" +
                                                 std::to_string(labels_size) +
                                                 ") does not equal the mesh's total cell count (" +
                                                 std::to_string(total) + ")");
        int64_t offset = 0;
        for (const NDArray& a : per_block) {
            const int64_t rows = a.Shape().empty() ? 0 : static_cast<int64_t>(a.Shape()[0]);
            std::memcpy(labels + offset, a.Data(),
                        static_cast<std::size_t>(rows) * sizeof(int64_t));
            offset += rows;
        }
        return MIO_OK;
    });
}

mio_status mio_stats(const mio_mesh* mesh, mio_stats_report* out) {
    return guarded([&]() -> mio_status {
        if (!mesh)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: mesh is NULL");
        if (!out)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: out is NULL");
        meshioplusplus::StatsReport r = meshioplusplus::compute_stats(mesh->mMesh);
        out->num_points = r.mNumPoints;
        out->num_cells = r.mNumCells;
        for (int d = 0; d < 3; ++d) {
            out->bbox_min[d] = r.mBBoxMin[d];
            out->bbox_max[d] = r.mBBoxMax[d];
            out->extent[d] = r.mExtent[d];
            out->centroid[d] = r.mCentroid[d];
        }
        out->total_area = r.mTotalArea;
        out->signed_volume = r.mSignedVolume;
        out->unsigned_volume = r.mUnsignedVolume;
        out->num_inverted = r.mNumInverted;
        return MIO_OK;
    });
}

/* --- data operations ---------------------------------------------------- */

/* The C enums duplicate the C++ ones, so pin every value: drift becomes a
 * compile error rather than a silently wrong argument (the same discipline the
 * MIO_CELL_TYPES X-macro applies to the cell-type list). */
static_assert(static_cast<int>(meshioplusplus::DataLocation::Point) == MIO_DATA_POINT, "");
static_assert(static_cast<int>(meshioplusplus::DataLocation::Cell) == MIO_DATA_CELL, "");
static_assert(static_cast<int>(meshioplusplus::DataLocation::Field) == MIO_DATA_FIELD, "");
static_assert(static_cast<int>(meshioplusplus::CellPointWeight::Uniform) == MIO_WEIGHT_UNIFORM, "");
static_assert(static_cast<int>(meshioplusplus::CellPointWeight::Measure) == MIO_WEIGHT_MEASURE, "");
static_assert(static_cast<int>(meshioplusplus::ConditionMode::Clamp) == MIO_COND_CLAMP, "");
static_assert(static_cast<int>(meshioplusplus::ConditionMode::Normalize) == MIO_COND_NORMALIZE, "");
static_assert(static_cast<int>(meshioplusplus::ConditionMode::Standardize) == MIO_COND_STANDARDIZE,
              "");
static_assert(static_cast<int>(meshioplusplus::ConditionScope::Component) == MIO_SCOPE_COMPONENT,
              "");
static_assert(static_cast<int>(meshioplusplus::ConditionScope::Magnitude) == MIO_SCOPE_MAGNITUDE,
              "");
static_assert(static_cast<int>(meshioplusplus::NanPolicy::Ignore) == MIO_NAN_IGNORE, "");
static_assert(static_cast<int>(meshioplusplus::NanPolicy::Replace) == MIO_NAN_REPLACE, "");
static_assert(static_cast<int>(meshioplusplus::NanPolicy::Fail) == MIO_NAN_FAIL, "");

namespace {

/* Turn the (names, count) pair the ABI uses into a vector, rejecting a NULL
 * entry rather than dereferencing it. */
std::vector<std::string> data_name_list(const char* const* pNames, int64_t count) {
    if (count < 0)
        throw meshioplusplus::ReadError("meshio++: name count is negative");
    if (count > 0 && !pNames)
        throw meshioplusplus::ReadError("meshio++: names is NULL but count > 0");
    std::vector<std::string> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        if (!pNames[i])
            throw meshioplusplus::ReadError("meshio++: names[" + std::to_string(i) + "] is NULL");
        out.emplace_back(pNames[i]);
    }
    return out;
}

meshioplusplus::DataLocation data_location_of(mio_data_location loc) {
    switch (loc) {
        case MIO_DATA_POINT:
            return meshioplusplus::DataLocation::Point;
        case MIO_DATA_CELL:
            return meshioplusplus::DataLocation::Cell;
        case MIO_DATA_FIELD:
            return meshioplusplus::DataLocation::Field;
    }
    throw meshioplusplus::ReadError("meshio++: invalid data location");
}

}  // namespace

mio_mesh* mio_data_drop(const mio_mesh* mesh, mio_data_location location, const char* const* names,
                        int64_t count, int ignore_missing) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        return new mio_mesh{meshioplusplus::data_drop(mesh->mMesh, data_location_of(location),
                                                      data_name_list(names, count),
                                                      ignore_missing != 0)};
    });
}

mio_mesh* mio_data_keep(const mio_mesh* mesh, mio_data_location location, const char* const* names,
                        int64_t count, int ignore_missing) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        return new mio_mesh{meshioplusplus::data_keep(mesh->mMesh, data_location_of(location),
                                                      data_name_list(names, count),
                                                      ignore_missing != 0)};
    });
}

mio_mesh* mio_data_rename(const mio_mesh* mesh, mio_data_location location, const char* from_name,
                          const char* to_name) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh || !from_name || !to_name)
            throw meshioplusplus::ReadError("meshio++: mesh/from_name/to_name is NULL");
        return new mio_mesh{meshioplusplus::data_rename(mesh->mMesh, data_location_of(location),
                                                        from_name, to_name)};
    });
}

mio_mesh* mio_data_point_to_cell(const mio_mesh* mesh, const char* const* names, int64_t count,
                                 const char* suffix) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        meshioplusplus::DataAverageOptions opts;
        opts.names = data_name_list(names, count);
        opts.suffix = suffix ? suffix : "";
        return new mio_mesh{meshioplusplus::point_data_to_cell_data(mesh->mMesh, opts)};
    });
}

mio_mesh* mio_data_cell_to_point(const mio_mesh* mesh, const char* const* names, int64_t count,
                                 mio_cell_point_weight weight, const char* suffix) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        meshioplusplus::DataAverageOptions opts;
        opts.names = data_name_list(names, count);
        opts.weight = weight == MIO_WEIGHT_MEASURE ? meshioplusplus::CellPointWeight::Measure
                                                   : meshioplusplus::CellPointWeight::Uniform;
        opts.suffix = suffix ? suffix : "";
        return new mio_mesh{meshioplusplus::cell_data_to_point_data(mesh->mMesh, opts)};
    });
}

mio_mesh* mio_data_calc(const mio_mesh* mesh, const char* expression, mio_data_location location,
                        const char* output_name, int overwrite) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh || !expression || !output_name)
            throw meshioplusplus::ReadError("meshio++: mesh/expression/output_name is NULL");
        meshioplusplus::DataCalcOptions opts;
        opts.location = data_location_of(location);
        opts.output = output_name;
        opts.overwrite = overwrite != 0;
        return new mio_mesh{meshioplusplus::data_calc(mesh->mMesh, expression, opts)};
    });
}

mio_mesh* mio_data_condition(const mio_mesh* mesh, mio_data_location location,
                             const char* const* names, int64_t count, mio_condition_mode mode,
                             double lo, double hi, mio_condition_scope scope,
                             mio_nan_policy nan_policy, double nan_replacement,
                             const char* suffix) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        meshioplusplus::DataConditionOptions opts;
        opts.location = data_location_of(location);
        opts.names = data_name_list(names, count);
        opts.mode = static_cast<meshioplusplus::ConditionMode>(mode);
        opts.scope = static_cast<meshioplusplus::ConditionScope>(scope);
        opts.lo = lo;
        opts.hi = hi;
        opts.nan_policy = static_cast<meshioplusplus::NanPolicy>(nan_policy);
        opts.nan_replacement = nan_replacement;
        opts.suffix = suffix ? suffix : "";
        return new mio_mesh{meshioplusplus::data_condition(mesh->mMesh, opts)};
    });
}

mio_data_info* mio_data_info_create(const mio_mesh* mesh) {
    return guarded_ptr(static_cast<mio_data_info*>(nullptr), [&]() -> mio_data_info* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        return new mio_data_info{meshioplusplus::data_info(mesh->mMesh)};
    });
}

int64_t mio_data_info_count(const mio_data_info* info) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!info)
            throw meshioplusplus::ReadError("meshio++: info is NULL");
        return static_cast<int64_t>(info->mReport.mArrays.size());
    });
}

int64_t mio_data_info_name(const mio_data_info* info, int64_t index, char* buf, int64_t buflen) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!info)
            throw meshioplusplus::ReadError("meshio++: info is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= info->mReport.mArrays.size())
            throw meshioplusplus::ReadError("meshio++: data array index out of range");
        return copy_string(info->mReport.mArrays[static_cast<std::size_t>(index)].mName, buf,
                           buflen);
    });
}

mio_status mio_data_info_entry(const mio_data_info* info, int64_t index, mio_data_array_info* out) {
    return guarded([&]() -> mio_status {
        if (!info)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: info is NULL");
        if (!out)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: out is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= info->mReport.mArrays.size())
            return fail(MIO_ERR_INVALID_ARG, "meshio++: data array index out of range");
        const meshioplusplus::DataArrayInfo& a =
            info->mReport.mArrays[static_cast<std::size_t>(index)];
        out->location = static_cast<int>(a.mLocation);
        out->dtype = static_cast<int>(a.mDtype);
        out->num_blocks = a.mNumBlocks;
        out->num_entries = a.mNumEntries;
        out->num_components = a.mNumComponents;
        out->num_values = a.mNumValues;
        out->min = a.mMin;
        out->max = a.mMax;
        out->mean = a.mMean;
        out->num_nan = a.mNumNan;
        out->num_inf = a.mNumInf;
        out->num_finite = a.mNumFinite;
        out->inconsistent_blocks = a.mInconsistentBlocks ? 1 : 0;
        return MIO_OK;
    });
}

mio_status mio_data_info_component(const mio_data_info* info, int64_t index, int64_t comp,
                                   double* min, double* max, double* mean) {
    return guarded([&]() -> mio_status {
        if (!info)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: info is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= info->mReport.mArrays.size())
            return fail(MIO_ERR_INVALID_ARG, "meshio++: data array index out of range");
        const meshioplusplus::DataArrayInfo& a =
            info->mReport.mArrays[static_cast<std::size_t>(index)];
        if (comp < 0 || static_cast<std::size_t>(comp) >= a.mMinPerComponent.size())
            return fail(MIO_ERR_INVALID_ARG, "meshio++: component index out of range");
        const std::size_t k = static_cast<std::size_t>(comp);
        if (min)
            *min = a.mMinPerComponent[k];
        if (max)
            *max = a.mMaxPerComponent[k];
        if (mean)
            *mean = a.mMeanPerComponent[k];
        return MIO_OK;
    });
}

void mio_data_info_free(mio_data_info* info) {
    delete info;
}

mio_data_integrate* mio_data_integrate_create(const mio_mesh* mesh, const char* const* names,
                                              int64_t count) {
    return guarded_ptr(static_cast<mio_data_integrate*>(nullptr), [&]() -> mio_data_integrate* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        meshioplusplus::DataIntegrateOptions opts;
        if (names) {
            for (int64_t i = 0; i < count; ++i)
                opts.mArrayNames.emplace_back(names[i] ? names[i] : "");
        }
        return new mio_data_integrate{meshioplusplus::data_integrate(mesh->mMesh, opts)};
    });
}

int64_t mio_data_integrate_count(const mio_data_integrate* result) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return static_cast<int64_t>(result->mReport.mArrays.size());
    });
}

int64_t mio_data_integrate_name(const mio_data_integrate* result, int64_t index, char* buf,
                                int64_t buflen) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= result->mReport.mArrays.size())
            throw meshioplusplus::ReadError("meshio++: array index out of range");
        return copy_string(result->mReport.mArrays[static_cast<std::size_t>(index)].mName, buf,
                           buflen);
    });
}

namespace {
mio_status fill_field_integral_info(const meshioplusplus::FieldIntegralRegion& rRegion,
                                    mio_field_integral_info* out) {
    if (!out)
        return fail(MIO_ERR_INVALID_ARG, "meshio++: out is NULL");
    out->num_components = static_cast<int64_t>(rRegion.mTotalPerComponent.size());
    out->num_cells = rRegion.mNumCells;
    out->num_skipped = rRegion.mNumSkipped;
    return MIO_OK;
}

mio_status fill_field_integral_component(const meshioplusplus::FieldIntegralRegion& rRegion,
                                         int64_t comp, double* total, double* mean,
                                         double* domain_measure, int64_t* num_nan) {
    if (comp < 0 || static_cast<std::size_t>(comp) >= rRegion.mTotalPerComponent.size())
        return fail(MIO_ERR_INVALID_ARG, "meshio++: component index out of range");
    const std::size_t k = static_cast<std::size_t>(comp);
    if (total)
        *total = rRegion.mTotalPerComponent[k];
    if (mean)
        *mean = rRegion.mMeanPerComponent[k];
    if (domain_measure)
        *domain_measure = rRegion.mDomainMeasurePerComponent[k];
    if (num_nan)
        *num_nan = rRegion.mNumNanPerComponent[k];
    return MIO_OK;
}
}  // namespace

mio_status mio_data_integrate_entry(const mio_data_integrate* result, int64_t index,
                                    mio_field_integral_info* out) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= result->mReport.mArrays.size())
            return fail(MIO_ERR_INVALID_ARG, "meshio++: array index out of range");
        return fill_field_integral_info(
            result->mReport.mArrays[static_cast<std::size_t>(index)].mDomain, out);
    });
}

mio_status mio_data_integrate_component(const mio_data_integrate* result, int64_t index,
                                        int64_t comp, double* total, double* mean,
                                        double* domain_measure, int64_t* num_nan) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= result->mReport.mArrays.size())
            return fail(MIO_ERR_INVALID_ARG, "meshio++: array index out of range");
        return fill_field_integral_component(
            result->mReport.mArrays[static_cast<std::size_t>(index)].mDomain, comp, total, mean,
            domain_measure, num_nan);
    });
}

int64_t mio_data_integrate_region_count(const mio_data_integrate* result, int64_t index) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= result->mReport.mArrays.size())
            throw meshioplusplus::ReadError("meshio++: array index out of range");
        return static_cast<int64_t>(
            result->mReport.mArrays[static_cast<std::size_t>(index)].mRegions.size());
    });
}

int64_t mio_data_integrate_region_name(const mio_data_integrate* result, int64_t index,
                                       int64_t region, char* buf, int64_t buflen) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= result->mReport.mArrays.size())
            throw meshioplusplus::ReadError("meshio++: array index out of range");
        const auto& regions = result->mReport.mArrays[static_cast<std::size_t>(index)].mRegions;
        if (region < 0 || static_cast<std::size_t>(region) >= regions.size())
            throw meshioplusplus::ReadError("meshio++: region index out of range");
        return copy_string(regions[static_cast<std::size_t>(region)].mName, buf, buflen);
    });
}

mio_status mio_data_integrate_region_entry(const mio_data_integrate* result, int64_t index,
                                           int64_t region, mio_field_integral_info* out) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= result->mReport.mArrays.size())
            return fail(MIO_ERR_INVALID_ARG, "meshio++: array index out of range");
        const auto& regions = result->mReport.mArrays[static_cast<std::size_t>(index)].mRegions;
        if (region < 0 || static_cast<std::size_t>(region) >= regions.size())
            return fail(MIO_ERR_INVALID_ARG, "meshio++: region index out of range");
        return fill_field_integral_info(regions[static_cast<std::size_t>(region)], out);
    });
}

mio_status mio_data_integrate_region_component(const mio_data_integrate* result, int64_t index,
                                               int64_t region, int64_t comp, double* total,
                                               double* mean, double* domain_measure,
                                               int64_t* num_nan) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= result->mReport.mArrays.size())
            return fail(MIO_ERR_INVALID_ARG, "meshio++: array index out of range");
        const auto& regions = result->mReport.mArrays[static_cast<std::size_t>(index)].mRegions;
        if (region < 0 || static_cast<std::size_t>(region) >= regions.size())
            return fail(MIO_ERR_INVALID_ARG, "meshio++: region index out of range");
        return fill_field_integral_component(regions[static_cast<std::size_t>(region)], comp, total,
                                             mean, domain_measure, num_nan);
    });
}

void mio_data_integrate_free(mio_data_integrate* result) {
    delete result;
}

mio_status mio_quality_counts(const mio_mesh* mesh, int64_t* num_cells, int64_t* num_inverted,
                              int64_t* num_degenerate) {
    return guarded([&]() -> mio_status {
        if (!mesh)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: mesh is NULL");
        meshioplusplus::QualityReport rep = meshioplusplus::compute_quality(mesh->mMesh);
        if (num_cells)
            *num_cells = rep.mNumCells;
        if (num_inverted)
            *num_inverted = rep.mNumInverted;
        if (num_degenerate)
            *num_degenerate = rep.mNumDegenerate;
        return MIO_OK;
    });
}

int64_t mio_sniff_format(const char* path, char* buf, int64_t buflen) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!path)
            throw meshioplusplus::ReadError("meshio++: path is NULL");
        return copy_string(meshioplusplus::sniff_format(path), buf, buflen);
    });
}

mio_reorder_result* mio_reorder(const mio_mesh* mesh, const char* method) {
    return guarded_ptr(static_cast<mio_reorder_result*>(nullptr), [&]() -> mio_reorder_result* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        if (!method)
            throw meshioplusplus::ReadError("meshio++: method is NULL");
        meshioplusplus::ReorderResult res =
            meshioplusplus::reorder(mesh->mMesh, meshioplusplus::reorder_method_from_name(method));
        auto* out =
            new mio_reorder_result{mio_mesh{std::move(res.mMesh)}, std::move(res.mNodePermutation),
                                   std::move(res.mCellPermutations)};
        return out;
    });
}

const mio_mesh* mio_reorder_result_mesh(const mio_reorder_result* result) {
    return result ? &result->mMesh : nullptr;
}

mio_status mio_reorder_result_node_perm(const mio_reorder_result* result, const void** data,
                                        mio_dtype* dtype, int64_t* n) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        const NDArray& a = result->mNodePerm;
        if (data)
            *data = a.Data();
        if (dtype)
            *dtype = from_dtype(a.Dtype());
        if (n)
            *n = a.Shape().empty() ? 0 : static_cast<int64_t>(a.Shape()[0]);
        return MIO_OK;
    });
}

int64_t mio_reorder_result_num_cell_perms(const mio_reorder_result* result) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return static_cast<int64_t>(result->mCellPerms.size());
    });
}

mio_status mio_reorder_result_cell_perm(const mio_reorder_result* result, int64_t block,
                                        const void** data, mio_dtype* dtype, int64_t* n) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        if (block < 0 || static_cast<std::size_t>(block) >= result->mCellPerms.size())
            return fail(MIO_ERR_NOT_FOUND, "meshio++: cell-permutation block index out of range");
        const NDArray& a = result->mCellPerms[static_cast<std::size_t>(block)];
        if (data)
            *data = a.Data();
        if (dtype)
            *dtype = from_dtype(a.Dtype());
        if (n)
            *n = a.Shape().empty() ? 0 : static_cast<int64_t>(a.Shape()[0]);
        return MIO_OK;
    });
}

mio_mesh* mio_reorder_result_take_mesh(mio_reorder_result* result) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return new mio_mesh{std::move(result->mMesh.mMesh)};
    });
}

void mio_reorder_result_free(mio_reorder_result* result) {
    delete result;
}

int64_t mio_compute_bandwidth(const mio_mesh* mesh) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        return meshioplusplus::compute_bandwidth(mesh->mMesh);
    });
}

mio_status mio_diff(const mio_mesh* a, const mio_mesh* b, double atol, double rtol, int unordered,
                    mio_diff_result** out) {
    return guarded([&]() -> mio_status {
        if (!a || !b)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: mesh is NULL");
        if (!out)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: out is NULL");
        meshioplusplus::DiffOptions opts;
        opts.atol = atol;
        opts.rtol = rtol;
        opts.unordered = unordered != 0;
        *out = new mio_diff_result{meshioplusplus::diff(a->mMesh, b->mMesh, opts)};
        return MIO_OK;
    });
}

mio_status mio_meshes_equal(const mio_mesh* a, const mio_mesh* b, double atol, double rtol,
                            int unordered, int* out_equal) {
    return guarded([&]() -> mio_status {
        if (!a || !b)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: mesh is NULL");
        meshioplusplus::DiffOptions opts;
        opts.atol = atol;
        opts.rtol = rtol;
        opts.unordered = unordered != 0;
        const bool eq = meshioplusplus::diff(a->mMesh, b->mMesh, opts).mVerdict !=
                        meshioplusplus::DiffVerdict::Different;
        if (out_equal)
            *out_equal = eq ? 1 : 0;
        return MIO_OK;
    });
}

mio_diff_verdict mio_diff_result_verdict(const mio_diff_result* result) {
    if (!result)
        return MIO_DIFF_DIFFERENT;
    switch (result->mReport.mVerdict) {
        case meshioplusplus::DiffVerdict::Identical:
            return MIO_DIFF_IDENTICAL;
        case meshioplusplus::DiffVerdict::EqualWithinTolerance:
            return MIO_DIFF_EQUAL_WITHIN_TOLERANCE;
        case meshioplusplus::DiffVerdict::Different:
            return MIO_DIFF_DIFFERENT;
    }
    return MIO_DIFF_DIFFERENT;
}

mio_status mio_diff_result_point_summary(const mio_diff_result* result, double* max_abs_error,
                                         double* max_rel_error, int64_t* worst_index,
                                         int64_t* num_exceeding) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        const meshioplusplus::ArrayDiff& p = result->mReport.mPoints;
        if (max_abs_error)
            *max_abs_error = p.mMaxAbsError;
        if (max_rel_error)
            *max_rel_error = p.mMaxRelError;
        if (worst_index)
            *worst_index = p.mWorstIndex;
        if (num_exceeding)
            *num_exceeding = p.mNumExceeding;
        return MIO_OK;
    });
}

int64_t mio_diff_result_num_block_diffs(const mio_diff_result* result) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!result)
            throw meshioplusplus::ReadError("meshio++: result is NULL");
        return static_cast<int64_t>(result->mReport.mBlocks.size());
    });
}

mio_status mio_diff_result_block(const mio_diff_result* result, int64_t block, int* type_mismatch,
                                 int* count_mismatch, int64_t* conn_mismatch_count) {
    return guarded([&]() -> mio_status {
        if (!result)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: result is NULL");
        if (block < 0 || static_cast<std::size_t>(block) >= result->mReport.mBlocks.size())
            return fail(MIO_ERR_NOT_FOUND, "meshio++: block index out of range");
        const meshioplusplus::BlockDiff& bd =
            result->mReport.mBlocks[static_cast<std::size_t>(block)];
        if (type_mismatch)
            *type_mismatch = bd.mTypeMismatch ? 1 : 0;
        if (count_mismatch)
            *count_mismatch = bd.mCountMismatch ? 1 : 0;
        if (conn_mismatch_count)
            *conn_mismatch_count = bd.mConnMismatchCount;
        return MIO_OK;
    });
}

void mio_diff_result_free(mio_diff_result* result) {
    delete result;
}

/* ------------------------------------------------------------------ */
/* Building a mesh (setters copy)                                      */
/* ------------------------------------------------------------------ */

mio_status mio_mesh_set_points(mio_mesh* mesh, mio_dtype dtype, int64_t num_points, int64_t dim,
                               const void* xyz) {
    return guarded([&]() -> mio_status {
        DType dt;
        if (!mesh || num_points < 0 || dim <= 0 || (!xyz && num_points > 0))
            return fail(MIO_ERR_INVALID_ARG, "meshio++: bad mio_mesh_set_points argument");
        if (!to_dtype(dtype, dt) || (dt != DType::Float32 && dt != DType::Float64))
            return fail(MIO_ERR_INVALID_ARG,
                        "meshio++: points dtype must be MIO_FLOAT32 or MIO_FLOAT64");
        mesh->mMesh.AssignPoints(copy_in(
            dt, {static_cast<std::size_t>(num_points), static_cast<std::size_t>(dim)}, xyz));
        return MIO_OK;
    });
}

mio_status mio_mesh_add_cell_block(mio_mesh* mesh, const char* cell_type, int64_t num_cells,
                                   int64_t nodes_per_cell, mio_dtype dtype,
                                   const void* connectivity) {
    return guarded([&]() -> mio_status {
        if (!mesh || !cell_type || num_cells < 0 || nodes_per_cell <= 0 ||
            (!connectivity && num_cells > 0))
            return fail(MIO_ERR_INVALID_ARG, "meshio++: bad mio_mesh_add_cell_block argument");
        if (dtype != MIO_INT32 && dtype != MIO_INT64)
            return fail(MIO_ERR_INVALID_ARG,
                        "meshio++: connectivity dtype must be MIO_INT32 or MIO_INT64");
        const auto ct = meshioplusplus::cell_type_from_name(cell_type);
        const int fixed = meshioplusplus::cell_type_num_nodes(ct);
        if (ct != meshioplusplus::CellType::Custom && fixed > 0 && fixed != nodes_per_cell)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: cell type '" + std::string(cell_type) +
                                                 "' has " + std::to_string(fixed) +
                                                 " nodes per cell, got " +
                                                 std::to_string(nodes_per_cell));
        // Widen to Int64 during the copy -- the core's connectivity type.
        NDArray conn = NDArray::Uninit(DType::Int64, {static_cast<std::size_t>(num_cells),
                                                      static_cast<std::size_t>(nodes_per_cell)});
        const std::size_t n = conn.Size();
        if (dtype == MIO_INT64) {
            if (n > 0)
                std::memcpy(conn.Data(), connectivity, conn.Nbytes());
        } else {
            const auto* src = static_cast<const std::int32_t*>(connectivity);
            std::int64_t* dst = conn.As<std::int64_t>();
            for (std::size_t i = 0; i < n; ++i)
                dst[i] = src[i];
        }
        mesh->mMesh.AddCellBlock(cell_type, std::move(conn));
        return MIO_OK;
    });
}

namespace {

// Validate one CSR offsets array of `count + 1` entries spanning [0, total].
// A malformed offsets array is the one way a caller can make the ragged setters
// read out of bounds, so every property is checked rather than assumed.
mio_status poly_check_offsets(const std::int64_t* pOffsets, std::int64_t Count, std::int64_t Total,
                              const char* pWhat) {
    if (!pOffsets)
        return fail(MIO_ERR_INVALID_ARG, "meshio++: " + std::string(pWhat) + " is NULL (expected " +
                                             std::to_string(Count + 1) + " entries)");
    if (pOffsets[0] != 0)
        return fail(MIO_ERR_INVALID_ARG, "meshio++: " + std::string(pWhat) + "[0] must be 0, got " +
                                             std::to_string(pOffsets[0]));
    for (std::int64_t i = 0; i < Count; ++i)
        if (pOffsets[i + 1] < pOffsets[i])
            return fail(MIO_ERR_INVALID_ARG,
                        "meshio++: " + std::string(pWhat) + " must be non-decreasing (entry " +
                            std::to_string(i + 1) + " is " + std::to_string(pOffsets[i + 1]) +
                            " after " + std::to_string(pOffsets[i]) + ")");
    if (pOffsets[Count] != Total)
        return fail(MIO_ERR_INVALID_ARG,
                    "meshio++: " + std::string(pWhat) + "[" + std::to_string(Count) + "] must be " +
                        std::to_string(Total) + ", got " + std::to_string(pOffsets[Count]));
    return MIO_OK;
}

// Node ids are checked non-negative but deliberately NOT range-checked against
// NumPoints(): points may legitimately be assigned after cells, and no other
// setter on this ABI range-checks either.
mio_status poly_check_nodes(const std::int64_t* pNodes, std::int64_t Count) {
    if (Count > 0 && !pNodes)
        return fail(MIO_ERR_INVALID_ARG, "meshio++: nodes is NULL");
    for (std::int64_t i = 0; i < Count; ++i)
        if (pNodes[i] < 0)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: node id at index " + std::to_string(i) +
                                                 " is negative (" + std::to_string(pNodes[i]) +
                                                 "); connectivity is 0-based");
    return MIO_OK;
}

}  // namespace

mio_status mio_mesh_add_polygon_block(mio_mesh* mesh, const char* cell_type, int64_t num_cells,
                                      const int64_t* row_offsets, const int64_t* nodes,
                                      int64_t num_nodes) {
    return guarded([&]() -> mio_status {
        if (!mesh || !cell_type || num_cells < 0 || num_nodes < 0)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: bad mio_mesh_add_polygon_block argument");
        if (mio_status s = poly_check_offsets(row_offsets, num_cells, num_nodes, "row_offsets");
            s != MIO_OK)
            return s;
        if (mio_status s = poly_check_nodes(nodes, num_nodes); s != MIO_OK)
            return s;
        std::vector<std::vector<std::int64_t>> rows(static_cast<std::size_t>(num_cells));
        for (std::int64_t c = 0; c < num_cells; ++c)
            rows[static_cast<std::size_t>(c)].assign(nodes + row_offsets[c],
                                                     nodes + row_offsets[c + 1]);
        mesh->mMesh.AddPolygonBlock(cell_type, std::move(rows));
        return MIO_OK;
    });
}

mio_status mio_mesh_add_polyhedron_block(mio_mesh* mesh, const char* cell_type, int64_t num_cells,
                                         const int64_t* cell_offsets, int64_t num_faces,
                                         const int64_t* face_offsets, const int64_t* nodes,
                                         int64_t num_nodes) {
    return guarded([&]() -> mio_status {
        if (!mesh || !cell_type || num_cells < 0 || num_faces < 0 || num_nodes < 0)
            return fail(MIO_ERR_INVALID_ARG,
                        "meshio++: bad mio_mesh_add_polyhedron_block argument");
        if (mio_status s = poly_check_offsets(cell_offsets, num_cells, num_faces, "cell_offsets");
            s != MIO_OK)
            return s;
        if (mio_status s = poly_check_offsets(face_offsets, num_faces, num_nodes, "face_offsets");
            s != MIO_OK)
            return s;
        if (mio_status s = poly_check_nodes(nodes, num_nodes); s != MIO_OK)
            return s;
        std::vector<std::vector<std::vector<std::int64_t>>> cells(
            static_cast<std::size_t>(num_cells));
        for (std::int64_t c = 0; c < num_cells; ++c) {
            auto& r_cell = cells[static_cast<std::size_t>(c)];
            r_cell.resize(static_cast<std::size_t>(cell_offsets[c + 1] - cell_offsets[c]));
            for (std::int64_t f = cell_offsets[c]; f < cell_offsets[c + 1]; ++f)
                r_cell[static_cast<std::size_t>(f - cell_offsets[c])].assign(
                    nodes + face_offsets[f], nodes + face_offsets[f + 1]);
        }
        mesh->mMesh.AddPolyhedronBlock(cell_type, std::move(cells));
        return MIO_OK;
    });
}

mio_status mio_mesh_add_point_data(mio_mesh* mesh, const char* name, mio_dtype dtype, int32_t ndim,
                                   const int64_t* shape, const void* data) {
    return guarded([&]() -> mio_status {
        const std::int64_t npoints = mesh ? static_cast<std::int64_t>(mesh->mMesh.NumPoints()) : -1;
        return add_named_array(mesh, name, dtype, ndim, shape, data, "mio_mesh_add_point_data",
                               npoints,
                               [&](NDArray a) { mesh->mMesh.AddPointData(name, std::move(a)); });
    });
}

mio_status mio_mesh_append_cell_data(mio_mesh* mesh, const char* name, mio_dtype dtype,
                                     int32_t ndim, const int64_t* shape, const void* data) {
    return guarded([&]() -> mio_status {
        // The array being appended belongs to the next block in order: block
        // index = how many arrays this field already has.
        std::int64_t required = -1;
        if (mesh && name) {
            const std::size_t next =
                mesh->mMesh.HasCellData(name) ? mesh->mMesh.CellDataNumBlocks(name) : 0;
            if (next >= mesh->mMesh.NumCellBlocks())
                return fail(MIO_ERR_INVALID_ARG,
                            std::string("meshio++: cell_data '") + name + "' already has one " +
                                "array per cell block (add cell blocks before their data)");
            required = static_cast<std::int64_t>(mesh->mMesh.Cells(next).NumCells());
        }
        return add_named_array(mesh, name, dtype, ndim, shape, data, "mio_mesh_append_cell_data",
                               required,
                               [&](NDArray a) { mesh->mMesh.AppendCellData(name, std::move(a)); });
    });
}

mio_status mio_mesh_add_field_data(mio_mesh* mesh, const char* name, mio_dtype dtype, int32_t ndim,
                                   const int64_t* shape, const void* data) {
    return guarded([&]() -> mio_status {
        return add_named_array(mesh, name, dtype, ndim, shape, data, "mio_mesh_add_field_data",
                               /*required_dim0=*/-1,
                               [&](NDArray a) { mesh->mMesh.AddFieldData(name, std::move(a)); });
    });
}

/* ------------------------------------------------------------------ */
/* Reading a mesh back (getters are zero-copy)                         */
/* ------------------------------------------------------------------ */

int64_t mio_mesh_num_points(const mio_mesh* mesh) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        return static_cast<int64_t>(mesh->mMesh.NumPoints());
    });
}

int64_t mio_mesh_point_dim(const mio_mesh* mesh) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        return static_cast<int64_t>(mesh->mMesh.PointDim());
    });
}

mio_status mio_mesh_get_points(const mio_mesh* mesh, const void** data, mio_dtype* dtype) {
    return guarded([&]() -> mio_status {
        if (!mesh)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: mesh is NULL");
        const NDArray& points = mesh->mMesh.Points();
        return array_out(points, data, dtype, nullptr, nullptr);
    });
}

int64_t mio_mesh_num_cell_blocks(const mio_mesh* mesh) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        return static_cast<int64_t>(mesh->mMesh.NumCellBlocks());
    });
}

mio_status mio_mesh_cell_block_info(const mio_mesh* mesh, int64_t block, int64_t* num_cells,
                                    int64_t* nodes_per_cell, int32_t* is_ragged) {
    return guarded([&]() -> mio_status {
        if (!mesh)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: mesh is NULL");
        if (!block_in_range(mesh, block))
            return fail(MIO_ERR_NOT_FOUND,
                        "meshio++: cell block " + std::to_string(block) + " out of range");
        const auto view = mesh->mMesh.Cells(static_cast<std::size_t>(block));
        if (num_cells)
            *num_cells = static_cast<int64_t>(view.NumCells());
        if (nodes_per_cell)
            *nodes_per_cell = static_cast<int64_t>(view.NodesPerCell());
        if (is_ragged)
            *is_ragged = view.IsRagged() ? 1 : 0;
        return MIO_OK;
    });
}

namespace {

// The one place a CellView is unrolled into flat CSR. Shared by
// mio_mesh_cell_block_info_ex (which needs the totals) and mio_poly_conn_create
// (which needs the arrays), so the two can never report different shapes.
// Goes through the uniform API only, hence it compiles under all three
// backends -- see the mio_poly_conn comment above.
void poly_flatten(const meshioplusplus::Mesh::CellView& rView, mio_poly_conn& rOut) {
    const std::size_t ncells = rView.NumCells();
    rOut.mIsPolyhedron = rView.IsPolyhedron();
    rOut.mNumCells = static_cast<std::int64_t>(ncells);
    rOut.mFaceOffsets.push_back(0);
    if (rOut.mIsPolyhedron) {
        rOut.mCellOffsets.push_back(0);
        for (std::size_t c = 0; c < ncells; ++c) {
            for (std::size_t f = 0; f < rView.NumFaces(c); ++f) {
                const auto face = rView.Face(c, f);
                rOut.mNodes.insert(rOut.mNodes.end(), face.first, face.first + face.second);
                rOut.mFaceOffsets.push_back(static_cast<std::int64_t>(rOut.mNodes.size()));
            }
            rOut.mCellOffsets.push_back(static_cast<std::int64_t>(rOut.mFaceOffsets.size() - 1));
        }
    } else {
        // 1-level: one face per cell, so face_offsets IS the row-offsets array
        // and cell_offsets stays empty (reported as NULL).
        for (std::size_t c = 0; c < ncells; ++c) {
            const std::int64_t* p_row = rView.Row(c);
            rOut.mNodes.insert(rOut.mNodes.end(), p_row, p_row + rView.RowSize(c));
            rOut.mFaceOffsets.push_back(static_cast<std::int64_t>(rOut.mNodes.size()));
        }
    }
}

}  // namespace

mio_status mio_mesh_cell_block_info_ex(const mio_mesh* mesh, int64_t block,
                                       mio_cell_block_info* out) {
    return guarded([&]() -> mio_status {
        if (!mesh || !out)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: mesh or out is NULL");
        if (!block_in_range(mesh, block))
            return fail(MIO_ERR_NOT_FOUND,
                        "meshio++: cell block " + std::to_string(block) + " out of range");
        const auto view = mesh->mMesh.Cells(static_cast<std::size_t>(block));
        *out = mio_cell_block_info{};
        out->num_cells = static_cast<int64_t>(view.NumCells());
        out->nodes_per_cell = static_cast<int64_t>(view.NodesPerCell());
        out->is_ragged = view.IsRagged() ? 1 : 0;
        out->is_polyhedron = view.IsPolyhedron() ? 1 : 0;
        if (!view.IsRagged()) {
            out->num_faces = out->num_cells;
            out->num_nodes = out->num_cells * out->nodes_per_cell;
        } else if (view.IsPolyhedron()) {
            for (std::size_t c = 0; c < view.NumCells(); ++c) {
                const std::size_t nf = view.NumFaces(c);
                out->num_faces += static_cast<int64_t>(nf);
                for (std::size_t f = 0; f < nf; ++f)
                    out->num_nodes += static_cast<int64_t>(view.Face(c, f).second);
            }
        } else {
            out->num_faces = out->num_cells;
            for (std::size_t c = 0; c < view.NumCells(); ++c)
                out->num_nodes += static_cast<int64_t>(view.RowSize(c));
        }
        return MIO_OK;
    });
}

int64_t mio_mesh_cell_block_type(const mio_mesh* mesh, int64_t block, char* buf, int64_t buflen) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!block_in_range(mesh, block))
            throw meshioplusplus::ReadError("meshio++: cell block " + std::to_string(block) +
                                            " out of range");
        // Bind through a std::string so per-backend Type() return categories
        // (reference vs temporary) both stay valid for the copy.
        const std::string type = mesh->mMesh.Cells(static_cast<std::size_t>(block)).Type();
        return copy_string(type, buf, buflen);
    });
}

mio_status mio_mesh_cell_block_conn(const mio_mesh* mesh, int64_t block, const void** conn,
                                    mio_dtype* dtype) {
    return guarded([&]() -> mio_status {
        if (!mesh)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: mesh is NULL");
        if (!block_in_range(mesh, block))
            return fail(MIO_ERR_NOT_FOUND,
                        "meshio++: cell block " + std::to_string(block) + " out of range");
        const auto view = mesh->mMesh.Cells(static_cast<std::size_t>(block));
        if (view.IsRagged())
            return fail(MIO_ERR_UNSUPPORTED,
                        "meshio++: cell block " + std::to_string(block) +
                            " is ragged and has no rectangular connectivity to borrow; "
                            "read it with mio_poly_conn_create()");
        const NDArray& c = view.Conn();
        return array_out(c, conn, dtype, nullptr, nullptr);
    });
}

/* Ragged connectivity: an owning snapshot rather than a rule-3 borrow. See the
 * mio_poly_conn struct comment above and doc/polyhedra.md. */

mio_poly_conn* mio_poly_conn_create(const mio_mesh* mesh, int64_t block) {
    return guarded_ptr(static_cast<mio_poly_conn*>(nullptr), [&]() -> mio_poly_conn* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        if (!block_in_range(mesh, block))
            throw meshioplusplus::ReadError("meshio++: cell block " + std::to_string(block) +
                                            " out of range");
        const auto view = mesh->mMesh.Cells(static_cast<std::size_t>(block));
        if (!view.IsRagged())
            throw meshioplusplus::ReadError(
                "meshio++: cell block " + std::to_string(block) +
                " is rectangular; borrow it with mio_mesh_cell_block_conn()");
        // Filled by value first, so a throw mid-flatten cannot leak the handle.
        mio_poly_conn poly;
        poly_flatten(view, poly);
        return new mio_poly_conn(std::move(poly));
    });
}

mio_status mio_poly_conn_get_shape(const mio_poly_conn* poly, mio_poly_conn_shape* out) {
    return guarded([&]() -> mio_status {
        if (!poly || !out)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: poly or out is NULL");
        *out = mio_poly_conn_shape{};
        out->is_polyhedron = poly->mIsPolyhedron ? 1 : 0;
        out->num_cells = poly->mNumCells;
        out->num_faces = static_cast<int64_t>(poly->mFaceOffsets.size()) - 1;
        out->num_nodes = static_cast<int64_t>(poly->mNodes.size());
        return MIO_OK;
    });
}

namespace {

const int64_t* poly_array_out(const mio_poly_conn* pPoly, const std::vector<std::int64_t>& rVec,
                              int64_t* pCount) {
    if (!pPoly)
        throw meshioplusplus::ReadError("meshio++: poly is NULL");
    if (pCount)
        *pCount = static_cast<int64_t>(rVec.size());
    return rVec.empty() ? nullptr : rVec.data();
}

}  // namespace

const int64_t* mio_poly_conn_nodes(const mio_poly_conn* poly, int64_t* count) {
    return guarded_ptr(static_cast<const int64_t*>(nullptr), [&]() -> const int64_t* {
        return poly_array_out(poly, poly->mNodes, count);
    });
}

const int64_t* mio_poly_conn_face_offsets(const mio_poly_conn* poly, int64_t* count) {
    return guarded_ptr(static_cast<const int64_t*>(nullptr), [&]() -> const int64_t* {
        return poly_array_out(poly, poly->mFaceOffsets, count);
    });
}

const int64_t* mio_poly_conn_cell_offsets(const mio_poly_conn* poly, int64_t* count) {
    // Empty (so NULL, count 0) for a 1-level block: a NULL nobody can mistake
    // for data, rather than a synthesized identity that looks like information.
    return guarded_ptr(static_cast<const int64_t*>(nullptr), [&]() -> const int64_t* {
        return poly_array_out(poly, poly->mCellOffsets, count);
    });
}

void mio_poly_conn_free(mio_poly_conn* poly) {
    delete poly;
}

/* Named-data accessors: the three families (point/cell/field) share the same
 * shape; small macros would obscure more than they save, so they are spelled
 * out. Names are returned in ascending lexicographic order -- the uniform
 * API's *DataNames() guarantee, identical on every backend. */

int64_t mio_mesh_num_point_data(const mio_mesh* mesh) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        return static_cast<int64_t>(mesh->mMesh.NumPointData());
    });
}

int64_t mio_mesh_point_data_name(const mio_mesh* mesh, int64_t index, char* buf, int64_t buflen) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        const auto names = mesh->mMesh.PointDataNames();
        if (index < 0 || static_cast<std::size_t>(index) >= names.size())
            throw meshioplusplus::ReadError("meshio++: point_data index " + std::to_string(index) +
                                            " out of range");
        return copy_string(names[static_cast<std::size_t>(index)], buf, buflen);
    });
}

mio_status mio_mesh_get_point_data(const mio_mesh* mesh, const char* name, const void** data,
                                   mio_dtype* dtype, int32_t* ndim, int64_t* shape) {
    return guarded([&]() -> mio_status {
        if (!mesh || !name)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: mesh/name is NULL");
        if (!mesh->mMesh.HasPointData(name))
            return fail(MIO_ERR_NOT_FOUND,
                        "meshio++: no point_data named '" + std::string(name) + "'");
        return array_out(mesh->mMesh.PointData(name), data, dtype, ndim, shape);
    });
}

int64_t mio_mesh_num_cell_data(const mio_mesh* mesh) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        return static_cast<int64_t>(mesh->mMesh.NumCellData());
    });
}

int64_t mio_mesh_cell_data_name(const mio_mesh* mesh, int64_t index, char* buf, int64_t buflen) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        const auto names = mesh->mMesh.CellDataNames();
        if (index < 0 || static_cast<std::size_t>(index) >= names.size())
            throw meshioplusplus::ReadError("meshio++: cell_data index " + std::to_string(index) +
                                            " out of range");
        return copy_string(names[static_cast<std::size_t>(index)], buf, buflen);
    });
}

int64_t mio_mesh_cell_data_num_blocks(const mio_mesh* mesh, const char* name) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!mesh || !name)
            throw meshioplusplus::ReadError("meshio++: mesh/name is NULL");
        if (!mesh->mMesh.HasCellData(name))
            throw meshioplusplus::ReadError("meshio++: no cell_data named '" + std::string(name) +
                                            "'");
        return static_cast<int64_t>(mesh->mMesh.CellDataNumBlocks(name));
    });
}

mio_status mio_mesh_get_cell_data(const mio_mesh* mesh, const char* name, int64_t block,
                                  const void** data, mio_dtype* dtype, int32_t* ndim,
                                  int64_t* shape) {
    return guarded([&]() -> mio_status {
        if (!mesh || !name)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: mesh/name is NULL");
        if (!mesh->mMesh.HasCellData(name))
            return fail(MIO_ERR_NOT_FOUND,
                        "meshio++: no cell_data named '" + std::string(name) + "'");
        if (block < 0 || static_cast<std::size_t>(block) >= mesh->mMesh.CellDataNumBlocks(name))
            return fail(MIO_ERR_NOT_FOUND, "meshio++: cell_data '" + std::string(name) +
                                               "' block " + std::to_string(block) +
                                               " out of range");
        return array_out(mesh->mMesh.CellData(name, static_cast<std::size_t>(block)), data, dtype,
                         ndim, shape);
    });
}

int64_t mio_mesh_num_field_data(const mio_mesh* mesh) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        return static_cast<int64_t>(mesh->mMesh.NumFieldData());
    });
}

int64_t mio_mesh_field_data_name(const mio_mesh* mesh, int64_t index, char* buf, int64_t buflen) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        const auto names = mesh->mMesh.FieldDataNames();
        if (index < 0 || static_cast<std::size_t>(index) >= names.size())
            throw meshioplusplus::ReadError("meshio++: field_data index " + std::to_string(index) +
                                            " out of range");
        return copy_string(names[static_cast<std::size_t>(index)], buf, buflen);
    });
}

mio_status mio_mesh_get_field_data(const mio_mesh* mesh, const char* name, const void** data,
                                   mio_dtype* dtype, int32_t* ndim, int64_t* shape) {
    return guarded([&]() -> mio_status {
        if (!mesh || !name)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: mesh/name is NULL");
        if (!mesh->mMesh.HasFieldData(name))
            return fail(MIO_ERR_NOT_FOUND,
                        "meshio++: no field_data named '" + std::string(name) + "'");
        return array_out(mesh->mMesh.FieldData(name), data, dtype, ndim, shape);
    });
}

/* ---------------------------------------------------------------------
 * Named regions (see the header, and doc/regions.md)
 * --------------------------------------------------------------------- */

static_assert(static_cast<int>(meshioplusplus::RegionKind::Point) == MIO_REGION_POINT, "");
static_assert(static_cast<int>(meshioplusplus::RegionKind::Cell) == MIO_REGION_CELL, "");
static_assert(static_cast<int>(meshioplusplus::RegionKind::Side) == MIO_REGION_SIDE, "");

mio_regions* mio_regions_create(const mio_mesh* mesh) {
    return guarded_ptr(static_cast<mio_regions*>(nullptr), [&]() -> mio_regions* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        auto* out = new mio_regions{};
        out->mRegions.reserve(mesh->mMesh.NumRegions());
        for (std::size_t i = 0; i < mesh->mMesh.NumRegions(); ++i)
            out->mRegions.push_back(mesh->mMesh.Region(i));
        return out;
    });
}

int64_t mio_regions_count(const mio_regions* regions) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!regions)
            throw meshioplusplus::ReadError("meshio++: regions is NULL");
        return static_cast<int64_t>(regions->mRegions.size());
    });
}

int64_t mio_regions_name(const mio_regions* regions, int64_t index, char* buf, int64_t buflen) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!regions)
            throw meshioplusplus::ReadError("meshio++: regions is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= regions->mRegions.size())
            throw meshioplusplus::ReadError("meshio++: region index " + std::to_string(index) +
                                            " out of range");
        return copy_string(regions->mRegions[static_cast<std::size_t>(index)].mName, buf, buflen);
    });
}

mio_status mio_regions_info(const mio_regions* regions, int64_t index, mio_region_info* out) {
    return guarded([&]() -> mio_status {
        if (!regions || !out)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: regions/out is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= regions->mRegions.size())
            return fail(MIO_ERR_INVALID_ARG,
                        "meshio++: region index " + std::to_string(index) + " out of range");
        const meshioplusplus::Region& r = regions->mRegions[static_cast<std::size_t>(index)];
        out->kind = static_cast<int32_t>(r.mKind);
        out->dim = static_cast<int32_t>(r.mDim);
        out->tag = r.mTag;
        out->num_entries = static_cast<int64_t>(r.NumEntries());
        out->stride = static_cast<int64_t>(r.Stride());
        return MIO_OK;
    });
}

const int64_t* mio_regions_entries(const mio_regions* regions, int64_t index, int64_t* count) {
    return guarded_ptr(static_cast<const int64_t*>(nullptr), [&]() -> const int64_t* {
        if (!regions)
            throw meshioplusplus::ReadError("meshio++: regions is NULL");
        if (index < 0 || static_cast<std::size_t>(index) >= regions->mRegions.size())
            throw meshioplusplus::ReadError("meshio++: region index " + std::to_string(index) +
                                            " out of range");
        const meshioplusplus::Region& r = regions->mRegions[static_cast<std::size_t>(index)];
        if (count)
            *count = static_cast<int64_t>(r.mEntries.Size());
        return r.Entries();
    });
}

void mio_regions_free(mio_regions* regions) {
    delete regions;
}

mio_status mio_mesh_add_region(mio_mesh* mesh, const char* name, mio_region_kind kind, int32_t dim,
                               int64_t tag, const int64_t* entries, int64_t count) {
    return guarded([&]() -> mio_status {
        if (!mesh || !name)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: mesh/name is NULL");
        if (count < 0 || (count > 0 && !entries))
            return fail(MIO_ERR_INVALID_ARG, "meshio++: region entries is NULL");
        const auto region_kind = static_cast<meshioplusplus::RegionKind>(kind);
        if (region_kind == meshioplusplus::RegionKind::Side && count % 2 != 0)
            return fail(MIO_ERR_INVALID_ARG,
                        "meshio++: a side region needs an even entry count "
                        "((cell, facet) pairs)");
        const auto n = static_cast<std::size_t>(count);
        std::vector<std::size_t> shape;
        if (region_kind == meshioplusplus::RegionKind::Side)
            shape = {n / 2, 2};
        else
            shape = {n};
        meshioplusplus::NDArray arr =
            meshioplusplus::NDArray::Uninit(meshioplusplus::DType::Int64, std::move(shape));
        for (std::size_t i = 0; i < n; ++i)
            arr.As<std::int64_t>()[i] = entries[i];
        mesh->mMesh.AddRegion(meshioplusplus::Region(name, region_kind, dim, tag, std::move(arr)));
        return MIO_OK;
    });
}

// ---- transient (time-series) XDMF ----------------------------------------
//
// The one writer that is a handle rather than a (path, mesh) call: the mesh is
// written once and each step appended, so there is no single call for mio_write
// to be. The C++ object is held by value; every entry point stays inside
// guarded()/guarded_ptr() like the rest of this file.

mio_xdmf_series* mio_xdmf_series_create(const char* path, const char* data_format,
                                        int32_t gzip_level) {
    return guarded_ptr(static_cast<mio_xdmf_series*>(nullptr), [&]() -> mio_xdmf_series* {
        if (!path)
            throw meshioplusplus::WriteError("meshio++: path is NULL");
        const std::string fmt = data_format ? data_format : "HDF";
        return new mio_xdmf_series{
            meshioplusplus::XdmfTimeSeriesWriter(path, fmt, static_cast<int>(gzip_level))};
    });
}

void mio_xdmf_series_opts_init(mio_xdmf_series_opts* opts) {
    if (!opts)
        return;
    *opts = mio_xdmf_series_opts{};  // value-initialized: HDF, truncate, no flush
}

mio_xdmf_series* mio_xdmf_series_create_ex(const char* path, const mio_xdmf_series_opts* opts) {
    return guarded_ptr(static_cast<mio_xdmf_series*>(nullptr), [&]() -> mio_xdmf_series* {
        if (!path)
            throw meshioplusplus::WriteError("meshio++: path is NULL");
        mio_xdmf_series_opts defaults{};
        const mio_xdmf_series_opts& r_o = opts ? *opts : defaults;
        const std::string fmt = r_o.data_format ? r_o.data_format : "HDF";
        const auto mode = r_o.mode == MIO_XDMF_SERIES_APPEND
                              ? meshioplusplus::XdmfSeriesMode::Append
                              : meshioplusplus::XdmfSeriesMode::Truncate;
        auto* p_out = new mio_xdmf_series{meshioplusplus::XdmfTimeSeriesWriter(
            path, fmt, static_cast<int>(r_o.gzip_level), mode)};
        p_out->mWriter.SetAutoFlush(r_o.auto_flush != 0);
        return p_out;
    });
}

mio_status mio_xdmf_series_write_points_cells(mio_xdmf_series* series, const mio_mesh* mesh) {
    return guarded([&]() -> mio_status {
        if (!series || !mesh)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: series/mesh is NULL");
        series->mWriter.WritePointsCells(mesh->mMesh);
        return MIO_OK;
    });
}

mio_status mio_xdmf_series_write_data(mio_xdmf_series* series, double time, const mio_mesh* mesh) {
    return guarded([&]() -> mio_status {
        if (!series || !mesh)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: series/mesh is NULL");
        series->mWriter.WriteData(time, mesh->mMesh);
        return MIO_OK;
    });
}

mio_status mio_xdmf_series_finalize(mio_xdmf_series* series) {
    return guarded([&]() -> mio_status {
        if (!series)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: series is NULL");
        series->mWriter.Finalize();
        return MIO_OK;
    });
}

mio_status mio_xdmf_series_write_data_arrays(mio_xdmf_series* series, double time,
                                             const mio_named_array* point_arrays,
                                             int64_t num_point_arrays,
                                             const mio_named_array* cell_arrays,
                                             int64_t num_cell_arrays) {
    return guarded([&]() -> mio_status {
        if (!series)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: series is NULL");
        const auto convert = [](const mio_named_array* p_src, std::int64_t n) {
            std::vector<meshioplusplus::XdmfTimeSeriesWriter::NamedArray> out;
            for (std::int64_t i = 0; i < n; ++i) {
                if (!p_src)
                    throw meshioplusplus::WriteError("meshio++: array list is NULL");
                meshioplusplus::XdmfTimeSeriesWriter::NamedArray a;
                a.mName = p_src[i].name ? p_src[i].name : "";
                a.mNumComponents = p_src[i].num_components > 0
                                       ? static_cast<std::size_t>(p_src[i].num_components)
                                       : 1u;
                if (p_src[i].num_values < 0)
                    throw meshioplusplus::WriteError("meshio++: num_values is negative");
                // Copied during the call, per the ABI's "setters copy" rule.
                if (p_src[i].values)
                    a.mValues.assign(p_src[i].values, p_src[i].values + p_src[i].num_values);
                out.push_back(std::move(a));
            }
            return out;
        };
        series->mWriter.WriteData(time, convert(point_arrays, num_point_arrays),
                                  convert(cell_arrays, num_cell_arrays));
        return MIO_OK;
    });
}

mio_status mio_xdmf_series_flush(mio_xdmf_series* series) {
    return guarded([&]() -> mio_status {
        if (!series)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: series is NULL");
        series->mWriter.Flush();
        return MIO_OK;
    });
}

int32_t mio_xdmf_series_finalized(const mio_xdmf_series* series) {
    return guarded_ptr(static_cast<std::int32_t>(-1), [&]() -> std::int32_t {
        if (!series)
            throw meshioplusplus::ReadError("meshio++: series is NULL");
        return series->mWriter.Finalized() ? 1 : 0;
    });
}

int64_t mio_xdmf_series_num_steps(const mio_xdmf_series* series) {
    return guarded_ptr(std::int64_t(-1), [&]() -> std::int64_t {
        if (!series)
            throw meshioplusplus::ReadError("meshio++: series is NULL");
        return static_cast<std::int64_t>(series->mWriter.NumSteps());
    });
}

void mio_xdmf_series_free(mio_xdmf_series* series) {
    // ~XdmfTimeSeriesWriter finalizes and swallows any failure (it must not
    // throw); a caller wanting to see one calls mio_xdmf_series_finalize first.
    delete series;
}

// --------------------------------------------------------------------------
// Settings pipeline. JSON text only across the ABI -- the typed step model
// never crosses it. When the build has no JSON parser the core throws naming
// -DMESHIOPLUSPLUS_WITH_JSON=ON, which guarded() surfaces through
// mio_last_error() as usual.
// --------------------------------------------------------------------------

mio_status mio_pipeline_run_file(const char* settings_path) {
    return guarded([&]() -> mio_status {
        if (!settings_path)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: settings_path is NULL");
        meshioplusplus::run_pipeline_file(settings_path);
        return MIO_OK;
    });
}

mio_status mio_pipeline_run_json(const char* json_text) {
    return guarded([&]() -> mio_status {
        if (!json_text)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: json_text is NULL");
        meshioplusplus::run_pipeline_json(json_text);
        return MIO_OK;
    });
}

namespace {

/// mio_sequence_opts -> SequenceInput, shared by the three open* entry points.
meshioplusplus::SequenceInput seq_input_from_opts(const mio_sequence_opts* pOpts) {
    meshioplusplus::SequenceInput in;
    if (!pOpts)
        return in;
    if (pOpts->format)
        in.mFormat = pOpts->format;
    if (pOpts->times && pOpts->num_times > 0)
        in.mTimes.assign(pOpts->times, pOpts->times + pOpts->num_times);
    switch (pOpts->time_from) {
        case 1:
            in.mTimeFrom = meshioplusplus::SequenceTimeFrom::File;
            break;
        case 2:
            in.mTimeFrom = meshioplusplus::SequenceTimeFrom::Filename;
            break;
        case 3:
            in.mTimeFrom = meshioplusplus::SequenceTimeFrom::Index;
            break;
        default:
            in.mTimeFrom = meshioplusplus::SequenceTimeFrom::Auto;
            break;
    }
    in.mSortExplicit = pOpts->sort != 0;
    return in;
}

mio_sequence* seq_open(meshioplusplus::SequenceInput in) {
    auto* out = new mio_sequence{};
    out->mEntries = meshioplusplus::sequence_expand(in);
    out->mFormat = in.mFormat;
    out->mOptions = in.mOptions;
    return out;
}

const meshioplusplus::SequenceEntry& seq_entry(const mio_sequence* pSeq, int64_t Index) {
    if (!pSeq)
        throw meshioplusplus::ReadError("meshio++: sequence is NULL");
    if (Index < 0 || static_cast<std::size_t>(Index) >= pSeq->mEntries.size())
        throw meshioplusplus::ReadError("meshio++: sequence entry index out of range");
    return pSeq->mEntries[static_cast<std::size_t>(Index)];
}

}  // namespace

void mio_sequence_opts_init(mio_sequence_opts* opts) {
    if (opts)
        *opts = mio_sequence_opts{};
}

mio_sequence* mio_sequence_open(const char* pattern) {
    return mio_sequence_open_ex(pattern, nullptr);
}

mio_sequence* mio_sequence_open_ex(const char* pattern, const mio_sequence_opts* opts) {
    return guarded_ptr(static_cast<mio_sequence*>(nullptr), [&]() -> mio_sequence* {
        if (!pattern)
            throw meshioplusplus::ReadError("meshio++: pattern is NULL");
        meshioplusplus::SequenceInput in = seq_input_from_opts(opts);
        in.mPattern = pattern;
        return seq_open(std::move(in));
    });
}

mio_sequence* mio_sequence_open_list(const char* const* paths, int64_t num_paths,
                                     const mio_sequence_opts* opts) {
    return guarded_ptr(static_cast<mio_sequence*>(nullptr), [&]() -> mio_sequence* {
        if (!paths || num_paths <= 0)
            throw meshioplusplus::ReadError("meshio++: paths is NULL or empty");
        meshioplusplus::SequenceInput in = seq_input_from_opts(opts);
        for (int64_t i = 0; i < num_paths; ++i) {
            if (!paths[i])
                throw meshioplusplus::ReadError("meshio++: a path in the list is NULL");
            in.mPaths.emplace_back(paths[i]);
        }
        return seq_open(std::move(in));
    });
}

int64_t mio_sequence_count(const mio_sequence* seq) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        if (!seq)
            throw meshioplusplus::ReadError("meshio++: sequence is NULL");
        return static_cast<int64_t>(seq->mEntries.size());
    });
}

int64_t mio_sequence_path(const mio_sequence* seq, int64_t index, char* buf, int64_t buflen) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        return copy_string(seq_entry(seq, index).mPath, buf, buflen);
    });
}

int64_t mio_sequence_step(const mio_sequence* seq, int64_t index) {
    return guarded_ptr(static_cast<int64_t>(-1), [&]() -> int64_t {
        return static_cast<int64_t>(seq_entry(seq, index).mStep);
    });
}

mio_status mio_sequence_time(const mio_sequence* seq, int64_t index, double* out_time) {
    return guarded([&]() -> mio_status {
        if (!out_time)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: out_time is NULL");
        *out_time = seq_entry(seq, index).mTime;
        return MIO_OK;
    });
}

int32_t mio_sequence_time_source(const mio_sequence* seq, int64_t index) {
    return guarded_ptr(static_cast<int32_t>(-1), [&]() -> int32_t {
        return static_cast<int32_t>(seq_entry(seq, index).mTimeSource);
    });
}

mio_mesh* mio_sequence_read(mio_sequence* seq, int64_t index) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        seq_entry(seq, index);  // bounds/NULL check with the shared message
        return new mio_mesh{meshioplusplus::sequence_read_step(
            seq->mEntries, static_cast<std::size_t>(index), seq->mFormat, seq->mOptions)};
    });
}

void mio_sequence_free(mio_sequence* seq) {
    delete seq;
}

mio_status mio_sequence_to_timeseries(const mio_sequence* seq, const char* out_path,
                                      const char* out_format) {
    return mio_sequence_to_timeseries_ex(seq, out_path, out_format, nullptr);
}

mio_status mio_sequence_to_timeseries_ex(const mio_sequence* seq, const char* out_path,
                                         const char* out_format, const mio_write_opts* opts) {
    return guarded([&]() -> mio_status {
        if (!seq || !out_path)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: sequence/out_path is NULL");
        // Rebuild the input from the plan rather than re-globbing: the handle's
        // entries (and their resolved times) are what the caller inspected.
        meshioplusplus::SequenceInput in;
        for (const meshioplusplus::SequenceEntry& e : seq->mEntries)
            in.mPaths.push_back(e.mPath);
        in.mFormat = seq->mFormat;
        in.mOptions = seq->mOptions;
        for (const meshioplusplus::SequenceEntry& e : seq->mEntries)
            in.mTimes.push_back(e.mTime);
        meshioplusplus::SequenceOutput out;
        out.mPath = out_path;
        if (out_format)
            out.mFormat = out_format;
        if (opts) {
            const mio_status opt_status = write_opts_to_cxx(*opts, out.mOptions);
            if (opt_status != MIO_OK)
                return opt_status;
        }
        meshioplusplus::sequence_to_timeseries(in, out);
        return MIO_OK;
    });
}

mio_status mio_timeseries_to_sequence(const char* in_path, const char* in_format,
                                      const char* out_pattern, const char* out_format) {
    return guarded([&]() -> mio_status {
        if (!in_path || !out_pattern)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: in_path/out_pattern is NULL");
        meshioplusplus::SequenceOutput out;
        out.mPath = out_pattern;
        if (out_format)
            out.mFormat = out_format;
        meshioplusplus::timeseries_to_sequence(in_path, in_format ? in_format : "",
                                               meshioplusplus::ReadOptions{}, out);
        return MIO_OK;
    });
}

mio_status mio_sequence_pipeline_run_file(const char* settings_path) {
    return guarded([&]() -> mio_status {
        if (!settings_path)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: settings_path is NULL");
        meshioplusplus::run_sequence_file(settings_path);
        return MIO_OK;
    });
}

mio_status mio_sequence_pipeline_run_json(const char* json_text) {
    return guarded([&]() -> mio_status {
        if (!json_text)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: json_text is NULL");
        meshioplusplus::run_sequence_json(json_text);
        return MIO_OK;
    });
}

/* ------------------------------------------------------------------------- */
/* Regular grids and signed distance                                         */
/* ------------------------------------------------------------------------- */

namespace {

// The C enums duplicate the C++ ones across the ABI, so pin every value: drift
// becomes a compile error rather than a silently wrong argument (the
// mio_region_kind / mio_refine_compare pattern).
static_assert(static_cast<int>(meshioplusplus::VoxelFill::All) == MIO_VOXEL_ALL &&
                  static_cast<int>(meshioplusplus::VoxelFill::Surface) == MIO_VOXEL_SURFACE &&
                  static_cast<int>(meshioplusplus::VoxelFill::Inside) == MIO_VOXEL_INSIDE,
              "mio_voxel_fill drifted from VoxelFill");
static_assert(static_cast<int>(meshioplusplus::SdfSign::Unsigned) == MIO_SDF_UNSIGNED &&
                  static_cast<int>(meshioplusplus::SdfSign::Pseudonormal) == MIO_SDF_PSEUDONORMAL &&
                  static_cast<int>(meshioplusplus::SdfSign::WindingNumber) ==
                      MIO_SDF_WINDING_NUMBER,
              "mio_sdf_sign drifted from SdfSign");
static_assert(static_cast<int>(meshioplusplus::SdfPseudonormalWeight::Angle) ==
                      MIO_SDF_WEIGHT_ANGLE &&
                  static_cast<int>(meshioplusplus::SdfPseudonormalWeight::Area) ==
                      MIO_SDF_WEIGHT_AREA,
              "mio_sdf_weight drifted from SdfPseudonormalWeight");
static_assert(static_cast<int>(meshioplusplus::SdfLocation::Corner) == MIO_SDF_CORNER &&
                  static_cast<int>(meshioplusplus::SdfLocation::Center) == MIO_SDF_CENTER,
              "mio_sdf_location drifted from SdfLocation");
static_assert(static_cast<int>(meshioplusplus::SdfWatertightCheck::Off) == MIO_SDF_WATERTIGHT_OFF &&
                  static_cast<int>(meshioplusplus::SdfWatertightCheck::Warn) ==
                      MIO_SDF_WATERTIGHT_WARN &&
                  static_cast<int>(meshioplusplus::SdfWatertightCheck::Error) ==
                      MIO_SDF_WATERTIGHT_ERROR,
              "mio_sdf_watertight_check drifted from SdfWatertightCheck");

// The sizes are ABI: the Fortran and Julia mirrors of these structs hard-code
// them, and the Julia binding checks them at load. Pinning them here makes a
// field added outside the `reserved` tail a compile error rather than silent
// corruption.
//
// Note mio_voxel_opts carries `sign`/`watertight_check` as its OWN int32 fields
// rather than embedding a mio_sdf_opts. That is deliberate: the C++ VoxelOptions
// does embed SurfaceDistanceOptions by value, and doc/abi.md records that
// growing the inner struct is a Tier A break of the outer one. Not repeating
// that nesting across the ABI keeps the two independently extensible.
static_assert(sizeof(mio_sdf_opts) == 104, "mio_sdf_opts grew outside its reserved tail");
static_assert(sizeof(mio_voxel_opts) == 112, "mio_voxel_opts grew outside its reserved tail");
static_assert(sizeof(mio_compute_sdf_opts) == 232,
              "mio_compute_sdf_opts grew outside its reserved tail");
static_assert(sizeof(mio_surface_quality) == 72,
              "mio_surface_quality grew outside its reserved tail");

/// Translate the flat option struct into the core's SurfaceDistanceOptions.
meshioplusplus::SurfaceDistanceOptions capi_sdf_options(const mio_sdf_opts& rOpts) {
    meshioplusplus::SurfaceDistanceOptions options;
    if (rOpts.sign < 0 || rOpts.sign > MIO_SDF_WINDING_NUMBER)
        throw meshioplusplus::ReadError("meshio++: sdf: unknown sign " +
                                        std::to_string(rOpts.sign));
    options.mSign = static_cast<meshioplusplus::SdfSign>(rOpts.sign);
    if (rOpts.weight < 0 || rOpts.weight > MIO_SDF_WEIGHT_AREA)
        throw meshioplusplus::ReadError("meshio++: sdf: unknown weight " +
                                        std::to_string(rOpts.weight));
    options.mWeight = static_cast<meshioplusplus::SdfPseudonormalWeight>(rOpts.weight);
    if (rOpts.location < 0 || rOpts.location > MIO_SDF_CENTER)
        throw meshioplusplus::ReadError("meshio++: sdf: unknown location " +
                                        std::to_string(rOpts.location));
    options.mLocation = static_cast<meshioplusplus::SdfLocation>(rOpts.location);
    if (rOpts.watertight_check < 0 || rOpts.watertight_check > MIO_SDF_WATERTIGHT_ERROR)
        throw meshioplusplus::ReadError("meshio++: sdf: unknown watertight check " +
                                        std::to_string(rOpts.watertight_check));
    options.mWatertightCheck =
        static_cast<meshioplusplus::SdfWatertightCheck>(rOpts.watertight_check);
    options.mBand = rOpts.band;
    options.mRecordClosestCell = rOpts.record_closest_cell != 0;
    options.mRecordInside = rOpts.record_inside != 0;
    if (rOpts.surface_region != nullptr)
        options.mSurfaceRegion = rOpts.surface_region;
    options.mGridCellSize = rOpts.grid_cell_size;
    options.mMaxWindingWork = rOpts.max_winding_work;
    return options;
}

/// Translate the flat option struct into the core's VoxelOptions.
meshioplusplus::VoxelOptions capi_voxel_options(const mio_voxel_opts& rOpts) {
    meshioplusplus::VoxelOptions options;
    if (rOpts.resolution != nullptr)
        options.mResolution = std::array<std::int64_t, 3>{
            {rOpts.resolution[0], rOpts.resolution[1], rOpts.resolution[2]}};
    if (rOpts.cell_size > 0.0)
        options.mCellSize = rOpts.cell_size;
    if (rOpts.bounds != nullptr)
        options.mBounds =
            std::array<double, 6>{{rOpts.bounds[0], rOpts.bounds[1], rOpts.bounds[2],
                                   rOpts.bounds[3], rOpts.bounds[4], rOpts.bounds[5]}};
    options.mPadding = rOpts.padding;
    options.mPaddingRelative = rOpts.padding_relative;
    if (rOpts.fill < 0 || rOpts.fill > MIO_VOXEL_INSIDE)
        throw meshioplusplus::ReadError("meshio++: voxelize: unknown fill " +
                                        std::to_string(rOpts.fill));
    options.mFill = static_cast<meshioplusplus::VoxelFill>(rOpts.fill);
    options.mAttachOccupancy = rOpts.attach_occupancy != 0;
    options.mMaxCells = rOpts.max_cells;
    if (rOpts.sign < 0 || rOpts.sign > MIO_SDF_WINDING_NUMBER)
        throw meshioplusplus::ReadError("meshio++: voxelize: unknown sign " +
                                        std::to_string(rOpts.sign));
    options.mDistance.mSign = static_cast<meshioplusplus::SdfSign>(rOpts.sign);
    if (rOpts.watertight_check < 0 || rOpts.watertight_check > MIO_SDF_WATERTIGHT_ERROR)
        throw meshioplusplus::ReadError("meshio++: voxelize: unknown watertight check " +
                                        std::to_string(rOpts.watertight_check));
    options.mDistance.mWatertightCheck =
        static_cast<meshioplusplus::SdfWatertightCheck>(rOpts.watertight_check);
    return options;
}

/// Translate the flat option struct into the core's SdfOptions.
meshioplusplus::SdfOptions capi_compute_sdf_options(const mio_compute_sdf_opts& rOpts) {
    meshioplusplus::SdfOptions options;
    if (rOpts.structure < 0 || rOpts.structure > MIO_SDF_OCTREE)
        throw meshioplusplus::ReadError("meshio++: sdf: unknown structure " +
                                        std::to_string(rOpts.structure));
    options.mStructure = static_cast<meshioplusplus::SdfStructure>(rOpts.structure);
    if (rOpts.resolution != nullptr)
        options.mResolution = std::array<std::int64_t, 3>{
            {rOpts.resolution[0], rOpts.resolution[1], rOpts.resolution[2]}};
    if (rOpts.cell_size > 0.0)
        options.mCellSize = rOpts.cell_size;
    if (rOpts.bounds != nullptr)
        options.mBounds =
            std::array<double, 6>{{rOpts.bounds[0], rOpts.bounds[1], rOpts.bounds[2],
                                   rOpts.bounds[3], rOpts.bounds[4], rOpts.bounds[5]}};
    options.mPadding = rOpts.padding;
    options.mPaddingRelative = rOpts.padding_relative;
    options.mRootResolution = rOpts.root_resolution;
    options.mMaxDepth = rOpts.max_depth;
    options.mBandCells = rOpts.band_cells;
    options.mRecordLevels = rOpts.record_levels != 0;
    options.mMaxCells = rOpts.max_cells;
    options.mDistance = capi_sdf_options(rOpts.distance);
    return options;
}

/// Translate the flat option struct into the core's RemeshVolumeOptions.
meshioplusplus::RemeshVolumeOptions capi_remesh_volume_options(
    const mio_remesh_volume_opts& rOpts) {
    meshioplusplus::RemeshVolumeOptions options;
    if (rOpts.resolution != nullptr)
        options.mResolution = std::array<std::int64_t, 3>{
            {rOpts.resolution[0], rOpts.resolution[1], rOpts.resolution[2]}};
    if (rOpts.cell_size > 0.0)
        options.mCellSize = rOpts.cell_size;
    if (rOpts.bounds != nullptr)
        options.mBounds =
            std::array<double, 6>{{rOpts.bounds[0], rOpts.bounds[1], rOpts.bounds[2],
                                   rOpts.bounds[3], rOpts.bounds[4], rOpts.bounds[5]}};
    options.mPadding = rOpts.padding;
    options.mPaddingRelative = rOpts.padding_relative;
    // Verbatim, not gated on > 0: mMaxCells/mMaxTets <= 0 means "unlimited"
    // to the C++ checks themselves (sdf.cpp/remesh_volume.cpp's own
    // `> 0 && ...` guards), so passing it through unconditionally is what
    // makes explicitly requesting "unlimited" (0) actually work, rather than
    // silently falling back to the struct's C++-side default of 20000000.
    options.mMaxCells = rOpts.max_cells;
    options.mMaxTets = rOpts.max_tets;
    options.mWarpFraction = rOpts.warp_fraction;
    options.mDistance = capi_sdf_options(rOpts.distance);
    return options;
}

void capi_fill_quality(const meshioplusplus::SurfaceQuality& rQuality, mio_surface_quality* pOut) {
    if (!pOut)
        return;
    *pOut = mio_surface_quality{};
    pOut->boundary_edges = rQuality.mBoundaryEdges;
    pOut->non_manifold_edges = rQuality.mNonManifoldEdges;
    pOut->inconsistent_pairs = rQuality.mInconsistentPairs;
    pOut->degenerate_triangles = rQuality.mDegenerateTriangles;
    pOut->watertight = rQuality.mWatertight ? 1 : 0;
}

}  // namespace

void mio_sdf_opts_init(mio_sdf_opts* opts) {
    if (!opts)
        return;
    *opts = mio_sdf_opts{};  // value-initialized: all zero == unsigned, full field
    opts->sign = MIO_SDF_PSEUDONORMAL;
    opts->watertight_check = MIO_SDF_WATERTIGHT_WARN;
    opts->max_winding_work = 2.0e9;
}

void mio_voxel_opts_init(mio_voxel_opts* opts) {
    if (!opts)
        return;
    *opts = mio_voxel_opts{};
    opts->max_cells = 20000000;
    opts->sign = MIO_SDF_PSEUDONORMAL;
    opts->watertight_check = MIO_SDF_WATERTIGHT_WARN;
}

void mio_compute_sdf_opts_init(mio_compute_sdf_opts* opts) {
    if (!opts)
        return;
    *opts = mio_compute_sdf_opts{};
    opts->padding_relative = 0.1;
    opts->band_cells = 1.0;
    opts->max_cells = 20000000;
    opts->root_resolution = 8;
    opts->max_depth = 4;
    opts->structure = MIO_SDF_VOXEL;
    opts->record_levels = 1;
    mio_sdf_opts_init(&opts->distance);
}

mio_mesh* mio_compute_sdf(const mio_mesh* surface, const mio_compute_sdf_opts* opts,
                          int64_t dims_out[3], double origin_out[3], double spacing_out[3],
                          int64_t* max_depth_out, int64_t* num_banded,
                          mio_surface_quality* quality) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!surface)
            throw meshioplusplus::ReadError("meshio++: surface is NULL");
        if (!opts)
            throw meshioplusplus::ReadError(
                "meshio++: sdf: options are NULL, but a grid sizing must be given");
        meshioplusplus::SdfResult r =
            meshioplusplus::compute_sdf(surface->mMesh, capi_compute_sdf_options(*opts));
        for (int k = 0; k < 3; ++k) {
            if (dims_out)
                dims_out[k] = r.mDims[static_cast<std::size_t>(k)];
            if (origin_out)
                origin_out[k] = r.mOrigin[static_cast<std::size_t>(k)];
            if (spacing_out)
                spacing_out[k] = r.mSpacing[static_cast<std::size_t>(k)];
        }
        if (max_depth_out)
            *max_depth_out = r.mMaxDepth;
        if (num_banded)
            *num_banded = r.mNumBanded;
        capi_fill_quality(r.mQuality, quality);
        return new mio_mesh{std::move(r.mMesh)};
    });
}

static_assert(sizeof(mio_remesh_volume_opts) == 216,
              "mio_remesh_volume_opts grew outside its reserved tail");
static_assert(sizeof(mio_remesh_volume_report) == 136,
              "mio_remesh_volume_report grew outside its reserved tail");

void mio_remesh_volume_opts_init(mio_remesh_volume_opts* opts) {
    if (!opts)
        return;
    *opts = mio_remesh_volume_opts{};
    opts->padding_relative = 0.1;
    opts->max_cells = 20000000;
    opts->max_tets = 20000000;
    opts->warp_fraction = 0.35;
    mio_sdf_opts_init(&opts->distance);
}

mio_mesh* mio_remesh_volume_ex(const mio_mesh* mesh, const mio_remesh_volume_opts* opts,
                               mio_remesh_volume_report* report) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        if (!opts)
            throw meshioplusplus::ReadError(
                "meshio++: remesh_volume: options are NULL, but exactly one of resolution and "
                "cell_size must be given");
        meshioplusplus::RemeshVolumeResult r =
            meshioplusplus::remesh_volume(mesh->mMesh, capi_remesh_volume_options(*opts));
        if (report) {
            *report = mio_remesh_volume_report{};
            capi_fill_quality(r.mQuality, &report->input_quality);
            report->num_tets = r.mNumTets;
            report->num_vertices_warped = r.mNumVerticesWarped;
            report->num_tets_rejected = r.mNumTetsRejected;
            report->num_non_manifold_edges = r.mNumNonManifoldEdges;
        }
        return new mio_mesh{std::move(r.mMesh)};
    });
}

mio_mesh* mio_grid(const int64_t dims[3], const double origin[3], const double spacing[3],
                   int64_t max_cells) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!dims)
            throw meshioplusplus::ReadError("meshio++: grid: dims is NULL");
        const std::array<std::int64_t, 3> d{{dims[0], dims[1], dims[2]}};
        const std::array<double, 3> o =
            origin ? std::array<double, 3>{{origin[0], origin[1], origin[2]}}
                   : std::array<double, 3>{{0.0, 0.0, 0.0}};
        const std::array<double, 3> s =
            spacing ? std::array<double, 3>{{spacing[0], spacing[1], spacing[2]}}
                    : std::array<double, 3>{{1.0, 1.0, 1.0}};
        return new mio_mesh{meshioplusplus::grid(d, o, s, max_cells)};
    });
}

mio_mesh* mio_voxelize(const mio_mesh* mesh, const mio_voxel_opts* opts, int64_t dims_out[3],
                       double origin_out[3], double spacing_out[3], int64_t* num_occupied) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        if (!opts)
            throw meshioplusplus::ReadError(
                "meshio++: voxelize: options are NULL, but exactly one of resolution and "
                "cell_size must be given");
        meshioplusplus::VoxelResult r =
            meshioplusplus::voxelize(mesh->mMesh, capi_voxel_options(*opts));
        for (int k = 0; k < 3; ++k) {
            if (dims_out)
                dims_out[k] = r.mDims[static_cast<std::size_t>(k)];
            if (origin_out)
                origin_out[k] = r.mOrigin[static_cast<std::size_t>(k)];
            if (spacing_out)
                spacing_out[k] = r.mSpacing[static_cast<std::size_t>(k)];
        }
        if (num_occupied)
            *num_occupied = r.mNumOccupied;
        return new mio_mesh{std::move(r.mMesh)};
    });
}

mio_status mio_surface_watertight_check(const mio_mesh* surface, mio_surface_quality* out) {
    return guarded([&]() -> mio_status {
        if (!surface)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: surface is NULL");
        if (!out)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: out is NULL");
        capi_fill_quality(meshioplusplus::surface_watertight_check(surface->mMesh), out);
        return MIO_OK;
    });
}

mio_status mio_sample_distance(const mio_mesh* surface, const double* points, int64_t n_points,
                               const mio_sdf_opts* opts, double* out) {
    return guarded([&]() -> mio_status {
        if (!surface)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: surface is NULL");
        if (!points || !out)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: points/out is NULL");
        if (n_points < 0)
            return fail(MIO_ERR_INVALID_ARG, "meshio++: n_points is negative");
        if (n_points == 0)
            return MIO_OK;
        // A view, not a copy: sample_distance only reads the query coordinates,
        // and the buffer is the caller's for the duration of the call.
        const meshioplusplus::NDArray query = meshioplusplus::NDArray::MakeView(
            meshioplusplus::DType::Float64, {static_cast<std::size_t>(n_points), std::size_t{3}},
            const_cast<std::byte*>(reinterpret_cast<const std::byte*>(points)));
        const meshioplusplus::SurfaceDistanceOptions options =
            opts ? capi_sdf_options(*opts) : meshioplusplus::SurfaceDistanceOptions{};
        const meshioplusplus::NDArray values =
            meshioplusplus::sample_distance(surface->mMesh, query, options);
        std::memcpy(out, values.Data(), values.Nbytes());
        return MIO_OK;
    });
}

mio_mesh* mio_distance_to_surface(const mio_mesh* query, const mio_mesh* surface,
                                  const mio_sdf_opts* opts, int64_t* num_banded,
                                  mio_surface_quality* quality) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!query || !surface)
            throw meshioplusplus::ReadError("meshio++: query/surface mesh is NULL");
        const meshioplusplus::SurfaceDistanceOptions options =
            opts ? capi_sdf_options(*opts) : meshioplusplus::SurfaceDistanceOptions{};
        meshioplusplus::SurfaceDistanceResult r =
            meshioplusplus::distance_to_surface(query->mMesh, surface->mMesh, options);
        if (num_banded)
            *num_banded = r.mNumBanded;
        capi_fill_quality(r.mQuality, quality);
        return new mio_mesh{std::move(r.mMesh)};
    });
}

static_assert(sizeof(mio_curvature_opts) == 80,
              "mio_curvature_opts grew outside its reserved tail");
static_assert(sizeof(mio_curvature_report) == 136,
              "mio_curvature_report grew outside its reserved tail");

void mio_curvature_opts_init(mio_curvature_opts* opts) {
    if (!opts)
        return;
    *opts = mio_curvature_opts{};  // value-initialized: every opt-in off
    opts->mean = 1;
    opts->gaussian = 1;
    opts->dual_area = MIO_CURVATURE_MIXED_VORONOI;
}

mio_mesh* mio_compute_curvature(const mio_mesh* mesh, const mio_curvature_opts* opts,
                                mio_curvature_report* report) {
    return guarded_ptr(static_cast<mio_mesh*>(nullptr), [&]() -> mio_mesh* {
        if (!mesh)
            throw meshioplusplus::ReadError("meshio++: mesh is NULL");
        meshioplusplus::CurvatureOptions options;
        if (opts) {
            options.mMean = opts->mean != 0;
            options.mGaussian = opts->gaussian != 0;
            if (opts->dual_area != MIO_CURVATURE_MIXED_VORONOI &&
                opts->dual_area != MIO_CURVATURE_BARYCENTRIC)
                throw std::invalid_argument(
                    "meshio++: curvature: dual_area must be MIO_CURVATURE_MIXED_VORONOI or "
                    "MIO_CURVATURE_BARYCENTRIC");
            options.mDualArea = opts->dual_area == MIO_CURVATURE_BARYCENTRIC
                                    ? meshioplusplus::CurvatureDualArea::Barycentric
                                    : meshioplusplus::CurvatureDualArea::MixedVoronoi;
            options.mIncludeBoundary = opts->include_boundary != 0;
            options.mRecordArea = opts->record_area != 0;
            options.mRecordPrincipal = opts->record_principal != 0;
            if (opts->region)
                options.mRegion = opts->region;
        }
        meshioplusplus::CurvatureResult r =
            meshioplusplus::compute_curvature(mesh->mMesh, options);
        if (report) {
            *report = mio_curvature_report{};
            capi_fill_quality(r.mQuality, &report->quality);
            report->num_boundary = r.mNumBoundary;
            report->num_isolated = r.mNumIsolated;
            report->num_degenerate = r.mNumDegenerate;
            report->total_angle_defect = r.mTotalAngleDefect;
        }
        return new mio_mesh{std::move(r.mMesh)};
    });
}

int32_t mio_pipeline_has_json(void) {
    return meshioplusplus::pipeline_has_json() ? 1 : 0;
}

}  // extern "C"
