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
 * @file js_bindings.cpp
 * @brief Emscripten/embind entry point for the `@meshioplusplus/wasm` npm
 *        package. Compiled only under `MESHIOPLUSPLUS_BUILD_WASM` (see the
 *        `if(EMSCRIPTEN)` block in the top-level `CMakeLists.txt`).
 *
 * Unlike `bindings/_core.cpp` (which exposes one `<fmt>_read`/`<fmt>_write`
 * pair per format to Python, letting `_helpers.py` own extension-dispatch and
 * `np_conversions.hpp` own zero-copy numpy<->NDArray conversion), this file
 * exposes a small, flat, copy-based JS API: `readMesh`/`writeMesh`/`convert`
 * plus the two pure cell-type metadata tables. There is no zero-copy path
 * here by design -- WASM linear memory and the JS heap are different address
 * spaces, so every value crossing the boundary is copied once (mirroring how
 * the existing pybind11 layer already treats ragged cell blocks: "just copy,
 * it's fine"). `NDArray`/`CellBlock`/`Mesh` are therefore kept entirely
 * internal; JS only ever sees plain objects of typed arrays (see
 * `mesh_to_val`/`val_to_mesh` below for the exact shape). Note: the
 * JS-facing names bound below (`"readMesh"`, `"writeMesh"`, ...) are string
 * literals independent of the C++ function names/symbols on the other side
 * of each `emscripten::function(...)` call, so this file's internal C++
 * identifiers can follow the project's snake_case free-function convention
 * without changing the JS API surface.
 *
 * Format scope: whatever `registry.cpp` was compiled with -- this file has no
 * format table of its own, so it needs no edit when one is added or when an
 * optional dependency comes or goes. Since v8.0.0 that is every format the
 * core has, CGNS/H5M/HMF/MED/Exodus and XDMF's HDF data path included:
 * build/build-wasm-deps.sh produces the wasm32 HDF5/netCDF the build links
 * (see doc/wasm.md). Ambiguous extensions (`.msh` shared
 * by ansys/freefem/gmsh, `.inp` shared by abaqus/ansysinp) require an
 * explicit `format` argument, mirroring Python's `file_format=` kwarg;
 * `.msh` defaults to gmsh and `.inp` to abaqus when `format` is omitted.
 *
 * File I/O goes through Emscripten's virtual filesystem (`Module.FS`,
 * exposed via `-sEXPORTED_RUNTIME_METHODS=['FS']` in CMakeLists.txt) -- JS
 * callers write bytes into the virtual FS themselves before calling
 * `readMesh`, and read them back out after `writeMesh`/`convert`.
 *
 * Exceptions: `meshioplusplus::ReadError`/`WriteError` derive from
 * `std::exception`, so Emscripten's default (exceptions enabled) embind
 * configuration automatically surfaces them as catchable JS errors -- no
 * explicit translator is needed here (unlike `bindings/_core.cpp`'s
 * `py::register_exception_translator`, which exists because CPython has no
 * such automatic C++-exception-to-host-exception bridge).
 */

// System includes
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// External includes
#include <emscripten/bind.h>
#include <emscripten/emscripten.h>
#include <emscripten/val.h>

// Project includes
#include "meshioplusplus/detail/value_io.hpp"
#include "meshioplusplus/exceptions.hpp"
#include "meshioplusplus/formats/cgns.hpp"
#include "meshioplusplus/formats/xdmf_time_series.hpp"
#include "meshioplusplus/mesh.hpp"
#include "meshioplusplus/parallel.hpp"
#include "meshioplusplus/region.hpp"
#include "meshioplusplus/operations/agglomerate.hpp"
#include "meshioplusplus/operations/clean.hpp"
#include "meshioplusplus/operations/conservative_interpolate.hpp"
#include "meshioplusplus/operations/convert_cells.hpp"
#include "meshioplusplus/operations/curvature.hpp"
#include "meshioplusplus/operations/repair.hpp"
#include "meshioplusplus/operations/shrinkwrap.hpp"
#include "meshioplusplus/operations/sobolev_deform.hpp"
#include "meshioplusplus/operations/crop.hpp"
#include "meshioplusplus/operations/data_average.hpp"
#include "meshioplusplus/operations/data_calc.hpp"
#include "meshioplusplus/operations/data_common.hpp"
#include "meshioplusplus/operations/data_condition.hpp"
#include "meshioplusplus/operations/data_info.hpp"
#include "meshioplusplus/operations/data_integrate.hpp"
#include "meshioplusplus/operations/data_manage.hpp"
#include "meshioplusplus/operations/decimate.hpp"
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
#include "meshioplusplus/operations/sdf.hpp"
#include "meshioplusplus/operations/voxelize.hpp"
#include "meshioplusplus/operations/error.hpp"
#include "meshioplusplus/operations/gradient.hpp"
#include "meshioplusplus/operations/hessian.hpp"
#include "meshioplusplus/operations/slice.hpp"
#include "meshioplusplus/operations/smooth.hpp"
#include "meshioplusplus/operations/sniff.hpp"
#include "meshioplusplus/operations/split.hpp"
#include "meshioplusplus/operations/stats.hpp"
#include "meshioplusplus/operations/subdivide.hpp"
#include "meshioplusplus/operations/surface.hpp"
#include "meshioplusplus/operations/transform.hpp"
#include "meshioplusplus/operations/undo_green.hpp"
#include "meshioplusplus/operations/remesh.hpp"
#include "meshioplusplus/operations/remesh_volume.hpp"
#include "meshioplusplus/operations/optimize_volume.hpp"
#include "meshioplusplus/read_options.hpp"
#include "meshioplusplus/registry.hpp"
#include "meshioplusplus/detail/provenance.hpp"
#include "meshioplusplus/skin.hpp"
#include "meshioplusplus/types.hpp"

using emscripten::val;

namespace {

using meshioplusplus::DType;
using meshioplusplus::Mesh;
using meshioplusplus::NDArray;

// ---------------------------------------------------------------------
// Typed-array helpers: copy a C++ buffer into a genuinely JS-owned typed
// array (a transient emscripten::typed_memory_view over Module memory is
// only valid for the duration of this call; `.set()` copies it into `arr`'s
// own backing store immediately, so the returned val outlives the view).
// ---------------------------------------------------------------------

val float64_array_from(const double* pData, std::size_t n) {
    val arr = val::global("Float64Array").new_(n);
    arr.call<void>("set", val(emscripten::typed_memory_view(n, pData)));
    return arr;
}

val int32_array_from(const std::int32_t* pData, std::size_t n) {
    val arr = val::global("Int32Array").new_(n);
    arr.call<void>("set", val(emscripten::typed_memory_view(n, pData)));
    return arr;
}

std::size_t cols_of(const NDArray& rA) {
    return rA.Shape().size() >= 2 ? rA.Shape()[1] : 1;
}

// Any-dtype NDArray -> Float64Array (upcasts ints / lower-precision floats;
// every point/data array in the JS API is double-precision for simplicity).
val ndarray_to_float64_array(const NDArray& rA) {
    return meshioplusplus::detail::dispatch_dtype(rA.Dtype(), [&]<class T>() -> val {
        if constexpr (std::is_same_v<T, double>) {
            return float64_array_from(rA.As<double>(), rA.Size());
        } else {
            std::vector<double> tmp(rA.Size());
            const T* src = rA.As<T>();
            for (std::size_t i = 0; i < rA.Size(); ++i)
                tmp[i] = static_cast<double>(src[i]);
            return float64_array_from(tmp.data(), tmp.size());
        }
    });
}

// Integer-dtype NDArray (mesh connectivity, always Int64 in the C++ core) ->
// Int32Array. Node/point counts for any mesh a browser can reasonably handle
// fit comfortably in 32 bits; Int32Array is far more JS-ergonomic than
// BigInt64Array for typical mesh-processing consumer code.
val ndarray_to_int32_array(const NDArray& rA) {
    std::vector<std::int32_t> tmp(rA.Size());
    meshioplusplus::detail::dispatch_dtype(rA.Dtype(), [&]<class T>() {
        const T* src = rA.As<T>();
        for (std::size_t i = 0; i < rA.Size(); ++i)
            tmp[i] = static_cast<std::int32_t>(src[i]);
    });
    return int32_array_from(tmp.data(), tmp.size());
}

/**
 * @brief Convert a C++ `Mesh` into a plain JS object of typed arrays.
 *
 * Shape: `{ points: Float64Array, dim: number, cells: [{type, data:
 * Int32Array, nodesPerCell}], point_data: {name: Float64Array}, cell_data:
 * {name: Float64Array[]} (one array per cell block, same order as `cells`),
 * field_data: {name: Float64Array} }` -- deliberately mirrors the Python
 * `Mesh`'s structure (points, a list of cell blocks, cell_data as one array
 * per block) for consistency with the rest of meshio++.
 *
 * Data arrays are flat, and a flat typed array carries no shape, so each of
 * the three data maps has a sibling `point_data_components` /
 * `cell_data_components` / `field_data_components` object giving the
 * per-entity width of any array that is not a scalar (the same device
 * `xdmfSeriesWriteDataArrays`' `components` argument uses). An array absent
 * from its `*_components` object has one component; only multi-component
 * arrays get an entry, so a scalar-only mesh emits three empty objects.
 * Before v9.9.0 there was no such metadata and every array round-tripped as
 * a scalar, silently reshaping an `(n,3)` vector field to `(3n,1)`.
 *
 * Ragged blocks cross as flat CSR instead of `data`/`nodesPerCell`: a polygon
 * block as `{data, rowOffsets}` and a polyhedron block as `{data, faceOffsets,
 * cellOffsets}`, discriminated by key presence. See doc/polyhedra.md.
 */
val mesh_to_val(const Mesh& rMesh) {
    val out = val::object();
    const NDArray& points = rMesh.Points();
    out.set("points", ndarray_to_float64_array(points));
    out.set("dim", static_cast<int>(cols_of(points)));

    val cells = val::array();
    for (const auto cb : rMesh.CellRange()) {
        val block = val::object();
        block.set("type", cb.Type());
        if (cb.IsPolyhedron()) {
            // 2-level ragged (cell -> faces -> node ids), flattened to three
            // CSR arrays: `data` is every face's node ids concatenated,
            // `faceOffsets` is each face's start index into `data` (length
            // totalFaces + 1), and `cellOffsets` is each cell's start index
            // into the face list (length numCells + 1). This shape crosses
            // the embind boundary as three flat typed arrays rather than a
            // nested JS array of arrays, which embind has no efficient
            // representation for.
            std::vector<std::int32_t> data;
            std::vector<std::int32_t> face_offsets = {0};
            std::vector<std::int32_t> cell_offsets = {0};
            for (std::size_t c = 0; c < cb.NumCells(); ++c) {
                for (std::size_t f = 0; f < cb.NumFaces(c); ++f) {
                    const auto face = cb.Face(c, f);
                    for (std::size_t k = 0; k < face.second; ++k)
                        data.push_back(static_cast<std::int32_t>(face.first[k]));
                    face_offsets.push_back(static_cast<std::int32_t>(data.size()));
                }
                cell_offsets.push_back(static_cast<std::int32_t>(face_offsets.size() - 1));
            }
            block.set("data", int32_array_from(data.data(), data.size()));
            block.set("faceOffsets", int32_array_from(face_offsets.data(), face_offsets.size()));
            block.set("cellOffsets", int32_array_from(cell_offsets.data(), cell_offsets.size()));
        } else if (cb.IsRagged()) {
            // 1-level ragged (jagged polygon rows), as two CSR arrays: `data`
            // is every row's node ids concatenated, `rowOffsets` is each
            // cell's start index into `data` (length numCells + 1).
            std::vector<std::int32_t> data;
            std::vector<std::int32_t> row_offsets = {0};
            for (std::size_t c = 0; c < cb.NumCells(); ++c) {
                const std::int64_t* row = cb.Row(c);
                const std::size_t row_size = cb.RowSize(c);
                for (std::size_t k = 0; k < row_size; ++k)
                    data.push_back(static_cast<std::int32_t>(row[k]));
                row_offsets.push_back(static_cast<std::int32_t>(data.size()));
            }
            block.set("data", int32_array_from(data.data(), data.size()));
            block.set("rowOffsets", int32_array_from(row_offsets.data(), row_offsets.size()));
        } else {
            const NDArray& conn = cb.Conn();
            block.set("data", ndarray_to_int32_array(conn));
            block.set("nodesPerCell", static_cast<int>(cols_of(conn)));
        }
        cells.call<void>("push", block);
    }
    out.set("cells", cells);

    // Data arrays cross as flat Float64Arrays, with the per-array component
    // count carried alongside in a sibling `*_components` object -- a flat
    // typed array has no shape of its own, exactly the rationale
    // `xdmfSeriesWriteDataArrays`' own `components` argument already states.
    // An entry is written ONLY for a genuinely multi-component array, so a
    // scalar-only mesh emits the same three empty objects it always did and
    // no existing consumer sees a new key. `val_to_mesh` reads them back.
    val point_data = val::object();
    val point_data_components = val::object();
    for (const auto& name : rMesh.PointDataNames()) {
        const NDArray& a = rMesh.PointData(name);
        point_data.set(name, ndarray_to_float64_array(a));
        if (cols_of(a) > 1)
            point_data_components.set(name, static_cast<double>(cols_of(a)));
    }
    out.set("point_data", point_data);
    out.set("point_data_components", point_data_components);

    val cell_data = val::object();
    val cell_data_components = val::object();
    for (const auto& name : rMesh.CellDataNames()) {
        val blocks = val::array();
        for (std::size_t b = 0; b < rMesh.CellDataNumBlocks(name); ++b)
            blocks.call<void>("push", ndarray_to_float64_array(rMesh.CellData(name, b)));
        cell_data.set(name, blocks);
        // The component count is a property of the ARRAY, not of one block:
        // the uniform mesh API guarantees every block of a named cell_data
        // array agrees on its trailing dimensions, so block 0 speaks for all.
        if (rMesh.CellDataNumBlocks(name) > 0 && cols_of(rMesh.CellData(name, 0)) > 1)
            cell_data_components.set(name, static_cast<double>(cols_of(rMesh.CellData(name, 0))));
    }
    out.set("cell_data", cell_data);
    out.set("cell_data_components", cell_data_components);

    val field_data = val::object();
    val field_data_components = val::object();
    for (const auto& name : rMesh.FieldDataNames()) {
        const NDArray& a = rMesh.FieldData(name);
        field_data.set(name, ndarray_to_float64_array(a));
        if (cols_of(a) > 1)
            field_data_components.set(name, static_cast<double>(cols_of(a)));
    }
    out.set("field_data", field_data);
    out.set("field_data_components", field_data_components);

    // Named regions (doc/regions.md) ride on the mesh object itself rather than
    // through a function of their own, so `readMesh` / `writeMesh` / `convert`
    // carry them with no extra call. `entries` is flat: `n` indices for
    // point/cell, `n` (cell, facet) pairs for side.
    val regions = val::array();
    for (std::size_t i = 0; i < rMesh.NumRegions(); ++i) {
        const meshioplusplus::Region& r = rMesh.Region(i);
        val jr = val::object();
        jr.set("name", r.mName);
        jr.set("kind", std::string(meshioplusplus::region_kind_name(r.mKind)));
        jr.set("dim", r.mDim);
        jr.set("tag", static_cast<double>(r.mTag));
        jr.set("entries", ndarray_to_int32_array(r.mEntries));
        regions.call<void>("push", jr);
    }
    out.set("regions", regions);

    return out;
}

// A JS array-like of numbers -> an owning Float64/Int64 NDArray of `shape`.
// `emscripten::vecFromJSArray` copies once into a std::vector; the second
// copy into the NDArray's own buffer is unavoidable without exposing
// NDArray's internals to JS, which the file-level design deliberately avoids.
NDArray float64_ndarray_from_val(const val& rJsArr, std::vector<std::size_t> shape) {
    std::vector<double> tmp = emscripten::vecFromJSArray<double>(rJsArr);
    NDArray out = NDArray::Uninit(DType::Float64, std::move(shape));
    std::copy(tmp.begin(), tmp.end(), out.As<double>());
    return out;
}

NDArray int64_ndarray_from_val(const val& rJsArr, std::vector<std::size_t> shape) {
    std::vector<std::int64_t> tmp = emscripten::vecFromJSArray<std::int64_t>(rJsArr);
    NDArray out = NDArray::Uninit(DType::Int64, std::move(shape));
    std::copy(tmp.begin(), tmp.end(), out.As<std::int64_t>());
    return out;
}

std::vector<std::string> js_object_keys(const val& rObj) {
    val keys = val::global("Object").call<val>("keys", rObj);
    return emscripten::vecFromJSArray<std::string>(keys);
}

/**
 * @brief The declared component count for data array `rName`, or 1.
 *
 * A missing/null/undefined container, or a name absent from it, means "one
 * component" -- so a caller who never heard of `*_components` keeps exactly
 * the pre-v9.9.0 1-D behaviour. Same null-tolerant lookup idiom as
 * `xdmf_series_write_data_arrays_js`, which has carried an explicit
 * `components` argument since the time-series binding landed.
 */
std::size_t js_components_of(const val& rComponents, const std::string& rName) {
    if (rComponents.isNull() || rComponents.isUndefined())
        return 1;
    const val k = rComponents[rName];
    if (k.isUndefined() || k.isNull())
        return 1;
    const double v = k.as<double>();
    if (!(v >= 1.0) || v != std::floor(v))
        throw meshioplusplus::WriteError(
            "meshio++ (wasm): data array '" + rName +
            "' declares a non-integer or non-positive component count");
    return static_cast<std::size_t>(v);
}

/**
 * @brief The `(rows, k)` shape for a flat JS data array of `rLen` values.
 *
 * `k == 1` yields a 1-D `{rLen}` shape rather than `{rLen, 1}`: the two are
 * indistinguishable to `detail::cols()` (which reports 1 for both) but the
 * 1-D form is what every pre-v9.9.0 caller produced, so keeping it makes the
 * scalar path byte-identical rather than merely equivalent.
 */
std::vector<std::size_t> js_data_shape(std::size_t rLen, std::size_t k, const std::string& rName) {
    if (k <= 1)
        return {rLen};
    if (rLen % k != 0)
        throw meshioplusplus::WriteError(
            "meshio++ (wasm): data array '" + rName + "' has length " + std::to_string(rLen) +
            ", which is not a multiple of its declared " + std::to_string(k) + " components");
    return {rLen / k, k};
}

/**
 * @brief Convert a plain JS mesh object (see `mesh_to_val`'s shape) into a C++
 * `Mesh`, for `writeMesh`/`convert`.
 *
 * The optional `*_components` objects are honoured symmetrically with
 * `mesh_to_val`, so a mesh obtained from `readMesh` re-enters C++ with every
 * array's shape intact. Omitting them (or the whole object) means every array
 * is a scalar, which is what every pre-v9.9.0 caller expressed.
 *
 * Ragged blocks are accepted in the same flat-CSR shape `mesh_to_val` emits:
 * `rowOffsets` for a polygon block, `cellOffsets` + `faceOffsets` for a
 * polyhedron block. See doc/polyhedra.md.
 *
 * @throws meshioplusplus::WriteError on malformed input (points/cell-block
 *   lengths not divisible by their declared dim/nodesPerCell, a data array's
 *   length not divisible by its declared component count, a non-positive
 *   /non-integer component count, or CSR offsets that are empty or out of
 *   range).
 */
Mesh val_to_mesh(const val& rObj) {
    Mesh mesh;
    val points_val = rObj["points"];
    auto dim = rObj["dim"].as<std::size_t>();
    auto npts_len = points_val["length"].as<std::size_t>();
    if (dim == 0 || npts_len % dim != 0)
        throw meshioplusplus::WriteError("meshio++ (wasm): points length is not a multiple of dim");
    mesh.AssignPoints(float64_ndarray_from_val(points_val, {npts_len / dim, dim}));

    val cells = rObj["cells"];
    auto ncells = cells["length"].as<unsigned>();
    for (unsigned i = 0; i < ncells; ++i) {
        val block = cells[i];
        std::string type = block["type"].as<std::string>();
        val data_val = block["data"];
        if (block.hasOwnProperty("cellOffsets")) {
            // Polyhedron: three flat CSR arrays (see mesh_to_val) -> the
            // nested vectors AddPolyhedronBlock wants.
            std::vector<std::int64_t> flat = emscripten::vecFromJSArray<std::int64_t>(data_val);
            std::vector<std::int64_t> face_offsets =
                emscripten::vecFromJSArray<std::int64_t>(block["faceOffsets"]);
            std::vector<std::int64_t> cell_offsets =
                emscripten::vecFromJSArray<std::int64_t>(block["cellOffsets"]);
            if (cell_offsets.empty())
                throw meshioplusplus::WriteError("meshio++ (wasm): cell block '" + type +
                                                 "' has an empty cellOffsets");
            std::vector<std::vector<std::vector<std::int64_t>>> cells_of_faces;
            cells_of_faces.reserve(cell_offsets.size() - 1);
            for (std::size_t c = 0; c + 1 < cell_offsets.size(); ++c) {
                std::vector<std::vector<std::int64_t>> faces;
                for (std::int64_t f = cell_offsets[c]; f < cell_offsets[c + 1]; ++f) {
                    if (f < 0 || static_cast<std::size_t>(f) + 1 >= face_offsets.size())
                        throw meshioplusplus::WriteError("meshio++ (wasm): cell block '" + type +
                                                         "' cellOffsets out of range");
                    // faceOffsets is checked here rather than assumed: a
                    // non-monotonic or over-long entry would otherwise build
                    // an out-of-range iterator pair straight into flat. The
                    // polygon branch below has always checked this.
                    const std::int64_t start = face_offsets[static_cast<std::size_t>(f)];
                    const std::int64_t end = face_offsets[static_cast<std::size_t>(f) + 1];
                    if (start < 0 || end < start || static_cast<std::size_t>(end) > flat.size())
                        throw meshioplusplus::WriteError("meshio++ (wasm): cell block '" + type +
                                                         "' faceOffsets out of range");
                    faces.emplace_back(flat.begin() + start, flat.begin() + end);
                }
                cells_of_faces.push_back(std::move(faces));
            }
            mesh.AddPolyhedronBlock(type, std::move(cells_of_faces));
        } else if (block.hasOwnProperty("rowOffsets")) {
            // Polygon: two flat CSR arrays -> jagged rows.
            std::vector<std::int64_t> flat = emscripten::vecFromJSArray<std::int64_t>(data_val);
            std::vector<std::int64_t> row_offsets =
                emscripten::vecFromJSArray<std::int64_t>(block["rowOffsets"]);
            if (row_offsets.empty())
                throw meshioplusplus::WriteError("meshio++ (wasm): cell block '" + type +
                                                 "' has an empty rowOffsets");
            std::vector<std::vector<std::int64_t>> rows;
            rows.reserve(row_offsets.size() - 1);
            for (std::size_t c = 0; c + 1 < row_offsets.size(); ++c) {
                const std::int64_t start = row_offsets[c], end = row_offsets[c + 1];
                if (start < 0 || end < start || static_cast<std::size_t>(end) > flat.size())
                    throw meshioplusplus::WriteError("meshio++ (wasm): cell block '" + type +
                                                     "' rowOffsets out of range");
                rows.emplace_back(flat.begin() + start, flat.begin() + end);
            }
            mesh.AddPolygonBlock(type, std::move(rows));
        } else {
            auto nodes_per_cell = block["nodesPerCell"].as<std::size_t>();
            auto data_len = data_val["length"].as<std::size_t>();
            if (nodes_per_cell == 0 || data_len % nodes_per_cell != 0)
                throw meshioplusplus::WriteError("meshio++ (wasm): cell block '" + type +
                                                 "' data length is not a multiple of nodesPerCell");
            mesh.AddCellBlock(type, int64_ndarray_from_val(
                                        data_val, {data_len / nodes_per_cell, nodes_per_cell}));
        }
    }

    // Data arrays, with their component counts taken from the optional
    // sibling `*_components` objects (see `mesh_to_val`, which emits them).
    // Absent => 1 component, i.e. exactly the pre-v9.9.0 flat-1-D behaviour,
    // which is what keeps every existing caller working unchanged.
    if (rObj.hasOwnProperty("point_data")) {
        val pd = rObj["point_data"];
        val comps = rObj.hasOwnProperty("point_data_components") ? rObj["point_data_components"]
                                                                 : val::undefined();
        for (const std::string& name : js_object_keys(pd)) {
            val arr = pd[name];
            const std::size_t len = arr["length"].as<std::size_t>();
            mesh.AddPointData(name,
                              float64_ndarray_from_val(
                                  arr, js_data_shape(len, js_components_of(comps, name), name)));
        }
    }
    if (rObj.hasOwnProperty("cell_data")) {
        val cd = rObj["cell_data"];
        val comps = rObj.hasOwnProperty("cell_data_components") ? rObj["cell_data_components"]
                                                                : val::undefined();
        for (const std::string& name : js_object_keys(cd)) {
            val blocks_val = cd[name];
            auto nb = blocks_val["length"].as<unsigned>();
            // One component count per ARRAY, applied to every block -- the
            // uniform mesh API requires the blocks of one named cell_data
            // array to agree on their trailing dimensions.
            const std::size_t k = js_components_of(comps, name);
            std::vector<NDArray> blocks;
            blocks.reserve(nb);
            for (unsigned b = 0; b < nb; ++b) {
                val arr = blocks_val[b];
                const std::size_t len = arr["length"].as<std::size_t>();
                blocks.push_back(float64_ndarray_from_val(arr, js_data_shape(len, k, name)));
            }
            mesh.AddCellData(name, std::move(blocks));
        }
    }
    if (rObj.hasOwnProperty("field_data")) {
        val fd = rObj["field_data"];
        val comps = rObj.hasOwnProperty("field_data_components") ? rObj["field_data_components"]
                                                                 : val::undefined();
        for (const std::string& name : js_object_keys(fd)) {
            val arr = fd[name];
            const std::size_t len = arr["length"].as<std::size_t>();
            mesh.AddFieldData(name,
                              float64_ndarray_from_val(
                                  arr, js_data_shape(len, js_components_of(comps, name), name)));
        }
    }
    // Named regions (see `mesh_to_val`). `dim`/`tag` default to -1, meaning
    // "unspecified", so a caller building a plain group need not supply them.
    if (rObj.hasOwnProperty("regions")) {
        val regions = rObj["regions"];
        const auto nregions = regions["length"].as<unsigned>();
        for (unsigned i = 0; i < nregions; ++i) {
            val jr = regions[i];
            meshioplusplus::Region region;
            region.mName = jr["name"].as<std::string>();
            region.mKind = meshioplusplus::region_kind_from_name(jr["kind"].as<std::string>());
            region.mDim = jr.hasOwnProperty("dim") ? jr["dim"].as<int>() : -1;
            region.mTag =
                jr.hasOwnProperty("tag") ? static_cast<std::int64_t>(jr["tag"].as<double>()) : -1;
            val ent = jr["entries"];
            const auto n = ent["length"].as<std::size_t>();
            std::vector<std::size_t> shape;
            if (region.mKind == meshioplusplus::RegionKind::Side)
                shape = {n / 2, 2};
            else
                shape = {n};
            region.mEntries = int64_ndarray_from_val(ent, std::move(shape));
            mesh.AddRegion(std::move(region));
        }
    }
    return mesh;
}

// ---------------------------------------------------------------------
// Format dispatch goes through the shared registry (registry.hpp), the
// JS-side analogue of Python's extension_to_filetypes in _helpers.py --
// bindings/_core.cpp exposes one function per format and leaves dispatch
// entirely to Python, while the flat bindings (this file and bindings/c/)
// share the C++-level tables in src/cpp/src/registry.cpp. Parameterized writers
// get a fixed default there (documented per-entry); per-call overrides are a
// possible future API addition, deliberately out of scope for v1. Under
// Emscripten the HDF5/netCDF-conditional registry entries are compiled out,
// so the WASM format set is exactly the non-HDF5/netCDF one described in the
// file-level docs above.
// ---------------------------------------------------------------------

using meshioplusplus::registry_readers;
using meshioplusplus::registry_writers;
using meshioplusplus::resolve_format;

// " (this build has no HDF5 support)" for extensions like `.med` whose format
// the registry knows but this build compiled out; "" otherwise.
std::string compiled_out_hint(const std::string& rFormat) {
    const char* dep = meshioplusplus::registry_compiled_out(rFormat);
    return dep ? " (this build has no " + std::string(dep) + " support)" : "";
}

// Throw a genuine, message-carrying JS `Error` from C++. Emscripten's
// -fwasm-exceptions support lets a C++ exception unwind out of an exported
// function without aborting the module, but the resulting JS-side value is a
// bare, message-less `WebAssembly.Exception` -- not useful to a JS caller.
// EM_ASM executes an inline JS snippet synchronously; throwing inside it is a
// plain JS throw at that call site, which propagates normally to the caller
// of the exported function (the standard documented technique for surfacing
// a readable message across the boundary).
[[noreturn]] void throw_js_error(const std::string& rMsg) {
    EM_ASM({ throw new Error(UTF8ToString($0)); }, rMsg.c_str());
    __builtin_unreachable();  // EM_ASM's throw never returns; satisfies -Wreturn-type
}

// Wraps the body of an exported function so any meshioplusplus::ReadError/
// WriteError (or other std::exception) becomes a proper JS Error instead of
// a message-less WebAssembly.Exception at the JS boundary.
template <class F>
auto with_js_errors(F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const std::exception& e) {
        throw_js_error(e.what());
    } catch (...) {
        throw_js_error("meshio++ (wasm): unknown error");
    }
}

// ---------------------------------------------------------------------
// The JS-facing API (see wasm/src/index.mjs for the ergonomic wrapper that
// defaults `format` to "" and awaits module instantiation).
// ---------------------------------------------------------------------

/**
 * @brief Read a mesh file from the Emscripten virtual filesystem.
 * @param rPath virtual FS path (write the bytes there first via `Module.FS`).
 * @param rFormat explicit format key (see `registry_extension_defaults()`), or "" to
 *   infer from `rPath`'s extension.
 * @return a plain JS mesh object (see `mesh_to_val`).
 * @throws meshioplusplus::ReadError on an unknown/unsupported format or a
 *   malformed file.
 */
/// @return a JS array of `rItems`. Built with the no-arg `val::array()` plus
/// per-index `set`, matching every other array construction in this file --
/// embind's container-taking overloads vary across Emscripten versions and this
/// form is unambiguously available.
template <class T>
val js_array_of(const std::vector<T>& rItems) {
    val out = val::array();
    for (std::size_t i = 0; i < rItems.size(); ++i)
        out.set(i, rItems[i]);
    return out;
}

/// Resolve a read format, falling back to a content sniff (read paths only).
std::string js_resolve_read_format(const std::string& rPath, const std::string& rFormat) {
    try {
        return resolve_format(rPath, rFormat);
    } catch (const meshioplusplus::ReadError&) {
        std::string sniffed = meshioplusplus::sniff_format(rPath);
        if (sniffed.empty())
            throw;
        return sniffed;
    }
}

val read_mesh(const std::string& rPath, const std::string& rFormat) {
    return with_js_errors([&]() -> val {
        const std::string fmt = js_resolve_read_format(rPath, rFormat);
        auto it = registry_readers().find(fmt);
        if (it == registry_readers().end())
            throw meshioplusplus::ReadError("meshio++ (wasm): unknown or unsupported format '" +
                                            fmt + "'" + compiled_out_hint(fmt));
        return mesh_to_val(it->second(rPath));
    });
}

/**
 * @brief Selective read: geometry only, or only the named data arrays.
 * @param rPath virtual FS path to read.
 * @param rFormat explicit format key, or "" to infer from the extension.
 * @param points_only skip every data array.
 * @param rArrays JS array of names to keep; `null`/`undefined` keeps every
 *   array, an empty array keeps none. That distinction is deliberate.
 * @param time_step which step of a multi-step file to materialize: 0 (the
 *   default) is the first, negative counts from the end. Out of range throws
 *   naming the available count. Honoured by formats carrying a time series
 *   (currently exodus); ignored by the rest.
 * @param lenient downgrade "this reader cannot represent construct X" errors to
 *   a warning plus a skip (currently mdpa's Table/Geometries/Mesh/Constraints
 *   blocks). Not "ignore all errors": a malformed file still throws.
 *
 * Formats without a native selective path are read whole and filtered, so the
 * result is the same either way -- only the cost differs.
 */
val read_mesh_selective(const std::string& rPath, const std::string& rFormat, bool points_only,
                        const val& rArrays, int time_step, bool lenient) {
    return with_js_errors([&]() -> val {
        const std::string fmt = js_resolve_read_format(rPath, rFormat);
        meshioplusplus::ReadOptions opts;
        opts.mPointsOnly = points_only;
        opts.mTimeStep = time_step;
        opts.mLenient = lenient;
        if (!rArrays.isNull() && !rArrays.isUndefined())
            opts.mDataArrays = emscripten::vecFromJSArray<std::string>(rArrays);
        return mesh_to_val(meshioplusplus::registry_read(rPath, fmt, opts));
    });
}

/**
 * @brief Summarize a file without loading its heavy arrays.
 * @return `{numPoints, pointDim, numCells, cellBlocks: [{type, numCells,
 *   nodesPerCell, ragged}], pointDataNames, cellDataNames, fieldDataNames,
 *   format, fellBackToFullRead, bboxMin?, bboxMax?}`.
 *
 * `bboxMin`/`bboxMax` are present only when a box was computed -- omitted
 * rather than null, so "not computed" cannot read as a box at the origin.
 * `fellBackToFullRead` says whether the summary was actually cheap.
 */
val read_metadata_js(const std::string& rPath, const std::string& rFormat) {
    return with_js_errors([&]() -> val {
        const std::string fmt = js_resolve_read_format(rPath, rFormat);
        const meshioplusplus::MeshMetadata meta =
            meshioplusplus::registry_read_metadata(rPath, fmt, meshioplusplus::ReadOptions{});

        val blocks = val::array();
        for (std::size_t i = 0; i < meta.mCellBlocks.size(); ++i) {
            val entry = val::object();
            entry.set("type", meta.mCellBlocks[i].mType);
            entry.set("numCells", static_cast<double>(meta.mCellBlocks[i].mNumCells));
            entry.set("nodesPerCell", static_cast<double>(meta.mCellBlocks[i].mNodesPerCell));
            entry.set("ragged", meta.mCellBlocks[i].mRagged);
            blocks.set(i, entry);
        }

        val out = val::object();
        out.set("numPoints", static_cast<double>(meta.mNumPoints));
        out.set("pointDim", static_cast<double>(meta.mPointDim));
        out.set("numCells", static_cast<double>(meta.NumCells()));
        out.set("cellBlocks", blocks);
        out.set("pointDataNames", js_array_of(meta.mPointDataNames));
        out.set("cellDataNames", js_array_of(meta.mCellDataNames));
        out.set("fieldDataNames", js_array_of(meta.mFieldDataNames));
        out.set("format", meta.mFormat);
        out.set("fellBackToFullRead", meta.mFellBackToFullRead);
        // Always present (empty for a format with no time concept), so a caller
        // can read `.timeValues.length` without first testing for the key. This
        // is the count `readMeshSelective`'s `timeStep` may name.
        out.set("timeValues", js_array_of(meta.mTimeValues));
        // Always present too (empty on a native metadata path, or for a format
        // with no regions), so a caller can iterate without testing the key.
        val regions = val::array();
        for (std::size_t i = 0; i < meta.mRegions.size(); ++i) {
            const meshioplusplus::RegionSummary& r = meta.mRegions[i];
            val entry = val::object();
            entry.set("name", r.mName);
            entry.set("kind", std::string(meshioplusplus::region_kind_name(r.mKind)));
            entry.set("dim", r.mDim);
            entry.set("tag", static_cast<double>(r.mTag));
            entry.set("numEntries", static_cast<double>(r.mNumEntries));
            regions.set(i, entry);
        }
        out.set("regions", regions);
        if (meta.mHasBBox) {
            out.set("bboxMin", js_array_of(std::vector<double>{meta.mBBoxMin[0], meta.mBBoxMin[1],
                                                               meta.mBBoxMin[2]}));
            out.set("bboxMax", js_array_of(std::vector<double>{meta.mBBoxMax[0], meta.mBBoxMax[1],
                                                               meta.mBBoxMax[2]}));
        }
        return out;
    });
}

/// @return whether `rFormat` has a native selective-read path.
bool reader_supports_options_js(const std::string& rFormat) {
    return meshioplusplus::registry_reader_supports_options(rFormat);
}

/**
 * @brief Write a mesh object to the Emscripten virtual filesystem.
 * @param rPath virtual FS path to write (read the bytes back out via
 *   `Module.FS` afterward).
 * @param rMeshObj a plain JS mesh object (see `mesh_to_val`'s shape).
 * @param rFormat explicit format key, or "" to infer from `rPath`'s extension.
 * @throws meshioplusplus::WriteError on an unknown/write-unsupported format
 *   or malformed input.
 */
void write_mesh(const std::string& rPath, const val& rMeshObj, const std::string& rFormat) {
    with_js_errors([&]() {
        std::string fmt = resolve_format(rPath, rFormat);
        auto it = registry_writers().find(fmt);
        if (it == registry_writers().end())
            throw meshioplusplus::WriteError(
                "meshio++ (wasm): unknown, read-only, or unsupported format '" + fmt + "'" +
                compiled_out_hint(fmt));
        // Bound scope-less provenance notes to this write, mirroring Python's
        // public write() -- otherwise a note left by an earlier scope-less
        // operation would leak into this file. No-op inside a caller's own
        // scope. See doc/provenance.md.
        meshioplusplus::detail::provenance_begin_write();
        it->second(rPath, val_to_mesh(rMeshObj));
    });
}

/**
 * @brief Read `inPath` and immediately write it to `outPath` (both on the
 * virtual FS), without round-tripping through a JS object. Mirrors the CLI's
 * `convert` subcommand.
 */
void convert(const std::string& rInPath, const std::string& rInFormat, const std::string& rOutPath,
             const std::string& rOutFormat) {
    with_js_errors([&]() {
        std::string rfmt = resolve_format(rInPath, rInFormat);
        std::string wfmt = resolve_format(rOutPath, rOutFormat);
        auto rit = registry_readers().find(rfmt);
        auto wit = registry_writers().find(wfmt);
        if (rit == registry_readers().end())
            throw meshioplusplus::ReadError(
                "meshio++ (wasm): unknown or unsupported input format '" + rfmt + "'" +
                compiled_out_hint(rfmt));
        if (wit == registry_writers().end())
            throw meshioplusplus::WriteError(
                "meshio++ (wasm): unknown, read-only, or unsupported output format '" + wfmt + "'" +
                compiled_out_hint(wfmt));
        Mesh mesh = rit->second(rInPath);
        meshioplusplus::detail::provenance_begin_write();
        wit->second(rOutPath, std::move(mesh));
    });
}

/**
 * @brief Read `rInPath` and write a *renderable surface* of it to `rOutPath`,
 * without round-tripping through a JS object.
 *
 * The one operation a browser viewer cannot express with the existing
 * bindings. `readMesh` -> `extractSkin` -> `writeMesh` would do the same
 * thing, but the JS mesh representation is flat and drops multi-component
 * (vector/tensor) data, so every such array would be lost on the way to the
 * renderer. Staying inside C++ keeps them.
 *
 * A mesh with skinnable 3D cells becomes its boundary; anything else (a
 * surface, a curve, a point cloud) passes through. Either way the result is
 * linearized, because the target is a triangle/polygon renderer with no
 * concept of a mid-side node -- drawing `triangle6` connectivity verbatim
 * produces visible garbage rather than a curved triangle.
 */
void convert_surface(const std::string& rInPath, const std::string& rInFormat,
                     const std::string& rOutPath, const std::string& rOutFormat) {
    with_js_errors([&]() {
        std::string rfmt = resolve_format(rInPath, rInFormat);
        std::string wfmt = resolve_format(rOutPath, rOutFormat);
        auto rit = registry_readers().find(rfmt);
        auto wit = registry_writers().find(wfmt);
        if (rit == registry_readers().end())
            throw meshioplusplus::ReadError(
                "meshio++ (wasm): unknown or unsupported input format '" + rfmt + "'" +
                compiled_out_hint(rfmt));
        if (wit == registry_writers().end())
            throw meshioplusplus::WriteError(
                "meshio++ (wasm): unknown, read-only, or unsupported output format '" + wfmt + "'" +
                compiled_out_hint(wfmt));

        Mesh mesh = rit->second(rInPath);
        meshioplusplus::ConvertCellsOptions linearize;
        linearize.mMode = meshioplusplus::ConvertCellsMode::Linearize;

        if (meshioplusplus::has_skinnable_cells(mesh)) {
            // record_parent_ids, then gather: extract_surface drops cell data,
            // and colouring a solid by its per-cell tag is the common case.
            Mesh surface = meshioplusplus::extract_surface(mesh, /*recordParentIds=*/true);
            gather_cell_data_onto_surface(mesh, surface);
            // Linearize after gathering; convert_cells carries cell data 1:1.
            surface = meshioplusplus::convert_cells(surface, linearize).mMesh;
            // The parent ids are plumbing, not something to colour by.
            surface = meshioplusplus::data_drop(surface, meshioplusplus::DataLocation::Cell,
                                                {"surface:parent_cell"},
                                                /*ignore_missing=*/true);
            // See write_mesh()'s comment: bound scope-less notes to this
            // write, called after the ops above so any notes THEY raised are
            // dropped rather than misattributed here, matching every other
            // scope-less write entry point.
            meshioplusplus::detail::provenance_begin_write();
            wit->second(rOutPath, surface);
            return;
        }
        Mesh linearized = meshioplusplus::convert_cells(mesh, linearize).mMesh;
        meshioplusplus::detail::provenance_begin_write();
        wit->second(rOutPath, std::move(linearized));
    });
}

namespace {

/**
 * The step dispatch itself lives in the core (`operations/pipeline.hpp`,
 * `apply_pipeline_step`) since v9.11.0 -- this file only converts between the
 * viewer's camelCase `{op, ...params}` objects and the core's PascalCase
 * `PipelineStep` vocabulary. The two casings differ by exactly the first
 * character, so the mapping is mechanical in both directions and the old
 * `apply_one_op` op table cannot drift from the settings.json one: they are
 * the same table.
 */

/// camelCase -> PascalCase (`removeOrphans` -> `RemoveOrphans`).
std::string pascal_case_key(std::string key) {
    if (!key.empty())
        key[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(key[0])));
    return key;
}

/// PascalCase -> camelCase (`PointsWelded` -> `pointsWelded`) for the step
/// report, whose key names predate the PascalCase vocabulary and are a
/// browser-viewer contract (`protocol.ts` `OpStepReport`).
std::string camel_case_key(std::string key) {
    if (!key.empty())
        key[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(key[0])));
    return key;
}

/// One step-parameter value out of a JS value. Arrays must be homogeneous
/// numbers or strings -- the same closed set the JSON front-end accepts.
meshioplusplus::PipelineValue val_to_pipeline_value(const val& rValue, const std::string& rKey) {
    const std::string type = rValue.typeOf().as<std::string>();
    if (type == "boolean")
        return rValue.as<bool>();
    if (type == "number")
        return rValue.as<double>();
    if (type == "string")
        return rValue.as<std::string>();
    if (rValue.isArray()) {
        const unsigned n = rValue["length"].as<unsigned>();
        if (n == 0)
            return std::vector<std::string>{};
        if (rValue[0].typeOf().as<std::string>() == "string") {
            std::vector<std::string> out;
            out.reserve(n);
            for (unsigned i = 0; i < n; ++i)
                out.push_back(rValue[i].as<std::string>());
            return out;
        }
        std::vector<double> out;
        out.reserve(n);
        for (unsigned i = 0; i < n; ++i)
            out.push_back(rValue[i].as<double>());
        return out;
    }
    throw meshioplusplus::ReadError("meshio++ (wasm): op parameter '" + rKey +
                                    "' has an unsupported value type");
}

/// A `{op, ...params}` spec -> the core `PipelineStep`. `null`/`undefined`
/// values mean "use the default", exactly as the old per-key readers did.
meshioplusplus::PipelineStep val_to_step(const val& rSpec) {
    meshioplusplus::PipelineStep step;
    step.mOp = pascal_case_key(rSpec["op"].as<std::string>());
    val keys = val::global("Object").call<val>("keys", rSpec);
    const unsigned n = keys["length"].as<unsigned>();
    for (unsigned i = 0; i < n; ++i) {
        const std::string key = keys[i].as<std::string>();
        if (key == "op")
            continue;
        val v = rSpec[key];
        if (v.isUndefined() || v.isNull())
            continue;
        step.mParams.emplace(pascal_case_key(key), val_to_pipeline_value(v, key));
    }
    return step;
}

/**
 * @brief Apply one operation, appending its counters to `rSteps`.
 *
 * Every operation runs on the **full-dimensional** mesh, before any boundary
 * extraction -- smoothing a solid's skin and smoothing a solid are different
 * things, and the second is what the user asked for.
 */
Mesh apply_one_op(Mesh mesh, const val& rSpec, val& rSteps, val& rWarnings) {
    // Echo the caller's own op spelling in the report; counters go back in
    // camelCase. Both are the pre-v9.11.0 contract, unchanged.
    const std::string op = rSpec["op"].as<std::string>();
    meshioplusplus::PipelineStep step = val_to_step(rSpec);
    {
        // The unknown-op error is a contract too: it must carry the CALLER'S
        // spelling ("teleport"), not the PascalCase one the converter made.
        bool known = false;
        for (const auto& entry : meshioplusplus::pipeline_op_table())
            known = known || entry.first == step.mOp;
        if (!known)
            throw meshioplusplus::ReadError("meshio++ (wasm): unknown operation '" + op + "'");
    }
    meshioplusplus::PipelineReport report;
    mesh = meshioplusplus::apply_pipeline_step(std::move(mesh), std::move(step), report);
    for (const auto& entry : report.mSteps) {
        val step = val::object();
        step.set("op", op);
        for (const auto& counter : entry.mCounters)
            step.set(camel_case_key(counter.first), counter.second);
        rSteps.call<void>("push", step);
    }
    for (const auto& warning : report.mWarnings)
        rWarnings.call<void>("push", warning);
    return mesh;
}

}  // namespace

/**
 * @brief Read `rInPath`, apply an operation pipeline, and write a renderable
 * surface to `rOutPath` -- all inside C++.
 *
 * This exists because every mesh operation in the JS API takes and returns a
 * JS `Mesh`, whose flat representation cannot carry multi-component
 * (vector/tensor) arrays. Chaining operations through that API would silently
 * destroy exactly the data `convertSurface` goes out of its way to preserve.
 * Here no mesh ever crosses the boundary, so nothing is lost.
 *
 * An **empty** pipeline is exactly `convertSurface`, which is deliberate: a
 * viewer can call this for both the plain display and the post-operation
 * display, so the two cannot drift apart. It is also what makes undo exact --
 * replaying a shortened pipeline from the original file needs no inverse
 * operations and no snapshots.
 *
 * @param rOps JS array of `{op, ...params}`; see the `OpSpec` union in
 *   `wasm/index.d.ts`.
 * @param keepProvenance keep `surface:parent_cell` in the output (the picker
 *   needs it; the colour-by menu must filter it out).
 * @return `{steps: [{op, ...counters}], warnings: [string]}`.
 */
val convert_surface_ops(const std::string& rInPath, const std::string& rInFormat,
                        const std::string& rOutPath, const std::string& rOutFormat, const val& rOps,
                        bool keepProvenance) {
    return with_js_errors([&]() -> val {
        std::string rfmt = resolve_format(rInPath, rInFormat);
        std::string wfmt = resolve_format(rOutPath, rOutFormat);
        auto rit = registry_readers().find(rfmt);
        auto wit = registry_writers().find(wfmt);
        if (rit == registry_readers().end())
            throw meshioplusplus::ReadError(
                "meshio++ (wasm): unknown or unsupported input format '" + rfmt + "'" +
                compiled_out_hint(rfmt));
        if (wit == registry_writers().end())
            throw meshioplusplus::WriteError(
                "meshio++ (wasm): unknown, read-only, or unsupported output format '" + wfmt + "'" +
                compiled_out_hint(wfmt));

        val steps = val::array();
        val warnings = val::array();

        Mesh mesh = rit->second(rInPath);
        const unsigned n = rOps["length"].as<unsigned>();
        for (unsigned i = 0; i < n; ++i)
            mesh = apply_one_op(std::move(mesh), rOps[i], steps, warnings);

        // From here on this is `convert_surface`'s tail, unchanged.
        meshioplusplus::ConvertCellsOptions linearize;
        linearize.mMode = meshioplusplus::ConvertCellsMode::Linearize;

        if (meshioplusplus::has_skinnable_cells(mesh)) {
            Mesh surface = meshioplusplus::extract_surface(mesh, /*recordParentIds=*/true);
            gather_cell_data_onto_surface(mesh, surface);
            surface = meshioplusplus::convert_cells(surface, linearize).mMesh;
            if (!keepProvenance)
                surface = meshioplusplus::data_drop(surface, meshioplusplus::DataLocation::Cell,
                                                    {"surface:parent_cell"},
                                                    /*ignore_missing=*/true);
            // See write_mesh()'s comment: bound scope-less notes to this
            // write, called after the pipeline steps and surface ops above so
            // any notes THEY raised are dropped rather than misattributed
            // here, matching every other scope-less write entry point.
            meshioplusplus::detail::provenance_begin_write();
            wit->second(rOutPath, surface);
        } else {
            Mesh linearized = meshioplusplus::convert_cells(mesh, linearize).mMesh;
            meshioplusplus::detail::provenance_begin_write();
            wit->second(rOutPath, std::move(linearized));
        }

        val out = val::object();
        out.set("steps", steps);
        out.set("warnings", warnings);
        return out;
    });
}

namespace {

/// Strict key check for the settings object, mirroring the JSON front-end's
/// rule: an unknown key is an error naming it, never silently ignored.
void check_settings_keys(const val& rObject, const char* pWhere,
                         std::initializer_list<const char*> rAllowed) {
    val keys = val::global("Object").call<val>("keys", rObject);
    const unsigned n = keys["length"].as<unsigned>();
    for (unsigned i = 0; i < n; ++i) {
        const std::string key = keys[i].as<std::string>();
        const bool known =
            std::any_of(rAllowed.begin(), rAllowed.end(), [&](const char* k) { return key == k; });
        if (!known)
            throw meshioplusplus::ReadError("meshio++ (wasm): unknown key '" + key + "' in " +
                                            pWhere);
    }
}

std::string settings_string(const val& rObject, const char* pKey, const char* pWhere,
                            bool required = false) {
    val v = rObject[pKey];
    if (v.isUndefined() || v.isNull()) {
        if (required)
            throw meshioplusplus::ReadError(std::string("meshio++ (wasm): ") + pWhere + "." + pKey +
                                            " is required");
        return "";
    }
    return v.as<std::string>();
}

/// A list of strings from a JS array (or a single string), for the settings
/// converter and the sequence entry points.
std::vector<std::string> val_to_string_list(const val& rValue) {
    std::vector<std::string> out;
    if (rValue.isUndefined() || rValue.isNull())
        return out;
    if (rValue.isString()) {
        out.push_back(rValue.as<std::string>());
        return out;
    }
    const unsigned n = rValue["length"].as<unsigned>();
    out.reserve(n);
    for (unsigned i = 0; i < n; ++i)
        out.push_back(rValue[i].as<std::string>());
    return out;
}

/// `{format, times, timeFrom, sort}` (all optional) + a source -> a
/// `SequenceInput`. `rSource` is a glob pattern string or an array of MEMFS
/// paths; a string containing `*`/`?` is a pattern, anything else a single
/// path -- the same rule the CLIs use.
meshioplusplus::SequenceInput val_to_sequence_input(const val& rSource, const val& rOptions) {
    meshioplusplus::SequenceInput in;
    if (rSource.isString()) {
        const std::string text = rSource.as<std::string>();
        if (text.find('*') != std::string::npos || text.find('?') != std::string::npos)
            in.mPattern = text;
        else
            in.mPaths.push_back(text);
    } else {
        in.mPaths = val_to_string_list(rSource);
        if (in.mPaths.empty())
            throw meshioplusplus::ReadError(
                "meshio++ (wasm): the sequence input must be a pattern string or a "
                "non-empty array of paths");
    }
    if (rOptions.isUndefined() || rOptions.isNull())
        return in;
    check_settings_keys(rOptions, "the sequence options", {"format", "times", "timeFrom", "sort"});
    in.mFormat = settings_string(rOptions, "format", "the sequence options");
    val times = rOptions["times"];
    if (!times.isUndefined() && !times.isNull()) {
        const unsigned n = times["length"].as<unsigned>();
        for (unsigned i = 0; i < n; ++i)
            in.mTimes.push_back(times[i].as<double>());
    }
    in.mTimeFrom = meshioplusplus::sequence_time_from_name(
        settings_string(rOptions, "timeFrom", "the sequence options"));
    val sort = rOptions["sort"];
    if (!sort.isUndefined() && !sort.isNull())
        in.mSortExplicit = sort.as<bool>();
    return in;
}

/// A parsed settings object, plus whether it used any sequence-only key.
struct SettingsDocument {
    meshioplusplus::SequencePipeline mSeq;
    bool mSequenceKeys = false;
};

/// A PascalCase settings object (the parsed form of a settings.json) -> the
/// core `SequencePipeline`. The wasm build carries no JSON parser --
/// `JSON.parse` is free on this platform, so the wrapper in
/// `src/wasm/src/index.mjs` parses text and this converter only walks the
/// already-parsed object.
///
/// ONE converter for both document shapes, exactly as the C++ JSON front-end
/// does: `run_pipeline_js` projects it down to the single-mesh `Pipeline` when
/// no sequence key was used, so a plain document takes the physically
/// unchanged path and a transient one is routed to the sequence driver.
SettingsDocument val_to_settings(const val& rSettings) {
    check_settings_keys(
        rSettings, "the settings object",
        {"Version", "Input", "Operations", "Output", "Mode", "Parallel", "Workers"});
    SettingsDocument parsed;
    meshioplusplus::SequencePipeline& pipeline = parsed.mSeq;
    val version = rSettings["Version"];
    if (!version.isUndefined() && !version.isNull())
        pipeline.mVersion = static_cast<int>(version.as<double>());

    val mode = rSettings["Mode"];
    if (!mode.isUndefined() && !mode.isNull()) {
        parsed.mSequenceKeys = true;
        pipeline.mMode = meshioplusplus::sequence_mode_from_name(mode.as<std::string>());
    }
    val parallel = rSettings["Parallel"];
    if (!parallel.isUndefined() && !parallel.isNull()) {
        parsed.mSequenceKeys = true;
        pipeline.mParallel = parallel.as<bool>();
    }
    val workers = rSettings["Workers"];
    if (!workers.isUndefined() && !workers.isNull()) {
        parsed.mSequenceKeys = true;
        pipeline.mWorkers = static_cast<int>(workers.as<double>());
    }

    val input = rSettings["Input"];
    if (input.isUndefined() || input.isNull())
        throw meshioplusplus::ReadError("meshio++ (wasm): the settings object needs 'Input'");
    check_settings_keys(input, "Input",
                        {"Path", "Format", "Options", "Pattern", "Paths", "Times", "TimeFrom"});
    val path = input["Path"];
    val pattern = input["Pattern"];
    val paths = input["Paths"];
    const int given = (path.isUndefined() || path.isNull() ? 0 : 1) +
                      (pattern.isUndefined() || pattern.isNull() ? 0 : 1) +
                      (paths.isUndefined() || paths.isNull() ? 0 : 1);
    if (given == 0)
        throw meshioplusplus::ReadError(
            "meshio++ (wasm): Input.Path is required (or Input.Pattern / Input.Paths)");
    if (given > 1)
        throw meshioplusplus::ReadError(
            "meshio++ (wasm): Input names more than one source; set exactly one of Path, "
            "Pattern or Paths");
    if (given == 1 && !(path.isUndefined() || path.isNull())) {
        pipeline.mInput.mPaths.push_back(settings_string(input, "Path", "Input", true));
    } else if (!(pattern.isUndefined() || pattern.isNull())) {
        parsed.mSequenceKeys = true;
        pipeline.mInput.mPattern = settings_string(input, "Pattern", "Input", true);
    } else {
        parsed.mSequenceKeys = true;
        pipeline.mInput.mPaths = val_to_string_list(paths);
        if (pipeline.mInput.mPaths.empty())
            throw meshioplusplus::ReadError(
                "meshio++ (wasm): Input.Paths must be a non-empty array of strings");
    }
    val times = input["Times"];
    if (!times.isUndefined() && !times.isNull()) {
        parsed.mSequenceKeys = true;
        const unsigned n = times["length"].as<unsigned>();
        for (unsigned i = 0; i < n; ++i)
            pipeline.mInput.mTimes.push_back(times[i].as<double>());
    }
    val time_from = input["TimeFrom"];
    if (!time_from.isUndefined() && !time_from.isNull()) {
        parsed.mSequenceKeys = true;
        pipeline.mInput.mTimeFrom =
            meshioplusplus::sequence_time_from_name(time_from.as<std::string>());
    }
    pipeline.mInput.mFormat = settings_string(input, "Format", "Input");
    val opts = input["Options"];
    if (!opts.isUndefined() && !opts.isNull()) {
        check_settings_keys(opts, "Input.Options",
                            {"PointsOnly", "DataArrays", "TimeStep", "Lenient", "Mmap"});
        val points_only = opts["PointsOnly"];
        if (!points_only.isUndefined() && !points_only.isNull())
            pipeline.mInput.mOptions.mPointsOnly = points_only.as<bool>();
        val lenient = opts["Lenient"];
        if (!lenient.isUndefined() && !lenient.isNull())
            pipeline.mInput.mOptions.mLenient = lenient.as<bool>();
        val time_step = opts["TimeStep"];
        if (!time_step.isUndefined() && !time_step.isNull())
            pipeline.mInput.mOptions.mTimeStep = static_cast<int>(time_step.as<double>());
        val arrays = opts["DataArrays"];
        if (!arrays.isUndefined() && !arrays.isNull()) {
            std::vector<std::string> names;
            const unsigned n = arrays["length"].as<unsigned>();
            names.reserve(n);
            for (unsigned i = 0; i < n; ++i)
                names.push_back(arrays[i].as<std::string>());
            pipeline.mInput.mOptions.mDataArrays = std::move(names);
        }
        pipeline.mInput.mOptions.mMmap =
            meshioplusplus::pipeline_mmap_from_name(settings_string(opts, "Mmap", "Input.Options"));
    }

    val output = rSettings["Output"];
    if (output.isUndefined() || output.isNull())
        throw meshioplusplus::ReadError("meshio++ (wasm): the settings object needs 'Output'");
    check_settings_keys(output, "Output", {"Path", "Format", "Encoding", "Codec", "FloatFormat"});
    pipeline.mOutput.mPath = settings_string(output, "Path", "Output", /*required=*/true);
    pipeline.mOutput.mFormat = settings_string(output, "Format", "Output");
    pipeline.mOutput.mOptions.mEncoding =
        meshioplusplus::pipeline_encoding_from_name(settings_string(output, "Encoding", "Output"));
    const std::string codec = settings_string(output, "Codec", "Output");
    if (!codec.empty()) {
        pipeline.mOutput.mOptions.mCodec = meshioplusplus::pipeline_codec_from_name(codec);
        pipeline.mOutput.mOptions.mCodecSet = true;
    }
    pipeline.mOutput.mOptions.mFloatFormat = settings_string(output, "FloatFormat", "Output");

    val ops = rSettings["Operations"];
    if (!ops.isUndefined() && !ops.isNull()) {
        const unsigned n = ops["length"].as<unsigned>();
        for (unsigned i = 0; i < n; ++i) {
            val spec = ops[i];
            val op_name = spec["Op"];
            if (op_name.isUndefined() || op_name.isNull())
                throw meshioplusplus::ReadError("meshio++ (wasm): Operations[" + std::to_string(i) +
                                                "] needs 'Op'");
            meshioplusplus::PipelineStep step;
            step.mOp = op_name.as<std::string>();
            val keys = val::global("Object").call<val>("keys", spec);
            const unsigned nk = keys["length"].as<unsigned>();
            for (unsigned k = 0; k < nk; ++k) {
                const std::string key = keys[k].as<std::string>();
                if (key == "Op")
                    continue;
                val v = spec[key];
                if (v.isUndefined() || v.isNull())
                    continue;
                step.mParams.emplace(key, val_to_pipeline_value(v, key));
            }
            meshioplusplus::validate_pipeline_step(step);
            pipeline.mSteps.push_back(std::move(step));
        }
    }
    return parsed;
}

/// Project a parsed document down to the single-mesh model. Only called when
/// no sequence key was used, so nothing can be lost here.
meshioplusplus::Pipeline settings_to_pipeline(const SettingsDocument& rParsed) {
    meshioplusplus::Pipeline out;
    out.mVersion = rParsed.mSeq.mVersion;
    out.mInput.mPath = rParsed.mSeq.mInput.mPaths.empty() ? "" : rParsed.mSeq.mInput.mPaths[0];
    out.mInput.mFormat = rParsed.mSeq.mInput.mFormat;
    out.mInput.mOptions = rParsed.mSeq.mInput.mOptions;
    out.mSteps = rParsed.mSeq.mSteps;
    out.mOutput.mPath = rParsed.mSeq.mOutput.mPath;
    out.mOutput.mFormat = rParsed.mSeq.mOutput.mFormat;
    out.mOutput.mOptions = rParsed.mSeq.mOutput.mOptions;
    return out;
}

/// The shared `{steps, warnings}` report shape.
val report_to_val(const meshioplusplus::PipelineReport& rReport) {
    val steps = val::array();
    for (const auto& entry : rReport.mSteps) {
        val step = val::object();
        step.set("op", entry.mOp);
        for (const auto& counter : entry.mCounters)
            step.set(counter.first, counter.second);
        steps.call<void>("push", step);
    }
    val warnings = val::array();
    for (const auto& warning : rReport.mWarnings)
        warnings.call<void>("push", warning);
    val out = val::object();
    out.set("steps", steps);
    out.set("warnings", warnings);
    return out;
}

}  // namespace

/**
 * @brief Run a whole settings pipeline (PascalCase settings object; see
 * `doc/pipeline.md`) against MEMFS paths.
 *
 * The wrapper accepts an object, a JSON string, or a MEMFS `.json` path --
 * string forms are `JSON.parse`d in `src/wasm/src/index.mjs`, so this build
 * needs no nlohmann and `wasm.yml` needs no submodule.
 *
 * @return `{steps: [{op, ...counters}], warnings: [string]}` -- `op` and the
 * counter keys are the canonical PascalCase vocabulary here, unlike
 * `convertSurfaceOps`' pre-existing camelCase report.
 */
val run_pipeline_js(const val& rSettings) {
    return with_js_errors([&]() -> val {
        const SettingsDocument parsed = val_to_settings(rSettings);
        // A document using no sequence key and naming a plain output IS a
        // single-mesh run and takes the unchanged path; anything else is
        // routed to the sequence driver, so nobody has to know which kind of
        // document they hold (the same rule both CLIs and Python follow).
        if (!parsed.mSequenceKeys &&
            !meshioplusplus::sequence_input_needs_driver(parsed.mSeq.mInput, parsed.mSeq.mOutput))
            return report_to_val(meshioplusplus::run_pipeline(settings_to_pipeline(parsed)));
        return report_to_val(meshioplusplus::run_sequence_pipeline(parsed.mSeq));
    });
}

/**
 * @brief The ordered plan for a sequence: which MEMFS files, which step inside
 * each, and each step's time value.
 *
 * Reads no heavy data. `source` is a glob pattern (`*` and `?` only; the
 * directory part is literal) or an array of paths; `options` is
 * `{format, times, timeFrom, sort}`.
 *
 * @return `[{path, step, time, timeSource}]` in natural-numeric order, so
 * `out_9.vtu` precedes `out_10.vtu`. `timeSource` is `"explicit"`, `"file"`,
 * `"filename"` or `"index"` -- reported rather than left to be guessed.
 */
val sequence_entries_js(const val& rSource, const val& rOptions) {
    return with_js_errors([&]() -> val {
        const std::vector<meshioplusplus::SequenceEntry> entries =
            meshioplusplus::sequence_expand(val_to_sequence_input(rSource, rOptions));
        val out = val::array();
        for (const meshioplusplus::SequenceEntry& e : entries) {
            val entry = val::object();
            entry.set("path", e.mPath);
            entry.set("step", static_cast<double>(e.mStep));
            entry.set("time", e.mTime);
            entry.set("timeSource",
                      std::string(meshioplusplus::sequence_time_source_name(e.mTimeSource)));
            out.call<void>("push", entry);
        }
        return out;
    });
}

/**
 * @brief Fan-in: every step of `source` into one multi-step file at `outPath`.
 *
 * Streams -- one mesh alive at a time, whatever the step count. A format that
 * cannot hold a series throws naming itself and pointing at `{step}`, never a
 * silent truncation to the first step.
 *
 * @return the number of steps written.
 */
double sequence_to_timeseries_js(const val& rSource, const std::string& rOutPath,
                                 const std::string& rOutFormat, const val& rOptions) {
    return with_js_errors([&]() -> double {
        meshioplusplus::SequenceInput in = val_to_sequence_input(rSource, rOptions);
        meshioplusplus::SequenceOutput out;
        out.mPath = rOutPath;
        out.mFormat = rOutFormat;
        const std::size_t n = meshioplusplus::sequence_expand(in).size();
        meshioplusplus::sequence_to_timeseries(in, out);
        return static_cast<double>(n);
    });
}

/**
 * @brief Fan-out: each step of the multi-step file `inPath` to `outPattern`,
 * which must contain `{step}` or `{index}`. Streams likewise.
 *
 * @return the MEMFS paths written, so the caller can read them straight back.
 */
val timeseries_to_sequence_js(const std::string& rInPath, const std::string& rInFormat,
                              const std::string& rOutPattern, const std::string& rOutFormat) {
    return with_js_errors([&]() -> val {
        meshioplusplus::SequenceOutput out;
        out.mPath = rOutPattern;
        out.mFormat = rOutFormat;
        meshioplusplus::ReadOptions opts;
        meshioplusplus::timeseries_to_sequence(rInPath, rInFormat, opts, out);
        // Recover the written names the same way the driver produced them.
        meshioplusplus::SequenceInput in;
        in.mPaths = {rInPath};
        in.mFormat = rInFormat;
        const std::size_t n = meshioplusplus::sequence_expand(in).size();
        val paths = val::array();
        for (std::size_t i = 0; i < n; ++i)
            paths.call<void>("push", meshioplusplus::sequence_expand_pattern(rOutPattern, i, n));
        return paths;
    });
}

/** @brief The shared `num_nodes_per_cell` metadata table, as a plain JS object. */
val num_nodes_per_cell_js() {
    val out = val::object();
    for (const auto& kv : meshioplusplus::num_nodes_per_cell())
        out.set(kv.first, kv.second);
    return out;
}

/** @brief The shared `topological_dimension` metadata table, as a plain JS object. */
val topological_dimension_js() {
    val out = val::object();
    for (const auto& kv : meshioplusplus::topological_dimension())
        out.set(kv.first, kv.second);
    return out;
}

/** @brief The compile-time mesh backend ("native" for the shipped wasm build). */
std::string mesh_backend_js() {
    return meshioplusplus::mesh_backend_name();
}

/**
 * @brief The compile-time parallel backend ("seq" for the sequential wasm
 * artifact, "openmp" for the threaded meshioplusplus_wasm_mt one). Lets a
 * consumer -- and the smoke test -- confirm which of the two artifacts loaded.
 */
std::string parallel_backend_js() {
    return meshioplusplus::parallel_backend_name();
}

/**
 * @brief Whether the optional cgnslib (CGNS MLL) backend is linked in.
 *
 * CGNS itself works either way -- meshio++ reads and writes it over raw HDF5,
 * polyhedral `NGON_n`/`NFACE_n` sections included. What this reports is whether
 * **ADF-backed containers** (which are not HDF5 at all) and the **CGNS 3.x**
 * section layout are reachable. Exposed so the smoke test can *assert* the
 * artifact was linked against it: without a probe, a build that silently
 * dropped the dependency reads identically for every file we produce ourselves,
 * and the regression would surface only on a user's ADF file.
 */
bool has_cgnslib_js() {
#ifdef MESHIOPLUSPLUS_HAS_HDF5
    return meshioplusplus::cgns_has_cgnslib();
#else
    return false;
#endif
}

/**
 * @brief The format names this build can actually read and write.
 *
 * `{readers: [...], writers: [...]}`, both sorted (the registry tables are
 * `std::map`). This is what a caller needs to build a file-picker filter or a
 * "convert to" menu without hardcoding a table that silently drifts from the
 * build -- the HDF5/netCDF-backed formats are absent from the wasm build, and
 * a few formats are read-only or write-only.
 */
val available_formats_js() {
    val readers = val::array();
    val writers = val::array();
    unsigned i = 0;
    for (const auto& kv : registry_readers())
        readers.set(i++, kv.first);
    i = 0;
    for (const auto& kv : registry_writers())
        writers.set(i++, kv.first);
    val out = val::object();
    out.set("readers", readers);
    out.set("writers", writers);
    return out;
}

// ---------------------------------------------------------------------
// Mesh operations (computations on a mesh, not file formats).
// ---------------------------------------------------------------------

/** @brief Extract the boundary (volume -> faces, surface -> edges). */
val extract_surface_js(const val& rMeshObj, bool recordParentIds) {
    return with_js_errors([&]() -> val {
        return mesh_to_val(meshioplusplus::extract_surface(val_to_mesh(rMeshObj), recordParentIds));
    });
}

/** @brief Extract the boundary skin of a volume mesh. */
val extract_skin_js(const val& rMeshObj, bool linearize) {
    return with_js_errors([&]() -> val {
        return mesh_to_val(meshioplusplus::extract_skin(val_to_mesh(rMeshObj), linearize));
    });
}

/** @brief Attach per-cell quality metrics as cell_data. */
val attach_quality_js(const val& rMeshObj) {
    return with_js_errors([&]() -> val {
        return mesh_to_val(meshioplusplus::attach_quality(val_to_mesh(rMeshObj)));
    });
}

/** @brief Guess a mesh file's format from its contents ("" if undetermined). */
std::string sniff_format_js(const std::string& rPath) {
    return with_js_errors([&]() -> std::string { return meshioplusplus::sniff_format(rPath); });
}

/**
 * @brief Renumber a mesh (method: "rcm", "morton", or "hilbert"). Returns an
 * object `{mesh, nodePermutation, cellPermutations}` (permutations old->new).
 */
val reorder_js(const val& rMeshObj, const std::string& rMethod) {
    return with_js_errors([&]() -> val {
        meshioplusplus::ReorderResult res = meshioplusplus::reorder(
            val_to_mesh(rMeshObj), meshioplusplus::reorder_method_from_name(rMethod));
        val out = val::object();
        out.set("mesh", mesh_to_val(res.mMesh));
        out.set("nodePermutation", ndarray_to_int32_array(res.mNodePermutation));
        val cell_perms = val::array();
        for (const NDArray& a : res.mCellPermutations)
            cell_perms.call<void>("push", ndarray_to_int32_array(a));
        out.set("cellPermutations", cell_perms);
        return out;
    });
}

/**
 * @brief Merge a JS array of mesh objects into one (concatenate, optional
 * welding). `dataPolicy` is "intersection" (default) or "fill". Returns a plain
 * JS mesh object (point_sets/cell_sets are not carried, as elsewhere in JS).
 */
val merge_js(const val& rMeshes, bool weld, double atol, bool sourceTag,
             const std::string& rDataPolicy, bool dropDuplicateCells) {
    return with_js_errors([&]() -> val {
        const unsigned n = rMeshes["length"].as<unsigned>();
        std::vector<Mesh> owned;
        owned.reserve(n);
        for (unsigned i = 0; i < n; ++i)
            owned.push_back(val_to_mesh(rMeshes[i]));
        std::vector<const Mesh*> ptrs;
        ptrs.reserve(owned.size());
        for (const Mesh& mm : owned)
            ptrs.push_back(&mm);
        meshioplusplus::MergeOptions opts;
        opts.weld = weld;
        opts.atol = atol;
        opts.source_tag = sourceTag;
        opts.drop_duplicate_cells = dropDuplicateCells;
        opts.data_policy = (rDataPolicy == "fill") ? meshioplusplus::MergeDataPolicy::Fill
                                                   : meshioplusplus::MergeDataPolicy::Intersection;
        return mesh_to_val(meshioplusplus::merge(ptrs, opts).mMesh);
    });
}

/**
 * @brief Apply an affine transform to a mesh's point coordinates. `matrix` is a
 * JS array of 16 numbers (row-major 4x4). Returns the transformed mesh object.
 */
val transform_js(const val& rMeshObj, const val& rMatrix, bool rotateVectorData) {
    return with_js_errors([&]() -> val {
        const unsigned n = rMatrix["length"].as<unsigned>();
        if (n != 16)
            throw meshioplusplus::WriteError("transform: matrix must have 16 elements (4x4)");
        double m[16];
        for (unsigned i = 0; i < 16; ++i)
            m[i] = rMatrix[i].as<double>();
        meshioplusplus::AffineTransform xf = meshioplusplus::transform_from_matrix(m);
        return mesh_to_val(meshioplusplus::transform(val_to_mesh(rMeshObj), xf, rotateVectorData));
    });
}

/**
 * @brief Clean a mesh (weld / prune / de-dup). Returns an object
 * `{mesh, pointsWelded, pointsRemovedOrphan, cellsDroppedDegenerate,
 * cellsDroppedDuplicate}` (sets/maps are not carried, as elsewhere in JS).
 */
val clean_js(const val& rMeshObj, bool weld, double atol, bool removeOrphans, bool dropDegenerate,
             bool dropDuplicateCells) {
    return with_js_errors([&]() -> val {
        meshioplusplus::CleanOptions opts;
        opts.weld = weld;
        opts.atol = atol;
        opts.remove_orphans = removeOrphans;
        opts.drop_degenerate = dropDegenerate;
        opts.drop_duplicate_cells = dropDuplicateCells;
        meshioplusplus::CleanResult r = meshioplusplus::clean(val_to_mesh(rMeshObj), opts);
        val out = val::object();
        out.set("mesh", mesh_to_val(r.mMesh));
        out.set("pointsWelded", static_cast<double>(r.mPointsWelded));
        out.set("pointsRemovedOrphan", static_cast<double>(r.mPointsRemovedOrphan));
        out.set("cellsDroppedDegenerate", static_cast<double>(r.mCellsDroppedDegenerate));
        out.set("cellsDroppedDuplicate", static_cast<double>(r.mCellsDroppedDuplicate));
        return out;
    });
}

/**
 * @brief Smooth a mesh's point coordinates (`"laplacian"` / `"taubin"`),
 * leaving connectivity and every data array untouched. A negative `lambda`
 * means "this method's own default" (0.5 Laplacian, 0.33 Taubin) and is passed
 * through unchanged. Returns an object `{mesh, numNodesMoved, maxDisplacement,
 * numSkippedInversion}`. The caller-supplied frozen-node mask is not exposed
 * here, as on the other flat bindings.
 */
val smooth_js(const val& rMeshObj, const std::string& rMethod, int iterations, double lambda,
              double mu, bool fixBoundary, bool preserveFeatures, double featureAngle,
              bool guardInversion) {
    return with_js_errors([&]() -> val {
        meshioplusplus::SmoothOptions options;
        options.mMethod = meshioplusplus::smooth_method_from_name(rMethod);
        options.mIterations = iterations;
        options.mLambda = lambda;
        options.mMu = mu;
        options.mFixBoundary = fixBoundary;
        options.mPreserveFeatures = preserveFeatures;
        options.mFeatureAngleDeg = featureAngle;
        options.mGuardInversion = guardInversion;
        meshioplusplus::SmoothResult r = meshioplusplus::smooth(val_to_mesh(rMeshObj), options);
        val out = val::object();
        out.set("mesh", mesh_to_val(r.mMesh));
        out.set("numNodesMoved", static_cast<double>(r.mNumNodesMoved));
        out.set("maxDisplacement", r.mMaxDisplacement);
        out.set("numSkippedInversion", static_cast<double>(r.mNumSkippedInversion));
        return out;
    });
}

/**
 * @brief Crop a mesh to a bounding box (`lo`/`hi` are 3-element JS arrays).
 * `mode` is "all" (default) or "any". Returns the pruned mesh object.
 */
val crop_bbox_js(const val& rMeshObj, const val& rLo, const val& rHi, const std::string& rMode,
                 bool recordIds) {
    return with_js_errors([&]() -> val {
        double lo[3], hi[3];
        for (unsigned i = 0; i < 3; ++i) {
            lo[i] = rLo[i].as<double>();
            hi[i] = rHi[i].as<double>();
        }
        meshioplusplus::CropMode m =
            (rMode == "any") ? meshioplusplus::CropMode::Any : meshioplusplus::CropMode::All;
        return mesh_to_val(
            meshioplusplus::crop_bbox(val_to_mesh(rMeshObj), lo, hi, m, recordIds).mMesh);
    });
}

/**
 * @brief Crop a mesh to the half-space (p - point) . normal >= 0 (`point`/
 * `normal` are 3-element JS arrays). Returns the pruned mesh object.
 */
/**
 * @brief Crop to the cells whose scalar `cell_data` value satisfies a comparison.
 *
 * There is deliberately no `mode`: `cropBbox`/`cropPlane` test points and then
 * need an all/any rule, whereas a `cell_data` predicate is already one value per
 * cell and has nothing to reduce.
 */
val crop_predicate_js(const val& rMeshObj, const std::string& rArray, const std::string& rCompare,
                      double value, bool recordIds) {
    return with_js_errors([&]() -> val {
        return mesh_to_val(meshioplusplus::crop_predicate(
                               val_to_mesh(rMeshObj), rArray,
                               meshioplusplus::refine_compare_from_name(rCompare), value, recordIds)
                               .mMesh);
    });
}

val crop_plane_js(const val& rMeshObj, const val& rPoint, const val& rNormal,
                  const std::string& rMode, bool recordIds) {
    return with_js_errors([&]() -> val {
        double point[3], normal[3];
        for (unsigned i = 0; i < 3; ++i) {
            point[i] = rPoint[i].as<double>();
            normal[i] = rNormal[i].as<double>();
        }
        meshioplusplus::CropMode m =
            (rMode == "any") ? meshioplusplus::CropMode::Any : meshioplusplus::CropMode::All;
        return mesh_to_val(
            meshioplusplus::crop_halfspace(val_to_mesh(rMeshObj), point, normal, m, recordIds)
                .mMesh);
    });
}

/**
 * @brief Planar cross-section of a mesh (marching tetrahedra on a simplexified
 * input). Returns a new mesh one dimension below the cut cells: a triangle/quad
 * surface for a volume mesh, a line mesh for a 2D surface mesh.
 */
val slice_js(const val& rMeshObj, const val& rOrigin, const val& rNormal, bool recordParentIds) {
    return with_js_errors([&]() -> val {
        meshioplusplus::SliceOptions options;
        for (unsigned i = 0; i < 3; ++i) {
            options.mOrigin[i] = rOrigin[i].as<double>();
            options.mNormal[i] = rNormal[i].as<double>();
        }
        options.mRecordParentIds = recordParentIds;
        return mesh_to_val(meshioplusplus::slice(val_to_mesh(rMeshObj), options));
    });
}

/**
 * @brief Isosurfaces / contours: the level set of a scalar point_data field,
 * cut with the same marching tetrahedra as slice. `component` is negative for
 * the row magnitude. Returns one mesh holding every contour, tagged per cell
 * with `iso:value` (Float64) and `iso:index` (Int64).
 */
/**
 * @brief A regular hexahedron lattice from nothing -- the only binding here
 * that takes no input mesh. Points run x fastest, then y, then z.
 */
val grid_js(const val& rDims, const val& rOrigin, const val& rSpacing, double maxCells) {
    return with_js_errors([&]() -> val {
        std::array<std::int64_t, 3> dims{{0, 0, 0}};
        std::array<double, 3> origin{{0.0, 0.0, 0.0}};
        std::array<double, 3> spacing{{1.0, 1.0, 1.0}};
        for (unsigned i = 0; i < 3; ++i) {
            dims[i] = static_cast<std::int64_t>(rDims[i].as<double>());
            if (!rOrigin.isUndefined() && !rOrigin.isNull())
                origin[i] = rOrigin[i].as<double>();
            if (!rSpacing.isUndefined() && !rSpacing.isNull())
                spacing[i] = rSpacing[i].as<double>();
        }
        return mesh_to_val(
            meshioplusplus::grid(dims, origin, spacing, static_cast<std::int64_t>(maxCells)));
    });
}

/**
 * @brief Build a regular grid around a mesh. Returns
 * `{mesh, dims, origin, spacing, numOccupied}`. Exactly one of `resolution` and
 * `cellSize` must be given -- pass an empty array / 0 for the other.
 */
val voxelize_js(const val& rMeshObj, const val& rResolution, double cellSize, const val& rBounds,
                double padding, double paddingRelative, const std::string& rFill,
                const std::string& rSign, bool attachOccupancy, double maxCells,
                const std::string& rWatertightCheck) {
    return with_js_errors([&]() -> val {
        meshioplusplus::VoxelOptions options;
        if (!rResolution.isUndefined() && !rResolution.isNull() &&
            rResolution["length"].as<unsigned>() == 3)
            options.mResolution = std::array<std::int64_t, 3>{
                {static_cast<std::int64_t>(rResolution[0].as<double>()),
                 static_cast<std::int64_t>(rResolution[1].as<double>()),
                 static_cast<std::int64_t>(rResolution[2].as<double>())}};
        if (cellSize > 0.0)
            options.mCellSize = cellSize;
        if (!rBounds.isUndefined() && !rBounds.isNull() && rBounds["length"].as<unsigned>() == 6) {
            std::array<double, 6> b{};
            for (unsigned i = 0; i < 6; ++i)
                b[i] = rBounds[i].as<double>();
            options.mBounds = b;
        }
        options.mPadding = padding;
        options.mPaddingRelative = paddingRelative;
        options.mFill = meshioplusplus::voxel_fill_from_name(rFill);
        options.mAttachOccupancy = attachOccupancy;
        options.mMaxCells = static_cast<std::int64_t>(maxCells);
        options.mDistance.mSign = meshioplusplus::sdf_sign_from_name(rSign);
        options.mDistance.mWatertightCheck =
            meshioplusplus::sdf_watertight_check_from_name(rWatertightCheck);

        meshioplusplus::VoxelResult r = meshioplusplus::voxelize(val_to_mesh(rMeshObj), options);
        const auto vec3i = [](const std::array<std::int64_t, 3>& a) {
            val out = val::array();
            for (int i = 0; i < 3; ++i)
                out.call<void>("push", static_cast<double>(a[static_cast<std::size_t>(i)]));
            return out;
        };
        const auto vec3d = [](const std::array<double, 3>& a) {
            val out = val::array();
            for (int i = 0; i < 3; ++i)
                out.call<void>("push", a[static_cast<std::size_t>(i)]);
            return out;
        };
        val out = val::object();
        out.set("mesh", mesh_to_val(r.mMesh));
        out.set("dims", vec3i(r.mDims));
        out.set("origin", vec3d(r.mOrigin));
        out.set("spacing", vec3d(r.mSpacing));
        out.set("numOccupied", static_cast<double>(r.mNumOccupied));
        return out;
    });
}

/// @brief What is wrong with a surface, in numbers.
val surface_watertight_check_js(const val& rMeshObj) {
    return with_js_errors([&]() -> val {
        const meshioplusplus::SurfaceQuality q =
            meshioplusplus::surface_watertight_check(val_to_mesh(rMeshObj));
        val out = val::object();
        out.set("boundaryEdges", static_cast<double>(q.mBoundaryEdges));
        out.set("nonManifoldEdges", static_cast<double>(q.mNonManifoldEdges));
        out.set("inconsistentPairs", static_cast<double>(q.mInconsistentPairs));
        out.set("degenerateTriangles", static_cast<double>(q.mDegenerateTriangles));
        out.set("watertight", q.mWatertight);
        return out;
    });
}

namespace {

meshioplusplus::SurfaceDistanceOptions sdf_options_js(const std::string& rSign,
                                                      const std::string& rLocation, double band,
                                                      bool recordInside,
                                                      const std::string& rWatertightCheck) {
    meshioplusplus::SurfaceDistanceOptions options;
    options.mSign = meshioplusplus::sdf_sign_from_name(rSign);
    options.mLocation = meshioplusplus::sdf_location_from_name(rLocation);
    options.mBand = band;
    options.mRecordInside = recordInside;
    options.mWatertightCheck = meshioplusplus::sdf_watertight_check_from_name(rWatertightCheck);
    return options;
}

}  // namespace

/**
 * @brief Signed distances from a flat `[x0,y0,z0, x1,y1,z1, ...]` array to a
 * surface. Returns a `Float64Array`; negative is inside.
 */
val sample_distance_js(const val& rSurfaceObj, const val& rPoints, const std::string& rSign,
                       double band, const std::string& rWatertightCheck) {
    return with_js_errors([&]() -> val {
        const unsigned n = rPoints["length"].as<unsigned>();
        if (n % 3 != 0)
            throw meshioplusplus::ReadError(
                "sample_distance: points must be a flat array of x, y, z triples");
        std::vector<double> flat(n);
        for (unsigned i = 0; i < n; ++i)
            flat[i] = rPoints[i].as<double>();
        const NDArray query = NDArray::MakeView(meshioplusplus::DType::Float64,
                                                {static_cast<std::size_t>(n / 3), std::size_t{3}},
                                                reinterpret_cast<std::byte*>(flat.data()));
        return ndarray_to_float64_array(meshioplusplus::sample_distance(
            val_to_mesh(rSurfaceObj), query,
            sdf_options_js(rSign, "corner", band, false, rWatertightCheck)));
    });
}

/**
 * @brief Attach the signed distance from a query mesh to a surface. Returns
 * `{mesh, numBanded, quality}`.
 */
val distance_to_surface_js(const val& rQueryObj, const val& rSurfaceObj, const std::string& rSign,
                           const std::string& rLocation, double band, bool recordInside,
                           const std::string& rWatertightCheck) {
    return with_js_errors([&]() -> val {
        meshioplusplus::SurfaceDistanceResult r = meshioplusplus::distance_to_surface(
            val_to_mesh(rQueryObj), val_to_mesh(rSurfaceObj),
            sdf_options_js(rSign, rLocation, band, recordInside, rWatertightCheck));
        val quality = val::object();
        quality.set("boundaryEdges", static_cast<double>(r.mQuality.mBoundaryEdges));
        quality.set("nonManifoldEdges", static_cast<double>(r.mQuality.mNonManifoldEdges));
        quality.set("inconsistentPairs", static_cast<double>(r.mQuality.mInconsistentPairs));
        quality.set("degenerateTriangles", static_cast<double>(r.mQuality.mDegenerateTriangles));
        quality.set("watertight", r.mQuality.mWatertight);
        val out = val::object();
        out.set("mesh", mesh_to_val(r.mMesh));
        out.set("numBanded", static_cast<double>(r.mNumBanded));
        out.set("quality", quality);
        return out;
    });
}

/**
 * @brief Generate a grid over a surface and fill it with signed distances.
 *
 * Returns `{mesh, dims, origin, spacing, maxDepth, numBanded, quality}`, where
 * `dims` are the ROOT cell counts and `spacing` the FINEST cell size.
 *
 * `structure` is `"voxel"` or `"octree"`; `resolution`/`cellSize` size a voxel
 * grid and are an error with `"octree"`, whose finest cell is
 * `rootResolution / 2^maxDepth` and is therefore already determined.
 */
val compute_sdf_js(const val& rSurfaceObj, const std::string& rStructure, const val& rResolution,
                   double cellSize, const val& rBounds, double padding, double paddingRelative,
                   double rootResolution, double maxDepth, double bandCells, bool recordLevels,
                   double maxCells, const std::string& rSign, const std::string& rLocation,
                   double band, const std::string& rWatertightCheck) {
    return with_js_errors([&]() -> val {
        meshioplusplus::SdfOptions options;
        options.mStructure = meshioplusplus::sdf_structure_from_name(rStructure);
        if (!rResolution.isUndefined() && !rResolution.isNull() &&
            rResolution["length"].as<unsigned>() == 3)
            options.mResolution = std::array<std::int64_t, 3>{
                {static_cast<std::int64_t>(rResolution[0].as<double>()),
                 static_cast<std::int64_t>(rResolution[1].as<double>()),
                 static_cast<std::int64_t>(rResolution[2].as<double>())}};
        if (cellSize > 0.0)
            options.mCellSize = cellSize;
        if (!rBounds.isUndefined() && !rBounds.isNull() && rBounds["length"].as<unsigned>() == 6) {
            std::array<double, 6> b{};
            for (unsigned i = 0; i < 6; ++i)
                b[i] = rBounds[i].as<double>();
            options.mBounds = b;
        }
        options.mPadding = padding;
        options.mPaddingRelative = paddingRelative;
        options.mRootResolution = static_cast<std::int64_t>(rootResolution);
        options.mMaxDepth = static_cast<std::int64_t>(maxDepth);
        options.mBandCells = bandCells;
        options.mRecordLevels = recordLevels;
        options.mMaxCells = static_cast<std::int64_t>(maxCells);
        options.mDistance =
            sdf_options_js(rSign, rLocation, band, /*recordInside=*/false, rWatertightCheck);

        meshioplusplus::SdfResult r =
            meshioplusplus::compute_sdf(val_to_mesh(rSurfaceObj), options);
        const auto vec3i = [](const std::array<std::int64_t, 3>& a) {
            val out = val::array();
            for (int i = 0; i < 3; ++i)
                out.call<void>("push", static_cast<double>(a[static_cast<std::size_t>(i)]));
            return out;
        };
        const auto vec3d = [](const std::array<double, 3>& a) {
            val out = val::array();
            for (int i = 0; i < 3; ++i)
                out.call<void>("push", a[static_cast<std::size_t>(i)]);
            return out;
        };
        val quality = val::object();
        quality.set("boundaryEdges", static_cast<double>(r.mQuality.mBoundaryEdges));
        quality.set("nonManifoldEdges", static_cast<double>(r.mQuality.mNonManifoldEdges));
        quality.set("inconsistentPairs", static_cast<double>(r.mQuality.mInconsistentPairs));
        quality.set("degenerateTriangles", static_cast<double>(r.mQuality.mDegenerateTriangles));
        quality.set("watertight", r.mQuality.mWatertight);
        val out = val::object();
        out.set("mesh", mesh_to_val(r.mMesh));
        out.set("dims", vec3i(r.mDims));
        out.set("origin", vec3d(r.mOrigin));
        out.set("spacing", vec3d(r.mSpacing));
        out.set("maxDepth", static_cast<double>(r.mMaxDepth));
        out.set("numBanded", static_cast<double>(r.mNumBanded));
        out.set("quality", quality);
        return out;
    });
}

val isosurface_js(const val& rMeshObj, const std::string& rArray, const val& rIsovalues,
                  int component, bool recordParentIds) {
    return with_js_errors([&]() -> val {
        meshioplusplus::IsosurfaceOptions options;
        options.mArrayName = rArray;
        const unsigned n = rIsovalues["length"].as<unsigned>();
        options.mIsovalues.reserve(n);
        for (unsigned i = 0; i < n; ++i)
            options.mIsovalues.push_back(rIsovalues[i].as<double>());
        if (component >= 0)
            options.mComponent = component;
        options.mRecordParentIds = recordParentIds;
        return mesh_to_val(meshioplusplus::isosurface(val_to_mesh(rMeshObj), options));
    });
}

/**
 * @brief Gradient / divergence / curl of a point_data field.
 *
 * `component` is negative for EVERY component -- deliberately the opposite of
 * `isosurface_js`, where negative means the row magnitude. An empty `output`
 * selects `<array>:<operator>`.
 *
 * The result rides the mesh boundary as an (n, 3) or (n, 9) array, so its width
 * travels in the `point_data_components`/`cell_data_components` sibling maps
 * `mesh_to_val` emits -- without those a gradient would come back flattened to
 * (3n, 1), which is exactly the class of bug v9.9.0 fixed.
 */
val gradient_js(const val& rMeshObj, const std::string& rArray, const std::string& rOperator,
                const std::string& rMethod, const std::string& rLocation,
                const std::string& rOutput, int component, bool overwrite) {
    return with_js_errors([&]() -> val {
        meshioplusplus::GradientOptions options;
        options.mArrayName = rArray;
        options.mOperator = meshioplusplus::gradient_operator_from_name(rOperator);
        options.mMethod = meshioplusplus::gradient_method_from_name(rMethod);
        options.mLocation =
            meshioplusplus::data_location_from_name(rLocation.empty() ? "cell" : rLocation);
        options.mOutputName = rOutput;
        if (component >= 0)
            options.mComponent = component;
        options.mOverwrite = overwrite;
        meshioplusplus::GradientResult r = meshioplusplus::gradient(val_to_mesh(rMeshObj), options);
        val out = val::object();
        out.set("mesh", mesh_to_val(r.mMesh));
        out.set("numSkipped", static_cast<double>(r.mNumSkipped));
        out.set("numFallback", static_cast<double>(r.mNumFallback));
        return out;
    });
}

/**
 * @brief The Hessian (second derivative) of a scalar point_data field --
 * gradient_js's companion one order further, for curvature-based adaptive
 * refinement.
 *
 * A composition of TWO gradient() calls, not a new numerical kernel: the
 * field is differentiated once (Point location), then that (n,3) gradient
 * is differentiated again with the default "gradient" operator, producing
 * (n,9) -- the flattened row-major 3x3 Hessian. `method` is forwarded to
 * BOTH internal passes. An empty `output` selects `<array>:hessian`.
 *
 * A field that is at most LINEAR has an exactly zero Hessian everywhere --
 * the one mesh-shape-independent guarantee. For a genuinely quadratic field
 * the composition is exact on a structured/symmetric mesh away from its own
 * boundary and a good, standard, but genuinely approximate curvature
 * estimate on an irregular mesh (see doc/hessian.md). Input must have
 * exactly one component.
 *
 * Like gradient_js, the result's (n,9) width travels in the
 * point_data_components/cell_data_components sibling maps mesh_to_val
 * emits.
 */
val hessian_js(const val& rMeshObj, const std::string& rArray, const std::string& rMethod,
               const std::string& rLocation, const std::string& rOutput, bool overwrite) {
    return with_js_errors([&]() -> val {
        meshioplusplus::HessianOptions options;
        options.mArrayName = rArray;
        options.mMethod = meshioplusplus::gradient_method_from_name(rMethod);
        options.mLocation =
            meshioplusplus::data_location_from_name(rLocation.empty() ? "cell" : rLocation);
        options.mOutputName = rOutput;
        options.mOverwrite = overwrite;
        meshioplusplus::HessianResult r = meshioplusplus::hessian(val_to_mesh(rMeshObj), options);
        val out = val::object();
        out.set("mesh", mesh_to_val(r.mMesh));
        out.set("numSkipped", static_cast<double>(r.mNumSkipped));
        out.set("numFallback", static_cast<double>(r.mNumFallback));
        return out;
    });
}

/**
 * @brief The ZZ recovery-based error indicator plus marking.
 *
 * A composition of `gradient_js` with the measure-weighted point<->cell
 * averaging round trip; see operations/error.hpp for the indicator/marking
 * contract. `marking` "none" (default) attaches only the Float64 `error:zz`
 * indicator; any other policy also attaches an Int64 0/1 `error:marked`
 * array, so `refine`'s own `--where` selector needs no change at all.
 */
val estimate_error_js(const val& rMeshObj, const std::string& rArray, const std::string& rMethod,
                      const std::string& rMarking, double markingValue, const std::string& rOutput,
                      const std::string& rMarked, bool overwrite) {
    return with_js_errors([&]() -> val {
        meshioplusplus::ErrorOptions options;
        options.mArrayName = rArray;
        options.mMethod = meshioplusplus::error_method_from_name(rMethod);
        options.mMarking = meshioplusplus::error_marking_from_name(rMarking);
        options.mMarkingValue = markingValue;
        options.mOutputName = rOutput;
        options.mMarkedName = rMarked;
        options.mOverwrite = overwrite;
        meshioplusplus::ErrorResult r =
            meshioplusplus::estimate_error(val_to_mesh(rMeshObj), options);
        val out = val::object();
        out.set("mesh", mesh_to_val(r.mMesh));
        out.set("globalError", r.mGlobalError);
        out.set("numSkipped", static_cast<double>(r.mNumSkipped));
        out.set("numMarked", static_cast<double>(r.mNumMarked));
        return out;
    });
}

/**
 * Remesh a surface by approximated centroidal Voronoi diagram (ACVD)
 * clustering: replace its triangulation with a new, near-uniformly-sized,
 * well-shaped one at `numClusters` vertices. Unlike every other
 * resolution-changing operation the output has NEW points and NEW
 * connectivity with no correspondence to the input -- point/cell data and
 * named regions are dropped, field data is carried. See doc/remesh.md.
 */
val remesh_js(const val& rMeshObj, std::int64_t numClusters, int subdivide, double subsampleRatio,
              int maxSubdivide, int maxIterations, int maxRepairPasses, const std::string& rMetric,
              double gradation, bool preserveBoundary, double maxAnisotropy) {
    return with_js_errors([&]() -> val {
        meshioplusplus::RemeshOptions options;
        options.mNumClusters = numClusters;
        options.mSubdivide = subdivide;
        options.mSubsampleRatio = subsampleRatio;
        options.mMaxSubdivide = maxSubdivide;
        options.mMaxIterations = maxIterations;
        options.mMaxRepairPasses = maxRepairPasses;
        options.mMetric = meshioplusplus::remesh_metric_from_name(rMetric);
        options.mGradation = gradation;
        options.mPreserveBoundary = preserveBoundary;
        options.mMaxAnisotropy = maxAnisotropy;
        meshioplusplus::RemeshResult r = meshioplusplus::remesh(val_to_mesh(rMeshObj), options);
        val out = val::object();
        out.set("mesh", mesh_to_val(r.mMesh));
        out.set("numClusters", static_cast<double>(r.mNumClusters));
        out.set("numIterations", static_cast<double>(r.mNumIterations));
        out.set("subdivideApplied", static_cast<double>(r.mSubdivideApplied));
        out.set("numIsolatedClusters", static_cast<double>(r.mNumIsolatedClusters));
        out.set("numNonManifoldVertices", static_cast<double>(r.mNumNonManifoldVertices));
        return out;
    });
}

/**
 * Retetrahedralize a volume mesh (or a closed surface) at a caller-chosen
 * resolution by isosurface stuffing -- remesh's volumetric sibling. Same
 * no-correspondence output contract (new points, new connectivity;
 * point/cell data and named regions are dropped, field data is carried).
 * `resolution`/`cellSize`/`bounds` share `voxelize_js`'s own convention:
 * `resolution` is a length-3 array or undefined/null, `cellSize <= 0` means
 * unset, `bounds` is a length-6 array or undefined/null. See
 * doc/remesh_volume.md.
 */
val remesh_volume_js(const val& rMeshObj, const val& rResolution, double cellSize,
                     const val& rBounds, double padding, double paddingRelative, double maxCells,
                     double maxTets, double warpFraction, const std::string& rSign,
                     const std::string& rWatertightCheck) {
    return with_js_errors([&]() -> val {
        meshioplusplus::RemeshVolumeOptions options;
        if (!rResolution.isUndefined() && !rResolution.isNull() &&
            rResolution["length"].as<unsigned>() == 3)
            options.mResolution = std::array<std::int64_t, 3>{
                {static_cast<std::int64_t>(rResolution[0].as<double>()),
                 static_cast<std::int64_t>(rResolution[1].as<double>()),
                 static_cast<std::int64_t>(rResolution[2].as<double>())}};
        if (cellSize > 0.0)
            options.mCellSize = cellSize;
        if (!rBounds.isUndefined() && !rBounds.isNull() && rBounds["length"].as<unsigned>() == 6) {
            std::array<double, 6> b{};
            for (unsigned i = 0; i < 6; ++i)
                b[i] = rBounds[i].as<double>();
            options.mBounds = b;
        }
        options.mPadding = padding;
        options.mPaddingRelative = paddingRelative;
        options.mMaxCells = static_cast<std::int64_t>(maxCells);
        options.mMaxTets = static_cast<std::int64_t>(maxTets);
        options.mWarpFraction = warpFraction;
        options.mDistance.mSign = meshioplusplus::sdf_sign_from_name(rSign);
        options.mDistance.mWatertightCheck =
            meshioplusplus::sdf_watertight_check_from_name(rWatertightCheck);
        meshioplusplus::RemeshVolumeResult r =
            meshioplusplus::remesh_volume(val_to_mesh(rMeshObj), options);
        val out = val::object();
        out.set("mesh", mesh_to_val(r.mMesh));
        out.set("numTets", static_cast<double>(r.mNumTets));
        out.set("numVerticesWarped", static_cast<double>(r.mNumVerticesWarped));
        out.set("numTetsRejected", static_cast<double>(r.mNumTetsRejected));
        out.set("numNonManifoldEdges", static_cast<double>(r.mNumNonManifoldEdges));
        return out;
    });
}

/**
 * @brief ODT-remesh a tetrahedral mesh: relocate vertices AND flip connectivity
 * (2-3/3-2, predicate-free). The point set is invariant, so point_data/
 * field_data and Point regions carry; cell_data + Cell/Side regions are
 * dropped. See operations/optimize_volume.hpp and doc/optimize_volume.md.
 */
/**
 * @brief Per-vertex mean and Gaussian curvature of a surface -- K by the
 * angle defect, H by the cotangent Laplace-Beltrami operator. The result
 * carries `totalAngleDefect` (2*pi*chi for a closed surface, 4*pi for a
 * sphere) as the Gauss-Bonnet oracle, and the input's `quality` since H's
 * sign is orientation-dependent.
 */
val compute_curvature_js(const val& rMeshObj, bool mean, bool gaussian,
                         const std::string& rDualArea, bool includeBoundary, bool recordArea,
                         bool recordPrincipal, const std::string& rRegion) {
    return with_js_errors([&]() -> val {
        meshioplusplus::CurvatureOptions options;
        options.mMean = mean;
        options.mGaussian = gaussian;
        options.mDualArea = meshioplusplus::curvature_dual_area_from_name(rDualArea);
        options.mIncludeBoundary = includeBoundary;
        options.mRecordArea = recordArea;
        options.mRecordPrincipal = recordPrincipal;
        options.mRegion = rRegion;
        meshioplusplus::CurvatureResult r =
            meshioplusplus::compute_curvature(val_to_mesh(rMeshObj), options);
        val quality = val::object();
        quality.set("boundaryEdges", static_cast<double>(r.mQuality.mBoundaryEdges));
        quality.set("nonManifoldEdges", static_cast<double>(r.mQuality.mNonManifoldEdges));
        quality.set("inconsistentPairs", static_cast<double>(r.mQuality.mInconsistentPairs));
        quality.set("degenerateTriangles", static_cast<double>(r.mQuality.mDegenerateTriangles));
        quality.set("watertight", r.mQuality.mWatertight);
        val out = val::object();
        out.set("mesh", mesh_to_val(r.mMesh));
        out.set("numBoundary", static_cast<double>(r.mNumBoundary));
        out.set("numIsolated", static_cast<double>(r.mNumIsolated));
        out.set("numDegenerate", static_cast<double>(r.mNumDegenerate));
        out.set("totalAngleDefect", r.mTotalAngleDefect);
        out.set("quality", quality);
        return out;
    });
}

/// A SurfaceQuality as a plain JS object (the four counts plus the verdict).
val quality_to_val(const meshioplusplus::SurfaceQuality& rQuality) {
    val quality = val::object();
    quality.set("boundaryEdges", static_cast<double>(rQuality.mBoundaryEdges));
    quality.set("nonManifoldEdges", static_cast<double>(rQuality.mNonManifoldEdges));
    quality.set("inconsistentPairs", static_cast<double>(rQuality.mInconsistentPairs));
    quality.set("degenerateTriangles", static_cast<double>(rQuality.mDegenerateTriangles));
    quality.set("watertight", rQuality.mWatertight);
    return quality;
}

/**
 * @brief Surface repair: orientation by the topological half-edge rule,
 * fan-filled holes wound to agree with the surrounding surface, and bowtie
 * splitting. See operations/repair.hpp.
 */
val repair_js(const val& rMeshObj, bool fixOrientation, bool orientOutward, bool fillHoles,
              bool splitNonManifold, double maxHoleEdges, double weldTolerance,
              bool recordProvenance) {
    return with_js_errors([&]() -> val {
        meshioplusplus::RepairOptions options;
        options.mFixOrientation = fixOrientation;
        options.mOrientOutward = orientOutward;
        options.mFillHoles = fillHoles;
        options.mSplitNonManifold = splitNonManifold;
        options.mMaxHoleEdges = static_cast<std::int64_t>(maxHoleEdges);
        options.mWeldTolerance = weldTolerance;
        options.mRecordProvenance = recordProvenance;
        meshioplusplus::RepairResult r = meshioplusplus::repair(val_to_mesh(rMeshObj), options);
        val out = val::object();
        out.set("mesh", mesh_to_val(r.mMesh));
        out.set("qualityBefore", quality_to_val(r.mQualityBefore));
        out.set("qualityAfter", quality_to_val(r.mQualityAfter));
        out.set("numFlipped", static_cast<double>(r.mNumFlipped));
        out.set("numComponents", static_cast<double>(r.mNumComponents));
        out.set("largestComponent", static_cast<double>(r.mLargestComponent));
        out.set("numOrientedOutward", static_cast<double>(r.mNumOrientedOutward));
        out.set("numUnorientable", static_cast<double>(r.mNumUnorientable));
        out.set("numVerticesSplit", static_cast<double>(r.mNumVerticesSplit));
        out.set("numHolesDetected", static_cast<double>(r.mNumHolesDetected));
        out.set("numHolesFilled", static_cast<double>(r.mNumHolesFilled));
        out.set("numHolesSkipped", static_cast<double>(r.mNumHolesSkipped));
        out.set("numFacesAdded", static_cast<double>(r.mNumFacesAdded));
        out.set("numPointsAdded", static_cast<double>(r.mNumPointsAdded));
        out.set("pointsWelded", static_cast<double>(r.mPointsWelded));
        return out;
    });
}

/**
 * @brief Project a mesh's points onto a target triangle surface. One
 * projection, offset along the hit FEATURE's pseudonormal. See
 * operations/shrinkwrap.hpp.
 */
val shrinkwrap_js(const val& rMeshObj, const val& rTargetObj, double offset, double maxDistance,
                  const std::string& rWeights, const std::string& rTargetRegion,
                  const std::string& rNormalWeight, bool recordDistance, bool recordClosestCell) {
    return with_js_errors([&]() -> val {
        meshioplusplus::ShrinkwrapOptions options;
        options.mOffset = offset;
        options.mMaxDistance = maxDistance;
        options.mWeights = rWeights;
        options.mTargetRegion = rTargetRegion;
        options.mNormalWeight = meshioplusplus::sdf_weight_from_name(rNormalWeight);
        options.mRecordDistance = recordDistance;
        options.mRecordClosestCell = recordClosestCell;
        meshioplusplus::ShrinkwrapResult r =
            meshioplusplus::shrinkwrap(val_to_mesh(rMeshObj), val_to_mesh(rTargetObj), options);
        val out = val::object();
        out.set("mesh", mesh_to_val(r.mMesh));
        out.set("quality", quality_to_val(r.mQuality));
        out.set("numProjected", static_cast<double>(r.mNumProjected));
        out.set("numMissed", static_cast<double>(r.mNumMissed));
        out.set("numSkipped", static_cast<double>(r.mNumSkipped));
        out.set("maxDisplacement", r.mMaxDisplacement);
        return out;
    });
}

/**
 * @brief Sobolev (Helmholtz-filtered) deformation over the mesh's own P1
 * operators. See operations/sobolev_deform.hpp.
 */
val sobolev_deform_js(const val& rMeshObj, const std::string& rArray, double lengthScale,
                      const std::string& rFixedPointsArray, bool fixBoundary, bool recordFiltered,
                      double maxIterations, double tolerance) {
    return with_js_errors([&]() -> val {
        meshioplusplus::SobolevOptions options;
        options.mArrayName = rArray;
        options.mLengthScale = lengthScale;
        options.mFixedPointsArray = rFixedPointsArray;
        options.mFixBoundary = fixBoundary;
        options.mRecordFiltered = recordFiltered;
        options.mMaxIterations = static_cast<int>(maxIterations);
        options.mTolerance = tolerance;
        meshioplusplus::SobolevResult r =
            meshioplusplus::sobolev_deform(val_to_mesh(rMeshObj), options);
        val out = val::object();
        out.set("mesh", mesh_to_val(r.mMesh));
        out.set("numIterations", static_cast<double>(r.mNumIterations));
        out.set("residual", r.mResidual);
        out.set("converged", r.mConverged);
        out.set("numFixed", static_cast<double>(r.mNumFixed));
        out.set("numIsolated", static_cast<double>(r.mNumIsolated));
        out.set("maxDisplacement", r.mMaxDisplacement);
        return out;
    });
}

val optimize_volume_js(const val& rMeshObj, double maxIterations, bool relocate, bool flip,
                       bool preserveBoundary, double minImprovement) {
    return with_js_errors([&]() -> val {
        meshioplusplus::OptimizeVolumeOptions options;
        options.mMaxIterations = static_cast<int>(maxIterations);
        options.mRelocate = relocate;
        options.mFlip = flip;
        options.mPreserveBoundary = preserveBoundary;
        options.mMinImprovement = minImprovement;
        meshioplusplus::OptimizeVolumeResult r =
            meshioplusplus::optimize_volume(val_to_mesh(rMeshObj), options);
        val out = val::object();
        out.set("mesh", mesh_to_val(r.mMesh));
        out.set("numFlips", static_cast<double>(r.mNumFlips));
        out.set("num23Flips", static_cast<double>(r.mNum23Flips));
        out.set("num32Flips", static_cast<double>(r.mNum32Flips));
        out.set("numVerticesMoved", static_cast<double>(r.mNumVerticesMoved));
        out.set("numTets", static_cast<double>(r.mNumTets));
        out.set("minQualityBefore", r.mMinQualityBefore);
        out.set("minQualityAfter", r.mMinQualityAfter);
        return out;
    });
}

/**
 * @brief Split a mesh into pieces (by "type" / "component" / "region"|"tag").
 * Returns a JS array of `{key, mesh}` objects. `tagName` selects the integer
 * cell_data for the tag criterion (empty = auto-detect).
 */
val split_js(const val& rMeshObj, const std::string& rBy, const std::string& rTagName) {
    return with_js_errors([&]() -> val {
        meshioplusplus::SplitResult r = meshioplusplus::split(
            val_to_mesh(rMeshObj), meshioplusplus::split_by_from_name(rBy), rTagName);
        val out = val::array();
        for (meshioplusplus::SplitPiece& p : r.mPieces) {
            val piece = val::object();
            piece.set("key", p.mKey);
            piece.set("mesh", mesh_to_val(p.mMesh));
            out.call<void>("push", piece);
        }
        return out;
    });
}

/**
 * @brief Convert the element representation of a mesh: `"linearize"`,
 * `"simplexify"`, or `"elevate"`. Returns the converted mesh; the index maps
 * are not carried across the JS boundary (use `recordParentIds` for the
 * `convert:parent_cell` cell_data instead).
 */
val convert_cells_js(const val& rMeshObj, const std::string& rMode, bool recordParentIds) {
    return with_js_errors([&]() -> val {
        meshioplusplus::ConvertCellsOptions options;
        options.mMode = meshioplusplus::convert_cells_mode_from_name(rMode);
        options.mRecordParentIds = recordParentIds;
        return mesh_to_val(meshioplusplus::convert_cells(val_to_mesh(rMeshObj), options).mMesh);
    });
}

/**
 * @brief Polyhedrally refine a mesh: one polyhedral child per face of every
 * eligible 3D cell, connected to a new interior point. Needs no per-type
 * template table -- tabulated types (reduced to corners for a quadratic
 * variant) and existing polyhedron blocks are handled uniformly.
 * Automatically conforming, unlike `refine`. Returns the subdivided mesh; the
 * cell maps are not carried across the JS boundary (use `recordParentIds` for
 * the `subdivide:parent_cell` cell_data instead) -- and unlike
 * `convertCells`, there is no point map at all, since subdivide never prunes
 * or renumbers an original point.
 */
val subdivide_js(const val& rMeshObj, bool recordParentIds) {
    return with_js_errors([&]() -> val {
        meshioplusplus::SubdivideOptions options;
        options.mRecordParentIds = recordParentIds;
        return mesh_to_val(meshioplusplus::subdivide(val_to_mesh(rMeshObj), options).mMesh);
    });
}

/**
 * @brief Polyhedrally coarsen a mesh: merge groups of cells into single
 * larger polyhedral cells via greedy seed-and-grow over the shared-face
 * dual. Non-volume blocks pass through unchanged; points are never pruned or
 * renumbered (`clean(mesh, ..., true)` is the follow-up for a minimal point
 * set). Returns the coarsened mesh; the flat cell map is not carried across
 * the JS boundary.
 */
val agglomerate_js(const val& rMeshObj, int targetGroupSize) {
    return with_js_errors([&]() -> val {
        meshioplusplus::AgglomerateOptions options;
        options.mTargetGroupSize = static_cast<std::size_t>(targetGroupSize);
        return mesh_to_val(meshioplusplus::agglomerate(val_to_mesh(rMeshObj), options).mMesh);
    });
}

/**
 * @brief Refine a mesh, subdividing cells into same-type children (line -> 2,
 * triangle -> 4, quad -> 4, tetra -> 8, wedge -> 8, hexahedron -> 8). Returns
 * the refined mesh; the index maps are not carried across the JS boundary (use
 * `recordParentIds` for the `refine:parent_cell` cell_data instead).
 *
 * `options` is an optional object selecting a SUBSET of the cells to refine —
 * `{cells, region, array/op/value, closure, recordLevels, recordHierarchy}`,
 * at most one selector — in which case the hanging nodes that leaves are
 * resolved by the closure and the output is still conforming. Omit it to
 * refine every cell.
 *
 * `recordHierarchy` attaches the `refine:cell_id`/`refine:parent_id`
 * cell_data arrays -- the persistent parent/child hierarchy a multigrid
 * caller resolves across the sequence of meshes it keeps. Also forces
 * `refine:entity` to be attached even when the closure leaves no hanging
 * node, since it already records the coarse corners each new fine node is
 * the mean of -- the multigrid prolongation weights.
 */
val refine_js(const val& rMeshObj, int levels, bool recordParentIds, const val& rOptions) {
    return with_js_errors([&]() -> val {
        meshioplusplus::RefineOptions options;
        options.mLevels = levels;
        options.mRecordParentIds = recordParentIds;
        if (!rOptions.isUndefined() && !rOptions.isNull()) {
            const auto text = [&rOptions](const char* key, const char* fallback) {
                val v = rOptions[key];
                return v.isUndefined() || v.isNull() ? std::string(fallback) : v.as<std::string>();
            };
            val cells = rOptions["cells"];
            if (!cells.isUndefined() && !cells.isNull()) {
                const unsigned n = cells["length"].as<unsigned>();
                options.mCells.reserve(n);
                for (unsigned i = 0; i < n; ++i)
                    options.mCells.push_back(static_cast<std::int64_t>(cells[i].as<double>()));
            }
            options.mRegion = text("region", "");
            options.mPredicateArray = text("array", "");
            if (!options.mPredicateArray.empty()) {
                options.mPredicateOp =
                    meshioplusplus::refine_compare_from_name(text("compare", "<"));
                val value = rOptions["value"];
                options.mPredicateValue =
                    value.isUndefined() || value.isNull() ? 0.0 : value.as<double>();
            }
            options.mClosure = meshioplusplus::refine_closure_from_name(text("closure", ""));
            val levels_flag = rOptions["recordLevels"];
            options.mRecordLevels =
                !levels_flag.isUndefined() && !levels_flag.isNull() && levels_flag.as<bool>();
            val hierarchy_flag = rOptions["recordHierarchy"];
            options.mRecordHierarchy = !hierarchy_flag.isUndefined() && !hierarchy_flag.isNull() &&
                                       hierarchy_flag.as<bool>();
        }
        return mesh_to_val(meshioplusplus::refine(val_to_mesh(rMeshObj), options).mMesh);
    });
}

/**
 * @brief Decimate a SURFACE mesh by quadric-error-metric edge collapse — the
 * resolution-reducing inverse of `refine`. Exactly one of `ratio` (fraction of
 * faces to keep, in (0, 1]), `targetFaces` and `maxError` must be non-negative.
 * Returns an object `{mesh, facesRemoved, pointsRemoved, collapsesRejected,
 * maxErrorApplied}`; the index maps are not carried across the JS boundary and
 * the frozen mask is not exposed here, as on the other flat bindings.
 */
val decimate_js(const val& rMeshObj, double ratio, double targetFaces, double maxError,
                const std::string& rPlacement, bool preserveBoundary, bool preserveFeatures,
                double featureAngle) {
    return with_js_errors([&]() -> val {
        meshioplusplus::DecimateOptions options;
        options.mTargetRatio = ratio;
        options.mTargetFaces = targetFaces < 0.0 ? static_cast<std::int64_t>(-1)
                                                 : static_cast<std::int64_t>(targetFaces);
        options.mMaxError = maxError;
        options.mPlacement = meshioplusplus::decimate_placement_from_name(rPlacement);
        options.mPreserveBoundary = preserveBoundary;
        options.mPreserveFeatures = preserveFeatures;
        options.mFeatureAngleDeg = featureAngle;
        meshioplusplus::DecimateResult r = meshioplusplus::decimate(val_to_mesh(rMeshObj), options);
        val out = val::object();
        out.set("mesh", mesh_to_val(r.mMesh));
        out.set("facesRemoved", static_cast<double>(r.mFacesRemoved));
        out.set("pointsRemoved", static_cast<double>(r.mPointsRemoved));
        out.set("collapsesRejected", static_cast<double>(r.mCollapsesRejected));
        out.set("maxErrorApplied", r.mMaxErrorApplied);
        return out;
    });
}

namespace {

meshioplusplus::PartitionOptions partition_options_js(int nparts, const std::string& rMethod,
                                                      double imbalance, const std::string& rMode,
                                                      int seed, bool recordIds, int ghostLayers,
                                                      const std::string& rWeightsKey) {
    meshioplusplus::PartitionOptions options;
    options.mNParts = nparts;
    options.mMethod = meshioplusplus::partition_method_from_name(rMethod);
    options.mImbalance = imbalance;
    options.mMode = meshioplusplus::partition_mode_from_name(rMode);
    options.mSeed = seed;
    options.mRecordIds = recordIds;
    options.mGhostLayers = ghostLayers;
    options.mWeightsKey = rWeightsKey;
    return options;
}

}  // namespace

/**
 * @brief Decompose a mesh into `nparts` balanced pieces (`"sfc"` / `"kahip"` /
 * `"auto"`). Returns a JS array of `{partId, mesh}` objects, exactly `nparts`
 * entries (pieces may be empty; blocks kept 1:1 with the input, unlike
 * `split`). The index maps are not carried across the JS boundary (use
 * `recordIds` for the `partition:original_*_id` arrays, or `partitionLabels`
 * for the assignment). KaHIP is never compiled into the WASM build, so
 * `method: "kahip"` throws the error naming `MESHIOPLUSPLUS_WITH_KAHIP`.
 */
val partition_js(const val& rMeshObj, int nparts, const std::string& rMethod, double imbalance,
                 const std::string& rMode, int seed, bool recordIds, int ghostLayers,
                 const std::string& rWeightsKey) {
    return with_js_errors([&]() -> val {
        meshioplusplus::PartitionResult r = meshioplusplus::partition(
            val_to_mesh(rMeshObj), partition_options_js(nparts, rMethod, imbalance, rMode, seed,
                                                        recordIds, ghostLayers, rWeightsKey));
        val out = val::array();
        for (meshioplusplus::PartitionPiece& p : r.mPieces) {
            val piece = val::object();
            piece.set("partId", p.mPartId);
            piece.set("mesh", mesh_to_val(p.mMesh));
            out.call<void>("push", piece);
        }
        return out;
    });
}

/**
 * @brief The per-cell part assignment only: a JS array with one Int32Array-like
 * array per cell block (block-aligned, like a cell_data entry), values in
 * `[0, nparts)`.
 */
val partition_labels_js(const val& rMeshObj, int nparts, const std::string& rMethod,
                        double imbalance, const std::string& rMode, int seed,
                        const std::string& rWeightsKey) {
    return with_js_errors([&]() -> val {
        std::vector<NDArray> labels = meshioplusplus::partition_labels(
            val_to_mesh(rMeshObj),
            partition_options_js(nparts, rMethod, imbalance, rMode, seed,
                                 /*recordIds=*/false, /*ghostLayers=*/0, rWeightsKey));
        val out = val::array();
        for (const NDArray& a : labels)
            out.call<void>("push", ndarray_to_int32_array(a));
        return out;
    });
}

/**
 * @brief Geometric statistics of a mesh (read-only). Returns an object with
 * `numPoints`, `numCells`, `bboxMin`/`bboxMax`/`extent`/`centroid` (3-arrays),
 * `cellTypeCounts` (object), `totalArea`, `signedVolume`, `unsignedVolume`,
 * `numInverted`.
 */
val stats_js(const val& rMeshObj) {
    return with_js_errors([&]() -> val {
        meshioplusplus::StatsReport r = meshioplusplus::compute_stats(val_to_mesh(rMeshObj));
        auto vec3 = [](const double* p) {
            val a = val::array();
            for (int i = 0; i < 3; ++i)
                a.call<void>("push", p[i]);
            return a;
        };
        val out = val::object();
        out.set("numPoints", static_cast<double>(r.mNumPoints));
        out.set("numCells", static_cast<double>(r.mNumCells));
        out.set("bboxMin", vec3(r.mBBoxMin));
        out.set("bboxMax", vec3(r.mBBoxMax));
        out.set("extent", vec3(r.mExtent));
        out.set("centroid", vec3(r.mCentroid));
        val counts = val::object();
        for (const auto& kv : r.mCellTypeCounts)
            counts.set(kv.first, static_cast<double>(kv.second));
        out.set("cellTypeCounts", counts);
        out.set("totalArea", r.mTotalArea);
        out.set("signedVolume", r.mSignedVolume);
        out.set("unsignedVolume", r.mUnsignedVolume);
        out.set("numInverted", static_cast<double>(r.mNumInverted));
        return out;
    });
}

/** @brief Connectivity bandwidth (max |i - j| over node pairs sharing a cell). */
int compute_bandwidth_js(const val& rMeshObj) {
    return with_js_errors([&]() -> int {
        return static_cast<int>(meshioplusplus::compute_bandwidth(val_to_mesh(rMeshObj)));
    });
}

// Convert an ArrayDiff to a JS object.
val array_diff_to_val(const meshioplusplus::ArrayDiff& rAd) {
    val d = val::object();
    d.set("name", rAd.mName);
    d.set("shapeMismatch", rAd.mShapeMismatch);
    d.set("sizeA", static_cast<double>(rAd.mSizeA));
    d.set("sizeB", static_cast<double>(rAd.mSizeB));
    d.set("maxAbsError", rAd.mMaxAbsError);
    d.set("maxRelError", rAd.mMaxRelError);
    d.set("worstIndex", static_cast<double>(rAd.mWorstIndex));
    d.set("numExceeding", static_cast<double>(rAd.mNumExceeding));
    d.set("exact", rAd.mExact);
    return d;
}

val data_diff_to_val(const meshioplusplus::DataDiff& rDd) {
    val d = val::object();
    val only_a = val::array(), only_b = val::array(), shared = val::array();
    for (const std::string& s : rDd.mOnlyInA)
        only_a.call<void>("push", s);
    for (const std::string& s : rDd.mOnlyInB)
        only_b.call<void>("push", s);
    for (const meshioplusplus::ArrayDiff& ad : rDd.mShared)
        shared.call<void>("push", array_diff_to_val(ad));
    d.set("onlyInA", only_a);
    d.set("onlyInB", only_b);
    d.set("shared", shared);
    return d;
}

/**
 * @brief Compare two meshes. Returns a report object with `verdict`
 * ("identical" / "equal within tolerance" / "different"), `points`, `blocks`,
 * `pointData`/`cellData`/`fieldData`, and `messages`. Named point/cell sets are
 * not compared (not visible to the core).
 */
val diff_js(const val& rMeshA, const val& rMeshB, double atol, double rtol, bool unordered) {
    return with_js_errors([&]() -> val {
        meshioplusplus::DiffOptions opts;
        opts.atol = atol;
        opts.rtol = rtol;
        opts.unordered = unordered;
        meshioplusplus::DiffReport rep =
            meshioplusplus::diff(val_to_mesh(rMeshA), val_to_mesh(rMeshB), opts);
        val out = val::object();
        out.set("verdict", meshioplusplus::diff_verdict_name(rep.mVerdict));
        out.set("unordered", rep.mUnordered);
        out.set("correspondenceFailed", rep.mCorrespondenceFailed);
        out.set("pointCountMismatch", rep.mPointCountMismatch);
        out.set("points", array_diff_to_val(rep.mPoints));
        out.set("blockCountMismatch", rep.mBlockCountMismatch);
        val blocks = val::array();
        for (const meshioplusplus::BlockDiff& bd : rep.mBlocks) {
            val d = val::object();
            d.set("block", static_cast<double>(bd.mBlock));
            d.set("typeA", bd.mTypeA);
            d.set("typeB", bd.mTypeB);
            d.set("countA", static_cast<double>(bd.mCountA));
            d.set("countB", static_cast<double>(bd.mCountB));
            d.set("typeMismatch", bd.mTypeMismatch);
            d.set("countMismatch", bd.mCountMismatch);
            d.set("connMismatchCount", static_cast<double>(bd.mConnMismatchCount));
            blocks.call<void>("push", d);
        }
        out.set("blocks", blocks);
        out.set("pointData", data_diff_to_val(rep.mPointData));
        out.set("cellData", data_diff_to_val(rep.mCellData));
        out.set("fieldData", data_diff_to_val(rep.mFieldData));
        val messages = val::array();
        for (const std::string& s : rep.mMessages)
            messages.call<void>("push", s);
        out.set("messages", messages);
        return out;
    });
}

/** @brief Are two meshes equal within tolerance? (verdict != "different") */
bool meshes_equal_js(const val& rMeshA, const val& rMeshB, double atol, double rtol,
                     bool unordered) {
    return with_js_errors([&]() -> bool {
        meshioplusplus::DiffOptions opts;
        opts.atol = atol;
        opts.rtol = rtol;
        opts.unordered = unordered;
        return meshioplusplus::diff(val_to_mesh(rMeshA), val_to_mesh(rMeshB), opts).mVerdict !=
               meshioplusplus::DiffVerdict::Different;
    });
}

// --- data operations -------------------------------------------------------
// These act on a mesh's point/cell/field data arrays; the geometry is never
// modified. Enumerations cross as strings (as `split`'s `by` already does),
// parsed by the same helpers the CLIs use.

/// A JS array of strings -> std::vector. This file deliberately avoids
/// `register_vector`, so the decode is manual.
std::vector<std::string> val_to_string_vector(const val& rArray) {
    std::vector<std::string> out;
    if (rArray.isUndefined() || rArray.isNull())
        return out;
    const unsigned n = rArray["length"].as<unsigned>();
    out.reserve(n);
    for (unsigned i = 0; i < n; ++i)
        out.push_back(rArray[i].as<std::string>());
    return out;
}

/** @brief Drop the named data arrays at `location` ("point"/"cell"/"field"). */
val data_drop_js(const val& rMeshObj, const std::string& rLocation, const val& rNames,
                 bool ignoreMissing) {
    return with_js_errors([&]() -> val {
        return mesh_to_val(meshioplusplus::data_drop(
            val_to_mesh(rMeshObj), meshioplusplus::data_location_from_name(rLocation),
            val_to_string_vector(rNames), ignoreMissing));
    });
}

/** @brief Keep only the named data arrays at `location`, dropping the rest
 *  there. The other locations are untouched. */
val data_keep_js(const val& rMeshObj, const std::string& rLocation, const val& rNames,
                 bool ignoreMissing) {
    return with_js_errors([&]() -> val {
        return mesh_to_val(meshioplusplus::data_keep(
            val_to_mesh(rMeshObj), meshioplusplus::data_location_from_name(rLocation),
            val_to_string_vector(rNames), ignoreMissing));
    });
}

/** @brief Rename one data array, preserving values, dtype and shape. */
val data_rename_js(const val& rMeshObj, const std::string& rLocation, const std::string& rFrom,
                   const std::string& rTo) {
    return with_js_errors([&]() -> val {
        return mesh_to_val(meshioplusplus::data_rename(
            val_to_mesh(rMeshObj), meshioplusplus::data_location_from_name(rLocation), rFrom, rTo));
    });
}

/** @brief Average point_data onto the cells (mean over each cell's nodes).
 *  An empty `names` converts every point_data array. Output is Float64. */
val data_point_to_cell_js(const val& rMeshObj, const val& rNames, const std::string& rSuffix) {
    return with_js_errors([&]() -> val {
        meshioplusplus::DataAverageOptions opts;
        opts.names = val_to_string_vector(rNames);
        opts.suffix = rSuffix;
        return mesh_to_val(meshioplusplus::point_data_to_cell_data(val_to_mesh(rMeshObj), opts));
    });
}

/** @brief Average cell_data onto the points. `weight` is "uniform" or
 *  "measure". Points touched by no cell get NaN. Output is Float64. */
val data_cell_to_point_js(const val& rMeshObj, const val& rNames, const std::string& rWeight,
                          const std::string& rSuffix) {
    return with_js_errors([&]() -> val {
        meshioplusplus::DataAverageOptions opts;
        opts.names = val_to_string_vector(rNames);
        opts.weight = meshioplusplus::cell_point_weight_from_name(rWeight);
        opts.suffix = rSuffix;
        return mesh_to_val(meshioplusplus::cell_data_to_point_data(val_to_mesh(rMeshObj), opts));
    });
}

/** @brief Evaluate an elementwise expression into a new array. The grammar
 *  accepts + - * /, unary minus, parentheses, numbers, array names and the
 *  functions abs/sqrt/min/max/norm -- nothing else is evaluated. */
val data_calc_js(const val& rMeshObj, const std::string& rExpression, const std::string& rLocation,
                 const std::string& rOutputName, bool overwrite) {
    return with_js_errors([&]() -> val {
        meshioplusplus::DataCalcOptions opts;
        opts.location = meshioplusplus::data_location_from_name(rLocation);
        opts.output = rOutputName;
        opts.overwrite = overwrite;
        return mesh_to_val(meshioplusplus::data_calc(val_to_mesh(rMeshObj), rExpression, opts));
    });
}

/** @brief Condition values: `mode` is "clamp"/"normalize"/"standardize",
 *  `scope` is "component"/"magnitude", `nanPolicy` is
 *  "ignore"/"replace"/"fail". */
val data_condition_js(const val& rMeshObj, const std::string& rLocation, const val& rNames,
                      const std::string& rMode, double lo, double hi, const std::string& rScope,
                      const std::string& rNanPolicy, double nanReplacement,
                      const std::string& rSuffix) {
    return with_js_errors([&]() -> val {
        meshioplusplus::DataConditionOptions opts;
        opts.location = meshioplusplus::data_location_from_name(rLocation);
        opts.names = val_to_string_vector(rNames);
        opts.mode = meshioplusplus::condition_mode_from_name(rMode);
        opts.scope = meshioplusplus::condition_scope_from_name(rScope);
        opts.lo = lo;
        opts.hi = hi;
        opts.nan_policy = meshioplusplus::nan_policy_from_name(rNanPolicy);
        opts.nan_replacement = nanReplacement;
        opts.suffix = rSuffix;
        return mesh_to_val(meshioplusplus::data_condition(val_to_mesh(rMeshObj), opts));
    });
}

/** @brief Read-only per-array data summary. Returns a JS array of objects with
 *  `location`, `name`, `dtype`, `shape`, `numBlocks`, `numEntries`,
 *  `numComponents`, `numValues`, `min`, `max`, `mean`, the three
 *  `*PerComponent` arrays, `numNan`, `numInf`, `numFinite` and
 *  `inconsistentBlocks`. */
val data_info_js(const val& rMeshObj) {
    return with_js_errors([&]() -> val {
        meshioplusplus::DataInfoReport r = meshioplusplus::data_info(val_to_mesh(rMeshObj));
        auto to_array = [](const std::vector<double>& rValues) {
            val a = val::array();
            for (double v : rValues)
                a.call<void>("push", v);
            return a;
        };
        val out = val::array();
        for (const meshioplusplus::DataArrayInfo& a : r.mArrays) {
            val o = val::object();
            o.set("location", std::string(meshioplusplus::data_location_name(a.mLocation)));
            o.set("name", a.mName);
            o.set("dtype", std::string(meshioplusplus::dtype_numpy_str(a.mDtype)));
            val shape = val::array();
            for (std::size_t d : a.mShape)
                shape.call<void>("push", static_cast<double>(d));
            o.set("shape", shape);
            o.set("numBlocks", static_cast<double>(a.mNumBlocks));
            o.set("numEntries", static_cast<double>(a.mNumEntries));
            o.set("numComponents", static_cast<double>(a.mNumComponents));
            o.set("numValues", static_cast<double>(a.mNumValues));
            o.set("min", a.mMin);
            o.set("max", a.mMax);
            o.set("mean", a.mMean);
            o.set("minPerComponent", to_array(a.mMinPerComponent));
            o.set("maxPerComponent", to_array(a.mMaxPerComponent));
            o.set("meanPerComponent", to_array(a.mMeanPerComponent));
            o.set("numNan", static_cast<double>(a.mNumNan));
            o.set("numInf", static_cast<double>(a.mNumInf));
            o.set("numFinite", static_cast<double>(a.mNumFinite));
            o.set("inconsistentBlocks", a.mInconsistentBlocks);
            out.call<void>("push", o);
        }
        return out;
    });
}

/** @brief Cell-measure-weighted total/mean of one or more cell_data arrays --
 *  gradient's integration counterpart (gradient differentiates a field, this
 *  integrates one). `arrays` is a JS array of names (undefined/null/empty =
 *  every cell_data array). Every sum is weighted by the cell's own
 *  length/area/volume, excluding both a cell with unmeasurable geometry and
 *  a component with a non-finite value from that component's numerator and
 *  denominator. Returns a JS array of objects with `name`, `numComponents`,
 *  `domain` (`numCells`, `numSkipped`, `totalPerComponent`,
 *  `meanPerComponent`, `domainMeasurePerComponent`, `numNanPerComponent`)
 *  and `regions` (the same shape plus `name`, one per named Cell region
 *  present -- a cell in two regions contributes fully to both, one in none
 *  contributes to neither). A point_data-only name throws naming
 *  pointDataToCellData as the fix. */
val data_integrate_js(const val& rMeshObj, const val& rArrays) {
    return with_js_errors([&]() -> val {
        meshioplusplus::DataIntegrateOptions opts;
        opts.mArrayNames = val_to_string_vector(rArrays);
        meshioplusplus::DataIntegrateReport r =
            meshioplusplus::data_integrate(val_to_mesh(rMeshObj), opts);
        auto to_array_d = [](const std::vector<double>& rValues) {
            val a = val::array();
            for (double v : rValues)
                a.call<void>("push", v);
            return a;
        };
        auto to_array_i = [](const std::vector<std::int64_t>& rValues) {
            val a = val::array();
            for (std::int64_t v : rValues)
                a.call<void>("push", static_cast<double>(v));
            return a;
        };
        auto region_to_val = [&](const meshioplusplus::FieldIntegralRegion& rRegion) {
            val o = val::object();
            o.set("numCells", static_cast<double>(rRegion.mNumCells));
            o.set("numSkipped", static_cast<double>(rRegion.mNumSkipped));
            o.set("totalPerComponent", to_array_d(rRegion.mTotalPerComponent));
            o.set("meanPerComponent", to_array_d(rRegion.mMeanPerComponent));
            o.set("domainMeasurePerComponent", to_array_d(rRegion.mDomainMeasurePerComponent));
            o.set("numNanPerComponent", to_array_i(rRegion.mNumNanPerComponent));
            return o;
        };
        val out = val::array();
        for (const meshioplusplus::FieldIntegralArray& a : r.mArrays) {
            val o = val::object();
            o.set("name", a.mName);
            o.set("numComponents", static_cast<double>(a.mNumComponents));
            o.set("domain", region_to_val(a.mDomain));
            val regions = val::array();
            for (const meshioplusplus::FieldIntegralRegion& rr : a.mRegions) {
                val ro = region_to_val(rr);
                ro.set("name", rr.mName);
                regions.call<void>("push", ro);
            }
            o.set("regions", regions);
            out.call<void>("push", o);
        }
        return out;
    });
}

/**
 * @brief Cross-mesh field transfer: sample the source's data arrays onto the
 * target (source point_data at the target's points, source cell_data by
 * nearest source-cell centroid regardless of the method). `method` is
 * "nearest" or "barycentric"; `arrays` is a JS array of source names
 * (undefined/null/empty = every source point_data array; cell_data only when
 * named); `onConflict` is "error", "overwrite" or "suffix" (name + "_interp").
 * Returns the target copy as a plain mesh object.
 */
val interpolate_js(const val& rSourceObj, const val& rTargetObj, const std::string& rMethod,
                   const val& rArrays, bool extrapolate, double defaultValue,
                   const std::string& rOnConflict) {
    return with_js_errors([&]() -> val {
        meshioplusplus::InterpolateOptions options;
        options.mMethod = meshioplusplus::interpolate_method_from_name(rMethod);
        options.mArrays = val_to_string_vector(rArrays);
        options.mExtrapolate = extrapolate;
        options.mDefaultValue = defaultValue;
        options.mOnConflict = meshioplusplus::interpolate_conflict_from_name(rOnConflict);
        return mesh_to_val(
            meshioplusplus::interpolate(val_to_mesh(rSourceObj), val_to_mesh(rTargetObj), options));
    });
}

/**
 * @brief Mass-preserving cross-mesh field transfer: an exact overlap-measure
 * weighted remap, so sum(target value * target measure) equals sum(source
 * value * source measure) over the shared region -- the property
 * `interpolate`'s "barycentric" mode does not have. Both meshes are
 * simplexified (accepting ragged/polyhedron blocks for free). Unlike
 * `interpolate`, `arrays` undefined/null/empty means every source
 * point_data AND cell_data array (one algorithm regardless of location).
 * `onConflict` is "error", "overwrite" or "suffix" (name + "_interp").
 * Returns the target copy as a plain mesh object (always Float64 arrays).
 */
val conservative_interpolate_js(const val& rSourceObj, const val& rTargetObj, const val& rArrays,
                                double defaultValue, const std::string& rOnConflict) {
    return with_js_errors([&]() -> val {
        meshioplusplus::ConservativeInterpolateOptions options;
        options.mArrays = val_to_string_vector(rArrays);
        options.mDefaultValue = defaultValue;
        options.mOnConflict =
            meshioplusplus::conservative_interpolate_conflict_from_name(rOnConflict);
        return mesh_to_val(meshioplusplus::conservative_interpolate(
            val_to_mesh(rSourceObj), val_to_mesh(rTargetObj), options));
    });
}

/**
 * @brief Green-element undo: restore `fine`'s transitional (closure-only)
 * cells back to their original parent, read verbatim from `coarse` -- a
 * lookup, not a reconstruction, since `refine` never renumbers or prunes
 * points. `fine` must carry `refine:cell_id`/`refine:parent_id`/
 * `refine:level` (i.e. must come from `refine(coarse, {recordHierarchy:
 * true, recordLevels: true})`); `coarse` must be the mesh that call was run
 * on. Returns `{ mesh, numGroupsUndone, numCellsRemoved }`.
 */
val undo_green_js(const val& rCoarseObj, const val& rFineObj) {
    return with_js_errors([&]() -> val {
        meshioplusplus::UndoGreenResult r =
            meshioplusplus::undo_green(val_to_mesh(rCoarseObj), val_to_mesh(rFineObj));
        val out = val::object();
        out.set("mesh", mesh_to_val(r.mMesh));
        out.set("numGroupsUndone", static_cast<double>(r.mNumGroupsUndone));
        out.set("numCellsRemoved", static_cast<double>(r.mNumCellsRemoved));
        return out;
    });
}

// ---------------------------------------------------------------------
// Transient (time-series) XDMF -- the one *stateful* binding in this file.
//
// Every other export here is a pure function over plain JS values: read a
// path, get an object; hand back an object, get a file. A series is not that
// shape at all -- the mesh is written once and each solve appends a step, and
// the `.xdmf` light data only lands at `Finalize()` -- so it needs an object
// that survives between calls.
//
// Shape chosen: an **opaque integer handle plus free functions**, not an
// embind `class_`. Three reasons, in order of weight:
//
//   1. This file's stated contract is that `NDArray`/`CellBlock`/`Mesh` stay
//      entirely internal and "JS only ever sees plain objects of typed
//      arrays". An embind `class_` hands JS a live C++ object whose lifetime
//      the caller manages with Emscripten's `.delete()` -- exactly the kind of
//      leaked C++ handle the rest of the API avoids, and a concept that has no
//      counterpart anywhere else in `@meshioplusplus/wasm`.
//   2. Free functions keep every export uniform: `src/wasm/src/index.mjs` can
//      wrap them the same way it wraps everything else (defaults + an
//      ergonomic object), and each one goes through `with_js_errors`, so a
//      `WriteError` is a readable JS `Error` here as it is everywhere else.
//      A bound member function would surface as a bare, message-less
//      `WebAssembly.Exception`.
//   3. It matches the C API (`mio_xdmf_series*`), so the two flat bindings
//      describe the same object the same way.
//
// The handle is an index into a module-local table rather than a
// `reinterpret_cast` pointer: JS numbers are trivially forgeable and a series
// is explicitly closed, so a stale handle must be a thrown `Error`, never a
// use-after-free inside linear memory.

std::unordered_map<int, std::unique_ptr<meshioplusplus::XdmfTimeSeriesWriter>>&
xdmf_series_table() {
    static std::unordered_map<int, std::unique_ptr<meshioplusplus::XdmfTimeSeriesWriter>> table;
    return table;
}

/// Resolve a handle or throw. Never returns null.
meshioplusplus::XdmfTimeSeriesWriter& xdmf_series_lookup(int handle) {
    auto it = xdmf_series_table().find(handle);
    if (it == xdmf_series_table().end())
        throw meshioplusplus::WriteError(
            "meshio++ (wasm): invalid or already-closed XDMF time-series handle " +
            std::to_string(handle));
    return *it->second;
}

/**
 * @brief Open a transient XDMF series on the virtual filesystem.
 * @param rPath virtual FS path of the `.xdmf` light-data file.
 * @param rDataFormat "HDF" (companion `<base>.h5`; this build has HDF5),
 *   "XML" (numbers inline, one file) or "Binary" (sibling `<base><n>.bin`).
 * @param gzipLevel gzip level for "HDF" datasets; negative = uncompressed.
 * @return an opaque handle to pass to the other `xdmfSeries*` functions.
 * @throws meshioplusplus::WriteError on an unknown data format.
 */
int xdmf_series_create_js(const std::string& rPath, const std::string& rDataFormat, int gzipLevel,
                          const std::string& rMode, bool autoFlush) {
    return with_js_errors([&]() -> int {
        static int next_handle = 1;
        if (rMode != "truncate" && rMode != "append")
            throw meshioplusplus::WriteError(
                "meshio++ (wasm): mode must be 'truncate' or "
                "'append', got '" +
                rMode + "'");
        auto writer = std::make_unique<meshioplusplus::XdmfTimeSeriesWriter>(
            rPath, rDataFormat, gzipLevel,
            rMode == "append" ? meshioplusplus::XdmfSeriesMode::Append
                              : meshioplusplus::XdmfSeriesMode::Truncate);
        writer->SetAutoFlush(autoFlush);
        const int handle = next_handle++;
        xdmf_series_table().emplace(handle, std::move(writer));
        return handle;
    });
}

/** @brief Write the static grid (points + cells) every step shares. Once. */
void xdmf_series_write_points_cells_js(int handle, const val& rMeshObj) {
    with_js_errors([&]() { xdmf_series_lookup(handle).WritePointsCells(val_to_mesh(rMeshObj)); });
}

/** @brief Append one step's `point_data`/`cell_data` at simulation time `time`. */
void xdmf_series_write_data_js(int handle, double time, const val& rMeshObj) {
    with_js_errors([&]() { xdmf_series_lookup(handle).WriteData(time, val_to_mesh(rMeshObj)); });
}

/**
 * @brief Append one step from `{name: Float64Array}` objects, with no mesh.
 *
 * The granularity a solver has once `xdmfSeriesWritePointsCells` has fixed the
 * geometry. Components come from an optional parallel `{name: n}` object,
 * defaulting to 1, since a flat typed array carries no shape of its own.
 */
void xdmf_series_write_data_arrays_js(int handle, double time, const val& rPointData,
                                      const val& rCellData, const val& rComponents) {
    with_js_errors([&]() {
        const auto convert = [&](const val& rSrc) {
            std::vector<meshioplusplus::XdmfTimeSeriesWriter::NamedArray> out;
            if (rSrc.isNull() || rSrc.isUndefined())
                return out;
            const val keys = val::global("Object").call<val>("keys", rSrc);
            const unsigned n = keys["length"].as<unsigned>();
            for (unsigned i = 0; i < n; ++i) {
                const std::string name = keys[i].as<std::string>();
                meshioplusplus::XdmfTimeSeriesWriter::NamedArray a;
                a.mName = name;
                a.mNumComponents = 1;
                if (!rComponents.isNull() && !rComponents.isUndefined()) {
                    const val nc = rComponents[name];
                    if (!nc.isUndefined() && !nc.isNull())
                        a.mNumComponents = static_cast<std::size_t>(nc.as<double>());
                }
                a.mValues = emscripten::convertJSArrayToNumberVector<double>(rSrc[name]);
                out.push_back(std::move(a));
            }
            return out;
        };
        xdmf_series_lookup(handle).WriteData(time, convert(rPointData), convert(rCellData));
    });
}

/**
 * @brief Write the `.xdmf` as it stands, without finalizing.
 *
 * So a run that is killed or still going leaves a readable file covering every
 * flushed step. Safe to call repeatedly; a no-op once finalized.
 */
void xdmf_series_flush_js(int handle) {
    with_js_errors([&]() { xdmf_series_lookup(handle).Flush(); });
}

/** @brief Write the `.xdmf` light data and close the heavy-data container.
 *  Idempotent; the handle stays valid (and queryable) afterwards. */
void xdmf_series_finalize_js(int handle) {
    with_js_errors([&]() { xdmf_series_lookup(handle).Finalize(); });
}

/** @brief How many steps have been written so far. */
double xdmf_series_num_steps_js(int handle) {
    return with_js_errors(
        [&]() -> double { return static_cast<double>(xdmf_series_lookup(handle).NumSteps()); });
}

/** @brief Whether `Finalize()` has already run for this handle. */
bool xdmf_series_finalized_js(int handle) {
    return with_js_errors([&]() -> bool { return xdmf_series_lookup(handle).Finalized(); });
}

/**
 * @brief Destroy the writer, finalizing it first if it has not been.
 *
 * Deliberately tolerant of an unknown handle (double-close is a no-op) --
 * a JS wrapper's `close()` runs in `finally` blocks, where throwing on the
 * second call would mask the original error. Note the destructor swallows a
 * finalize failure (it must not throw); call `xdmfSeriesFinalize` first to
 * see one.
 */
void xdmf_series_free_js(int handle) {
    xdmf_series_table().erase(handle);
}

}  // namespace

// --- Provenance (see doc/provenance.md) -----------------------------------
//
// Binds the C++ core directly (this package's pattern) rather than the flat C
// ABI. Exposed as begin/end plus setters; `index.mjs` wraps them in a
// `withProvenance(mode, fn)` callback so a throw inside the body cannot strand
// the scope. A thread-local stack of owned scopes gives the two calls a
// lifetime, exactly as the Python bridge and the C ABI already do -- the scope
// object itself never crosses into JS.
namespace {

std::vector<std::unique_ptr<meshioplusplus::detail::ProvenanceScope>>& js_provenance_stack() {
    static std::vector<std::unique_ptr<meshioplusplus::detail::ProvenanceScope>> stack;
    return stack;
}

void provenance_begin_js(int mode) {
    with_js_errors([&]() {
        if (mode < 0 || mode > 2)
            throw meshioplusplus::WriteError("meshio++: unknown provenance mode " +
                                             std::to_string(mode));
        js_provenance_stack().push_back(std::make_unique<meshioplusplus::detail::ProvenanceScope>(
            static_cast<meshioplusplus::detail::ProvenanceMode>(mode)));
        return 0;
    });
}

void provenance_end_js() {
    auto& stack = js_provenance_stack();
    if (!stack.empty())
        stack.pop_back();
}

void provenance_note_js(std::string category, std::string detail) {
    meshioplusplus::detail::provenance_note(category, detail);
}

void provenance_set_source_js(std::string path, std::string format) {
    meshioplusplus::detail::provenance_set_source(path, format);
}

void provenance_set_target_js(std::string format, std::string encoding, std::string codec,
                              std::string float_format) {
    meshioplusplus::detail::provenance_set_target(format, encoding, codec, float_format);
}

val read_provenance_js(std::string path) {
    return with_js_errors([&]() -> val {
        auto found = meshioplusplus::detail::read_provenance_lines(path);
        val lines = val::array();
        for (std::size_t i = 0; i < found.mLines.size(); ++i)
            lines.set(i, found.mLines[i]);
        val out = val::object();
        out.set("lines", lines);
        out.set("recognised", found.mRecognised);
        return out;
    });
}

}  // namespace

EMSCRIPTEN_BINDINGS(meshioplusplus_wasm) {
    emscripten::function("readMesh", &read_mesh);
    emscripten::function("readMeshSelective", &read_mesh_selective);
    emscripten::function("readMetadata", &read_metadata_js);
    emscripten::function("readerSupportsOptions", &reader_supports_options_js);
    emscripten::function("writeMesh", &write_mesh);
    emscripten::function("convert", &convert);
    emscripten::function("convertSurface", &convert_surface);
    emscripten::function("convertSurfaceOps", &convert_surface_ops);
    emscripten::function("runPipeline", &run_pipeline_js);
    // Sequences (multi-file / transient datasets) over MEMFS paths -- the same
    // surface `convert`/`runPipeline` already work on. See doc/sequences.md.
    emscripten::function("sequenceEntries", &sequence_entries_js);
    emscripten::function("sequenceToTimeseries", &sequence_to_timeseries_js);
    emscripten::function("timeseriesToSequence", &timeseries_to_sequence_js);
    emscripten::function("numNodesPerCell", &num_nodes_per_cell_js);
    emscripten::function("topologicalDimension", &topological_dimension_js);
    emscripten::function("meshBackend", &mesh_backend_js);
    emscripten::function("parallelBackend", &parallel_backend_js);
    emscripten::function("hasCgnslib", &has_cgnslib_js);
    emscripten::function("availableFormats", &available_formats_js);
    emscripten::function("extractSurface", &extract_surface_js);
    emscripten::function("extractSkin", &extract_skin_js);
    emscripten::function("attachQuality", &attach_quality_js);
    emscripten::function("sniffFormat", &sniff_format_js);
    emscripten::function("reorder", &reorder_js);
    emscripten::function("computeBandwidth", &compute_bandwidth_js);
    emscripten::function("diff", &diff_js);
    emscripten::function("meshesEqual", &meshes_equal_js);
    emscripten::function("merge", &merge_js);
    emscripten::function("transform", &transform_js);
    emscripten::function("clean", &clean_js);
    emscripten::function("smooth", &smooth_js);
    emscripten::function("interpolate", &interpolate_js);
    emscripten::function("conservativeInterpolate", &conservative_interpolate_js);
    emscripten::function("undoGreen", &undo_green_js);
    emscripten::function("slice", &slice_js);
    emscripten::function("isosurface", &isosurface_js);
    emscripten::function("grid", &grid_js);
    emscripten::function("voxelize", &voxelize_js);
    emscripten::function("surfaceWatertightCheck", &surface_watertight_check_js);
    emscripten::function("sampleDistance", &sample_distance_js);
    emscripten::function("distanceToSurface", &distance_to_surface_js);
    emscripten::function("computeSdf", &compute_sdf_js);
    emscripten::function("gradient", &gradient_js);
    emscripten::function("hessian", &hessian_js);
    emscripten::function("estimateError", &estimate_error_js);
    emscripten::function("remesh", &remesh_js);
    emscripten::function("remeshVolume", &remesh_volume_js);
    emscripten::function("optimizeVolume", &optimize_volume_js);
    emscripten::function("computeCurvature", &compute_curvature_js);
    emscripten::function("repair", &repair_js);
    emscripten::function("shrinkwrap", &shrinkwrap_js);
    emscripten::function("sobolevDeform", &sobolev_deform_js);
    emscripten::function("cropBbox", &crop_bbox_js);
    emscripten::function("cropPlane", &crop_plane_js);
    emscripten::function("cropPredicate", &crop_predicate_js);
    emscripten::function("split", &split_js);
    emscripten::function("convertCells", &convert_cells_js);
    emscripten::function("subdivide", &subdivide_js);
    emscripten::function("agglomerate", &agglomerate_js);
    emscripten::function("refine", &refine_js);
    emscripten::function("decimate", &decimate_js);
    emscripten::function("partition", &partition_js);
    emscripten::function("partitionLabels", &partition_labels_js);
    emscripten::function("stats", &stats_js);
    emscripten::function("provenanceBegin", &provenance_begin_js);
    emscripten::function("provenanceEnd", &provenance_end_js);
    emscripten::function("provenanceNote", &provenance_note_js);
    emscripten::function("provenanceSetSource", &provenance_set_source_js);
    emscripten::function("provenanceSetTarget", &provenance_set_target_js);
    emscripten::function("readProvenance", &read_provenance_js);
    emscripten::function("dataDrop", &data_drop_js);
    emscripten::function("dataKeep", &data_keep_js);
    emscripten::function("dataRename", &data_rename_js);
    emscripten::function("dataPointToCell", &data_point_to_cell_js);
    emscripten::function("dataCellToPoint", &data_cell_to_point_js);
    emscripten::function("dataCalc", &data_calc_js);
    emscripten::function("dataCondition", &data_condition_js);
    emscripten::function("dataInfo", &data_info_js);
    emscripten::function("dataIntegrate", &data_integrate_js);
    // Transient XDMF: the one stateful surface -- an opaque handle plus these
    // seven calls (see the block comment above their definitions for why it is
    // not an embind class_).
    emscripten::function("xdmfSeriesCreate", &xdmf_series_create_js);
    emscripten::function("xdmfSeriesWritePointsCells", &xdmf_series_write_points_cells_js);
    emscripten::function("xdmfSeriesWriteData", &xdmf_series_write_data_js);
    emscripten::function("xdmfSeriesWriteDataArrays", &xdmf_series_write_data_arrays_js);
    emscripten::function("xdmfSeriesFlush", &xdmf_series_flush_js);
    emscripten::function("xdmfSeriesFinalize", &xdmf_series_finalize_js);
    emscripten::function("xdmfSeriesNumSteps", &xdmf_series_num_steps_js);
    emscripten::function("xdmfSeriesFinalized", &xdmf_series_finalized_js);
    emscripten::function("xdmfSeriesFree", &xdmf_series_free_js);
}
