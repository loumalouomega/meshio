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
// System includes
#include <memory>
#include <vector>

// External includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Project includes
#include "meshioplusplus/detail/colormap.hpp"
#include "meshioplusplus/detail/provenance.hpp"
#include "meshioplusplus/detail/cell_subdivision.hpp"
#include "meshioplusplus/detail/refine_templates.hpp"
#include "meshioplusplus/exceptions.hpp"
#include "meshioplusplus/version.hpp"
#include "meshioplusplus/formats/abaqus.hpp"
#include "meshioplusplus/formats/ansys.hpp"
#include "meshioplusplus/formats/ansysinp.hpp"
#include "meshioplusplus/formats/avsucd.hpp"
#ifdef MESHIOPLUSPLUS_HAS_HDF5
#include "meshioplusplus/formats/cgns.hpp"
#include "meshioplusplus/formats/h5m.hpp"
#include "meshioplusplus/formats/hmf.hpp"
#include "meshioplusplus/formats/med.hpp"
#endif
#include "meshioplusplus/formats/dolfin.hpp"
#include "meshioplusplus/formats/ensight.hpp"
#ifdef MESHIOPLUSPLUS_HAS_NETCDF
#include "meshioplusplus/formats/exodus.hpp"
#endif
#include "meshioplusplus/formats/dex.hpp"
#include "meshioplusplus/formats/flac3d.hpp"
#include "meshioplusplus/formats/flux.hpp"
#include "meshioplusplus/formats/freefem.hpp"
#include "meshioplusplus/formats/gid.hpp"
#include "meshioplusplus/formats/gmsh.hpp"
#include "meshioplusplus/formats/ip.hpp"
#include "meshioplusplus/formats/mdpa.hpp"
#include "meshioplusplus/formats/medit.hpp"
#include "meshioplusplus/formats/mff.hpp"
#include "meshioplusplus/formats/mfm.hpp"
#include "meshioplusplus/formats/mphtxt.hpp"
#include "meshioplusplus/formats/nastran.hpp"
#include "meshioplusplus/formats/netgen.hpp"
#include "meshioplusplus/formats/obj_off.hpp"
#include "meshioplusplus/formats/openfoam.hpp"
#include "meshioplusplus/formats/permas.hpp"
#include "meshioplusplus/formats/ply.hpp"
#include "meshioplusplus/formats/stl.hpp"
#include "meshioplusplus/formats/su2.hpp"
#include "meshioplusplus/formats/svg.hpp"
#include "meshioplusplus/formats/tecplot.hpp"
#include "meshioplusplus/formats/tetgen.hpp"
#include "meshioplusplus/formats/triangle.hpp"
#include "meshioplusplus/formats/vtp.hpp"
#include "meshioplusplus/formats/tikz.hpp"
#include "meshioplusplus/formats/ugrid.hpp"
#include "meshioplusplus/formats/unv.hpp"
#include "meshioplusplus/formats/vti.hpp"
#include "meshioplusplus/formats/vtk.hpp"
#include "meshioplusplus/formats/wkt.hpp"
#include "meshioplusplus/formats/vtu.hpp"
#include "meshioplusplus/formats/xdmf.hpp"
#include "meshioplusplus/formats/xdmf_time_series.hpp"
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
#include "meshioplusplus/operations/decimate.hpp"
#include "meshioplusplus/operations/decimate_volume.hpp"
#include "meshioplusplus/operations/data_calc.hpp"
#include "meshioplusplus/operations/data_common.hpp"
#include "meshioplusplus/operations/data_condition.hpp"
#include "meshioplusplus/operations/data_info.hpp"
#include "meshioplusplus/operations/data_integrate.hpp"
#include "meshioplusplus/operations/data_manage.hpp"
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
#include "meshioplusplus/parallel.hpp"
#include "meshioplusplus/read_options.hpp"
#include "meshioplusplus/registry.hpp"
#include "meshioplusplus/skin.hpp"
#include "meshioplusplus/types.hpp"
#include "np_conversions.hpp"

namespace py = pybind11;

namespace meshioplusplus_py {

/**
 * @brief The thread-local stack of open provenance scopes bridged in from
 * Python (see `_provenance.scope` and the `provenance_scope_push`/`_pop`
 * bindings at the bottom of this file).
 *
 * Owning `unique_ptr`s so a pushed scope stays alive -- and its RAII
 * destructor therefore does NOT fire -- until the matching pop, at which
 * point popping the vector destroys it and its destructor restores the
 * previous C++-side state exactly as a directly-nested `ProvenanceScope`
 * would.
 */
inline std::vector<std::unique_ptr<meshioplusplus::detail::ProvenanceScope>>&
provenance_scope_stack() {
    thread_local std::vector<std::unique_ptr<meshioplusplus::detail::ProvenanceScope>> stack;
    return stack;
}

}  // namespace meshioplusplus_py

namespace {

/**
 * @brief Build a `ReadOptions` from the reader bindings' keyword arguments.
 *
 * `arrays=None` means every array; an empty list means none. Passing them
 * through as `std::optional` keeps that distinction intact -- collapsing it
 * would make `arrays=[]` silently mean "everything".
 */
meshioplusplus::ReadOptions core_read_options(bool points_only, const py::object& rArrays,
                                              int time_step = 0) {
    meshioplusplus::ReadOptions opts;
    opts.mPointsOnly = points_only;
    opts.mTimeStep = time_step;
    if (!rArrays.is_none())
        opts.mDataArrays = rArrays.cast<std::vector<std::string>>();
    return opts;
}

/** @brief `--codec` / `compression=` name -> VtkCodec. */
meshioplusplus::detail::VtkCodec core_codec_from_name(const std::string& rName) {
    using meshioplusplus::detail::VtkCodec;
    if (rName.empty() || rName == "none")
        return VtkCodec::None;
    if (rName == "zlib")
        return VtkCodec::Zlib;
    if (rName == "lz4")
        return VtkCodec::LZ4;
    if (rName == "zstd")
        return VtkCodec::ZSTD;
    if (rName == "lzma")
        return VtkCodec::LZMA;
    throw meshioplusplus::WriteError("meshio++: unknown codec '" + rName +
                                     "' (expected zlib, lz4, zstd or none)");
}

/** @brief `MeshMetadata` -> the dict shape the Python layer exposes. */
py::dict core_metadata_to_py(const meshioplusplus::MeshMetadata& rMeta) {
    py::list blocks;
    for (const meshioplusplus::CellBlockInfo& block : rMeta.mCellBlocks) {
        py::dict entry;
        entry["type"] = block.mType;
        entry["num_cells"] = block.mNumCells;
        entry["nodes_per_cell"] = block.mNodesPerCell;
        entry["ragged"] = block.mRagged;
        blocks.append(std::move(entry));
    }

    py::dict out;
    out["num_points"] = rMeta.mNumPoints;
    out["point_dim"] = rMeta.mPointDim;
    out["num_cells"] = rMeta.NumCells();
    out["cell_blocks"] = std::move(blocks);
    out["point_data_names"] = rMeta.mPointDataNames;
    out["cell_data_names"] = rMeta.mCellDataNames;
    out["field_data_names"] = rMeta.mFieldDataNames;
    out["fell_back_to_full_read"] = rMeta.mFellBackToFullRead;
    out["format"] = rMeta.mFormat;
    // Always present (empty for a format with no time concept), so a caller can
    // write `len(meta["time_values"])` without first testing for the key.
    out["time_values"] = rMeta.mTimeValues;
    // Always present too (empty on a native metadata path, or for a format
    // with no regions), so a caller can iterate without testing the key.
    py::list regions;
    for (const meshioplusplus::RegionSummary& r : rMeta.mRegions) {
        py::dict entry;
        entry["name"] = r.mName;
        entry["kind"] = std::string(meshioplusplus::region_kind_name(r.mKind));
        entry["dim"] = r.mDim;
        entry["tag"] = r.mTag;
        entry["num_entries"] = r.mNumEntries;
        regions.append(std::move(entry));
    }
    out["regions"] = std::move(regions);
    // Always present too (empty for a file carrying no block), matching
    // time_values/regions above, so a caller can iterate without a key test.
    out["provenance"] = rMeta.mProvenance;
    out["provenance_recognised"] = rMeta.mProvenanceRecognised;
    // Absent rather than None-valued when not computed, so callers must ask
    // for it explicitly instead of accidentally treating "not computed" as a
    // real box at the origin.
    if (rMeta.mHasBBox) {
        out["bbox_min"] = py::make_tuple(rMeta.mBBoxMin[0], rMeta.mBBoxMin[1], rMeta.mBBoxMin[2]);
        out["bbox_max"] = py::make_tuple(rMeta.mBBoxMax[0], rMeta.mBBoxMax[1], rMeta.mBBoxMax[2]);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "meshio++ C++ core (pybind11)";
    m.attr("__cpp_core__") = true;

    // Build-time capability flags: the HDF5/netCDF formats are optional and
    // compile out when the libraries are absent (Python is the fallback).
#ifdef MESHIOPLUSPLUS_HAS_HDF5
    m.attr("__has_hdf5__") = true;
#else
    m.attr("__has_hdf5__") = false;
#endif
#ifdef MESHIOPLUSPLUS_HAS_NETCDF
    m.attr("__has_netcdf__") = true;
#else
    m.attr("__has_netcdf__") = false;
#endif
#ifdef MESHIOPLUSPLUS_HAS_CGNSLIB
    m.attr("__has_cgnslib__") = true;
#else
    m.attr("__has_cgnslib__") = false;
#endif
#ifdef MESHIOPLUSPLUS_HAS_ZLIB
    m.attr("__has_zlib__") = true;
#else
    m.attr("__has_zlib__") = false;
#endif
#ifdef MESHIOPLUSPLUS_HAS_ZSTD
    m.attr("__has_zstd__") = true;
#else
    m.attr("__has_zstd__") = false;
#endif
#ifdef MESHIOPLUSPLUS_HAS_LZ4
    m.attr("__has_lz4__") = true;
#else
    m.attr("__has_lz4__") = false;
#endif
#ifdef MESHIOPLUSPLUS_HAS_KAHIP
    m.attr("__has_kahip__") = true;
#else
    m.attr("__has_kahip__") = false;
#endif
#ifdef MESHIOPLUSPLUS_HAS_GIDPOST
    m.attr("__has_gidpost__") = true;
#else
    m.attr("__has_gidpost__") = false;
#endif
#ifdef MESHIOPLUSPLUS_HAS_GIDPOST_HDF5
    m.attr("__has_gidpost_hdf5__") = true;
#else
    m.attr("__has_gidpost_hdf5__") = false;
#endif
#ifdef MESHIOPLUSPLUS_HAS_JSON
    m.attr("__has_json__") = true;
#else
    m.attr("__has_json__") = false;
#endif
    // Active compile-time parallel backend ("seq"/"stl"/"openmp"/"tbb"): lets
    // callers verify that parallel_for actually threads (STL without TBB is
    // effectively sequential).
    m.attr("__parallel_backend__") = meshioplusplus::parallel_backend_name();
    // Active compile-time mesh backend — always "meshio" here (the Python
    // extension refuses to build against any other; see CMakeLists.txt),
    // exposed for symmetry with the standalone/WASM builds.
    m.attr("__mesh_backend__") = meshioplusplus::mesh_backend_name();
    // The release this extension was compiled against (MESHIOPLUSPLUS_VERSION_STRING),
    // so the Python-side provenance tag (meshioplusplus._provenance.TAG, built from
    // installed package metadata) can be pinned against what the C++ core actually
    // compiled with, rather than the two silently drifting.
    m.attr("__version__") = MESHIOPLUSPLUS_VERSION_STRING;

    // Translate C++ I/O errors to the existing Python exception classes.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (const meshioplusplus::ReadError& e) {
            py::object exc = py::module_::import("meshioplusplus").attr("ReadError");
            PyErr_SetString(exc.ptr(), e.what());
        } catch (const meshioplusplus::WriteError& e) {
            py::object exc = py::module_::import("meshioplusplus").attr("WriteError");
            PyErr_SetString(exc.ptr(), e.what());
        }
    });

    // Shared cell-type metadata (single source of truth with Python).
    m.def("num_nodes_per_cell", []() { return meshioplusplus::num_nodes_per_cell(); });
    m.def("topological_dimension", []() { return meshioplusplus::topological_dimension(); });

    // Debug helper: Python mesh -> C++ mesh (zero-copy views) -> Python mesh
    // (capsule-backed arrays). Exercises both conversion directions. With
    // allow_ragged=True it also round-trips polyhedron / jagged-polygon blocks
    // through the C++ ragged CellBlock representation (a copy).
    m.def(
        "_roundtrip",
        [](py::object pymesh, bool allow_ragged) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp =
                meshioplusplus_py::py_to_mesh(pymesh, refs,
                                              /*lenient_field_data=*/false, allow_ragged);
            return meshioplusplus_py::mesh_to_py(std::move(cpp));
        },
        py::arg("pymesh"), py::arg("allow_ragged") = false);

    // VTU writer (ascii / binary / zlib), zero-copy input from the Python mesh.
    m.def(
        "vtu_write_codec",
        [](const std::string& path, py::object pymesh, bool binary, const std::string& codec) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs,
                                                                     /*lenient_field_data=*/false,
                                                                     /*allow_ragged=*/false);
            meshioplusplus::write_vtu_codec(path, cpp, binary, core_codec_from_name(codec));
        },
        py::arg("path"), py::arg("mesh"), py::arg("binary") = true, py::arg("codec") = "zlib");

    m.def("vtu_write", [](const std::string& path, py::object pymesh, bool binary, bool zlib) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
        meshioplusplus::write_vtu(path, cpp, binary, zlib);
    });

    // VTI (ImageData) writer / reader. A lattice is rectangular by definition,
    // so allow_ragged stays false like VTU's.
    m.def(
        "vti_write_codec",
        [](const std::string& path, py::object pymesh, bool binary, const std::string& codec) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs,
                                                                     /*lenient_field_data=*/false,
                                                                     /*allow_ragged=*/false);
            meshioplusplus::write_vti_codec(path, cpp, binary, core_codec_from_name(codec));
        },
        py::arg("path"), py::arg("mesh"), py::arg("binary") = true, py::arg("codec") = "zlib");

    m.def("vti_write", [](const std::string& path, py::object pymesh, bool binary, bool zlib) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
        meshioplusplus::write_vti(path, cpp, binary, zlib);
    });

    m.def(
        "vti_read",
        [](const std::string& path, bool points_only, py::object arrays) {
            return meshioplusplus_py::mesh_to_py(
                meshioplusplus::read_vti(path, core_read_options(points_only, arrays)));
        },
        py::arg("path"), py::arg("points_only") = false, py::arg("arrays") = py::none());

    // VTP (PolyData) writer / reader; allow_ragged so jagged polygon blocks
    // reach the C++ writer (they are legal PolyData Polys rows).
    m.def(
        "vtp_write_codec",
        [](const std::string& path, py::object pymesh, bool binary, const std::string& codec) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs,
                                                                     /*lenient_field_data=*/false,
                                                                     /*allow_ragged=*/true);
            meshioplusplus::write_vtp_codec(path, cpp, binary, core_codec_from_name(codec));
        },
        py::arg("path"), py::arg("mesh"), py::arg("binary") = true, py::arg("codec") = "zlib");

    m.def("vtp_write", [](const std::string& path, py::object pymesh, bool binary, bool zlib) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs,
                                                                 /*lenient_field_data=*/false,
                                                                 /*allow_ragged=*/true);
        meshioplusplus::write_vtp(path, cpp, binary, zlib);
    });
    m.def(
        "vtp_read",
        [](const std::string& path, bool points_only, py::object arrays) {
            return meshioplusplus_py::mesh_to_py(
                meshioplusplus::read_vtp(path, core_read_options(points_only, arrays)));
        },
        py::arg("path"), py::arg("points_only") = false, py::arg("arrays") = py::none());

    // VTU reader -> Python mesh (zero-copy capsule-backed arrays).
    m.def(
        "vtu_read",
        [](const std::string& path, bool points_only, py::object arrays) {
            return meshioplusplus_py::mesh_to_py(
                meshioplusplus::read_vtu(path, core_read_options(points_only, arrays)));
        },
        py::arg("path"), py::arg("points_only") = false, py::arg("arrays") = py::none());

    // VTK writer (version 5.1 or 4.2; ascii or big-endian binary).
    m.def("vtk_write", [](const std::string& path, py::object pymesh, bool binary, bool v51) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
        meshioplusplus::write_vtk(path, cpp, binary, v51);
    });

    // VTK 5.1 reader -> Python mesh (zero-copy capsule-backed arrays).
    m.def("vtk_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_vtk(path));
    });

    // Boundary-skin extraction (volume mesh -> surface mesh).
    m.def(
        "extract_skin",
        [](py::object pymesh, bool linearize) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
            return meshioplusplus_py::mesh_to_py(meshioplusplus::extract_skin(cpp, linearize));
        },
        py::arg("mesh"), py::arg("linearize") = false);

    // Surface/boundary extraction (auto: volume -> faces, surface -> edges).
    m.def(
        "extract_surface",
        [](py::object pymesh, bool record_parent_ids) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
            return meshioplusplus_py::mesh_to_py(
                meshioplusplus::extract_surface(cpp, record_parent_ids));
        },
        py::arg("mesh"), py::arg("record_parent_ids") = false);

    // Mesh quality: attach per-cell metrics as cell_data.
    m.def(
        "attach_quality",
        [](py::object pymesh) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
            return meshioplusplus_py::mesh_to_py(meshioplusplus::attach_quality(cpp));
        },
        py::arg("mesh"));

    // Mesh quality: full per-cell report (summaries + arrays + counts).
    m.def(
        "compute_quality",
        [](py::object pymesh) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
            meshioplusplus::QualityReport rep = meshioplusplus::compute_quality(cpp);
            py::dict out;
            out["num_cells"] = rep.mNumCells;
            out["num_inverted"] = rep.mNumInverted;
            out["num_degenerate"] = rep.mNumDegenerate;
            py::dict metrics;
            for (auto& entry : rep.mMetrics) {
                const meshioplusplus::QualityMetricSummary& s = entry.second;
                py::dict d;
                d["min"] = s.mMin;
                d["max"] = s.mMax;
                d["mean"] = s.mMean;
                d["count"] = s.mCount;
                py::list hist;
                for (std::int64_t h : s.mHistogram)
                    hist.append(h);
                d["histogram"] = hist;
                metrics[py::str(entry.first)] = d;
            }
            out["metrics"] = metrics;
            py::dict cell_arrays;
            for (auto& entry : rep.mCellArrays) {
                py::list lst;
                for (meshioplusplus::NDArray& a : entry.second)
                    lst.append(meshioplusplus_py::numpy_from_ndarray(std::move(a)));
                cell_arrays[py::str(entry.first)] = lst;
            }
            out["cell_arrays"] = cell_arrays;
            return out;
        },
        py::arg("mesh"));

    // Content-based format detection ("" if unsure).
    m.def(
        "sniff_format", [](const std::string& path) { return meshioplusplus::sniff_format(path); },
        py::arg("path"));

    // Lightweight file summary: counts, cell-block shapes and data-array names
    // without materializing the heavy arrays. Formats lacking a native metadata
    // path are read in full and report fell_back_to_full_read=True -- the
    // answer is always correct, and always honest about whether it was cheap.
    m.def(
        "read_metadata",
        [](const std::string& path, const std::string& format) {
            std::string fmt = format;
            if (fmt.empty()) {
                try {
                    fmt = meshioplusplus::resolve_format(path, "");
                } catch (const meshioplusplus::ReadError&) {
                    fmt = meshioplusplus::sniff_format(path);  // read-only fallback
                    if (fmt.empty())
                        throw;
                }
            }
            return core_metadata_to_py(
                meshioplusplus::registry_read_metadata(path, fmt, meshioplusplus::ReadOptions{}));
        },
        py::arg("path"), py::arg("format") = "");

    // Whether `format` has a native selective-read path (rather than being read
    // whole and filtered afterwards).
    m.def(
        "reader_supports_options",
        [](const std::string& format) {
            return meshioplusplus::registry_reader_supports_options(format);
        },
        py::arg("format"));

    // Mesh renumbering (RCM / Morton / Hilbert). Returns a dict with the
    // permuted mesh and the applied node/cell permutations (old->new).
    m.def(
        "reorder",
        [](py::object pymesh, const std::string& method) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
            meshioplusplus::ReorderResult res =
                meshioplusplus::reorder(cpp, meshioplusplus::reorder_method_from_name(method));
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(res.mMesh));
            out["node_permutation"] =
                meshioplusplus_py::numpy_from_ndarray(std::move(res.mNodePermutation));
            py::list cell_perms;
            for (meshioplusplus::NDArray& a : res.mCellPermutations)
                cell_perms.append(meshioplusplus_py::numpy_from_ndarray(std::move(a)));
            out["cell_permutations"] = cell_perms;
            return out;
        },
        py::arg("mesh"), py::arg("method") = "rcm");

    // Connectivity bandwidth (max |i - j| over node pairs sharing a cell).
    m.def(
        "compute_bandwidth",
        [](py::object pymesh) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
            return meshioplusplus::compute_bandwidth(cpp);
        },
        py::arg("mesh"));

    // Mesh merge/weld. Takes a sequence of meshes and returns a dict with the
    // combined mesh and the per-input point/cell index maps (so the Python shim
    // can remap the shim-only point_sets/cell_sets). See operations/merge.hpp.
    m.def(
        "merge",
        [](py::sequence pymeshes, bool weld, double atol, bool source_tag,
           const std::string& data_policy, bool drop_duplicate_cells) {
            // Convert every input, keeping its numpy buffers alive for the call.
            std::vector<meshioplusplus_py::PyMeshRefs> refs(py::len(pymeshes));
            std::vector<meshioplusplus::Mesh> owned;
            owned.reserve(py::len(pymeshes));
            std::size_t idx = 0;
            for (py::handle h : pymeshes) {
                owned.push_back(meshioplusplus_py::py_to_mesh(
                    py::reinterpret_borrow<py::object>(h), refs[idx],
                    /*lenient_field_data=*/false, /*allow_ragged=*/true));
                ++idx;
            }
            std::vector<const meshioplusplus::Mesh*> ptrs;
            ptrs.reserve(owned.size());
            for (const meshioplusplus::Mesh& mm : owned)
                ptrs.push_back(&mm);

            meshioplusplus::MergeOptions opts;
            opts.weld = weld;
            opts.atol = atol;
            opts.source_tag = source_tag;
            opts.drop_duplicate_cells = drop_duplicate_cells;
            opts.data_policy = (data_policy == "fill")
                                   ? meshioplusplus::MergeDataPolicy::Fill
                                   : meshioplusplus::MergeDataPolicy::Intersection;
            meshioplusplus::MergeResult r = meshioplusplus::merge(ptrs, opts);

            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            py::list point_maps, cell_maps;
            for (meshioplusplus::NDArray& a : r.mPointMaps)
                point_maps.append(meshioplusplus_py::numpy_from_ndarray(std::move(a)));
            for (meshioplusplus::NDArray& a : r.mCellMaps)
                cell_maps.append(meshioplusplus_py::numpy_from_ndarray(std::move(a)));
            out["point_maps"] = point_maps;
            out["cell_maps"] = cell_maps;
            return out;
        },
        py::arg("meshes"), py::arg("weld") = false, py::arg("atol") = 1e-8,
        py::arg("source_tag") = true, py::arg("data_policy") = "intersection",
        py::arg("drop_duplicate_cells") = false);

    // Affine transform of point coordinates. Takes a row-major 4x4 matrix (16
    // doubles) and returns the transformed mesh. See operations/transform.hpp.
    m.def(
        "transform",
        [](py::object pymesh, const std::vector<double>& matrix, bool rotate_vector_data) {
            if (matrix.size() != 16)
                throw std::invalid_argument("transform: matrix must have 16 elements (4x4)");
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
            meshioplusplus::AffineTransform xf =
                meshioplusplus::transform_from_matrix(matrix.data());
            meshioplusplus::Mesh out = meshioplusplus::transform(cpp, xf, rotate_vector_data);
            return meshioplusplus_py::mesh_to_py(std::move(out));
        },
        py::arg("mesh"), py::arg("matrix"), py::arg("rotate_vector_data") = false);

    // Mesh cleanup (weld / prune / de-dup). Returns a dict with the cleaned mesh,
    // the point/cell index maps, and the removal counts. See operations/clean.hpp.
    m.def(
        "clean",
        [](py::object pymesh, bool weld, double atol, bool remove_orphans, bool drop_degenerate,
           bool drop_duplicate_cells) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::CleanOptions opts;
            opts.weld = weld;
            opts.atol = atol;
            opts.remove_orphans = remove_orphans;
            opts.drop_degenerate = drop_degenerate;
            opts.drop_duplicate_cells = drop_duplicate_cells;
            meshioplusplus::CleanResult r = meshioplusplus::clean(cpp, opts);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["point_map"] = meshioplusplus_py::numpy_from_ndarray(std::move(r.mPointMap));
            py::list cell_maps;
            for (meshioplusplus::NDArray& a : r.mCellMaps)
                cell_maps.append(meshioplusplus_py::numpy_from_ndarray(std::move(a)));
            out["cell_maps"] = cell_maps;
            out["points_welded"] = r.mPointsWelded;
            out["points_removed_orphan"] = r.mPointsRemovedOrphan;
            out["cells_dropped_degenerate"] = r.mCellsDroppedDegenerate;
            out["cells_dropped_duplicate"] = r.mCellsDroppedDuplicate;
            return out;
        },
        py::arg("mesh"), py::arg("weld") = false, py::arg("atol") = 1e-8,
        py::arg("remove_orphans") = true, py::arg("drop_degenerate") = true,
        py::arg("drop_duplicate_cells") = true);

    // Crop by bounding box / half-space. Return a dict with the pruned mesh and
    // the point/cell index maps. See operations/crop.hpp.
    auto crop_result_to_dict = [](meshioplusplus::CropResult&& r) {
        py::dict out;
        out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
        out["point_map"] = meshioplusplus_py::numpy_from_ndarray(std::move(r.mPointMap));
        py::list cell_maps;
        for (meshioplusplus::NDArray& a : r.mCellMaps)
            cell_maps.append(meshioplusplus_py::numpy_from_ndarray(std::move(a)));
        out["cell_maps"] = cell_maps;
        return out;
    };
    m.def(
        "crop_bbox",
        [crop_result_to_dict](py::object pymesh, const std::vector<double>& lo,
                              const std::vector<double>& hi, const std::string& mode,
                              bool record_ids) {
            if (lo.size() != 3 || hi.size() != 3)
                throw std::invalid_argument("crop_bbox: lo/hi must have 3 elements");
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::CropMode m =
                (mode == "any") ? meshioplusplus::CropMode::Any : meshioplusplus::CropMode::All;
            return crop_result_to_dict(
                meshioplusplus::crop_bbox(cpp, lo.data(), hi.data(), m, record_ids));
        },
        py::arg("mesh"), py::arg("lo"), py::arg("hi"), py::arg("mode") = "all",
        py::arg("record_ids") = false);
    m.def(
        "crop_plane",
        [crop_result_to_dict](py::object pymesh, const std::vector<double>& point,
                              const std::vector<double>& normal, const std::string& mode,
                              bool record_ids) {
            if (point.size() != 3 || normal.size() != 3)
                throw std::invalid_argument("crop_plane: point/normal must have 3 elements");
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::CropMode m =
                (mode == "any") ? meshioplusplus::CropMode::Any : meshioplusplus::CropMode::All;
            return crop_result_to_dict(
                meshioplusplus::crop_halfspace(cpp, point.data(), normal.data(), m, record_ids));
        },
        py::arg("mesh"), py::arg("point"), py::arg("normal"), py::arg("mode") = "all",
        py::arg("record_ids") = false);

    // The predicate crop: keep cells whose scalar cell_data satisfies a
    // comparison. No `mode` -- a cell_data value is already one per cell, so
    // there is nothing for CropMode's all/any to reduce. See operations/crop.hpp.
    m.def(
        "crop_predicate",
        [crop_result_to_dict](py::object pymesh, const std::string& array,
                              const std::string& compare, double value, bool record_ids) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            return crop_result_to_dict(meshioplusplus::crop_predicate(
                cpp, array, meshioplusplus::refine_compare_from_name(compare), value, record_ids));
        },
        py::arg("mesh"), py::arg("array"), py::arg("compare") = "<", py::arg("value") = 0.0,
        py::arg("record_ids") = false);

    // Split into pieces (by type / component / integer cell_data tag). Returns a
    // list of dicts {key, mesh, point_map, cell_maps}. See operations/split.hpp.
    m.def(
        "split",
        [](py::object pymesh, const std::string& by, const std::string& tag_name) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::SplitResult r =
                meshioplusplus::split(cpp, meshioplusplus::split_by_from_name(by), tag_name);
            py::list pieces;
            for (meshioplusplus::SplitPiece& p : r.mPieces) {
                py::dict d;
                d["key"] = p.mKey;
                d["mesh"] = meshioplusplus_py::mesh_to_py(std::move(p.mMesh));
                d["point_map"] = meshioplusplus_py::numpy_from_ndarray(std::move(p.mPointMap));
                py::list cell_maps;
                for (meshioplusplus::NDArray& a : p.mCellMaps)
                    cell_maps.append(meshioplusplus_py::numpy_from_ndarray(std::move(a)));
                d["cell_maps"] = cell_maps;
                pieces.append(d);
            }
            return pieces;
        },
        py::arg("mesh"), py::arg("by"), py::arg("tag_name") = "");

    // Convert the element representation (linearize / simplexify / elevate).
    // Returns a dict {mesh, point_map, cell_maps}. See operations/convert_cells.hpp.
    m.def(
        "convert_cells",
        [](py::object pymesh, const std::string& mode, bool record_parent_ids) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::ConvertCellsOptions options;
            options.mMode = meshioplusplus::convert_cells_mode_from_name(mode);
            options.mRecordParentIds = record_parent_ids;
            meshioplusplus::ConvertCellsResult r = meshioplusplus::convert_cells(cpp, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["point_map"] = meshioplusplus_py::numpy_from_ndarray(std::move(r.mPointMap));
            py::list cell_maps;
            for (meshioplusplus::NDArray& a : r.mCellMaps)
                cell_maps.append(meshioplusplus_py::numpy_from_ndarray(std::move(a)));
            out["cell_maps"] = cell_maps;
            return out;
        },
        py::arg("mesh"), py::arg("mode") = "linearize", py::arg("record_parent_ids") = false);

    // Polyhedral coarsening: merge groups of cells into single larger
    // polyhedral cells. Returns a dict {mesh, cell_map} -- a single FLAT
    // array (unlike subdivide's per-block cell_maps), since an output cell's
    // index is a function of which group it joined, not which input block it
    // came from. See operations/agglomerate.hpp.
    m.def(
        "agglomerate",
        [](py::object pymesh, std::size_t target_group_size) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::AgglomerateOptions options;
            options.mTargetGroupSize = target_group_size;
            meshioplusplus::AgglomerateResult r = meshioplusplus::agglomerate(cpp, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["cell_map"] = meshioplusplus_py::numpy_from_ndarray(std::move(r.mCellMap));
            return out;
        },
        py::arg("mesh"), py::arg("target_group_size") = 8);

    // Polyhedral refinement: one polyhedral child per face, connected to a new
    // interior point. Returns a dict {mesh, cell_maps} -- no point_map, since
    // this never prunes or renumbers an original point. See
    // operations/subdivide.hpp.
    m.def(
        "subdivide",
        [](py::object pymesh, bool record_parent_ids) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::SubdivideOptions options;
            options.mRecordParentIds = record_parent_ids;
            meshioplusplus::SubdivideResult r = meshioplusplus::subdivide(cpp, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            py::list cell_maps;
            for (meshioplusplus::NDArray& a : r.mCellMaps)
                cell_maps.append(meshioplusplus_py::numpy_from_ndarray(std::move(a)));
            out["cell_maps"] = cell_maps;
            return out;
        },
        py::arg("mesh"), py::arg("record_parent_ids") = false);

    // Green-element undo: restore `fine`'s transitional (closure-only) cells
    // back to their original parent, read verbatim from `coarse` (never
    // reconstructed). Returns a dict {mesh, cell_maps, num_groups_undone,
    // num_cells_removed}. See operations/undo_green.hpp.
    m.def(
        "undo_green",
        [](py::object pycoarse, py::object pyfine) {
            meshioplusplus_py::PyMeshRefs refs_coarse;
            meshioplusplus_py::PyMeshRefs refs_fine;
            meshioplusplus::Mesh coarse = meshioplusplus_py::py_to_mesh(
                pycoarse, refs_coarse, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::Mesh fine = meshioplusplus_py::py_to_mesh(
                pyfine, refs_fine, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::UndoGreenResult r = meshioplusplus::undo_green(coarse, fine);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            py::list cell_maps;
            for (meshioplusplus::NDArray& a : r.mCellMaps)
                cell_maps.append(meshioplusplus_py::numpy_from_ndarray(std::move(a)));
            out["cell_maps"] = cell_maps;
            out["num_groups_undone"] = r.mNumGroupsUndone;
            out["num_cells_removed"] = r.mNumCellsRemoved;
            return out;
        },
        py::arg("coarse"), py::arg("fine"));

    // Refinement: subdivide every cell (uniform) or a selected subset with a
    // conforming closure (selective) into same-type children. Returns a dict
    // {mesh, point_map, cell_maps}. See operations/refine.hpp.
    m.def(
        "refine",
        [](py::object pymesh, int levels, bool record_parent_ids, py::object cells,
           const std::string& region, const std::string& predicate_array,
           const std::string& predicate_op, double predicate_value, const std::string& closure,
           bool record_levels, bool record_hierarchy) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::RefineOptions options;
            options.mLevels = levels;
            options.mRecordParentIds = record_parent_ids;
            if (!cells.is_none()) {
                auto ids = cells.cast<
                    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>>();
                options.mCells.assign(ids.data(), ids.data() + ids.size());
            }
            options.mRegion = region;
            options.mPredicateArray = predicate_array;
            options.mPredicateOp = meshioplusplus::refine_compare_from_name(predicate_op);
            options.mPredicateValue = predicate_value;
            options.mClosure = meshioplusplus::refine_closure_from_name(closure);
            options.mRecordLevels = record_levels;
            options.mRecordHierarchy = record_hierarchy;
            meshioplusplus::RefineResult r = meshioplusplus::refine(cpp, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["point_map"] = meshioplusplus_py::numpy_from_ndarray(std::move(r.mPointMap));
            py::list cell_maps;
            for (meshioplusplus::NDArray& a : r.mCellMaps)
                cell_maps.append(meshioplusplus_py::numpy_from_ndarray(std::move(a)));
            out["cell_maps"] = cell_maps;
            return out;
        },
        py::arg("mesh"), py::arg("levels") = 1, py::arg("record_parent_ids") = false,
        py::arg("cells") = py::none(), py::arg("region") = "", py::arg("predicate_array") = "",
        py::arg("predicate_op") = "<", py::arg("predicate_value") = 0.0,
        py::arg("closure") = "redgreen", py::arg("record_levels") = false,
        py::arg("record_hierarchy") = false);

    // The selective-refinement subdivision table for one cell type, as
    // {mask: [[child node ids], ...]} plus the tie-break metadata. Exported so
    // the numpy reference implementation is pinned against the C++ tables
    // rather than merely transcribed from them -- the colormap_table precedent.
    m.def(
        "refine_mask_table",
        [](const std::string& type_name) {
            const meshioplusplus::CellType type = meshioplusplus::cell_type_from_name(type_name);
            py::dict out;
            const std::size_t nedges = meshioplusplus::detail::cell_refine_edges(type).size();
            for (std::uint32_t mask = 0; mask < (1u << nedges); ++mask) {
                const meshioplusplus::detail::RefineMaskTemplate& tpl =
                    meshioplusplus::detail::refine_mask_template(type,
                                                                 static_cast<std::uint16_t>(mask));
                if (!tpl.IsAdmissible())
                    continue;
                py::dict entry;
                const auto rows = [](const std::vector<std::vector<std::uint8_t>>& children) {
                    py::list out_rows;
                    for (const std::vector<std::uint8_t>& child : children) {
                        py::list row;
                        for (std::uint8_t n : child)
                            row.append(static_cast<int>(n));
                        out_rows.append(row);
                    }
                    return out_rows;
                };
                entry["children"] = rows(tpl.mChildren);
                entry["children_alt"] = rows(tpl.mChildrenAlt);
                entry["tie_a"] = static_cast<int>(tpl.mTieA);
                entry["tie_b"] = static_cast<int>(tpl.mTieB);
                out[py::int_(mask)] = entry;
            }
            return out;
        },
        py::arg("cell_type"));

    // The smallest admissible split-edge mask containing `mask` -- the closure
    // operator selective refinement iterates. See detail/refine_templates.hpp.
    m.def(
        "refine_promote_mask",
        [](const std::string& type_name, int mask, bool propagate) {
            return static_cast<int>(meshioplusplus::detail::refine_promote_mask(
                meshioplusplus::cell_type_from_name(type_name), static_cast<std::uint16_t>(mask),
                propagate));
        },
        py::arg("cell_type"), py::arg("mask"), py::arg("propagate") = false);

    // Quadric-error-metric surface decimation. Returns a dict {mesh, point_map,
    // cell_maps, faces_removed, points_removed, collapses_rejected,
    // max_error_applied}. See operations/decimate.hpp.
    m.def(
        "decimate",
        [](py::object pymesh, double target_ratio, std::int64_t target_faces, double max_error,
           const std::string& placement, bool preserve_boundary, bool preserve_features,
           double feature_angle, py::object frozen) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::DecimateOptions options;
            options.mTargetRatio = target_ratio;
            options.mTargetFaces = target_faces;
            options.mMaxError = max_error;
            options.mPlacement = meshioplusplus::decimate_placement_from_name(placement);
            options.mPreserveBoundary = preserve_boundary;
            options.mPreserveFeatures = preserve_features;
            options.mFeatureAngleDeg = feature_angle;
            if (!frozen.is_none()) {
                py::array_t<std::int64_t> ids =
                    py::cast<py::array_t<std::int64_t>>(py::array::ensure(frozen));
                options.mFrozen.assign(cpp.NumPoints(), 0);
                auto v = ids.unchecked<1>();
                for (py::ssize_t k = 0; k < v.shape(0); ++k) {
                    const std::int64_t id = v(k);
                    if (id < 0 || static_cast<std::size_t>(id) >= cpp.NumPoints())
                        throw std::invalid_argument("meshio++: decimate: frozen node id " +
                                                    std::to_string(id) + " is out of range");
                    options.mFrozen[static_cast<std::size_t>(id)] = 1;
                }
            }
            meshioplusplus::DecimateResult r = meshioplusplus::decimate(cpp, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["point_map"] = meshioplusplus_py::numpy_from_ndarray(std::move(r.mPointMap));
            py::list cell_maps;
            for (meshioplusplus::NDArray& a : r.mCellMaps)
                cell_maps.append(meshioplusplus_py::numpy_from_ndarray(std::move(a)));
            out["cell_maps"] = cell_maps;
            out["faces_removed"] = r.mFacesRemoved;
            out["points_removed"] = r.mPointsRemoved;
            out["collapses_rejected"] = r.mCollapsesRejected;
            out["max_error_applied"] = r.mMaxErrorApplied;
            return out;
        },
        py::arg("mesh"), py::arg("target_ratio") = -1.0, py::arg("target_faces") = -1,
        py::arg("max_error") = -1.0, py::arg("placement") = "optimal",
        py::arg("preserve_boundary") = true, py::arg("preserve_features") = true,
        py::arg("feature_angle") = 30.0, py::arg("frozen") = py::none());

    // Quadric-error-metric tet-edge collapse volume decimation. Returns a dict
    // {mesh, point_map, cell_maps, tets_removed, points_removed,
    // collapses_rejected, max_error_applied}. See operations/decimate_volume.hpp.
    // A separate operation from `decimate` -- boundary vertices participate by
    // default (`preserve_boundary` defaults False here, unlike `decimate`'s True).
    m.def(
        "decimate_volume",
        [](py::object pymesh, double target_ratio, std::int64_t target_cells, double max_error,
           const std::string& placement, bool preserve_boundary, bool preserve_features,
           double feature_angle, py::object frozen) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::DecimateVolumeOptions options;
            options.mTargetRatio = target_ratio;
            options.mTargetCells = target_cells;
            options.mMaxError = max_error;
            options.mPlacement = meshioplusplus::decimate_placement_from_name(placement);
            options.mPreserveBoundary = preserve_boundary;
            options.mPreserveFeatures = preserve_features;
            options.mFeatureAngleDeg = feature_angle;
            if (!frozen.is_none()) {
                py::array_t<std::int64_t> ids =
                    py::cast<py::array_t<std::int64_t>>(py::array::ensure(frozen));
                options.mFrozen.assign(cpp.NumPoints(), 0);
                auto v = ids.unchecked<1>();
                for (py::ssize_t k = 0; k < v.shape(0); ++k) {
                    const std::int64_t id = v(k);
                    if (id < 0 || static_cast<std::size_t>(id) >= cpp.NumPoints())
                        throw std::invalid_argument("meshio++: decimate_volume: frozen node id " +
                                                    std::to_string(id) + " is out of range");
                    options.mFrozen[static_cast<std::size_t>(id)] = 1;
                }
            }
            meshioplusplus::DecimateVolumeResult r = meshioplusplus::decimate_volume(cpp, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["point_map"] = meshioplusplus_py::numpy_from_ndarray(std::move(r.mPointMap));
            py::list cell_maps;
            for (meshioplusplus::NDArray& a : r.mCellMaps)
                cell_maps.append(meshioplusplus_py::numpy_from_ndarray(std::move(a)));
            out["cell_maps"] = cell_maps;
            out["tets_removed"] = r.mTetsRemoved;
            out["points_removed"] = r.mPointsRemoved;
            out["collapses_rejected"] = r.mCollapsesRejected;
            out["max_error_applied"] = r.mMaxErrorApplied;
            return out;
        },
        py::arg("mesh"), py::arg("target_ratio") = -1.0, py::arg("target_cells") = -1,
        py::arg("max_error") = -1.0, py::arg("placement") = "optimal",
        py::arg("preserve_boundary") = false, py::arg("preserve_features") = true,
        py::arg("feature_angle") = 30.0, py::arg("frozen") = py::none());

    // Laplacian / Taubin smoothing. Returns a dict with the smoothed mesh and
    // the run summary. A coordinates-only change. See operations/smooth.hpp.
    m.def(
        "smooth",
        [](py::object pymesh, const std::string& method, int iterations, double lambda, double mu,
           bool fix_boundary, bool preserve_features, double feature_angle, bool guard_inversion,
           py::object frozen) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::SmoothOptions options;
            options.mMethod = meshioplusplus::smooth_method_from_name(method);
            options.mIterations = iterations;
            options.mLambda = lambda;
            options.mMu = mu;
            options.mFixBoundary = fix_boundary;
            options.mPreserveFeatures = preserve_features;
            options.mFeatureAngleDeg = feature_angle;
            options.mGuardInversion = guard_inversion;
            if (!frozen.is_none()) {
                py::array_t<std::int64_t> ids =
                    py::cast<py::array_t<std::int64_t>>(py::array::ensure(frozen));
                options.mFrozen.assign(cpp.NumPoints(), 0);
                auto v = ids.unchecked<1>();
                for (py::ssize_t k = 0; k < v.shape(0); ++k) {
                    const std::int64_t id = v(k);
                    if (id < 0 || static_cast<std::size_t>(id) >= cpp.NumPoints())
                        throw std::invalid_argument("meshio++: smooth: frozen node id " +
                                                    std::to_string(id) + " is out of range");
                    options.mFrozen[static_cast<std::size_t>(id)] = 1;
                }
            }
            meshioplusplus::SmoothResult r = meshioplusplus::smooth(cpp, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["num_nodes_moved"] = r.mNumNodesMoved;
            out["max_displacement"] = r.mMaxDisplacement;
            out["num_skipped_inversion"] = r.mNumSkippedInversion;
            return out;
        },
        py::arg("mesh"), py::arg("method") = "taubin", py::arg("iterations") = 10,
        py::arg("lambda_") = -1.0, py::arg("mu") = -0.34, py::arg("fix_boundary") = true,
        py::arg("preserve_features") = true, py::arg("feature_angle") = 30.0,
        py::arg("guard_inversion") = true, py::arg("frozen") = py::none());

    // Cross-mesh field transfer: sample the source's data arrays onto the
    // target. Returns the target copy as a plain mesh (no maps/counters).
    // See operations/interpolate.hpp.
    m.def(
        "interpolate",
        [](py::object pysource, py::object pytarget, const std::string& method, py::object arrays,
           bool extrapolate, double default_value, const std::string& on_conflict) {
            meshioplusplus_py::PyMeshRefs refs_src;
            meshioplusplus_py::PyMeshRefs refs_tgt;
            meshioplusplus::Mesh src = meshioplusplus_py::py_to_mesh(
                pysource, refs_src, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::Mesh tgt = meshioplusplus_py::py_to_mesh(
                pytarget, refs_tgt, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::InterpolateOptions options;
            options.mMethod = meshioplusplus::interpolate_method_from_name(method);
            if (!arrays.is_none())
                options.mArrays = py::cast<std::vector<std::string>>(arrays);
            options.mExtrapolate = extrapolate;
            options.mDefaultValue = default_value;
            options.mOnConflict = meshioplusplus::interpolate_conflict_from_name(on_conflict);
            meshioplusplus::Mesh out = meshioplusplus::interpolate(src, tgt, options);
            return meshioplusplus_py::mesh_to_py(std::move(out));
        },
        py::arg("source"), py::arg("target"), py::arg("method") = "nearest",
        py::arg("arrays") = py::none(), py::arg("extrapolate") = false,
        py::arg("default_value") = 0.0, py::arg("on_conflict") = "error");

    // Planar cross-section (marching tetrahedra on a simplexified input).
    // Returns a plain new mesh one dimension below the cut cells. See
    // operations/slice.hpp.
    m.def(
        "slice",
        [](py::object pymesh, const std::vector<double>& origin, const std::vector<double>& normal,
           bool record_parent_ids) {
            if (origin.size() != 3 || normal.size() != 3)
                throw std::invalid_argument("slice: origin and normal must each have 3 elements");
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::SliceOptions options;
            options.mOrigin = {origin[0], origin[1], origin[2]};
            options.mNormal = {normal[0], normal[1], normal[2]};
            options.mRecordParentIds = record_parent_ids;
            meshioplusplus::Mesh out = meshioplusplus::slice(cpp, options);
            return meshioplusplus_py::mesh_to_py(std::move(out));
        },
        py::arg("mesh"), py::arg("origin"), py::arg("normal"),
        py::arg("record_parent_ids") = false);

    // Mass-preserving cross-mesh field transfer: exact overlap-measure
    // weighted remap (as opposed to interpolate's pointwise sampling).
    // Returns the target copy as a plain mesh (no maps/counters). See
    // operations/conservative_interpolate.hpp.
    m.def(
        "conservative_interpolate",
        [](py::object pysource, py::object pytarget, py::object arrays, double default_value,
           const std::string& on_conflict) {
            meshioplusplus_py::PyMeshRefs refs_src;
            meshioplusplus_py::PyMeshRefs refs_tgt;
            meshioplusplus::Mesh src = meshioplusplus_py::py_to_mesh(
                pysource, refs_src, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::Mesh tgt = meshioplusplus_py::py_to_mesh(
                pytarget, refs_tgt, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::ConservativeInterpolateOptions options;
            if (!arrays.is_none())
                options.mArrays = py::cast<std::vector<std::string>>(arrays);
            options.mDefaultValue = default_value;
            options.mOnConflict =
                meshioplusplus::conservative_interpolate_conflict_from_name(on_conflict);
            meshioplusplus::Mesh out = meshioplusplus::conservative_interpolate(src, tgt, options);
            return meshioplusplus_py::mesh_to_py(std::move(out));
        },
        py::arg("source"), py::arg("target"), py::arg("arrays") = py::none(),
        py::arg("default_value") = 0.0, py::arg("on_conflict") = "error");

    // Isosurfaces / contours: the level set of a scalar point_data field, cut
    // with the same marching tetrahedra as slice. `component` is negative for
    // the row magnitude. See operations/isosurface.hpp.
    m.def(
        "isosurface",
        [](py::object pymesh, const std::string& array, const std::vector<double>& isovalues,
           int component, bool record_parent_ids) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::IsosurfaceOptions options;
            options.mArrayName = array;
            options.mIsovalues = isovalues;
            if (component >= 0)
                options.mComponent = component;
            options.mRecordParentIds = record_parent_ids;
            meshioplusplus::Mesh out = meshioplusplus::isosurface(cpp, options);
            return meshioplusplus_py::mesh_to_py(std::move(out));
        },
        py::arg("mesh"), py::arg("array"), py::arg("isovalues"), py::arg("component") = -1,
        py::arg("record_parent_ids") = false);

    // Regular grids. `grid` builds a lattice from nothing -- the library's first
    // mesh *generator* -- and `voxelize` builds one around a mesh, optionally
    // keeping only the cells its surface passes through or encloses.
    m.def(
        "grid",
        [](const std::array<std::int64_t, 3>& dims, const std::array<double, 3>& origin,
           const std::array<double, 3>& spacing, std::int64_t max_cells) {
            return meshioplusplus_py::mesh_to_py(
                meshioplusplus::grid(dims, origin, spacing, max_cells));
        },
        py::arg("dims"), py::arg("origin") = std::array<double, 3>{{0.0, 0.0, 0.0}},
        py::arg("spacing") = std::array<double, 3>{{1.0, 1.0, 1.0}},
        py::arg("max_cells") = 20000000);

    m.def(
        "voxelize",
        [](py::object pymesh, py::object resolution, py::object cell_size, py::object bounds,
           double padding, double padding_relative, const std::string& fill, bool attach_occupancy,
           std::int64_t max_cells, const std::string& sign, const std::string& watertight_check) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::VoxelOptions options;
            if (!resolution.is_none())
                options.mResolution = resolution.cast<std::array<std::int64_t, 3>>();
            if (!cell_size.is_none())
                options.mCellSize = cell_size.cast<double>();
            if (!bounds.is_none())
                options.mBounds = bounds.cast<std::array<double, 6>>();
            options.mPadding = padding;
            options.mPaddingRelative = padding_relative;
            options.mFill = meshioplusplus::voxel_fill_from_name(fill);
            options.mAttachOccupancy = attach_occupancy;
            options.mMaxCells = max_cells;
            options.mDistance.mSign = meshioplusplus::sdf_sign_from_name(sign);
            options.mDistance.mWatertightCheck =
                meshioplusplus::sdf_watertight_check_from_name(watertight_check);
            meshioplusplus::VoxelResult r = meshioplusplus::voxelize(cpp, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["dims"] = r.mDims;
            out["origin"] = r.mOrigin;
            out["spacing"] = r.mSpacing;
            out["num_occupied"] = r.mNumOccupied;
            return out;
        },
        py::arg("mesh"), py::arg("resolution") = py::none(), py::arg("cell_size") = py::none(),
        py::arg("bounds") = py::none(), py::arg("padding") = 0.0, py::arg("padding_relative") = 0.0,
        py::arg("fill") = "all", py::arg("attach_occupancy") = false,
        py::arg("max_cells") = 20000000, py::arg("sign") = "pseudonormal",
        py::arg("watertight_check") = "warn");

    // Distance to a surface. `sample_distance` takes bare query points;
    // `distance_to_surface` attaches the result to a mesh as ordinary data.
    m.def(
        "sample_distance",
        [](py::object pysurface, py::array points, const std::string& sign,
           const std::string& weight, double band, const std::string& watertight_check,
           const std::string& surface_region, double grid_cell_size, double max_winding_work) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh surface = meshioplusplus_py::py_to_mesh(
                pysurface, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::SurfaceDistanceOptions options;
            options.mSign = meshioplusplus::sdf_sign_from_name(sign);
            options.mWeight = meshioplusplus::sdf_weight_from_name(weight);
            options.mBand = band;
            options.mWatertightCheck =
                meshioplusplus::sdf_watertight_check_from_name(watertight_check);
            options.mSurfaceRegion = surface_region;
            options.mGridCellSize = grid_cell_size;
            options.mMaxWindingWork = max_winding_work;
            meshioplusplus_py::PyMeshRefs qrefs;
            py::array contiguous = meshioplusplus_py::ensure_contiguous(points, qrefs);
            meshioplusplus::NDArray query = meshioplusplus_py::view_from_numpy(contiguous);
            return meshioplusplus_py::numpy_from_ndarray(
                meshioplusplus::sample_distance(surface, query, options));
        },
        py::arg("surface"), py::arg("points"), py::arg("sign") = "pseudonormal",
        py::arg("weight") = "angle", py::arg("band") = 0.0, py::arg("watertight_check") = "warn",
        py::arg("surface_region") = "", py::arg("grid_cell_size") = 0.0,
        py::arg("max_winding_work") = 2.0e9);

    m.def(
        "distance_to_surface",
        [](py::object pyquery, py::object pysurface, const std::string& sign,
           const std::string& weight, const std::string& location, double band,
           bool record_closest_cell, bool record_inside, const std::string& watertight_check,
           const std::string& surface_region, double grid_cell_size, double max_winding_work) {
            meshioplusplus_py::PyMeshRefs qrefs;
            meshioplusplus::Mesh query = meshioplusplus_py::py_to_mesh(
                pyquery, qrefs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus_py::PyMeshRefs srefs;
            meshioplusplus::Mesh surface = meshioplusplus_py::py_to_mesh(
                pysurface, srefs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::SurfaceDistanceOptions options;
            options.mSign = meshioplusplus::sdf_sign_from_name(sign);
            options.mWeight = meshioplusplus::sdf_weight_from_name(weight);
            options.mLocation = meshioplusplus::sdf_location_from_name(location);
            options.mBand = band;
            options.mRecordClosestCell = record_closest_cell;
            options.mRecordInside = record_inside;
            options.mWatertightCheck =
                meshioplusplus::sdf_watertight_check_from_name(watertight_check);
            options.mSurfaceRegion = surface_region;
            options.mGridCellSize = grid_cell_size;
            options.mMaxWindingWork = max_winding_work;
            meshioplusplus::SurfaceDistanceResult r =
                meshioplusplus::distance_to_surface(query, surface, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["num_banded"] = r.mNumBanded;
            py::dict q;
            q["boundary_edges"] = r.mQuality.mBoundaryEdges;
            q["non_manifold_edges"] = r.mQuality.mNonManifoldEdges;
            q["inconsistent_pairs"] = r.mQuality.mInconsistentPairs;
            q["degenerate_triangles"] = r.mQuality.mDegenerateTriangles;
            q["watertight"] = r.mQuality.mWatertight;
            out["quality"] = q;
            return out;
        },
        py::arg("query"), py::arg("surface"), py::arg("sign") = "pseudonormal",
        py::arg("weight") = "angle", py::arg("location") = "corner", py::arg("band") = 0.0,
        py::arg("record_closest_cell") = false, py::arg("record_inside") = false,
        py::arg("watertight_check") = "warn", py::arg("surface_region") = "",
        py::arg("grid_cell_size") = 0.0, py::arg("max_winding_work") = 2.0e9);

    m.def(
        "surface_watertight_check",
        [](py::object pysurface) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh surface = meshioplusplus_py::py_to_mesh(
                pysurface, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            const meshioplusplus::SurfaceQuality q =
                meshioplusplus::surface_watertight_check(surface);
            py::dict out;
            out["boundary_edges"] = q.mBoundaryEdges;
            out["non_manifold_edges"] = q.mNonManifoldEdges;
            out["inconsistent_pairs"] = q.mInconsistentPairs;
            out["degenerate_triangles"] = q.mDegenerateTriangles;
            out["watertight"] = q.mWatertight;
            return out;
        },
        py::arg("surface"));

    // The umbrella: generate a grid over the surface and fill it with signed
    // distances. `structure` picks a dense lattice or an adaptive octree; the
    // octree's finest cell is root/2^depth, so resolution/cell_size are a
    // voxel-only pair and an error with 'octree'. See operations/sdf.hpp.
    m.def(
        "compute_sdf",
        [](py::object pysurface, const std::string& structure, py::object resolution,
           py::object cell_size, py::object bounds, double padding, double padding_relative,
           std::int64_t root_resolution, std::int64_t max_depth, double band_cells,
           bool record_levels, std::int64_t max_cells, const std::string& sign,
           const std::string& weight, const std::string& location, double band,
           bool record_closest_cell, bool record_inside, const std::string& watertight_check,
           const std::string& surface_region, double grid_cell_size, double max_winding_work) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh surface = meshioplusplus_py::py_to_mesh(
                pysurface, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::SdfOptions options;
            options.mStructure = meshioplusplus::sdf_structure_from_name(structure);
            if (!resolution.is_none())
                options.mResolution = resolution.cast<std::array<std::int64_t, 3>>();
            if (!cell_size.is_none())
                options.mCellSize = cell_size.cast<double>();
            if (!bounds.is_none())
                options.mBounds = bounds.cast<std::array<double, 6>>();
            options.mPadding = padding;
            options.mPaddingRelative = padding_relative;
            options.mRootResolution = root_resolution;
            options.mMaxDepth = max_depth;
            options.mBandCells = band_cells;
            options.mRecordLevels = record_levels;
            options.mMaxCells = max_cells;
            options.mDistance.mSign = meshioplusplus::sdf_sign_from_name(sign);
            options.mDistance.mWeight = meshioplusplus::sdf_weight_from_name(weight);
            options.mDistance.mLocation = meshioplusplus::sdf_location_from_name(location);
            options.mDistance.mBand = band;
            options.mDistance.mRecordClosestCell = record_closest_cell;
            options.mDistance.mRecordInside = record_inside;
            options.mDistance.mWatertightCheck =
                meshioplusplus::sdf_watertight_check_from_name(watertight_check);
            options.mDistance.mSurfaceRegion = surface_region;
            options.mDistance.mGridCellSize = grid_cell_size;
            options.mDistance.mMaxWindingWork = max_winding_work;
            meshioplusplus::SdfResult r = meshioplusplus::compute_sdf(surface, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["dims"] = r.mDims;
            out["origin"] = r.mOrigin;
            out["spacing"] = r.mSpacing;
            out["max_depth"] = r.mMaxDepth;
            out["num_banded"] = r.mNumBanded;
            py::dict q;
            q["boundary_edges"] = r.mQuality.mBoundaryEdges;
            q["non_manifold_edges"] = r.mQuality.mNonManifoldEdges;
            q["inconsistent_pairs"] = r.mQuality.mInconsistentPairs;
            q["degenerate_triangles"] = r.mQuality.mDegenerateTriangles;
            q["watertight"] = r.mQuality.mWatertight;
            out["quality"] = q;
            return out;
        },
        py::arg("surface"), py::arg("structure") = "voxel", py::arg("resolution") = py::none(),
        py::arg("cell_size") = py::none(), py::arg("bounds") = py::none(), py::arg("padding") = 0.0,
        py::arg("padding_relative") = 0.1, py::arg("root_resolution") = 8, py::arg("max_depth") = 4,
        py::arg("band_cells") = 1.0, py::arg("record_levels") = true,
        py::arg("max_cells") = 20000000, py::arg("sign") = "pseudonormal",
        py::arg("weight") = "angle", py::arg("location") = "corner", py::arg("band") = 0.0,
        py::arg("record_closest_cell") = false, py::arg("record_inside") = false,
        py::arg("watertight_check") = "warn", py::arg("surface_region") = "",
        py::arg("grid_cell_size") = 0.0, py::arg("max_winding_work") = 2.0e9);

    // Volumetric retetrahedralization by isosurface stuffing -- the
    // volumetric sibling of `remesh`, generating an entirely new tet mesh
    // (no point/cell maps, point_data/cell_data/regions dropped) rather than
    // working on the input's own cells. Same lattice-sizing and distance
    // kwargs as `compute_sdf`, since it is a lattice generator too. See
    // operations/remesh_volume.hpp and doc/remesh_volume.md.
    m.def(
        "remesh_volume",
        [](py::object pymesh, py::object resolution, py::object cell_size, py::object bounds,
           double padding, double padding_relative, std::int64_t max_cells, std::int64_t max_tets,
           double warp_fraction, const std::string& sign, const std::string& weight,
           const std::string& watertight_check, const std::string& surface_region,
           double grid_cell_size, double max_winding_work) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::RemeshVolumeOptions options;
            if (!resolution.is_none())
                options.mResolution = resolution.cast<std::array<std::int64_t, 3>>();
            if (!cell_size.is_none())
                options.mCellSize = cell_size.cast<double>();
            if (!bounds.is_none())
                options.mBounds = bounds.cast<std::array<double, 6>>();
            options.mPadding = padding;
            options.mPaddingRelative = padding_relative;
            options.mMaxCells = max_cells;
            options.mMaxTets = max_tets;
            options.mWarpFraction = warp_fraction;
            options.mDistance.mSign = meshioplusplus::sdf_sign_from_name(sign);
            options.mDistance.mWeight = meshioplusplus::sdf_weight_from_name(weight);
            options.mDistance.mWatertightCheck =
                meshioplusplus::sdf_watertight_check_from_name(watertight_check);
            options.mDistance.mSurfaceRegion = surface_region;
            options.mDistance.mGridCellSize = grid_cell_size;
            options.mDistance.mMaxWindingWork = max_winding_work;
            meshioplusplus::RemeshVolumeResult r = meshioplusplus::remesh_volume(cpp, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            py::dict q;
            q["boundary_edges"] = r.mQuality.mBoundaryEdges;
            q["non_manifold_edges"] = r.mQuality.mNonManifoldEdges;
            q["inconsistent_pairs"] = r.mQuality.mInconsistentPairs;
            q["degenerate_triangles"] = r.mQuality.mDegenerateTriangles;
            q["watertight"] = r.mQuality.mWatertight;
            out["input_quality"] = q;
            out["num_tets"] = r.mNumTets;
            out["num_vertices_warped"] = r.mNumVerticesWarped;
            out["num_tets_rejected"] = r.mNumTetsRejected;
            out["num_non_manifold_edges"] = r.mNumNonManifoldEdges;
            return out;
        },
        py::arg("mesh"), py::arg("resolution") = py::none(), py::arg("cell_size") = py::none(),
        py::arg("bounds") = py::none(), py::arg("padding") = 0.0, py::arg("padding_relative") = 0.1,
        py::arg("max_cells") = 20000000, py::arg("max_tets") = 20000000,
        py::arg("warp_fraction") = 0.35, py::arg("sign") = "pseudonormal",
        py::arg("weight") = "angle", py::arg("watertight_check") = "warn",
        py::arg("surface_region") = "", py::arg("grid_cell_size") = 0.0,
        py::arg("max_winding_work") = 2.0e9);

    // ODT remeshing: raise a tet mesh's worst element quality by relocating
    // vertices AND flipping connectivity (2-3/3-2, predicate-free). The point
    // set is invariant, so point_data/field_data and Point regions carry;
    // cell_data + Cell/Side regions are dropped. See
    // operations/optimize_volume.hpp and doc/optimize_volume.md.
    m.def(
        "optimize_volume",
        [](py::object pymesh, int max_iterations, bool relocate, bool flip, bool preserve_boundary,
           double min_improvement, py::object frozen) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
            meshioplusplus::OptimizeVolumeOptions options;
            options.mMaxIterations = max_iterations;
            options.mRelocate = relocate;
            options.mFlip = flip;
            options.mPreserveBoundary = preserve_boundary;
            options.mMinImprovement = min_improvement;
            if (!frozen.is_none())
                options.mFrozen = frozen.cast<std::vector<std::uint8_t>>();
            meshioplusplus::OptimizeVolumeResult r = meshioplusplus::optimize_volume(cpp, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["num_flips"] = r.mNumFlips;
            out["num_23_flips"] = r.mNum23Flips;
            out["num_32_flips"] = r.mNum32Flips;
            out["num_vertices_moved"] = r.mNumVerticesMoved;
            out["num_tets"] = r.mNumTets;
            out["min_quality_before"] = r.mMinQualityBefore;
            out["min_quality_after"] = r.mMinQualityAfter;
            return out;
        },
        py::arg("mesh"), py::arg("max_iterations") = 10, py::arg("relocate") = true,
        py::arg("flip") = true, py::arg("preserve_boundary") = true,
        py::arg("min_improvement") = 1e-6, py::arg("frozen") = py::none());

    // Field differential operators: the gradient / divergence / curl of a
    // point_data field, by Green-Gauss or least squares. `component` is negative
    // for "every component" -- note this is the OPPOSITE of isosurface's
    // sentinel, where negative means the row magnitude. An empty `output` name
    // selects the operator's default suffix. See operations/gradient.hpp.
    m.def(
        "gradient",
        [](py::object pymesh, const std::string& array, const std::string& op,
           const std::string& method, const std::string& location, const std::string& output,
           int component, bool overwrite) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::GradientOptions options;
            options.mArrayName = array;
            options.mOperator = meshioplusplus::gradient_operator_from_name(op);
            options.mMethod = meshioplusplus::gradient_method_from_name(method);
            options.mLocation = meshioplusplus::data_location_from_name(location);
            options.mOutputName = output;
            if (component >= 0)
                options.mComponent = component;
            options.mOverwrite = overwrite;
            meshioplusplus::GradientResult r = meshioplusplus::gradient(cpp, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["num_skipped"] = r.mNumSkipped;
            out["num_fallback"] = r.mNumFallback;
            return out;
        },
        py::arg("mesh"), py::arg("array"), py::arg("operator_") = "gradient",
        py::arg("method") = "green-gauss", py::arg("location") = "cell", py::arg("output") = "",
        py::arg("component") = -1, py::arg("overwrite") = false);

    // The Hessian (second derivative) of a scalar point_data field, built
    // entirely as a composition of two `gradient` calls. See
    // operations/hessian.hpp for the exactness argument and the scalar-only
    // scope.
    // Per-vertex mean and Gaussian curvature of a surface. The result carries
    // `total_angle_defect` on purpose: for a closed surface it is `2*pi*chi`
    // whatever the tessellation and whichever dual area, so the Gauss-Bonnet
    // invariant is checkable from Python and not only from a gtest. See
    // operations/curvature.hpp.
    m.def(
        "compute_curvature",
        [](py::object pymesh, bool mean, bool gaussian, const std::string& dual_area,
           bool include_boundary, bool record_area, bool record_principal,
           const std::string& region) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::CurvatureOptions options;
            options.mMean = mean;
            options.mGaussian = gaussian;
            options.mDualArea = meshioplusplus::curvature_dual_area_from_name(dual_area);
            options.mIncludeBoundary = include_boundary;
            options.mRecordArea = record_area;
            options.mRecordPrincipal = record_principal;
            options.mRegion = region;
            meshioplusplus::CurvatureResult r = meshioplusplus::compute_curvature(cpp, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["num_boundary"] = r.mNumBoundary;
            out["num_isolated"] = r.mNumIsolated;
            out["num_degenerate"] = r.mNumDegenerate;
            out["total_angle_defect"] = r.mTotalAngleDefect;
            py::dict q;
            q["boundary_edges"] = r.mQuality.mBoundaryEdges;
            q["non_manifold_edges"] = r.mQuality.mNonManifoldEdges;
            q["inconsistent_pairs"] = r.mQuality.mInconsistentPairs;
            q["degenerate_triangles"] = r.mQuality.mDegenerateTriangles;
            q["watertight"] = r.mQuality.mWatertight;
            out["quality"] = q;
            return out;
        },
        py::arg("mesh"), py::arg("mean") = true, py::arg("gaussian") = true,
        py::arg("dual_area") = "mixed-voronoi", py::arg("include_boundary") = false,
        py::arg("record_area") = false, py::arg("record_principal") = false,
        py::arg("region") = "");

    // Surface repair: orientation, holes, bowties. {mesh, point_map, cell_maps,
    // counters, quality_before, quality_after}. See operations/repair.hpp.
    m.def(
        "repair",
        [](py::object pymesh, bool fix_orientation, bool orient_outward, bool fill_holes,
           bool split_non_manifold, std::int64_t max_hole_edges, double weld_tolerance,
           bool record_provenance) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::RepairOptions options;
            options.mFixOrientation = fix_orientation;
            options.mOrientOutward = orient_outward;
            options.mFillHoles = fill_holes;
            options.mSplitNonManifold = split_non_manifold;
            options.mMaxHoleEdges = max_hole_edges;
            options.mWeldTolerance = weld_tolerance;
            options.mRecordProvenance = record_provenance;
            meshioplusplus::RepairResult r = meshioplusplus::repair(cpp, options);
            auto quality = [](const meshioplusplus::SurfaceQuality& q) {
                py::dict d;
                d["boundary_edges"] = q.mBoundaryEdges;
                d["non_manifold_edges"] = q.mNonManifoldEdges;
                d["inconsistent_pairs"] = q.mInconsistentPairs;
                d["degenerate_triangles"] = q.mDegenerateTriangles;
                d["watertight"] = q.mWatertight;
                return d;
            };
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["point_map"] = meshioplusplus_py::numpy_from_ndarray(std::move(r.mPointMap));
            py::list cell_maps;
            for (auto& a : r.mCellMaps)
                cell_maps.append(meshioplusplus_py::numpy_from_ndarray(std::move(a)));
            out["cell_maps"] = cell_maps;
            out["quality_before"] = quality(r.mQualityBefore);
            out["quality_after"] = quality(r.mQualityAfter);
            out["num_flipped"] = r.mNumFlipped;
            out["num_components"] = r.mNumComponents;
            out["largest_component"] = r.mLargestComponent;
            out["num_oriented_outward"] = r.mNumOrientedOutward;
            out["num_unorientable"] = r.mNumUnorientable;
            out["num_vertices_split"] = r.mNumVerticesSplit;
            out["num_holes_detected"] = r.mNumHolesDetected;
            out["num_holes_filled"] = r.mNumHolesFilled;
            out["num_holes_skipped"] = r.mNumHolesSkipped;
            out["num_faces_added"] = r.mNumFacesAdded;
            out["num_points_added"] = r.mNumPointsAdded;
            out["points_welded"] = r.mPointsWelded;
            return out;
        },
        py::arg("mesh"), py::arg("fix_orientation") = true, py::arg("orient_outward") = true,
        py::arg("fill_holes") = true, py::arg("split_non_manifold") = true,
        py::arg("max_hole_edges") = 10, py::arg("weld_tolerance") = 0.0,
        py::arg("record_provenance") = false);

    // Shrinkwrap: one projection of the source's points onto the target
    // surface. {mesh, quality (of the target), counters}. See
    // operations/shrinkwrap.hpp.
    m.def(
        "shrinkwrap",
        [](py::object pymesh, py::object pytarget, double offset, double max_distance,
           const std::string& weights, const std::string& target_region,
           const std::string& normal_weight, bool record_distance, bool record_closest_cell,
           double grid_cell_size) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus_py::PyMeshRefs trefs;
            meshioplusplus::Mesh target = meshioplusplus_py::py_to_mesh(
                pytarget, trefs, /*lenient_field_data=*/true, /*allow_ragged=*/true);
            meshioplusplus::ShrinkwrapOptions options;
            options.mOffset = offset;
            options.mMaxDistance = max_distance;
            options.mWeights = weights;
            options.mTargetRegion = target_region;
            options.mNormalWeight = meshioplusplus::sdf_weight_from_name(normal_weight);
            options.mRecordDistance = record_distance;
            options.mRecordClosestCell = record_closest_cell;
            options.mGridCellSize = grid_cell_size;
            meshioplusplus::ShrinkwrapResult r = meshioplusplus::shrinkwrap(cpp, target, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["num_projected"] = r.mNumProjected;
            out["num_missed"] = r.mNumMissed;
            out["num_skipped"] = r.mNumSkipped;
            out["max_displacement"] = r.mMaxDisplacement;
            py::dict q;
            q["boundary_edges"] = r.mQuality.mBoundaryEdges;
            q["non_manifold_edges"] = r.mQuality.mNonManifoldEdges;
            q["inconsistent_pairs"] = r.mQuality.mInconsistentPairs;
            q["degenerate_triangles"] = r.mQuality.mDegenerateTriangles;
            q["watertight"] = r.mQuality.mWatertight;
            out["quality"] = q;
            return out;
        },
        py::arg("mesh"), py::arg("target"), py::arg("offset") = 0.0, py::arg("max_distance") = 0.0,
        py::arg("weights") = "", py::arg("target_region") = "", py::arg("normal_weight") = "angle",
        py::arg("record_distance") = false, py::arg("record_closest_cell") = false,
        py::arg("grid_cell_size") = 0.0);

    // Sobolev deformation: a Helmholtz-filtered displacement over the mesh's
    // P1 operators. {mesh, counters}. `fixed_points` is a numpy array of point
    // ids (smooth's `frozen` shape); `fixed_points_array` names a point_data
    // mask. See operations/sobolev_deform.hpp.
    m.def(
        "sobolev_deform",
        [](py::object pymesh, const std::string& array, double length_scale,
           py::object fixed_points, const std::string& fixed_points_array, bool fix_boundary,
           bool record_filtered, int max_iterations, double tolerance) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::SobolevOptions options;
            options.mArrayName = array;
            options.mLengthScale = length_scale;
            options.mFixedPointsArray = fixed_points_array;
            options.mFixBoundary = fix_boundary;
            options.mRecordFiltered = record_filtered;
            options.mMaxIterations = max_iterations;
            options.mTolerance = tolerance;
            if (!fixed_points.is_none()) {
                py::array_t<std::int64_t> ids =
                    py::cast<py::array_t<std::int64_t>>(py::array::ensure(fixed_points));
                options.mFixedPoints.assign(cpp.NumPoints(), 0);
                auto v = ids.unchecked<1>();
                for (py::ssize_t k = 0; k < v.shape(0); ++k) {
                    const std::int64_t id = v(k);
                    if (id < 0 || static_cast<std::size_t>(id) >= cpp.NumPoints())
                        throw std::invalid_argument("meshio++: sobolev_deform: fixed point id " +
                                                    std::to_string(id) + " is out of range");
                    options.mFixedPoints[static_cast<std::size_t>(id)] = 1;
                }
            }
            meshioplusplus::SobolevResult r = meshioplusplus::sobolev_deform(cpp, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["num_iterations"] = r.mNumIterations;
            out["residual"] = r.mResidual;
            out["converged"] = r.mConverged;
            out["num_fixed"] = r.mNumFixed;
            out["num_isolated"] = r.mNumIsolated;
            out["max_displacement"] = r.mMaxDisplacement;
            return out;
        },
        py::arg("mesh"), py::arg("array"), py::arg("length_scale"),
        py::arg("fixed_points") = py::none(), py::arg("fixed_points_array") = "",
        py::arg("fix_boundary") = false, py::arg("record_filtered") = false,
        py::arg("max_iterations") = 128, py::arg("tolerance") = 1e-10);

    m.def(
        "hessian",
        [](py::object pymesh, const std::string& array, const std::string& method,
           const std::string& location, const std::string& output, bool overwrite) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::HessianOptions options;
            options.mArrayName = array;
            options.mMethod = meshioplusplus::gradient_method_from_name(method);
            options.mLocation = meshioplusplus::data_location_from_name(location);
            options.mOutputName = output;
            options.mOverwrite = overwrite;
            meshioplusplus::HessianResult r = meshioplusplus::hessian(cpp, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["num_skipped"] = r.mNumSkipped;
            out["num_fallback"] = r.mNumFallback;
            return out;
        },
        py::arg("mesh"), py::arg("array"), py::arg("method") = "green-gauss",
        py::arg("location") = "cell", py::arg("output") = "", py::arg("overwrite") = false);

    // The Zienkiewicz-Zhu recovery-based error estimator plus marking, built
    // entirely as a composition of `gradient` + the two `data_average`
    // directions. See operations/error.hpp for the indicator/marking contract.
    m.def(
        "estimate_error",
        [](py::object pymesh, const std::string& array, const std::string& method,
           const std::string& marking, double marking_value, const std::string& output,
           const std::string& marked_name, bool overwrite) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::ErrorOptions options;
            options.mArrayName = array;
            options.mMethod = meshioplusplus::error_method_from_name(method);
            options.mMarking = meshioplusplus::error_marking_from_name(marking);
            options.mMarkingValue = marking_value;
            options.mOutputName = output;
            options.mMarkedName = marked_name;
            options.mOverwrite = overwrite;
            meshioplusplus::ErrorResult r = meshioplusplus::estimate_error(cpp, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["global_error"] = r.mGlobalError;
            out["num_skipped"] = r.mNumSkipped;
            out["num_marked"] = r.mNumMarked;
            return out;
        },
        py::arg("mesh"), py::arg("array"), py::arg("method") = "zz", py::arg("marking") = "none",
        py::arg("marking_value") = 0.0, py::arg("output") = "", py::arg("marked_name") = "",
        py::arg("overwrite") = false);

    // Isotropic / feature-preserving surface remeshing by approximated
    // centroidal Voronoi diagram clustering. Output is a brand-new,
    // unrelated mesh (no point/cell maps, unlike decimate_volume) with
    // point_data/cell_data/regions dropped, so the dict carries only the
    // mesh plus the run's scalar counters. See operations/remesh.hpp.
    m.def(
        "remesh",
        [](py::object pymesh, std::int64_t num_clusters, int subdivide, double subsample_ratio,
           int max_subdivide, int max_iterations, int max_repair_passes, const std::string& metric,
           double gradation, bool preserve_boundary, double max_anisotropy) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::RemeshOptions options;
            options.mNumClusters = num_clusters;
            options.mSubdivide = subdivide;
            options.mSubsampleRatio = subsample_ratio;
            options.mMaxSubdivide = max_subdivide;
            options.mMaxIterations = max_iterations;
            options.mMaxRepairPasses = max_repair_passes;
            options.mMetric = meshioplusplus::remesh_metric_from_name(metric);
            options.mGradation = gradation;
            options.mPreserveBoundary = preserve_boundary;
            options.mMaxAnisotropy = max_anisotropy;
            meshioplusplus::RemeshResult r = meshioplusplus::remesh(cpp, options);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["num_clusters"] = r.mNumClusters;
            out["num_iterations"] = r.mNumIterations;
            out["subdivide_applied"] = r.mSubdivideApplied;
            out["num_isolated_clusters"] = r.mNumIsolatedClusters;
            out["num_non_manifold_vertices"] = r.mNumNonManifoldVertices;
            return out;
        },
        py::arg("mesh"), py::arg("num_clusters"), py::arg("subdivide") = -1,
        py::arg("subsample_ratio") = 10.0, py::arg("max_subdivide") = 4,
        py::arg("max_iterations") = 100, py::arg("max_repair_passes") = 10,
        py::arg("metric") = "isotropic", py::arg("gradation") = 0.0,
        py::arg("preserve_boundary") = true,
        py::arg("max_anisotropy") = meshioplusplus::kRemeshDefaultMaxAnisotropy);

    // The settings.json pipeline (read -> operation chain -> write), run
    // entirely in C++ against file paths. Bound for parity tests and for
    // symmetry with the flat bindings -- the public Python `run_pipeline`
    // (`_pipeline.py`) dispatches over the Python API instead, which is what
    // gives it the per-format Python fallbacks. Both return the same report
    // shape {steps: [{op, ...counters}], warnings: [...]}, PascalCase counter
    // keys. See operations/pipeline.hpp and doc/pipeline.md.
    {
        auto report_to_py = [](const meshioplusplus::PipelineReport& r) {
            py::list steps;
            for (const auto& entry : r.mSteps) {
                py::dict step;
                step["op"] = entry.mOp;
                for (const auto& counter : entry.mCounters)
                    step[py::str(counter.first)] = counter.second;
                steps.append(step);
            }
            py::list warnings;
            for (const auto& warning : r.mWarnings)
                warnings.append(warning);
            py::dict out;
            out["steps"] = steps;
            out["warnings"] = warnings;
            return out;
        };
        m.def(
            "run_pipeline_file",
            [report_to_py](const std::string& path) {
                return report_to_py(meshioplusplus::run_pipeline_file(path));
            },
            py::arg("path"));
        m.def(
            "run_pipeline_json",
            [report_to_py](const std::string& text) {
                return report_to_py(meshioplusplus::run_pipeline_json(text));
            },
            py::arg("text"));
        // The step vocabulary (op -> parameter keys), for pinning the pure
        // Python runner's transcribed table (the refine_mask_table precedent).
        m.def("pipeline_op_table", []() {
            py::dict out;
            for (const auto& entry : meshioplusplus::pipeline_op_table()) {
                py::list keys;
                for (const auto& key : entry.second)
                    keys.append(key);
                out[py::str(entry.first)] = keys;
            }
            return out;
        });

        // The sequence document (a whole transient run: glob/list input, the
        // chain applied per step, fan-out/fan-in output). Same relationship to
        // the public Python `run_sequence_pipeline` as above.
        m.def(
            "run_sequence_file",
            [report_to_py](const std::string& path) {
                return report_to_py(meshioplusplus::run_sequence_file(path));
            },
            py::arg("path"));
        m.def(
            "run_sequence_json",
            [report_to_py](const std::string& text) {
                return report_to_py(meshioplusplus::run_sequence_json(text));
            },
            py::arg("text"));
    }

    // The sequence layer's pure units, exposed so the Python twins in
    // `_sequence.py` are pinned against these rather than transcribed and
    // hoped for (the `pipeline_op_table` / `refine_mask_table` / `colormap_table`
    // precedent). A natural-numeric comparator and a glob matcher that disagree
    // across the boundary would order a transient dataset differently depending
    // on which engine ran it -- silently.
    m.def("sequence_natural_less", &meshioplusplus::sequence_natural_less, py::arg("a"),
          py::arg("b"));
    m.def("sequence_glob_match", &meshioplusplus::sequence_glob_match, py::arg("pattern"),
          py::arg("name"));
    m.def("sequence_expand_pattern", &meshioplusplus::sequence_expand_pattern, py::arg("pattern"),
          py::arg("index"), py::arg("count"));

    // The plan for a sequence: the ordered entries with their time values and
    // where each came from. Reads no heavy data.
    m.def(
        "sequence_entries",
        [](const std::vector<std::string>& paths, const std::string& pattern,
           const std::string& file_format, const std::vector<double>& times,
           const std::string& time_from, bool sort) {
            meshioplusplus::SequenceInput in;
            in.mPaths = paths;
            in.mPattern = pattern;
            in.mFormat = file_format;
            in.mTimes = times;
            in.mTimeFrom = meshioplusplus::sequence_time_from_name(time_from);
            in.mSortExplicit = sort;
            py::list out;
            for (const meshioplusplus::SequenceEntry& e : meshioplusplus::sequence_expand(in)) {
                py::dict d;
                d["path"] = e.mPath;
                d["step"] = e.mStep;
                d["time"] = e.mTime;
                d["time_source"] = meshioplusplus::sequence_time_source_name(e.mTimeSource);
                out.append(d);
            }
            return out;
        },
        py::arg("paths") = std::vector<std::string>{}, py::arg("pattern") = "",
        py::arg("file_format") = "", py::arg("times") = std::vector<double>{},
        py::arg("time_from") = "auto", py::arg("sort") = false);

    // Partition into nparts balanced pieces (SFC or KaHIP). Returns a list of
    // dicts {part_id, mesh, point_map, cell_maps}. See operations/partition.hpp.
    m.def(
        "partition",
        [](py::object pymesh, int nparts, const std::string& method, double imbalance,
           const std::string& mode, int seed, bool record_ids, int ghost_layers,
           const std::string& weights_key) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::PartitionOptions options;
            options.mNParts = nparts;
            options.mMethod = meshioplusplus::partition_method_from_name(method);
            options.mImbalance = imbalance;
            options.mMode = meshioplusplus::partition_mode_from_name(mode);
            options.mSeed = seed;
            options.mRecordIds = record_ids;
            options.mGhostLayers = ghost_layers;
            options.mWeightsKey = weights_key;
            meshioplusplus::PartitionResult r = meshioplusplus::partition(cpp, options);
            py::list pieces;
            for (meshioplusplus::PartitionPiece& p : r.mPieces) {
                py::dict d;
                d["part_id"] = p.mPartId;
                d["mesh"] = meshioplusplus_py::mesh_to_py(std::move(p.mMesh));
                d["point_map"] = meshioplusplus_py::numpy_from_ndarray(std::move(p.mPointMap));
                py::list cell_maps;
                for (meshioplusplus::NDArray& a : p.mCellMaps)
                    cell_maps.append(meshioplusplus_py::numpy_from_ndarray(std::move(a)));
                d["cell_maps"] = cell_maps;
                pieces.append(d);
            }
            return pieces;
        },
        py::arg("mesh"), py::arg("nparts"), py::arg("method") = "auto", py::arg("imbalance") = 0.03,
        py::arg("mode") = "eco", py::arg("seed") = 0, py::arg("record_ids") = false,
        py::arg("ghost_layers") = 0, py::arg("weights_key") = "");

    // The per-cell part assignment only (block-aligned Int64 arrays). See
    // operations/partition.hpp.
    m.def(
        "partition_labels",
        [](py::object pymesh, int nparts, const std::string& method, double imbalance,
           const std::string& mode, int seed, const std::string& weights_key) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::PartitionOptions options;
            options.mNParts = nparts;
            options.mMethod = meshioplusplus::partition_method_from_name(method);
            options.mImbalance = imbalance;
            options.mMode = meshioplusplus::partition_mode_from_name(mode);
            options.mSeed = seed;
            options.mWeightsKey = weights_key;
            std::vector<meshioplusplus::NDArray> labels =
                meshioplusplus::partition_labels(cpp, options);
            py::list out;
            for (meshioplusplus::NDArray& a : labels)
                out.append(meshioplusplus_py::numpy_from_ndarray(std::move(a)));
            return out;
        },
        py::arg("mesh"), py::arg("nparts"), py::arg("method") = "auto", py::arg("imbalance") = 0.03,
        py::arg("mode") = "eco", py::arg("seed") = 0, py::arg("weights_key") = "");

    // Geometric statistics (read-only). Returns a dict of the StatsReport fields.
    // See operations/stats.hpp.
    m.def(
        "compute_stats",
        [](py::object pymesh) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::StatsReport r = meshioplusplus::compute_stats(cpp);
            py::dict out;
            out["num_points"] = r.mNumPoints;
            out["num_cells"] = r.mNumCells;
            out["bbox_min"] = py::make_tuple(r.mBBoxMin[0], r.mBBoxMin[1], r.mBBoxMin[2]);
            out["bbox_max"] = py::make_tuple(r.mBBoxMax[0], r.mBBoxMax[1], r.mBBoxMax[2]);
            out["extent"] = py::make_tuple(r.mExtent[0], r.mExtent[1], r.mExtent[2]);
            out["centroid"] = py::make_tuple(r.mCentroid[0], r.mCentroid[1], r.mCentroid[2]);
            py::dict counts;
            for (const auto& kv : r.mCellTypeCounts)
                counts[py::str(kv.first)] = kv.second;
            out["cell_type_counts"] = counts;
            out["total_area"] = r.mTotalArea;
            out["signed_volume"] = r.mSignedVolume;
            out["unsigned_volume"] = r.mUnsignedVolume;
            out["num_inverted"] = r.mNumInverted;
            return out;
        },
        py::arg("mesh"));

    // --- data operations ---------------------------------------------------
    // Operations on the point_data / cell_data / field_data arrays rather than
    // on the geometry. Enums cross as strings (the same names the CLIs and the
    // WASM binding use). See operations/data_*.hpp.

    // Array management: keep / drop / rename, applied in that order. Returns a
    // dict {mesh, dropped, renamed}. See operations/data_manage.hpp.
    m.def(
        "data_manage",
        [](py::object pymesh, const std::vector<std::pair<std::string, std::string>>& keep,
           const std::vector<std::pair<std::string, std::string>>& drop,
           const std::vector<std::tuple<std::string, std::string, std::string>>& rename,
           bool ignore_missing) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::DataManageOptions opts;
            opts.ignore_missing = ignore_missing;
            for (const auto& kv : keep)
                opts.keep.push_back(meshioplusplus::DataKey{
                    meshioplusplus::data_location_from_name(kv.first), kv.second});
            for (const auto& kv : drop)
                opts.drop.push_back(meshioplusplus::DataKey{
                    meshioplusplus::data_location_from_name(kv.first), kv.second});
            for (const auto& t : rename)
                opts.rename.push_back(meshioplusplus::DataRename{
                    meshioplusplus::data_location_from_name(std::get<0>(t)), std::get<1>(t),
                    std::get<2>(t)});
            meshioplusplus::DataManageResult r = meshioplusplus::data_manage(cpp, opts);
            py::dict out;
            out["mesh"] = meshioplusplus_py::mesh_to_py(std::move(r.mMesh));
            out["dropped"] = py::cast(r.mDropped);
            py::list renamed;
            for (const auto& kv : r.mRenamed)
                renamed.append(py::make_tuple(kv.first, kv.second));
            out["renamed"] = renamed;
            return out;
        },
        py::arg("mesh"), py::arg("keep") = std::vector<std::pair<std::string, std::string>>{},
        py::arg("drop") = std::vector<std::pair<std::string, std::string>>{},
        py::arg("rename") = std::vector<std::tuple<std::string, std::string, std::string>>{},
        py::arg("ignore_missing") = false);

    // point_data -> cell_data averaging. See operations/data_average.hpp.
    m.def(
        "point_data_to_cell_data",
        [](py::object pymesh, const std::vector<std::string>& names, const std::string& prefix,
           const std::string& suffix, bool overwrite, const std::string& nan_policy,
           double nan_replacement) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::DataAverageOptions opts;
            opts.names = names;
            opts.prefix = prefix;
            opts.suffix = suffix;
            opts.overwrite = overwrite;
            opts.nan_policy = meshioplusplus::nan_policy_from_name(nan_policy);
            opts.nan_replacement = nan_replacement;
            return meshioplusplus_py::mesh_to_py(
                meshioplusplus::point_data_to_cell_data(cpp, opts));
        },
        py::arg("mesh"), py::arg("names") = std::vector<std::string>{}, py::arg("prefix") = "",
        py::arg("suffix") = "", py::arg("overwrite") = true, py::arg("nan_policy") = "ignore",
        py::arg("nan_replacement") = 0.0);

    // cell_data -> point_data averaging. See operations/data_average.hpp.
    m.def(
        "cell_data_to_point_data",
        [](py::object pymesh, const std::vector<std::string>& names, const std::string& weight,
           const std::string& prefix, const std::string& suffix, bool overwrite,
           const std::string& nan_policy, double nan_replacement) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::DataAverageOptions opts;
            opts.names = names;
            opts.weight = meshioplusplus::cell_point_weight_from_name(weight);
            opts.prefix = prefix;
            opts.suffix = suffix;
            opts.overwrite = overwrite;
            opts.nan_policy = meshioplusplus::nan_policy_from_name(nan_policy);
            opts.nan_replacement = nan_replacement;
            return meshioplusplus_py::mesh_to_py(
                meshioplusplus::cell_data_to_point_data(cpp, opts));
        },
        py::arg("mesh"), py::arg("names") = std::vector<std::string>{},
        py::arg("weight") = "uniform", py::arg("prefix") = "", py::arg("suffix") = "",
        py::arg("overwrite") = true, py::arg("nan_policy") = "ignore",
        py::arg("nan_replacement") = 0.0);

    // Elementwise expression evaluator. See operations/data_calc.hpp.
    m.def(
        "data_calc",
        [](py::object pymesh, const std::string& expression, const std::string& location,
           const std::string& output, bool overwrite) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::DataCalcOptions opts;
            opts.location = meshioplusplus::data_location_from_name(location);
            opts.output = output;
            opts.overwrite = overwrite;
            return meshioplusplus_py::mesh_to_py(meshioplusplus::data_calc(cpp, expression, opts));
        },
        py::arg("mesh"), py::arg("expression"), py::arg("location") = "point",
        py::arg("output") = "", py::arg("overwrite") = false);

    // Value conditioning (clamp / normalize / standardize).
    // See operations/data_condition.hpp.
    m.def(
        "data_condition",
        [](py::object pymesh, const std::string& location, const std::vector<std::string>& names,
           const std::string& mode, const std::string& scope, double lo, double hi,
           const std::string& nan_policy, double nan_replacement, const std::string& suffix,
           bool preserve_dtype) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::DataConditionOptions opts;
            opts.location = meshioplusplus::data_location_from_name(location);
            opts.names = names;
            opts.mode = meshioplusplus::condition_mode_from_name(mode);
            opts.scope = meshioplusplus::condition_scope_from_name(scope);
            opts.lo = lo;
            opts.hi = hi;
            opts.nan_policy = meshioplusplus::nan_policy_from_name(nan_policy);
            opts.nan_replacement = nan_replacement;
            opts.suffix = suffix;
            opts.preserve_dtype = preserve_dtype;
            return meshioplusplus_py::mesh_to_py(meshioplusplus::data_condition(cpp, opts));
        },
        py::arg("mesh"), py::arg("location") = "point",
        py::arg("names") = std::vector<std::string>{}, py::arg("mode") = "clamp",
        py::arg("scope") = "component", py::arg("lo") = 0.0, py::arg("hi") = 1.0,
        py::arg("nan_policy") = "ignore", py::arg("nan_replacement") = 0.0, py::arg("suffix") = "",
        py::arg("preserve_dtype") = true);

    // Read-only per-array data summary. Returns a list of dicts, one per array.
    // See operations/data_info.hpp.
    m.def(
        "data_info",
        [](py::object pymesh) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::DataInfoReport r = meshioplusplus::data_info(cpp);
            py::list arrays;
            for (const meshioplusplus::DataArrayInfo& a : r.mArrays) {
                py::dict d;
                d["location"] = meshioplusplus::data_location_name(a.mLocation);
                d["name"] = a.mName;
                d["dtype"] = meshioplusplus::dtype_numpy_str(a.mDtype);
                d["shape"] = py::cast(a.mShape);
                d["num_blocks"] = a.mNumBlocks;
                d["num_entries"] = a.mNumEntries;
                d["num_components"] = a.mNumComponents;
                d["num_values"] = a.mNumValues;
                d["min"] = a.mMin;
                d["max"] = a.mMax;
                d["mean"] = a.mMean;
                d["min_per_component"] = py::cast(a.mMinPerComponent);
                d["max_per_component"] = py::cast(a.mMaxPerComponent);
                d["mean_per_component"] = py::cast(a.mMeanPerComponent);
                d["num_nan"] = a.mNumNan;
                d["num_inf"] = a.mNumInf;
                d["num_finite"] = a.mNumFinite;
                d["inconsistent_blocks"] = a.mInconsistentBlocks;
                arrays.append(d);
            }
            return arrays;
        },
        py::arg("mesh"));

    // Cell-measure-weighted field integration (total/mean, whole mesh and
    // per named Cell region). Returns a list of dicts, one per array. See
    // operations/data_integrate.hpp.
    m.def(
        "data_integrate",
        [](py::object pymesh, py::object arrays) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::DataIntegrateOptions opts;
            if (!arrays.is_none())
                opts.mArrayNames = py::cast<std::vector<std::string>>(arrays);
            meshioplusplus::DataIntegrateReport r = meshioplusplus::data_integrate(cpp, opts);

            auto region_dict = [](const meshioplusplus::FieldIntegralRegion& reg) {
                py::dict d;
                d["name"] = reg.mName;
                d["num_cells"] = reg.mNumCells;
                d["num_skipped"] = reg.mNumSkipped;
                d["domain_measure_per_component"] = py::cast(reg.mDomainMeasurePerComponent);
                d["total_per_component"] = py::cast(reg.mTotalPerComponent);
                d["mean_per_component"] = py::cast(reg.mMeanPerComponent);
                d["num_nan_per_component"] = py::cast(reg.mNumNanPerComponent);
                return d;
            };

            py::list out;
            for (const meshioplusplus::FieldIntegralArray& a : r.mArrays) {
                py::dict d;
                d["name"] = a.mName;
                d["num_components"] = a.mNumComponents;
                d["domain"] = region_dict(a.mDomain);
                py::list regions;
                for (const meshioplusplus::FieldIntegralRegion& reg : a.mRegions)
                    regions.append(region_dict(reg));
                d["regions"] = regions;
                out.append(d);
            }
            return out;
        },
        py::arg("mesh"), py::arg("arrays") = py::none());

    // Mesh comparison ("diff"). Returns a nested dict mirroring DiffReport, with
    // an overall verdict string. See operations/diff.hpp.
    m.def(
        "diff",
        [](py::object pymesh_a, py::object pymesh_b, double atol, double rtol, bool unordered,
           std::int64_t max_reported) {
            meshioplusplus_py::PyMeshRefs refs_a, refs_b;
            meshioplusplus::Mesh a = meshioplusplus_py::py_to_mesh(pymesh_a, refs_a);
            meshioplusplus::Mesh b = meshioplusplus_py::py_to_mesh(pymesh_b, refs_b);
            meshioplusplus::DiffOptions opts;
            opts.atol = atol;
            opts.rtol = rtol;
            opts.unordered = unordered;
            opts.max_reported = max_reported;
            meshioplusplus::DiffReport rep = meshioplusplus::diff(a, b, opts);

            auto array_diff = [](const meshioplusplus::ArrayDiff& ad) {
                py::dict d;
                d["name"] = ad.mName;
                d["shape_mismatch"] = ad.mShapeMismatch;
                d["size_a"] = ad.mSizeA;
                d["size_b"] = ad.mSizeB;
                d["max_abs_error"] = ad.mMaxAbsError;
                d["max_rel_error"] = ad.mMaxRelError;
                d["worst_index"] = ad.mWorstIndex;
                d["num_exceeding"] = ad.mNumExceeding;
                d["exact"] = ad.mExact;
                return d;
            };
            auto data_diff = [&](const meshioplusplus::DataDiff& dd) {
                py::dict d;
                py::list only_a, only_b, shared;
                for (const std::string& s : dd.mOnlyInA)
                    only_a.append(s);
                for (const std::string& s : dd.mOnlyInB)
                    only_b.append(s);
                for (const meshioplusplus::ArrayDiff& ad : dd.mShared)
                    shared.append(array_diff(ad));
                d["only_in_a"] = only_a;
                d["only_in_b"] = only_b;
                d["shared"] = shared;
                return d;
            };

            py::dict out;
            out["verdict"] = meshioplusplus::diff_verdict_name(rep.mVerdict);
            out["unordered"] = rep.mUnordered;
            out["correspondence_failed"] = rep.mCorrespondenceFailed;
            out["point_count_mismatch"] = rep.mPointCountMismatch;
            out["num_points_a"] = rep.mNumPointsA;
            out["num_points_b"] = rep.mNumPointsB;
            out["points"] = array_diff(rep.mPoints);
            out["block_count_mismatch"] = rep.mBlockCountMismatch;
            out["num_blocks_a"] = rep.mNumBlocksA;
            out["num_blocks_b"] = rep.mNumBlocksB;
            py::list blocks;
            for (const meshioplusplus::BlockDiff& bd : rep.mBlocks) {
                py::dict d;
                d["block"] = bd.mBlock;
                d["type_a"] = bd.mTypeA;
                d["type_b"] = bd.mTypeB;
                d["count_a"] = bd.mCountA;
                d["count_b"] = bd.mCountB;
                d["type_mismatch"] = bd.mTypeMismatch;
                d["count_mismatch"] = bd.mCountMismatch;
                d["conn_mismatch_count"] = bd.mConnMismatchCount;
                py::list firsts;
                for (std::int64_t f : bd.mFirstMismatches)
                    firsts.append(f);
                d["first_mismatches"] = firsts;
                blocks.append(d);
            }
            out["blocks"] = blocks;
            out["point_data"] = data_diff(rep.mPointData);
            out["cell_data"] = data_diff(rep.mCellData);
            out["field_data"] = data_diff(rep.mFieldData);
            py::list messages;
            for (const std::string& s : rep.mMessages)
                messages.append(s);
            out["messages"] = messages;
            return out;
        },
        py::arg("mesh_a"), py::arg("mesh_b"), py::arg("atol") = 1e-12, py::arg("rtol") = 1e-9,
        py::arg("unordered") = false, py::arg("max_reported") = 10);

    // Convenience boolean: are two meshes equal within tolerance (ordered)?
    m.def(
        "meshes_equal",
        [](py::object pymesh_a, py::object pymesh_b, double atol, double rtol) {
            meshioplusplus_py::PyMeshRefs refs_a, refs_b;
            meshioplusplus::Mesh a = meshioplusplus_py::py_to_mesh(pymesh_a, refs_a);
            meshioplusplus::Mesh b = meshioplusplus_py::py_to_mesh(pymesh_b, refs_b);
            return meshioplusplus::meshes_equal(a, b, atol, rtol);
        },
        py::arg("mesh_a"), py::arg("mesh_b"), py::arg("atol") = 1e-12, py::arg("rtol") = 1e-9);

    // STL writer / reader (ascii or binary).
    m.def(
        "stl_write",
        [](const std::string& path, py::object pymesh, bool binary, bool skin) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
            meshioplusplus::write_stl(path, cpp, binary, skin);
        },
        py::arg("path"), py::arg("mesh"), py::arg("binary") = false, py::arg("skin") = true);
    m.def("stl_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_stl(path));
    });

    // OFF writer / reader.
    m.def("off_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_off(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("off_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_off(path));
    });

    // OBJ writer / reader.
    m.def("obj_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_obj(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("obj_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_obj(path));
    });

    // Gmsh 2.2 writer / reader.
    m.def("gmsh22_write", [](const std::string& path, py::object pymesh, bool binary) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
        meshioplusplus::write_gmsh22(path, cpp, binary);
    });
    m.def(
        "gmsh41_write",
        [](const std::string& path, py::object pymesh, bool binary, py::object bounding_entities) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
            // The $Entities bounding entities cannot live on the C++ Mesh (they
            // are signed entity tags, not indices), so the shim hands them over
            // separately -- the read path's GmshInfo channel, in reverse.
            meshioplusplus::GmshInfo info;
            if (!bounding_entities.is_none()) {
                for (py::handle blk : py::cast<py::sequence>(bounding_entities)) {
                    std::vector<std::int32_t> tags;
                    if (!blk.is_none()) {
                        // forcecast: the shim may hand over a list or any
                        // integer-dtype array, not necessarily int32.
                        auto arr = py::cast<py::array_t<std::int32_t, py::array::forcecast>>(blk);
                        tags.assign(arr.data(), arr.data() + arr.size());
                    }
                    info.mBoundingEntities.push_back(std::move(tags));
                }
            }
            meshioplusplus::write_gmsh41(path, cpp, binary, info);
        },
        py::arg("path"), py::arg("mesh"), py::arg("binary"),
        py::arg("bounding_entities") = py::none());
    m.def(
        "gmsh_read",
        [](const std::string& path, bool points_only, py::object arrays) {
            meshioplusplus::GmshInfo info;
            py::object pymesh = meshioplusplus_py::mesh_to_py(
                meshioplusplus::read_gmsh(path, info, core_read_options(points_only, arrays)));
            // The 4.1 $Entities bounding entities are signed entity tags, not
            // cell indices, so they ride the GmshInfo side channel and land in
            // cell_sets here -- where the Mesh's own predicate routes them to
            // the verbatim passthrough, exactly as the Python reference's do.
            if (!info.mBoundingEntities.empty()) {
                py::list blocks;
                for (const auto& tags : info.mBoundingEntities)
                    blocks.append(py::array_t<std::int32_t>(static_cast<py::ssize_t>(tags.size()),
                                                            tags.data()));
                pymesh.attr("cell_sets")["gmsh:bounding_entities"] = std::move(blocks);
            }
            return pymesh;
        },
        py::arg("path"), py::arg("points_only") = false, py::arg("arrays") = py::none());

    // PLY writer / reader (ascii or binary).
    m.def(
        "ply_write",
        [](const std::string& path, py::object pymesh, bool binary, bool skin) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
            meshioplusplus::write_ply(path, cpp, binary, skin);
        },
        py::arg("path"), py::arg("mesh"), py::arg("binary") = true, py::arg("skin") = true);
    m.def("ply_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_ply(path));
    });

    // Medit ascii writer / reader (.mesh).
    m.def("medit_write_ascii", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_medit_ascii(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("medit_read_ascii", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_medit_ascii(path));
    });

    // Kratos MDPA writer / reader (.mdpa).
    m.def("mdpa_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_mdpa(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("mdpa_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_mdpa(path));
    });

    // Abaqus writer / reader (.inp).
    m.def("abaqus_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_abaqus(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("abaqus_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_abaqus(path));
    });

    // AVS-UCD writer / reader (.avs).
    m.def("avsucd_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_avsucd(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("avsucd_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_avsucd(path));
    });

    // Nastran writer / reader (.bdf/.fem/.nas) — meshio++-C++ files only.
    m.def("nastran_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_nastran(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("nastran_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_nastran(path));
    });

    // SU2 writer / reader (.su2).
    m.def("su2_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_su2(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("su2_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_su2(path));
    });

    // Tecplot writer / reader (.dat/.tec).
    m.def("tecplot_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_tecplot(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("tecplot_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_tecplot(path));
    });

    // UGRID writer / reader (.ugrid, ascii + binary variants).
    m.def("ugrid_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_ugrid(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("ugrid_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_ugrid(path));
    });

    // UNV (I-DEAS Universal) writer / reader (.unv).  point_data/cell_data
    // become field datasets 2414 (or 55/57 in Code-Aster mode); permanent
    // groups (point_sets/cell_sets) travel via the UnvInfo side-channel.
    m.def(
        "unv_write",
        [](const std::string& path, py::object pymesh,
           std::map<std::string, std::vector<std::int64_t>> point_sets,
           std::map<std::string, std::vector<std::vector<std::int64_t>>> cell_sets, bool code_aster,
           int node_dataset) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
            meshioplusplus::UnvInfo info;
            info.mPointSets = std::move(point_sets);
            info.mCellSets = std::move(cell_sets);
            meshioplusplus::write_unv(path, cpp, info, code_aster, node_dataset);
        },
        py::arg("path"), py::arg("mesh"), py::arg("point_sets"), py::arg("cell_sets"),
        py::arg("code_aster") = false, py::arg("node_dataset") = 2411);
    m.def("unv_read", [](const std::string& path) {
        meshioplusplus::UnvInfo info;
        py::object pymesh = meshioplusplus_py::mesh_to_py(meshioplusplus::read_unv(path, info));
        py::dict psets, csets;
        for (const auto& kv : info.mPointSets)
            psets[py::str(kv.first)] = py::array_t<std::int64_t>(
                static_cast<py::ssize_t>(kv.second.size()), kv.second.data());
        for (const auto& kv : info.mCellSets) {
            py::list blocks;
            for (const auto& blk : kv.second)
                blocks.append(
                    py::array_t<std::int64_t>(static_cast<py::ssize_t>(blk.size()), blk.data()));
            csets[py::str(kv.first)] = blocks;
        }
        if (py::len(psets) > 0)
            pymesh.attr("point_sets") = psets;
        if (py::len(csets) > 0)
            pymesh.attr("cell_sets") = csets;
        return pymesh;
    });

    // EnSight Gold writer / reader (.case/.geo pair, geometry only).
    m.def("ensight_write", [](const std::string& path, py::object pymesh, bool binary) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_ensight(path, meshioplusplus_py::py_to_mesh(pymesh, refs), binary);
    });
    m.def("ensight_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_ensight(path));
    });

    // TetGen writer / reader (.node/.ele pair).
    m.def("tetgen_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_tetgen(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("tetgen_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_tetgen(path));
    });

    // Triangle writer / reader (.node/.ele pair or .poly).
    m.def("triangle_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_triangle(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("triangle_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_triangle(path));
    });

    // XDMF writer / reader (.xdmf/.xmf) — XML/Binary always; HDF when built
    // with HDF5.
    m.def(
        "xdmf_write",
        [](const std::string& path, py::object pymesh, const std::string& data_format,
           int gzip_level) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
            meshioplusplus::write_xdmf(path, cpp, data_format, gzip_level);
        },
        py::arg("path"), py::arg("mesh"), py::arg("data_format"), py::arg("gzip_level") = -1);
    m.def(
        "xdmf_read",
        [](const std::string& path, bool points_only, py::object arrays, int time_step) {
            return meshioplusplus_py::mesh_to_py(
                meshioplusplus::read_xdmf(path, core_read_options(points_only, arrays, time_step)));
        },
        py::arg("path"), py::arg("points_only") = false, py::arg("arrays") = py::none(),
        // XDMF is *the* multi-step format, and `read_xdmf` has honoured
        // ReadOptions::mTimeStep since v9.0.0 -- this binding simply never
        // passed it through, so a temporal collection was unreachable from
        // Python except through `xdmf.TimeSeriesReader`. Exposed in v9.12.0 for
        // the sequence layer's fan-out (0 = first step, negative counts from
        // the end, out of range is an error naming the count).
        py::arg("time_step") = 0);

    // Transient (time-series) XDMF — the C++ `XdmfTimeSeriesWriter`, exposed
    // ADDITIONALLY and EXPLICITLY rather than swapped in under the pure-Python
    // `meshioplusplus.xdmf.TimeSeriesWriter`.
    //
    // This is deliberately NOT the try-C++/fall-back-to-Python shim pattern
    // every *format* module uses, for the same reason `.mdpa` is not: the two
    // writers are not byte-for-byte interchangeable, and the Python one has a
    // documented, tested API (`write_points_cells(points, cells)` /
    // `write_data(t, point_data=, cell_data=)`, i.e. raw arrays) that this one
    // does not share (it takes whole `Mesh` objects). Quietly changing what a
    // tested public API writes is the failure mode to avoid; a caller who wants
    // the C++ writer asks for it by name.
    //
    // Two behavioural differences that make the swap unsafe, both intentional
    // on the C++ side: the `"HDF"` companion is a sibling of the `.xdmf`
    // (`<path minus extension>.h5`) rather than `<stem>.h5` in the process CWD,
    // and arrays are emitted in the uniform API's sorted-name order.
    //
    // No `#ifdef` guard: the class is always built (XML/Binary need no HDF5),
    // and `"HDF"` on a build without HDF5 throws `WriteError` from the
    // constructor, which the translator above turns into a clean
    // `meshioplusplus.WriteError` rather than a missing symbol or a crash.
    py::class_<meshioplusplus::XdmfTimeSeriesWriter>(m, "XdmfTimeSeriesWriter",
                                                     R"doc(
Transient XDMF3 writer: one static grid plus one <Grid> per time step.

The C++ core's writer, reachable explicitly. It is *not* what
``meshioplusplus.xdmf.TimeSeriesWriter`` uses — that one is the pure-Python
reference writer and keeps its own behaviour untouched.

Unlike the Python writer, both methods take a whole ``Mesh``:
``write_points_cells`` uses its points/cells, ``write_data`` its
``point_data``/``cell_data``. Usable as a context manager; ``__exit__``
finalizes.

>>> with _core.XdmfTimeSeriesWriter("out.xdmf") as w:
...     w.write_points_cells(mesh)
...     for k in range(3):
...         w.write_data(k * 0.5, mesh)
)doc")
        .def(py::init([](const std::string& rPath, const std::string& rDataFormat, int gzip_level,
                         const std::string& rMode) {
                 if (rMode != "truncate" && rMode != "append")
                     throw std::invalid_argument("mode must be 'truncate' or 'append', got '" +
                                                 rMode + "'");
                 return std::make_unique<meshioplusplus::XdmfTimeSeriesWriter>(
                     rPath, rDataFormat, gzip_level,
                     rMode == "append" ? meshioplusplus::XdmfSeriesMode::Append
                                       : meshioplusplus::XdmfSeriesMode::Truncate);
             }),
             py::arg("path"), py::arg("data_format") = "HDF", py::arg("gzip_level") = -1,
             py::arg("mode") = "truncate")
        .def(
            "write_points_cells",
            [](meshioplusplus::XdmfTimeSeriesWriter& rSelf, py::object pymesh) {
                meshioplusplus_py::PyMeshRefs refs;
                meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
                rSelf.WritePointsCells(cpp);
            },
            py::arg("mesh"), "Write the static grid (points + cells). Once, before write_data.")
        .def(
            "write_data",
            [](meshioplusplus::XdmfTimeSeriesWriter& rSelf, double time, py::object pymesh) {
                meshioplusplus_py::PyMeshRefs refs;
                meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
                rSelf.WriteData(time, cpp);
            },
            py::arg("time"), py::arg("mesh"),
            "Append one step's point_data/cell_data at simulation time `time`.")
        .def(
            "write_data_arrays",
            [](meshioplusplus::XdmfTimeSeriesWriter& rSelf, double time, const py::dict& rPointData,
               const py::dict& rCellData) {
                const auto convert = [](const py::dict& rSrc) {
                    std::vector<meshioplusplus::XdmfTimeSeriesWriter::NamedArray> out;
                    for (const auto& r_item : rSrc) {
                        auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>(
                            py::reinterpret_borrow<py::object>(r_item.second));
                        meshioplusplus::XdmfTimeSeriesWriter::NamedArray a;
                        a.mName = py::cast<std::string>(r_item.first);
                        // Components are the trailing dimension, the same rule
                        // the rest of the repo uses for a multi-component array.
                        a.mNumComponents = arr.ndim() >= 2
                                               ? static_cast<std::size_t>(arr.shape(arr.ndim() - 1))
                                               : 1u;
                        a.mValues.assign(arr.data(), arr.data() + arr.size());
                        out.push_back(std::move(a));
                    }
                    return out;
                };
                rSelf.WriteData(time, convert(rPointData), convert(rCellData));
            },
            py::arg("time"), py::arg("point_data"), py::arg("cell_data") = py::dict(),
            "Append one step from name -> array dicts, with no Mesh in between -- "
            "the granularity a solver has once write_points_cells has fixed the "
            "geometry. Arrays are emitted in dict order.")
        .def(
            "flush", [](meshioplusplus::XdmfTimeSeriesWriter& rSelf) { rSelf.Flush(); },
            "Write the .xdmf as it currently stands without finalizing, so a run "
            "that is killed or still going leaves a readable file covering every "
            "flushed step. Safe to call repeatedly.")
        .def_property(
            "auto_flush",
            [](const meshioplusplus::XdmfTimeSeriesWriter& rSelf) { return rSelf.AutoFlush(); },
            [](meshioplusplus::XdmfTimeSeriesWriter& rSelf, bool enable) {
                rSelf.SetAutoFlush(enable);
            },
            "Flush after every write_data (default False). Off by default because "
            "a flush re-serializes the whole document, making per-step flushing "
            "quadratic in the step count.")
        .def(
            "finalize", [](meshioplusplus::XdmfTimeSeriesWriter& rSelf) { rSelf.Finalize(); },
            "Write the .xdmf light data and close the heavy-data container. "
            "Idempotent; the destructor would do this too, but only an explicit "
            "call can raise on failure.")
        .def_property_readonly(
            "num_steps",
            [](const meshioplusplus::XdmfTimeSeriesWriter& rSelf) { return rSelf.NumSteps(); },
            "How many steps have been written so far.")
        .def_property_readonly(
            "finalized",
            [](const meshioplusplus::XdmfTimeSeriesWriter& rSelf) { return rSelf.Finalized(); },
            "Whether finalize() has already run.")
        .def("__enter__", [](py::object self) { return self; })
        // Finalizes unconditionally, mirroring the Python writer's __exit__
        // (which always writes the XML). Returning False never suppresses the
        // body's exception; if Finalize() also fails, Python chains the two.
        .def("__exit__", [](meshioplusplus::XdmfTimeSeriesWriter& rSelf, const py::object&,
                            const py::object&, const py::object&) {
            rSelf.Finalize();
            return false;
        });

#ifdef MESHIOPLUSPLUS_HAS_HDF5
    // CGNS writer / reader (.cgns).
    // `allow_ragged`: polygon/polyhedron blocks became NGON_n/NFACE_n
    // sections in v9.21.0, so they must reach the C++ writer rather than being
    // bounced back to the (deliberately NGON-less) Python twin.
    m.def("cgns_write", [](const std::string& path, py::object pymesh, int gzip_level) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
            pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
        meshioplusplus::write_cgns(path, cpp, gzip_level);
    });
    m.def("cgns_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_cgns(path));
    });

    // HMF writer / reader (.hmf).
    m.def("hmf_write", [](const std::string& path, py::object pymesh, int gzip_level) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
        meshioplusplus::write_hmf(path, cpp, gzip_level);
    });
    m.def("hmf_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_hmf(path));
    });

    // MOAB h5m writer / reader (.h5m).
    m.def("h5m_write",
          [](const std::string& path, py::object pymesh, bool add_global_ids, int gzip_level) {
              meshioplusplus_py::PyMeshRefs refs;
              meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
              meshioplusplus::write_h5m(path, cpp, add_global_ids, gzip_level);
          });
    m.def("h5m_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_h5m(path));
    });

    // MED/Salome writer / reader (.med). point_tags/cell_tags are custom Mesh
    // attributes and med:nom is a list of string-lists, so they travel outside
    // the Mesh conversion layer. Named regions (FAS/GRO group names) need no
    // such plumbing here -- write_med/read_med attach/consult them on the
    // Mesh directly (see med_attach_point_regions/med_attach_cell_regions and
    // med_point_regions_to_tags/med_cell_regions_to_tags in med.cpp), so they
    // already cross this binding through py_to_mesh/mesh_to_py's existing
    // `.regions` handling like every other format.
    m.def("med_write",
          [](const std::string& path, py::object pymesh,
             std::map<std::int64_t, std::vector<std::string>> point_tags,
             std::map<std::int64_t, std::vector<std::string>> cell_tags,
             std::vector<std::vector<std::string>> med_nom, std::string mesh_name,
             std::string description, std::string unit_time, std::string unit_coords,
             std::map<std::int64_t, std::string> point_tag_groups,
             std::map<std::int64_t, std::string> cell_tag_groups, std::string med_version) {
              meshioplusplus_py::PyMeshRefs refs;
              meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs,
                                                                       /*lenient_field_data=*/true,
                                                                       /*allow_ragged=*/true);
              meshioplusplus::MedInfo info;
              info.mPointTags = std::move(point_tags);
              info.mCellTags = std::move(cell_tags);
              info.mMedNom = std::move(med_nom);
              info.mMeshName = mesh_name.empty() ? "mesh" : std::move(mesh_name);
              info.mDescription = std::move(description);
              info.mUnitTime = std::move(unit_time);
              info.mUnitCoords = std::move(unit_coords);
              info.mPointTagGroups = std::move(point_tag_groups);
              info.mCellTagGroups = std::move(cell_tag_groups);
              meshioplusplus::write_med(path, cpp, info, med_version);
          });
    m.def("med_read", [](const std::string& path) {
        meshioplusplus::MedInfo info;
        py::object pymesh = meshioplusplus_py::mesh_to_py(meshioplusplus::read_med(path, info));
        py::dict ptags, ctags, pgroups, cgroups;
        for (const auto& kv : info.mPointTags)
            ptags[py::int_(kv.first)] = kv.second;
        for (const auto& kv : info.mCellTags)
            ctags[py::int_(kv.first)] = kv.second;
        for (const auto& kv : info.mPointTagGroups)
            pgroups[py::int_(kv.first)] = kv.second;
        for (const auto& kv : info.mCellTagGroups)
            cgroups[py::int_(kv.first)] = kv.second;
        pymesh.attr("point_tags") = ptags;
        pymesh.attr("cell_tags") = ctags;
        pymesh.attr("point_tag_groups") = pgroups;
        pymesh.attr("cell_tag_groups") = cgroups;
        pymesh.attr("mesh_name") = info.mMeshName;
        pymesh.attr("description") = info.mDescription;
        pymesh.attr("unit_time") = info.mUnitTime;
        pymesh.attr("unit_coords") = info.mUnitCoords;
        if (!info.mMedNom.empty())
            pymesh.attr("field_data")[py::str("med:nom")] = py::cast(info.mMedNom);
        return pymesh;
    });
#endif

#ifdef MESHIOPLUSPLUS_HAS_NETCDF
    // Exodus II writer / reader (.e/.exo/.ex2).
    m.def("exodus_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_exodus(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def(
        "exodus_read",
        [](const std::string& path, int time_step) {
            meshioplusplus::ReadOptions opts;
            opts.mTimeStep = time_step;
            meshioplusplus::ExodusInfo info;
            py::object pymesh =
                meshioplusplus_py::mesh_to_py(meshioplusplus::read_exodus(path, info, opts));
            // qa_records/info_records are strings, which NDArray cannot hold --
            // they ride the ExodusInfo side channel and land here, so the C++
            // path produces the same `mesh.info` the Python reference does.
            pymesh.attr("info") = py::cast(info.mInfoRecords);
            return pymesh;
        },
        py::arg("path"), py::arg("time_step") = 0);
#endif

    // DOLFIN XML writer / reader (.xml).
    m.def("dolfin_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_dolfin(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("dolfin_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_dolfin(path));
    });

    // Ansys/Fluent writer / reader (.msh, ascii + binary).
    m.def("ansys_write", [](const std::string& path, py::object pymesh, bool binary) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
        meshioplusplus::write_ansys(path, cpp, binary);
    });
    m.def("ansys_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_ansys(path));
    });

    // Ansys MAPDL coded database (.cdb/.inp). CMBLOCK components are point_sets
    // / cell_sets, custom Mesh attributes carried through the AnsysInfo
    // side-channel.
    m.def("ansysinp_write",
          [](const std::string& path, py::object pymesh,
             std::map<std::string, std::vector<std::int64_t>> point_sets,
             std::map<std::string, std::vector<std::vector<std::int64_t>>> cell_sets) {
              meshioplusplus_py::PyMeshRefs refs;
              meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
              meshioplusplus::AnsysInfo info;
              info.mPointSets = std::move(point_sets);
              info.mCellSets = std::move(cell_sets);
              meshioplusplus::write_ansysinp(path, cpp, info);
          });
    m.def("ansysinp_read", [](const std::string& path) {
        meshioplusplus::AnsysInfo info;
        py::object pymesh =
            meshioplusplus_py::mesh_to_py(meshioplusplus::read_ansysinp(path, info));
        py::dict psets, csets;
        for (const auto& kv : info.mPointSets)
            psets[py::str(kv.first)] = py::array_t<std::int64_t>(
                static_cast<py::ssize_t>(kv.second.size()), kv.second.data());
        for (const auto& kv : info.mCellSets) {
            py::list blocks;
            for (const auto& blk : kv.second)
                blocks.append(
                    py::array_t<std::int64_t>(static_cast<py::ssize_t>(blk.size()), blk.data()));
            csets[py::str(kv.first)] = blocks;
        }
        pymesh.attr("point_sets") = psets;
        pymesh.attr("cell_sets") = csets;
        return pymesh;
    });

    // OpenFOAM polyMesh reader. Boundary patch names and types are
    // mesh.cell_tags, carried through the OpenFoamInfo side-channel.
    m.def("openfoam_read", [](const std::string& path) {
        meshioplusplus::OpenFoamInfo info;
        py::object pymesh =
            meshioplusplus_py::mesh_to_py(meshioplusplus::read_openfoam(path, info));
        py::dict ctags;
        for (const auto& kv : info.mCellTags)
            ctags[py::int_(kv.first)] = kv.second;
        pymesh.attr("cell_tags") = ctags;
        pymesh.attr("point_tags") = py::dict();
        py::dict ptypes;
        for (const auto& kv : info.mPatchTypes)
            ptypes[py::int_(kv.first)] = kv.second;
        pymesh.attr("openfoam_patch_types") = ptypes;
        return pymesh;
    });

    // OpenFOAM polyMesh writer. `allow_ragged` is mandatory: a polyhedron block
    // is this format's native cell shape, and the reader emits them, so a
    // round-trip write would otherwise throw on our own output.
    m.def(
        "openfoam_write",
        [](const std::string& path, py::object pymesh, py::dict cell_tags, py::dict patch_types) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(
                pymesh, refs, /*lenient_field_data=*/false, /*allow_ragged=*/true);
            meshioplusplus::OpenFoamInfo info;
            for (auto item : cell_tags) {
                std::vector<std::string> names;
                for (auto n : py::cast<py::iterable>(item.second))
                    names.push_back(py::cast<std::string>(n));
                info.mCellTags[py::cast<std::int64_t>(item.first)] = std::move(names);
            }
            for (auto item : patch_types)
                info.mPatchTypes[py::cast<std::int64_t>(item.first)] =
                    py::cast<std::string>(item.second);
            meshioplusplus::write_openfoam(path, cpp, info);
        },
        py::arg("path"), py::arg("mesh"), py::arg("cell_tags") = py::dict(),
        py::arg("patch_types") = py::dict());

    // WKT (TIN) writer / reader (.wkt).
    m.def("wkt_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_wkt(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("wkt_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_wkt(path));
    });

    // Built-in colormap tables, exported so tests/python/test_colormap.py can pin the
    // C++ table against its `_colormap.py` twin byte for byte (the two copies
    // are both emitted by tools/gen_colormaps.py).
    m.def(
        "colormap_table",
        [](const std::string& name) {
            const std::uint8_t* table = meshioplusplus::detail::colormap_table(name);
            return py::bytes(reinterpret_cast<const char*>(table),
                             meshioplusplus::detail::kColormapSize * 3);
        },
        py::arg("name"));
    m.def("colormap_names", []() { return meshioplusplus::detail::colormap_names(); });

    // SVG writer (write-only visualization; 3D input renders the projected
    // skin — camera args appended AFTER the existing ones, the Python shim
    // passes everything positionally).
    m.def(
        "svg_write",
        [](const std::string& path, py::object pymesh, const std::string& float_fmt,
           const std::optional<std::string>& stroke_width, const std::optional<double>& image_width,
           const std::string& fill, const std::string& stroke, double azimuth, double elevation,
           double roll, const std::string& color_by, const std::optional<int>& component,
           const std::string& cmap, const std::optional<double>& vmin,
           const std::optional<double>& vmax, const std::string& nan_color, bool colorbar) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
            meshioplusplus::write_svg(path, cpp, float_fmt, stroke_width, image_width, fill, stroke,
                                      azimuth, elevation, roll, color_by, component, cmap, vmin,
                                      vmax, nan_color, colorbar);
        },
        py::arg("path"), py::arg("mesh"), py::arg("float_fmt") = ".3f",
        py::arg("stroke_width") = std::nullopt, py::arg("image_width") = 100.0,
        py::arg("fill") = "#c8c5bd", py::arg("stroke") = "#000080", py::arg("azimuth") = 45.0,
        py::arg("elevation") = 35.264389682754654, py::arg("roll") = 0.0, py::arg("color_by") = "",
        py::arg("component") = std::nullopt, py::arg("cmap") = "viridis",
        py::arg("vmin") = std::nullopt, py::arg("vmax") = std::nullopt,
        py::arg("nan_color") = "#808080", py::arg("colorbar") = false);

    // TikZ writer (write-only LaTeX visualization; 3D input renders the
    // projected skin — camera args appended AFTER the existing ones).
    m.def(
        "tikz_write",
        [](const std::string& path, py::object pymesh, const std::string& float_fmt,
           bool standalone, const std::optional<std::string>& line_width, const std::string& fill,
           const std::string& draw, const std::optional<double>& scale, double azimuth,
           double elevation, double roll, const std::string& color_by,
           const std::optional<int>& component, const std::string& cmap,
           const std::optional<double>& vmin, const std::optional<double>& vmax,
           const std::string& nan_color, bool colorbar) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
            meshioplusplus::write_tikz(path, cpp, float_fmt, standalone, line_width, fill, draw,
                                       scale, azimuth, elevation, roll, color_by, component, cmap,
                                       vmin, vmax, nan_color, colorbar);
        },
        py::arg("path"), py::arg("mesh"), py::arg("float_fmt") = ".6f",
        py::arg("standalone") = true, py::arg("line_width") = std::nullopt,
        py::arg("fill") = "gray!30", py::arg("draw") = "black", py::arg("scale") = std::nullopt,
        py::arg("azimuth") = 45.0, py::arg("elevation") = 35.264389682754654, py::arg("roll") = 0.0,
        py::arg("color_by") = "", py::arg("component") = std::nullopt, py::arg("cmap") = "viridis",
        py::arg("vmin") = std::nullopt, py::arg("vmax") = std::nullopt,
        py::arg("nan_color") = "gray", py::arg("colorbar") = false);

    // PERMAS writer / reader (.post/.dato).
    m.def("permas_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_permas(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("permas_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_permas(path));
    });

    // FLAC3D writer / reader (.f3grid, ascii + binary, common path).
    m.def("flac3d_write", [](const std::string& path, py::object pymesh,
                             const std::string& float_fmt, bool binary) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
        meshioplusplus::write_flac3d(path, cpp, float_fmt, binary);
    });
    m.def("flac3d_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_flac3d(path));
    });

    // FLUX .pf3 writer / reader.
    m.def("flux_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_flux(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("flux_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_flux(path));
    });

    // Modulef Formatted Field (.mff), FLUX field (.dex), ANSYS Fluent
    // interpolation (.ip) -- field-only formats read/written as geometry-less
    // meshes (point_data carried by the normal conversion layer).
    m.def("mff_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_mff(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("mff_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_mff(path));
    });
    m.def("dex_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_dex(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("dex_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_dex(path));
    });
    m.def("ip_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_ip(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("ip_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_ip(path));
    });

    // COMSOL .mphtxt writer / reader.
    m.def("mphtxt_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_mphtxt(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("mphtxt_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_mphtxt(path));
    });

    // FreeFem++ writer / reader (.msh).
    m.def("freefem_write", [](const std::string& path, py::object pymesh) {
        meshioplusplus_py::PyMeshRefs refs;
        meshioplusplus::write_freefem(path, meshioplusplus_py::py_to_mesh(pymesh, refs));
    });
    m.def("freefem_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_freefem(path));
    });

    // GiD postprocess writer, on a vendored gidpost (write-only -- gidpost
    // has no read functions at all). `mode` is one of "auto"/"ascii"/
    // "binary"/"hdf5", the flat-binding spelling of GidMode.
    m.def(
        "gid_write",
        [](const std::string& path, py::object pymesh, const std::string& mode,
           const std::string& analysis_name, double step) {
            meshioplusplus_py::PyMeshRefs refs;
            meshioplusplus::write_gid(path, meshioplusplus_py::py_to_mesh(pymesh, refs),
                                      meshioplusplus::gid_mode_from_name(mode), analysis_name,
                                      step);
        },
        py::arg("path"), py::arg("mesh"), py::arg("mode") = "auto",
        py::arg("analysis_name") = "meshio++", py::arg("step") = 1.0);
    // GiD multi-step writer. Python's sequence engine (`_sequence.py`) is a
    // separate pure-Python implementation, not a shim over the C++ one, so it
    // cannot reach `sequence_to_timeseries`' own gid branch and needs this.
    // `next_step()` returns `(time, mesh)` or None when exhausted -- pull, not
    // push, so a generator drives it and only one mesh is ever alive.
    m.def(
        "gid_write_series",
        [](const std::string& path, const py::function& next_step, const std::string& mode,
           const std::string& analysis_name) {
            meshioplusplus::write_gid_series(
                path,
                [&](std::size_t, double& time, meshioplusplus::Mesh& mesh) {
                    py::object item = next_step();
                    if (item.is_none())
                        return false;
                    auto pair = item.cast<py::tuple>();
                    time = pair[0].cast<double>();
                    // The refs must outlive the conversion only until the mesh
                    // is written, which happens before the next pull.
                    meshioplusplus_py::PyMeshRefs refs;
                    mesh = meshioplusplus_py::py_to_mesh(pair[1], refs);
                    return true;
                },
                meshioplusplus::gid_mode_from_name(mode), analysis_name);
        },
        py::arg("path"), py::arg("next_step"), py::arg("mode") = "auto",
        py::arg("analysis_name") = "meshio++");
    // GiD postprocess reader. Needs no gidpost (that library is write-only),
    // so it is available in builds that cannot write the format at all.
    m.def(
        "gid_read",
        [](const std::string& path, int time_step) {
            meshioplusplus::ReadOptions opts;
            opts.mTimeStep = time_step;
            return meshioplusplus_py::mesh_to_py(meshioplusplus::read_gid(path, opts));
        },
        py::arg("path"), py::arg("time_step") = 0);

    // MFM (Modulef Formatted Mesh) writer / reader (.mfm).
    m.def("mfm_write",
          [](const std::string& path, py::object pymesh, const std::string& float_fmt) {
              meshioplusplus_py::PyMeshRefs refs;
              meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
              meshioplusplus::write_mfm(path, cpp, float_fmt);
          });
    m.def("mfm_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_mfm(path));
    });

    // Netgen writer / reader (.vol, common path).
    m.def("netgen_write",
          [](const std::string& path, py::object pymesh, const std::string& float_fmt) {
              meshioplusplus_py::PyMeshRefs refs;
              meshioplusplus::Mesh cpp = meshioplusplus_py::py_to_mesh(pymesh, refs);
              meshioplusplus::write_netgen(path, cpp, float_fmt);
          });
    m.def("netgen_read", [](const std::string& path) {
        return meshioplusplus_py::mesh_to_py(meshioplusplus::read_netgen(path));
    });

    // Provenance bridge (v10.16.0): keeps Python's `_provenance.scope` and the
    // C++ writers' `detail::ProvenanceScope` reading from the SAME thread-local
    // state, so a scope opened from Python is honoured by every `_core.<fmt>_write`
    // call -- without this, opting in from Python would silently produce Off-mode
    // output for every C++-backed writer (~40 of the 44 formats), since C++'s own
    // thread-local state is otherwise never told Python opened anything.
    //
    // A thread-local stack of owned scopes on this side mirrors Python's own list-
    // based stack one-for-one: `push` constructs and keeps one alive, `pop`
    // destroys the most recently pushed one (LIFO), which is what makes nesting
    // work identically to a plain `ProvenanceScope` used directly from C++.
    m.def("provenance_scope_push", [](int mode, py::dict record) {
        using meshioplusplus::detail::ProvenanceMode;
        using meshioplusplus::detail::ProvenanceNote;
        using meshioplusplus::detail::ProvenanceRecord;
        ProvenanceRecord rec;
        auto get_str = [&](const char* key) -> std::string {
            return record.contains(key) ? py::cast<std::string>(record[key]) : std::string();
        };
        rec.mSourcePath = get_str("source_path");
        rec.mSourceFormat = get_str("source_format");
        rec.mTargetFormat = get_str("target_format");
        rec.mEncoding = get_str("encoding");
        rec.mCodec = get_str("codec");
        rec.mFloatFormat = get_str("float_format");
        rec.mTimestamp = get_str("timestamp");
        if (record.contains("operations"))
            for (auto op : record["operations"])
                rec.mOperations.push_back(py::cast<std::string>(op));
        if (record.contains("notes"))
            for (auto n : record["notes"]) {
                auto pair = py::cast<py::tuple>(n);
                rec.mNotes.push_back(
                    ProvenanceNote{py::cast<std::string>(pair[0]), py::cast<std::string>(pair[1])});
            }
        meshioplusplus_py::provenance_scope_stack().push_back(
            std::make_unique<meshioplusplus::detail::ProvenanceScope>(
                static_cast<ProvenanceMode>(mode), std::move(rec)));
    });
    m.def("provenance_scope_pop", []() {
        auto& stack = meshioplusplus_py::provenance_scope_stack();
        if (!stack.empty())
            stack.pop_back();
    });
    m.def("provenance_note", [](const std::string& category, const std::string& detail) {
        meshioplusplus::detail::provenance_note(category, detail);
    });
    m.def("provenance_set_source", [](const std::string& path, const std::string& format) {
        meshioplusplus::detail::provenance_set_source(path, format);
    });
    m.def("provenance_set_target", [](const std::string& format, const std::string& encoding,
                                      const std::string& codec, const std::string& float_format) {
        meshioplusplus::detail::provenance_set_target(format, encoding, codec, float_format);
    });
    m.def("provenance_add_operation", [](const std::string& rendered) {
        meshioplusplus::detail::provenance_add_operation(rendered);
    });
    m.def("provenance_current_mode",
          []() { return static_cast<int>(meshioplusplus::detail::current_provenance_mode()); });
    m.def("provenance_begin_write", []() { meshioplusplus::detail::provenance_begin_write(); });
    m.def("provenance_default_mode",
          []() { return static_cast<int>(meshioplusplus::detail::default_provenance_mode()); });
    m.def("provenance_set_default_mode", [](int mode) {
        if (mode < 0 || mode > 2)
            throw std::invalid_argument("meshio++: unknown provenance mode " +
                                        std::to_string(mode));
        meshioplusplus::detail::set_default_provenance_mode(
            static_cast<meshioplusplus::detail::ProvenanceMode>(mode));
    });
    m.def("provenance_current_record", []() {
        const auto& rec = meshioplusplus::detail::current_provenance();
        py::dict out;
        out["source_path"] = rec.mSourcePath;
        out["source_format"] = rec.mSourceFormat;
        out["target_format"] = rec.mTargetFormat;
        out["encoding"] = rec.mEncoding;
        out["codec"] = rec.mCodec;
        out["float_format"] = rec.mFloatFormat;
        out["timestamp"] = rec.mTimestamp;
        out["operations"] = rec.mOperations;
        py::list notes;
        for (const auto& n : rec.mNotes)
            notes.append(py::make_tuple(n.mCategory, n.mDetail));
        out["notes"] = notes;
        return out;
    });
    m.def(
        "read_provenance_lines",
        [](const std::string& path, std::size_t max_bytes) {
            auto r = meshioplusplus::detail::read_provenance_lines(path, max_bytes);
            return py::make_tuple(r.mLines, r.mRecognised);
        },
        py::arg("path"), py::arg("max_bytes") = 8192);
    m.def("provenance_lines", [](int tier) {
        return meshioplusplus::detail::provenance_lines(
            static_cast<meshioplusplus::detail::SlotTier>(tier));
    });
}
