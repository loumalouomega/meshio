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
#pragma once

/**
 * @file mesh_fixtures.hpp
 * @brief Fixture-mesh builders and round-trip helpers shared by the C++ unit
 *        test suite (`tests/cpp/test_*.cpp`).
 *
 * This is the C++ analogue of Python's `tests/python/helpers.py`: rather than each
 * `test_<format>.cpp` hand-rolling its own small meshes, it builds a
 * `meshioplusplus::Mesh` fixture (e.g. `mt::tri_mesh()`, `mt::tet_mesh()`,
 * `mt::hex_mesh()`) once here and every format's test file exercises the
 * same handful of geometries against `mt::roundtrip()`, which writes a
 * mesh to a temp file with a format's writer, reads it back with the
 * matching reader, and asserts the two meshes agree (`mt::expect_mesh_eq`).
 *
 * Everything in this header lives in the `mt` namespace (short for "mesh
 * test") to keep it out of the way of the `meshioplusplus` production
 * namespace it pulls types from (`Mesh`, `NDArray`, `DType`). Meshes are
 * built and inspected exclusively through the uniform format-facing API
 * (see `mesh_api.hpp`) so the whole suite compiles under every mesh
 * backend.
 */

// System includes
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

// External includes
#include <gtest/gtest.h>

// Project includes
#include "meshioplusplus/detail/value_io.hpp"
#include "meshioplusplus/mesh.hpp"

namespace mt {

using meshioplusplus::DType;
using meshioplusplus::Mesh;
using meshioplusplus::NDArray;

// ---- builders ----
//
// Low-level helpers (`points_from`, `conn_from`, `make_mesh`) that turn
// nested `std::vector` literals into `meshioplusplus::NDArray`/`Mesh`
// objects, followed by a set of named single-cell-type fixture meshes
// (`line_mesh`, `tri_mesh`, `tet_mesh`, `hex_mesh`, ...) that mirror the
// fixtures in Python's `tests/python/helpers.py`. Each fixture is a small,
// hand-picked, geometrically valid mesh (consistent winding / positive
// volume for solid cells) meant to be fed straight into `roundtrip()` by a
// format's test file.

/**
 * @brief Build a `Float64` `NDArray` of point coordinates from a nested vector.
 *
 * @param rPts One row per point, each an equal-length sequence of
 *            coordinates (2 for 2-D fixtures, 3 for 3-D ones). If empty,
 *            the resulting array has 0 rows and a default dimensionality
 *            of 3.
 * @return An owning `NDArray` of shape `{rPts.size(), rPts[0].size()}` and
 *         dtype `Float64`, laid out row-major matching `rPts`.
 */
inline NDArray points_from(const std::vector<std::vector<double>>& rPts) {
    std::size_t n = rPts.size();
    std::size_t d = n ? rPts[0].size() : 3;
    NDArray a(DType::Float64, {n, d});
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < d; ++j)
            a.As<double>()[i * d + j] = rPts[i][j];
    return a;
}

/**
 * @brief Build an `Int64` `NDArray` of cell connectivity from a nested vector.
 *
 * @param rRows One row per cell, each an equal-length sequence of point
 *             indices (node count per cell for the target cell type). If
 *             empty, the resulting array has 0 rows and 0 columns.
 * @return An owning `NDArray` of shape `{rRows.size(), rRows[0].size()}` and
 *         dtype `Int64`, laid out row-major matching `rRows`.
 */
inline NDArray conn_from(const std::vector<std::vector<std::int64_t>>& rRows) {
    std::size_t n = rRows.size();
    std::size_t k = n ? rRows[0].size() : 0;
    NDArray a(DType::Int64, {n, k});
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < k; ++j)
            a.As<std::int64_t>()[i * k + j] = rRows[i][j];
    return a;
}

/**
 * @brief Build a single-cell-block `Mesh` from point and connectivity literals.
 *
 * Convenience wrapper combining `points_from` and `conn_from`: assigns the
 * points and appends exactly one cell block of the given `type`. Used
 * by every single-cell-type fixture below (`tri_mesh`, `tet_mesh`, etc.);
 * multi-block fixtures (e.g. `tri_quad_mesh`) build the `Mesh` by hand
 * instead since they need more than one block.
 *
 * @param pts Point coordinates, as for `points_from`.
 * @param rType The meshio++ cell type name for the single block (e.g.
 *             `"triangle"`, `"tetra10"`).
 * @param cells Cell connectivity rows, as for `conn_from`.
 * @return A `Mesh` with points set and one cell block of `rType`.
 */
inline Mesh make_mesh(std::vector<std::vector<double>> pts, const std::string& rType,
                      std::vector<std::vector<std::int64_t>> cells) {
    Mesh m;
    m.AssignPoints(points_from(pts));
    m.AddCellBlock(rType, conn_from(cells));
    return m;
}

// Fixture meshes mirroring tests/python/helpers.py (geometry chosen to be valid,
// right-handed volume cells so FLAC3D's determinant reorder round-trips).

/**
 * @brief A 2-D planar fixture (4 corners of a unit square, in 3-D coordinates
 *        with z=0) with a single `"line"` block of 5 edges.
 * @return A single-block `line` `Mesh`.
 */
inline Mesh line_mesh() {
    return make_mesh({{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}, "line",
                     {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {2, 3}});
}
/**
 * @brief A unit-square fixture (3-D coordinates, z=0) split into 2
 *        `"triangle"` cells.
 * @return A single-block `triangle` `Mesh`.
 */
inline Mesh tri_mesh() {
    return make_mesh({{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}, "triangle",
                     {{0, 1, 2}, {0, 2, 3}});
}
/**
 * @brief Same geometry as `tri_mesh` but with genuinely 2-D points (no z
 *        coordinate), for exercising formats/paths that handle 2-D point
 *        arrays directly rather than always padding to 3-D.
 * @return A single-block `triangle` `Mesh` with 2-D points.
 */
inline Mesh tri_mesh_2d() {
    return make_mesh({{0, 0}, {1, 0}, {1, 1}, {0, 1}}, "triangle", {{0, 1, 2}, {0, 2, 3}});
}
/**
 * @brief A fixture with 2 `"quad"` cells over 6 points (3-D coordinates, z=0).
 * @return A single-block `quad` `Mesh`.
 */
inline Mesh quad_mesh() {
    return make_mesh({{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {2, 1, 0}, {1, 1, 0}, {0, 1, 0}}, "quad",
                     {{0, 1, 4, 5}, {1, 2, 3, 4}});
}
/**
 * @brief A fixture with 2 `"tetra"` cells (5 points, one raised out of the
 *        z=0 plane) chosen for a valid, right-handed (positive-volume)
 *        tetrahedron winding.
 * @return A single-block `tetra` `Mesh`.
 */
inline Mesh tet_mesh() {
    return make_mesh({{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0.5, 0.5, 0.5}}, "tetra",
                     {{0, 1, 2, 4}, {0, 2, 3, 4}});
}
/**
 * @brief A single unit-cube `"hexahedron"` cell (8 points).
 * @return A single-block `hexahedron` `Mesh`.
 */
inline Mesh hex_mesh() {
    return make_mesh(
        {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}},
        "hexahedron", {{0, 1, 2, 3, 4, 5, 6, 7}});
}
/**
 * @brief A single triangular-prism `"wedge"` cell (6 points).
 * @return A single-block `wedge` `Mesh`.
 */
inline Mesh wedge_mesh() {
    return make_mesh({{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}}, "wedge",
                     {{0, 1, 2, 3, 4, 5}});
}
/**
 * @brief A single 2nd-order `"triangle6"` cell: 3 corner points plus 3
 *        mid-edge points (perturbed off the exact edge midpoints so the
 *        fixture also exercises non-trivial mid-node coordinates).
 * @return A single-block `triangle6` `Mesh`.
 */
inline Mesh triangle6_mesh() {
    return make_mesh(
        {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0.5, 0.25, 0}, {1.25, 0.5, 0}, {0.25, 0.75, 0}},
        "triangle6", {{0, 1, 2, 3, 4, 5}});
}
/**
 * @brief A single 2nd-order `"quad8"` cell: 4 corner points plus 4 mid-edge
 *        points.
 * @return A single-block `quad8` `Mesh`.
 */
inline Mesh quad8_mesh() {
    return make_mesh({{0, 0, 0},
                      {1, 0, 0},
                      {1, 1, 0},
                      {0, 1, 0},
                      {0.5, 0.1, 0},
                      {0.9, 0.5, 0},
                      {0.5, 0.9, 0},
                      {0.1, 0.5, 0}},
                     "quad8", {{0, 1, 2, 3, 4, 5, 6, 7}});
}
/**
 * @brief A single 2nd-order `"tetra10"` cell with 10 procedurally generated
 *        (evenly spaced along a line, not geometrically meaningful) points -
 *        the fixture exists to exercise the node count/ordering of the
 *        format round-trip rather than a "real" tetrahedron shape.
 * @return A single-block `tetra10` `Mesh`.
 */
inline Mesh tet10_mesh() {
    std::vector<std::vector<double>> p;
    for (int i = 0; i < 10; ++i)
        p.push_back({0.1 * i, 0.2 * i + 0.05, 0.3 * i + 0.01});
    return make_mesh(p, "tetra10", {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}});
}
/**
 * @brief A single 2nd-order `"hexahedron20"` cell with 20 procedurally
 *        generated points, analogous to `tet10_mesh`.
 * @return A single-block `hexahedron20` `Mesh`.
 */
inline Mesh hex20_mesh() {
    std::vector<std::vector<double>> p;
    for (int i = 0; i < 20; ++i)
        p.push_back({0.11 * i, 0.07 * i, 0.03 * i});
    std::vector<std::int64_t> row(20);
    for (int i = 0; i < 20; ++i)
        row[i] = i;
    return make_mesh(p, "hexahedron20", {row});
}
/**
 * @brief A hybrid mesh with three cell blocks of mixed types (`triangle`,
 *        `quad`, `triangle`, in that order) sharing one point set, for
 *        exercising formats that must preserve multiple heterogeneous cell
 *        blocks rather than a single uniform one. Built by hand (rather
 *        than via `make_mesh`, which only supports a single block).
 * @return A 3-block (`triangle`, `quad`, `triangle`) `Mesh`.
 */
inline Mesh tri_quad_mesh() {
    Mesh m;
    m.AssignPoints(
        points_from({{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 1, 0}, {2, 1, 0}, {1, 1, 0}, {0, 1, 0}}));
    m.AddCellBlock("triangle", conn_from({{0, 1, 5}, {0, 5, 6}}));
    m.AddCellBlock("quad", conn_from({{1, 2, 4, 5}}));
    m.AddCellBlock("triangle", conn_from({{2, 3, 4}}));
    return m;
}

// ---- comparison / round-trip ----

/**
 * @brief Generate a unique path in the system temp directory for a round-trip
 *        test file.
 *
 * Uniqueness comes from a process-local, monotonically increasing atomic
 * counter (not from any filesystem check), so this only guarantees
 * distinct paths across calls within the same test binary run, which is
 * sufficient for `roundtrip()`'s write-then-read-then-delete usage.
 *
 * @param rSuffix File suffix/extension to append (e.g. `".vtk"`), including
 *               the leading dot if desired.
 * @return An absolute path of the form
 *         `<temp_dir>/meshio_cpp_<counter><suffix>`.
 */
inline std::string temp_path(const std::string& rSuffix) {
    static std::atomic<unsigned> counter{0};
    auto dir = std::filesystem::temp_directory_path();
    return (dir / ("meshio_cpp_" + std::to_string(counter++) + rSuffix)).string();
}

/**
 * @brief Flatten a `Mesh`'s cell blocks into `{type -> multiset of connectivity
 *        rows}` for order-insensitive comparison.
 *
 * Grouping by type into a `std::multiset` (rather than comparing
 * cell blocks block-by-block/row-by-row in original order) makes the
 * comparison robust to formats that split or merge same-type blocks on
 * read (e.g. a writer emitting two separate `"triangle"` blocks that a
 * reader recombines into one, or vice versa) - as long as the *set* of
 * rows for each type matches, block boundaries and row order don't matter.
 * Node order *within* a single cell's row is preserved and does matter (a
 * cell with permuted node order is a different row).
 *
 * @param rM The mesh to summarize.
 * @return A map from cell type name to a multiset of that type's
 *         connectivity rows (as `std::int64_t` vectors), pooled across all
 *         of `rM`'s blocks of that type.
 */
inline std::map<std::string, std::multiset<std::vector<std::int64_t>>> cell_rows(const Mesh& rM) {
    std::map<std::string, std::multiset<std::vector<std::int64_t>>> out;
    for (const auto cb : rM.CellRange()) {
        const NDArray& conn = cb.Conn();
        std::size_t n = cb.NumCells();
        std::size_t k = meshioplusplus::detail::cols(conn);
        for (std::size_t r = 0; r < n; ++r) {
            std::vector<std::int64_t> row(k);
            for (std::size_t j = 0; j < k; ++j)
                row[j] = meshioplusplus::detail::read_int(conn, r * k + j);
            out[cb.Type()].insert(std::move(row));
        }
    }
    return out;
}

/**
 * @brief GoogleTest assertion: check that two meshes' point coordinates agree
 *        within a tolerance, allowing for 2-D -> 3-D padding.
 *
 * Asserts equal point counts (`ASSERT_EQ`, fatal) and that `rOut`'s point
 * dimensionality is at least `rIn`'s (`ASSERT_GE`, fatal) - some formats
 * always write 3-D points even when the input was 2-D, padding with an
 * implicit zero z-coordinate, but never truncate 3-D down to 2-D. Only the
 * first `din` (the input's dimensionality) components of each point are
 * then compared (`EXPECT_NEAR`, non-fatal so all mismatches are reported).
 *
 * @param rIn The reference (pre-round-trip) mesh.
 * @param rOut The mesh produced by writing `rIn` and reading it back.
 * @param atol Absolute tolerance for the per-component `EXPECT_NEAR` check.
 */
inline void expect_points_close(const Mesh& rIn, const Mesh& rOut, double atol) {
    ASSERT_EQ(rIn.NumPoints(), rOut.NumPoints());
    const NDArray& pin = rIn.Points();
    const NDArray& pout = rOut.Points();
    std::size_t din = meshioplusplus::detail::cols(pin);
    std::size_t dout = meshioplusplus::detail::cols(pout);
    ASSERT_GE(dout, din);  // formats may pad 2D -> 3D, never truncate
    for (std::size_t i = 0; i < rIn.NumPoints(); ++i)
        for (std::size_t j = 0; j < din; ++j) {
            double a = meshioplusplus::detail::read_double(pin, i * din + j);
            double b = meshioplusplus::detail::read_double(pout, i * dout + j);
            EXPECT_NEAR(a, b, atol) << "point " << i << " comp " << j;
        }
}

/**
 * @brief GoogleTest assertion: check that two meshes are equivalent for
 *        round-trip purposes (points within tolerance, cells identical as
 *        sets per type).
 *
 * Combines `expect_points_close` (point coordinates) with an `EXPECT_EQ`
 * over `cell_rows` (connectivity, compared per-type as multisets so block
 * splitting/merging and cross-block reordering don't cause false
 * failures). This is the top-level check every `roundtrip()` call ends
 * with.
 *
 * @param rIn The reference (pre-round-trip) mesh.
 * @param rOut The mesh produced by writing `rIn` and reading it back.
 * @param atol Absolute tolerance forwarded to `expect_points_close`
 *             (default `1e-12`; format-specific tests widen it for
 *             formats with lossy/lower-precision storage).
 */
inline void expect_mesh_eq(const Mesh& rIn, const Mesh& rOut, double atol = 1e-12) {
    expect_points_close(rIn, rOut, atol);
    EXPECT_EQ(cell_rows(rIn), cell_rows(rOut));
}

/** @brief A format writer: `(path, mesh) -> void`, writes `mesh` to `path`. */
using Writer = std::function<void(const std::string&, const Mesh&)>;
/** @brief A format reader: `(path) -> Mesh`, reads and returns the mesh at `path`. */
using Reader = std::function<Mesh(const std::string&)>;

/**
 * @brief Write -> read -> compare, then remove the temp file(s); the standard
 *        per-format-test round-trip pattern.
 *
 * Generates a temp path (`temp_path(suffix)`), calls `rWriter(path, rMesh)`,
 * then `rReader(path)`, and asserts the result matches `rMesh` via
 * `expect_mesh_eq`. The temp file is removed afterward regardless of
 * whether the comparison assertions passed (`std::filesystem::remove` with
 * an `std::error_code` overload, so a failed cleanup does not itself throw
 * or fail the test).
 *
 * Typical call site (see `tests/cpp/test_vtk.cpp`):
 * @code
 * mt::roundtrip(
 *     [=](const std::string& p, const mt::Mesh& m) {
 *         meshioplusplus::write_vtk(p, m, binary, v51);
 *     },
 *     [](const std::string& p) { return meshioplusplus::read_vtk(p); },
 *     mt::tri_mesh(), ".vtk");
 * @endcode
 *
 * @param rWriter Callable writing a `Mesh` to a file path.
 * @param rReader Callable reading a `Mesh` back from a file path.
 * @param rMesh The fixture mesh to round-trip (e.g. `mt::tri_mesh()`).
 * @param rSuffix File suffix/extension for the generated temp path (e.g.
 *               `".vtk"`).
 * @param atol Absolute point-coordinate tolerance forwarded to
 *             `expect_mesh_eq` (default `1e-12`).
 */
inline void roundtrip(const Writer& rWriter, const Reader& rReader, const Mesh& rMesh,
                      const std::string& rSuffix, double atol = 1e-12) {
    std::string path = temp_path(rSuffix);
    rWriter(path, rMesh);
    Mesh out = rReader(path);
    expect_mesh_eq(rMesh, out, atol);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// ---------------------------------------------------------------------------
// Fixtures for the data operations
// ---------------------------------------------------------------------------

/**
 * @brief A `Float64` array of @p rValues laid out as `(rows, cols)`, or
 * `(rows,)` when @p cols is 1 (the "scalar data stays 1-D" convention).
 * @param rValues Row-major values.
 * @param cols Component count.
 * @return The array.
 */
inline NDArray data_array(const std::vector<double>& rValues, std::size_t cols = 1) {
    const std::size_t rows = cols > 0 ? rValues.size() / cols : 0;
    NDArray a(DType::Float64,
              cols <= 1 ? std::vector<std::size_t>{rows} : std::vector<std::size_t>{rows, cols});
    double* p = a.As<double>();
    for (std::size_t i = 0; i < rValues.size(); ++i)
        p[i] = rValues[i];
    return a;
}

/**
 * @brief An `Int32` scalar array — used to check that dtype *kind* survives an
 * operation. Note that the NATIVE and KRATOS backends canonicalize integers to
 * `Int64` on ingest, so tests must assert the kind (`detail::is_float_dtype`)
 * rather than the exact width.
 * @param rValues The values.
 * @return The array.
 */
inline NDArray int_data_array(const std::vector<std::int32_t>& rValues) {
    NDArray a(DType::Int32, {rValues.size()});
    std::int32_t* p = a.As<std::int32_t>();
    for (std::size_t i = 0; i < rValues.size(); ++i)
        p[i] = rValues[i];
    return a;
}

/**
 * @brief Two unit squares side by side, split into a `triangle` block (2 cells)
 * and a `quad` block (1 cell), carrying point, cell and field data.
 *
 * Point layout (all z = 0):
 * @code
 *   3---2---5
 *   |  /|   |
 *   | / |   |
 *   0---1---4
 * @endcode
 * `triangle` = {0,1,2} and {0,2,3}; `quad` = {1,4,5,2}.
 *
 * Data carried: `point_data` `T` (scalar, = x + 10*y) and `v` (3-vector);
 * `cell_data` `mat` (scalar, per block) and `tag` (`Int32` scalar);
 * `field_data` `meta`.
 * @return The fixture mesh.
 */
inline Mesh data_mesh() {
    Mesh m;
    m.AssignPoints(points_from({{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {2, 0, 0}, {2, 1, 0}}));
    m.AddCellBlock("triangle", conn_from({{0, 1, 2}, {0, 2, 3}}));
    m.AddCellBlock("quad", conn_from({{1, 4, 5, 2}}));

    // T = x + 10*y at each of the six points.
    m.AddPointData("T", data_array({0.0, 1.0, 11.0, 10.0, 2.0, 12.0}));
    m.AddPointData("v", data_array({1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 2, 0, 0, 0, 2, 0}, 3));

    std::vector<NDArray> mat;
    mat.push_back(data_array({1.0, 2.0}));  // triangle block: 2 cells
    mat.push_back(data_array({3.0}));       // quad block: 1 cell
    m.AddCellData("mat", std::move(mat));

    std::vector<NDArray> tag;
    tag.push_back(int_data_array({10, 20}));
    tag.push_back(int_data_array({30}));
    m.AddCellData("tag", std::move(tag));

    m.AddFieldData("meta", data_array({1.0, 2.0, 3.0}));
    return m;
}

/**
 * @brief Asserts that two meshes have identical geometry — same points, same
 * block count, and per block the same type, cell count and connectivity.
 *
 * Every data operation must leave geometry bit-identical, so this is the shared
 * invariant check for all of them.
 * @param rA First mesh.
 * @param rB Second mesh.
 */
inline void expect_same_geometry(const Mesh& rA, const Mesh& rB) {
    ASSERT_EQ(rA.NumPoints(), rB.NumPoints());
    ASSERT_EQ(rA.PointDim(), rB.PointDim());
    ASSERT_EQ(rA.Points().Size(), rB.Points().Size());
    for (std::size_t i = 0; i < rA.Points().Size(); ++i)
        EXPECT_DOUBLE_EQ(meshioplusplus::detail::read_double(rA.Points(), i),
                         meshioplusplus::detail::read_double(rB.Points(), i));
    ASSERT_EQ(rA.NumCellBlocks(), rB.NumCellBlocks());
    for (std::size_t b = 0; b < rA.NumCellBlocks(); ++b) {
        const auto ca = rA.Cells(b);
        const auto cb = rB.Cells(b);
        EXPECT_EQ(std::string(ca.Type()), std::string(cb.Type()));
        ASSERT_EQ(ca.NumCells(), cb.NumCells());
        ASSERT_EQ(ca.IsRagged(), cb.IsRagged());
        if (ca.IsRagged() || ca.IsPolyhedron())
            continue;
        ASSERT_EQ(ca.Conn().Size(), cb.Conn().Size());
        for (std::size_t i = 0; i < ca.Conn().Size(); ++i)
            EXPECT_EQ(meshioplusplus::detail::read_int(ca.Conn(), i),
                      meshioplusplus::detail::read_int(cb.Conn(), i));
    }
}

// --- surface fixtures shared by the curvature/repair/shrinkwrap suites -------
inline constexpr double mt_kPi = 3.141592653589793238462643383279;

/// A closed icosahedron of radius R, then `Subdivisions` rounds of 1-to-4
/// splitting with every new point pushed back onto the sphere.
///
/// meshio++ has no `sphere` primitive (roadmap section 6), and the analytic
/// answers on a sphere -- H = 1/R and K = 1/R^2 at every vertex -- are the
/// only exact per-vertex oracle either estimator has.
inline Mesh icosphere(int Subdivisions, double Radius) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    std::vector<std::vector<double>> p = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0},
                                          {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
                                          {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
    std::vector<std::vector<std::int64_t>> f = {
        {0, 11, 5},  {0, 5, 1},  {0, 1, 7},  {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
        {11, 10, 2}, {10, 7, 6}, {7, 1, 8},  {3, 9, 4},  {3, 4, 2},   {3, 2, 6}, {3, 6, 8},
        {3, 8, 9},   {4, 9, 5},  {2, 4, 11}, {6, 2, 10}, {8, 6, 7},   {9, 8, 1}};

    for (int s = 0; s < Subdivisions; ++s) {
        std::map<std::pair<std::int64_t, std::int64_t>, std::int64_t> mid;
        std::vector<std::vector<std::int64_t>> out;
        auto midpoint = [&](std::int64_t a, std::int64_t b) {
            const auto key = std::make_pair(std::min(a, b), std::max(a, b));
            auto it = mid.find(key);
            if (it != mid.end())
                return it->second;
            const std::int64_t id = static_cast<std::int64_t>(p.size());
            p.push_back(
                {0.5 * (p[static_cast<std::size_t>(a)][0] + p[static_cast<std::size_t>(b)][0]),
                 0.5 * (p[static_cast<std::size_t>(a)][1] + p[static_cast<std::size_t>(b)][1]),
                 0.5 * (p[static_cast<std::size_t>(a)][2] + p[static_cast<std::size_t>(b)][2])});
            mid.emplace(key, id);
            return id;
        };
        for (const auto& tri : f) {
            const std::int64_t ab = midpoint(tri[0], tri[1]);
            const std::int64_t bc = midpoint(tri[1], tri[2]);
            const std::int64_t ca = midpoint(tri[2], tri[0]);
            out.push_back({tri[0], ab, ca});
            out.push_back({tri[1], bc, ab});
            out.push_back({tri[2], ca, bc});
            out.push_back({ab, bc, ca});
        }
        f = out;
    }
    for (auto& v : p) {
        const double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        for (auto& x : v)
            x *= Radius / n;
    }
    return make_mesh(p, "triangle", f);
}

/// An OPEN cylinder of radius R and height H: `Around` columns, `Along` rows,
/// no caps. The one fixture that separates the two estimators -- K is exactly
/// 0 everywhere (a cylinder is developable) while H is 1/(2R), non-zero.
inline Mesh open_cylinder(int Around, int Along, double Radius, double Height) {
    std::vector<std::vector<double>> p;
    for (int j = 0; j <= Along; ++j) {
        const double z = Height * (static_cast<double>(j) / Along - 0.5);
        for (int i = 0; i < Around; ++i) {
            const double a = 2.0 * mt_kPi * static_cast<double>(i) / Around;
            p.push_back({Radius * std::cos(a), Radius * std::sin(a), z});
        }
    }
    auto id = [&](int j, int i) { return static_cast<std::int64_t>(j * Around + (i % Around)); };
    std::vector<std::vector<std::int64_t>> f;
    for (int j = 0; j < Along; ++j)
        for (int i = 0; i < Around; ++i) {
            f.push_back({id(j, i), id(j, i + 1), id(j + 1, i + 1)});
            f.push_back({id(j, i), id(j + 1, i + 1), id(j + 1, i)});
        }
    return make_mesh(p, "triangle", f);
}

/// A flat triangulated square in the z = 0 plane, deliberately with an
/// irregular diagonal pattern so a "flat" answer cannot come from symmetry.
inline Mesh plane_grid(int N) {
    std::vector<std::vector<double>> p;
    for (int j = 0; j <= N; ++j)
        for (int i = 0; i <= N; ++i)
            p.push_back({static_cast<double>(i), static_cast<double>(j), 0.0});
    auto id = [&](int i, int j) { return static_cast<std::int64_t>(j * (N + 1) + i); };
    std::vector<std::vector<std::int64_t>> f;
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            if ((i + j) % 2 == 0) {
                f.push_back({id(i, j), id(i + 1, j), id(i + 1, j + 1)});
                f.push_back({id(i, j), id(i + 1, j + 1), id(i, j + 1)});
            } else {
                f.push_back({id(i, j), id(i + 1, j), id(i, j + 1)});
                f.push_back({id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)});
            }
        }
    return make_mesh(p, "triangle", f);
}

}  // namespace mt
