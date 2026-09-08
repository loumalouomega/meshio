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
 * @file test_c_api.cpp
 * @brief Tests for the C API (bindings/c/). Compiled into the gtest suite
 *        only when MESHIOPLUSPLUS_BUILD_C_API=ON; written purely against the
 *        public C surface (plus mt:: fixtures for reference meshes), so it
 *        runs identically under every mesh backend.
 */

// System includes
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// External includes
#include <gtest/gtest.h>

#include "meshioplusplus/version.hpp"

// Project includes
#include "meshioplusplus/meshioplusplus.h"

#include "mesh_fixtures.hpp"
#ifdef MESHIOPLUSPLUS_HAS_HDF5
#include "meshioplusplus/formats/med.hpp"
#include "meshioplusplus/formats/stl.hpp"
#include "meshioplusplus/formats/vtu.hpp"
#endif

namespace {

// Deliberately non-square everywhere (5 points x 3 dims, 2 cells x 4 nodes)
// with asymmetric coordinates so a transposed layout cannot cancel out.
const std::vector<double> kPoints = {
    0.0, 0.0, 0.0,  //
    1.1, 0.2, 0.3,  //
    0.4, 1.2, 0.5,  //
    0.6, 0.7, 1.3,  //
    1.4, 1.5, 1.6,  //
};
const std::vector<std::int64_t> kConn = {0, 1, 2, 3, 1, 2, 3, 4};

// Build the reference tet mesh through the C API.
mio_mesh* build_tet_mesh() {
    mio_mesh* m = mio_mesh_create();
    EXPECT_NE(m, nullptr);
    EXPECT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 5, 3, kPoints.data()), MIO_OK);
    EXPECT_EQ(mio_mesh_add_cell_block(m, "tetra", 2, 4, MIO_INT64, kConn.data()), MIO_OK);
    return m;
}

std::string block_type(const mio_mesh* pMesh, std::int64_t block) {
    char buf[64] = {};
    const std::int64_t n = mio_mesh_cell_block_type(pMesh, block, buf, sizeof(buf));
    EXPECT_GE(n, 0);
    return buf;
}

TEST(CApi, VersionAndBackend) {
    EXPECT_STRNE(mio_version(), "");
    const std::string backend = mio_mesh_backend();
    EXPECT_TRUE(backend == "meshio" || backend == "native" || backend == "kratos") << backend;
}

// The compile-time macros describe the HEADER; mio_version() describes the
// linked LIBRARY. With a shared build the two genuinely can differ, which is why
// both exist -- but in this build they are the same artifact, so they must agree,
// and that is what catches a version file bumped without its twin.
TEST(CApi, CompileTimeVersionMatchesTheLinkedLibrary) {
    const std::string expected = std::to_string(MIO_VERSION_MAJOR) + "." +
                                 std::to_string(MIO_VERSION_MINOR) + "." +
                                 std::to_string(MIO_VERSION_PATCH);
    EXPECT_EQ(std::string(mio_version()), expected);
    EXPECT_EQ(std::string(MESHIOPLUSPLUS_VERSION_STRING), expected);

    // The feature-detection macros are what consumers actually write.
    EXPECT_TRUE(MIO_VERSION_AT_LEAST(MIO_VERSION_MAJOR, MIO_VERSION_MINOR, MIO_VERSION_PATCH));
    EXPECT_FALSE(MIO_VERSION_BEFORE(MIO_VERSION_MAJOR, MIO_VERSION_MINOR, MIO_VERSION_PATCH));
    EXPECT_TRUE(MIO_VERSION_AT_LEAST(1, 0, 0));
    EXPECT_FALSE(MIO_VERSION_AT_LEAST(MIO_VERSION_MAJOR + 1, 0, 0));
    // Ordering must not break across a component boundary: 9.6.0 is after 9.5.99.
    EXPECT_TRUE(MESHIOPLUSPLUS_VERSION_AT_LEAST(9, 5, 0));
    EXPECT_TRUE(MESHIOPLUSPLUS_VERSION > (9 * 10000 + 5 * 100 + 99));

    // selective refinement landed in 9.5.0; this is the shape a consumer writes.
#if MIO_VERSION_AT_LEAST(9, 5, 0)
    EXPECT_NE(&mio_refine_ex, nullptr);
#endif
}

TEST(CApi, FormatAvailability) {
    EXPECT_EQ(mio_format_readable("vtu"), 1);
    EXPECT_EQ(mio_format_writable("vtu"), 1);
    EXPECT_EQ(mio_format_readable("openfoam"), 1);
    EXPECT_EQ(mio_format_writable("openfoam"), 1);  // writable since v9.20.0
    EXPECT_EQ(mio_format_readable("nonexistent"), 0);
    EXPECT_EQ(mio_format_readable(nullptr), 0);
#ifdef MESHIOPLUSPLUS_HAS_HDF5
    EXPECT_EQ(mio_format_readable("med"), 1);
    EXPECT_EQ(mio_format_writable("med"), 1);
#else
    EXPECT_EQ(mio_format_readable("med"), 0);
#endif
}

TEST(CApi, CellTypeMetadata) {
    EXPECT_STREQ(mio_cell_type_name(MIO_CELL_Tetra10), "tetra10");
    EXPECT_EQ(mio_cell_type_from_name("tetra10"), MIO_CELL_Tetra10);
    EXPECT_EQ(mio_cell_type_from_name("not_a_type"), MIO_CELL_Custom);
    EXPECT_EQ(mio_cell_type_from_name(nullptr), MIO_CELL_Custom);
    EXPECT_EQ(mio_cell_type_num_nodes(MIO_CELL_Hexahedron20), 20);
    EXPECT_EQ(mio_cell_type_num_nodes(MIO_CELL_Polygon), -1);
    EXPECT_EQ(mio_cell_type_dimension(MIO_CELL_Triangle), 2);
    EXPECT_EQ(mio_cell_type_dimension(MIO_CELL_Custom), -1);
    EXPECT_STREQ(mio_cell_type_name(MIO_CELL_Custom), "");
}

TEST(CApi, BuildAndInspect) {
    mio_mesh* m = build_tet_mesh();

    EXPECT_EQ(mio_mesh_num_points(m), 5);
    EXPECT_EQ(mio_mesh_point_dim(m), 3);
    EXPECT_EQ(mio_mesh_num_cell_blocks(m), 1);
    EXPECT_EQ(block_type(m, 0), "tetra");

    std::int64_t num_cells = 0, npc = 0;
    std::int32_t ragged = -1;
    ASSERT_EQ(mio_mesh_cell_block_info(m, 0, &num_cells, &npc, &ragged), MIO_OK);
    EXPECT_EQ(num_cells, 2);
    EXPECT_EQ(npc, 4);
    EXPECT_EQ(ragged, 0);

    const void* pts = nullptr;
    mio_dtype dt = MIO_FLOAT32;
    ASSERT_EQ(mio_mesh_get_points(m, &pts, &dt), MIO_OK);
    ASSERT_EQ(dt, MIO_FLOAT64);
    const double* d = static_cast<const double*>(pts);
    for (std::size_t i = 0; i < kPoints.size(); ++i)
        EXPECT_DOUBLE_EQ(d[i], kPoints[i]);

    const void* conn = nullptr;
    ASSERT_EQ(mio_mesh_cell_block_conn(m, 0, &conn, &dt), MIO_OK);
    ASSERT_EQ(dt, MIO_INT64);
    const std::int64_t* c = static_cast<const std::int64_t*>(conn);
    for (std::size_t i = 0; i < kConn.size(); ++i)
        EXPECT_EQ(c[i], kConn[i]);

    mio_mesh_free(m);
}

TEST(CApi, Int32ConnectivityWidens) {
    const std::vector<std::int32_t> conn32 = {0, 1, 2, 3, 1, 2, 3, 4};
    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 5, 3, kPoints.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "tetra", 2, 4, MIO_INT32, conn32.data()), MIO_OK);
    const void* conn = nullptr;
    mio_dtype dt = MIO_FLOAT32;
    ASSERT_EQ(mio_mesh_cell_block_conn(m, 0, &conn, &dt), MIO_OK);
    EXPECT_EQ(dt, MIO_INT64);
    const std::int64_t* c = static_cast<const std::int64_t*>(conn);
    for (std::size_t i = 0; i < conn32.size(); ++i)
        EXPECT_EQ(c[i], conn32[i]);
    mio_mesh_free(m);
}

TEST(CApi, NamedDataRoundTrip) {
    mio_mesh* m = build_tet_mesh();

    const std::vector<double> temperature = {1.0, 2.0, 3.0, 4.0, 5.0};
    const std::int64_t shape1[] = {5};
    ASSERT_EQ(mio_mesh_add_point_data(m, "temperature", MIO_FLOAT64, 1, shape1, temperature.data()),
              MIO_OK);
    std::vector<double> velocity(15);
    for (std::size_t i = 0; i < velocity.size(); ++i)
        velocity[i] = 0.5 * static_cast<double>(i);
    const std::int64_t shape2[] = {5, 3};
    ASSERT_EQ(mio_mesh_add_point_data(m, "velocity", MIO_FLOAT64, 2, shape2, velocity.data()),
              MIO_OK);

    const std::vector<double> quality = {0.5, 0.75};
    const std::int64_t shapec[] = {2};
    ASSERT_EQ(mio_mesh_append_cell_data(m, "quality", MIO_FLOAT64, 1, shapec, quality.data()),
              MIO_OK);

    const std::vector<double> gravity = {0.0, 0.0, -9.81};
    const std::int64_t shapef[] = {3};
    ASSERT_EQ(mio_mesh_add_field_data(m, "gravity", MIO_FLOAT64, 1, shapef, gravity.data()),
              MIO_OK);

    // Names come back sorted on every backend.
    EXPECT_EQ(mio_mesh_num_point_data(m), 2);
    char buf[64] = {};
    ASSERT_GE(mio_mesh_point_data_name(m, 0, buf, sizeof(buf)), 0);
    EXPECT_STREQ(buf, "temperature");
    ASSERT_GE(mio_mesh_point_data_name(m, 1, buf, sizeof(buf)), 0);
    EXPECT_STREQ(buf, "velocity");

    const void* data = nullptr;
    mio_dtype dt = MIO_FLOAT32;
    std::int32_t ndim = 0;
    std::int64_t shape[MIO_MAX_NDIM] = {};
    ASSERT_EQ(mio_mesh_get_point_data(m, "velocity", &data, &dt, &ndim, shape), MIO_OK);
    EXPECT_EQ(ndim, 2);
    EXPECT_EQ(shape[0], 5);
    EXPECT_EQ(shape[1], 3);
    const double* v = static_cast<const double*>(data);
    for (std::size_t i = 0; i < velocity.size(); ++i)
        EXPECT_DOUBLE_EQ(v[i], velocity[i]);

    EXPECT_EQ(mio_mesh_num_cell_data(m), 1);
    EXPECT_EQ(mio_mesh_cell_data_num_blocks(m, "quality"), 1);
    ASSERT_EQ(mio_mesh_get_cell_data(m, "quality", 0, &data, &dt, &ndim, shape), MIO_OK);
    EXPECT_EQ(ndim, 1);
    EXPECT_EQ(shape[0], 2);
    EXPECT_DOUBLE_EQ(static_cast<const double*>(data)[1], 0.75);

    EXPECT_EQ(mio_mesh_num_field_data(m), 1);
    ASSERT_GE(mio_mesh_field_data_name(m, 0, buf, sizeof(buf)), 0);
    EXPECT_STREQ(buf, "gravity");
    ASSERT_EQ(mio_mesh_get_field_data(m, "gravity", &data, &dt, &ndim, shape), MIO_OK);
    EXPECT_EQ(shape[0], 3);
    EXPECT_DOUBLE_EQ(static_cast<const double*>(data)[2], -9.81);

    mio_mesh_free(m);
}

TEST(CApi, FileRoundTripAndConvert) {
    const std::string vtu = mt::temp_path("_capi.vtu");
    const std::string vtk = mt::temp_path("_capi.vtk");

    mio_mesh* m = build_tet_mesh();
    ASSERT_EQ(mio_write(vtu.c_str(), m, nullptr), MIO_OK);
    mio_mesh_free(m);

    mio_mesh* r = mio_read(vtu.c_str(), nullptr);
    ASSERT_NE(r, nullptr) << mio_last_error();
    EXPECT_EQ(mio_mesh_num_points(r), 5);
    EXPECT_EQ(mio_mesh_num_cell_blocks(r), 1);
    EXPECT_EQ(block_type(r, 0), "tetra");
    const void* pts = nullptr;
    mio_dtype dt;
    ASSERT_EQ(mio_mesh_get_points(r, &pts, &dt), MIO_OK);
    EXPECT_DOUBLE_EQ(static_cast<const double*>(pts)[4], 0.2);  // point 1, y
    mio_mesh_free(r);

    ASSERT_EQ(mio_convert(vtu.c_str(), nullptr, vtk.c_str(), nullptr), MIO_OK);
    mio_mesh* r2 = mio_read(vtk.c_str(), "vtk");
    ASSERT_NE(r2, nullptr) << mio_last_error();
    EXPECT_EQ(mio_mesh_num_points(r2), 5);
    mio_mesh_free(r2);

    std::remove(vtu.c_str());
    std::remove(vtk.c_str());
}

TEST(CApi, MeshOperations) {
    mio_mesh* m = build_tet_mesh();  // 2 tetra sharing a face

    // extract_surface -> boundary triangles.
    mio_mesh* surf = mio_extract_surface(m, /*record_parent_ids=*/1);
    ASSERT_NE(surf, nullptr) << mio_last_error();
    EXPECT_GT(mio_mesh_num_cell_blocks(surf), 0);
    EXPECT_EQ(block_type(surf, 0), "triangle");
    mio_mesh_free(surf);

    // extract_skin -> also boundary triangles.
    mio_mesh* skin = mio_extract_skin(m, /*linearize=*/0);
    ASSERT_NE(skin, nullptr) << mio_last_error();
    EXPECT_EQ(block_type(skin, 0), "triangle");
    mio_mesh_free(skin);

    // attach_quality preserves the cell block(s) and adds cell_data.
    mio_mesh* q = mio_attach_quality(m);
    ASSERT_NE(q, nullptr) << mio_last_error();
    EXPECT_EQ(mio_mesh_num_cell_blocks(q), 1);
    EXPECT_EQ(block_type(q, 0), "tetra");
    EXPECT_GT(mio_mesh_num_cell_data(q), 0);
    mio_mesh_free(q);

    // quality counts.
    std::int64_t nc = -1, ninv = -1, ndeg = -1;
    ASSERT_EQ(mio_quality_counts(m, &nc, &ninv, &ndeg), MIO_OK) << mio_last_error();
    EXPECT_EQ(nc, 2);
    EXPECT_GE(ninv, 0);
    EXPECT_GE(ndeg, 0);

    // bandwidth + reorder (RCM): result carries a renumbered mesh + node perm.
    EXPECT_GE(mio_compute_bandwidth(m), 0);
    mio_reorder_result* res = mio_reorder(m, "rcm");
    ASSERT_NE(res, nullptr) << mio_last_error();
    const mio_mesh* rmesh = mio_reorder_result_mesh(res);
    ASSERT_NE(rmesh, nullptr);
    EXPECT_EQ(mio_mesh_num_cell_blocks(rmesh), 1);
    const void* np = nullptr;
    mio_dtype dt = MIO_FLOAT64;
    std::int64_t n = -1;
    ASSERT_EQ(mio_reorder_result_node_perm(res, &np, &dt, &n), MIO_OK) << mio_last_error();
    EXPECT_EQ(dt, MIO_INT64);
    EXPECT_EQ(n, 5);  // build_tet_mesh has 5 points
    // node permutation is a bijection over [0, n).
    {
        const std::int64_t* p = static_cast<const std::int64_t*>(np);
        std::vector<int> seen(static_cast<std::size_t>(n), 0);
        for (std::int64_t i = 0; i < n; ++i) {
            ASSERT_GE(p[i], 0);
            ASSERT_LT(p[i], n);
            EXPECT_EQ(seen[static_cast<std::size_t>(p[i])], 0);
            seen[static_cast<std::size_t>(p[i])] = 1;
        }
    }
    EXPECT_EQ(mio_reorder_result_num_cell_perms(res), 1);
    // Ownership transfer keeps the mesh alive after freeing the result.
    mio_mesh* taken = mio_reorder_result_take_mesh(res);
    ASSERT_NE(taken, nullptr) << mio_last_error();
    mio_reorder_result_free(res);
    EXPECT_EQ(mio_mesh_num_cell_blocks(taken), 1);
    mio_mesh_free(taken);
    // Unknown method fails cleanly.
    EXPECT_EQ(mio_reorder(m, "bogus"), nullptr);

    mio_mesh_free(m);
}

TEST(CApi, Decimate) {
    // A 4-triangle fan around a centre vertex: the centre collapses into the
    // pinned boundary, leaving 2 triangles.
    const std::array<double, 15> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0.5, 0.5, 0};
    const std::array<std::int64_t, 12> conn = {0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4};
    mio_mesh* m = mio_mesh_create();
    ASSERT_NE(m, nullptr);
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 5, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "triangle", 4, 3, MIO_INT64, conn.data()), MIO_OK);

    mio_decimate_result* res =
        mio_decimate(m, /*target_ratio=*/-1.0, /*target_faces=*/1, /*max_error=*/-1.0,
                     /*placement=*/nullptr, /*preserve_boundary=*/1, /*preserve_features=*/1,
                     /*feature_angle=*/30.0);
    ASSERT_NE(res, nullptr) << mio_last_error();
    const mio_mesh* dm = mio_decimate_result_mesh(res);
    ASSERT_NE(dm, nullptr);
    EXPECT_EQ(mio_mesh_num_cell_blocks(dm), 1);
    EXPECT_EQ(block_type(dm, 0), "triangle");
    EXPECT_EQ(mio_decimate_result_faces_removed(res), 2);
    EXPECT_EQ(mio_decimate_result_points_removed(res), 1);
    EXPECT_GE(mio_decimate_result_collapses_rejected(res), 0);
    EXPECT_GE(mio_decimate_result_max_error_applied(res), 0.0);

    // The point map lands every input point on a live output index (the
    // collapsed centre maps to its survivor, not -1).
    const void* pm = nullptr;
    mio_dtype dt = MIO_FLOAT64;
    std::int64_t n = -1;
    ASSERT_EQ(mio_decimate_result_point_map(res, &pm, &dt, &n), MIO_OK) << mio_last_error();
    EXPECT_EQ(dt, MIO_INT64);
    EXPECT_EQ(n, 5);
    {
        const std::int64_t* p = static_cast<const std::int64_t*>(pm);
        for (std::int64_t i = 0; i < n; ++i) {
            EXPECT_GE(p[i], 0);
            EXPECT_LT(p[i], 4);
        }
    }
    ASSERT_EQ(mio_decimate_result_num_cell_maps(res), 1);
    const void* cm = nullptr;
    ASSERT_EQ(mio_decimate_result_cell_map(res, 0, &cm, &dt, &n), MIO_OK) << mio_last_error();
    EXPECT_EQ(dt, MIO_INT64);
    EXPECT_EQ(n, 4);
    EXPECT_EQ(mio_decimate_result_cell_map(res, 7, &cm, &dt, &n), MIO_ERR_NOT_FOUND);

    // Ownership transfer keeps the mesh alive after freeing the result.
    mio_mesh* taken = mio_decimate_result_take_mesh(res);
    ASSERT_NE(taken, nullptr) << mio_last_error();
    mio_decimate_result_free(res);
    EXPECT_EQ(mio_mesh_num_cell_blocks(taken), 1);
    mio_mesh_free(taken);

    // Error paths fail cleanly: no criterion, a bad placement, a volume mesh.
    EXPECT_EQ(mio_decimate(m, -1.0, -1, -1.0, nullptr, 1, 1, 30.0), nullptr);
    EXPECT_EQ(mio_decimate(m, -1.0, 1, -1.0, "nearest", 1, 1, 30.0), nullptr);
    mio_mesh_free(m);
    mio_mesh* tet = build_tet_mesh();
    EXPECT_EQ(mio_decimate(tet, 0.5, -1, -1.0, nullptr, 1, 1, 30.0), nullptr);
    EXPECT_NE(std::string(mio_last_error()).find("extract_surface"), std::string::npos);
    mio_mesh_free(tet);
}

TEST(CApi, DecimateVolume) {
    // A unit cube split into 6 positively-oriented tets sharing the main
    // diagonal 0-6 (the same fixture as test_decimate_volume.cpp).
    const std::array<double, 24> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                                        0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    const std::array<std::int64_t, 24> conn = {0, 1, 2, 6, 0, 2, 3, 6, 0, 3, 7, 6,
                                               0, 7, 4, 6, 0, 4, 5, 6, 0, 5, 1, 6};
    mio_mesh* m = mio_mesh_create();
    ASSERT_NE(m, nullptr);
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 8, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "tetra", 6, 4, MIO_INT64, conn.data()), MIO_OK);

    mio_decimate_volume_result* res =
        mio_decimate_volume(m, /*target_ratio=*/-1.0, /*target_cells=*/1, /*max_error=*/-1.0,
                            /*placement=*/nullptr, /*preserve_boundary=*/0, /*preserve_features=*/0,
                            /*feature_angle=*/30.0);
    ASSERT_NE(res, nullptr) << mio_last_error();
    const mio_mesh* dm = mio_decimate_volume_result_mesh(res);
    ASSERT_NE(dm, nullptr);
    EXPECT_EQ(mio_mesh_num_cell_blocks(dm), 1);
    EXPECT_EQ(block_type(dm, 0), "tetra");
    EXPECT_GT(mio_decimate_volume_result_tets_removed(res), 0);
    EXPECT_GE(mio_decimate_volume_result_points_removed(res), 0);
    EXPECT_GE(mio_decimate_volume_result_collapses_rejected(res), 0);
    EXPECT_GE(mio_decimate_volume_result_max_error_applied(res), 0.0);

    const void* pm = nullptr;
    mio_dtype dt = MIO_FLOAT64;
    std::int64_t n = -1;
    ASSERT_EQ(mio_decimate_volume_result_point_map(res, &pm, &dt, &n), MIO_OK) << mio_last_error();
    EXPECT_EQ(dt, MIO_INT64);
    EXPECT_EQ(n, 8);

    ASSERT_EQ(mio_decimate_volume_result_num_cell_maps(res), 1);
    const void* cm = nullptr;
    ASSERT_EQ(mio_decimate_volume_result_cell_map(res, 0, &cm, &dt, &n), MIO_OK)
        << mio_last_error();
    EXPECT_EQ(dt, MIO_INT64);
    EXPECT_EQ(n, 6);
    EXPECT_EQ(mio_decimate_volume_result_cell_map(res, 7, &cm, &dt, &n), MIO_ERR_NOT_FOUND);

    mio_mesh* taken = mio_decimate_volume_result_take_mesh(res);
    ASSERT_NE(taken, nullptr) << mio_last_error();
    mio_decimate_volume_result_free(res);
    EXPECT_EQ(mio_mesh_num_cell_blocks(taken), 1);
    mio_mesh_free(taken);

    // Error paths fail cleanly: no criterion, and decimate() itself still
    // refuses a tet mesh, pointing here.
    EXPECT_EQ(mio_decimate_volume(m, -1.0, -1, -1.0, nullptr, 0, 1, 30.0), nullptr);
    EXPECT_EQ(mio_decimate(m, -1.0, 1, -1.0, nullptr, 1, 1, 30.0), nullptr);
    EXPECT_NE(std::string(mio_last_error()).find("extract_surface"), std::string::npos);
    mio_mesh_free(m);

    // A non-tet 3D block is refused by name.
    mio_mesh* hex = mio_mesh_create();
    ASSERT_NE(hex, nullptr);
    const std::array<double, 24> hpts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                                         0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    const std::array<std::int64_t, 8> hconn = {0, 1, 2, 3, 4, 5, 6, 7};
    ASSERT_EQ(mio_mesh_set_points(hex, MIO_FLOAT64, 8, 3, hpts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(hex, "hexahedron", 1, 8, MIO_INT64, hconn.data()), MIO_OK);
    EXPECT_EQ(mio_decimate_volume(hex, -1.0, 1, -1.0, nullptr, 0, 1, 30.0), nullptr);
    EXPECT_NE(std::string(mio_last_error()).find("tet-only"), std::string::npos);
    mio_mesh_free(hex);
}

TEST(CApi, Merge) {
    mio_mesh* a = build_tet_mesh();  // 5 points, 2 tetra
    mio_mesh* b = build_tet_mesh();
    const mio_mesh* inputs[2] = {a, b};

    // Concatenate: point/cell counts sum; source tag present.
    mio_mesh* cat = mio_merge(inputs, 2, /*weld=*/0, 1e-8, /*source_tag=*/1,
                              /*data_policy=*/0, /*drop_dup=*/0);
    ASSERT_NE(cat, nullptr) << mio_last_error();
    EXPECT_EQ(mio_mesh_num_points(cat), 10);
    EXPECT_EQ(mio_mesh_num_cell_blocks(cat), 1);
    EXPECT_GT(mio_mesh_num_cell_data(cat), 0);  // source_mesh_id
    mio_mesh_free(cat);

    // Weld two identical meshes -> the 5 coincident points collapse.
    mio_mesh* welded = mio_merge(inputs, 2, /*weld=*/1, 1e-9, /*source_tag=*/1,
                                 /*data_policy=*/0, /*drop_dup=*/0);
    ASSERT_NE(welded, nullptr) << mio_last_error();
    EXPECT_EQ(mio_mesh_num_points(welded), 5);
    mio_mesh_free(welded);

    // Error paths.
    EXPECT_EQ(mio_merge(nullptr, 2, 0, 1e-8, 1, 0, 0), nullptr);
    EXPECT_EQ(mio_merge(inputs, 0, 0, 1e-8, 1, 0, 0), nullptr);

    mio_mesh_free(a);
    mio_mesh_free(b);
}

TEST(CApi, Interpolate) {
    // Source: two points on the x-axis carrying a scalar field.
    mio_mesh* src = mio_mesh_create();
    const std::vector<double> spts = {0, 0, 0, 1, 0, 0};
    ASSERT_EQ(mio_mesh_set_points(src, MIO_FLOAT64, 2, 3, spts.data()), MIO_OK);
    const std::vector<double> f = {10.0, 20.0};
    const std::int64_t shape1[] = {2};
    ASSERT_EQ(mio_mesh_add_point_data(src, "f", MIO_FLOAT64, 1, shape1, f.data()), MIO_OK);

    // Target: one point near the second source point.
    mio_mesh* tgt = mio_mesh_create();
    const std::vector<double> tpts = {0.9, 0, 0};
    ASSERT_EQ(mio_mesh_set_points(tgt, MIO_FLOAT64, 1, 3, tpts.data()), MIO_OK);

    // Default arrays (= all source point_data), default method (nearest).
    mio_mesh* out = mio_interpolate(src, tgt, nullptr, nullptr, 0, 0, 0.0, nullptr);
    ASSERT_NE(out, nullptr) << mio_last_error();
    const void* data = nullptr;
    mio_dtype dt = MIO_FLOAT32;
    ASSERT_EQ(mio_mesh_get_point_data(out, "f", &data, &dt, nullptr, nullptr), MIO_OK);
    EXPECT_EQ(dt, MIO_FLOAT64);
    EXPECT_DOUBLE_EQ(static_cast<const double*>(data)[0], 20.0);
    mio_mesh_free(out);

    // An explicit name list goes through the char** + count convention.
    const char* names[] = {"f"};
    out = mio_interpolate(src, tgt, "nearest", names, 1, 0, 0.0, "error");
    ASSERT_NE(out, nullptr) << mio_last_error();
    mio_mesh_free(out);

    // Error paths: bad method / unknown array / NULL meshes.
    EXPECT_EQ(mio_interpolate(src, tgt, "bogus", nullptr, 0, 0, 0.0, nullptr), nullptr);
    EXPECT_STRNE(mio_last_error(), "");
    const char* bad[] = {"nope"};
    EXPECT_EQ(mio_interpolate(src, tgt, nullptr, bad, 1, 0, 0.0, nullptr), nullptr);
    EXPECT_EQ(mio_interpolate(nullptr, tgt, nullptr, nullptr, 0, 0, 0.0, nullptr), nullptr);
    EXPECT_EQ(mio_interpolate(src, nullptr, nullptr, nullptr, 0, 0, 0.0, nullptr), nullptr);

    mio_mesh_free(src);
    mio_mesh_free(tgt);
}

TEST(CApi, ConservativeInterpolate) {
    // Two identical single-triangle meshes: the full-value-recovery oracle.
    mio_mesh* src = mio_mesh_create();
    const std::vector<double> spts = {0, 0, 0, 1, 0, 0, 0, 1, 0};
    ASSERT_EQ(mio_mesh_set_points(src, MIO_FLOAT64, 3, 3, spts.data()), MIO_OK);
    const std::vector<std::int64_t> sconn = {0, 1, 2};
    ASSERT_EQ(mio_mesh_add_cell_block(src, "triangle", 1, 3, MIO_INT64, sconn.data()), MIO_OK);
    const std::vector<double> f = {7.0};
    const std::int64_t shapec[] = {1};
    ASSERT_EQ(mio_mesh_append_cell_data(src, "f", MIO_FLOAT64, 1, shapec, f.data()), MIO_OK);

    mio_mesh* tgt = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(tgt, MIO_FLOAT64, 3, 3, spts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(tgt, "triangle", 1, 3, MIO_INT64, sconn.data()), MIO_OK);

    mio_mesh* out = mio_conservative_interpolate(src, tgt, nullptr, 0, 0.0, nullptr);
    ASSERT_NE(out, nullptr) << mio_last_error();
    const void* data = nullptr;
    mio_dtype dt = MIO_FLOAT32;
    ASSERT_EQ(mio_mesh_get_cell_data(out, "f", 0, &data, &dt, nullptr, nullptr), MIO_OK);
    EXPECT_EQ(dt, MIO_FLOAT64);
    EXPECT_DOUBLE_EQ(static_cast<const double*>(data)[0], 7.0);
    mio_mesh_free(out);

    // An explicit name list goes through the char** + count convention.
    const char* names[] = {"f"};
    out = mio_conservative_interpolate(src, tgt, names, 1, 0.0, "error");
    ASSERT_NE(out, nullptr) << mio_last_error();
    mio_mesh_free(out);

    // Error paths: unknown array / NULL meshes.
    const char* bad[] = {"nope"};
    EXPECT_EQ(mio_conservative_interpolate(src, tgt, bad, 1, 0.0, nullptr), nullptr);
    EXPECT_STRNE(mio_last_error(), "");
    EXPECT_EQ(mio_conservative_interpolate(nullptr, tgt, nullptr, 0, 0.0, nullptr), nullptr);
    EXPECT_EQ(mio_conservative_interpolate(src, nullptr, nullptr, 0, 0.0, nullptr), nullptr);

    mio_mesh_free(src);
    mio_mesh_free(tgt);
}

TEST(CApi, Diff) {
    mio_mesh* a = build_tet_mesh();
    mio_mesh* b = build_tet_mesh();

    // Identical meshes -> equal, verdict identical.
    int equal = -1;
    ASSERT_EQ(mio_meshes_equal(a, b, 1e-12, 1e-9, 0, &equal), MIO_OK) << mio_last_error();
    EXPECT_EQ(equal, 1);
    mio_diff_result* res = nullptr;
    ASSERT_EQ(mio_diff(a, b, 1e-12, 1e-9, 0, &res), MIO_OK) << mio_last_error();
    ASSERT_NE(res, nullptr);
    EXPECT_EQ(mio_diff_result_verdict(res), MIO_DIFF_IDENTICAL);
    EXPECT_EQ(mio_diff_result_num_block_diffs(res), 1);
    int type_mism = -1, count_mism = -1;
    std::int64_t conn = -1;
    ASSERT_EQ(mio_diff_result_block(res, 0, &type_mism, &count_mism, &conn), MIO_OK);
    EXPECT_EQ(type_mism, 0);
    EXPECT_EQ(count_mism, 0);
    EXPECT_EQ(conn, 0);
    mio_diff_result_free(res);
    mio_mesh_free(b);

    // A mesh with different connectivity -> different.
    const std::vector<std::int64_t> conn2 = {0, 1, 2, 3, 0, 2, 3, 4};
    mio_mesh* c = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(c, MIO_FLOAT64, 5, 3, kPoints.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(c, "tetra", 2, 4, MIO_INT64, conn2.data()), MIO_OK);
    ASSERT_EQ(mio_meshes_equal(a, c, 1e-12, 1e-9, 0, &equal), MIO_OK);
    EXPECT_EQ(equal, 0);
    res = nullptr;
    ASSERT_EQ(mio_diff(a, c, 1e-12, 1e-9, 0, &res), MIO_OK);
    EXPECT_EQ(mio_diff_result_verdict(res), MIO_DIFF_DIFFERENT);
    ASSERT_EQ(mio_diff_result_block(res, 0, nullptr, nullptr, &conn), MIO_OK);
    EXPECT_EQ(conn, 1);
    mio_diff_result_free(res);
    mio_mesh_free(c);

    // NULL mesh is a clean error.
    EXPECT_EQ(mio_diff(nullptr, a, 1e-12, 1e-9, 0, &res), MIO_ERR_INVALID_ARG);

    mio_mesh_free(a);
}

TEST(CApi, SniffFormat) {
    const std::string vtu = mt::temp_path("_capi_sniff.vtu");
    mio_mesh* m = build_tet_mesh();
    ASSERT_EQ(mio_write(vtu.c_str(), m, nullptr), MIO_OK);
    mio_mesh_free(m);

    char buf[32] = {};
    const std::int64_t n = mio_sniff_format(vtu.c_str(), buf, sizeof(buf));
    EXPECT_EQ(std::string(buf), "vtu");
    EXPECT_EQ(n, 3);
    std::remove(vtu.c_str());
}

TEST(CApi, ZeroCopyPointerStability) {
    mio_mesh* m = build_tet_mesh();
    const void* pts1 = nullptr;
    mio_dtype dt;
    ASSERT_EQ(mio_mesh_get_points(m, &pts1, &dt), MIO_OK);
    // Non-mutating traffic must not invalidate or move the borrow.
    (void)mio_mesh_num_cell_blocks(m);
    (void)block_type(m, 0);
    const void* pts2 = nullptr;
    ASSERT_EQ(mio_mesh_get_points(m, &pts2, &dt), MIO_OK);
    EXPECT_EQ(pts1, pts2);
    EXPECT_DOUBLE_EQ(static_cast<const double*>(pts1)[3], 1.1);
    mio_mesh_free(m);
}

TEST(CApi, ErrorPaths) {
    // NULL / invalid arguments.
    EXPECT_EQ(mio_read(nullptr, nullptr), nullptr);
    EXPECT_STRNE(mio_last_error(), "");
    EXPECT_EQ(mio_write("out.vtu", nullptr, nullptr), MIO_ERR_INVALID_ARG);
    EXPECT_EQ(mio_mesh_num_points(nullptr), -1);
    EXPECT_EQ(mio_mesh_set_points(nullptr, MIO_FLOAT64, 1, 3, kPoints.data()), MIO_ERR_INVALID_ARG);

    mio_mesh* m = build_tet_mesh();

    // Wrong dtypes / shapes.
    EXPECT_EQ(mio_mesh_set_points(m, MIO_INT32, 5, 3, kPoints.data()), MIO_ERR_INVALID_ARG);
    EXPECT_EQ(mio_mesh_add_cell_block(m, "tetra", 2, 5, MIO_INT64, kConn.data()),
              MIO_ERR_INVALID_ARG);        // tetra has 4 nodes, not 5
    const std::int64_t bad_shape[] = {4};  // 5 points in the mesh
    const std::vector<double> four(4, 1.0);
    EXPECT_EQ(mio_mesh_add_point_data(m, "bad", MIO_FLOAT64, 1, bad_shape, four.data()),
              MIO_ERR_INVALID_ARG);

    // Out-of-range lookups.
    EXPECT_EQ(mio_mesh_cell_block_info(m, 7, nullptr, nullptr, nullptr), MIO_ERR_NOT_FOUND);
    const void* data = nullptr;
    mio_dtype dt;
    EXPECT_EQ(mio_mesh_get_point_data(m, "nope", &data, &dt, nullptr, nullptr), MIO_ERR_NOT_FOUND);
    EXPECT_EQ(mio_mesh_point_data_name(m, 0, nullptr, 0), -1);  // no point data yet

    // Unknown format / extension.
    EXPECT_EQ(mio_write("mesh.not_an_extension", m, nullptr), MIO_ERR_READ);
    EXPECT_STRNE(mio_last_error(), "");
    EXPECT_EQ(mio_write("mesh.vtu", m, "no_such_format"), MIO_ERR_NOT_FOUND);
    // openfoam gained a writer in v9.20.0, so the "resolvable format with no
    // writer" case it used to demonstrate needs a different format: svg/tikz
    // are write-only, so pick the mirror case -- a read-only key no longer
    // exists in the registry at all.
    EXPECT_EQ(mio_read("mesh.svg", "svg"), nullptr);

#ifndef MESHIOPLUSPLUS_HAS_HDF5
    // Compiled-out formats name the missing dependency.
    EXPECT_EQ(mio_write("mesh.med", m, nullptr), MIO_ERR_NOT_FOUND);
    EXPECT_NE(std::strstr(mio_last_error(), "HDF5"), nullptr) << mio_last_error();
#endif

    mio_mesh_free(m);
    mio_mesh_free(nullptr);  // NULL-safe
}

TEST(CApi, StringBufferProtocol) {
    mio_mesh* m = build_tet_mesh();
    // Full length is returned even when the buffer is too small ("tetra" = 5).
    char tiny[3] = {'x', 'x', 'x'};
    EXPECT_EQ(mio_mesh_cell_block_type(m, 0, tiny, sizeof(tiny)), 5);
    EXPECT_STREQ(tiny, "te");                                  // truncated + NUL-terminated
    EXPECT_EQ(mio_mesh_cell_block_type(m, 0, nullptr, 0), 5);  // pure length query
    mio_mesh_free(m);
}

/* Ragged (polygon / polyhedron) connectivity.
 *
 * These used to need MED (the one C++ writer that serializes ragged blocks) to
 * construct a ragged mesh at all, and so only ran on an HDF5 build. The setters
 * mean the C API can now build one directly, which is why they are unguarded. */

TEST(CApi, PolygonBlockRoundTripsThroughTheCApi) {
    mio_mesh* m = mio_mesh_create();
    ASSERT_NE(m, nullptr);
    const double xyz[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 2, 0.5, 0};
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 5, 3, xyz), MIO_OK);

    // A quad then a triangle -- the point of a jagged block.
    const std::int64_t row_offsets[] = {0, 4, 7};
    const std::int64_t nodes[] = {0, 1, 2, 3, 1, 4, 2};
    ASSERT_EQ(mio_mesh_add_polygon_block(m, "polygon", 2, row_offsets, nodes, 7), MIO_OK)
        << mio_last_error();

    mio_cell_block_info info{};
    ASSERT_EQ(mio_mesh_cell_block_info_ex(m, 0, &info), MIO_OK);
    EXPECT_EQ(info.num_cells, 2);
    EXPECT_EQ(info.nodes_per_cell, 0);
    EXPECT_EQ(info.is_ragged, 1);
    EXPECT_EQ(info.is_polyhedron, 0);
    EXPECT_EQ(info.num_faces, 2);  // 1-level: one face per cell
    EXPECT_EQ(info.num_nodes, 7);

    // The five-argument original must keep agreeing on the fields it shares.
    std::int64_t nc = 0, npc = -1;
    std::int32_t ragged = 0;
    ASSERT_EQ(mio_mesh_cell_block_info(m, 0, &nc, &npc, &ragged), MIO_OK);
    EXPECT_EQ(nc, info.num_cells);
    EXPECT_EQ(npc, info.nodes_per_cell);
    EXPECT_EQ(ragged, info.is_ragged);

    // A ragged block has no rectangular buffer to borrow.
    const void* conn = nullptr;
    mio_dtype dt;
    EXPECT_EQ(mio_mesh_cell_block_conn(m, 0, &conn, &dt), MIO_ERR_UNSUPPORTED);

    mio_poly_conn* pc = mio_poly_conn_create(m, 0);
    ASSERT_NE(pc, nullptr) << mio_last_error();
    mio_poly_conn_shape shape{};
    ASSERT_EQ(mio_poly_conn_get_shape(pc, &shape), MIO_OK);
    EXPECT_EQ(shape.is_polyhedron, 0);
    EXPECT_EQ(shape.num_cells, 2);
    EXPECT_EQ(shape.num_faces, 2);
    EXPECT_EQ(shape.num_nodes, 7);

    std::int64_t n = -1;
    const std::int64_t* out_nodes = mio_poly_conn_nodes(pc, &n);
    ASSERT_NE(out_nodes, nullptr);
    EXPECT_EQ(n, 7);
    EXPECT_TRUE(std::equal(nodes, nodes + 7, out_nodes));
    const std::int64_t* out_rows = mio_poly_conn_face_offsets(pc, &n);
    ASSERT_NE(out_rows, nullptr);
    EXPECT_EQ(n, 3);
    EXPECT_TRUE(std::equal(row_offsets, row_offsets + 3, out_rows));

    // A 1-level block has no cell-offsets array: NULL, not a synthesized
    // identity that a caller could mistake for information.
    n = -1;
    EXPECT_EQ(mio_poly_conn_cell_offsets(pc, &n), nullptr);
    EXPECT_EQ(n, 0);

    mio_poly_conn_free(pc);
    mio_mesh_free(m);
}

TEST(CApi, PolyhedronBlockRoundTripsThroughTheCApi) {
    mio_mesh* m = mio_mesh_create();
    ASSERT_NE(m, nullptr);
    const double xyz[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 5, 3, xyz), MIO_OK);

    // Two cells: a 4-face tetrahedron, then a 3-face sliver (deliberately not
    // the same face count, so cell_offsets carries real information).
    const std::int64_t cell_offsets[] = {0, 4, 7};
    const std::int64_t face_offsets[] = {0, 3, 6, 9, 12, 15, 18, 21};
    const std::int64_t nodes[] = {0, 1, 2, 0, 3, 1, 1, 3, 2, 2, 3, 0, 1, 2, 4, 2, 3, 4, 3, 1, 4};
    ASSERT_EQ(
        mio_mesh_add_polyhedron_block(m, "polyhedron", 2, cell_offsets, 7, face_offsets, nodes, 21),
        MIO_OK)
        << mio_last_error();

    mio_cell_block_info info{};
    ASSERT_EQ(mio_mesh_cell_block_info_ex(m, 0, &info), MIO_OK);
    EXPECT_EQ(info.num_cells, 2);
    EXPECT_EQ(info.is_ragged, 1);
    EXPECT_EQ(info.is_polyhedron, 1);
    EXPECT_EQ(info.num_faces, 7);
    EXPECT_EQ(info.num_nodes, 21);

    mio_poly_conn* pc = mio_poly_conn_create(m, 0);
    ASSERT_NE(pc, nullptr) << mio_last_error();
    mio_poly_conn_shape shape{};
    ASSERT_EQ(mio_poly_conn_get_shape(pc, &shape), MIO_OK);
    EXPECT_EQ(shape.is_polyhedron, 1);
    EXPECT_EQ(shape.num_cells, 2);
    EXPECT_EQ(shape.num_faces, 7);
    EXPECT_EQ(shape.num_nodes, 21);

    std::int64_t n = -1;
    const std::int64_t* out_nodes = mio_poly_conn_nodes(pc, &n);
    ASSERT_NE(out_nodes, nullptr);
    ASSERT_EQ(n, 21);
    EXPECT_TRUE(std::equal(nodes, nodes + 21, out_nodes));
    const std::int64_t* out_faces = mio_poly_conn_face_offsets(pc, &n);
    ASSERT_NE(out_faces, nullptr);
    ASSERT_EQ(n, 8);
    EXPECT_TRUE(std::equal(face_offsets, face_offsets + 8, out_faces));
    const std::int64_t* out_cells = mio_poly_conn_cell_offsets(pc, &n);
    ASSERT_NE(out_cells, nullptr);
    ASSERT_EQ(n, 3);
    EXPECT_TRUE(std::equal(cell_offsets, cell_offsets + 3, out_cells));

    // Face f of cell c spans nodes[face_offsets[cell_offsets[c] + f] .. +1) --
    // the documented indexing, checked rather than described.
    const std::int64_t f = out_cells[1] + 0;  // cell 1's first face
    EXPECT_EQ(out_faces[f + 1] - out_faces[f], 3);
    EXPECT_EQ(out_nodes[out_faces[f]], 1);

    mio_poly_conn_free(pc);
    mio_mesh_free(m);
}

TEST(CApi, PolyConnSnapshotSurvivesMutation) {
    // The contract that distinguishes the snapshot from every other getter on
    // this ABI: rule 3 borrows die at the next mutating call, this does not.
    mio_mesh* m = mio_mesh_create();
    const double xyz[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 4, 3, xyz), MIO_OK);
    const std::int64_t row_offsets[] = {0, 4};
    const std::int64_t nodes[] = {0, 1, 2, 3};
    ASSERT_EQ(mio_mesh_add_polygon_block(m, "polygon", 1, row_offsets, nodes, 4), MIO_OK);

    mio_poly_conn* pc = mio_poly_conn_create(m, 0);
    ASSERT_NE(pc, nullptr) << mio_last_error();
    const std::int64_t* borrowed = mio_poly_conn_nodes(pc, nullptr);
    ASSERT_NE(borrowed, nullptr);

    const std::int64_t tri[] = {0, 1, 2};
    ASSERT_EQ(mio_mesh_add_cell_block(m, "triangle", 1, 3, MIO_INT64, tri), MIO_OK);
    EXPECT_EQ(mio_mesh_num_cell_blocks(m), 2);
    // Still readable, and still the values it was created from.
    EXPECT_TRUE(std::equal(nodes, nodes + 4, borrowed));

    mio_poly_conn_free(pc);
    mio_mesh_free(m);
}

TEST(CApi, PolyConnRejectsRectangularBlocks) {
    mio_mesh* m = mio_mesh_create();
    const double xyz[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 3, 3, xyz), MIO_OK);
    const std::int64_t tri[] = {0, 1, 2};
    ASSERT_EQ(mio_mesh_add_cell_block(m, "triangle", 1, 3, MIO_INT64, tri), MIO_OK);

    EXPECT_EQ(mio_poly_conn_create(m, 0), nullptr);
    EXPECT_STRNE(mio_last_error(), "");
    EXPECT_EQ(mio_poly_conn_create(m, 7), nullptr);  // out of range

    mio_cell_block_info info{};
    ASSERT_EQ(mio_mesh_cell_block_info_ex(m, 0, &info), MIO_OK);
    EXPECT_EQ(info.is_ragged, 0);
    EXPECT_EQ(info.is_polyhedron, 0);
    EXPECT_EQ(info.num_faces, 1);
    EXPECT_EQ(info.num_nodes, 3);
    mio_mesh_free(m);
}

TEST(CApi, RaggedSettersRejectMalformedOffsets) {
    mio_mesh* m = mio_mesh_create();
    const std::int64_t nodes[] = {0, 1, 2, 3};

    const std::int64_t not_zero_based[] = {1, 4};
    EXPECT_EQ(mio_mesh_add_polygon_block(m, "polygon", 1, not_zero_based, nodes, 4),
              MIO_ERR_INVALID_ARG);
    const std::int64_t decreasing[] = {0, 3, 2};
    EXPECT_EQ(mio_mesh_add_polygon_block(m, "polygon", 2, decreasing, nodes, 2),
              MIO_ERR_INVALID_ARG);
    const std::int64_t wrong_total[] = {0, 3};
    EXPECT_EQ(mio_mesh_add_polygon_block(m, "polygon", 1, wrong_total, nodes, 4),
              MIO_ERR_INVALID_ARG);
    EXPECT_EQ(mio_mesh_add_polygon_block(m, "polygon", 1, nullptr, nodes, 4), MIO_ERR_INVALID_ARG);

    const std::int64_t negative[] = {0, -1, 2, 3};
    const std::int64_t ok_offsets[] = {0, 4};
    EXPECT_EQ(mio_mesh_add_polygon_block(m, "polygon", 1, ok_offsets, negative, 4),
              MIO_ERR_INVALID_ARG);

    // The 2-level setter validates both offset arrays independently.
    const std::int64_t cell_offsets[] = {0, 2};
    const std::int64_t bad_faces[] = {0, 3, 5};  // must end at num_nodes == 6
    EXPECT_EQ(
        mio_mesh_add_polyhedron_block(m, "polyhedron", 1, cell_offsets, 2, bad_faces, nodes, 6),
        MIO_ERR_INVALID_ARG);
    const std::int64_t bad_cells[] = {0, 3};  // must end at num_faces == 2
    const std::int64_t ok_faces[] = {0, 3, 6};
    EXPECT_EQ(mio_mesh_add_polyhedron_block(m, "polyhedron", 1, bad_cells, 2, ok_faces, nodes, 6),
              MIO_ERR_INVALID_ARG);

    // Nothing was added by any of the rejected calls.
    EXPECT_EQ(mio_mesh_num_cell_blocks(m), 0);
    mio_mesh_free(m);
}

#ifdef MESHIOPLUSPLUS_HAS_HDF5
TEST(CApi, RaggedBlockBuiltThroughTheCApiSurvivesAMedRoundTrip) {
    // The end-to-end path the setters exist for: build ragged through C, write
    // it with the one C++ writer that serializes ragged blocks (MED's POG),
    // read it back and compare through the snapshot.
    mio_mesh* m = mio_mesh_create();
    const double xyz[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 2, 0.5, 0};
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 5, 3, xyz), MIO_OK);
    const std::int64_t row_offsets[] = {0, 4, 7};
    const std::int64_t nodes[] = {0, 1, 2, 3, 1, 4, 2};
    ASSERT_EQ(mio_mesh_add_polygon_block(m, "polygon", 2, row_offsets, nodes, 7), MIO_OK);

    const std::string med = mt::temp_path("_capi_ragged.med");
    ASSERT_EQ(mio_write(med.c_str(), m, nullptr), MIO_OK) << mio_last_error();
    mio_mesh_free(m);

    mio_mesh* back = mio_read(med.c_str(), nullptr);
    ASSERT_NE(back, nullptr) << mio_last_error();
    ASSERT_EQ(mio_mesh_num_cell_blocks(back), 1);
    mio_poly_conn* pc = mio_poly_conn_create(back, 0);
    ASSERT_NE(pc, nullptr) << mio_last_error();
    std::int64_t n = -1;
    const std::int64_t* out_nodes = mio_poly_conn_nodes(pc, &n);
    ASSERT_EQ(n, 7);
    EXPECT_TRUE(std::equal(nodes, nodes + 7, out_nodes));
    const std::int64_t* out_rows = mio_poly_conn_face_offsets(pc, &n);
    ASSERT_EQ(n, 3);
    EXPECT_TRUE(std::equal(row_offsets, row_offsets + 3, out_rows));
    mio_poly_conn_free(pc);
    mio_mesh_free(back);
    std::remove(med.c_str());
}
#endif

#ifdef MESHIOPLUSPLUS_HAS_HDF5
TEST(CApi, HdfFormatConvert) {
    const std::string vtu = mt::temp_path("_capi_h5.vtu");
    const std::string med = mt::temp_path("_capi_h5.med");
    mio_mesh* m = build_tet_mesh();
    ASSERT_EQ(mio_write(vtu.c_str(), m, nullptr), MIO_OK);
    mio_mesh_free(m);
    ASSERT_EQ(mio_convert(vtu.c_str(), nullptr, med.c_str(), nullptr), MIO_OK) << mio_last_error();
    mio_mesh* r = mio_read(med.c_str(), nullptr);
    ASSERT_NE(r, nullptr) << mio_last_error();
    EXPECT_EQ(mio_mesh_num_points(r), 5);
    mio_mesh_free(r);
    std::remove(vtu.c_str());
    std::remove(med.c_str());
}
#endif

/* --- data operations ---------------------------------------------------- */

// A tet mesh carrying point, cell and field data, for the data operations.
mio_mesh* build_data_mesh() {
    mio_mesh* m = build_tet_mesh();  // 5 points, 2 tetra
    static const std::array<double, 5> temperature = {0.0, 1.0, 2.0, 3.0, 4.0};
    static const std::array<double, 15> velocity = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 2, 0, 0};
    static const std::array<double, 2> mat = {10.0, 20.0};
    static const std::array<double, 3> meta = {1.0, 2.0, 3.0};
    std::int64_t s1[1] = {5};
    std::int64_t s2[2] = {5, 3};
    std::int64_t sc[1] = {2};
    std::int64_t sf[1] = {3};
    EXPECT_EQ(mio_mesh_add_point_data(m, "T", MIO_FLOAT64, 1, s1, temperature.data()), MIO_OK);
    EXPECT_EQ(mio_mesh_add_point_data(m, "v", MIO_FLOAT64, 2, s2, velocity.data()), MIO_OK);
    EXPECT_EQ(mio_mesh_append_cell_data(m, "mat", MIO_FLOAT64, 1, sc, mat.data()), MIO_OK);
    EXPECT_EQ(mio_mesh_add_field_data(m, "meta", MIO_FLOAT64, 1, sf, meta.data()), MIO_OK);
    return m;
}

TEST(CApi, DataDropAndKeep) {
    mio_mesh* m = build_data_mesh();
    const char* names[] = {"T"};
    mio_mesh* dropped = mio_data_drop(m, MIO_DATA_POINT, names, 1, 0);
    ASSERT_NE(dropped, nullptr) << mio_last_error();
    EXPECT_EQ(mio_mesh_num_point_data(dropped), 1);  // only "v" left
    // Geometry is never modified.
    EXPECT_EQ(mio_mesh_num_points(dropped), 5);
    EXPECT_EQ(mio_mesh_num_cell_blocks(dropped), 1);
    mio_mesh_free(dropped);

    mio_mesh* kept = mio_data_keep(m, MIO_DATA_POINT, names, 1, 0);
    ASSERT_NE(kept, nullptr) << mio_last_error();
    EXPECT_EQ(mio_mesh_num_point_data(kept), 1);
    mio_mesh_free(kept);
    mio_mesh_free(m);
}

TEST(CApi, DataDropUnknownKeyFails) {
    mio_mesh* m = build_data_mesh();
    const char* names[] = {"nope"};
    EXPECT_EQ(mio_data_drop(m, MIO_DATA_POINT, names, 1, 0), nullptr);
    EXPECT_STRNE(mio_last_error(), "");
    // ignore_missing makes it a no-op instead.
    mio_mesh* ok = mio_data_drop(m, MIO_DATA_POINT, names, 1, 1);
    ASSERT_NE(ok, nullptr) << mio_last_error();
    EXPECT_EQ(mio_mesh_num_point_data(ok), 2);
    mio_mesh_free(ok);
    mio_mesh_free(m);
}

TEST(CApi, DataRename) {
    mio_mesh* m = build_data_mesh();
    mio_mesh* out = mio_data_rename(m, MIO_DATA_POINT, "T", "temperature");
    ASSERT_NE(out, nullptr) << mio_last_error();
    char buf[64] = {};
    bool found = false;
    for (std::int64_t i = 0; i < mio_mesh_num_point_data(out); ++i) {
        mio_mesh_point_data_name(out, i, buf, sizeof(buf));
        if (std::string(buf) == "temperature")
            found = true;
    }
    EXPECT_TRUE(found);
    mio_mesh_free(out);
    mio_mesh_free(m);
}

TEST(CApi, DataAveraging) {
    mio_mesh* m = build_data_mesh();
    mio_mesh* to_cell = mio_data_point_to_cell(m, nullptr, 0, nullptr);
    ASSERT_NE(to_cell, nullptr) << mio_last_error();
    EXPECT_GT(mio_mesh_num_cell_data(to_cell), 0);
    mio_mesh_free(to_cell);

    const char* names[] = {"mat"};
    mio_mesh* to_point = mio_data_cell_to_point(m, names, 1, MIO_WEIGHT_UNIFORM, nullptr);
    ASSERT_NE(to_point, nullptr) << mio_last_error();
    const void* data = nullptr;
    mio_dtype dt;
    std::int32_t ndim = 0;
    std::int64_t shape[MIO_MAX_NDIM] = {};
    ASSERT_EQ(mio_mesh_get_point_data(to_point, "mat", &data, &dt, &ndim, shape), MIO_OK);
    EXPECT_EQ(shape[0], 5);
    // A mean is not an integer: the output is always Float64.
    EXPECT_EQ(dt, MIO_FLOAT64);
    mio_mesh_free(to_point);

    mio_mesh* weighted = mio_data_cell_to_point(m, names, 1, MIO_WEIGHT_MEASURE, "_w");
    ASSERT_NE(weighted, nullptr) << mio_last_error();
    EXPECT_EQ(mio_mesh_get_point_data(weighted, "mat_w", &data, &dt, &ndim, shape), MIO_OK);
    mio_mesh_free(weighted);
    mio_mesh_free(m);
}

TEST(CApi, DataCalc) {
    mio_mesh* m = build_data_mesh();
    mio_mesh* out = mio_data_calc(m, "norm(v)", MIO_DATA_POINT, "speed", 0);
    ASSERT_NE(out, nullptr) << mio_last_error();
    const void* data = nullptr;
    mio_dtype dt;
    std::int32_t ndim = 0;
    std::int64_t shape[MIO_MAX_NDIM] = {};
    ASSERT_EQ(mio_mesh_get_point_data(out, "speed", &data, &dt, &ndim, shape), MIO_OK);
    ASSERT_EQ(dt, MIO_FLOAT64);
    const double* speed = static_cast<const double*>(data);
    EXPECT_NEAR(speed[0], 1.0, 1e-12);
    EXPECT_NEAR(speed[3], std::sqrt(2.0), 1e-12);
    mio_mesh_free(out);
    mio_mesh_free(m);
}

TEST(CApi, DataCalcRejectsBadExpressions) {
    mio_mesh* m = build_data_mesh();
    EXPECT_EQ(mio_data_calc(m, "log(T)", MIO_DATA_POINT, "o", 0), nullptr);
    EXPECT_STRNE(mio_last_error(), "");
    EXPECT_EQ(mio_data_calc(m, "nope + 1", MIO_DATA_POINT, "o", 0), nullptr);
    EXPECT_STRNE(mio_last_error(), "");
    EXPECT_EQ(mio_data_calc(m, "T +", MIO_DATA_POINT, "o", 0), nullptr);
    EXPECT_STRNE(mio_last_error(), "");
    mio_mesh_free(m);
}

TEST(CApi, DataCondition) {
    mio_mesh* m = build_data_mesh();
    const char* names[] = {"T"};
    mio_mesh* out = mio_data_condition(m, MIO_DATA_POINT, names, 1, MIO_COND_NORMALIZE, 0.0, 1.0,
                                       MIO_SCOPE_COMPONENT, MIO_NAN_IGNORE, 0.0, nullptr);
    ASSERT_NE(out, nullptr) << mio_last_error();
    const void* data = nullptr;
    mio_dtype dt;
    std::int32_t ndim = 0;
    std::int64_t shape[MIO_MAX_NDIM] = {};
    ASSERT_EQ(mio_mesh_get_point_data(out, "T", &data, &dt, &ndim, shape), MIO_OK);
    const double* t = static_cast<const double*>(data);
    // T = {0,1,2,3,4} -> min maps to 0, max to 1.
    EXPECT_NEAR(t[0], 0.0, 1e-12);
    EXPECT_NEAR(t[4], 1.0, 1e-12);
    mio_mesh_free(out);
    mio_mesh_free(m);
}

TEST(CApi, DataInfoHandle) {
    mio_mesh* m = build_data_mesh();
    mio_data_info* info = mio_data_info_create(m);
    ASSERT_NE(info, nullptr) << mio_last_error();
    const std::int64_t n = mio_data_info_count(info);
    EXPECT_EQ(n, 4);  // T, v, mat, meta

    bool saw_t = false;
    for (std::int64_t i = 0; i < n; ++i) {
        char buf[64] = {};
        const std::int64_t len = mio_data_info_name(info, i, buf, sizeof(buf));
        EXPECT_GE(len, 0);
        mio_data_array_info entry;
        ASSERT_EQ(mio_data_info_entry(info, i, &entry), MIO_OK);
        if (std::string(buf) == "T" && entry.location == MIO_DATA_POINT) {
            saw_t = true;
            EXPECT_EQ(entry.num_entries, 5);
            EXPECT_EQ(entry.num_components, 1);
            EXPECT_EQ(entry.num_finite, 5);
            EXPECT_EQ(entry.num_nan, 0);
            EXPECT_NEAR(entry.min, 0.0, 1e-12);
            EXPECT_NEAR(entry.max, 4.0, 1e-12);
            EXPECT_NEAR(entry.mean, 2.0, 1e-12);
            double cmin = 0, cmax = 0, cmean = 0;
            ASSERT_EQ(mio_data_info_component(info, i, 0, &cmin, &cmax, &cmean), MIO_OK);
            EXPECT_NEAR(cmax, 4.0, 1e-12);
            // Every out pointer is optional.
            EXPECT_EQ(mio_data_info_component(info, i, 0, nullptr, nullptr, nullptr), MIO_OK);
        }
    }
    EXPECT_TRUE(saw_t);

    // Out-of-range indices are rejected, not dereferenced.
    mio_data_array_info entry;
    EXPECT_NE(mio_data_info_entry(info, n, &entry), MIO_OK);
    EXPECT_EQ(mio_data_info_name(info, -1, nullptr, 0), -1);
    EXPECT_NE(mio_data_info_component(info, 0, 999, nullptr, nullptr, nullptr), MIO_OK);

    mio_data_info_free(info);
    mio_data_info_free(nullptr);  // must tolerate NULL
    mio_mesh_free(m);
}

TEST(CApi, DataInfoNameBufferTooSmall) {
    mio_mesh* m = build_data_mesh();
    mio_data_info* info = mio_data_info_create(m);
    ASSERT_NE(info, nullptr);
    // The required length is returned even when the buffer cannot hold it.
    char tiny[2] = {};
    const std::int64_t needed = mio_data_info_name(info, 0, tiny, sizeof(tiny));
    EXPECT_GE(needed, 0);
    EXPECT_EQ(tiny[1], '\0');  // always NUL-terminated
    mio_data_info_free(info);
    mio_mesh_free(m);
}

TEST(CApi, DataNullArgumentsAreRejected) {
    mio_mesh* m = build_data_mesh();
    const char* names[] = {"T"};
    EXPECT_EQ(mio_data_drop(nullptr, MIO_DATA_POINT, names, 1, 0), nullptr);
    EXPECT_EQ(mio_data_rename(m, MIO_DATA_POINT, nullptr, "x"), nullptr);
    EXPECT_EQ(mio_data_calc(m, nullptr, MIO_DATA_POINT, "o", 0), nullptr);
    EXPECT_EQ(mio_data_info_create(nullptr), nullptr);
    EXPECT_EQ(mio_data_info_count(nullptr), -1);
    // A NULL names array with a positive count must be caught, not dereferenced.
    EXPECT_EQ(mio_data_drop(m, MIO_DATA_POINT, nullptr, 3, 0), nullptr);
    EXPECT_STRNE(mio_last_error(), "");
    mio_mesh_free(m);
}

TEST(CApi, DataIntegrateHandle) {
    mio_mesh* m = build_data_mesh();  // cell_data "mat" = {10.0, 20.0} over 2 tets
    const int64_t region_cells[1] = {0};
    ASSERT_EQ(mio_mesh_add_region(m, "one_cell", MIO_REGION_CELL, -1, -1, region_cells, 1),
              MIO_OK);

    const char* names[] = {"mat"};
    mio_data_integrate* result = mio_data_integrate_create(m, names, 1);
    ASSERT_NE(result, nullptr) << mio_last_error();
    EXPECT_EQ(mio_data_integrate_count(result), 1);

    char buf[64] = {};
    const std::int64_t len = mio_data_integrate_name(result, 0, buf, sizeof(buf));
    EXPECT_GE(len, 0);
    EXPECT_STREQ(buf, "mat");

    mio_field_integral_info entry;
    ASSERT_EQ(mio_data_integrate_entry(result, 0, &entry), MIO_OK);
    EXPECT_EQ(entry.num_components, 1);
    EXPECT_EQ(entry.num_cells, 2);
    EXPECT_EQ(entry.num_skipped, 0);

    double total = 0, mean = 0, domain_measure = 0;
    std::int64_t num_nan = 0;
    ASSERT_EQ(mio_data_integrate_component(result, 0, 0, &total, &mean, &domain_measure, &num_nan),
              MIO_OK);
    EXPECT_GT(total, 0.0);
    EXPECT_GT(domain_measure, 0.0);
    EXPECT_NEAR(mean, total / domain_measure, 1e-9);
    EXPECT_EQ(num_nan, 0);
    // Every out pointer is optional.
    EXPECT_EQ(mio_data_integrate_component(result, 0, 0, nullptr, nullptr, nullptr, nullptr),
              MIO_OK);

    ASSERT_EQ(mio_data_integrate_region_count(result, 0), 1);
    char rbuf[64] = {};
    EXPECT_GE(mio_data_integrate_region_name(result, 0, 0, rbuf, sizeof(rbuf)), 0);
    EXPECT_STREQ(rbuf, "one_cell");
    mio_field_integral_info rentry;
    ASSERT_EQ(mio_data_integrate_region_entry(result, 0, 0, &rentry), MIO_OK);
    EXPECT_EQ(rentry.num_cells, 1);
    double rtotal = 0;
    ASSERT_EQ(mio_data_integrate_region_component(result, 0, 0, 0, &rtotal, nullptr, nullptr,
                                                   nullptr),
              MIO_OK);
    EXPECT_LT(rtotal, total);  // one cell contributes less than both

    // Out-of-range indices are rejected, not dereferenced.
    EXPECT_NE(mio_data_integrate_entry(result, 1, &entry), MIO_OK);
    EXPECT_EQ(mio_data_integrate_name(result, -1, nullptr, 0), -1);
    EXPECT_NE(mio_data_integrate_component(result, 0, 999, nullptr, nullptr, nullptr, nullptr),
              MIO_OK);
    EXPECT_NE(mio_data_integrate_region_entry(result, 0, 1, &rentry), MIO_OK);

    mio_data_integrate_free(result);
    mio_data_integrate_free(nullptr);  // must tolerate NULL

    // A point_data-only name fails, naming the fix.
    const char* point_only[] = {"T"};
    EXPECT_EQ(mio_data_integrate_create(m, point_only, 1), nullptr);
    EXPECT_STRNE(mio_last_error(), "");

    EXPECT_EQ(mio_data_integrate_create(nullptr, nullptr, 0), nullptr);
    EXPECT_EQ(mio_data_integrate_count(nullptr), -1);

    mio_mesh_free(m);
}

}  // namespace

// ---------------------------------------------------------------------------
// Selective reads (mio_read_ex) and the opaque file summary (mio_read_metadata)
// ---------------------------------------------------------------------------

TEST(CApi, WriteOptsInitMatchesPlainWrite) {
    mio_write_opts opts;
    mio_write_opts_init(&opts);
    EXPECT_EQ(opts.encoding, MIO_ENCODING_DEFAULT);
    EXPECT_EQ(opts.codec, MIO_CODEC_DEFAULT);
    EXPECT_EQ(opts.float_format, nullptr);
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(opts.reserved[i], 0) << "reserved must stay zero for ABI growth";
    // Same discipline as mio_read_opts: the tail is the growth budget, so its
    // width is part of the ABI and a field added later must come out of it.
    static_assert(sizeof(mio_write_opts::reserved) == 5 * sizeof(int64_t),
                  "mio_write_opts.reserved width is ABI");
}

TEST(CApi, WriteExHonoursEncodingAndCodec) {
    mio_mesh* m = build_tet_mesh();
    ASSERT_NE(m, nullptr);
    const std::string ascii_path = mt::temp_path("_wex_ascii.vtu");
    const std::string binary_path = mt::temp_path("_wex_binary.vtu");

    mio_write_opts opts;
    mio_write_opts_init(&opts);
    opts.encoding = MIO_ENCODING_ASCII;
    ASSERT_EQ(mio_write_ex(ascii_path.c_str(), m, "vtu", &opts), MIO_OK) << mio_last_error();

    mio_write_opts_init(&opts);
    opts.encoding = MIO_ENCODING_BINARY;
    opts.codec = MIO_CODEC_NONE;
    ASSERT_EQ(mio_write_ex(binary_path.c_str(), m, "vtu", &opts), MIO_OK) << mio_last_error();

    // The two encodings really produced different files, and both read back.
    mio_mesh* a = mio_read(ascii_path.c_str(), "vtu");
    mio_mesh* b = mio_read(binary_path.c_str(), "vtu");
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(mio_mesh_num_points(a), mio_mesh_num_points(m));
    EXPECT_EQ(mio_mesh_num_points(b), mio_mesh_num_points(m));
    mio_mesh_free(a);
    mio_mesh_free(b);
    std::remove(ascii_path.c_str());
    std::remove(binary_path.c_str());
    mio_mesh_free(m);
}

TEST(CApi, WriteExRejectsAnOptionTheFormatCannotHonour) {
    mio_mesh* m = build_tet_mesh();
    ASSERT_NE(m, nullptr);
    const std::string path = mt::temp_path("_wex_bad.msh");
    mio_write_opts opts;
    mio_write_opts_init(&opts);
    opts.codec = MIO_CODEC_ZSTD;  // gmsh has no block codec
    // Failing beats silently ignoring: asking for zstd here is a mistake.
    EXPECT_NE(mio_write_ex(path.c_str(), m, "gmsh", &opts), MIO_OK);
    EXPECT_NE(std::string(mio_last_error()).find("codec"), std::string::npos);
    std::remove(path.c_str());
    mio_mesh_free(m);
}

TEST(CApi, WriteExWithNullOptsIsPlainWrite) {
    mio_mesh* m = build_tet_mesh();
    ASSERT_NE(m, nullptr);
    const std::string path = mt::temp_path("_wex_null.vtu");
    EXPECT_EQ(mio_write_ex(path.c_str(), m, "vtu", nullptr), MIO_OK) << mio_last_error();
    std::remove(path.c_str());
    mio_mesh_free(m);
}

TEST(CApi, ReadOptsInitIsReadEverything) {
    mio_read_opts opts;
    mio_read_opts_init(&opts);
    EXPECT_EQ(opts.points_only, 0);
    EXPECT_EQ(opts.metadata_only, 0);
    EXPECT_EQ(opts.arrays, nullptr);
    EXPECT_EQ(opts.num_arrays, 0);
    EXPECT_EQ(opts.mmap_mode, 0);
    // 0 = the first step, which is the historical behaviour -- so a
    // default-initialized options struct still reads exactly what it always did.
    EXPECT_EQ(opts.time_step, 0);
    // 0 = throw on a construct the reader cannot represent, the historical
    // behaviour, so the defaults still read exactly what they always did.
    EXPECT_EQ(opts.lenient, 0);
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(opts.reserved[i], 0) << "reserved must stay zero for ABI growth";
    // `time_step` and `lenient` each took one of the six former reserved int64
    // slots rather than growing the struct, so the tail is still exactly six
    // int64s wide and a caller compiled against an older header passes a
    // correctly-sized object. Stated as the tail's width rather than sizeof(the
    // whole struct), which would be a padding assertion rather than an ABI one.
    static_assert(sizeof(mio_read_opts::time_step) + sizeof(mio_read_opts::lenient) +
                          sizeof(mio_read_opts::reserved) ==
                      6 * sizeof(std::int64_t),
                  "mio_read_opts grew: that is an ABI break, not additive growth");
}

TEST(CApi, ReadExPointsOnlyDropsDataKeepsGeometry) {
    const std::string path = mt::temp_path(".vtu");
    meshioplusplus::write_vtu(path, mt::data_mesh(), /*binary=*/true, /*zlib=*/false);

    mio_mesh* full = mio_read(path.c_str(), "vtu");
    ASSERT_NE(full, nullptr);

    mio_read_opts opts;
    mio_read_opts_init(&opts);
    opts.points_only = 1;
    mio_mesh* bare = mio_read_ex(path.c_str(), "vtu", &opts);
    ASSERT_NE(bare, nullptr) << mio_last_error();

    EXPECT_EQ(mio_mesh_num_point_data(bare), 0);
    EXPECT_EQ(mio_mesh_num_cell_data(bare), 0);
    EXPECT_GT(mio_mesh_num_point_data(full), 0);
    EXPECT_EQ(mio_mesh_num_points(bare), mio_mesh_num_points(full));
    EXPECT_EQ(mio_mesh_num_cell_blocks(bare), mio_mesh_num_cell_blocks(full));

    mio_mesh_free(bare);
    mio_mesh_free(full);
    std::filesystem::remove(path);
}

TEST(CApi, ReadExArraysSubsetAndExplicitNone) {
    const std::string path = mt::temp_path(".vtu");
    const meshioplusplus::Mesh source = mt::data_mesh();
    ASSERT_GE(source.PointDataNames().size(), 2u);
    const std::string keep = source.PointDataNames().front();
    meshioplusplus::write_vtu(path, source, /*binary=*/true, /*zlib=*/false);

    mio_read_opts opts;
    mio_read_opts_init(&opts);
    const char* names[1] = {keep.c_str()};
    opts.arrays = names;
    opts.num_arrays = 1;
    mio_mesh* subset = mio_read_ex(path.c_str(), "vtu", &opts);
    ASSERT_NE(subset, nullptr) << mio_last_error();
    EXPECT_EQ(mio_mesh_num_point_data(subset), 1);
    mio_mesh_free(subset);

    // Non-NULL pointer with count 0 means "no arrays" -- distinct from NULL.
    mio_read_opts none;
    mio_read_opts_init(&none);
    none.arrays = names;
    none.num_arrays = 0;
    mio_mesh* empty = mio_read_ex(path.c_str(), "vtu", &none);
    ASSERT_NE(empty, nullptr) << mio_last_error();
    EXPECT_EQ(mio_mesh_num_point_data(empty), 0);
    mio_mesh_free(empty);

    std::filesystem::remove(path);
}

TEST(CApi, ReadExWithNullOptsMatchesPlainRead) {
    const std::string path = mt::temp_path(".vtu");
    meshioplusplus::write_vtu(path, mt::data_mesh(), /*binary=*/true, /*zlib=*/false);

    mio_mesh* a = mio_read(path.c_str(), "vtu");
    mio_mesh* b = mio_read_ex(path.c_str(), "vtu", nullptr);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(mio_mesh_num_points(a), mio_mesh_num_points(b));
    EXPECT_EQ(mio_mesh_num_point_data(a), mio_mesh_num_point_data(b));
    mio_mesh_free(a);
    mio_mesh_free(b);
    std::filesystem::remove(path);
}

TEST(CApi, ReadMetadataReportsShapeAndNames) {
    const std::string path = mt::temp_path(".vtu");
    const meshioplusplus::Mesh source = mt::data_mesh();
    meshioplusplus::write_vtu(path, source, /*binary=*/true, /*zlib=*/false);

    mio_read_metadata* meta = mio_read_metadata_create(path.c_str(), "vtu");
    ASSERT_NE(meta, nullptr) << mio_last_error();

    EXPECT_EQ(mio_read_metadata_num_points(meta), static_cast<int64_t>(source.NumPoints()));
    EXPECT_EQ(mio_read_metadata_num_cell_blocks(meta),
              static_cast<int64_t>(source.NumCellBlocks()));
    EXPECT_GT(mio_read_metadata_num_cells(meta), 0);
    EXPECT_EQ(mio_read_metadata_fell_back(meta), 0) << "vtu has a native metadata path";

    int64_t ncells = 0, npc = 0;
    int ragged = -1;
    ASSERT_EQ(mio_read_metadata_cell_block(meta, 0, &ncells, &npc, &ragged), MIO_OK);
    EXPECT_EQ(ncells, static_cast<int64_t>(source.Cells(0).NumCells()));
    EXPECT_EQ(ragged, 0);

    char buf[64];
    const int64_t n = mio_read_metadata_cell_block_type(meta, 0, buf, sizeof(buf));
    ASSERT_GT(n, 0);
    EXPECT_EQ(std::string(buf), std::string(source.Cells(0).Type()));

    EXPECT_EQ(mio_read_metadata_num_names(meta, MIO_DATA_POINT),
              static_cast<int64_t>(source.PointDataNames().size()));
    ASSERT_EQ(mio_read_metadata_name(meta, MIO_DATA_POINT, 0, buf, sizeof(buf)),
              static_cast<int64_t>(source.PointDataNames().front().size()));
    EXPECT_EQ(std::string(buf), source.PointDataNames().front());

    // A native summary never decodes the coordinates, so it has no bbox.
    EXPECT_EQ(mio_read_metadata_bbox(meta, nullptr, nullptr), MIO_ERR_NOT_FOUND);

    mio_read_metadata_free(meta);
    std::filesystem::remove(path);
}

TEST(CApi, ReadMetadataFallbackFlagsItselfAndHasBBox) {
    const std::string path = mt::temp_path(".stl");
    meshioplusplus::write_stl(path, mt::tri_mesh(), /*binary=*/false, /*skin=*/true);

    mio_read_metadata* meta = mio_read_metadata_create(path.c_str(), "stl");
    ASSERT_NE(meta, nullptr) << mio_last_error();
    EXPECT_EQ(mio_read_metadata_fell_back(meta), 1);

    double lo[3], hi[3];
    EXPECT_EQ(mio_read_metadata_bbox(meta, lo, hi), MIO_OK);
    for (int d = 0; d < 3; ++d)
        EXPECT_LE(lo[d], hi[d]);

    mio_read_metadata_free(meta);
    std::filesystem::remove(path);
}

TEST(CApi, ReadMetadataErrorsAreGuardedNotThrown) {
    EXPECT_EQ(mio_read_metadata_create(nullptr, "vtu"), nullptr);
    EXPECT_EQ(mio_read_metadata_num_points(nullptr), -1);
    EXPECT_EQ(mio_read_metadata_fell_back(nullptr), -1);
    EXPECT_EQ(mio_read_metadata_cell_block(nullptr, 0, nullptr, nullptr, nullptr),
              MIO_ERR_INVALID_ARG);
    EXPECT_EQ(mio_reader_supports_options(nullptr), -1);
    mio_read_metadata_free(nullptr);  // NULL-safe
    EXPECT_NE(std::string(mio_last_error()), "");
}

TEST(CApi, ReaderSupportsOptions) {
    EXPECT_EQ(mio_reader_supports_options("vtu"), 1);
    EXPECT_EQ(mio_reader_supports_options("stl"), 0);
}

TEST(CApi, ConvertCellsSimplexifyThroughTheAbi) {
    // One unit-cube hexahedron -> 6 tetra.
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                                     0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    const std::vector<std::int64_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 8, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "hexahedron", 1, 8, MIO_INT64, conn.data()), MIO_OK);

    mio_convert_cells_result* r = mio_convert_cells(m, "simplexify", /*record_parent_ids=*/1);
    ASSERT_NE(r, nullptr);
    const mio_mesh* out = mio_convert_cells_result_mesh(r);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(mio_mesh_num_points(out), 8);
    ASSERT_EQ(mio_mesh_num_cell_blocks(out), 1);
    EXPECT_EQ(block_type(out, 0), "tetra");
    std::int64_t num_cells = 0, nodes_per_cell = 0;
    ASSERT_EQ(mio_mesh_cell_block_info(out, 0, &num_cells, &nodes_per_cell, nullptr), MIO_OK);
    EXPECT_EQ(num_cells, 6);
    EXPECT_EQ(nodes_per_cell, 4);

    // Zero-copy borrows of the index maps.
    const void* data = nullptr;
    mio_dtype dtype = MIO_FLOAT64;
    std::int64_t n = -1;
    ASSERT_EQ(mio_convert_cells_result_point_map(r, &data, &dtype, &n), MIO_OK);
    EXPECT_EQ(dtype, MIO_INT64);
    EXPECT_EQ(n, 8);
    EXPECT_EQ(static_cast<const std::int64_t*>(data)[0], 0);

    ASSERT_EQ(mio_convert_cells_result_num_cell_maps(r), 1);
    ASSERT_EQ(mio_convert_cells_result_cell_map(r, 0, &data, &dtype, &n), MIO_OK);
    EXPECT_EQ(n, 1);
    EXPECT_EQ(static_cast<const std::int64_t*>(data)[0], 0);  // parent 0's first child

    mio_mesh* owned = mio_convert_cells_result_take_mesh(r);
    ASSERT_NE(owned, nullptr);
    EXPECT_EQ(mio_mesh_num_points(owned), 8);
    mio_mesh_free(owned);
    mio_convert_cells_result_free(r);
    mio_mesh_free(m);
}

TEST(CApi, ConvertCellsLinearizeAndElevateRoundTrip) {
    mio_mesh* m = build_tet_mesh();

    mio_convert_cells_result* up = mio_convert_cells(m, "elevate", 0);
    ASSERT_NE(up, nullptr);
    const mio_mesh* quad = mio_convert_cells_result_mesh(up);
    EXPECT_EQ(block_type(quad, 0), "tetra10");
    EXPECT_GT(mio_mesh_num_points(quad), 5);

    mio_convert_cells_result* down = mio_convert_cells(quad, "linearize", 0);
    ASSERT_NE(down, nullptr);
    const mio_mesh* back = mio_convert_cells_result_mesh(down);
    EXPECT_EQ(block_type(back, 0), "tetra");
    EXPECT_EQ(mio_mesh_num_points(back), 5);

    mio_convert_cells_result_free(down);
    mio_convert_cells_result_free(up);
    mio_mesh_free(m);
}

TEST(CApi, ConvertCellsErrorsAreGuardedNotThrown) {
    mio_mesh* m = build_tet_mesh();
    // An unknown mode fails through the status/last-error contract, not by
    // letting the C++ exception escape the ABI.
    EXPECT_EQ(mio_convert_cells(m, "nope", 0), nullptr);
    EXPECT_NE(std::string(mio_last_error()), "");
    EXPECT_EQ(mio_convert_cells(nullptr, "linearize", 0), nullptr);
    EXPECT_EQ(mio_convert_cells(m, nullptr, 0), nullptr);
    EXPECT_EQ(mio_convert_cells_result_mesh(nullptr), nullptr);
    EXPECT_EQ(mio_convert_cells_result_num_cell_maps(nullptr), -1);
    EXPECT_EQ(mio_convert_cells_result_point_map(nullptr, nullptr, nullptr, nullptr),
              MIO_ERR_INVALID_ARG);
    mio_convert_cells_result_free(nullptr);  // NULL-safe
    mio_mesh_free(m);
}

TEST(CApi, SubdivideHexahedronThroughTheAbi) {
    // One unit-cube hexahedron -> 6 polyhedral children, one per face.
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                                     0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    const std::vector<std::int64_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 8, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "hexahedron", 1, 8, MIO_INT64, conn.data()), MIO_OK);

    mio_subdivide_result* r = mio_subdivide(m, /*record_parent_ids=*/1);
    ASSERT_NE(r, nullptr);
    const mio_mesh* out = mio_subdivide_result_mesh(r);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(mio_mesh_num_points(out), 9);  // one new interior point (apex)
    ASSERT_EQ(mio_mesh_num_cell_blocks(out), 1);
    EXPECT_EQ(block_type(out, 0), "polyhedron");

    mio_cell_block_info info{};
    ASSERT_EQ(mio_mesh_cell_block_info_ex(out, 0, &info), MIO_OK);
    EXPECT_EQ(info.num_cells, 6);
    EXPECT_EQ(info.is_ragged, 1);
    EXPECT_EQ(info.is_polyhedron, 1);

    // Zero-copy borrow of the cell map -- there is no point map (subdivide
    // never prunes or renumbers an original point).
    const void* data = nullptr;
    mio_dtype dtype = MIO_FLOAT64;
    std::int64_t n = -1;
    ASSERT_EQ(mio_subdivide_result_num_cell_maps(r), 1);
    ASSERT_EQ(mio_subdivide_result_cell_map(r, 0, &data, &dtype, &n), MIO_OK);
    EXPECT_EQ(dtype, MIO_INT64);
    EXPECT_EQ(n, 1);
    EXPECT_EQ(static_cast<const std::int64_t*>(data)[0], 0);

    mio_mesh* owned = mio_subdivide_result_take_mesh(r);
    ASSERT_NE(owned, nullptr);
    EXPECT_EQ(mio_mesh_num_points(owned), 9);
    mio_mesh_free(owned);
    mio_subdivide_result_free(r);
    mio_mesh_free(m);
}

TEST(CApi, SubdivideErrorsAreGuardedNotThrown) {
    mio_mesh* m = build_tet_mesh();
    EXPECT_EQ(mio_subdivide(nullptr, 0), nullptr);
    EXPECT_NE(std::string(mio_last_error()), "");
    EXPECT_EQ(mio_subdivide_result_mesh(nullptr), nullptr);
    EXPECT_EQ(mio_subdivide_result_num_cell_maps(nullptr), -1);
    EXPECT_EQ(mio_subdivide_result_cell_map(nullptr, 0, nullptr, nullptr, nullptr),
              MIO_ERR_INVALID_ARG);
    mio_subdivide_result_free(nullptr);  // NULL-safe
    mio_mesh_free(m);
}

TEST(CApi, AgglomerateTwoHexesThroughTheAbi) {
    // Two unit hexahedra sharing one face (x=1 plane).
    const std::vector<double> pts = {0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
                                     1, 1, 1, 1, 0, 1, 2, 0, 0, 2, 1, 0, 2, 1, 1, 2, 0, 1};
    const std::vector<std::int64_t> conn = {0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9, 10, 11};
    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 12, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "hexahedron", 2, 8, MIO_INT64, conn.data()), MIO_OK);

    mio_agglomerate_result* r = mio_agglomerate(m, /*target_group_size=*/2);
    ASSERT_NE(r, nullptr);
    const mio_mesh* out = mio_agglomerate_result_mesh(r);
    ASSERT_NE(out, nullptr);
    ASSERT_EQ(mio_mesh_num_cell_blocks(out), 1);
    EXPECT_EQ(block_type(out, 0), "polyhedron");

    mio_cell_block_info info{};
    ASSERT_EQ(mio_mesh_cell_block_info_ex(out, 0, &info), MIO_OK);
    EXPECT_EQ(info.num_cells, 1);
    EXPECT_EQ(info.is_polyhedron, 1);

    // Zero-copy borrow of the FLAT cell map -- both hexes land in the one
    // merged cell.
    const void* data = nullptr;
    mio_dtype dtype = MIO_FLOAT64;
    std::int64_t n = -1;
    ASSERT_EQ(mio_agglomerate_result_cell_map(r, &data, &dtype, &n), MIO_OK);
    EXPECT_EQ(dtype, MIO_INT64);
    EXPECT_EQ(n, 2);
    const auto* cm = static_cast<const std::int64_t*>(data);
    EXPECT_EQ(cm[0], cm[1]);

    mio_mesh* owned = mio_agglomerate_result_take_mesh(r);
    ASSERT_NE(owned, nullptr);
    EXPECT_EQ(mio_mesh_num_cell_blocks(owned), 1);
    mio_mesh_free(owned);
    mio_agglomerate_result_free(r);
    mio_mesh_free(m);
}

TEST(CApi, AgglomerateErrorsAreGuardedNotThrown) {
    mio_mesh* m = build_tet_mesh();
    EXPECT_EQ(mio_agglomerate(nullptr, 8), nullptr);
    EXPECT_NE(std::string(mio_last_error()), "");
    EXPECT_EQ(mio_agglomerate(m, -1), nullptr);
    EXPECT_EQ(mio_agglomerate_result_mesh(nullptr), nullptr);
    EXPECT_EQ(mio_agglomerate_result_cell_map(nullptr, nullptr, nullptr, nullptr),
              MIO_ERR_INVALID_ARG);
    mio_agglomerate_result_free(nullptr);  // NULL-safe
    mio_mesh_free(m);
}

TEST(CApi, SmoothMovesInteriorAndPinsBoundary) {
    // A 3x3 grid of quads: only the centre node is interior, so it is the only
    // one free to move -- which makes both halves of the assertion sharp.
    std::vector<double> pts;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            pts.insert(pts.end(), {static_cast<double>(i), static_cast<double>(j), 0.0});
    pts[4 * 3 + 0] += 0.4;  // push the centre node off-centre
    pts[4 * 3 + 1] -= 0.3;
    const std::vector<std::int64_t> conn = {0, 3, 4, 1, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5};

    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 9, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "quad", 4, 4, MIO_INT64, conn.data()), MIO_OK);

    std::int64_t moved = -1, skipped = -1;
    double max_disp = -1.0;
    mio_mesh* out =
        mio_smooth(m, "laplacian", /*iterations=*/10, /*lambda=*/-1.0, /*mu=*/-0.34,
                   /*fix_boundary=*/1, /*preserve_features=*/1,
                   /*feature_angle=*/30.0, /*guard_inversion=*/1, &moved, &max_disp, &skipped);
    ASSERT_NE(out, nullptr);

    // Geometry only: same counts, same connectivity.
    EXPECT_EQ(mio_mesh_num_points(out), 9);
    ASSERT_EQ(mio_mesh_num_cell_blocks(out), 1);
    EXPECT_EQ(block_type(out, 0), "quad");

    EXPECT_EQ(moved, 1);  // exactly the one interior node
    EXPECT_GT(max_disp, 0.0);
    EXPECT_EQ(skipped, 0);

    // The centre node was pulled back toward (1, 1); the corners never moved.
    const void* data = nullptr;
    mio_dtype dtype = MIO_INT64;
    ASSERT_EQ(mio_mesh_get_points(out, &data, &dtype), MIO_OK);
    const double* p = static_cast<const double*>(data);
    // Laplacian converges geometrically: after 10 passes at lambda = 0.5 the
    // residual is the initial 0.4 offset times 0.5^10, i.e. ~3.9e-4.
    EXPECT_NEAR(p[4 * 3 + 0], 1.0, 1e-3);
    EXPECT_NEAR(p[4 * 3 + 1], 1.0, 1e-3);
    EXPECT_EQ(p[0], 0.0);
    EXPECT_EQ(p[8 * 3 + 0], 2.0);

    // A NULL method defaults to taubin rather than failing, and the counter
    // out-params are all individually optional.
    mio_mesh* dflt =
        mio_smooth(m, nullptr, 3, -1.0, -0.34, 1, 1, 30.0, 1, nullptr, nullptr, nullptr);
    EXPECT_NE(dflt, nullptr);
    mio_mesh_free(dflt);

    mio_mesh_free(out);
    mio_mesh_free(m);
}

TEST(CApi, SmoothOdtMovesATetVertexTowardItsCircumcenter) {
    // A single, non-degenerate tet with one free vertex: ODT's closed-form
    // update pulls it toward the tet's own circumcenter -- mio_smooth's
    // method string is passed through to smooth_method_from_name unchanged,
    // so "odt" needs no C API surface of its own beyond the string.
    const std::vector<double> pts = {0, 0, 0, 4, 0, 0, 0, 4, 0, 1, 1, 5};
    const std::vector<std::int64_t> conn = {0, 1, 2, 3};
    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 4, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "tetra", 1, 4, MIO_INT64, conn.data()), MIO_OK);

    std::int64_t moved = -1, skipped = -1;
    double max_disp = -1.0;
    mio_mesh* out = mio_smooth(m, "odt", /*iterations=*/1, /*lambda=*/-1.0, /*mu=*/-0.34,
                               /*fix_boundary=*/0, /*preserve_features=*/0,
                               /*feature_angle=*/30.0, /*guard_inversion=*/1, &moved, &max_disp,
                               &skipped);
    ASSERT_NE(out, nullptr) << mio_last_error();
    EXPECT_EQ(mio_mesh_num_points(out), 4);
    EXPECT_GT(moved, 0);
    EXPECT_GT(max_disp, 0.0);
    mio_mesh_free(out);
    mio_mesh_free(m);
}

TEST(CApi, SmoothRejectsBadArguments) {
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
    const std::vector<std::int64_t> conn = {0, 1, 2, 3};
    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 4, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "quad", 1, 4, MIO_INT64, conn.data()), MIO_OK);

    // Unknown method, lambda out of (0, 1), and a taubin mu that would amplify.
    EXPECT_EQ(mio_smooth(m, "bogus", 1, -1.0, -0.34, 1, 1, 30.0, 1, nullptr, nullptr, nullptr),
              nullptr);
    EXPECT_NE(std::string(mio_last_error()), "");
    EXPECT_EQ(mio_smooth(m, "taubin", 1, 1.5, -0.34, 1, 1, 30.0, 1, nullptr, nullptr, nullptr),
              nullptr);
    EXPECT_EQ(mio_smooth(m, "taubin", 1, 0.4, -0.2, 1, 1, 30.0, 1, nullptr, nullptr, nullptr),
              nullptr);
    // No exception may cross the ABI for a NULL mesh either.
    EXPECT_EQ(
        mio_smooth(nullptr, "taubin", 1, -1.0, -0.34, 1, 1, 30.0, 1, nullptr, nullptr, nullptr),
        nullptr);
    EXPECT_NE(std::string(mio_last_error()), "");

    mio_mesh_free(m);
}

TEST(CApi, OptimizeVolumeFlipsAndImproves) {
    // The 2-tet fixture whose 2-3 flip raises the worst scaled Jacobian from
    // ~0.066 to ~0.362 (see test_optimize_volume.cpp). relocate off so only the
    // flip acts; every out-param is exercised.
    const std::vector<double> pts = {0.0,    0.0,    0.0,    1.0,    0.0,   0.0,   0.3,   0.9,
                                     0.0,    0.2252, 0.2808, 0.6977, 0.4659, 0.3149, -0.0367};
    const std::vector<std::int64_t> conn = {0, 1, 2, 3, 0, 1, 2, 4};
    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 5, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "tetra", 2, 4, MIO_INT64, conn.data()), MIO_OK);

    std::int64_t nflips = -1, n23 = -1, n32 = -1, moved = -1, ntets = -1;
    double qbefore = -1.0, qafter = -1.0;
    mio_mesh* out = mio_optimize_volume(m, /*max_iterations=*/5, /*relocate=*/0, /*flip=*/1,
                                        /*preserve_boundary=*/1, /*min_improvement=*/1e-6, &nflips,
                                        &n23, &n32, &moved, &ntets, &qbefore, &qafter);
    ASSERT_NE(out, nullptr) << mio_last_error();
    EXPECT_EQ(n23, 1);
    EXPECT_EQ(n32, 0);
    EXPECT_EQ(nflips, 1);
    EXPECT_EQ(ntets, 3);
    EXPECT_EQ(mio_mesh_num_points(out), 5);  // point set invariant
    EXPECT_GT(qafter, qbefore + 0.2);
    mio_mesh_free(out);

    // A non-tet mesh is refused by name; no exception crosses the ABI.
    mio_mesh* q = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(q, MIO_FLOAT64, 4, 3, pts.data()), MIO_OK);
    const std::vector<std::int64_t> quad = {0, 1, 2, 3};
    ASSERT_EQ(mio_mesh_add_cell_block(q, "quad", 1, 4, MIO_INT64, quad.data()), MIO_OK);
    EXPECT_EQ(mio_optimize_volume(q, 5, 1, 1, 1, 1e-6, nullptr, nullptr, nullptr, nullptr, nullptr,
                                  nullptr, nullptr),
              nullptr);
    EXPECT_NE(std::string(mio_last_error()), "");
    mio_mesh_free(q);
    mio_mesh_free(m);
}

TEST(CApi, RefineHexIntoEight) {
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                                     0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    const std::vector<std::int64_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 8, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "hexahedron", 1, 8, MIO_INT64, conn.data()), MIO_OK);

    mio_refine_result* r = mio_refine(m, /*levels=*/1, /*record_parent_ids=*/1);
    ASSERT_NE(r, nullptr);
    const mio_mesh* out = mio_refine_result_mesh(r);
    ASSERT_NE(out, nullptr);
    // 8 corners + 12 edge mids + 6 face centres + 1 body.
    EXPECT_EQ(mio_mesh_num_points(out), 27);
    ASSERT_EQ(mio_mesh_num_cell_blocks(out), 1);
    EXPECT_EQ(block_type(out, 0), "hexahedron");
    std::int64_t num_cells = 0, nodes_per_cell = 0;
    ASSERT_EQ(mio_mesh_cell_block_info(out, 0, &num_cells, &nodes_per_cell, nullptr), MIO_OK);
    EXPECT_EQ(num_cells, 8);
    EXPECT_EQ(nodes_per_cell, 8);

    // Zero-copy borrows of the index maps.
    const void* data = nullptr;
    mio_dtype dtype = MIO_FLOAT64;
    std::int64_t n = -1;
    ASSERT_EQ(mio_refine_result_point_map(r, &data, &dtype, &n), MIO_OK);
    EXPECT_EQ(dtype, MIO_INT64);
    EXPECT_EQ(n, 8);
    EXPECT_EQ(static_cast<const std::int64_t*>(data)[0], 0);

    ASSERT_EQ(mio_refine_result_num_cell_maps(r), 1);
    ASSERT_EQ(mio_refine_result_cell_map(r, 0, &data, &dtype, &n), MIO_OK);
    EXPECT_EQ(n, 1);
    EXPECT_EQ(static_cast<const std::int64_t*>(data)[0], 0);  // parent 0's first child

    mio_mesh* owned = mio_refine_result_take_mesh(r);
    ASSERT_NE(owned, nullptr);
    EXPECT_EQ(mio_mesh_num_points(owned), 27);
    mio_mesh_free(owned);
    mio_refine_result_free(r);
    mio_mesh_free(m);
}

TEST(CApi, RefineExWithDefaultsMatchesMioRefine) {
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                                     0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    const std::vector<std::int64_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 8, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "hexahedron", 1, 8, MIO_INT64, conn.data()), MIO_OK);

    // A NULL options pointer and a zero-initialized struct must both reproduce
    // mio_refine(mesh, 1, 0) exactly -- the append-only-tail contract.
    mio_refine_opts opts;
    mio_refine_opts_init(&opts);
    mio_refine_result* a = mio_refine(m, 1, 0);
    mio_refine_result* b = mio_refine_ex(m, &opts);
    mio_refine_result* c = mio_refine_ex(m, nullptr);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(mio_mesh_num_points(mio_refine_result_mesh(a)), 27);
    EXPECT_EQ(mio_mesh_num_points(mio_refine_result_mesh(b)), 27);
    EXPECT_EQ(mio_mesh_num_points(mio_refine_result_mesh(c)), 27);
    mio_refine_result_free(a);
    mio_refine_result_free(b);
    mio_refine_result_free(c);
    mio_mesh_free(m);
}

TEST(CApi, RefineExRecordHierarchyAttachesCellIdAndParentId) {
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                                     0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    const std::vector<std::int64_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 8, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "hexahedron", 1, 8, MIO_INT64, conn.data()), MIO_OK);

    mio_refine_opts opts;
    mio_refine_opts_init(&opts);
    opts.record_hierarchy = 1;
    mio_refine_result* r = mio_refine_ex(m, &opts);
    ASSERT_NE(r, nullptr);
    const mio_mesh* out = mio_refine_result_mesh(r);

    const void* data = nullptr;
    mio_dtype dt;
    int32_t ndim = 0;
    int64_t shape[8] = {};
    ASSERT_EQ(mio_mesh_get_cell_data(out, "refine:cell_id", 0, &data, &dt, &ndim, shape), MIO_OK);
    EXPECT_EQ(dt, MIO_INT64);
    EXPECT_EQ(shape[0], 8);  // one hexahedron -> 8 children, uniform refinement
    ASSERT_EQ(mio_mesh_get_cell_data(out, "refine:parent_id", 0, &data, &dt, &ndim, shape), MIO_OK);
    const std::int64_t* parents = static_cast<const std::int64_t*>(data);
    for (int64_t i = 0; i < shape[0]; ++i)
        EXPECT_EQ(parents[i], 0) << "every child of the sole input cell shares parent 0";

    // Also proves the multigrid-stencil fix: RedGreen leaves no hanging nodes,
    // so refine:entity would normally never be attached at all.
    EXPECT_EQ(mio_mesh_get_point_data(out, "refine:entity", &data, &dt, &ndim, shape), MIO_OK);

    mio_refine_result_free(r);
    mio_mesh_free(m);
}

TEST(CApi, RefineExSelectsASubsetAndClosesUp) {
    // A 3 x 3 grid of quadrilaterals; refine the middle one only.
    std::vector<double> pts;
    for (int j = 0; j <= 3; ++j)
        for (int i = 0; i <= 3; ++i) {
            pts.push_back(i);
            pts.push_back(j);
            pts.push_back(0);
        }
    std::vector<std::int64_t> conn;
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) {
            const std::int64_t a = j * 4 + i;
            conn.insert(conn.end(), {a, a + 1, a + 5, a + 4});
        }
    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 16, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "quad", 9, 4, MIO_INT64, conn.data()), MIO_OK);

    const std::int64_t selected[] = {4};
    mio_refine_opts opts;
    mio_refine_opts_init(&opts);
    opts.cells = selected;
    opts.num_cells = 1;
    opts.record_levels = 1;
    mio_refine_result* r = mio_refine_ex(m, &opts);
    ASSERT_NE(r, nullptr);
    std::int64_t num_cells = 0, npc = 0;
    ASSERT_EQ(mio_mesh_cell_block_info(mio_refine_result_mesh(r), 0, &num_cells, &npc, nullptr),
              MIO_OK);
    // More than the input, far fewer than the 36 a uniform refinement gives.
    EXPECT_GT(num_cells, 9);
    EXPECT_LT(num_cells, 36);
    EXPECT_EQ(npc, 4) << "green quadrilaterals stay quadrilaterals";
    mio_refine_result_free(r);

    // Two selectors at once is an error, reported rather than thrown.
    opts.region = "anything";
    EXPECT_EQ(mio_refine_ex(m, &opts), nullptr);
    EXPECT_NE(mio_last_error(), nullptr);
    mio_mesh_free(m);
}

TEST(CApi, UndoGreenRestoresTheCoarseParentThroughTheAbi) {
    // A 3 x 3 grid of quadrilaterals; refine the middle one only, so the
    // neighbours around it get transitional (green) closures.
    std::vector<double> pts;
    for (int j = 0; j <= 3; ++j)
        for (int i = 0; i <= 3; ++i) {
            pts.push_back(i);
            pts.push_back(j);
            pts.push_back(0);
        }
    std::vector<std::int64_t> conn;
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) {
            const std::int64_t a = j * 4 + i;
            conn.insert(conn.end(), {a, a + 1, a + 5, a + 4});
        }
    mio_mesh* coarse = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(coarse, MIO_FLOAT64, 16, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(coarse, "quad", 9, 4, MIO_INT64, conn.data()), MIO_OK);

    const std::int64_t selected[] = {4};
    mio_refine_opts opts;
    mio_refine_opts_init(&opts);
    opts.cells = selected;
    opts.num_cells = 1;
    opts.record_hierarchy = 1;
    opts.record_levels = 1;
    mio_refine_result* r = mio_refine_ex(coarse, &opts);
    ASSERT_NE(r, nullptr);
    const mio_mesh* fine = mio_refine_result_mesh(r);

    std::int64_t fine_cells = 0;
    ASSERT_EQ(mio_mesh_cell_block_info(fine, 0, &fine_cells, nullptr, nullptr), MIO_OK);

    std::int64_t num_groups_undone = -1, num_cells_removed = -1;
    mio_mesh* undone = mio_undo_green(coarse, fine, &num_groups_undone, &num_cells_removed);
    ASSERT_NE(undone, nullptr);
    EXPECT_GT(num_groups_undone, 0);
    EXPECT_GT(num_cells_removed, 0);

    std::int64_t undone_cells = 0, coarse_cells = 0;
    ASSERT_EQ(mio_mesh_cell_block_info(undone, 0, &undone_cells, nullptr, nullptr), MIO_OK);
    ASSERT_EQ(mio_mesh_cell_block_info(coarse, 0, &coarse_cells, nullptr, nullptr), MIO_OK);
    EXPECT_LT(undone_cells, fine_cells);
    EXPECT_GT(undone_cells, coarse_cells);

    // The reserved refine:* arrays are dropped entirely.
    const void* data = nullptr;
    mio_dtype dt;
    int32_t ndim = 0;
    int64_t shape[8] = {};
    EXPECT_NE(mio_mesh_get_cell_data(undone, "refine:cell_id", 0, &data, &dt, &ndim, shape),
              MIO_OK);
    EXPECT_NE(mio_mesh_get_point_data(undone, "refine:entity", &data, &dt, &ndim, shape), MIO_OK);

    // Nullable counters really are nullable.
    mio_mesh* undone2 = mio_undo_green(coarse, fine, nullptr, nullptr);
    ASSERT_NE(undone2, nullptr);
    mio_mesh_free(undone2);

    mio_mesh_free(undone);
    mio_refine_result_free(r);
    mio_mesh_free(coarse);
}

TEST(CApi, UndoGreenErrorsAreGuardedNotThrown) {
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
    const std::vector<std::int64_t> conn = {0, 1, 2, 3};
    mio_mesh* coarse = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(coarse, MIO_FLOAT64, 4, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(coarse, "quad", 1, 4, MIO_INT64, conn.data()), MIO_OK);

    // No hierarchy at all on "fine" (here just the coarse mesh itself).
    EXPECT_EQ(mio_undo_green(coarse, coarse, nullptr, nullptr), nullptr);
    EXPECT_NE(std::string(mio_last_error()), "");

    // NULL meshes never let an exception cross the ABI.
    EXPECT_EQ(mio_undo_green(nullptr, coarse, nullptr, nullptr), nullptr);
    EXPECT_NE(std::string(mio_last_error()), "");
    EXPECT_EQ(mio_undo_green(coarse, nullptr, nullptr, nullptr), nullptr);
    EXPECT_NE(std::string(mio_last_error()), "");

    mio_mesh_free(coarse);
}

TEST(CApi, RefineChainsOnABorrowedMesh) {
    mio_mesh* m = build_tet_mesh();  // 2 tetrahedra

    mio_refine_result* one = mio_refine(m, 1, 0);
    ASSERT_NE(one, nullptr);
    const mio_mesh* once = mio_refine_result_mesh(one);
    EXPECT_EQ(block_type(once, 0), "tetra");

    // Feed the borrowed mesh straight back in: levels=1 twice == levels=2.
    mio_refine_result* two = mio_refine(once, 1, 0);
    ASSERT_NE(two, nullptr);
    mio_refine_result* direct = mio_refine(m, 2, 0);
    ASSERT_NE(direct, nullptr);
    EXPECT_EQ(mio_mesh_num_points(mio_refine_result_mesh(two)),
              mio_mesh_num_points(mio_refine_result_mesh(direct)));

    mio_refine_result_free(direct);
    mio_refine_result_free(two);
    mio_refine_result_free(one);
    mio_mesh_free(m);
}

TEST(CApi, PartitionPiecesAndMaps) {
    // A 2x2 quad grid split into 2 parts: exactly nparts pieces, blocks kept
    // 1:1, cell maps assign every input cell to exactly one piece.
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 1, 1,
                                     0, 2, 1, 0, 0, 2, 0, 1, 2, 0, 2, 2, 0};
    const std::vector<std::int64_t> conn = {0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7};
    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 9, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "quad", 4, 4, MIO_INT64, conn.data()), MIO_OK);

    mio_partition_result* r =
        mio_partition(m, 2, "sfc", 0.03, "eco", 0, /*record_ids=*/1, /*ghost_layers=*/0, "");
    ASSERT_NE(r, nullptr);
    ASSERT_EQ(mio_partition_result_num_pieces(r), 2);
    ASSERT_EQ(mio_partition_result_num_cell_maps(r), 1);

    std::int64_t total_out = 0;
    std::vector<int> owners(4, 0);
    for (std::int64_t i = 0; i < 2; ++i) {
        EXPECT_EQ(mio_partition_result_part_id(r, i), static_cast<int>(i));
        const mio_mesh* piece = mio_partition_result_mesh(r, i);
        ASSERT_NE(piece, nullptr);
        ASSERT_EQ(mio_mesh_num_cell_blocks(piece), 1);  // blocks kept 1:1
        std::int64_t num_cells = 0;
        ASSERT_EQ(mio_mesh_cell_block_info(piece, 0, &num_cells, nullptr, nullptr), MIO_OK);
        total_out += num_cells;

        const void* data = nullptr;
        mio_dtype dtype = MIO_FLOAT64;
        std::int64_t n = -1;
        ASSERT_EQ(mio_partition_result_point_map(r, i, &data, &dtype, &n), MIO_OK);
        EXPECT_EQ(dtype, MIO_INT64);
        EXPECT_EQ(n, 9);
        ASSERT_EQ(mio_partition_result_cell_map(r, i, 0, &data, &dtype, &n), MIO_OK);
        EXPECT_EQ(n, 4);
        for (std::int64_t c = 0; c < n; ++c)
            if (static_cast<const std::int64_t*>(data)[c] >= 0)
                ++owners[static_cast<std::size_t>(c)];
    }
    EXPECT_EQ(total_out, 4);
    for (int o : owners)
        EXPECT_EQ(o, 1);  // partition of unity

    mio_mesh* owned = mio_partition_result_take_mesh(r, 0);
    ASSERT_NE(owned, nullptr);
    mio_mesh_free(owned);
    mio_partition_result_free(r);
    mio_mesh_free(m);
}

TEST(CApi, PartitionLabelsFillsTheCallerBuffer) {
    mio_mesh* m = build_tet_mesh();  // 2 tetrahedra
    std::vector<std::int64_t> labels(2, -1);
    ASSERT_EQ(mio_partition_labels(m, 2, "sfc", 0.03, "eco", 0, "", labels.data(),
                                   static_cast<std::int64_t>(labels.size())),
              MIO_OK);
    for (std::int64_t v : labels) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 2);
    }
    // A wrong buffer size is rejected with a named message, not written past.
    EXPECT_EQ(mio_partition_labels(m, 2, "sfc", 0.03, "eco", 0, "", labels.data(), 5),
              MIO_ERR_INVALID_ARG);
    EXPECT_NE(std::string(mio_last_error()).find("labels_size"), std::string::npos);
    mio_mesh_free(m);
}

TEST(CApi, PartitionGhostLayers) {
    mio_mesh* m = build_tet_mesh();
    ASSERT_NE(m, nullptr);
    mio_partition_result* plain = mio_partition(m, 2, "sfc", 0.03, "eco", 0, 0, 0, "");
    mio_partition_result* halo = mio_partition(m, 2, "sfc", 0.03, "eco", 0, 0, 1, "");
    ASSERT_NE(plain, nullptr) << mio_last_error();
    ASSERT_NE(halo, nullptr) << mio_last_error();
    ASSERT_EQ(mio_partition_result_num_pieces(halo), mio_partition_result_num_pieces(plain));

    // With only two tetras sharing a face, one ghost layer makes each piece the
    // whole mesh -- and the tag must come along to say which cell is owned.
    const mio_mesh* piece = mio_partition_result_mesh(halo, 0);
    ASSERT_NE(piece, nullptr);
    EXPECT_GE(mio_mesh_num_cell_blocks(piece), 1);
    const void* data = nullptr;
    mio_dtype dt{};
    int32_t ndim = 0;
    int64_t shape[8]{};
    ASSERT_EQ(mio_mesh_get_cell_data(piece, "partition:ghost", 0, &data, &dt, &ndim, shape), MIO_OK)
        << mio_last_error();
    EXPECT_EQ(dt, MIO_INT64);

    mio_partition_result_free(plain);
    mio_partition_result_free(halo);
    mio_mesh_free(m);
}

TEST(CApi, PartitionErrorsAreGuardedNotThrown) {
    mio_mesh* m = build_tet_mesh();
    // Bad method name, bad nparts, negative ghost layers: NULL + last_error,
    // no throw. (A POSITIVE ghost_layers is a supported request since v9.0.0
    // and is exercised in PartitionGhostLayers below.)
    EXPECT_EQ(mio_partition(m, 2, "metis", 0.03, "eco", 0, 0, 0, ""), nullptr);
    EXPECT_NE(std::string(mio_last_error()), "");
    EXPECT_EQ(mio_partition(m, 0, "sfc", 0.03, "eco", 0, 0, 0, ""), nullptr);
    EXPECT_EQ(mio_partition(m, 2, "sfc", 0.03, "eco", 0, 0, /*ghost_layers=*/-1, ""), nullptr);
    EXPECT_NE(std::string(mio_last_error()).find("ghost_layers"), std::string::npos);
    EXPECT_EQ(mio_partition(nullptr, 2, "sfc", 0.03, "eco", 0, 0, 0, ""), nullptr);
    EXPECT_EQ(mio_partition_result_num_pieces(nullptr), -1);
    EXPECT_EQ(mio_partition_result_part_id(nullptr, 0), -1);
    EXPECT_EQ(mio_partition_result_mesh(nullptr, 0), nullptr);
    EXPECT_EQ(mio_partition_result_point_map(nullptr, 0, nullptr, nullptr, nullptr),
              MIO_ERR_INVALID_ARG);
    EXPECT_EQ(mio_partition_result_cell_map(nullptr, 0, 0, nullptr, nullptr, nullptr),
              MIO_ERR_INVALID_ARG);
    mio_partition_result_free(nullptr);  // NULL-safe
    EXPECT_EQ(mio_partition_labels(nullptr, 2, "sfc", 0.03, "eco", 0, "", nullptr, 0),
              MIO_ERR_INVALID_ARG);
    mio_mesh_free(m);
}

TEST(CApi, RefineErrorsAreGuardedNotThrown) {
    // A pyramid has no same-type subdivision: the C++ exception must be turned
    // into NULL + last_error, never allowed to escape the ABI.
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0.5, 0.5, 1};
    const std::vector<std::int64_t> conn = {0, 1, 2, 3, 4};
    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 5, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "pyramid", 1, 5, MIO_INT64, conn.data()), MIO_OK);

    EXPECT_EQ(mio_refine(m, 1, 0), nullptr);
    EXPECT_NE(std::string(mio_last_error()), "");
    EXPECT_EQ(mio_refine(nullptr, 1, 0), nullptr);
    EXPECT_EQ(mio_refine_result_mesh(nullptr), nullptr);
    EXPECT_EQ(mio_refine_result_num_cell_maps(nullptr), -1);
    EXPECT_EQ(mio_refine_result_point_map(nullptr, nullptr, nullptr, nullptr), MIO_ERR_INVALID_ARG);
    EXPECT_EQ(mio_refine_result_cell_map(nullptr, 0, nullptr, nullptr, nullptr),
              MIO_ERR_INVALID_ARG);
    mio_refine_result_free(nullptr);  // NULL-safe
    mio_mesh_free(m);
}

TEST(CApi, IsosurfaceContoursAScalarPointField) {
    // The unit cube with f = x on its corners: the 0.5 level set is the unit
    // square at x = 0.5, on which f reads back as exactly the isovalue.
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                                     0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    const std::vector<std::int64_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<double> fx = {0, 1, 1, 0, 0, 1, 1, 0};
    const std::int64_t shape[1] = {8};

    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 8, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "hexahedron", 1, 8, MIO_INT64, conn.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_point_data(m, "fx", MIO_FLOAT64, 1, shape, fx.data()), MIO_OK);

    const double isovalues[2] = {0.25, 0.75};
    mio_mesh* out = mio_isosurface(m, "fx", isovalues, 1, /*component=*/-1,
                                   /*record_parent_ids=*/1);
    ASSERT_NE(out, nullptr);
    ASSERT_GT(mio_mesh_num_points(out), 0);
    ASSERT_GT(mio_mesh_num_cell_blocks(out), 0);

    const void* data = nullptr;
    mio_dtype dt = MIO_FLOAT64;
    std::int32_t ndim = 0;
    std::int64_t got_shape[MIO_MAX_NDIM] = {0};
    ASSERT_EQ(mio_mesh_get_point_data(out, "fx", &data, &dt, &ndim, got_shape), MIO_OK);
    ASSERT_EQ(dt, MIO_FLOAT64);
    const double* v = static_cast<const double*>(data);
    for (std::int64_t i = 0; i < got_shape[0]; ++i)
        EXPECT_EQ(v[i], 0.25) << "the contoured field must be exact";

    // Every contour cell is tagged with its value and its ordinal.
    EXPECT_GT(mio_mesh_cell_data_num_blocks(out, "iso:value"), 0);
    EXPECT_GT(mio_mesh_cell_data_num_blocks(out, "iso:index"), 0);
    EXPECT_GT(mio_mesh_cell_data_num_blocks(out, "iso:parent_cell"), 0);
    mio_mesh_free(out);

    // Two isovalues land in one mesh.
    mio_mesh* both = mio_isosurface(m, "fx", isovalues, 2, -1, 0);
    ASSERT_NE(both, nullptr);
    ASSERT_EQ(mio_mesh_get_cell_data(both, "iso:index", 0, &data, &dt, &ndim, got_shape), MIO_OK);
    EXPECT_EQ(dt, MIO_INT64);
    mio_mesh_free(both);

    mio_mesh_free(m);
}

TEST(CApi, GradientCarriesTheOperatorsAndCounters) {
    // A frustum, not a cube: on a cube every face is a parallelogram whose
    // corner average IS its area centroid, so the exactness assertion below
    // would pass even with a broken quadrature.
    const std::vector<double> pts = {0,   0,   0, 2,   0,   0, 2,   2,   0, 0,   2,   0,
                                     0.5, 0.5, 1, 1.5, 0.5, 1, 1.5, 1.5, 1, 0.5, 1.5, 1};
    const std::vector<std::int64_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<double> f(8), u(24);
    for (std::size_t i = 0; i < 8; ++i) {
        const double x = pts[i * 3 + 0], y = pts[i * 3 + 1], z = pts[i * 3 + 2];
        f[i] = 3.0 * x - 2.0 * y + 5.0 * z + 7.0;
        u[i * 3 + 0] = 7.0 * z;
        u[i * 3 + 1] = 11.0 * x;
        u[i * 3 + 2] = 13.0 * y;
    }
    const std::int64_t sshape[1] = {8};
    const std::int64_t vshape[2] = {8, 3};

    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 8, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "hexahedron", 1, 8, MIO_INT64, conn.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_point_data(m, "f", MIO_FLOAT64, 1, sshape, f.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_point_data(m, "u", MIO_FLOAT64, 2, vshape, u.data()), MIO_OK);

    const void* data = nullptr;
    mio_dtype dt = MIO_FLOAT64;
    std::int32_t ndim = 0;
    std::int64_t got_shape[MIO_MAX_NDIM] = {0};

    std::int64_t skipped = -1, fallback = -1;
    mio_mesh* g =
        mio_gradient(m, "f", nullptr, nullptr, nullptr, nullptr, -1, 0, &skipped, &fallback);
    ASSERT_NE(g, nullptr) << mio_last_error();
    EXPECT_EQ(skipped, 0);
    EXPECT_EQ(fallback, 0);
    ASSERT_EQ(mio_mesh_get_cell_data(g, "f:gradient", 0, &data, &dt, &ndim, got_shape), MIO_OK);
    EXPECT_EQ(dt, MIO_FLOAT64);
    ASSERT_EQ(ndim, 2);
    EXPECT_EQ(got_shape[1], 3);
    const double* v = static_cast<const double*>(data);
    EXPECT_NEAR(v[0], 3.0, 1e-12);
    EXPECT_NEAR(v[1], -2.0, 1e-12);
    EXPECT_NEAR(v[2], 5.0, 1e-12);
    mio_mesh_free(g);

    // curl of (7z, 11x, 13y) is (13, 7, 11): three distinct nonzero components,
    // so any index permutation or sign flip fails.
    mio_mesh* c = mio_gradient(m, "u", "curl", "green-gauss", "cell", "w", -1, 0, nullptr, nullptr);
    ASSERT_NE(c, nullptr) << mio_last_error();
    ASSERT_EQ(mio_mesh_get_cell_data(c, "w", 0, &data, &dt, &ndim, got_shape), MIO_OK);
    v = static_cast<const double*>(data);
    EXPECT_NEAR(v[0], 13.0, 1e-12);
    EXPECT_NEAR(v[1], 7.0, 1e-12);
    EXPECT_NEAR(v[2], 11.0, 1e-12);
    mio_mesh_free(c);

    // Least squares on a lone cell has no neighbourhood at all, so the fallback
    // counter must fire -- asserting it stays 0 on a nice mesh would be inert.
    skipped = fallback = -1;
    mio_mesh* l = mio_gradient(m, "f", "gradient", "least-squares", "cell", nullptr, -1, 0,
                               &skipped, &fallback);
    ASSERT_NE(l, nullptr) << mio_last_error();
    EXPECT_EQ(fallback, 1);
    EXPECT_EQ(skipped, 0);
    mio_mesh_free(l);

    // Point location moves the array to point_data and drops the intermediate.
    mio_mesh* p =
        mio_gradient(m, "f", "gradient", "green-gauss", "point", nullptr, -1, 0, nullptr, nullptr);
    ASSERT_NE(p, nullptr) << mio_last_error();
    ASSERT_EQ(mio_mesh_get_point_data(p, "f:gradient", &data, &dt, &ndim, got_shape), MIO_OK);
    EXPECT_LE(mio_mesh_cell_data_num_blocks(p, "f:gradient"), 0)
        << "the intermediate cell array must be dropped";
    mio_mesh_free(p);

    mio_mesh_free(m);
}

TEST(CApi, GradientErrorsAreGuardedNotThrown) {
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
    const std::vector<std::int64_t> conn = {0, 1, 2, 3};
    const std::vector<double> h = {0, 1, 1, 0};
    const std::int64_t shape[1] = {4};
    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 4, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "quad", 1, 4, MIO_INT64, conn.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_point_data(m, "h", MIO_FLOAT64, 1, shape, h.data()), MIO_OK);

    // Unknown array, unknown operator/method, a scalar divergence, an
    // out-of-range component and NULL arguments: NULL + last_error, never an
    // exception across the ABI.
    EXPECT_EQ(mio_gradient(m, "nope", nullptr, nullptr, nullptr, nullptr, -1, 0, nullptr, nullptr),
              nullptr);
    EXPECT_NE(std::string(mio_last_error()), "");
    EXPECT_EQ(mio_gradient(m, "h", "laplacian", nullptr, nullptr, nullptr, -1, 0, nullptr, nullptr),
              nullptr);
    EXPECT_EQ(mio_gradient(m, "h", nullptr, "magic", nullptr, nullptr, -1, 0, nullptr, nullptr),
              nullptr);
    EXPECT_EQ(
        mio_gradient(m, "h", "divergence", nullptr, nullptr, nullptr, -1, 0, nullptr, nullptr),
        nullptr)
        << "a scalar has no divergence";
    EXPECT_EQ(mio_gradient(m, "h", nullptr, nullptr, nullptr, nullptr, 7, 0, nullptr, nullptr),
              nullptr);
    EXPECT_EQ(
        mio_gradient(nullptr, "h", nullptr, nullptr, nullptr, nullptr, -1, 0, nullptr, nullptr),
        nullptr);
    EXPECT_EQ(mio_gradient(m, nullptr, nullptr, nullptr, nullptr, nullptr, -1, 0, nullptr, nullptr),
              nullptr);
    mio_mesh_free(m);
}

TEST(CApi, HessianCarriesCountersAndShape) {
    // A single hexahedron; a linear field has an exactly zero Hessian
    // regardless of mesh shape (see hessian.hpp), so this needs no
    // structured-grid fixture the way the C++ gtest suite's own
    // interior-exactness test does.
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                                     0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    const std::vector<std::int64_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<double> f(8);
    for (std::size_t i = 0; i < 8; ++i) {
        const double x = pts[i * 3 + 0], y = pts[i * 3 + 1], z = pts[i * 3 + 2];
        f[i] = 3.0 * x - 2.0 * y + 5.0 * z + 7.0;
    }
    const std::int64_t sshape[1] = {8};

    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 8, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "hexahedron", 1, 8, MIO_INT64, conn.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_point_data(m, "f", MIO_FLOAT64, 1, sshape, f.data()), MIO_OK);

    const void* data = nullptr;
    mio_dtype dt = MIO_FLOAT64;
    std::int32_t ndim = 0;
    std::int64_t got_shape[MIO_MAX_NDIM] = {0};

    std::int64_t skipped = -1, fallback = -1;
    mio_mesh* h =
        mio_hessian(m, "f", nullptr, nullptr, nullptr, 0, &skipped, &fallback);
    ASSERT_NE(h, nullptr) << mio_last_error();
    EXPECT_EQ(skipped, 0);
    EXPECT_EQ(fallback, 0);
    ASSERT_EQ(mio_mesh_get_cell_data(h, "f:hessian", 0, &data, &dt, &ndim, got_shape), MIO_OK);
    EXPECT_EQ(dt, MIO_FLOAT64);
    ASSERT_EQ(ndim, 2);
    EXPECT_EQ(got_shape[1], 9);
    const double* v = static_cast<const double*>(data);
    for (int i = 0; i < 9; ++i)
        EXPECT_NEAR(v[i], 0.0, 1e-9) << "entry " << i;
    mio_mesh_free(h);

    // Point location moves the array to point_data and drops the intermediate.
    mio_mesh* p = mio_hessian(m, "f", "green-gauss", "point", nullptr, 0, nullptr, nullptr);
    ASSERT_NE(p, nullptr) << mio_last_error();
    ASSERT_EQ(mio_mesh_get_point_data(p, "f:hessian", &data, &dt, &ndim, got_shape), MIO_OK);
    EXPECT_LE(mio_mesh_cell_data_num_blocks(p, "f:hessian"), 0)
        << "the intermediate cell array must be dropped";
    mio_mesh_free(p);

    // Unknown array, a vector (multi-component) input, and NULL arguments:
    // NULL + last_error, never an exception across the ABI.
    EXPECT_EQ(mio_hessian(m, "nope", nullptr, nullptr, nullptr, 0, nullptr, nullptr), nullptr);
    EXPECT_NE(std::string(mio_last_error()), "");
    const std::vector<double> u(24, 0.0);
    const std::int64_t vshape[2] = {8, 3};
    ASSERT_EQ(mio_mesh_add_point_data(m, "u", MIO_FLOAT64, 2, vshape, u.data()), MIO_OK);
    EXPECT_EQ(mio_hessian(m, "u", nullptr, nullptr, nullptr, 0, nullptr, nullptr), nullptr)
        << "hessian is scalar-only";
    EXPECT_EQ(mio_hessian(nullptr, "f", nullptr, nullptr, nullptr, 0, nullptr, nullptr), nullptr);
    EXPECT_EQ(mio_hessian(m, nullptr, nullptr, nullptr, nullptr, 0, nullptr, nullptr), nullptr);

    mio_mesh_free(m);
}

TEST(CApi, EstimateError) {
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                                     0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    const std::vector<std::int64_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<double> f(8);
    for (std::size_t i = 0; i < 8; ++i) {
        const double x = pts[i * 3 + 0], y = pts[i * 3 + 1], z = pts[i * 3 + 2];
        f[i] = 3.0 * x - 2.0 * y + 5.0 * z + 7.0;
    }
    const std::int64_t sshape[1] = {8};

    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 8, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "hexahedron", 1, 8, MIO_INT64, conn.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_point_data(m, "f", MIO_FLOAT64, 1, sshape, f.data()), MIO_OK);

    double global_error = -1.0;
    std::int64_t skipped = -1, marked = -1;
    mio_mesh* out = mio_estimate_error(m, "f", nullptr, nullptr, 0.0, nullptr, nullptr, 0,
                                       &global_error, &skipped, &marked);
    ASSERT_NE(out, nullptr) << mio_last_error();
    EXPECT_EQ(skipped, 0);
    EXPECT_EQ(marked, 0);
    EXPECT_NEAR(global_error, 0.0, 1e-9);

    const void* data = nullptr;
    mio_dtype dt = MIO_FLOAT64;
    std::int32_t ndim = 0;
    std::int64_t got_shape[MIO_MAX_NDIM] = {0};
    ASSERT_EQ(mio_mesh_get_cell_data(out, "error:zz", 0, &data, &dt, &ndim, got_shape), MIO_OK);
    EXPECT_EQ(dt, MIO_FLOAT64);
    EXPECT_LE(mio_mesh_cell_data_num_blocks(out, "error:marked"), 0)
        << "no marking requested, so no marked array";
    mio_mesh_free(out);

    // Custom names + marking.
    double ge2 = -1.0;
    std::int64_t sk2 = -1, mk2 = -1;
    mio_mesh* out2 =
        mio_estimate_error(m, "f", "zz", "absolute", 1e-6, "ind", "flag", 1, &ge2, &sk2, &mk2);
    ASSERT_NE(out2, nullptr) << mio_last_error();
    EXPECT_EQ(mk2, 0);
    ASSERT_EQ(mio_mesh_get_cell_data(out2, "ind", 0, &data, &dt, &ndim, got_shape), MIO_OK);
    ASSERT_EQ(mio_mesh_get_cell_data(out2, "flag", 0, &data, &dt, &ndim, got_shape), MIO_OK);
    EXPECT_EQ(dt, MIO_INT64);
    mio_mesh_free(out2);

    mio_mesh_free(m);
}

TEST(CApi, EstimateErrorErrorsAreGuardedNotThrown) {
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                                     0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    const std::vector<std::int64_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<double> f(8, 1.0);
    const std::int64_t sshape[1] = {8};
    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 8, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "hexahedron", 1, 8, MIO_INT64, conn.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_point_data(m, "f", MIO_FLOAT64, 1, sshape, f.data()), MIO_OK);

    EXPECT_EQ(mio_estimate_error(m, "nope", nullptr, nullptr, 0.0, nullptr, nullptr, 0, nullptr,
                                 nullptr, nullptr),
              nullptr);
    EXPECT_NE(std::string(mio_last_error()), "");
    EXPECT_EQ(mio_estimate_error(m, "f", "kelly", nullptr, 0.0, nullptr, nullptr, 0, nullptr,
                                 nullptr, nullptr),
              nullptr);
    EXPECT_EQ(mio_estimate_error(m, "f", nullptr, "median", 0.0, nullptr, nullptr, 0, nullptr,
                                 nullptr, nullptr),
              nullptr);
    EXPECT_EQ(mio_estimate_error(m, "f", nullptr, "fraction", 1.5, nullptr, nullptr, 0, nullptr,
                                 nullptr, nullptr),
              nullptr)
        << "out-of-range marking_value";
    EXPECT_EQ(mio_estimate_error(nullptr, "f", nullptr, nullptr, 0.0, nullptr, nullptr, 0, nullptr,
                                 nullptr, nullptr),
              nullptr);
    EXPECT_EQ(mio_estimate_error(m, nullptr, nullptr, nullptr, 0.0, nullptr, nullptr, 0, nullptr,
                                 nullptr, nullptr),
              nullptr);
    mio_mesh_free(m);
}

namespace {
// A regular octahedron -- the smallest closed 2-manifold triangle mesh with
// no coplanar faces, small enough to hand-write for a C API smoke test.
mio_mesh* c_api_octahedron() {
    const std::vector<double> pts = {1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1};
    const std::vector<std::int64_t> conn = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4,
                                            2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
    mio_mesh* m = mio_mesh_create();
    mio_mesh_set_points(m, MIO_FLOAT64, 6, 3, pts.data());
    mio_mesh_add_cell_block(m, "triangle", 8, 3, MIO_INT64, conn.data());
    return m;
}
}  // namespace

TEST(CApi, RemeshProducesTheRequestedNumberOfClusters) {
    mio_mesh* m = c_api_octahedron();

    std::int64_t num_clusters = -1, num_iterations = -1, num_isolated = -1, num_nonmanifold = -1;
    int subdivide_applied = -1;
    mio_mesh* out =
        mio_remesh(m, 30, -1, 10.0, 4, 100, 10, "isotropic", 0.0, 1, &num_clusters,
                   &num_iterations, &subdivide_applied, &num_isolated, &num_nonmanifold);
    ASSERT_NE(out, nullptr) << mio_last_error();
    EXPECT_EQ(num_clusters, 30);
    EXPECT_GT(subdivide_applied, 0) << "6 vertices cannot support 30 clusters without subdividing";
    EXPECT_EQ(mio_mesh_num_points(out), 30);
    EXPECT_GE(num_nonmanifold, 0);
    mio_mesh_free(out);

    // The quadric ("feature-preserving") metric, gradation and boundary
    // preservation are all accepted too.
    mio_mesh* out2 = mio_remesh(m, 30, -1, 10.0, 4, 100, 10, "quadric", 1.5, 0, nullptr, nullptr,
                                nullptr, nullptr, nullptr);
    ASSERT_NE(out2, nullptr) << mio_last_error();
    mio_mesh_free(out2);

    mio_mesh_free(m);
}

TEST(CApi, RemeshErrorsAreGuardedNotThrown) {
    mio_mesh* m = c_api_octahedron();

    EXPECT_EQ(mio_remesh(m, 3, -1, 10.0, 4, 100, 10, nullptr, 0.0, 1, nullptr, nullptr, nullptr,
                         nullptr, nullptr),
              nullptr)
        << "too few clusters";
    EXPECT_NE(std::string(mio_last_error()), "");
    EXPECT_EQ(mio_remesh(m, 10, -1, 10.0, 4, 100, 10, "bogus", 0.0, 1, nullptr, nullptr, nullptr,
                         nullptr, nullptr),
              nullptr)
        << "unknown metric";
    EXPECT_EQ(mio_remesh(nullptr, 10, -1, 10.0, 4, 100, 10, nullptr, 0.0, 1, nullptr, nullptr,
                         nullptr, nullptr, nullptr),
              nullptr);

    mio_mesh_free(m);
}

TEST(CApi, RemeshExAcceptsAnisotropicAndReportsCounters) {
    mio_mesh* m = c_api_octahedron();

    mio_remesh_opts opts;
    mio_remesh_opts_init(&opts);
    opts.num_clusters = 30;
    opts.metric = "anisotropic";
    opts.max_anisotropy = 3.0;
    mio_remesh_report report;
    mio_mesh* out = mio_remesh_ex(m, &opts, &report);
    ASSERT_NE(out, nullptr) << mio_last_error();
    EXPECT_GT(report.num_clusters, 0);
    EXPECT_EQ(mio_mesh_num_points(out), report.num_clusters);
    EXPECT_GE(report.num_isolated_clusters, 0);
    EXPECT_GE(report.num_non_manifold_vertices, 0);
    mio_mesh_free(out);

    // NULL report is fine.
    mio_mesh* out2 = mio_remesh_ex(m, &opts, nullptr);
    ASSERT_NE(out2, nullptr) << mio_last_error();
    mio_mesh_free(out2);

    // NULL opts falls back to mio_remesh_opts_init's own defaults, whose
    // num_clusters == 0 fails validation exactly like the flat mio_remesh's
    // own "too few clusters" case above.
    EXPECT_EQ(mio_remesh_ex(m, nullptr, nullptr), nullptr);

    // max_anisotropy away from the default under a non-anisotropic metric
    // is guarded, not silently ignored, exactly as in the C++ core.
    mio_remesh_opts bad;
    mio_remesh_opts_init(&bad);
    bad.num_clusters = 30;
    bad.metric = "isotropic";
    bad.max_anisotropy = 2.0;
    EXPECT_EQ(mio_remesh_ex(m, &bad, nullptr), nullptr);
    EXPECT_NE(std::string(mio_last_error()), "");

    mio_mesh_free(m);
}

TEST(CApi, RemeshVolumeOptsInitDefaultsAndRemeshVolume) {
    mio_mesh* octa = c_api_octahedron();

    mio_remesh_volume_opts opts;
    mio_remesh_volume_opts_init(&opts);
    EXPECT_EQ(opts.cell_size, 0.0);
    EXPECT_DOUBLE_EQ(opts.padding_relative, 0.1);
    EXPECT_EQ(opts.max_cells, 20000000);
    EXPECT_EQ(opts.max_tets, 20000000);
    EXPECT_DOUBLE_EQ(opts.warp_fraction, 0.35);
    // mio_sdf_opts_init's own defaults are honoured for the embedded distance
    // options -- the mio_compute_sdf_opts nesting precedent.
    EXPECT_EQ(opts.distance.sign, MIO_SDF_PSEUDONORMAL);
    opts.cell_size = 0.4;
    opts.distance.watertight_check = MIO_SDF_WATERTIGHT_OFF;

    mio_remesh_volume_report report;
    mio_mesh* out = mio_remesh_volume_ex(octa, &opts, &report);
    ASSERT_NE(out, nullptr) << mio_last_error();
    EXPECT_GT(report.num_tets, 0);
    EXPECT_EQ(mio_mesh_num_cell_blocks(out), 1);
    EXPECT_EQ(block_type(out, 0), "tetra");
    EXPECT_GE(report.num_vertices_warped, 0);
    EXPECT_GE(report.num_tets_rejected, 0);
    EXPECT_GE(report.num_non_manifold_edges, 0);
    mio_mesh_free(out);

    // NULL report is fine.
    mio_mesh* out2 = mio_remesh_volume_ex(octa, &opts, nullptr);
    ASSERT_NE(out2, nullptr) << mio_last_error();
    mio_mesh_free(out2);

    // NULL opts is an error, mio_voxelize's own rule: a resolution or cell
    // size must be given, and there is no sensible all-defaults fallback.
    EXPECT_EQ(mio_remesh_volume_ex(octa, nullptr, nullptr), nullptr);
    EXPECT_NE(std::string(mio_last_error()), "");

    mio_mesh_free(octa);
}

TEST(CApi, RemeshVolumeErrorsAreGuardedNotThrown) {
    mio_mesh* octa = c_api_octahedron();
    mio_remesh_volume_opts opts;
    mio_remesh_volume_opts_init(&opts);
    opts.cell_size = 0.4;
    opts.warp_fraction = -0.1;  // negative warp fraction

    EXPECT_EQ(mio_remesh_volume_ex(octa, &opts, nullptr), nullptr);
    EXPECT_NE(std::string(mio_last_error()), "");

    mio_mesh_free(octa);
}

TEST(CApi, IsosurfaceErrorsAreGuardedNotThrown) {
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
    const std::vector<std::int64_t> conn = {0, 1, 2, 3};
    const std::vector<double> h = {0, 1, 1, 0};
    const std::int64_t shape[1] = {4};
    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 4, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "quad", 1, 4, MIO_INT64, conn.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_point_data(m, "h", MIO_FLOAT64, 1, shape, h.data()), MIO_OK);

    const double iso[1] = {0.5};
    // Unknown array, no isovalues, and NULL arguments: NULL + last_error, never
    // an exception across the ABI.
    EXPECT_EQ(mio_isosurface(m, "nope", iso, 1, -1, 0), nullptr);
    EXPECT_NE(std::string(mio_last_error()), "");
    EXPECT_EQ(mio_isosurface(m, "h", iso, 0, -1, 0), nullptr);
    EXPECT_EQ(mio_isosurface(m, "h", iso, 1, 7, 0), nullptr);
    EXPECT_EQ(mio_isosurface(nullptr, "h", iso, 1, -1, 0), nullptr);
    EXPECT_EQ(mio_isosurface(m, nullptr, iso, 1, -1, 0), nullptr);
    EXPECT_EQ(mio_isosurface(m, "h", nullptr, 1, -1, 0), nullptr);
    mio_mesh_free(m);
}

// --- named regions (doc/regions.md) ------------------------------------------
// Regions are the first thing on this ABI that carries named *groups* of
// entities; before meshio++ 8.1 they never left the Python layer.

TEST(CApi, RegionsRoundTripThroughTheOpaqueHandle) {
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0.5, 0.5, 1};
    const std::vector<std::int64_t> conn = {0, 1, 2, 4, 0, 2, 3, 4};
    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 5, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "tetra", 2, 4, MIO_INT64, conn.data()), MIO_OK);

    // Entries are given unsorted and with a duplicate: canonicalization is part
    // of the contract, so what comes back is sorted and de-duplicated.
    const std::int64_t nodes[4] = {3, 0, 3, 1};
    ASSERT_EQ(mio_mesh_add_region(m, "fixed", MIO_REGION_POINT, -1, -1, nodes, 4), MIO_OK);
    const std::int64_t cells[2] = {1, 0};
    ASSERT_EQ(mio_mesh_add_region(m, "solid", MIO_REGION_CELL, 3, 42, cells, 2), MIO_OK);
    const std::int64_t sides[4] = {1, 3, 0, 1};  // (cell, facet) pairs
    ASSERT_EQ(mio_mesh_add_region(m, "wall", MIO_REGION_SIDE, 2, -1, sides, 4), MIO_OK);

    mio_regions* r = mio_regions_create(m);
    ASSERT_NE(r, nullptr);
    ASSERT_EQ(mio_regions_count(r), 3);

    // Order is (kind, name, dim, tag): Point, then Cell, then Side.
    char buf[64];
    ASSERT_GT(mio_regions_name(r, 0, buf, sizeof(buf)), 0);
    EXPECT_EQ(std::string(buf), "fixed");
    ASSERT_GT(mio_regions_name(r, 2, buf, sizeof(buf)), 0);
    EXPECT_EQ(std::string(buf), "wall");

    mio_region_info info{};
    ASSERT_EQ(mio_regions_info(r, 0, &info), MIO_OK);
    EXPECT_EQ(info.kind, MIO_REGION_POINT);
    EXPECT_EQ(info.stride, 1);
    EXPECT_EQ(info.num_entries, 3);  // the duplicate 3 was dropped

    ASSERT_EQ(mio_regions_info(r, 1, &info), MIO_OK);
    EXPECT_EQ(info.kind, MIO_REGION_CELL);
    EXPECT_EQ(info.dim, 3);
    EXPECT_EQ(info.tag, 42);

    ASSERT_EQ(mio_regions_info(r, 2, &info), MIO_OK);
    EXPECT_EQ(info.kind, MIO_REGION_SIDE);
    EXPECT_EQ(info.stride, 2);
    EXPECT_EQ(info.num_entries, 2);

    std::int64_t count = 0;
    const std::int64_t* e = mio_regions_entries(r, 0, &count);
    ASSERT_NE(e, nullptr);
    ASSERT_EQ(count, 3);
    EXPECT_EQ(e[0], 0);
    EXPECT_EQ(e[1], 1);
    EXPECT_EQ(e[2], 3);

    e = mio_regions_entries(r, 2, &count);
    ASSERT_NE(e, nullptr);
    ASSERT_EQ(count, 4);  // 2 pairs, lexicographically sorted
    EXPECT_EQ(e[0], 0);
    EXPECT_EQ(e[1], 1);
    EXPECT_EQ(e[2], 1);
    EXPECT_EQ(e[3], 3);

    mio_regions_free(r);
    mio_mesh_free(m);
}

TEST(CApi, RegionErrorsAreGuardedNotThrown) {
    mio_mesh* m = mio_mesh_create();
    const std::int64_t one[1] = {0};

    // No exception may cross the ABI: every one of these is a status/NULL.
    EXPECT_EQ(mio_mesh_add_region(nullptr, "x", MIO_REGION_POINT, -1, -1, one, 1),
              MIO_ERR_INVALID_ARG);
    EXPECT_EQ(mio_mesh_add_region(m, nullptr, MIO_REGION_POINT, -1, -1, one, 1),
              MIO_ERR_INVALID_ARG);
    EXPECT_EQ(mio_mesh_add_region(m, "x", MIO_REGION_POINT, -1, -1, nullptr, 1),
              MIO_ERR_INVALID_ARG);
    // A side region needs (cell, facet) pairs.
    EXPECT_EQ(mio_mesh_add_region(m, "x", MIO_REGION_SIDE, -1, -1, one, 1), MIO_ERR_INVALID_ARG);

    EXPECT_EQ(mio_regions_create(nullptr), nullptr);
    mio_regions* r = mio_regions_create(m);
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(mio_regions_count(r), 0);
    mio_region_info info{};
    EXPECT_EQ(mio_regions_info(r, 0, &info), MIO_ERR_INVALID_ARG);
    EXPECT_EQ(mio_regions_entries(r, 5, nullptr), nullptr);
    EXPECT_EQ(mio_regions_name(r, 5, nullptr, 0), -1);
    mio_regions_free(r);
    mio_regions_free(nullptr);  // NULL is safe
    mio_mesh_free(m);
}

TEST(CApi, RegionsSurviveAnOperation) {
    // crop keeps both tetra, so the cell region survives whole; the point
    // region is remapped through the pruned point numbering.
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0.5, 0.5, 1};
    const std::vector<std::int64_t> conn = {0, 1, 2, 4, 0, 2, 3, 4};
    mio_mesh* m = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 5, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(m, "tetra", 2, 4, MIO_INT64, conn.data()), MIO_OK);
    const std::int64_t cells[1] = {0};
    ASSERT_EQ(mio_mesh_add_region(m, "solid", MIO_REGION_CELL, 3, 7, cells, 1), MIO_OK);

    const double lo[3] = {-1, -1, -1};
    const double hi[3] = {2, 2, 2};
    mio_mesh* out = mio_crop_bbox(m, lo, hi, 0, 0);
    ASSERT_NE(out, nullptr);

    mio_regions* r = mio_regions_create(out);
    ASSERT_NE(r, nullptr);
    ASSERT_EQ(mio_regions_count(r), 1);
    mio_region_info info{};
    ASSERT_EQ(mio_regions_info(r, 0, &info), MIO_OK);
    EXPECT_EQ(info.num_entries, 1);
    EXPECT_EQ(info.tag, 7);  // the format-native id rides along
    mio_regions_free(r);

    mio_mesh_free(out);
    mio_mesh_free(m);
}

// ---- transient XDMF ------------------------------------------------------
//
// The one writer the flat API exposes as a handle rather than a (path, mesh)
// call. Written here, then read back through the C reader one step at a time.

TEST(CApi, XdmfTimeSeries) {
    mio_mesh* m = mio_mesh_create();
    ASSERT_NE(m, nullptr);
    const double pts[15] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0.5, 0.5, 0.5};
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 5, 3, pts), MIO_OK) << mio_last_error();
    const int64_t conn[8] = {0, 1, 2, 4, 0, 2, 3, 4};
    ASSERT_EQ(mio_mesh_add_cell_block(m, "tetra", 2, 4, MIO_INT64, conn), MIO_OK)
        << mio_last_error();

    const std::string path = mt::temp_path(".xdmf");

    mio_xdmf_series* s = mio_xdmf_series_create(path.c_str(), "XML", -1);
    ASSERT_NE(s, nullptr) << mio_last_error();
    ASSERT_EQ(mio_xdmf_series_write_points_cells(s, m), MIO_OK) << mio_last_error();
    const int64_t shape[1] = {5};
    for (int k = 0; k < 3; ++k) {
        const double t[5] = {100.0 * k, 100.0 * k + 1, 100.0 * k + 2, 100.0 * k + 3, 100.0 * k + 4};
        ASSERT_EQ(mio_mesh_add_point_data(m, "T", MIO_FLOAT64, 1, shape, t), MIO_OK)
            << mio_last_error();
        ASSERT_EQ(mio_xdmf_series_write_data(s, 0.25 * k, m), MIO_OK) << mio_last_error();
    }
    EXPECT_EQ(mio_xdmf_series_num_steps(s), 3);
    ASSERT_EQ(mio_xdmf_series_finalize(s), MIO_OK) << mio_last_error();
    mio_xdmf_series_free(s);

    // Every step's time value is reachable without decoding a payload.
    mio_read_metadata* meta = mio_read_metadata_create(path.c_str(), nullptr);
    ASSERT_NE(meta, nullptr) << mio_last_error();
    ASSERT_EQ(mio_read_metadata_num_time_values(meta), 3);
    double times[3] = {0, 0, 0};
    ASSERT_EQ(mio_read_metadata_time_values(meta, times, 3), 3);
    for (int k = 0; k < 3; ++k)
        EXPECT_DOUBLE_EQ(times[k], 0.25 * k);
    mio_read_metadata_free(meta);

    // ... and each step reads back with its own values.
    for (int k = 0; k < 3; ++k) {
        mio_read_opts opts;
        mio_read_opts_init(&opts);
        opts.time_step = k;
        mio_mesh* out = mio_read_ex(path.c_str(), nullptr, &opts);
        ASSERT_NE(out, nullptr) << mio_last_error();
        EXPECT_EQ(mio_mesh_num_points(out), 5);
        const void* data = nullptr;
        mio_dtype dt = MIO_FLOAT64;
        int32_t ndim = 0;
        int64_t got_shape[MIO_MAX_NDIM] = {0};
        ASSERT_EQ(mio_mesh_get_point_data(out, "T", &data, &dt, &ndim, got_shape), MIO_OK)
            << mio_last_error();
        ASSERT_EQ(dt, MIO_FLOAT64);
        ASSERT_EQ(got_shape[0], 5);
        const double* values = static_cast<const double*>(data);
        for (int i = 0; i < 5; ++i)
            EXPECT_DOUBLE_EQ(values[i], 100.0 * k + i) << "step " << k << " entry " << i;
        mio_mesh_free(out);
    }

    mio_mesh_free(m);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(CApi, XdmfTimeSeriesUnknownFormat) {
    const std::string path = mt::temp_path(".xdmf");
    EXPECT_EQ(mio_xdmf_series_create(path.c_str(), "Zarr", -1), nullptr);
    EXPECT_NE(std::string(mio_last_error()).find("Zarr"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Settings pipeline (v9.11.0): JSON text only across the ABI.
// ---------------------------------------------------------------------------

#ifdef MESHIOPLUSPLUS_HAS_JSON

TEST(CApi, PipelineRunsASettingsFile) {
    ASSERT_EQ(mio_pipeline_has_json(), 1);
    const std::string in_path = mt::temp_path("_capi_pipe.vtk");
    const std::string out_path = mt::temp_path("_capi_pipe_out.vtk");
    const std::string settings_path = mt::temp_path("_capi_settings.json");
    {
        mio_mesh* m = build_tet_mesh();
        ASSERT_EQ(mio_write(in_path.c_str(), m, "vtk"), MIO_OK) << mio_last_error();
        mio_mesh_free(m);
    }
    {
        std::ofstream settings(settings_path);
        settings << R"({"Input": {"Path": ")" << in_path << R"("},
                       "Operations": [{"Op": "Quality"}],
                       "Output": {"Path": ")"
                 << out_path << R"("}})";
    }
    ASSERT_EQ(mio_pipeline_run_file(settings_path.c_str()), MIO_OK) << mio_last_error();
    mio_mesh* out = mio_read(out_path.c_str(), "vtk");
    ASSERT_NE(out, nullptr) << mio_last_error();
    EXPECT_GE(mio_mesh_cell_data_num_blocks(out, "quality:scaled_jacobian"), 1);
    mio_mesh_free(out);
    std::error_code ec;
    std::filesystem::remove(in_path, ec);
    std::filesystem::remove(out_path, ec);
    std::filesystem::remove(settings_path, ec);
}

TEST(CApi, PipelineJsonSchemaErrorsNameTheOffender) {
    EXPECT_NE(mio_pipeline_run_json(R"({"Input": {"Path": "a"}, "Output": {"Path": "b"},
                                        "Operations": [{"Op": "Nope"}]})"),
              MIO_OK);
    EXPECT_NE(std::string(mio_last_error()).find("Nope"), std::string::npos);
    EXPECT_NE(mio_pipeline_run_file(nullptr), MIO_OK);
    EXPECT_NE(mio_pipeline_run_json(nullptr), MIO_OK);
}

TEST(CApi, SequencePipelineRunsASettingsFile) {
    // The sequence document shares the pipeline's parser and guard.
    const std::string dir = mt::temp_path("_capi_seqdoc");
    std::filesystem::create_directories(dir);
    for (int i = 0; i < 3; ++i) {
        mio_mesh* m = build_tet_mesh();
        ASSERT_EQ(mio_write((dir + "/in_" + std::to_string(i) + ".vtu").c_str(), m, "vtu"), MIO_OK);
        mio_mesh_free(m);
    }
    const std::string settings = dir + "/settings.json";
    {
        std::ofstream f(settings);
        f << R"({"Version": 1, "Input": {"Pattern": ")" << dir
          << R"(/in_*.vtu"}, "Operations": [{"Op": "Quality"}], "Output": {"Path": ")" << dir
          << R"(/out_{step}.vtu"}})";
    }
    ASSERT_EQ(mio_sequence_pipeline_run_file(settings.c_str()), MIO_OK) << mio_last_error();
    EXPECT_TRUE(std::filesystem::exists(dir + "/out_0002.vtu"));
    EXPECT_NE(mio_sequence_pipeline_run_file(nullptr), MIO_OK);
    EXPECT_NE(mio_sequence_pipeline_run_json(nullptr), MIO_OK);
    std::filesystem::remove_all(dir);
}

#else  // !MESHIOPLUSPLUS_HAS_JSON

TEST(CApi, PipelineCompiledOutFailsNamingTheFlag) {
    // Never a link error, never a silent no-op: the entry points exist and
    // the error names the CMake option (fires on the WITH_JSON=OFF CI leg).
    EXPECT_EQ(mio_pipeline_has_json(), 0);
    EXPECT_NE(mio_pipeline_run_json("{}"), MIO_OK);
    EXPECT_NE(std::string(mio_last_error()).find("MESHIOPLUSPLUS_WITH_JSON"), std::string::npos);
}

#endif  // MESHIOPLUSPLUS_HAS_JSON

// ---------------------------------------------------------------------------
// Sequences. The handle is a PLAN, not a cache: mio_sequence_read hands back an
// OWNED mesh precisely so nothing accumulates, which is the C ABI's expression
// of the streaming guarantee.
// ---------------------------------------------------------------------------

namespace {

/// Write `count` single-step .vtu files named `<dir>/in_<i>.vtu`.
std::string capi_seq_dir(int count) {
    const std::string dir = mt::temp_path("_capi_seq");
    std::filesystem::create_directories(dir);
    for (int i = 0; i < count; ++i) {
        mio_mesh* m = build_tet_mesh();
        mio_write((dir + "/in_" + std::to_string(i) + ".vtu").c_str(), m, "vtu");
        mio_mesh_free(m);
    }
    return dir;
}

}  // namespace

TEST(CApi, SequenceOpensAPatternInNaturalOrder) {
    const std::string dir = capi_seq_dir(12);
    mio_sequence* seq = mio_sequence_open((dir + "/in_*.vtu").c_str());
    ASSERT_NE(seq, nullptr) << mio_last_error();
    ASSERT_EQ(mio_sequence_count(seq), 12);

    // in_9 must precede in_10 -- the whole point of the ordering rule.
    char buf[512];
    ASSERT_GT(mio_sequence_path(seq, 9, buf, sizeof(buf)), 0);
    EXPECT_NE(std::string(buf).find("in_9.vtu"), std::string::npos);
    ASSERT_GT(mio_sequence_path(seq, 10, buf, sizeof(buf)), 0);
    EXPECT_NE(std::string(buf).find("in_10.vtu"), std::string::npos);

    double t = -1.0;
    ASSERT_EQ(mio_sequence_time(seq, 3, &t), MIO_OK);
    EXPECT_DOUBLE_EQ(t, 3.0);
    EXPECT_EQ(mio_sequence_time_source(seq, 3), 2);  // from the filename
    EXPECT_EQ(mio_sequence_step(seq, 3), 0);

    // The string-getter protocol: a short buffer reports the needed length.
    char tiny[2];
    EXPECT_GT(mio_sequence_path(seq, 0, tiny, sizeof(tiny)), 2);

    mio_sequence_free(seq);
    std::filesystem::remove_all(dir);
}

TEST(CApi, SequenceReadHandsBackAnOwnedMesh) {
    const std::string dir = capi_seq_dir(3);
    mio_sequence* seq = mio_sequence_open((dir + "/in_*.vtu").c_str());
    ASSERT_NE(seq, nullptr);
    for (int64_t i = 0; i < 3; ++i) {
        mio_mesh* m = mio_sequence_read(seq, i);
        ASSERT_NE(m, nullptr) << mio_last_error();
        EXPECT_EQ(mio_mesh_num_points(m), 5);
        // Owned: freeing it here must be correct, and the sequence must stay
        // usable afterwards (it holds no reference to what it produced).
        mio_mesh_free(m);
    }
    EXPECT_EQ(mio_sequence_count(seq), 3);
    mio_sequence_free(seq);
    std::filesystem::remove_all(dir);
}

TEST(CApi, SequenceExplicitListAndOptions) {
    const std::string dir = capi_seq_dir(3);
    const std::string a = dir + "/in_0.vtu";
    const std::string b = dir + "/in_1.vtu";
    const char* paths[2] = {b.c_str(), a.c_str()};  // a caller's stated order

    mio_sequence_opts opts;
    mio_sequence_opts_init(&opts);
    const double times[2] = {10.0, 20.0};
    opts.times = times;
    opts.num_times = 2;

    mio_sequence* seq = mio_sequence_open_list(paths, 2, &opts);
    ASSERT_NE(seq, nullptr) << mio_last_error();
    char buf[512];
    ASSERT_GT(mio_sequence_path(seq, 0, buf, sizeof(buf)), 0);
    EXPECT_NE(std::string(buf).find("in_1.vtu"), std::string::npos)
        << "an explicit list is a stated order and must not be re-sorted";
    double t = 0.0;
    ASSERT_EQ(mio_sequence_time(seq, 1, &t), MIO_OK);
    EXPECT_DOUBLE_EQ(t, 20.0);
    EXPECT_EQ(mio_sequence_time_source(seq, 1), 0);  // explicit
    mio_sequence_free(seq);

    // sort=1 reorders it.
    opts.sort = 1;
    seq = mio_sequence_open_list(paths, 2, &opts);
    ASSERT_NE(seq, nullptr);
    ASSERT_GT(mio_sequence_path(seq, 0, buf, sizeof(buf)), 0);
    EXPECT_NE(std::string(buf).find("in_0.vtu"), std::string::npos);
    mio_sequence_free(seq);
    std::filesystem::remove_all(dir);
}

TEST(CApi, SequenceFanInAndFanOut) {
    const std::string dir = capi_seq_dir(4);
    mio_sequence* seq = mio_sequence_open((dir + "/in_*.vtu").c_str());
    ASSERT_NE(seq, nullptr);

    const std::string series = dir + "/series.xdmf";
    ASSERT_EQ(mio_sequence_to_timeseries(seq, series.c_str(), nullptr), MIO_OK) << mio_last_error();
    mio_sequence_free(seq);
    ASSERT_TRUE(std::filesystem::exists(series));

    const std::string pattern = dir + "/back_{step}.vtu";
    ASSERT_EQ(mio_timeseries_to_sequence(series.c_str(), nullptr, pattern.c_str(), nullptr), MIO_OK)
        << mio_last_error();
    EXPECT_TRUE(std::filesystem::exists(dir + "/back_0003.vtu"));
    std::filesystem::remove_all(dir);
}

TEST(CApi, SequenceToTimeseriesExSelectsEncodingAndRejectsWhatItCannotHonour) {
    // The transient writer bypasses mio_write_ex's registry path entirely, so
    // it needs its own opt-in encoding selection -- and the same "reject an
    // option you cannot honour" rule (this is what a build without HDF5, e.g.
    // the Julia/R notebook environments, needs to select the XML data format).
    const std::string dir = capi_seq_dir(2);
    mio_sequence* seq = mio_sequence_open((dir + "/in_*.vtu").c_str());
    ASSERT_NE(seq, nullptr);

    mio_write_opts ascii;
    mio_write_opts_init(&ascii);
    ascii.encoding = MIO_ENCODING_ASCII;
    const std::string xml = dir + "/xml_series.xdmf";
    EXPECT_EQ(mio_sequence_to_timeseries_ex(seq, xml.c_str(), nullptr, &ascii), MIO_OK)
        << mio_last_error();
    EXPECT_TRUE(std::filesystem::exists(xml));
    // "XML" means the light data is inline -- no sibling .h5 companion.
    EXPECT_FALSE(std::filesystem::exists(dir + "/xml_series.h5"));

    mio_write_opts bad;
    mio_write_opts_init(&bad);
    bad.codec = MIO_CODEC_ZLIB;
    EXPECT_NE(mio_sequence_to_timeseries_ex(seq, (dir + "/s2.xdmf").c_str(), nullptr, &bad),
              MIO_OK);
    EXPECT_NE(std::string(mio_last_error()).find("Codec"), std::string::npos);

    mio_sequence_free(seq);
    std::filesystem::remove_all(dir);
}

TEST(CApi, SequenceErrorsAreReportedNotCrashed) {
    EXPECT_EQ(mio_sequence_open(nullptr), nullptr);
    EXPECT_EQ(mio_sequence_count(nullptr), -1);
    EXPECT_EQ(mio_sequence_step(nullptr, 0), -1);
    EXPECT_EQ(mio_sequence_time_source(nullptr, 0), -1);
    EXPECT_EQ(mio_sequence_read(nullptr, 0), nullptr);
    EXPECT_NE(mio_sequence_time(nullptr, 0, nullptr), MIO_OK);
    EXPECT_EQ(mio_sequence_open_list(nullptr, 0, nullptr), nullptr);
    // A pattern matching nothing is an error, never an empty sequence.
    EXPECT_EQ(mio_sequence_open("/nonexistent-dir-for-meshio/out_*.vtu"), nullptr);
    mio_sequence_free(nullptr);  // must be a no-op

    const std::string dir = capi_seq_dir(2);
    mio_sequence* seq = mio_sequence_open((dir + "/in_*.vtu").c_str());
    ASSERT_NE(seq, nullptr);
    EXPECT_EQ(mio_sequence_read(seq, 99), nullptr);
    EXPECT_NE(std::string(mio_last_error()).find("out of range"), std::string::npos);
    // A target that cannot hold a series fails by name.
    EXPECT_NE(mio_sequence_to_timeseries(seq, (dir + "/no.vtu").c_str(), nullptr), MIO_OK);
    EXPECT_NE(std::string(mio_last_error()).find("{step}"), std::string::npos);
    mio_sequence_free(seq);
    std::filesystem::remove_all(dir);
}

// --- regular grids and signed distance --------------------------------------

namespace {

/// The unit cube [0,1]^3 as a closed, outward-wound triangle surface.
mio_mesh* capi_cube_surface() {
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                                     0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    const std::vector<std::int64_t> tris = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
                                            1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7};
    mio_mesh* m = mio_mesh_create();
    EXPECT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 8, 3, pts.data()), MIO_OK);
    EXPECT_EQ(mio_mesh_add_cell_block(m, "triangle", 12, 3, MIO_INT64, tris.data()), MIO_OK);
    return m;
}

}  // namespace

TEST(CApi, GridBuildsALattice) {
    const std::int64_t dims[3] = {2, 3, 4};
    const double origin[3] = {-1.0, 0.0, 0.5};
    const double spacing[3] = {0.5, 2.0, 0.25};
    mio_mesh* g = mio_grid(dims, origin, spacing, 20000000);
    ASSERT_NE(g, nullptr);
    EXPECT_EQ(mio_mesh_num_points(g), 3 * 4 * 5);
    ASSERT_EQ(mio_mesh_num_cell_blocks(g), 1);
    EXPECT_EQ(block_type(g, 0), "hexahedron");
    std::int64_t num_cells = 0, npc = 0;
    ASSERT_EQ(mio_mesh_cell_block_info(g, 0, &num_cells, &npc, nullptr), MIO_OK);
    EXPECT_EQ(num_cells, 24);
    EXPECT_EQ(npc, 8);
    mio_mesh_free(g);

    // NULL dims and an exceeded budget both fail rather than crash.
    EXPECT_EQ(mio_grid(nullptr, origin, spacing, 0), nullptr);
    const std::int64_t big[3] = {100, 100, 100};
    EXPECT_EQ(mio_grid(big, nullptr, nullptr, 1000), nullptr);
}

TEST(CApi, GridDefaultsOriginAndSpacingWhenNull) {
    const std::int64_t dims[3] = {1, 1, 1};
    mio_mesh* g = mio_grid(dims, nullptr, nullptr, 0);
    ASSERT_NE(g, nullptr);
    EXPECT_EQ(mio_mesh_num_points(g), 8);
    mio_mesh_free(g);
}

TEST(CApi, VoxelOptsInitDefaultsAndVoxelize) {
    mio_voxel_opts opts;
    mio_voxel_opts_init(&opts);
    EXPECT_EQ(opts.fill, MIO_VOXEL_ALL);
    EXPECT_EQ(opts.sign, MIO_SDF_PSEUDONORMAL);
    EXPECT_EQ(opts.max_cells, 20000000);
    EXPECT_EQ(opts.resolution, nullptr);

    mio_mesh* cube = capi_cube_surface();
    const std::int64_t res[3] = {4, 4, 4};
    opts.resolution = res;

    std::int64_t dims[3] = {0, 0, 0};
    double origin[3] = {0, 0, 0}, spacing[3] = {0, 0, 0};
    std::int64_t occupied = -1;
    mio_mesh* g = mio_voxelize(cube, &opts, dims, origin, spacing, &occupied);
    ASSERT_NE(g, nullptr);
    EXPECT_EQ(occupied, 64);
    EXPECT_EQ(dims[0], 4);
    EXPECT_DOUBLE_EQ(spacing[0], 0.25);
    EXPECT_DOUBLE_EQ(origin[0], 0.0);
    mio_mesh_free(g);

    // Every out-param is nullable.
    mio_mesh* g2 = mio_voxelize(cube, &opts, nullptr, nullptr, nullptr, nullptr);
    ASSERT_NE(g2, nullptr);
    mio_mesh_free(g2);

    // NULL options is an error, not a default: a resolution must be chosen.
    EXPECT_EQ(mio_voxelize(cube, nullptr, nullptr, nullptr, nullptr, nullptr), nullptr);
    mio_mesh_free(cube);
}

TEST(CApi, ComputeSdfOptsInitDefaultsAndVoxelStructure) {
    mio_compute_sdf_opts opts;
    mio_compute_sdf_opts_init(&opts);
    EXPECT_EQ(opts.structure, MIO_SDF_VOXEL);
    EXPECT_EQ(opts.root_resolution, 8);
    EXPECT_EQ(opts.max_depth, 4);
    EXPECT_DOUBLE_EQ(opts.band_cells, 1.0);
    EXPECT_DOUBLE_EQ(opts.padding_relative, 0.1);
    EXPECT_EQ(opts.max_cells, 20000000);
    EXPECT_EQ(opts.resolution, nullptr);
    // The embedded distance options are initialized too, or a caller who only
    // touched the outer struct would get sign = Unsigned by accident.
    EXPECT_EQ(opts.distance.sign, MIO_SDF_PSEUDONORMAL);
    EXPECT_DOUBLE_EQ(opts.distance.max_winding_work, 2.0e9);

    mio_mesh* cube = capi_cube_surface();
    const std::int64_t res[3] = {4, 4, 4};
    opts.resolution = res;
    opts.distance.watertight_check = MIO_SDF_WATERTIGHT_OFF;

    std::int64_t dims[3] = {0, 0, 0}, depth = -1, banded = -1;
    double origin[3] = {0, 0, 0}, spacing[3] = {0, 0, 0};
    mio_surface_quality q{};
    mio_mesh* g = mio_compute_sdf(cube, &opts, dims, origin, spacing, &depth, &banded, &q);
    ASSERT_NE(g, nullptr);
    EXPECT_EQ(dims[0], 4);
    EXPECT_EQ(depth, 0);
    EXPECT_EQ(banded, 0);
    EXPECT_NE(q.watertight, 0);
    EXPECT_EQ(mio_mesh_num_point_data(g), 1);
    mio_mesh_free(g);

    // Every out-param is nullable.
    mio_mesh* g2 =
        mio_compute_sdf(cube, &opts, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    ASSERT_NE(g2, nullptr);
    mio_mesh_free(g2);

    // NULL options is an error, not a default: a sizing must be chosen.
    EXPECT_EQ(mio_compute_sdf(cube, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
              nullptr);
    mio_mesh_free(cube);
}

TEST(CApi, ComputeSdfOctreeRefinesAndRejectsAVoxelSizing) {
    mio_mesh* cube = capi_cube_surface();
    mio_compute_sdf_opts opts;
    mio_compute_sdf_opts_init(&opts);
    opts.structure = MIO_SDF_OCTREE;
    opts.root_resolution = 4;
    opts.max_depth = 2;
    opts.distance.watertight_check = MIO_SDF_WATERTIGHT_OFF;

    std::int64_t depth = -1;
    mio_mesh* g = mio_compute_sdf(cube, &opts, nullptr, nullptr, nullptr, &depth, nullptr, nullptr);
    ASSERT_NE(g, nullptr);
    EXPECT_EQ(depth, 2);
    std::int64_t nc = 0, npc = 0;
    ASSERT_EQ(mio_mesh_cell_block_info(g, 0, &nc, &npc, nullptr), MIO_OK);
    EXPECT_GT(nc, 64);
    EXPECT_LT(nc, 16 * 16 * 16);
    mio_mesh_free(g);

    // resolution/cell_size size a voxel grid; an octree's finest cell is already
    // determined, so accepting either would silently ignore one of the two.
    const std::int64_t res[3] = {8, 8, 8};
    opts.resolution = res;
    EXPECT_EQ(mio_compute_sdf(cube, &opts, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
              nullptr);
    mio_mesh_free(cube);
}

TEST(CApi, CropPredicate) {
    mio_mesh* m = mio_mesh_create();
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0,
                                     0, 1, 0, 1, 1, 0, 2, 1, 0, 3, 1, 0};
    ASSERT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 8, 3, pts.data()), MIO_OK);
    const std::vector<std::int64_t> conn = {0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6};
    ASSERT_EQ(mio_mesh_add_cell_block(m, "quad", 3, 4, MIO_INT64, conn.data()), MIO_OK);
    const std::vector<double> t = {0.0, 1.0, 2.0};
    const std::int64_t shape[1] = {3};
    ASSERT_EQ(mio_mesh_append_cell_data(m, "t", MIO_FLOAT64, 1, shape, t.data()), MIO_OK);

    mio_mesh* kept = mio_crop_predicate(m, "t", MIO_REFINE_LT, 1.5, 0);
    ASSERT_NE(kept, nullptr);
    std::int64_t nc = 0, npc = 0;
    ASSERT_EQ(mio_mesh_cell_block_info(kept, 0, &nc, &npc, nullptr), MIO_OK);
    EXPECT_EQ(nc, 2);
    EXPECT_EQ(mio_mesh_num_points(kept), 6);
    mio_mesh_free(kept);

    // An unknown array and an out-of-range comparison both fail rather than
    // defaulting.
    EXPECT_EQ(mio_crop_predicate(m, "nope", MIO_REFINE_LT, 0.0, 0), nullptr);
    EXPECT_EQ(mio_crop_predicate(m, "t", 99, 0.0, 0), nullptr);
    EXPECT_EQ(mio_crop_predicate(nullptr, "t", MIO_REFINE_LT, 0.0, 0), nullptr);
    mio_mesh_free(m);
}

TEST(CApi, VoxelizeInsideKeepsTheInterior) {
    mio_mesh* cube = capi_cube_surface();
    mio_voxel_opts opts;
    mio_voxel_opts_init(&opts);
    const std::int64_t res[3] = {5, 5, 5};
    const double bounds[6] = {-0.5, -0.5, -0.5, 1.5, 1.5, 1.5};
    opts.resolution = res;
    opts.bounds = bounds;
    opts.fill = MIO_VOXEL_INSIDE;
    opts.watertight_check = MIO_SDF_WATERTIGHT_OFF;

    std::int64_t occupied = -1;
    mio_mesh* g = mio_voxelize(cube, &opts, nullptr, nullptr, nullptr, &occupied);
    ASSERT_NE(g, nullptr);
    EXPECT_EQ(occupied, 27);
    mio_mesh_free(g);
    mio_mesh_free(cube);
}

TEST(CApi, VoxelizeRejectsAnUnknownFill) {
    mio_mesh* cube = capi_cube_surface();
    mio_voxel_opts opts;
    mio_voxel_opts_init(&opts);
    const std::int64_t res[3] = {2, 2, 2};
    opts.resolution = res;
    opts.fill = 99;  // drift guard: an out-of-range enum fails, never reinterprets
    EXPECT_EQ(mio_voxelize(cube, &opts, nullptr, nullptr, nullptr, nullptr), nullptr);
    mio_mesh_free(cube);
}

TEST(CApi, SurfaceWatertightCheckCountsDefects) {
    mio_mesh* cube = capi_cube_surface();
    mio_surface_quality q;
    ASSERT_EQ(mio_surface_watertight_check(cube, &q), MIO_OK);
    EXPECT_NE(q.watertight, 0);
    EXPECT_EQ(q.boundary_edges, 0);
    EXPECT_EQ(q.degenerate_triangles, 0);
    mio_mesh_free(cube);

    // A lone triangle is a sheet: three boundary edges, reported as a number.
    const std::vector<double> pts = {0, 0, 0, 1, 0, 0, 0, 1, 0};
    const std::vector<std::int64_t> tri = {0, 1, 2};
    mio_mesh* sheet = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(sheet, MIO_FLOAT64, 3, 3, pts.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(sheet, "triangle", 1, 3, MIO_INT64, tri.data()), MIO_OK);
    ASSERT_EQ(mio_surface_watertight_check(sheet, &q), MIO_OK);
    EXPECT_EQ(q.watertight, 0);
    EXPECT_EQ(q.boundary_edges, 3);
    mio_mesh_free(sheet);

    EXPECT_EQ(mio_surface_watertight_check(nullptr, &q), MIO_ERR_INVALID_ARG);
}

TEST(CApi, SampleDistanceWritesIntoTheCallersBuffer) {
    mio_mesh* cube = capi_cube_surface();
    mio_sdf_opts opts;
    mio_sdf_opts_init(&opts);
    opts.watertight_check = MIO_SDF_WATERTIGHT_OFF;

    const double points[9] = {0.5, 0.5, 0.5, 2.0, 0.5, 0.5, -1.0, 0.5, 0.5};
    double out[3] = {0, 0, 0};
    ASSERT_EQ(mio_sample_distance(cube, points, 3, &opts, out), MIO_OK);
    EXPECT_NEAR(out[0], -0.5, 1e-12);  // the centre, inside
    EXPECT_NEAR(out[1], 1.0, 1e-12);
    EXPECT_NEAR(out[2], 1.0, 1e-12);

    // NULL options means the defaults, not a failure.
    ASSERT_EQ(mio_sample_distance(cube, points, 1, nullptr, out), MIO_OK);
    EXPECT_NEAR(out[0], -0.5, 1e-12);

    // Zero queries is a no-op, not an error; NULL buffers are refused.
    EXPECT_EQ(mio_sample_distance(cube, points, 0, &opts, out), MIO_OK);
    EXPECT_EQ(mio_sample_distance(cube, nullptr, 1, &opts, out), MIO_ERR_INVALID_ARG);
    EXPECT_EQ(mio_sample_distance(cube, points, 1, &opts, nullptr), MIO_ERR_INVALID_ARG);
    EXPECT_EQ(mio_sample_distance(nullptr, points, 1, &opts, out), MIO_ERR_INVALID_ARG);
    mio_mesh_free(cube);
}

TEST(CApi, DistanceToSurfaceAttachesPointData) {
    mio_mesh* cube = capi_cube_surface();
    const std::int64_t dims[3] = {2, 2, 2};
    const double origin[3] = {-0.5, -0.5, -0.5};
    const double spacing[3] = {1.0, 1.0, 1.0};
    mio_mesh* q = mio_grid(dims, origin, spacing, 0);
    ASSERT_NE(q, nullptr);

    mio_sdf_opts opts;
    mio_sdf_opts_init(&opts);
    opts.watertight_check = MIO_SDF_WATERTIGHT_OFF;
    opts.record_inside = 1;

    std::int64_t banded = -1;
    mio_surface_quality quality;
    mio_mesh* out = mio_distance_to_surface(q, cube, &opts, &banded, &quality);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(banded, 0);  // no band in force
    EXPECT_NE(quality.watertight, 0);
    EXPECT_EQ(mio_mesh_num_point_data(out), 2);  // sdf:distance and sdf:inside
    mio_mesh_free(out);

    EXPECT_EQ(mio_distance_to_surface(nullptr, cube, &opts, nullptr, nullptr), nullptr);
    mio_mesh_free(q);
    mio_mesh_free(cube);
}

// ---------------------------------------------------------------------------
// Provenance (v10.16.0): the C ABI bridge over detail::ProvenanceScope.
// ---------------------------------------------------------------------------

TEST(CApi, ProvenanceOffByDefaultIsByteIdenticalToNoScope) {
    mio_mesh* m = capi_cube_surface();
    const std::string a = mt::temp_path("_capi_prov_a.obj");
    const std::string b = mt::temp_path("_capi_prov_b.obj");
    ASSERT_EQ(mio_write(a.c_str(), m, "obj"), MIO_OK);

    ASSERT_EQ(mio_provenance_scope_begin(MIO_PROVENANCE_OFF), MIO_OK);
    ASSERT_EQ(mio_write(b.c_str(), m, "obj"), MIO_OK);
    mio_provenance_scope_end();

    std::ifstream fa(a, std::ios::binary), fb(b, std::ios::binary);
    std::string sa((std::istreambuf_iterator<char>(fa)), std::istreambuf_iterator<char>());
    std::string sb((std::istreambuf_iterator<char>(fb)), std::istreambuf_iterator<char>());
    EXPECT_EQ(sa, sb);

    mio_mesh_free(m);
    std::error_code ec;
    std::filesystem::remove(a, ec);
    std::filesystem::remove(b, ec);
}

TEST(CApi, ProvenanceBestEffortRecordsSourceAndNotes) {
    mio_mesh* m = capi_cube_surface();
    const std::string path = mt::temp_path("_capi_prov.obj");

    ASSERT_EQ(mio_provenance_scope_begin(MIO_PROVENANCE_BEST_EFFORT), MIO_OK);
    mio_provenance_set_source("in.vtu", "vtu");
    mio_provenance_note("regions-dropped", "1 named region(s) dropped");
    ASSERT_EQ(mio_write(path.c_str(), m, "obj"), MIO_OK);
    mio_provenance_scope_end();

    std::ifstream f(path);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    EXPECT_NE(text.find("in.vtu"), std::string::npos);
    EXPECT_NE(text.find("regions-dropped"), std::string::npos);

    // Scope closed: the next write must not see the stale note.
    const std::string path2 = mt::temp_path("_capi_prov2.obj");
    ASSERT_EQ(mio_write(path2.c_str(), m, "obj"), MIO_OK);
    std::ifstream f2(path2);
    std::string text2((std::istreambuf_iterator<char>(f2)), std::istreambuf_iterator<char>());
    EXPECT_EQ(text2.find("in.vtu"), std::string::npos);

    mio_mesh_free(m);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(path2, ec);
}

TEST(CApi, ProvenanceScopeBeginRejectsAnUnknownMode) {
    EXPECT_EQ(mio_provenance_scope_begin(99), MIO_ERR_INVALID_ARG);
    EXPECT_EQ(mio_provenance_scope_begin(-1), MIO_ERR_INVALID_ARG);
}

TEST(CApi, ProvenanceScopeEndWithNoneOpenIsANoOp) {
    mio_provenance_scope_end();  // must not crash
    SUCCEED();
}

TEST(CApi, ProvenanceReadBackThroughMetadata) {
    mio_mesh* m = capi_cube_surface();
    const std::string path = mt::temp_path("_capi_rb.obj");

    ASSERT_EQ(mio_provenance_scope_begin(MIO_PROVENANCE_BEST_EFFORT), MIO_OK);
    mio_provenance_set_source("in.msh", "gmsh");
    mio_provenance_note("regions-dropped", "Side regions dropped");
    ASSERT_EQ(mio_write(path.c_str(), m, "obj"), MIO_OK);
    mio_provenance_scope_end();

    mio_read_metadata* meta = mio_read_metadata_create(path.c_str(), "obj");
    ASSERT_NE(meta, nullptr);
    EXPECT_EQ(mio_read_metadata_provenance_recognised(meta), 1);
    const int64_t n = mio_read_metadata_num_provenance_lines(meta);
    ASSERT_GE(n, 3);

    bool saw_source = false, saw_note = false;
    char buf[512];
    for (int64_t i = 0; i < n; ++i) {
        ASSERT_GT(mio_read_metadata_provenance_line(meta, i, buf, sizeof(buf)), 0);
        std::string line(buf);
        if (line.find("in.msh") != std::string::npos)
            saw_source = true;
        if (line.find("regions-dropped") != std::string::npos)
            saw_note = true;
    }
    EXPECT_TRUE(saw_source);
    EXPECT_TRUE(saw_note);

    // Out-of-range and NULL are errors, not silent zeros.
    EXPECT_EQ(mio_read_metadata_provenance_line(meta, n, buf, sizeof(buf)), -1);
    EXPECT_EQ(mio_read_metadata_num_provenance_lines(nullptr), -1);
    EXPECT_EQ(mio_read_metadata_provenance_recognised(nullptr), -1);

    mio_read_metadata_free(meta);
    mio_mesh_free(m);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(CApi, ProvenanceReadBackIsHonestAboutAForeignFile) {
    const std::string path = mt::temp_path("_capi_foreign.obj");
    {
        std::ofstream f(path);
        f << "# Created by SomeOtherTool 3.2\nv 0 0 0\nv 1 0 0\nv 1 1 0\nf 1 2 3\n";
    }
    mio_read_metadata* meta = mio_read_metadata_create(path.c_str(), "obj");
    ASSERT_NE(meta, nullptr);
    EXPECT_EQ(mio_read_metadata_provenance_recognised(meta), 0);
    EXPECT_EQ(mio_read_metadata_num_provenance_lines(meta), 0);
    mio_read_metadata_free(meta);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

namespace {

/// A regular octahedron: the smallest closed, consistently wound triangle
/// surface, so Gauss-Bonnet applies and every edge is used exactly twice.
mio_mesh* capi_curv_octahedron() {
    const std::vector<double> pts = {1.0,  0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0,  0.0,
                                     0.0, -1.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0, -1.0};
    const std::vector<std::int64_t> conn = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4,
                                            2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
    mio_mesh* m = mio_mesh_create();
    EXPECT_EQ(mio_mesh_set_points(m, MIO_FLOAT64, 6, 3, pts.data()), MIO_OK);
    EXPECT_EQ(mio_mesh_add_cell_block(m, "triangle", 8, 3, MIO_INT64, conn.data()), MIO_OK);
    return m;
}

}  // namespace

TEST(CApi, ComputeCurvatureSatisfiesGaussBonnet) {
    // The tessellation-independent oracle: on a closed surface the angle
    // defects sum to 2*pi*chi, which is 4*pi for anything sphere-like. It is
    // reported through the C ABI precisely so a flat-binding consumer can check
    // a result without reimplementing the estimator.
    mio_mesh* m = capi_curv_octahedron();

    mio_curvature_opts opts;
    mio_curvature_opts_init(&opts);
    EXPECT_EQ(opts.mean, 1);      // ON by default -- an all-zero struct is NOT the default
    EXPECT_EQ(opts.gaussian, 1);
    EXPECT_EQ(opts.dual_area, MIO_CURVATURE_MIXED_VORONOI);
    opts.record_area = 1;
    opts.record_principal = 1;

    mio_curvature_report report;
    mio_mesh* out = mio_compute_curvature(m, &opts, &report);
    ASSERT_NE(out, nullptr) << mio_last_error();
    EXPECT_NEAR(report.total_angle_defect, 4.0 * 3.14159265358979323846, 1e-12);
    EXPECT_EQ(report.num_boundary, 0);
    EXPECT_EQ(report.num_isolated, 0);
    EXPECT_EQ(report.num_degenerate, 0);
    EXPECT_NE(report.quality.watertight, 0);
    EXPECT_EQ(report.quality.inconsistent_pairs, 0);
    EXPECT_EQ(mio_mesh_num_points(out), 6);  // a pure data step
    mio_mesh_free(out);

    // NULL options means every mio_curvature_opts_init default, and a NULL
    // report is simply not written.
    mio_mesh* plain = mio_compute_curvature(m, nullptr, nullptr);
    ASSERT_NE(plain, nullptr) << mio_last_error();
    mio_mesh_free(plain);

    // Barycentric gives the same defect: it changes the dual area, and the
    // defect never touches it.
    opts.dual_area = MIO_CURVATURE_BARYCENTRIC;
    mio_curvature_report bary;
    mio_mesh* out2 = mio_compute_curvature(m, &opts, &bary);
    ASSERT_NE(out2, nullptr) << mio_last_error();
    EXPECT_EQ(bary.total_angle_defect, report.total_angle_defect);
    mio_mesh_free(out2);

    // An out-of-range dual area is refused rather than silently clamped.
    opts.dual_area = 7;
    EXPECT_EQ(mio_compute_curvature(m, &opts, nullptr), nullptr);
    mio_mesh_free(m);

    // A volume mesh is refused by name; no exception crosses the ABI.
    const std::vector<double> tp = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const std::vector<std::int64_t> tc = {0, 1, 2, 3};
    mio_mesh* vol = mio_mesh_create();
    ASSERT_EQ(mio_mesh_set_points(vol, MIO_FLOAT64, 4, 3, tp.data()), MIO_OK);
    ASSERT_EQ(mio_mesh_add_cell_block(vol, "tetra", 1, 4, MIO_INT64, tc.data()), MIO_OK);
    EXPECT_EQ(mio_compute_curvature(vol, nullptr, nullptr), nullptr);
    EXPECT_NE(std::string(mio_last_error()).find("extract_surface"), std::string::npos);
    mio_mesh_free(vol);
}
