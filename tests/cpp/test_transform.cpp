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
// Tests for the affine transform operation.

// System includes
#include <cmath>

// External includes
#include <gtest/gtest.h>

// Project includes
#include "mesh_fixtures.hpp"
#include "meshioplusplus/detail/value_io.hpp"
#include "meshioplusplus/operations/transform.hpp"

namespace {

using meshioplusplus::Mesh;
using meshioplusplus::transform;

Mesh unit_cube() {
    return mt::make_mesh(
        {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}},
        "hexahedron", {{0, 1, 2, 3, 4, 5, 6, 7}});
}

double px(const Mesh& m, std::size_t node, std::size_t comp) {
    return meshioplusplus::detail::read_double(m.Points(), node * m.PointDim() + comp);
}

TEST(Transform, TranslateMovesPointsOnly) {
    Mesh m = unit_cube();
    Mesh out = transform(m, meshioplusplus::transform_translation(10, 20, 30));
    EXPECT_NEAR(px(out, 1, 0), 11.0, 1e-12);
    EXPECT_NEAR(px(out, 1, 1), 20.0, 1e-12);
    // connectivity unchanged
    ASSERT_EQ(out.NumCellBlocks(), 1u);
    EXPECT_EQ(out.Cells(0).NumCells(), 1u);
}

TEST(Transform, Rotate90AboutZ) {
    Mesh m = unit_cube();
    Mesh out = transform(m, meshioplusplus::transform_rotation(0, 0, 1, M_PI / 2.0));
    // (1,0,0) -> (0,1,0)
    EXPECT_NEAR(px(out, 1, 0), 0.0, 1e-12);
    EXPECT_NEAR(px(out, 1, 1), 1.0, 1e-12);
}

TEST(Transform, UnitScale) {
    Mesh m = unit_cube();
    Mesh out = transform(m, meshioplusplus::transform_units(0.001));
    EXPECT_NEAR(px(out, 6, 0), 0.001, 1e-15);
    EXPECT_NEAR(px(out, 6, 2), 0.001, 1e-15);
}

TEST(Transform, MatrixEqualsTranslate) {
    Mesh m = unit_cube();
    double mat[16] = {1, 0, 0, 5, 0, 1, 0, 6, 0, 0, 1, 7, 0, 0, 0, 1};
    Mesh out = transform(m, meshioplusplus::transform_from_matrix(mat));
    EXPECT_NEAR(px(out, 0, 0), 5.0, 1e-12);
    EXPECT_NEAR(px(out, 0, 1), 6.0, 1e-12);
    EXPECT_NEAR(px(out, 0, 2), 7.0, 1e-12);
}

double cd(const Mesh& m, const char* name, std::size_t comp) {
    return meshioplusplus::detail::read_double(m.CellData(name, 0), comp);
}

// A vector living on a CELL rotates exactly as one living on a point does.
// Before v10.33.0 `rotate_vector_data` reached point data only, and cell data
// rode through in the old frame -- silently, since nothing about the array
// says which frame it is in.
TEST(Transform, RotateVectorDataReachesCellDataToo) {
    Mesh m = unit_cube();
    m.AddCellData("vel", {mt::data_array({1.0, 0.0, 0.0}, 3)});
    Mesh out = transform(m, meshioplusplus::transform_rotation(0, 0, 1, M_PI / 2.0), true);
    // (1,0,0) -> (0,1,0)
    EXPECT_NEAR(cd(out, "vel", 0), 0.0, 1e-12);
    EXPECT_NEAR(cd(out, "vel", 1), 1.0, 1e-12);
    EXPECT_NEAR(cd(out, "vel", 2), 0.0, 1e-12);
}

TEST(Transform, RotateVectorDataRotatesACellTensor) {
    Mesh m = unit_cube();
    // diag(1, 2, 3) under a quarter turn about z becomes diag(2, 1, 3).
    m.AddCellData("stress", {mt::data_array({1, 0, 0, 0, 2, 0, 0, 0, 3}, 9)});
    Mesh out = transform(m, meshioplusplus::transform_rotation(0, 0, 1, M_PI / 2.0), true);
    EXPECT_NEAR(cd(out, "stress", 0), 2.0, 1e-12);
    EXPECT_NEAR(cd(out, "stress", 4), 1.0, 1e-12);
    EXPECT_NEAR(cd(out, "stress", 8), 3.0, 1e-12);
}

TEST(Transform, CellDataIsUntouchedWithoutTheFlag) {
    Mesh m = unit_cube();
    m.AddCellData("vel", {mt::data_array({1.0, 0.0, 0.0}, 3)});
    Mesh out = transform(m, meshioplusplus::transform_rotation(0, 0, 1, M_PI / 2.0));
    EXPECT_NEAR(cd(out, "vel", 0), 1.0, 1e-12);
    EXPECT_NEAR(cd(out, "vel", 1), 0.0, 1e-12);
}

TEST(Transform, IntegerCellDataIsNeverRotated) {
    Mesh m = unit_cube();
    m.AddCellData("tag", {mt::int_data_array({7, 8, 9})});
    Mesh out = transform(m, meshioplusplus::transform_rotation(0, 0, 1, M_PI / 2.0), true);
    EXPECT_EQ(cd(out, "tag", 0), 7.0);
    EXPECT_EQ(cd(out, "tag", 1), 8.0);
}

}  // namespace
