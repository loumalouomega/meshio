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
// Tests for shrinkwrap. See operations/shrinkwrap.hpp for the contract; the
// oracles here are stated in the plan that shipped it: projection onto a
// sphere measures zero distance afterwards, a planar target with an offset
// puts every point exactly at that height, the accelerator is unobservable,
// and a crease offsets along the bisector (the one deliberate divergence from
// upstream, sabotage-verified by switching to the face normal).

// System includes
#include <cmath>
#include <cstdint>
#include <vector>

// External includes
#include <gtest/gtest.h>

// Project includes
#include "meshioplusplus/operations/sdf.hpp"
#include "meshioplusplus/operations/shrinkwrap.hpp"
#include "mesh_fixtures.hpp"

using namespace meshioplusplus;

namespace {

using mt::icosphere;
using mt::plane_grid;

/// A point cloud (vertex cells) at the given positions.
Mesh cloud(const std::vector<std::vector<double>>& rPts) {
    std::vector<std::vector<std::int64_t>> v;
    for (std::size_t i = 0; i < rPts.size(); ++i)
        v.push_back({static_cast<std::int64_t>(i)});
    return mt::make_mesh(rPts, "vertex", v);
}

/// Two unit squares meeting at the x axis at 90 degrees (a "book"): one in the
/// z = 0 plane (y >= 0), one in the y = 0 plane (z >= 0), both wound so the
/// normals point into the +y/+z quadrant. The spine is the x axis.
Mesh book() {
    std::vector<std::vector<double>> p = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0},
                                          {0, 1, 0}, {1, 0, 1}, {0, 0, 1}};
    std::vector<std::vector<std::int64_t>> f = {{0, 1, 2}, {0, 2, 3}, {0, 4, 1}, {0, 5, 4}};
    return mt::make_mesh(p, "triangle", f);
}

std::vector<double> coords(const Mesh& rMesh) {
    const NDArray& p = rMesh.Points();
    std::vector<double> out(p.Size());
    for (std::size_t i = 0; i < p.Size(); ++i)
        out[i] = detail::read_double(p, i);
    return out;
}

}  // namespace

TEST(Shrinkwrap, ProjectsOntoASphereSoTheDistanceIsZeroAfterwards) {
    const Mesh target = icosphere(3, 1.0);
    const Mesh source = icosphere(2, 1.3);
    const ShrinkwrapResult r = shrinkwrap(source, target);
    EXPECT_EQ(r.mNumProjected, static_cast<std::int64_t>(source.NumPoints()));
    EXPECT_EQ(r.mNumMissed, 0);
    EXPECT_EQ(r.mNumSkipped, 0);
    EXPECT_NEAR(r.mMaxDisplacement, 0.3, 0.02);
    EXPECT_TRUE(r.mQuality.mWatertight);
    const NDArray d = sample_distance(target, r.mMesh.Points());
    for (std::size_t i = 0; i < d.Size(); ++i)
        EXPECT_NEAR(detail::read_double(d, i), 0.0, 1e-12);
    // Connectivity and cell count untouched: a pure coordinate move.
    EXPECT_EQ(r.mMesh.NumCellBlocks(), source.NumCellBlocks());
    EXPECT_EQ(r.mMesh.Cells(0).NumCells(), source.Cells(0).NumCells());
}

TEST(Shrinkwrap, OffsetOnAPlanePutsEveryPointAtThatHeight) {
    const Mesh target = plane_grid(4);  // z = 0, wound +z
    std::vector<std::vector<double>> pts;
    for (int i = 0; i < 12; ++i)
        pts.push_back({0.7 + 0.2 * i, 1.3 + 0.1 * i, (i % 3 == 0 ? -1.0 : 1.0) * (0.1 + 0.05 * i)});
    const Mesh source = cloud(pts);
    ShrinkwrapOptions o;
    o.mOffset = 0.25;
    const ShrinkwrapResult r = shrinkwrap(source, target, o);
    const std::vector<double> c = coords(r.mMesh);
    double max_disp = 0.0;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        EXPECT_DOUBLE_EQ(c[i * 3 + 0], pts[i][0]);
        EXPECT_DOUBLE_EQ(c[i * 3 + 1], pts[i][1]);
        EXPECT_NEAR(c[i * 3 + 2], 0.25, 1e-15);
        max_disp = std::max(max_disp, std::fabs(pts[i][2] - 0.25));
    }
    EXPECT_NEAR(r.mMaxDisplacement, max_disp, 1e-15);
}

TEST(Shrinkwrap, MaxDistanceLeavesFarPointsAloneAndCountsThem) {
    const Mesh target = plane_grid(4);
    const Mesh source = cloud({{1, 1, 0.05}, {2, 2, 0.5}, {1.5, 2.5, -0.02}, {3, 1, -2.0}});
    ShrinkwrapOptions o;
    o.mMaxDistance = 0.1;
    o.mRecordDistance = true;
    o.mRecordClosestCell = true;
    const ShrinkwrapResult r = shrinkwrap(source, target, o);
    EXPECT_EQ(r.mNumProjected, 2);
    EXPECT_EQ(r.mNumMissed, 2);
    const std::vector<double> c = coords(r.mMesh);
    EXPECT_DOUBLE_EQ(c[2], 0.0);
    EXPECT_DOUBLE_EQ(c[5], 0.5);  // untouched
    EXPECT_DOUBLE_EQ(c[8], 0.0);
    EXPECT_DOUBLE_EQ(c[11], -2.0);  // untouched
    // The distance is recorded for every queried point, hit or miss.
    const double* d = r.mMesh.PointData(kShrinkwrapDistanceName).As<double>();
    EXPECT_NEAR(d[1], 0.5, 1e-15);
    EXPECT_NEAR(d[3], 2.0, 1e-15);
    const std::int64_t* cc = r.mMesh.PointData(kShrinkwrapClosestCellName).As<std::int64_t>();
    EXPECT_GE(cc[0], 0);
    EXPECT_GE(cc[3], 0);
}

TEST(Shrinkwrap, WeightsBlendOrSelect) {
    const Mesh target = plane_grid(4);
    Mesh source = cloud({{1, 1, 1.0}, {2, 2, 1.0}, {3, 3, 1.0}});
    // Float weights blend, bit-exactly x + w*(p - x).
    source.AddPointData("w", mt::data_array({0.5, 0.0, 1.0}));
    ShrinkwrapOptions o;
    o.mWeights = "w";
    const ShrinkwrapResult r = shrinkwrap(source, target, o);
    const std::vector<double> c = coords(r.mMesh);
    EXPECT_DOUBLE_EQ(c[2], 1.0 + 0.5 * (0.0 - 1.0));
    EXPECT_DOUBLE_EQ(c[5], 1.0);  // weight 0: never queried
    EXPECT_DOUBLE_EQ(c[8], 0.0);
    EXPECT_EQ(r.mNumSkipped, 1);
    EXPECT_EQ(r.mNumProjected, 2);
    // Integer weights select.
    Mesh sel = cloud({{1, 1, 1.0}, {2, 2, 1.0}});
    sel.AddPointData("pick", mt::int_data_array({0, 1}));
    o.mWeights = "pick";
    const ShrinkwrapResult r2 = shrinkwrap(sel, target, o);
    const std::vector<double> c2 = coords(r2.mMesh);
    EXPECT_DOUBLE_EQ(c2[2], 1.0);
    EXPECT_DOUBLE_EQ(c2[5], 0.0);
    EXPECT_EQ(r2.mNumSkipped, 1);
    // A wrong name is refused by name.
    o.mWeights = "nope";
    EXPECT_THROW(shrinkwrap(sel, target, o), std::invalid_argument);
}

TEST(Shrinkwrap, TheBucketSizeDoesNotChangeTheAnswer) {
    const Mesh target = icosphere(2, 1.0);
    const Mesh source = icosphere(1, 1.4);
    ShrinkwrapOptions o;
    o.mOffset = 0.05;
    const std::vector<double> ref = coords(shrinkwrap(source, target, o).mMesh);
    for (const double cell : {0.05, 0.3, 5.0}) {
        o.mGridCellSize = cell;
        const ShrinkwrapResult r = shrinkwrap(source, target, o);
        const std::vector<double> c = coords(r.mMesh);
        ASSERT_EQ(c.size(), ref.size());
        for (std::size_t i = 0; i < c.size(); ++i)
            EXPECT_EQ(c[i], ref[i]) << "bucket " << cell << " entry " << i;
    }
}

// The one deliberate divergence from upstream. A point on the CONVEX side of
// the spine, on its bisector, projects onto the spine (an EDGE hit, two
// equidistant faces -- on the concave side the nearest feature would be a
// face), and
// the offset must go along the bisector -- the feature pseudonormal -- not
// along whichever face won the tie-break. Sabotaging the kernel to return the
// face normal moves the point along one face and fails this.
TEST(Shrinkwrap, ACreaseOffsetsAlongTheBisector) {
    const Mesh target = book();
    const double s = 0.5 / std::sqrt(2.0);
    const Mesh source = cloud({{0.5, -s, -s}});
    ShrinkwrapOptions o;
    o.mOffset = 0.1;
    const ShrinkwrapResult r = shrinkwrap(source, target, o);
    const std::vector<double> c = coords(r.mMesh);
    const double b = 0.1 / std::sqrt(2.0);
    EXPECT_NEAR(c[0], 0.5, 1e-15);
    EXPECT_NEAR(c[1], b, 1e-15);
    EXPECT_NEAR(c[2], b, 1e-15);
    // Under area weighting the same answer: the edge normal is the plain sum
    // of the two unit face normals under either weighting.
    o.mNormalWeight = SdfPseudonormalWeight::Area;
    const std::vector<double> c2 = coords(shrinkwrap(source, target, o).mMesh);
    EXPECT_NEAR(c2[1], b, 1e-15);
    EXPECT_NEAR(c2[2], b, 1e-15);
}

TEST(Shrinkwrap, AVolumeSourceMovesEveryPointIncludingInteriorOnes) {
    const Mesh target = icosphere(3, 1.0);
    // A tetra whose corners are all inside the unit sphere.
    const Mesh source = mt::tet_mesh();
    const ShrinkwrapResult r = shrinkwrap(source, target);
    EXPECT_EQ(r.mNumProjected, static_cast<std::int64_t>(source.NumPoints()));
    const NDArray d = sample_distance(target, r.mMesh.Points());
    for (std::size_t i = 0; i < d.Size(); ++i)
        EXPECT_NEAR(detail::read_double(d, i), 0.0, 1e-12);
    EXPECT_EQ(std::string(r.mMesh.Cells(0).Type()), "tetra");
}

TEST(Shrinkwrap, DataRegionsAndDtypeSurvive) {
    const Mesh target = plane_grid(3);
    Mesh source = mt::data_mesh();
    const ShrinkwrapResult r = shrinkwrap(source, target);
    EXPECT_EQ(r.mMesh.Points().Dtype(), source.Points().Dtype());
    EXPECT_EQ(r.mMesh.PointDataNames(), source.PointDataNames());
    EXPECT_EQ(r.mMesh.CellDataNames(), source.CellDataNames());
    EXPECT_EQ(r.mMesh.NumRegions(), source.NumRegions());
}

TEST(Shrinkwrap, RefusesAVolumeTargetByName) {
    const Mesh source = mt::tri_mesh();
    try {
        shrinkwrap(source, mt::tet_mesh());
        FAIL() << "expected a throw";
    } catch (const std::invalid_argument& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("meshio++: shrinkwrap: target:"), std::string::npos) << what;
        EXPECT_NE(what.find("extract_surface"), std::string::npos) << what;
    }
}
