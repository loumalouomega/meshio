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
// Tests for repair. The oracles are structural: soup_quality of the OUTPUT
// (no inconsistent pairs, no boundary edges after a fill), the signed volume
// of a closed component, Euler's formula for a filled surface, and -- the one
// that discriminates the topological half-edge rule from upstream's normal
// test -- a consistently wound strip folded past 90 degrees that must NOT be
// touched.

// System includes
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

// External includes
#include <gtest/gtest.h>

// Project includes
#include "meshioplusplus/detail/surface_distance.hpp"
#include "meshioplusplus/operations/repair.hpp"
#include "meshioplusplus/detail/geometry.hpp"
#include "mesh_fixtures.hpp"

using namespace meshioplusplus;

namespace {

using mt::icosphere;
using mt::open_cylinder;
using mt::plane_grid;

/// Flip the winding of every triangle whose index is in @p rWhich.
Mesh flipped(const Mesh& rMesh, const std::vector<std::size_t>& rWhich) {
    std::vector<std::vector<double>> p;
    const NDArray& pts = rMesh.Points();
    for (std::size_t i = 0; i < rMesh.NumPoints(); ++i)
        p.push_back({detail::read_double(pts, i * 3), detail::read_double(pts, i * 3 + 1),
                     detail::read_double(pts, i * 3 + 2)});
    std::vector<std::vector<std::int64_t>> f;
    const NDArray& conn = rMesh.Cells(0).Conn();
    for (std::size_t c = 0; c < rMesh.Cells(0).NumCells(); ++c)
        f.push_back({detail::read_int(conn, c * 3), detail::read_int(conn, c * 3 + 1),
                     detail::read_int(conn, c * 3 + 2)});
    for (const std::size_t w : rWhich)
        std::swap(f[w][1], f[w][2]);
    return mt::make_mesh(p, "triangle", f);
}

std::vector<std::size_t> every_triangle(const Mesh& rMesh) {
    std::vector<std::size_t> all;
    for (std::size_t c = 0; c < rMesh.Cells(0).NumCells(); ++c)
        all.push_back(c);
    return all;
}

/// Two quads (four triangles) sharing the edge x = 1, the second folded back
/// over the first at 150 degrees, wound CONSISTENTLY: the shared edge is
/// traversed B->C by one side and C->B by the other, while the two normals
/// have a dot product of cos(150) < 0.
Mesh folded_strip() {
    const double th = 150.0 * mt::mt_kPi / 180.0;
    std::vector<std::vector<double>> p = {{0, 0, 0},
                                          {1, 0, 0},
                                          {1, 1, 0},
                                          {0, 1, 0},
                                          {1 + std::cos(th), 0, std::sin(th)},
                                          {1 + std::cos(th), 1, std::sin(th)}};
    std::vector<std::vector<std::int64_t>> f = {{0, 1, 2}, {0, 2, 3}, {2, 1, 4}, {2, 4, 5}};
    return mt::make_mesh(p, "triangle", f);
}

/// Two closed tetrahedron skins that share exactly one vertex (a bowtie).
Mesh two_tets_pinched() {
    std::vector<std::vector<double>> p = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
                                          {0, 0, 2}, {1, 0, 2}, {0, 1, 2}};
    // Outward-wound skins of tetra (0,1,2,3) and tetra (3,4,5,6)... with 3 the
    // apex of the first and the base corner of the second.
    std::vector<std::vector<std::int64_t>> f = {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {0, 3, 2},
                                                {3, 5, 4}, {3, 4, 6}, {4, 5, 6}, {3, 6, 5}};
    return mt::make_mesh(p, "triangle", f);
}

/// The volume a triangle surface encloses, by the divergence theorem over its
/// triangles (compute_stats' mSignedVolume counts 3-D cells, of which a
/// surface has none).
double signed_volume(const Mesh& rMesh) {
    const NDArray& p = rMesh.Points();
    double vol = 0.0;
    for (const auto cb : rMesh.CellRange()) {
        if (std::string(cb.Type()) != "triangle")
            continue;
        const NDArray& conn = cb.Conn();
        for (std::size_t c = 0; c < cb.NumCells(); ++c) {
            detail::Vec3 v[3];
            for (std::size_t k = 0; k < 3; ++k)
                v[k] = detail::read_point(p, 3, detail::read_int(conn, c * 3 + k));
            vol += detail::triple_product(v[0], v[1], v[2]) / 6.0;
        }
    }
    return vol;
}

std::size_t num_edges(const Mesh& rMesh) {
    return detail::build_surface_edges(detail::build_triangle_soup(rMesh, "")).size();
}

std::size_t num_cells(const Mesh& rMesh) {
    std::size_t n = 0;
    for (const auto cb : rMesh.CellRange())
        n += cb.NumCells();
    return n;
}

}  // namespace

// --------------------------------------------------------------------------- //
// Orientation
// --------------------------------------------------------------------------- //

TEST(Repair, RewindsARandomlyFlippedSphere) {
    const Mesh sphere = icosphere(2, 1.0);
    std::mt19937 rng(7);
    std::vector<std::size_t> which;
    for (std::size_t c = 0; c < sphere.Cells(0).NumCells(); ++c)
        if (rng() % 10 == 0)
            which.push_back(c);
    ASSERT_GT(which.size(), 5u);
    const Mesh broken = flipped(sphere, which);
    const RepairResult r = repair(broken);
    EXPECT_GT(r.mQualityBefore.mInconsistentPairs, 0);
    EXPECT_EQ(r.mQualityAfter.mInconsistentPairs, 0);
    EXPECT_TRUE(r.mQualityAfter.mWatertight);
    EXPECT_EQ(r.mNumFlipped, static_cast<std::int64_t>(which.size()));
    EXPECT_EQ(r.mNumComponents, 1);
    EXPECT_EQ(r.mLargestComponent, static_cast<std::int64_t>(sphere.Cells(0).NumCells()));
    EXPECT_EQ(r.mNumUnorientable, 0);
    EXPECT_GT(signed_volume(r.mMesh), 0.0);
    // Nothing was added or removed.
    EXPECT_EQ(r.mMesh.NumPoints(), sphere.NumPoints());
    EXPECT_EQ(r.mMesh.NumCellBlocks(), 1u);
    EXPECT_EQ(num_cells(r.mMesh), num_cells(sphere));
}

TEST(Repair, OrientsAnInwardSphereOutwardUnlessToldNotTo) {
    const Mesh sphere = icosphere(1, 1.0);
    const Mesh inward = flipped(sphere, every_triangle(sphere));
    ASSERT_LT(signed_volume(inward), 0.0);
    const RepairResult r = repair(inward);
    EXPECT_EQ(r.mNumFlipped, static_cast<std::int64_t>(sphere.Cells(0).NumCells()));
    EXPECT_EQ(r.mNumOrientedOutward, 1);
    EXPECT_GT(signed_volume(r.mMesh), 0.0);
    RepairOptions o;
    o.mOrientOutward = false;
    const RepairResult r2 = repair(inward, o);
    EXPECT_EQ(r2.mNumFlipped, 0);  // already consistent; fewest flips keeps it
    EXPECT_EQ(r2.mNumOrientedOutward, 0);
    EXPECT_LT(signed_volume(r2.mMesh), 0.0);
    EXPECT_EQ(r2.mQualityAfter.mInconsistentPairs, 0);
}

// The discriminating oracle: the strip is consistently wound by the
// half-edge rule even though its normals disagree by 150 degrees, so nothing
// may be flipped. Upstream's normal-dot test flips here.
TEST(Repair, DoesNotTouchAConsistentlyWoundCrease) {
    const Mesh strip = folded_strip();
    RepairOptions o;
    o.mFillHoles = false;  // isolate orientation: a filled strip closes and gets an outward pass
    const RepairResult r = repair(strip, o);
    EXPECT_EQ(r.mQualityBefore.mInconsistentPairs, 0);
    EXPECT_EQ(r.mNumFlipped, 0);
    EXPECT_EQ(r.mNumComponents, 1);
    EXPECT_EQ(r.mQualityAfter.mInconsistentPairs, 0);
    // And when one panel IS flipped, exactly that panel comes back.
    const Mesh broken = flipped(strip, {2, 3});
    const RepairResult r2 = repair(broken, o);
    EXPECT_EQ(r2.mNumFlipped, 2);
    EXPECT_EQ(r2.mQualityAfter.mInconsistentPairs, 0);
}

TEST(Repair, FixOrientationOffLeavesTheWindingAlone) {
    const Mesh sphere = icosphere(1, 1.0);
    const Mesh broken = flipped(sphere, {0, 5, 9});
    RepairOptions o;
    o.mFixOrientation = false;
    const RepairResult r = repair(broken, o);
    EXPECT_EQ(r.mNumFlipped, 0);
    EXPECT_EQ(r.mQualityAfter.mInconsistentPairs, r.mQualityBefore.mInconsistentPairs);
    EXPECT_EQ(r.mNumComponents, 1);  // components are still reported
}

// --------------------------------------------------------------------------- //
// Holes
// --------------------------------------------------------------------------- //

TEST(Repair, FillsAnOpenCylindersTwoRimsAndClosesIt) {
    const Mesh cyl = open_cylinder(12, 4, 1.0, 2.0);
    RepairOptions o;
    o.mMaxHoleEdges = 0;  // no limit
    const RepairResult r = repair(cyl, o);
    EXPECT_EQ(r.mQualityBefore.mBoundaryEdges, 24);
    EXPECT_EQ(r.mNumHolesDetected, 2);
    EXPECT_EQ(r.mNumHolesFilled, 2);
    EXPECT_EQ(r.mNumHolesSkipped, 0);
    EXPECT_EQ(r.mNumFacesAdded, 24);
    EXPECT_EQ(r.mNumPointsAdded, 2);
    EXPECT_TRUE(r.mQualityAfter.mWatertight);
    EXPECT_EQ(r.mMesh.NumCellBlocks(), 2u);  // the input block plus the fill block
    EXPECT_EQ(r.mMesh.Cells(1).NumCells(), 24u);
    // Euler: V - E + F == 2 for a closed sphere-like surface.
    const std::size_t V = r.mMesh.NumPoints(), E = num_edges(r.mMesh), F = num_cells(r.mMesh);
    EXPECT_EQ(static_cast<long>(V) - static_cast<long>(E) + static_cast<long>(F), 2);
    // Closed and outward: the volume is the inscribed 12-gon prism's,
    // 12 * (1/2) sin(30 deg) * h = 6, exactly.
    const double vol = signed_volume(r.mMesh);
    EXPECT_NEAR(vol, 12.0 * 0.5 * std::sin(mt::mt_kPi / 6.0) * 2.0, 1e-12);
    // A quiet run leaves the block count alone.
    const RepairResult closed = repair(r.mMesh, o);
    EXPECT_EQ(closed.mNumHolesDetected, 0);
    EXPECT_EQ(closed.mMesh.NumCellBlocks(), 2u);
}

TEST(Repair, TheHoleSizeLimitIsHonoured) {
    const Mesh cyl = open_cylinder(12, 4, 1.0, 2.0);
    RepairOptions o;
    o.mMaxHoleEdges = 3;
    const RepairResult r = repair(cyl, o);
    EXPECT_EQ(r.mNumHolesDetected, 2);
    EXPECT_EQ(r.mNumHolesSkipped, 2);
    EXPECT_EQ(r.mNumFacesAdded, 0);
    EXPECT_EQ(r.mQualityAfter.mBoundaryEdges, 24);
    EXPECT_EQ(r.mMesh.NumCellBlocks(), 1u);
    o.mMaxHoleEdges = 12;
    const RepairResult r2 = repair(cyl, o);
    EXPECT_EQ(r2.mNumHolesFilled, 2);
    o.mFillHoles = false;
    const RepairResult r3 = repair(cyl, o);
    EXPECT_EQ(r3.mNumHolesDetected, 0);
    EXPECT_EQ(r3.mQualityAfter.mBoundaryEdges, 24);
}

// The fill's winding: a sphere minus one triangle comes back closed and
// consistent. Winding the fan the other way (upstream's rule) leaves three
// inconsistent pairs here.
TEST(Repair, FillsASingleMissingTriangleConsistently) {
    const Mesh sphere = icosphere(2, 1.0);
    std::vector<std::vector<double>> p;
    const NDArray& pts = sphere.Points();
    for (std::size_t i = 0; i < sphere.NumPoints(); ++i)
        p.push_back({detail::read_double(pts, i * 3), detail::read_double(pts, i * 3 + 1),
                     detail::read_double(pts, i * 3 + 2)});
    std::vector<std::vector<std::int64_t>> f;
    const NDArray& conn = sphere.Cells(0).Conn();
    for (std::size_t c = 1; c < sphere.Cells(0).NumCells(); ++c)  // drop triangle 0
        f.push_back({detail::read_int(conn, c * 3), detail::read_int(conn, c * 3 + 1),
                     detail::read_int(conn, c * 3 + 2)});
    const Mesh holed = mt::make_mesh(p, "triangle", f);
    const RepairResult r = repair(holed);
    EXPECT_EQ(r.mQualityBefore.mBoundaryEdges, 3);
    EXPECT_EQ(r.mNumHolesFilled, 1);
    EXPECT_EQ(r.mNumFacesAdded, 3);
    EXPECT_EQ(r.mNumPointsAdded, 1);
    EXPECT_EQ(r.mNumFlipped, 0);
    EXPECT_TRUE(r.mQualityAfter.mWatertight);
    const std::size_t V = r.mMesh.NumPoints(), E = num_edges(r.mMesh), F = num_cells(r.mMesh);
    EXPECT_EQ(static_cast<long>(V) - static_cast<long>(E) + static_cast<long>(F), 2);
    EXPECT_GT(signed_volume(r.mMesh), 0.0);
}

TEST(Repair, ANonTraceableLoopIsSkippedNotGuessed) {
    // Two triangles sharing only a vertex, with no other triangles: their
    // boundary vertex 0 has boundary degree 4, so with splitting OFF the loop
    // through it cannot be traced. With splitting on it becomes two loops of
    // length 3, each filled.
    const Mesh bow = mt::make_mesh({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}},
                                   "triangle", {{0, 1, 2}, {0, 3, 4}});
    RepairOptions o;
    o.mSplitNonManifold = false;
    const RepairResult r = repair(bow, o);
    EXPECT_EQ(r.mNumVerticesSplit, 0);
    EXPECT_GT(r.mNumHolesSkipped, 0);
    EXPECT_EQ(r.mNumHolesFilled, 0);
    o.mSplitNonManifold = true;
    const RepairResult r2 = repair(bow, o);
    EXPECT_EQ(r2.mNumVerticesSplit, 1);
    EXPECT_EQ(r2.mNumHolesDetected, 2);
    EXPECT_EQ(r2.mNumHolesFilled, 2);
}

// --------------------------------------------------------------------------- //
// Bowties
// --------------------------------------------------------------------------- //

TEST(Repair, SplitsAPinchedVertexIntoTwoClosedComponents) {
    const Mesh pinched = two_tets_pinched();
    const RepairResult r = repair(pinched);
    EXPECT_EQ(r.mNumVerticesSplit, 1);
    EXPECT_EQ(r.mNumPointsAdded, 1);
    EXPECT_EQ(r.mMesh.NumPoints(), pinched.NumPoints() + 1);
    EXPECT_EQ(r.mNumComponents, 2);
    EXPECT_TRUE(r.mQualityAfter.mWatertight);
    EXPECT_EQ(r.mNumHolesDetected, 0);
    EXPECT_GT(signed_volume(r.mMesh), 0.0);
    // The copy sits exactly on its source.
    const NDArray& p = r.mMesh.Points();
    for (std::size_t d = 0; d < 3; ++d)
        EXPECT_EQ(detail::read_double(p, 7 * 3 + d), detail::read_double(p, 3 * 3 + d));
    RepairOptions o;
    o.mSplitNonManifold = false;
    const RepairResult r2 = repair(pinched, o);
    EXPECT_EQ(r2.mNumVerticesSplit, 0);
    EXPECT_EQ(r2.mMesh.NumPoints(), pinched.NumPoints());
    EXPECT_EQ(r2.mNumComponents, 2);  // no shared EDGE, so still two components
}

// --------------------------------------------------------------------------- //
// Bookkeeping
// --------------------------------------------------------------------------- //

TEST(Repair, QuadsAreTriangulatedBlocksStayOneToOneAndMapsAreFirstChild) {
    Mesh m = mt::data_mesh();  // a triangle block and a quad block, with data
    RepairOptions o;
    o.mFillHoles = false;  // an open patch; keep the block count at the input's
    const RepairResult r = repair(m, o);
    ASSERT_EQ(r.mMesh.NumCellBlocks(), 2u);
    EXPECT_EQ(std::string(r.mMesh.Cells(0).Type()), "triangle");
    EXPECT_EQ(std::string(r.mMesh.Cells(1).Type()), "triangle");
    EXPECT_EQ(r.mMesh.Cells(0).NumCells(), 2u);
    EXPECT_EQ(r.mMesh.Cells(1).NumCells(), 2u);  // one quad -> two triangles
    ASSERT_EQ(r.mCellMaps.size(), 2u);
    EXPECT_EQ(r.mCellMaps[1].Size(), 1u);
    EXPECT_EQ(detail::read_int(r.mCellMaps[1], 0), 0);
    EXPECT_EQ(r.mPointMap.Size(), m.NumPoints());
    for (const std::string& name : m.CellDataNames()) {
        EXPECT_EQ(r.mMesh.CellDataNumBlocks(name), 2u);
        EXPECT_EQ(detail::rows(r.mMesh.CellData(name, 1)), 2u);
    }
    EXPECT_EQ(r.mMesh.PointDataNames(), m.PointDataNames());
    EXPECT_EQ(r.mMesh.Points().Dtype(), m.Points().Dtype());
}

TEST(Repair, DataAndRegionsFollowTheNewPointsAndCells) {
    // An open cylinder with point data T = z and a cell tag, plus a Point
    // region on the top rim and a Cell region on the first strip.
    Mesh cyl = open_cylinder(6, 2, 1.0, 2.0);
    const std::size_t n = cyl.NumPoints();
    std::vector<double> T(n);
    const NDArray& pts = cyl.Points();
    for (std::size_t i = 0; i < n; ++i)
        T[i] = detail::read_double(pts, i * 3 + 2);
    cyl.AddPointData("T", mt::data_array(T));
    std::vector<std::int32_t> tag(cyl.Cells(0).NumCells(), 7);
    cyl.AddCellData("tag", {mt::int_data_array(tag)});
    {
        meshioplusplus::Region top;
        top.mName = "top";
        top.mKind = RegionKind::Point;
        std::vector<std::int64_t> ids;
        for (std::size_t i = 12; i < 18; ++i)
            ids.push_back(static_cast<std::int64_t>(i));
        top.mEntries = NDArray::Uninit(DType::Int64, {ids.size()});
        std::memcpy(top.mEntries.Data(), ids.data(), ids.size() * sizeof(std::int64_t));
        cyl.AddRegion(top);
    }
    RepairOptions o;
    o.mMaxHoleEdges = 0;
    o.mRecordProvenance = true;
    const RepairResult r = repair(cyl, o);
    ASSERT_EQ(r.mNumHolesFilled, 2);
    ASSERT_EQ(r.mMesh.NumPoints(), n + 2);
    // Centroid point data = the mean of the rim's rows: z = -1 and z = +1.
    const double* t = r.mMesh.PointData("T").As<double>();
    const double c0 = t[n], c1 = t[n + 1];
    EXPECT_NEAR(std::min(c0, c1), -1.0, 1e-12);
    EXPECT_NEAR(std::max(c0, c1), 1.0, 1e-12);
    // Fill cell data: 0 for an integer array.
    ASSERT_EQ(r.mMesh.CellDataNumBlocks("tag"), 2u);
    const NDArray& fill = r.mMesh.CellData("tag", 1);
    EXPECT_EQ(detail::rows(fill), 12u);
    for (std::size_t i = 0; i < 12; ++i)
        EXPECT_EQ(detail::read_int(fill, i), 0);
    // Provenance.
    const std::int64_t* pp = r.mMesh.PointData(kRepairParentPointName).As<std::int64_t>();
    EXPECT_EQ(pp[0], 0);
    EXPECT_EQ(pp[n], -1);
    EXPECT_EQ(pp[n + 1], -1);
    const NDArray& hole = r.mMesh.CellData(kRepairHoleName, 1);
    EXPECT_EQ(detail::read_int(hole, 0), 0);
    EXPECT_EQ(detail::read_int(hole, 11), 1);
    EXPECT_EQ(detail::read_int(r.mMesh.CellData(kRepairHoleName, 0), 0), -1);
    // The Point region survived with its six entries.
    ASSERT_TRUE(r.mMesh.HasRegion("top", RegionKind::Point));
    EXPECT_EQ(r.mMesh.Region(r.mMesh.FindRegion("top", RegionKind::Point)).mEntries.Size(), 6u);
}

TEST(Repair, ASplitCopyJoinsItsSourcesPointRegion) {
    Mesh pinched = two_tets_pinched();
    meshioplusplus::Region reg;
    reg.mName = "apex";
    reg.mKind = RegionKind::Point;
    const std::int64_t ids[2] = {3, 5};
    reg.mEntries = NDArray::Uninit(DType::Int64, {2});
    std::memcpy(reg.mEntries.Data(), ids, sizeof(ids));
    pinched.AddRegion(reg);
    const RepairResult r = repair(pinched);
    ASSERT_EQ(r.mNumVerticesSplit, 1);
    const meshioplusplus::Region& out =
        r.mMesh.Region(r.mMesh.FindRegion("apex", RegionKind::Point));
    ASSERT_EQ(out.mEntries.Size(), 3u);
    EXPECT_EQ(detail::read_int(out.mEntries, 2), 7);  // the copy of point 3
}

TEST(Repair, LowerDimensionalBlocksRideAlong) {
    Mesh m = plane_grid(2);
    m.AddCellBlock("line", mt::conn_from({{0, 1}, {1, 2}}));
    const RepairResult r = repair(m);
    // The surface block, the line block carried verbatim, then the rim's fill.
    ASSERT_EQ(r.mMesh.NumCellBlocks(), 3u);
    EXPECT_EQ(std::string(r.mMesh.Cells(1).Type()), "line");
    EXPECT_EQ(r.mMesh.Cells(1).NumCells(), 2u);
    EXPECT_EQ(r.mCellMaps.size(), 2u);
    EXPECT_EQ(r.mNumHolesDetected, 1);  // the grid's rim
    EXPECT_EQ(r.mNumHolesFilled, 1);
    EXPECT_EQ(r.mMesh.Cells(2).NumCells(), 8u);
}

TEST(Repair, WeldsFirstWhenAsked) {
    // A square split into two triangles that do NOT share their diagonal's
    // points: four boundary edges become two internal ones after the weld.
    const Mesh split =
        mt::make_mesh({{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 0, 0}, {1, 1, 0}, {0, 1, 0}},
                      "triangle", {{0, 1, 2}, {3, 4, 5}});
    RepairOptions o;
    o.mFillHoles = false;
    const RepairResult plain = repair(split, o);
    EXPECT_EQ(plain.mQualityAfter.mBoundaryEdges, 6);
    EXPECT_EQ(plain.mNumComponents, 2);
    o.mWeldTolerance = 1e-9;
    const RepairResult welded = repair(split, o);
    EXPECT_EQ(welded.mPointsWelded, 2);
    EXPECT_EQ(welded.mMesh.NumPoints(), 4u);
    EXPECT_EQ(welded.mQualityAfter.mBoundaryEdges, 4);
    EXPECT_EQ(welded.mNumComponents, 1);
    EXPECT_EQ(welded.mPointMap.Size(), 6u);
}

TEST(Repair, RefusesOutOfScopeInputByName) {
    auto expect_msg = [](const Mesh& m, const char* pNeedle) {
        try {
            repair(m);
            FAIL() << "expected a throw";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find(pNeedle), std::string::npos) << e.what();
        }
    };
    expect_msg(mt::tet_mesh(), "extract_surface");
    expect_msg(mt::triangle6_mesh(), "linearize");
    expect_msg(mt::line_mesh(), "no surface");
}

TEST(Repair, IsDeterministicAcrossRuns) {
    const Mesh sphere = icosphere(2, 1.0);
    const Mesh broken = flipped(sphere, {1, 4, 8, 15, 16, 23});
    const RepairResult a = repair(broken);
    const RepairResult b = repair(broken);
    const NDArray& ca = a.mMesh.Cells(0).Conn();
    const NDArray& cb = b.mMesh.Cells(0).Conn();
    ASSERT_EQ(ca.Size(), cb.Size());
    for (std::size_t i = 0; i < ca.Size(); ++i)
        EXPECT_EQ(detail::read_int(ca, i), detail::read_int(cb, i));
}
