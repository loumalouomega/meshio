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
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

// External includes
#include <gtest/gtest.h>

// Project includes
#include "meshioplusplus/operations/curvature.hpp"
#include "mesh_fixtures.hpp"

using namespace meshioplusplus;

namespace {

constexpr double kPi = 3.141592653589793238462643383279;

using mt::icosphere;
using mt::open_cylinder;
using mt::plane_grid;

const double* array_of(const Mesh& rMesh, const char* pName) {
    return rMesh.PointData(pName).As<double>();
}

}  // namespace

// --------------------------------------------------------------------------- //
// Gauss-Bonnet: the structural oracle. For a closed surface the angle defects
// sum to 2*pi*chi -- 4*pi for a sphere -- whatever the tessellation and
// whichever dual area was chosen. Nothing else here is tessellation-independent.
// --------------------------------------------------------------------------- //
TEST(Curvature, GaussBonnetOnClosedSurfaces) {
    const double four_pi = 4.0 * kPi;
    for (int s = 0; s <= 3; ++s) {
        const CurvatureResult r = compute_curvature(icosphere(s, 1.0));
        EXPECT_NEAR(r.mTotalAngleDefect, four_pi, 1e-9) << "icosphere subdivision " << s;
    }
    // A radius the sphere formulas care about but Gauss-Bonnet does not.
    EXPECT_NEAR(compute_curvature(icosphere(2, 7.0)).mTotalAngleDefect, four_pi, 1e-9);
}

TEST(Curvature, GaussBonnetIsIndependentOfTheDualArea) {
    CurvatureOptions vor;
    CurvatureOptions bary;
    bary.mDualArea = CurvatureDualArea::Barycentric;
    const Mesh m = icosphere(2, 1.0);
    // Bit-identical: the defect never touches the area at all.
    EXPECT_EQ(compute_curvature(m, vor).mTotalAngleDefect,
              compute_curvature(m, bary).mTotalAngleDefect);
}

// --------------------------------------------------------------------------- //
// The sphere: the exact per-vertex oracle.
// --------------------------------------------------------------------------- //
TEST(Curvature, SphereHasConstantMeanAndGaussianCurvature) {
    for (double radius : {1.0, 7.0})
        for (CurvatureDualArea mode :
             {CurvatureDualArea::MixedVoronoi, CurvatureDualArea::Barycentric}) {
            CurvatureOptions o;
            o.mDualArea = mode;
            const Mesh m = icosphere(3, radius);
            const CurvatureResult r = compute_curvature(m, o);
            const double* h = array_of(r.mMesh, kCurvatureMeanName);
            const double* k = array_of(r.mMesh, kCurvatureGaussianName);
            const double want_h = 1.0 / radius;
            const double want_k = 1.0 / (radius * radius);
            // Tolerances are MEASURED per mode, not guessed, and they differ by
            // more than an order of magnitude -- which is the whole reason
            // MixedVoronoi is the default. On this fixture the worst relative
            // errors are: MixedVoronoi H = 6.4e-6 and K = 0.0055; Barycentric
            // H = 0.1437 and K = 0.1474. Asserting one loose bound for both
            // would let a real regression in the good mode hide behind the
            // crude mode's error.
            const bool voronoi = mode == CurvatureDualArea::MixedVoronoi;
            const double tol_h = voronoi ? 1e-4 : 0.16;
            const double tol_k = voronoi ? 0.01 : 0.16;
            for (std::size_t i = 0; i < r.mMesh.NumPoints(); ++i) {
                ASSERT_FALSE(std::isnan(h[i])) << "vertex " << i;
                // The SIGN matters: a sphere with outward normals is convex, so
                // H is +1/R. Asserting |H| would let a sign flip through.
                EXPECT_NEAR(h[i], want_h, tol_h * want_h) << "radius " << radius;
                EXPECT_NEAR(k[i], want_k, tol_k * want_k) << "radius " << radius;
            }
        }
}

TEST(Curvature, SphereAccuracyImprovesWithRefinement) {
    auto worst = [](int s) {
        const CurvatureResult r = compute_curvature(icosphere(s, 1.0));
        const double* k = array_of(r.mMesh, kCurvatureGaussianName);
        double e = 0.0;
        for (std::size_t i = 0; i < r.mMesh.NumPoints(); ++i)
            e = std::max(e, std::abs(k[i] - 1.0));
        return e;
    };
    EXPECT_LT(worst(3), worst(1));
}

// --------------------------------------------------------------------------- //
// A developable surface has K == 0 and H == 1/(2R) -- the cleanest separation
// of the two estimators, since K is exactly zero rather than merely different.
// The unit sphere cannot separate them at all (H == K == 1 there); the R = 7
// sphere can, and does (verified: swapping the two outputs fires both). This
// fixture is the one that says WHY they differ rather than merely that they do.
// --------------------------------------------------------------------------- //
TEST(Curvature, CylinderSeparatesMeanFromGaussian) {
    const double radius = 2.0;
    const CurvatureResult r = compute_curvature(open_cylinder(48, 12, radius, 4.0));
    const double* h = array_of(r.mMesh, kCurvatureMeanName);
    const double* k = array_of(r.mMesh, kCurvatureGaussianName);
    int interior = 0;
    for (std::size_t i = 0; i < r.mMesh.NumPoints(); ++i) {
        if (std::isnan(h[i]))
            continue;  // a boundary ring
        ++interior;
        EXPECT_NEAR(k[i], 0.0, 1e-9) << "vertex " << i;
        EXPECT_NEAR(h[i], 1.0 / (2.0 * radius), 0.02) << "vertex " << i;
    }
    EXPECT_GT(interior, 0);
}

TEST(Curvature, PlaneIsFlat) {
    const CurvatureResult r = compute_curvature(plane_grid(6));
    const double* h = array_of(r.mMesh, kCurvatureMeanName);
    const double* k = array_of(r.mMesh, kCurvatureGaussianName);
    int interior = 0;
    for (std::size_t i = 0; i < r.mMesh.NumPoints(); ++i) {
        if (std::isnan(h[i]))
            continue;
        ++interior;
        EXPECT_NEAR(h[i], 0.0, 1e-12);
        EXPECT_NEAR(k[i], 0.0, 1e-12);
    }
    EXPECT_GT(interior, 0);
}

// --------------------------------------------------------------------------- //
// Boundary, isolated and degenerate handling.
// --------------------------------------------------------------------------- //
TEST(Curvature, BoundaryIsNaNUnlessOptedIn) {
    const Mesh m = open_cylinder(24, 6, 1.0, 2.0);
    const CurvatureResult closed_off = compute_curvature(m);
    const double* h = array_of(closed_off.mMesh, kCurvatureMeanName);
    EXPECT_GT(closed_off.mNumBoundary, 0);
    std::size_t nan_count = 0;
    for (std::size_t i = 0; i < closed_off.mMesh.NumPoints(); ++i)
        nan_count += std::isnan(h[i]) ? 1u : 0u;
    EXPECT_EQ(nan_count, static_cast<std::size_t>(closed_off.mNumBoundary));

    CurvatureOptions on;
    on.mIncludeBoundary = true;
    const CurvatureResult opted = compute_curvature(m, on);
    const double* h2 = array_of(opted.mMesh, kCurvatureMeanName);
    for (std::size_t i = 0; i < opted.mMesh.NumPoints(); ++i)
        EXPECT_FALSE(std::isnan(h2[i])) << "vertex " << i;
}

TEST(Curvature, AClosedSurfaceHasNoBoundaryVertices) {
    EXPECT_EQ(compute_curvature(icosphere(1, 1.0)).mNumBoundary, 0);
}

TEST(Curvature, DegenerateTrianglesAreSkippedNotPropagated) {
    // A zero-area triangle bolted onto a sphere must not poison its 1-ring.
    const Mesh clean_mesh = icosphere(2, 1.0);
    const CurvatureResult before = compute_curvature(clean_mesh);

    Mesh m = icosphere(2, 1.0);
    std::vector<std::vector<double>> p;
    for (std::size_t i = 0; i < m.NumPoints(); ++i) {
        const double* row = m.Points().As<double>() + i * 3;
        p.push_back({row[0], row[1], row[2]});
    }
    std::vector<std::vector<std::int64_t>> f;
    const auto cb = *m.CellRange().begin();
    for (std::size_t c = 0; c < cb.NumCells(); ++c) {
        const std::int64_t* row = cb.Conn().As<std::int64_t>() + c * 3;
        f.push_back({row[0], row[1], row[2]});
    }
    f.push_back({0, 0, 1});  // zero area, repeated corner
    const CurvatureResult after = compute_curvature(mt::make_mesh(p, "triangle", f));

    EXPECT_EQ(after.mNumDegenerate, 1);
    const double* a = array_of(before.mMesh, kCurvatureGaussianName);
    const double* b = array_of(after.mMesh, kCurvatureGaussianName);
    for (std::size_t i = 0; i < before.mMesh.NumPoints(); ++i)
        EXPECT_EQ(a[i], b[i]) << "vertex " << i;  // bit-identical, not merely close
}

// --------------------------------------------------------------------------- //
// Options and contracts.
// --------------------------------------------------------------------------- //
TEST(Curvature, PrincipalCurvaturesAgreeWithMeanAndGaussian) {
    CurvatureOptions o;
    o.mRecordPrincipal = true;
    const CurvatureResult r = compute_curvature(icosphere(2, 1.0), o);
    const double* h = array_of(r.mMesh, kCurvatureMeanName);
    const double* k = array_of(r.mMesh, kCurvatureGaussianName);
    const double* pr = array_of(r.mMesh, kCurvaturePrincipalName);
    int clamped = 0;
    for (std::size_t i = 0; i < r.mMesh.NumPoints(); ++i) {
        const double k1 = pr[i * 2];
        const double k2 = pr[i * 2 + 1];
        EXPECT_GE(k1, k2);
        // The mean is exact whether or not the radicand was clamped: the clamp
        // is symmetric, so it moves k1 and k2 by equal and opposite amounts.
        EXPECT_NEAR(0.5 * (k1 + k2), h[i], 1e-12) << "vertex " << i;
        if (h[i] * h[i] >= k[i]) {
            EXPECT_NEAR(k1 * k2, k[i], 1e-9) << "vertex " << i;
        } else {
            // H^2 < K cannot happen on a smooth surface, but it happens all the
            // time on a discrete one -- on this icosphere it is EVERY vertex,
            // because K is over-estimated by ~1.4% while H is nearly exact.
            // The radicand is clamped to zero there, which collapses the two
            // principal curvatures onto H. Documented behaviour, asserted here
            // so a change to the clamp cannot pass unnoticed.
            ++clamped;
            EXPECT_EQ(k1, k2) << "vertex " << i;
            EXPECT_NEAR(k1, h[i], 1e-12) << "vertex " << i;
        }
    }
    EXPECT_GT(clamped, 0) << "the clamp branch was never exercised";
}

TEST(Curvature, OptionalArraysAreOptIn) {
    const CurvatureResult off = compute_curvature(icosphere(1, 1.0));
    EXPECT_FALSE(off.mMesh.HasPointData(kCurvatureAreaName));
    EXPECT_FALSE(off.mMesh.HasPointData(kCurvaturePrincipalName));
    CurvatureOptions o;
    o.mRecordArea = true;
    o.mMean = false;
    const CurvatureResult on = compute_curvature(icosphere(1, 1.0), o);
    EXPECT_TRUE(on.mMesh.HasPointData(kCurvatureAreaName));
    EXPECT_FALSE(on.mMesh.HasPointData(kCurvatureMeanName));
}

TEST(Curvature, TheDualAreasPartitionTheSurface) {
    // Both modes must total the surface area -- the property that makes
    // Gauss-Bonnet independent of the choice.
    for (CurvatureDualArea mode :
         {CurvatureDualArea::MixedVoronoi, CurvatureDualArea::Barycentric}) {
        CurvatureOptions o;
        o.mDualArea = mode;
        o.mRecordArea = true;
        const CurvatureResult r = compute_curvature(icosphere(2, 1.0), o);
        const double* a = array_of(r.mMesh, kCurvatureAreaName);
        double total = 0.0;
        for (std::size_t i = 0; i < r.mMesh.NumPoints(); ++i)
            total += a[i];
        EXPECT_NEAR(total, 4.0 * kPi, 0.05 * 4.0 * kPi);
    }
}

TEST(Curvature, GeometryAndExistingDataSurvive) {
    Mesh m = icosphere(1, 1.0);
    const std::size_t npts = m.NumPoints();
    NDArray tag(DType::Float64, {npts});
    for (std::size_t i = 0; i < npts; ++i)
        tag.As<double>()[i] = static_cast<double>(i);
    m.AddPointData("tag", std::move(tag));

    const CurvatureResult r = compute_curvature(m);
    mt::expect_same_geometry(m, r.mMesh);
    ASSERT_TRUE(r.mMesh.HasPointData("tag"));
    for (std::size_t i = 0; i < npts; ++i)
        EXPECT_EQ(r.mMesh.PointData("tag").As<double>()[i], static_cast<double>(i));
}

TEST(Curvature, RefusesAVolumeMeshByName) {
    try {
        compute_curvature(mt::tet_mesh());
        FAIL() << "expected a throw on a volume mesh";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("extract_surface"), std::string::npos) << e.what();
    }
}
