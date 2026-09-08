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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// Project includes
#include "meshioplusplus/operations/curvature.hpp"
#include "meshioplusplus/detail/data_ops.hpp"
#include "meshioplusplus/detail/geometry.hpp"
#include "meshioplusplus/detail/surface_distance.hpp"
#include "meshioplusplus/log.hpp"
#include "meshioplusplus/parallel.hpp"

namespace meshioplusplus {
namespace {

using detail::Vec3;

constexpr double kCurvTwoPi = 6.283185307179586476925286766559;
constexpr double kCurvPi = 3.141592653589793238462643383279;

/// One triangle corner's angle and the cotangent of it.
///
/// Both come from the same `(cross, dot)` pair, which is what keeps them
/// consistent. The angle is `atan2(|cross|, dot)` rather than `acos` of a
/// clamped ratio: `detail::sd_corner_angle` takes the `acos` route because a
/// pseudonormal weight does not care about the last few digits, but an angle
/// DEFECT is a sum of angles minus `2*pi`, so the digits are exactly what
/// survives -- do not unify the two.
///
/// The cotangent is `dot / |cross|`, trig-free and correctly NEGATIVE at an
/// obtuse corner. Never clamp it: clamping is the usual "fix" and it destroys
/// the operator's linear precision, which is the whole reason to use cotangent
/// weights rather than a uniform graph Laplacian.
struct CurvCorner {
    double mAngle = 0.0;
    double mCotangent = 0.0;
};

CurvCorner curv_corner(const Vec3& rApex, const Vec3& rB, const Vec3& rC) {
    const Vec3 u = detail::vec3_sub(rB, rApex);
    const Vec3 v = detail::vec3_sub(rC, rApex);
    const Vec3 cross = detail::vec3_cross(u, v);
    const double cross_norm = std::sqrt(detail::vec3_norm_sq(cross));
    const double dot = detail::vec3_dot(u, v);
    CurvCorner out;
    out.mAngle = std::atan2(cross_norm, dot);
    out.mCotangent = cross_norm > 0.0 ? dot / cross_norm : 0.0;
    return out;
}

/// Everything one serial pass over the triangles accumulates.
struct CurvAccumulators {
    std::vector<double> mAngleSum;  ///< sum of incident corner angles
    std::vector<Vec3> mLaplacian;   ///< the cotangent Laplacian of the positions
    std::vector<double> mArea;      ///< the dual area
    std::vector<Vec3> mNormal;      ///< area-weighted vertex normal
    std::vector<char> mTouched;     ///< referenced by at least one live triangle
    std::int64_t mNumDegenerate = 0;
};

/// The per-vertex dual-area contributions of one triangle.
///
/// Mixed Voronoi (Meyer et al. 2003): a non-obtuse triangle contributes
/// `(1/8) * (cot(alpha) * |e1|^2 + cot(beta) * |e2|^2)` to each of its corners,
/// where the cotangents are at the two corners OPPOSITE the respective edges.
/// An obtuse triangle has no valid Voronoi region, so it contributes half its
/// area at the obtuse corner and a quarter at each of the others.
void curv_dual_area(const Vec3& rA, const Vec3& rB, const Vec3& rC, const CurvCorner& rCa,
                    const CurvCorner& rCb, const CurvCorner& rCc, double TriArea,
                    CurvatureDualArea Mode, double* pOut) {
    if (Mode == CurvatureDualArea::Barycentric) {
        // Branch-free, hence bit-exactly twinnable in numpy.
        const double third = TriArea / 3.0;
        pOut[0] = third;
        pOut[1] = third;
        pOut[2] = third;
        return;
    }
    const bool obtuse_a = rCa.mAngle > 0.5 * kCurvPi;
    const bool obtuse_b = rCb.mAngle > 0.5 * kCurvPi;
    const bool obtuse_c = rCc.mAngle > 0.5 * kCurvPi;
    if (obtuse_a || obtuse_b || obtuse_c) {
        pOut[0] = obtuse_a ? 0.5 * TriArea : 0.25 * TriArea;
        pOut[1] = obtuse_b ? 0.5 * TriArea : 0.25 * TriArea;
        pOut[2] = obtuse_c ? 0.5 * TriArea : 0.25 * TriArea;
        return;
    }
    const double ab = detail::vec3_norm_sq(detail::vec3_sub(rB, rA));
    const double bc = detail::vec3_norm_sq(detail::vec3_sub(rC, rB));
    const double ca = detail::vec3_norm_sq(detail::vec3_sub(rA, rC));
    // Edge AB is opposite corner C, BC opposite A, CA opposite B.
    pOut[0] = 0.125 * (rCc.mCotangent * ab + rCb.mCotangent * ca);
    pOut[1] = 0.125 * (rCc.mCotangent * ab + rCa.mCotangent * bc);
    pOut[2] = 0.125 * (rCa.mCotangent * bc + rCb.mCotangent * ca);
}

/// The serial accumulation pass.
///
/// Serial deliberately: this is a scatter into shared per-vertex slots, so a
/// `parallel_for` would be both racy and -- once made safe -- order-dependent
/// in the last bits. The finalize pass below is the parallel half, and it is
/// per-vertex and order-independent. The same phase split `surface_distance`
/// and `surface.cpp` already use.
CurvAccumulators curv_accumulate(const detail::TriangleSoup& rSoup, CurvatureDualArea Mode) {
    const std::size_t npts = rSoup.mPoints.size();
    CurvAccumulators acc;
    acc.mAngleSum.assign(npts, 0.0);
    acc.mLaplacian.assign(npts, Vec3{0.0, 0.0, 0.0});
    acc.mArea.assign(npts, 0.0);
    acc.mNormal.assign(npts, Vec3{0.0, 0.0, 0.0});
    acc.mTouched.assign(npts, 0);

    const std::size_t ntri = rSoup.NumTriangles();
    for (std::size_t t = 0; t < ntri; ++t) {
        const std::array<std::int64_t, 3>& v = rSoup.mVertices[t];
        const Vec3& a = rSoup.mCorners[t * 3 + 0];
        const Vec3& b = rSoup.mCorners[t * 3 + 1];
        const Vec3& c = rSoup.mCorners[t * 3 + 2];
        const Vec3 cross = detail::vec3_cross(detail::vec3_sub(b, a), detail::vec3_sub(c, a));
        // Exactly `soup_quality`'s predicate, so the two cannot disagree about
        // which triangles are degenerate.
        if (!(detail::vec3_norm_sq(cross) > 0.0)) {
            ++acc.mNumDegenerate;
            continue;
        }
        const double tri_area = 0.5 * std::sqrt(detail::vec3_norm_sq(cross));

        const CurvCorner ca = curv_corner(a, b, c);
        const CurvCorner cb = curv_corner(b, c, a);
        const CurvCorner cc = curv_corner(c, a, b);

        double dual[3] = {0.0, 0.0, 0.0};
        curv_dual_area(a, b, c, ca, cb, cc, tri_area, Mode, dual);

        const Vec3* corner[3] = {&a, &b, &c};
        const CurvCorner* angle[3] = {&ca, &cb, &cc};
        for (std::size_t i = 0; i < 3; ++i) {
            const std::size_t vi = static_cast<std::size_t>(v[i]);
            acc.mTouched[vi] = 1;
            acc.mAngleSum[vi] += angle[i]->mAngle;
            acc.mArea[vi] += dual[i];
            acc.mNormal[vi] = detail::vec3_add(acc.mNormal[vi], cross);  // area-weighted
        }
        // The cotangent Laplacian of the positions. Edge (i, j)'s weight is the
        // cotangent at the corner OPPOSITE it; each triangle contributes one
        // half of each of its three edges' weights, and the two triangles
        // sharing an edge complete the usual (cot a + cot b) / 2.
        for (std::size_t e = 0; e < 3; ++e) {
            const std::size_t i = e;
            const std::size_t j = (e + 1) % 3;
            const std::size_t k = (e + 2) % 3;  // the opposite corner
            const double w = 0.5 * angle[k]->mCotangent;
            const Vec3 d = detail::vec3_sub(*corner[i], *corner[j]);
            const std::size_t vi = static_cast<std::size_t>(v[i]);
            const std::size_t vj = static_cast<std::size_t>(v[j]);
            acc.mLaplacian[vi] = detail::vec3_add(acc.mLaplacian[vi], detail::vec3_scale(d, w));
            acc.mLaplacian[vj] = detail::vec3_add(acc.mLaplacian[vj], detail::vec3_scale(d, -w));
        }
    }
    return acc;
}

/// Which vertices sit on a boundary, from the SAME edge map `soup_quality` and
/// `repair` read -- so the three cannot disagree about what a boundary is.
std::vector<char> curv_boundary_vertices(const detail::TriangleSoup& rSoup,
                                         const detail::SurfaceEdgeMap& rEdges) {
    std::vector<char> is_boundary(rSoup.mPoints.size(), 0);
    for (const auto& kv : rEdges) {
        if (kv.second.mUsed != 1)
            continue;
        // A self-loop can only come from a triangle with a repeated corner,
        // which is degenerate and was skipped by the accumulation pass. Letting
        // it mark a boundary would NaN out a perfectly good vertex because of a
        // triangle that contributed nothing -- and `soup_quality` counts it as
        // a boundary edge (correctly, for its own purpose), so the shared map
        // carries it and this is where it has to be filtered.
        if (kv.first[0] == kv.first[1])
            continue;
        is_boundary[static_cast<std::size_t>(kv.first[0])] = 1;
        is_boundary[static_cast<std::size_t>(kv.first[1])] = 1;
    }
    return is_boundary;
}

}  // namespace

CurvatureDualArea curvature_dual_area_from_name(const std::string& rName) {
    if (rName == "mixed-voronoi")
        return CurvatureDualArea::MixedVoronoi;
    if (rName == "barycentric")
        return CurvatureDualArea::Barycentric;
    throw std::invalid_argument("meshio++: curvature: unknown dual area '" + rName +
                                "' (expected 'mixed-voronoi' or 'barycentric')");
}

const char* curvature_dual_area_name(CurvatureDualArea Mode) {
    switch (Mode) {
        case CurvatureDualArea::MixedVoronoi:
            return "mixed-voronoi";
        case CurvatureDualArea::Barycentric:
            return "barycentric";
    }
    return "mixed-voronoi";
}

CurvatureResult compute_curvature(const Mesh& rMesh, const CurvatureOptions& rOptions) {
    const detail::TriangleSoup soup = detail::build_triangle_soup(rMesh, rOptions.mRegion);
    const detail::SurfaceEdgeMap edges = detail::build_surface_edges(soup);

    CurvatureResult out;
    out.mQuality = detail::soup_quality(soup);
    out.mMesh = detail::clone_mesh(
        rMesh, [](DataLocation, const std::string&, std::string&) { return true; });

    const std::size_t npts = soup.mPoints.size();
    const CurvAccumulators acc = curv_accumulate(soup, rOptions.mDualArea);
    const std::vector<char> is_boundary = curv_boundary_vertices(soup, edges);
    out.mNumDegenerate = acc.mNumDegenerate;

    const double nan = std::numeric_limits<double>::quiet_NaN();
    NDArray mean(DType::Float64, {npts});
    NDArray gauss(DType::Float64, {npts});
    NDArray area(DType::Float64, {npts});
    NDArray principal(DType::Float64, {npts, 2});
    double* p_mean = mean.As<double>();
    double* p_gauss = gauss.As<double>();
    double* p_area = area.As<double>();
    double* p_principal = principal.As<double>();

    // Per-vertex and order-independent: the parallel half of the phase split.
    parallel_for(npts, [&](std::size_t i) {
        const double a = acc.mArea[i];
        p_area[i] = acc.mTouched[i] ? a : nan;
        if (!acc.mTouched[i] || !(a > 0.0) || (is_boundary[i] && !rOptions.mIncludeBoundary)) {
            p_mean[i] = nan;
            p_gauss[i] = nan;
            p_principal[i * 2] = nan;
            p_principal[i * 2 + 1] = nan;
            return;
        }
        // The geodesic form on a boundary vertex, the closed one otherwise.
        const double turn = is_boundary[i] ? kCurvPi : kCurvTwoPi;
        const double k = (turn - acc.mAngleSum[i]) / a;
        // ||L p|| / 2A is |H|; the sign comes from whether the mean-curvature
        // normal points along the surface normal or against it. That is what
        // makes H orientation-dependent and K not.
        const Vec3& lp = acc.mLaplacian[i];
        const double h_mag = 0.5 * std::sqrt(detail::vec3_norm_sq(lp)) / a;
        // `mLaplacian` accumulates sum(w * (p_i - p_j)), which is the NEGATIVE
        // of the usual Laplacian sum(w * (p_j - p_i)), so the mean-curvature
        // normal is -lp. On a convex outward-oriented surface that points along
        // the vertex normal, and H must come out positive there.
        const double sign = detail::vec3_dot(lp, acc.mNormal[i]) > 0.0 ? 1.0 : -1.0;
        const double h = h_mag * sign;
        p_mean[i] = h;
        p_gauss[i] = k;
        // k1, k2 = H +- sqrt(H^2 - K). The radicand is negative only through
        // discretization error, so clamp rather than produce NaN.
        const double disc = h * h - k;
        const double root = std::sqrt(disc > 0.0 ? disc : 0.0);
        p_principal[i * 2] = h + root;
        p_principal[i * 2 + 1] = h - root;
    });

    for (std::size_t i = 0; i < npts; ++i) {
        if (!acc.mTouched[i])
            ++out.mNumIsolated;
        else if (is_boundary[i])
            ++out.mNumBoundary;
        out.mTotalAngleDefect += acc.mTouched[i] ? (kCurvTwoPi - acc.mAngleSum[i]) : 0.0;
    }

    if (rOptions.mMean)
        out.mMesh.AddPointData(kCurvatureMeanName, std::move(mean));
    if (rOptions.mGaussian)
        out.mMesh.AddPointData(kCurvatureGaussianName, std::move(gauss));
    if (rOptions.mRecordArea)
        out.mMesh.AddPointData(kCurvatureAreaName, std::move(area));
    if (rOptions.mRecordPrincipal)
        out.mMesh.AddPointData(kCurvaturePrincipalName, std::move(principal));

    if (out.mQuality.mInconsistentPairs != 0)
        log::warn(
            "curvature: {} edge pair(s) wind the same way, so the sign of "
            "'{}' is not trustworthy; run repair(mesh, {{.mFixOrientation = true}}) first",
            out.mQuality.mInconsistentPairs, kCurvatureMeanName);
    return out;
}

}  // namespace meshioplusplus
