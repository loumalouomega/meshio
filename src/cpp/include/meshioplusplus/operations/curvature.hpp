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
 * @file operations/curvature.hpp
 * @brief Per-vertex mean and Gaussian curvature of a surface, by the standard
 * discrete estimators: the angle defect for `K` and the cotangent
 * Laplace-Beltrami operator for `H`.
 *
 * These are the estimators the discrete-differential-geometry literature's
 * convergence results are about, and the ones NVIDIA PhysicsNeMo's own
 * `gaussian_curvature_vertices`/`mean_curvature_vertices` use -- so the numbers
 * here are comparable with a model's. Implemented from the published
 * definitions; no upstream code is read or vendored.
 *
 * **Deliberately NOT `remesh`'s estimator.** `remesh.cpp` fits an osculating
 * paraboloid over each 1-ring to drive `RemeshOptions::mGradation` and the
 * anisotropic metric. That is a legitimate estimator, but it returns only the
 * larger-magnitude principal curvature as a magnitude, it runs on remesh's own
 * subdivided working copy rather than the caller's mesh, and it has no
 * structural invariant to test against. It is left exactly as it is: this
 * operation adds a second, independent estimator rather than changing one two
 * other features depend on.
 *
 * **The oracle that makes this testable is Gauss-Bonnet.** For a closed surface
 * the angle defects sum to `2*pi*chi` -- `4*pi` for a sphere, whatever the
 * tessellation and whichever dual area is chosen. `CurvatureResult` reports that
 * sum (`mTotalAngleDefect`) precisely so the invariant is observable from every
 * binding rather than only from a gtest. Note it is `2*pi*chi` only for a
 * **closed** surface; on an open one it is that minus the boundary's turning.
 *
 * **`H` is orientation-dependent** and `K` is not. The mean curvature's sign
 * comes from the surface's own winding, so a mesh whose facets disagree about
 * which side is out yields sign-flipped patches with no error raised. That is
 * why the result carries the input's `SurfaceQuality`: check
 * `mQuality.mInconsistentPairs` before trusting a sign, and run
 * `repair(mesh, {.mFixOrientation = true})` if it is non-zero. This operation
 * never silently repairs its input.
 */

// System includes
#include <cstdint>
#include <string>

// Project includes
#include "meshioplusplus/export.hpp"
#include "meshioplusplus/mesh.hpp"
#include "meshioplusplus/operations/sdf.hpp"

namespace meshioplusplus {

/// Point data: the mean curvature `H`, one Float64 per point.
inline constexpr const char* kCurvatureMeanName = "curvature:mean";
/// Point data: the Gaussian curvature `K`, one Float64 per point.
inline constexpr const char* kCurvatureGaussianName = "curvature:gaussian";
/// Point data: the dual (vertex) area each curvature was divided by. Opt-in.
inline constexpr const char* kCurvatureAreaName = "curvature:area";
/// Point data: the two principal curvatures as `(n, 2)`, `k1 >= k2`. Opt-in.
inline constexpr const char* kCurvaturePrincipalName = "curvature:principal";

/**
 * @brief Which dual area a per-vertex curvature is divided by.
 *
 * Both partition the surface exactly, so `mTotalAngleDefect` -- and therefore
 * the Gauss-Bonnet invariant -- is identical under either.
 */
enum class CurvatureDualArea : std::uint8_t {
    /// Meyer et al.'s mixed Voronoi area: the Voronoi cell where the triangle is
    /// non-obtuse, and a bisected area where it is. Reuses the cotangents the
    /// mean-curvature pass already computes, so it is nearly free, and it
    /// converges better on an irregular tessellation. The default.
    MixedVoronoi = 0,
    /// A third of each incident triangle's area. Cruder, but **branch-free**,
    /// which is what makes it bit-exactly reproducible by the numpy twin --
    /// the same second-mode-for-twinnability argument `SdfPseudonormalWeight`
    /// already makes.
    Barycentric = 1,
};

/**
 * @brief Parses a dual-area name.
 * @param rName One of `"mixed-voronoi"`, `"barycentric"` (case-sensitive, as
 *        elsewhere in the operations layer).
 * @return The matching enumerator.
 * @throws std::invalid_argument if the name is not recognised.
 */
MESHIOPLUSPLUS_API CurvatureDualArea curvature_dual_area_from_name(const std::string& rName);

/// The spelling `curvature_dual_area_from_name` accepts for `Mode`, so the flat
/// bindings, both CLIs and the pipeline report a name a caller can pass back.
MESHIOPLUSPLUS_API const char* curvature_dual_area_name(CurvatureDualArea Mode);

/// What `compute_curvature` should compute and attach.
struct CurvatureOptions {
    /// Attach `curvature:mean`.
    bool mMean = true;
    /// Attach `curvature:gaussian`.
    bool mGaussian = true;
    /// Which dual area to divide by.
    CurvatureDualArea mDualArea = CurvatureDualArea::MixedVoronoi;
    /// Compute a value at boundary vertices instead of leaving them NaN.
    ///
    /// Off by default, matching upstream: a boundary vertex has no closed
    /// 1-ring, so both estimators are biased there and the honest answer is
    /// "not defined". On, `K` uses the geodesic form `pi - sum(theta)` and `H`
    /// the raw one-sided operator; both are biased and documented as such.
    /// Isolated vertices are NaN either way -- there is nothing to average.
    bool mIncludeBoundary = false;
    /// Also attach `curvature:area`.
    bool mRecordArea = false;
    /// Also attach `curvature:principal`, the `(n, 2)` pair `k1 >= k2` recovered
    /// as `H +- sqrt(H^2 - K)`. Free: no new machinery, just the two outputs.
    bool mRecordPrincipal = false;
    /// Restrict to this named `Cell` region; empty takes every surface cell.
    std::string mRegion;
};

/// What `compute_curvature` computed, and what it found on the way.
struct CurvatureResult {
    /// The input mesh with the requested arrays attached.
    Mesh mMesh;
    /// The INPUT surface's defect counts. `mInconsistentPairs != 0` means the
    /// sign of `H` is not trustworthy; see this header's file comment.
    SurfaceQuality mQuality;
    /// Vertices left NaN because they sit on a boundary.
    std::int64_t mNumBoundary = 0;
    /// Vertices left NaN because no surviving triangle references them.
    std::int64_t mNumIsolated = 0;
    /// Triangles skipped for zero area, by the same predicate `soup_quality`
    /// uses -- so the two agree about what "degenerate" means.
    std::int64_t mNumDegenerate = 0;
    /// The sum of every vertex's angle defect, boundary vertices included.
    ///
    /// For a CLOSED surface this is `2*pi*chi` exactly -- `4*pi` for a sphere --
    /// whatever the tessellation and whichever `CurvatureDualArea` was chosen.
    /// On an open surface it is that minus the boundary's total turning, which
    /// is a different (still meaningful) quantity.
    double mTotalAngleDefect = 0.0;
};

/**
 * @brief Per-vertex mean and Gaussian curvature of a surface mesh.
 *
 * Triangles come from `detail::build_triangle_soup`, which fans quads and
 * rectangular polygons on the same diagonal `convert_cells(Simplexify)` uses,
 * refuses a 3-D or polyhedron block by name pointing at `extract_surface`, and
 * refuses a higher-order block pointing at `linearize`. A quad mesh's curvature
 * is therefore the curvature of its canonical triangulation, not of the quad
 * surface itself.
 *
 * @param rMesh a surface mesh.
 * @param rOptions what to compute; see `CurvatureOptions`.
 * @return the mesh with the requested `point_data` attached, plus the counters.
 * @throws std::invalid_argument on a non-surface input (naming the fix) or an
 *         unknown region name.
 */
MESHIOPLUSPLUS_API CurvatureResult compute_curvature(const Mesh& rMesh,
                                                     const CurvatureOptions& rOptions = {});

}  // namespace meshioplusplus
