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
 * @file operations/sobolev_deform.hpp
 * @brief Sobolev (Helmholtz-filtered) deformation: smooth a raw per-point
 * displacement field through the mesh's own P1 finite-element operators, then
 * move the points by the smoothed field.
 *
 * Adapted from NVIDIA PhysicsNeMo's `physicsnemo.mesh.sobolev_deform` (2.2):
 * solve, per ambient component,
 *
 *     (M + l^2 K) u = M d,    x' = x + u
 *
 * with `K` the P1 stiffness matrix, `M` a uniform vertex mass and `l` the
 * `mLengthScale` -- a screened-Poisson low-pass filter of the raw field `d`
 * whose cutoff wavelength is `l`, which is what turns a jagged per-node
 * displacement (a shape gradient, a scattered measurement, a model's raw
 * output) into one a mesh can follow without tangling. Implemented from the
 * published description over meshio++'s own machinery; no upstream code is
 * read or vendored.
 *
 * **What is matched exactly, and why.** Three of upstream's choices are
 * reproduced rather than "improved", because they are what make a parity test
 * against the reference a valid oracle and because each is defensible on its
 * own: (1) `K` is assembled dimension-generically from the simplex edge Gram
 * matrix, `K_loc = |c| B G^-1 B^T` with `B = [-1; I]`, so one formula serves a
 * polyline in 2-D, a triangle surface in 3-D and a tetrahedral volume alike;
 * (2) `M` is UNIFORM -- the mean of the positive lumped P1 vertex masses,
 * applied to every vertex -- which makes the filter's response depend on
 * `l` alone rather than on the local element size, and is what upstream
 * needs for the operator to be self-adjoint in plain vertex coordinates;
 * (3) ONE global Jacobi-preconditioned conjugate-gradient solve with one
 * global stopping test `||r|| <= tol * ||b||`, the components decoupling
 * through `K`'s block structure but sharing the convergence history.
 *
 * **Scope.** Every cell block at the mesh's top topological dimension must be
 * a linear simplex -- `line`, `triangle` or `tetra` -- since that is what the
 * Gram-matrix assembly is defined on; a quadratic block is refused naming
 * `linearize`, any other type naming `convert_cells(Simplexify)`. Lower-
 * dimensional blocks ride along untouched (their points are still points).
 * A point in no top-dimensional cell is isolated: it receives its raw
 * displacement, as upstream does, and is counted.
 *
 * **Boundaries.** Nothing is pinned by default: an unfixed boundary carries
 * the natural homogeneous Neumann condition, so a constant displacement is
 * preserved exactly when nothing is fixed. `mFixedPoints` / `mFixedPointsArray`
 * / `mFixBoundary` impose zero-Dirichlet rows instead.
 *
 * **Determinism.** The operator is applied in GATHER form -- each vertex's
 * row is evaluated by one thread from a fixed sequence of incident cells in
 * ascending cell order, no scatter, no atomics -- and every inner product is a
 * fixed-chunk parallel partial sum folded serially, so the iterate is
 * byte-identical across backends and thread counts. There is deliberately NO
 * numpy twin: the stopping test is a branch on a rounded reduction and the
 * iterate depends on how many iterations ran, so a second implementation
 * could stop one iteration early or late and disagree macroscopically.
 *
 * **What survives.** Everything: this is a pure coordinate move, so
 * connectivity, every data array, regions and property sets pass through and
 * the points keep their input dtype. Non-convergence within `mMaxIterations`
 * is a warning plus `mConverged == false`, and the last iterate is returned --
 * a partially smoothed field is still a usable one, and `mResidual` says how
 * far it got.
 */

// System includes
#include <cstdint>
#include <string>
#include <vector>

// Project includes
#include "meshioplusplus/export.hpp"
#include "meshioplusplus/mesh.hpp"

namespace meshioplusplus {

/// Point data (opt-in): Float64 `(n, PointDim)`, the filtered displacement `u`.
inline constexpr const char* kSobolevDisplacementName = "sobolev:displacement";

/// How `sobolev_deform` filters.
struct SobolevOptions {
    /// The `point_data` array holding the raw displacement `d`: `(n, PointDim)`,
    /// or `(n, 3)` on a 2-D mesh (the z column is ignored, with a warning).
    std::string mArrayName;
    /// The smoothing length `l`, in mesh coordinate units. 0 applies `d`
    /// directly at the free points (no solve).
    double mLengthScale = 0.0;
    /// Optional pin mask, empty or `(n,)`: nonzero pins the point (`u = 0`).
    /// The same shape contract as `SmoothOptions::mFrozen`.
    std::vector<std::uint8_t> mFixedPoints;
    /// The flat bindings' route to the same thing: an integer/bool `(n,)`
    /// `point_data` array, nonzero pins. Unioned with `mFixedPoints`.
    std::string mFixedPointsArray;
    /// Also pin every point on a boundary facet of the top-dimensional cells
    /// (a facet used by exactly one cell).
    bool mFixBoundary = false;
    /// Attach `sobolev:displacement`.
    bool mRecordFiltered = false;
    /// Conjugate-gradient iteration cap.
    int mMaxIterations = 128;
    /// Relative residual tolerance: stop once `||r|| <= mTolerance * ||b||`.
    double mTolerance = 1e-10;
};

/// What `sobolev_deform` did.
struct SobolevResult {
    /// The input mesh with its points moved by the filtered displacement.
    Mesh mMesh;
    /// Conjugate-gradient iterations run (0 when `l == 0` or `d == 0`).
    std::int64_t mNumIterations = 0;
    /// The final relative residual `||r|| / ||b||` (0 when no solve ran).
    double mResidual = 0.0;
    /// False when the cap was hit or the iteration broke down; the last
    /// iterate is still returned.
    bool mConverged = true;
    /// Points pinned by any of the three mechanisms.
    std::int64_t mNumFixed = 0;
    /// Points in no top-dimensional cell (they receive `d` verbatim).
    std::int64_t mNumIsolated = 0;
    /// The largest `|u|` over every point.
    double mMaxDisplacement = 0.0;
};

/**
 * @brief Move @p rMesh's points by the Sobolev-filtered version of a raw
 * displacement field.
 *
 * @param rMesh the mesh; see the file comment for the scope.
 * @param rOptions see `SobolevOptions` (`mArrayName` is required).
 * @return the moved mesh and the solve's counters.
 * @throws std::invalid_argument on a missing/mis-shaped displacement array, a
 *         non-simplex top-dimensional block (naming the fix), a degenerate
 *         cell when `l > 0`, or a mis-sized pin mask.
 */
MESHIOPLUSPLUS_API SobolevResult sobolev_deform(const Mesh& rMesh, const SobolevOptions& rOptions);

}  // namespace meshioplusplus
