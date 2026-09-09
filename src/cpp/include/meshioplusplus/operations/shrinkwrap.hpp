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
 * @file operations/shrinkwrap.hpp
 * @brief Project a mesh's points onto a target triangle surface, optionally
 * offset along the surface normal -- the fitting step a scanned skin, a CAD
 * shell or a coarse solve needs before it can be used as a template.
 *
 * Adapted from NVIDIA PhysicsNeMo's `physicsnemo.mesh.shrinkwrap` (2.2):
 * `x' = x + w * (p + offset * n - x)` with `p` the closest point on the
 * target surface, `n` a unit normal there and `w` a per-point weight. It is ONE
 * projection, not an iteration: there is no self-intersection guard and no
 * inversion guard, exactly as upstream, because a wrap is a fit and not a
 * smoothing. Rewritten over meshio++'s own bucket-grid nearest-triangle search
 * (`detail/surface_distance.hpp`, the one `sample_distance` uses), so the point
 * a query is projected to is the point `sample_distance` measures to.
 *
 * **One deliberate divergence from upstream, and the reason for it.** Upstream
 * offsets along the normal of the SELECTED triangle. At an edge or a vertex hit
 * that is the wrong direction: the offset surface of a creased mesh is the
 * rounded one (its Minkowski sum with a ball), whose normal at the crease is
 * the bisector of the two incident faces, and offsetting along one face's
 * normal from the crease lands off that surface by a factor `1/cos(theta/2)` --
 * and, worse, makes the result depend on which of the two equidistant faces
 * won the tie-break. meshio++ offsets along the pseudonormal of the hit
 * FEATURE (face, edge or vertex), the same tables the signed distance's sign
 * already reads, which is a property of the feature and not of the tie-break.
 * A 90-degree "book" target pins the difference.
 *
 * **What moves.** Every point of the source mesh, whatever cells it carries --
 * a volume mesh's interior points are projected too (the source is not
 * required to be a surface; only the TARGET is). Use `mWeights` to select or
 * blend: an integer/bool `point_data` array selects (nonzero moves), a float
 * one blends (`w` is applied unclamped, matching upstream, so a caller can
 * overshoot on purpose). A point farther than `mMaxDistance` from the target
 * is left where it is and counted, as is a point whose hit feature has no
 * direction to offset along (every triangle touching it degenerate).
 *
 * **What survives.** Everything: connectivity, `point_data`, `cell_data`,
 * `field_data`, regions and property sets pass through verbatim, since this is
 * a pure coordinate move. The points array keeps its input dtype.
 *
 * **Determinism.** The nearest-triangle search is totally ordered on
 * `(distance^2, triangle id)`, so the bucket size cannot change the answer
 * (`mGridCellSize` is public precisely so a test can prove it); the update is
 * per-point and elementwise. Byte-identical across backends and thread counts,
 * and -- the pseudonormal's own `acos` aside -- across the C++/numpy boundary.
 */

// System includes
#include <cstdint>
#include <string>

// Project includes
#include "meshioplusplus/export.hpp"
#include "meshioplusplus/mesh.hpp"
#include "meshioplusplus/operations/sdf.hpp"

namespace meshioplusplus {

/// Point data (opt-in): Float64 `(n,)`, each point's distance to the target
/// BEFORE the move; NaN where the point was not queried (weight zero).
inline constexpr const char* kShrinkwrapDistanceName = "shrinkwrap:distance";
/// Point data (opt-in): Int64 `(n,)`, the target cell (block-major global id)
/// each point was projected onto; -1 where not queried.
inline constexpr const char* kShrinkwrapClosestCellName = "shrinkwrap:closest_cell";

/// How `shrinkwrap` projects.
struct ShrinkwrapOptions {
    /// Signed offset along the hit feature's unit pseudonormal. Positive is
    /// the target's outward side when the target is consistently wound.
    double mOffset = 0.0;
    /// A point farther than this from the target is left where it is and
    /// counted in `mNumMissed`. Non-positive means unlimited.
    double mMaxDistance = 0.0;
    /// Name of a `(n,)` `point_data` array on the SOURCE. Float dtype: the
    /// blend factor `w` (unclamped). Integer/bool dtype: a selection, nonzero
    /// moves with `w = 1`. Empty: every point moves with `w = 1`.
    std::string mWeights;
    /// Restrict the TARGET to this named `Cell` region; empty takes all of it.
    std::string mTargetRegion;
    /// How the target's vertex pseudonormals are weighted. `Area` is the mode
    /// the numpy twin reproduces bit for bit under a nonzero offset.
    SdfPseudonormalWeight mNormalWeight = SdfPseudonormalWeight::Angle;
    /// Attach `shrinkwrap:distance`.
    bool mRecordDistance = false;
    /// Attach `shrinkwrap:closest_cell`.
    bool mRecordClosestCell = false;
    /// Bucket size of the nearest-triangle accelerator; 0 derives one from the
    /// target. Public so a test can prove the accelerator is unobservable.
    double mGridCellSize = 0.0;
};

/// What `shrinkwrap` did.
struct ShrinkwrapResult {
    /// The source mesh with its points moved.
    Mesh mMesh;
    /// The TARGET surface's defect counts. A nonzero offset on a target with
    /// inconsistent winding offsets different points to different sides.
    SurfaceQuality mQuality;
    /// Points moved onto the target.
    std::int64_t mNumProjected = 0;
    /// Points queried but left alone: beyond `mMaxDistance`, or no direction
    /// to offset along.
    std::int64_t mNumMissed = 0;
    /// Points never queried (weight zero / unselected).
    std::int64_t mNumSkipped = 0;
    /// The largest `|x' - x|` over every point.
    double mMaxDisplacement = 0.0;
};

/**
 * @brief Project @p rMesh's points onto the surface of @p rTarget.
 *
 * @param rMesh the mesh whose points move (any cell types).
 * @param rTarget the surface to project onto, through `detail::build_triangle_soup`:
 *        quads and polygons are fanned, a volume or higher-order block is
 *        refused by name.
 * @param rOptions see `ShrinkwrapOptions`.
 * @return the moved mesh and the counters.
 * @throws std::invalid_argument on an unusable target, an unknown region or
 *         weights array, or a weights array of the wrong shape.
 */
MESHIOPLUSPLUS_API ShrinkwrapResult shrinkwrap(const Mesh& rMesh, const Mesh& rTarget,
                                               const ShrinkwrapOptions& rOptions = {});

}  // namespace meshioplusplus
