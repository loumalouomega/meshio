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
 * @file operations/repair.hpp
 * @brief Surface repair beyond `clean`: consistent orientation, hole filling
 * and non-manifold (pinched-vertex) splitting -- the three defects
 * `SurfaceQuality` counts and `clean` does not touch.
 *
 * Adapted from NVIDIA PhysicsNeMo's `physicsnemo.mesh.repair` (2.2:
 * `fix_orientation`, `fill_holes`, and the generator's private
 * `split_pinched_vertices`); algorithms only, rewritten over meshio++'s own
 * machinery, no upstream code read or vendored. `clean` remains the place for
 * welding, degenerate/duplicate removal and orphan pruning -- `repair`
 * composes it (`mWeldTolerance`) rather than duplicating it.
 *
 * **Two deliberate divergences from upstream, and why.** (1) Orientation is
 * propagated by the TOPOLOGICAL half-edge rule -- two triangles sharing an
 * edge agree iff they traverse it in opposite directions -- which is exact,
 * where upstream's `dot(n_child, n_parent) < 0` test is a local-flatness
 * approximation that mis-orients across any crease sharper than 90 degrees
 * (a strip folded at 150 degrees pins the difference). It is the rule
 * `detail::orient_rings` already applies to one polyhedron's faces, made
 * tolerant of boundaries, non-manifold edges and several components. (2) A
 * hole's fan is wound to AGREE with the surrounding surface (against the one
 * triangle on each boundary edge), where upstream winds it from the loop's
 * own traversal direction and so may leave the fill inconsistent with its
 * neighbours -- which `mQualityAfter.mInconsistentPairs` would then report.
 *
 * **What it does, in order.** Weld (opt-in) -> triangulate (quads and polygons
 * fan exactly as `convert_cells(Simplexify)` does, blocks staying 1:1) ->
 * split bowties (a vertex whose triangle star is edge-disconnected is
 * duplicated once per extra component, geometry unchanged) -> orient (BFS per
 * connected component over manifold edges, fewest flips wins ties) -> fill
 * holes (every traceable boundary loop of at most `mMaxHoleEdges` edges gets
 * one centroid point and one triangle per loop edge) -> orient outward
 * (every CLOSED component whose divergence-theorem volume is negative is
 * flipped whole). Split first so a pinched boundary vertex traces as two
 * loops rather than one aborted figure-eight; orient before fill so the fill
 * inherits a consistent neighbourhood; outward after fill so a sphere with a
 * hole is closed when its volume is taken.
 *
 * **What it does not do.** Non-manifold EDGES (used by three or more
 * triangles) are neither split nor crossed by the orientation BFS; they are
 * counted in `mQualityAfter`. Nested cavities are not detected: every closed
 * component is oriented outward on its own. A boundary loop through a vertex
 * of boundary degree other than two is left open and counted as skipped.
 *
 * **Output shape.** All-triangle at the surface, blocks 1:1 with the input
 * (a quad block becomes a triangle block of twice the rows), lower-dimensional
 * blocks (boundary `line`s) carried verbatim, plus ONE trailing `triangle`
 * block holding every fill triangle -- only when there is one, so a closed
 * input keeps its block count. Points: the originals, then the split copies,
 * then the hole centroids. `cell_data` rows follow their cell (the trailing
 * block gets NaN for float and 0 for integer arrays); `point_data` copies
 * inherit their source row and centroids the mean of their loop's rows,
 * dtype preserved. Point and Cell regions survive (a copy joins its source's
 * regions); Side regions are dropped by name, since a flip permutes a
 * triangle's edge numbering. There is deliberately no numpy twin: the
 * outward test is a branch on the sign of a rounded volume.
 */

// System includes
#include <cstdint>
#include <vector>

// Project includes
#include "meshioplusplus/export.hpp"
#include "meshioplusplus/mesh.hpp"
#include "meshioplusplus/ndarray.hpp"
#include "meshioplusplus/operations/sdf.hpp"

namespace meshioplusplus {

/// Point data (opt-in): Int64 `(n_out,)`, the INPUT point each output point
/// came from -- itself for an original, its source for a split copy, -1 for
/// a hole centroid.
inline constexpr const char* kRepairParentPointName = "repair:parent_point";
/// Cell data (opt-in): Int64 per block, -1 for every input triangle and the
/// hole's ordinal (0-based, in detection order) for each fill triangle.
inline constexpr const char* kRepairHoleName = "repair:hole";

/// What `repair` should do.
struct RepairOptions {
    /// Rewind triangles so neighbours across every manifold edge agree.
    bool mFixOrientation = true;
    /// After orientation and filling, flip every CLOSED component whose
    /// signed volume is negative. Ignored when `mFixOrientation` is off.
    bool mOrientOutward = true;
    /// Fan-fill boundary loops of at most `mMaxHoleEdges` edges.
    bool mFillHoles = true;
    /// Duplicate a vertex whose triangle star is edge-disconnected.
    bool mSplitNonManifold = true;
    /// Attach `repair:parent_point` and `repair:hole`.
    bool mRecordProvenance = false;
    /// Longest boundary loop that is still filled; longer ones are counted as
    /// skipped and left open. Non-positive means no limit.
    std::int64_t mMaxHoleEdges = 10;
    /// Weld coincident points within this distance FIRST (through `clean`),
    /// so two stacked-but-distinct points stop reading as a boundary. 0 = off.
    double mWeldTolerance = 0.0;
};

/// What `repair` did.
struct RepairResult {
    Mesh mMesh;
    /// Per input block, Int64 `(cells_in_block,)`: input cell -> first output
    /// cell (the `FirstChild` shape, identity on an all-triangle input). The
    /// trailing hole-fill block has no input counterpart and no entry.
    std::vector<NDArray> mCellMaps;
    /// Int64 `(points_in,)`, input point -> output point. The identity unless
    /// `mWeldTolerance > 0`; copies and centroids are appended, never mapped to.
    NDArray mPointMap;
    /// The (welded, triangulated) input's defect counts.
    SurfaceQuality mQualityBefore;
    /// The output's defect counts.
    SurfaceQuality mQualityAfter;
    std::int64_t mNumFlipped = 0;          ///< Input triangles rewound.
    std::int64_t mNumComponents = 0;       ///< Edge-connected components.
    std::int64_t mLargestComponent = 0;    ///< Triangles in the largest one.
    std::int64_t mNumOrientedOutward = 0;  ///< Closed components flipped whole.
    std::int64_t mNumUnorientable = 0;     ///< Components with a parity conflict.
    std::int64_t mNumVerticesSplit = 0;    ///< Bowtie vertices duplicated.
    std::int64_t mNumHolesDetected = 0;    ///< Boundary loops found.
    std::int64_t mNumHolesFilled = 0;
    std::int64_t mNumHolesSkipped = 0;  ///< Too long, or not traceable.
    std::int64_t mNumFacesAdded = 0;    ///< Fill triangles.
    std::int64_t mNumPointsAdded = 0;   ///< Copies plus centroids.
    std::int64_t mPointsWelded = 0;     ///< From the optional weld.
};

/**
 * @brief Repair a surface mesh's orientation, holes and pinched vertices.
 *
 * @param rMesh a surface mesh: triangle/quad/polygon blocks, optionally with
 *        lower-dimensional blocks that ride along. A 3-D or polyhedron block
 *        is refused naming `extract_surface`, a higher-order surface block
 *        naming `linearize`.
 * @param rOptions see `RepairOptions`.
 * @return the repaired mesh, the maps and the counters.
 * @throws std::invalid_argument on an out-of-scope input.
 */
MESHIOPLUSPLUS_API RepairResult repair(const Mesh& rMesh, const RepairOptions& rOptions = {});

}  // namespace meshioplusplus
