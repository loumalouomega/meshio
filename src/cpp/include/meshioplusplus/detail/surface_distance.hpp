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
 * @file detail/surface_distance.hpp
 * @brief The signed-distance kernel: a triangle soup, its pseudonormal tables,
 * a bucket-grid accelerator, and the nearest-point query over them.
 *
 * `operations/sdf.cpp` is the driver; everything geometric lives here so that it
 * can be tested against closed-form distance fields with no grid, no mesh
 * generation and no binding in the way.
 *
 * ### The accelerator is deliberately not a BVH
 *
 * `detail/spatial_hash.hpp` already provides `InsertBox` (which exists precisely
 * to register a simplex's quantized bounding box) and `ForEachInShell` with a
 * documented stopping rule, and `interpolate.cpp` already runs an expanding-shell
 * nearest search over it. The workload here -- many queries against a
 * roughly uniform triangle soup -- is the uniform grid's best case and the BVH's
 * worst.
 *
 * The decisive reason, though, is not speed: **a BVH would make the accelerator
 * observable in the output.** Its build order and split heuristic determine the
 * order candidates are visited in, which determines which of two equidistant
 * triangles wins. The grid can be made provably unobservable instead, by giving
 * every comparison a total order -- `(distance squared, triangle id)` -- so that
 * the answer cannot depend on visit order at all. `SurfaceDistanceOptions`
 * exposes `mGridCellSize` mainly so a test can *prove* that: the same query at
 * several bucket sizes must return byte-identical distances and nearest cells.
 *
 * ### The sign
 *
 * `closest_point_on_triangle` reports which feature of the triangle the nearest
 * point lies on, and that selects the normal:
 *
 *  - face   -> the triangle's own normal;
 *  - edge   -> the sum of the two incident triangles' unit normals;
 *  - vertex -> the angle-weighted sum of the incident triangles' unit normals.
 *
 * Using the nearest *triangle's* normal in all three cases is the classic bug:
 * exactly right on convex geometry, exactly wrong on the concave side of every
 * crease, and it renders beautifully either way.
 *
 * Building those tables is the one place with a floating-point ordering hazard,
 * because summing unit normals in a non-deterministic order changes the last
 * bits and a last-bit change can flip a sign at a near-tangent query point. The
 * tables are therefore accumulated by a **serial** pass in ascending
 * (triangle, corner) order -- the phase-split idiom `surface.cpp` uses.
 */

// System includes
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// Project includes
#include "meshioplusplus/export.hpp"
#include "meshioplusplus/mesh.hpp"
#include "meshioplusplus/detail/geometry.hpp"
#include "meshioplusplus/detail/point_triangle.hpp"
#include "meshioplusplus/detail/spatial_hash.hpp"
#include "meshioplusplus/operations/sdf.hpp"

namespace meshioplusplus {
namespace detail {

/**
 * @brief A surface reduced to triangles, with each triangle remembering which
 * input cell it came from.
 *
 * The provenance matters because `sdf:closest_cell` must name a cell of the
 * mesh the caller passed in, not of a triangulation they never saw.
 */
struct TriangleSoup {
    /// Corner coordinates, three per triangle, in `(a, b, c)` order.
    std::vector<Vec3> mCorners;
    /// Per triangle, the global (block-major) index of the input cell it came from.
    std::vector<std::int64_t> mSourceCell;
    /// Per triangle, the three vertex ids -- the INPUT MESH's own point ids.
    /// `build_triangle_soup` does not weld: it copies every mesh point verbatim,
    /// so `mPoints[i]` is mesh point `i`, index for index, orphans included.
    /// Two coincident-but-distinct points therefore read as two vertices here,
    /// which is why an edge between them counts as a boundary edge and why
    /// `repair` offers `mWeldTolerance` to merge points before it looks at
    /// orientation or holes.
    std::vector<std::array<std::int64_t, 3>> mVertices;
    /// Every input point, in the input's own order (see `mVertices`).
    std::vector<Vec3> mPoints;

    std::size_t NumTriangles() const { return mSourceCell.size(); }
};

/**
 * @brief Reduce a surface mesh to a triangle soup.
 * @param rSurface the mesh; `triangle` blocks are taken as they are, `quad` and
 *        rectangular `polygon` blocks are fanned exactly as
 *        `convert_cells(Simplexify)` fans them so the two cannot disagree.
 * @param rRegion restrict to this named `Cell` region; empty takes everything.
 * @throws std::invalid_argument on a 3-D or polyhedron block (pointing at
 *         `extract_surface`), on a higher-order block (pointing at `linearize`),
 *         and on an unknown region name.
 */
MESHIOPLUSPLUS_API TriangleSoup build_triangle_soup(const Mesh& rSurface,
                                                    const std::string& rRegion);

/// An undirected edge, as the sorted pair of its endpoints' vertex ids.
using SurfaceEdgeKey = std::array<std::int64_t, 2>;

/// Hash for `SurfaceEdgeKey`, the `GridKeyHash` mixing constant and shape.
struct SurfaceEdgeKeyHash {
    std::size_t operator()(const SurfaceEdgeKey& rKey) const {
        std::size_t h = 0;
        for (std::int64_t v : rKey)
            h ^= std::hash<std::int64_t>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

/**
 * @brief What one undirected edge of a soup is used by.
 *
 * `mUsed` is how many triangles reference the edge and `mForward` how many
 * traverse it low->high. A consistently wound closed surface has `mUsed == 2`
 * and `mForward == 1` on every edge: the two triangles walk their shared edge
 * in opposite directions, which is exactly what "they agree about which side
 * is out" means. `mUsed == 1` is a boundary edge, `mUsed > 2` non-manifold,
 * and `mUsed == 2 && mForward != 1` a wound-the-same-way pair.
 *
 * `mFirstTriangle` is the lowest-indexed triangle using the edge, which is
 * what lets a caller walk the face dual without building a second incidence
 * structure.
 */
struct SurfaceEdgeRecord {
    std::int64_t mUsed = 0;
    std::int64_t mForward = 0;
    std::int64_t mFirstTriangle = -1;
};

/// Every undirected edge of a soup, keyed by its sorted endpoint pair.
using SurfaceEdgeMap = std::unordered_map<SurfaceEdgeKey, SurfaceEdgeRecord, SurfaceEdgeKeyHash>;

/**
 * @brief The per-edge use record of a soup.
 *
 * `soup_quality` is a fold over this, and it is what a reorientation BFS walks
 * and what a boundary-loop walk starts from -- so the three cannot disagree
 * about what a boundary edge is. Hoisted out of `soup_quality`'s body, which
 * built exactly this map and then discarded it.
 */
MESHIOPLUSPLUS_API SurfaceEdgeMap build_surface_edges(const TriangleSoup& rSoup);

/// The four edge defect counts of a soup, and the resulting verdict.
MESHIOPLUSPLUS_API SurfaceQuality soup_quality(const TriangleSoup& rSoup);

/**
 * @brief A soup prepared for querying: the accelerator plus the normal tables.
 *
 * Held by value and moved; building one is O(triangles) and is paid once per
 * call rather than once per query, which is why the public API is a batch one.
 */
struct DistanceQuery {
    const TriangleSoup* mpSoup = nullptr;
    /// The bucket grid, holding triangle ids by quantized bounding box.
    SpatialGrid mGrid{1.0};
    /// Per triangle, its unnormalized normal (`cross(ab, ac)`).
    std::vector<Vec3> mFaceNormal;
    /// Per welded vertex, the weighted sum of incident unit face normals.
    std::vector<Vec3> mVertexNormal;
    /// Per edge (sorted endpoint ids), the sum of incident unit face normals.
    std::unordered_map<SurfaceEdgeKey, Vec3, SurfaceEdgeKeyHash> mEdgeNormal;
    /// The bucket size actually used, after the auto rule or the caller's override.
    double mCellSize = 1.0;
};

/// Build the accelerator and normal tables for @p rSoup.
MESHIOPLUSPLUS_API DistanceQuery build_distance_query(const TriangleSoup& rSoup,
                                                      const SurfaceDistanceOptions& rOptions);

/// What a single query point resolved to.
struct DistanceHit {
    double mSignedDistance = 0.0;   ///< Negative inside, per the usual convention.
    std::int64_t mSourceCell = -1;  ///< Nearest input cell, or -1 outside the band.
    bool mInBand = true;            ///< False when the value was clamped to the band.
};

/**
 * @brief Distances from @p rPoints to the soup @p rQuery was built from.
 *
 * Parallel over query points, which are independent; each point's own search is
 * serial and totally ordered, so the result does not depend on thread count.
 */
MESHIOPLUSPLUS_API std::vector<DistanceHit> query_distances(const DistanceQuery& rQuery,
                                                            const std::vector<Vec3>& rPoints,
                                                            const SurfaceDistanceOptions& rOptions);

/// What `query_closest_points` resolved a single query point to. A new,
/// additive type (Tier C) rather than a field added to `DistanceHit` -- see
/// that function's own doc comment for why.
struct ClosestPointHit {
    Vec3 mPoint{0.0, 0.0, 0.0};     ///< The nearest point on the soup.
    double mDistance = 0.0;         ///< Its distance from the query (unsigned).
    std::int64_t mSourceCell = -1;  ///< The input cell it came from, or -1 (empty soup only).
    bool mFound = false;            ///< False only when the soup has no triangles at all.
};

/**
 * @brief The actual nearest POINT on the soup to each of @p rPoints, not just
 * its distance.
 *
 * `query_distances`'s own `PointTriangleHit` already computes this internally
 * (`best_hit.mPoint`) and then discards it, keeping only the distance and the
 * sign -- which is all `sample_distance`/`distance_to_surface` ever needed.
 * `remesh_volume`'s warp step is the first caller that needs the point itself
 * (to move a lattice vertex onto the surface, not merely learn how far it
 * is), hence this sibling function rather than growing `DistanceHit`, whose
 * layout is an ABI boundary (`doc/abi.md`) this header already participates
 * in. Both functions share the identical bucket-grid nearest-triangle search
 * (`sd_nearest_triangle`, private to `surface_distance.cpp`) so they cannot
 * disagree about which triangle is nearest.
 *
 * No banding and no sign here -- a warp threshold is the caller's own
 * decision (`remesh_volume.cpp`'s own threshold, compared against
 * `mDistance` after the fact), and a warp target has no notion of inside or
 * outside to report.
 *
 * @param rQuery the accelerator built by `build_distance_query`.
 * @param rPoints the query points.
 * @return one hit per query point, `mFound == false` only when the soup this
 *         query was built from has no triangles (impossible in practice,
 *         since `build_distance_query` itself refuses an empty soup, but the
 *         field exists so a caller need not assume).
 */
MESHIOPLUSPLUS_API std::vector<ClosestPointHit> query_closest_points(
    const DistanceQuery& rQuery, const std::vector<Vec3>& rPoints);

/// What `query_surface_projections` resolved a single query point to. A
/// second additive sibling (Tier C), for the same reason `ClosestPointHit` is
/// one: `ClosestPointHit`'s layout is installed-header state, and `shrinkwrap`
/// needs three things it does not carry -- the soup-local triangle, the
/// feature the hit landed on, and that feature's pseudonormal.
struct SurfaceProjection {
    Vec3 mPoint{0.0, 0.0, 0.0};  ///< The nearest point on the soup.
    /// The UNNORMALIZED pseudonormal of the hit FEATURE, read straight from
    /// the `DistanceQuery` tables: the face's cross product for a face hit, the
    /// sum of the two incident unit normals for an edge hit, the weighted sum
    /// of the incident unit normals for a vertex hit. Zero only when every
    /// triangle touching the feature is degenerate.
    Vec3 mNormal{0.0, 0.0, 0.0};
    double mDistance = 0.0;                            ///< Its distance from the query (unsigned).
    std::int64_t mTriangle = -1;                       ///< The soup-local triangle the hit is on.
    std::int64_t mSourceCell = -1;                     ///< The input cell it came from.
    TriangleFeature mFeature = TriangleFeature::Face;  ///< Which feature was nearest.
    bool mFound = false;  ///< False only when the soup has no triangles at all.
};

/**
 * @brief The nearest point on the soup to each of @p rPoints, together with
 * the feature it lies on and that feature's pseudonormal.
 *
 * `shrinkwrap`'s primitive. It runs the identical `sd_nearest_triangle` search
 * `query_distances`/`query_closest_points` run and reads the feature normal
 * through the same lookup `query_distances`' pseudonormal sign uses
 * (`sd_feature_normal`), so the three cannot disagree about which triangle is
 * nearest or which way its feature faces. The normal is the FEATURE's -- at an
 * edge or vertex hit the offset surface's normal is the bisector, which is a
 * property of the feature and not of whichever of the equidistant triangles
 * won the tie-break -- and it is returned unnormalized so a caller can tell a
 * degenerate feature (zero) from a genuine direction.
 */
MESHIOPLUSPLUS_API std::vector<SurfaceProjection> query_surface_projections(
    const DistanceQuery& rQuery, const std::vector<Vec3>& rPoints);

}  // namespace detail
}  // namespace meshioplusplus
