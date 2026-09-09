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
// The signed-distance kernel. See detail/surface_distance.hpp for the contract,
// in particular why the accelerator is a bucket grid rather than a BVH and why
// the normal tables are accumulated serially.

// System includes
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Project includes
#include "meshioplusplus/detail/surface_distance.hpp"
#include "meshioplusplus/cell_type.hpp"
#include "meshioplusplus/detail/cell_index.hpp"
#include "meshioplusplus/detail/point_triangle.hpp"
#include "meshioplusplus/detail/value_io.hpp"
#include "meshioplusplus/parallel.hpp"
#include "meshioplusplus/region.hpp"

namespace meshioplusplus {
namespace detail {

namespace {

constexpr const char* kSdPrefix = "meshio++: surface distance: ";

// The cells a named Cell region selects, as a per-global-cell flag. An empty
// name means "everything", which is reported as an empty vector rather than an
// all-true one so the caller can skip the test entirely.
std::vector<char> sd_region_mask(const Mesh& rMesh, const std::string& rRegion) {
    if (rRegion.empty())
        return {};
    const std::size_t idx = rMesh.FindRegion(rRegion, RegionKind::Cell);
    if (idx == Mesh::npos) {
        std::string names;
        for (const std::string& n : rMesh.RegionNames())
            names += (names.empty() ? "" : ", ") + n;
        throw std::invalid_argument(std::string(kSdPrefix) + "no cell region named '" + rRegion +
                                    "' (available: " + (names.empty() ? "none" : names) + ")");
    }
    const std::vector<std::int64_t> bases = block_bases(rMesh);
    std::vector<char> mask(static_cast<std::size_t>(total_cells(bases)), 0);
    const NDArray& entries = rMesh.Region(idx).mEntries;
    for (std::size_t e = 0; e < entries.Size(); ++e) {
        const std::int64_t g = read_int(entries, e);
        if (g >= 0 && static_cast<std::size_t>(g) < mask.size())
            mask[static_cast<std::size_t>(g)] = 1;
    }
    return mask;
}

// The angle triangle (a, b, c) subtends at corner a. Used only as a positive
// weight on a unit normal, so its last-ulp behaviour cannot change a sign
// except where the distance is already zero to within rounding -- see
// doc/sdf.md on the one place the numpy twin excludes.
double sd_corner_angle(const Vec3& rA, const Vec3& rB, const Vec3& rC) {
    const Vec3 u = vec3_sub(rB, rA);
    const Vec3 v = vec3_sub(rC, rA);
    const double nu = vec3_norm(u);
    const double nv = vec3_norm(v);
    if (!(nu > 0.0) || !(nv > 0.0))
        return 0.0;
    double c = vec3_dot(u, v) / (nu * nv);
    c = c < -1.0 ? -1.0 : (c > 1.0 ? 1.0 : c);
    return std::acos(c);
}

}  // namespace

TriangleSoup build_triangle_soup(const Mesh& rSurface, const std::string& rRegion) {
    TriangleSoup soup;
    const std::size_t dim = rSurface.PointDim();
    const NDArray& points = rSurface.Points();
    soup.mPoints.resize(rSurface.NumPoints());
    for (std::size_t p = 0; p < rSurface.NumPoints(); ++p)
        soup.mPoints[p] = read_point(points, dim, static_cast<std::int64_t>(p));

    const std::vector<char> mask = sd_region_mask(rSurface, rRegion);
    const std::vector<std::int64_t> bases = block_bases(rSurface);

    std::size_t bi = 0;
    for (const auto cb : rSurface.CellRange()) {
        const std::int64_t base = bases[bi++];
        const std::string type(cb.Type());
        const CellType ct = cell_type_from_name(type);

        if (cb.IsPolyhedron() || cell_type_dimension(ct) == 3)
            throw std::invalid_argument(
                std::string(kSdPrefix) + "cell block '" + type +
                "' is a volume; distance is measured to a surface (run extract_surface first)");

        // A block whose cells happen to share a node count stores rectangularly
        // and so is not IsRagged(), which is why the type name is what decides
        // whether it is a polygon -- the trap cgns.cpp records.
        const bool polygon = type.rfind("polygon", 0) == 0;
        if (!polygon && ct != CellType::Triangle && ct != CellType::Quad) {
            if (cell_type_dimension(ct) < 2)
                continue;  // lines and vertices carry no area; silently skipped
            throw std::invalid_argument(std::string(kSdPrefix) + "cell block '" + type +
                                        "' is not a linear surface cell (linearize the mesh "
                                        "first, then run extract_surface if needed)");
        }

        const std::size_t ncells = cb.NumCells();
        for (std::size_t c = 0; c < ncells; ++c) {
            const std::int64_t global = base + static_cast<std::int64_t>(c);
            if (!mask.empty() && !mask[static_cast<std::size_t>(global)])
                continue;

            // Gather this cell's corners, ragged or not.
            std::vector<std::int64_t> ids;
            if (cb.IsRagged()) {
                const std::size_t n = cb.RowSize(c);
                const std::int64_t* row = cb.Row(c);
                ids.assign(row, row + n);
            } else {
                const NDArray& conn = cb.Conn();
                const std::size_t npc = cb.NodesPerCell();
                ids.resize(npc);
                for (std::size_t i = 0; i < npc; ++i)
                    ids[i] = read_int(conn, c * npc + i);
            }
            if (ids.size() < 3)
                continue;

            // The same fan convert_cells(Simplexify) uses: corner 0 to every
            // non-adjacent edge. Transcribing a different fan here would make
            // the two disagree about which diagonal a quad is split on.
            for (std::size_t k = 1; k + 1 < ids.size(); ++k) {
                const std::array<std::int64_t, 3> tri{ids[0], ids[k], ids[k + 1]};
                soup.mVertices.push_back(tri);
                soup.mSourceCell.push_back(global);
                for (std::size_t i = 0; i < 3; ++i)
                    soup.mCorners.push_back(soup.mPoints[static_cast<std::size_t>(tri[i])]);
            }
        }
    }
    return soup;
}

SurfaceEdgeMap build_surface_edges(const TriangleSoup& rSoup) {
    // Per undirected edge: how many triangles use it, and how many use it in the
    // low->high direction. A consistently wound closed surface has every edge
    // used exactly twice, once in each direction.
    const std::size_t ntri = rSoup.NumTriangles();
    SurfaceEdgeMap edges;
    edges.reserve(ntri * 3 * 2);
    for (std::size_t t = 0; t < ntri; ++t) {
        const std::array<std::int64_t, 3>& v = rSoup.mVertices[t];
        for (std::size_t e = 0; e < 3; ++e) {
            const std::int64_t u = v[e];
            const std::int64_t w = v[(e + 1) % 3];
            const SurfaceEdgeKey key{u < w ? u : w, u < w ? w : u};
            SurfaceEdgeRecord& rec = edges[key];
            ++rec.mUsed;
            if (u < w)
                ++rec.mForward;
            if (rec.mFirstTriangle < 0)
                rec.mFirstTriangle = static_cast<std::int64_t>(t);
        }
    }
    return edges;
}

SurfaceQuality soup_quality(const TriangleSoup& rSoup) {
    SurfaceQuality q;
    const std::size_t ntri = rSoup.NumTriangles();

    for (std::size_t t = 0; t < ntri; ++t) {
        const Vec3& a = rSoup.mCorners[t * 3 + 0];
        const Vec3& b = rSoup.mCorners[t * 3 + 1];
        const Vec3& c = rSoup.mCorners[t * 3 + 2];
        if (!(vec3_norm_sq(vec3_cross(vec3_sub(b, a), vec3_sub(c, a))) > 0.0))
            ++q.mDegenerateTriangles;
    }

    const SurfaceEdgeMap edges = build_surface_edges(rSoup);
    for (const auto& kv : edges) {
        const std::int64_t used = kv.second.mUsed;
        const std::int64_t forward = kv.second.mForward;
        if (used == 1)
            ++q.mBoundaryEdges;
        else if (used > 2)
            ++q.mNonManifoldEdges;
        else if (used == 2 && forward != 1)
            ++q.mInconsistentPairs;  // both the same way round: they disagree on "out"
    }
    q.mWatertight = q.mBoundaryEdges == 0 && q.mNonManifoldEdges == 0 &&
                    q.mInconsistentPairs == 0 && q.mDegenerateTriangles == 0;
    return q;
}

DistanceQuery build_distance_query(const TriangleSoup& rSoup,
                                   const SurfaceDistanceOptions& rOptions) {
    const std::size_t ntri = rSoup.NumTriangles();
    if (ntri == 0)
        throw std::invalid_argument(std::string(kSdPrefix) +
                                    "the surface has no triangles to measure against");

    DistanceQuery q;
    q.mpSoup = &rSoup;
    q.mFaceNormal.resize(ntri);

    // Bucket size. It affects only how many candidates each query examines --
    // never the answer, because every comparison below is totally ordered -- so
    // the rule here is a pure performance heuristic. That is not a throwaway
    // remark: it is what let this rule be retuned after measurement without
    // re-validating a single distance, and `TheBucketSizeDoesNotChangeTheAnswer`
    // is the proof.
    //
    // Sizing buckets by the mean triangle alone is the obvious rule and the wrong
    // one. On a finely tessellated model the triangles are tiny, so the buckets
    // are tiny, and a query far from the surface has to expand through hundreds
    // of empty shells before it finds anything -- a 64^3 inside-fill of the
    // 112k-triangle Stanford bunny took 19 seconds that way. The domain's own
    // size has to enter the rule, so the base is the extent divided by the cube
    // root of the triangle count (roughly "one bucket per triangle's worth of
    // volume"), floored at the mean triangle so buckets never split a single
    // triangle needlessly and capped a few multiples above it.
    double cell = rOptions.mGridCellSize;
    if (!(cell > 0.0)) {
        Vec3 lo = rSoup.mCorners[0];
        Vec3 hi = lo;
        double sum = 0.0;
        for (std::size_t t = 0; t < ntri; ++t) {
            Vec3 tlo = rSoup.mCorners[t * 3];
            Vec3 thi = tlo;
            for (std::size_t i = 1; i < 3; ++i)
                for (std::size_t k = 0; k < 3; ++k) {
                    const double v = rSoup.mCorners[t * 3 + i][k];
                    tlo[k] = tlo[k] < v ? tlo[k] : v;
                    thi[k] = thi[k] > v ? thi[k] : v;
                }
            for (std::size_t k = 0; k < 3; ++k) {
                lo[k] = lo[k] < tlo[k] ? lo[k] : tlo[k];
                hi[k] = hi[k] > thi[k] ? hi[k] : thi[k];
            }
            sum += vec3_norm(vec3_sub(thi, tlo));
        }
        const double mean_tri = sum / static_cast<double>(ntri);
        double extent = 0.0;
        for (std::size_t k = 0; k < 3; ++k)
            extent = extent > (hi[k] - lo[k]) ? extent : (hi[k] - lo[k]);
        const double base = extent / std::cbrt(static_cast<double>(ntri));
        cell = base < mean_tri ? mean_tri : (base > 8.0 * mean_tri ? 8.0 * mean_tri : base);
    }
    if (!(cell > 0.0))
        cell = 1.0;  // every triangle degenerate to a point: any bucket size will do
    q.mCellSize = cell;
    q.mGrid = SpatialGrid(cell);

    // Serial ascending insert. The bucket contents order is not observable given
    // the tie-break, but keeping the insert serial costs nothing here and keeps
    // the structure's documented determinism contract intact.
    for (std::size_t t = 0; t < ntri; ++t) {
        Vec3 lo = rSoup.mCorners[t * 3];
        Vec3 hi = lo;
        for (std::size_t i = 1; i < 3; ++i)
            for (std::size_t k = 0; k < 3; ++k) {
                const double v = rSoup.mCorners[t * 3 + i][k];
                lo[k] = lo[k] < v ? lo[k] : v;
                hi[k] = hi[k] > v ? hi[k] : v;
            }
        q.mGrid.InsertBox(q.mGrid.KeyOf(lo.data()), q.mGrid.KeyOf(hi.data()),
                          static_cast<std::int64_t>(t));
    }

    // Face normals, then the vertex and edge tables. The table pass is SERIAL
    // and in ascending (triangle, corner) order: summing unit normals in a
    // different order changes the last bits, and a last-bit change can flip the
    // sign of a query point sitting almost exactly on the surface.
    q.mVertexNormal.assign(rSoup.mPoints.size(), Vec3{0.0, 0.0, 0.0});
    for (std::size_t t = 0; t < ntri; ++t) {
        const Vec3& a = rSoup.mCorners[t * 3 + 0];
        const Vec3& b = rSoup.mCorners[t * 3 + 1];
        const Vec3& c = rSoup.mCorners[t * 3 + 2];
        q.mFaceNormal[t] = vec3_cross(vec3_sub(b, a), vec3_sub(c, a));
    }
    for (std::size_t t = 0; t < ntri; ++t) {
        const Vec3 n = q.mFaceNormal[t];
        const double len = vec3_norm(n);
        if (!(len > 0.0))
            continue;  // degenerate: no direction to contribute
        const Vec3 unit = vec3_scale(n, 1.0 / len);
        const std::array<std::int64_t, 3>& v = rSoup.mVertices[t];
        const Vec3* corner = &rSoup.mCorners[t * 3];
        for (std::size_t i = 0; i < 3; ++i) {
            const double w =
                rOptions.mWeight == SdfPseudonormalWeight::Angle
                    ? sd_corner_angle(corner[i], corner[(i + 1) % 3], corner[(i + 2) % 3])
                    : len;  // area weighting: |cross| is twice the area, a positive scale
            Vec3& acc = q.mVertexNormal[static_cast<std::size_t>(v[i])];
            acc = vec3_add(acc, vec3_scale(unit, w));

            const std::int64_t p = v[i];
            const std::int64_t r = v[(i + 1) % 3];
            const SurfaceEdgeKey key{p < r ? p : r, p < r ? r : p};
            Vec3& e = q.mEdgeNormal[key];
            e = vec3_add(e, unit);
        }
    }
    return q;
}

namespace {

// The bucket-grid nearest-triangle search shared by query_distances and
// query_closest_points, extracted verbatim from what used to be inline in
// query_distances (a pure refactor -- query_distances' own test suite is the
// regression guard that its output is unchanged). @p MaxShell caps expansion,
// used by query_distances to stop early under a band; a plain search (as
// query_closest_points needs) passes the int64 max.
struct SdNearestTriangle {
    std::int64_t mTri = -1;
    PointTriangleHit mHit;
};

SdNearestTriangle sd_nearest_triangle(const DistanceQuery& rQuery, const TriangleSoup& rSoup,
                                      const Vec3& rQueryPoint, std::int64_t MaxShell) {
    const GridKey centre = rQuery.mGrid.KeyOf(rQueryPoint.data());

    double best_d2 = std::numeric_limits<double>::infinity();
    std::int64_t best_tri = -1;
    PointTriangleHit best_hit;

    // The largest shell radius that can still reach an occupied bucket. Note
    // ForEachInShell clamps to the occupied box, so an empty shell does NOT
    // mean "no more candidates" for a query far outside it -- without this
    // bound the loop would stop early on exactly those points.
    std::int64_t reach = 0;
    if (!rQuery.mGrid.Empty()) {
        const GridKey lo = rQuery.mGrid.OccupiedLo();
        const GridKey hi = rQuery.mGrid.OccupiedHi();
        const std::int64_t dx = std::max(std::abs(centre.x - lo.x), std::abs(centre.x - hi.x));
        const std::int64_t dy = std::max(std::abs(centre.y - lo.y), std::abs(centre.y - hi.y));
        const std::int64_t dz = std::max(std::abs(centre.z - lo.z), std::abs(centre.z - hi.z));
        reach = std::max(dx, std::max(dy, dz));
    }
    if (reach > MaxShell)
        reach = MaxShell;

    for (std::int64_t r = 0; r <= reach; ++r) {
        // A hit in a bucket at Chebyshev radius r is at least (r - 1) * cell
        // away, so once that bound exceeds the best found there is nothing
        // left to find.
        if (r >= 1 && best_tri >= 0) {
            const double bound = static_cast<double>(r - 1) * rQuery.mCellSize;
            if (bound > 0.0 && bound * bound > best_d2)
                break;
        }
        rQuery.mGrid.ForEachInShell(centre, r, [&](const std::vector<std::int64_t>& rIds) {
            for (std::int64_t t : rIds) {
                const std::size_t ti = static_cast<std::size_t>(t);
                const PointTriangleHit hit = closest_point_on_triangle(
                    rQueryPoint, rSoup.mCorners[ti * 3 + 0], rSoup.mCorners[ti * 3 + 1],
                    rSoup.mCorners[ti * 3 + 2]);
                // The total order that makes the accelerator unobservable:
                // distance first, then the triangle id, so two equidistant
                // triangles always resolve the same way regardless of which
                // bucket happened to be visited first.
                if (hit.mDistanceSq < best_d2 || (hit.mDistanceSq == best_d2 && t < best_tri)) {
                    best_d2 = hit.mDistanceSq;
                    best_tri = t;
                    best_hit = hit;
                }
            }
        });
    }
    return {best_tri, best_hit};
}

// The pseudonormal of the FEATURE a hit landed on, not of the nearest
// triangle: using the triangle's own normal is right on convex geometry and
// wrong on the concave side of every crease. Hoisted verbatim out of
// query_distances (a pure refactor -- that function's own suite is the
// regression guard) so query_surface_projections reads the same tables the
// same way. Returned unnormalized.
Vec3 sd_feature_normal(const DistanceQuery& rQuery, const TriangleSoup& rSoup, std::int64_t Tri,
                       TriangleFeature Feature) {
    const std::size_t ti = static_cast<std::size_t>(Tri);
    const std::array<std::int64_t, 3>& v = rSoup.mVertices[ti];
    Vec3 normal = rQuery.mFaceNormal[ti];
    switch (Feature) {
        case TriangleFeature::VertexA:
            normal = rQuery.mVertexNormal[static_cast<std::size_t>(v[0])];
            break;
        case TriangleFeature::VertexB:
            normal = rQuery.mVertexNormal[static_cast<std::size_t>(v[1])];
            break;
        case TriangleFeature::VertexC:
            normal = rQuery.mVertexNormal[static_cast<std::size_t>(v[2])];
            break;
        case TriangleFeature::EdgeAB:
        case TriangleFeature::EdgeBC:
        case TriangleFeature::EdgeCA: {
            const std::size_t e = Feature == TriangleFeature::EdgeAB
                                      ? 0
                                      : (Feature == TriangleFeature::EdgeBC ? 1 : 2);
            const std::int64_t a = v[e];
            const std::int64_t b = v[(e + 1) % 3];
            const SurfaceEdgeKey key{a < b ? a : b, a < b ? b : a};
            auto it = rQuery.mEdgeNormal.find(key);
            if (it != rQuery.mEdgeNormal.end())
                normal = it->second;
            break;
        }
        case TriangleFeature::Face:
        default:
            break;
    }
    return normal;
}

}  // namespace

std::vector<DistanceHit> query_distances(const DistanceQuery& rQuery,
                                         const std::vector<Vec3>& rPoints,
                                         const SurfaceDistanceOptions& rOptions) {
    const TriangleSoup& soup = *rQuery.mpSoup;
    const std::size_t ntri = soup.NumTriangles();
    const std::size_t n = rPoints.size();
    std::vector<DistanceHit> out(n);

    if (rOptions.mSign == SdfSign::WindingNumber && rOptions.mMaxWindingWork > 0.0) {
        const double work = static_cast<double>(n) * static_cast<double>(ntri);
        if (work > rOptions.mMaxWindingWork)
            throw std::invalid_argument(
                std::string(kSdPrefix) + "sign='winding-number' is O(triangles) per query and " +
                std::to_string(n) + " queries x " + std::to_string(ntri) + " triangles exceeds " +
                "max_winding_work (raise it, use a band, or use sign='pseudonormal')");
    }

    const double band = rOptions.mBand;
    const bool banded = band > 0.0;
    const double band2 = band * band;
    // With a band there is no point expanding past it: a hit found beyond this
    // radius would be clamped anyway.
    const std::int64_t max_shell =
        banded ? static_cast<std::int64_t>(std::ceil(band / rQuery.mCellSize)) + 1
               : std::numeric_limits<std::int64_t>::max();

    parallel_for(n, [&](std::size_t p) {
        const Vec3& query = rPoints[p];
        const SdNearestTriangle found = sd_nearest_triangle(rQuery, soup, query, max_shell);
        const std::int64_t best_tri = found.mTri;
        const PointTriangleHit& best_hit = found.mHit;
        const double best_d2 = best_hit.mDistanceSq;

        DistanceHit& res = out[p];
        if (best_tri < 0) {
            // Nothing within reach: only possible under a band.
            res.mSignedDistance = band;
            res.mSourceCell = -1;
            res.mInBand = false;
            return;
        }
        const double dist = std::sqrt(best_d2);
        if (banded && best_d2 > band2) {
            res.mSignedDistance = band;
            res.mSourceCell = -1;
            res.mInBand = false;
            return;
        }
        res.mSourceCell = soup.mSourceCell[static_cast<std::size_t>(best_tri)];
        res.mInBand = true;

        if (rOptions.mSign == SdfSign::Unsigned) {
            res.mSignedDistance = dist;
            return;
        }
        if (rOptions.mSign == SdfSign::WindingNumber) {
            // Van Oosterom-Strackee solid angle, summed in ascending triangle
            // order. O(triangles) per query, which is why it is guarded above.
            double w = 0.0;
            for (std::size_t t = 0; t < ntri; ++t) {
                const Vec3 a = vec3_sub(soup.mCorners[t * 3 + 0], query);
                const Vec3 b = vec3_sub(soup.mCorners[t * 3 + 1], query);
                const Vec3 c = vec3_sub(soup.mCorners[t * 3 + 2], query);
                const double la = vec3_norm(a);
                const double lb = vec3_norm(b);
                const double lc = vec3_norm(c);
                const double num = triple_product(a, b, c);
                const double den =
                    la * lb * lc + vec3_dot(a, b) * lc + vec3_dot(b, c) * la + vec3_dot(c, a) * lb;
                w += 2.0 * std::atan2(num, den);
            }
            const bool inside = w / (4.0 * 3.14159265358979323846) > 0.5;
            res.mSignedDistance = inside ? -dist : dist;
            return;
        }

        // Pseudonormal: the normal of the nearest FEATURE, not of the nearest
        // triangle (sd_feature_normal says why).
        const Vec3 normal = sd_feature_normal(rQuery, soup, best_tri, best_hit.mFeature);
        const double side = vec3_dot(vec3_sub(query, best_hit.mPoint), normal);
        res.mSignedDistance = side < 0.0 ? -dist : dist;
    });

    return out;
}

std::vector<ClosestPointHit> query_closest_points(const DistanceQuery& rQuery,
                                                  const std::vector<Vec3>& rPoints) {
    const TriangleSoup& soup = *rQuery.mpSoup;
    const std::size_t n = rPoints.size();
    std::vector<ClosestPointHit> out(n);

    parallel_for(n, [&](std::size_t p) {
        const SdNearestTriangle found =
            sd_nearest_triangle(rQuery, soup, rPoints[p], std::numeric_limits<std::int64_t>::max());
        ClosestPointHit& res = out[p];
        if (found.mTri < 0) {
            res.mFound = false;
            return;
        }
        res.mFound = true;
        res.mPoint = found.mHit.mPoint;
        res.mDistance = std::sqrt(found.mHit.mDistanceSq);
        res.mSourceCell = soup.mSourceCell[static_cast<std::size_t>(found.mTri)];
    });

    return out;
}

std::vector<SurfaceProjection> query_surface_projections(const DistanceQuery& rQuery,
                                                         const std::vector<Vec3>& rPoints) {
    const TriangleSoup& soup = *rQuery.mpSoup;
    const std::size_t n = rPoints.size();
    std::vector<SurfaceProjection> out(n);

    parallel_for(n, [&](std::size_t p) {
        const SdNearestTriangle found =
            sd_nearest_triangle(rQuery, soup, rPoints[p], std::numeric_limits<std::int64_t>::max());
        SurfaceProjection& res = out[p];
        if (found.mTri < 0) {
            res.mFound = false;
            return;
        }
        res.mFound = true;
        res.mPoint = found.mHit.mPoint;
        res.mDistance = std::sqrt(found.mHit.mDistanceSq);
        res.mTriangle = found.mTri;
        res.mSourceCell = soup.mSourceCell[static_cast<std::size_t>(found.mTri)];
        res.mFeature = found.mHit.mFeature;
        res.mNormal = sd_feature_normal(rQuery, soup, found.mTri, found.mHit.mFeature);
    });

    return out;
}

}  // namespace detail
}  // namespace meshioplusplus
