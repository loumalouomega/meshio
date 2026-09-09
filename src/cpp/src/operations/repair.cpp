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
// Surface repair: bowtie splitting, half-edge orientation BFS, fan hole
// filling and outward orientation of closed components. See
// operations/repair.hpp for the contract, the pass order and the two
// deliberate divergences from upstream.

// System includes
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Project includes
#include "meshioplusplus/operations/repair.hpp"
#include "meshioplusplus/cell_type.hpp"
#include "meshioplusplus/detail/data_ops.hpp"
#include "meshioplusplus/detail/geometry.hpp"
#include "meshioplusplus/detail/region_remap.hpp"
#include "meshioplusplus/detail/surface_distance.hpp"
#include "meshioplusplus/detail/value_io.hpp"
#include "meshioplusplus/operations/clean.hpp"
#include "meshioplusplus/operations/convert_cells.hpp"
#include "meshioplusplus/parallel.hpp"
#include "meshioplusplus/region.hpp"

namespace meshioplusplus {
namespace {

using detail::Vec3;
using RepairTri = std::array<std::int64_t, 3>;

constexpr const char* kRepairPrefix = "meshio++: repair: ";

// Reject, by name, every construct outside the surface scope -- BEFORE the
// simplexify call, whose own policy is to pass unsupported blocks through.
// Lower-dimensional blocks are allowed and ride along (repair.hpp says why).
void repair_check_blocks(const Mesh& rMesh) {
    bool has_surface = false;
    for (const auto cb : rMesh.CellRange()) {
        const std::string type(cb.Type());
        if (cb.IsPolyhedron())
            throw std::invalid_argument(std::string(kRepairPrefix) +
                                        "mesh contains a polyhedron cell block; repair operates "
                                        "on surface meshes (run extract_surface first)");
        const CellType ct = cell_type_from_name(type);
        const int dim = cell_type_dimension(ct);
        if (dim == 3)
            throw std::invalid_argument(std::string(kRepairPrefix) +
                                        "mesh contains 3D volume cell block '" + type +
                                        "'; repair operates on surface meshes (extract_surface "
                                        "first)");
        if (dim != 2)
            continue;
        has_surface = true;
        const bool polygon = type.rfind("polygon", 0) == 0;
        if (!polygon && ct != CellType::Triangle && ct != CellType::Quad)
            throw std::invalid_argument(std::string(kRepairPrefix) +
                                        "cannot repair higher-order cell block '" + type +
                                        "' (linearize the mesh first)");
        if (cb.IsRagged() && !polygon)
            throw std::invalid_argument(std::string(kRepairPrefix) +
                                        "cannot repair ragged cell block '" + type + "'");
    }
    if (!has_surface)
        throw std::invalid_argument(std::string(kRepairPrefix) +
                                    "mesh contains no surface (2D) cell block");
}

// --- the half-edge table ------------------------------------------------------
// 3T records sorted by (lo, hi, tri, slot); an undirected edge is a run of
// equal (lo, hi). Parallel fill, SERIAL sort, so the edge ids are a function
// of the connectivity alone. A self-loop (a repeated corner) gets no edge id
// and is never crossed or traced.
struct RepairHalfEdge {
    std::int64_t mLo = 0;
    std::int64_t mHi = 0;
    std::int64_t mTri = 0;
    std::uint8_t mSlot = 0;     // which of the triangle's three edges
    std::uint8_t mForward = 0;  // traversed lo -> hi as stored
    bool operator<(const RepairHalfEdge& rO) const {
        if (mLo != rO.mLo)
            return mLo < rO.mLo;
        if (mHi != rO.mHi)
            return mHi < rO.mHi;
        if (mTri != rO.mTri)
            return mTri < rO.mTri;
        return mSlot < rO.mSlot;
    }
};

struct RepairEdges {
    std::vector<RepairHalfEdge> mRecs;  // sorted
    std::vector<std::size_t> mBegin;    // per edge: first record
    std::vector<std::size_t> mEnd;      // per edge: one past the last record
    std::vector<std::int64_t> mEdgeOf;  // per (tri, slot): edge id, -1 for a self-loop
    std::size_t NumEdges() const { return mBegin.size(); }
    std::size_t Count(std::size_t e) const { return mEnd[e] - mBegin[e]; }
};

RepairEdges repair_build_edges(const std::vector<RepairTri>& rTris) {
    const std::size_t ntri = rTris.size();
    RepairEdges out;
    out.mRecs.resize(ntri * 3);
    parallel_for(ntri, [&](std::size_t t) {
        for (std::size_t e = 0; e < 3; ++e) {
            const std::int64_t u = rTris[t][e];
            const std::int64_t w = rTris[t][(e + 1) % 3];
            RepairHalfEdge& r = out.mRecs[t * 3 + e];
            r.mLo = u < w ? u : w;
            r.mHi = u < w ? w : u;
            r.mTri = static_cast<std::int64_t>(t);
            r.mSlot = static_cast<std::uint8_t>(e);
            r.mForward = u < w ? 1 : 0;
        }
    });
    std::sort(out.mRecs.begin(), out.mRecs.end());
    out.mEdgeOf.assign(ntri * 3, -1);
    std::size_t i = 0;
    while (i < out.mRecs.size()) {
        std::size_t j = i + 1;
        while (j < out.mRecs.size() && out.mRecs[j].mLo == out.mRecs[i].mLo &&
               out.mRecs[j].mHi == out.mRecs[i].mHi)
            ++j;
        if (out.mRecs[i].mLo != out.mRecs[i].mHi) {
            const std::int64_t id = static_cast<std::int64_t>(out.mBegin.size());
            out.mBegin.push_back(i);
            out.mEnd.push_back(j);
            for (std::size_t k = i; k < j; ++k)
                out.mEdgeOf[static_cast<std::size_t>(out.mRecs[k].mTri) * 3 + out.mRecs[k].mSlot] =
                    id;
        }
        i = j;
    }
    return out;
}

// --- bowtie splitting -----------------------------------------------------------
// A vertex whose star is edge-disconnected is duplicated once per extra
// component (geometry unchanged); the lowest-triangle group keeps the id.
// Detection runs in parallel per vertex over the ORIGINAL connectivity into
// disjoint slots; ids are handed out in a serial ascending-vertex pass, so
// they are stable. Decisions at different vertices cannot interfere: two
// triangles in different groups at v share no edge (v, w) by definition, so
// w's own connectivity through v-edges is unaffected by the split.
struct RepairStar {
    std::vector<std::size_t> mOffsets;  // n + 1
    std::vector<std::int64_t> mTri;     // ascending per vertex
};

RepairStar repair_build_star(const std::vector<RepairTri>& rTris, std::size_t n) {
    RepairStar s;
    s.mOffsets.assign(n + 1, 0);
    for (const RepairTri& t : rTris)
        for (const std::int64_t v : t)
            ++s.mOffsets[static_cast<std::size_t>(v) + 1];
    for (std::size_t i = 0; i < n; ++i)
        s.mOffsets[i + 1] += s.mOffsets[i];
    s.mTri.resize(s.mOffsets[n]);
    std::vector<std::size_t> fill(s.mOffsets.begin(), s.mOffsets.end() - 1);
    for (std::size_t t = 0; t < rTris.size(); ++t)
        for (const std::int64_t v : rTris[t])
            s.mTri[fill[static_cast<std::size_t>(v)]++] = static_cast<std::int64_t>(t);
    return s;
}

std::int64_t repair_uf_find(std::vector<std::int64_t>& rParent, std::int64_t i) {
    while (rParent[static_cast<std::size_t>(i)] != i) {
        rParent[static_cast<std::size_t>(i)] =
            rParent[static_cast<std::size_t>(rParent[static_cast<std::size_t>(i)])];
        i = rParent[static_cast<std::size_t>(i)];
    }
    return i;
}

/// Splits bowties in place. Returns the number of vertices split; appends the
/// source of every new point to @p rParentOfNew and returns the new point count
/// through @p rNumPoints.
std::int64_t repair_split_bowties(std::vector<RepairTri>& rTris, std::size_t& rNumPoints,
                                  std::vector<std::int64_t>& rParentOfNew) {
    const std::size_t n = rNumPoints;
    const RepairStar star = repair_build_star(rTris, n);
    // Per incident slot: the group (0 = keeps the id) of that triangle at v.
    std::vector<std::int32_t> slot_group(star.mTri.size(), 0);
    std::vector<std::int32_t> ngroups(n, 0);
    parallel_for(n, [&](std::size_t v) {
        const std::size_t lo = star.mOffsets[v], hi = star.mOffsets[v + 1];
        const std::size_t m = hi - lo;
        if (m < 2) {
            ngroups[v] = m == 0 ? 0 : 1;
            return;
        }
        std::vector<std::int64_t> parent(m);
        for (std::size_t k = 0; k < m; ++k)
            parent[k] = static_cast<std::int64_t>(k);
        // The other two corners of every incident triangle; two triangles
        // that share one of them share the edge (v, w).
        std::vector<std::pair<std::int64_t, std::size_t>> others;
        others.reserve(m * 2);
        for (std::size_t k = 0; k < m; ++k) {
            const RepairTri& t = rTris[static_cast<std::size_t>(star.mTri[lo + k])];
            for (const std::int64_t w : t)
                if (w != static_cast<std::int64_t>(v))
                    others.push_back({w, k});
        }
        std::sort(others.begin(), others.end());
        for (std::size_t a = 0; a + 1 < others.size(); ++a)
            if (others[a].first == others[a + 1].first) {
                const std::int64_t ra =
                    repair_uf_find(parent, static_cast<std::int64_t>(others[a].second));
                const std::int64_t rb =
                    repair_uf_find(parent, static_cast<std::int64_t>(others[a + 1].second));
                if (ra != rb)
                    parent[static_cast<std::size_t>(std::max(ra, rb))] = std::min(ra, rb);
            }
        // Groups numbered by first appearance in ascending triangle order.
        std::vector<std::int32_t> group_of_root(m, -1);
        std::int32_t g = 0;
        for (std::size_t k = 0; k < m; ++k) {
            const std::size_t r =
                static_cast<std::size_t>(repair_uf_find(parent, static_cast<std::int64_t>(k)));
            if (group_of_root[r] < 0)
                group_of_root[r] = g++;
            slot_group[lo + k] = group_of_root[r];
        }
        ngroups[v] = g;
    });

    std::int64_t split = 0;
    std::size_t next_id = n;
    for (std::size_t v = 0; v < n; ++v) {
        if (ngroups[v] <= 1)
            continue;
        ++split;
        std::vector<std::int64_t> new_id(static_cast<std::size_t>(ngroups[v]), -1);
        for (std::int32_t g = 1; g < ngroups[v]; ++g) {
            new_id[static_cast<std::size_t>(g)] = static_cast<std::int64_t>(next_id++);
            rParentOfNew.push_back(static_cast<std::int64_t>(v));
        }
        for (std::size_t k = star.mOffsets[v]; k < star.mOffsets[v + 1]; ++k) {
            const std::int32_t g = slot_group[k];
            if (g == 0)
                continue;
            RepairTri& t = rTris[static_cast<std::size_t>(star.mTri[k])];
            for (std::int64_t& c : t)
                if (c == static_cast<std::int64_t>(v))
                    c = new_id[static_cast<std::size_t>(g)];
        }
    }
    rNumPoints = next_id;
    return split;
}

// --- orientation ------------------------------------------------------------------
struct RepairOrientation {
    std::vector<std::uint8_t> mFlip;       // per triangle, after the fewest-flips rule
    std::vector<std::int64_t> mComponent;  // per triangle
    std::vector<std::int64_t> mSize;       // per component
    std::vector<std::uint8_t> mUnorientable;
};

/// The topological rule: two triangles sharing a manifold edge agree iff they
/// traverse it in opposite directions. Boundary and non-manifold edges are
/// walls; a parity conflict is recorded, never thrown.
RepairOrientation repair_orient(const std::vector<RepairTri>& rTris, const RepairEdges& rEdges) {
    const std::size_t ntri = rTris.size();
    RepairOrientation o;
    std::vector<std::int8_t> flip(ntri, -1);
    o.mComponent.assign(ntri, -1);
    std::vector<std::int64_t> stack;
    std::vector<std::int64_t> members;
    for (std::size_t seed = 0; seed < ntri; ++seed) {
        if (flip[seed] >= 0)
            continue;
        const std::int64_t comp = static_cast<std::int64_t>(o.mSize.size());
        bool conflict = false;
        flip[seed] = 0;
        stack.assign(1, static_cast<std::int64_t>(seed));
        members.clear();
        while (!stack.empty()) {
            const std::int64_t t = stack.back();
            stack.pop_back();
            members.push_back(t);
            o.mComponent[static_cast<std::size_t>(t)] = comp;
            for (std::size_t e = 0; e < 3; ++e) {
                const std::int64_t edge = rEdges.mEdgeOf[static_cast<std::size_t>(t) * 3 + e];
                if (edge < 0 || rEdges.Count(static_cast<std::size_t>(edge)) != 2)
                    continue;
                const RepairHalfEdge& r0 =
                    rEdges.mRecs[rEdges.mBegin[static_cast<std::size_t>(edge)]];
                const RepairHalfEdge& r1 =
                    rEdges.mRecs[rEdges.mBegin[static_cast<std::size_t>(edge)] + 1];
                const RepairHalfEdge& mine = r0.mTri == t && r0.mSlot == e ? r0 : r1;
                const RepairHalfEdge& other = &mine == &r0 ? r1 : r0;
                if (other.mTri == t)
                    continue;  // the same triangle on both sides: nothing to propagate
                const std::uint8_t mine_fwd = mine.mForward ^ flip[static_cast<std::size_t>(t)];
                // Agree iff opposite directions: other must end up != mine_fwd.
                const std::int8_t want = other.mForward == mine_fwd ? 1 : 0;
                std::int8_t& of = flip[static_cast<std::size_t>(other.mTri)];
                if (of < 0) {
                    of = want;
                    stack.push_back(other.mTri);
                } else if (of != want) {
                    conflict = true;
                }
            }
        }
        std::int64_t nflip = 0;
        for (const std::int64_t m : members)
            nflip += flip[static_cast<std::size_t>(m)];
        if (nflip * 2 > static_cast<std::int64_t>(members.size()))
            for (const std::int64_t m : members)
                flip[static_cast<std::size_t>(m)] ^= 1;
        o.mSize.push_back(static_cast<std::int64_t>(members.size()));
        o.mUnorientable.push_back(conflict ? 1 : 0);
    }
    o.mFlip.resize(ntri);
    for (std::size_t t = 0; t < ntri; ++t)
        o.mFlip[t] = static_cast<std::uint8_t>(flip[t]);
    return o;
}

// --- hole filling ---------------------------------------------------------------------
struct RepairHole {
    std::vector<std::int64_t> mLoop;  // boundary vertices, in fill order
    std::int64_t mComponent = -1;     // of the triangle owning its first edge
};

struct RepairHoles {
    std::vector<RepairHole> mFilled;
    std::int64_t mDetected = 0;
    std::int64_t mSkipped = 0;
};

/// Trace every boundary loop against the surface's own traversal (so the fan
/// `(v_i, v_{i+1}, c)` winds opposite to the neighbour across each edge).
RepairHoles repair_trace_holes(const std::vector<RepairTri>& rTris, const RepairEdges& rEdges,
                               const std::vector<std::int64_t>& rComponent, std::size_t n,
                               std::int64_t MaxEdges) {
    std::vector<std::int64_t> next(n, -1);
    std::vector<std::int64_t> owner(n, -1);
    std::vector<std::uint8_t> indeg(n, 0), outdeg(n, 0);
    for (std::size_t t = 0; t < rTris.size(); ++t)
        for (std::size_t e = 0; e < 3; ++e) {
            const std::int64_t edge = rEdges.mEdgeOf[t * 3 + e];
            if (edge < 0 || rEdges.Count(static_cast<std::size_t>(edge)) != 1)
                continue;
            const std::int64_t a = rTris[t][e];
            const std::int64_t b = rTris[t][(e + 1) % 3];
            // The triangle traverses a -> b; the loop runs b -> a.
            next[static_cast<std::size_t>(b)] = a;
            owner[static_cast<std::size_t>(b)] = static_cast<std::int64_t>(t);
            if (outdeg[static_cast<std::size_t>(b)] < 255)
                ++outdeg[static_cast<std::size_t>(b)];
            if (indeg[static_cast<std::size_t>(a)] < 255)
                ++indeg[static_cast<std::size_t>(a)];
        }
    RepairHoles out;
    std::vector<std::uint8_t> visited(n, 0);
    for (std::size_t s = 0; s < n; ++s) {
        if (outdeg[s] == 0 || visited[s])
            continue;
        RepairHole hole;
        bool ok = true;
        std::int64_t cur = static_cast<std::int64_t>(s);
        while (true) {
            const std::size_t c = static_cast<std::size_t>(cur);
            if (visited[c]) {
                ok = ok && cur == static_cast<std::int64_t>(s) && !hole.mLoop.empty();
                break;
            }
            visited[c] = 1;
            hole.mLoop.push_back(cur);
            if (indeg[c] != 1 || outdeg[c] != 1) {
                ok = false;
                break;
            }
            cur = next[c];
            if (cur < 0) {
                ok = false;
                break;
            }
        }
        ++out.mDetected;
        const std::int64_t len = static_cast<std::int64_t>(hole.mLoop.size());
        if (!ok || len < 3 || (MaxEdges > 0 && len > MaxEdges)) {
            ++out.mSkipped;
            continue;
        }
        hole.mComponent = rComponent[static_cast<std::size_t>(owner[s])];
        out.mFilled.push_back(std::move(hole));
    }
    return out;
}

// The signed volume of a set of triangles by the divergence theorem, recentred
// on their corner mean (poly_measure's numerical-stability lesson); serial in
// ascending triangle order.
double repair_signed_volume(const std::vector<RepairTri>& rTris,
                            const std::vector<std::int64_t>& rIds, const std::vector<Vec3>& rXyz) {
    Vec3 c0{0.0, 0.0, 0.0};
    std::size_t count = 0;
    for (const std::int64_t t : rIds)
        for (const std::int64_t v : rTris[static_cast<std::size_t>(t)]) {
            c0 = detail::vec3_add(c0, rXyz[static_cast<std::size_t>(v)]);
            ++count;
        }
    if (count == 0)
        return 0.0;
    c0 = detail::vec3_scale(c0, 1.0 / static_cast<double>(count));
    double vol = 0.0;
    for (const std::int64_t t : rIds) {
        const RepairTri& tri = rTris[static_cast<std::size_t>(t)];
        const Vec3 a = detail::vec3_sub(rXyz[static_cast<std::size_t>(tri[0])], c0);
        const Vec3 b = detail::vec3_sub(rXyz[static_cast<std::size_t>(tri[1])], c0);
        const Vec3 c = detail::vec3_sub(rXyz[static_cast<std::size_t>(tri[2])], c0);
        vol += detail::triple_product(a, b, c) / 6.0;
    }
    return vol;
}

}  // namespace

RepairResult repair(const Mesh& rMesh, const RepairOptions& rOptions) {
    repair_check_blocks(rMesh);
    RepairResult result;

    // --- phase 0: optional weld, through clean -----------------------------------
    Mesh welded;
    const Mesh* p_base = &rMesh;
    const std::size_t n_in = rMesh.NumPoints();
    if (rOptions.mWeldTolerance > 0.0) {
        CleanOptions co;
        co.weld = true;
        co.atol = rOptions.mWeldTolerance;
        co.remove_orphans = false;
        co.drop_degenerate = false;
        co.drop_duplicate_cells = false;
        CleanResult cr = clean(rMesh, co);
        welded = std::move(cr.mMesh);
        result.mPointMap = std::move(cr.mPointMap);
        result.mPointsWelded = cr.mPointsWelded;
        p_base = &welded;
    } else {
        result.mPointMap = NDArray::Uninit(DType::Int64, {n_in});
        std::int64_t* pm = result.mPointMap.As<std::int64_t>();
        for (std::size_t i = 0; i < n_in; ++i)
            pm[i] = static_cast<std::int64_t>(i);
    }
    const Mesh& base = *p_base;
    result.mQualityBefore = detail::soup_quality(detail::build_triangle_soup(base, ""));

    // --- phase 1: triangulate, blocks 1:1 ---------------------------------------
    ConvertCellsResult prep =
        convert_cells(base, {ConvertCellsMode::Simplexify, /*mRecordParentIds=*/true});
    const Mesh& simp = prep.mMesh;
    const std::size_t nblocks = simp.NumCellBlocks();
    const std::size_t n = simp.NumPoints();  // the ORIGINAL count; copies/centroids append
    const std::size_t dim = simp.PointDim();
    std::vector<Vec3> xyz(n);
    {
        const NDArray& points = simp.Points();
        parallel_for_bw(n, [&](std::size_t i) {
            xyz[i] = detail::read_point(points, dim, static_cast<std::int64_t>(i));
        });
    }
    // The flat triangle table: every 2-D block's rows, in block order.
    std::vector<RepairTri> tris;
    std::vector<std::size_t> block_tri_base(nblocks + 1, 0);
    std::vector<std::uint8_t> block_is_surface(nblocks, 0);
    {
        std::size_t bi = 0;
        for (const auto cb : simp.CellRange()) {
            block_tri_base[bi] = tris.size();
            const CellType ct = cell_type_from_name(std::string(cb.Type()));
            if (cell_type_dimension(ct) == 2) {
                if (ct != CellType::Triangle || cb.IsRagged())
                    throw std::invalid_argument(std::string(kRepairPrefix) +
                                                "internal error: simplexified block '" +
                                                std::string(cb.Type()) + "' is not triangles");
                block_is_surface[bi] = 1;
                const NDArray& conn = cb.Conn();
                for (std::size_t c = 0; c < cb.NumCells(); ++c)
                    tris.push_back({detail::read_int(conn, c * 3),
                                    detail::read_int(conn, c * 3 + 1),
                                    detail::read_int(conn, c * 3 + 2)});
            }
            ++bi;
        }
        block_tri_base[nblocks] = tris.size();
    }
    const std::size_t ntri_in = tris.size();
    for (const RepairTri& t : tris)
        for (const std::int64_t v : t)
            if (v < 0 || static_cast<std::size_t>(v) >= n)
                throw std::invalid_argument(std::string(kRepairPrefix) +
                                            "connectivity references point " + std::to_string(v) +
                                            " but the mesh has " + std::to_string(n) + " points");

    // --- phase 2: bowties ------------------------------------------------------------
    std::vector<std::int64_t> parent_of_new;  // per point beyond n: its source
    if (rOptions.mSplitNonManifold) {
        std::size_t n_split = n;
        result.mNumVerticesSplit = repair_split_bowties(tris, n_split, parent_of_new);
        for (const std::int64_t p : parent_of_new)
            xyz.push_back(xyz[static_cast<std::size_t>(p)]);
    }
    const std::size_t n_copies = parent_of_new.size();

    // --- phase 3: orientation BFS ------------------------------------------------------
    RepairEdges edges = repair_build_edges(tris);
    RepairOrientation orient = repair_orient(tris, edges);
    result.mNumComponents = static_cast<std::int64_t>(orient.mSize.size());
    for (std::size_t c = 0; c < orient.mSize.size(); ++c) {
        result.mLargestComponent = std::max(result.mLargestComponent, orient.mSize[c]);
        result.mNumUnorientable += orient.mUnorientable[c];
    }
    if (!rOptions.mFixOrientation)
        std::fill(orient.mFlip.begin(), orient.mFlip.end(), 0);
    parallel_for(ntri_in, [&](std::size_t t) {
        if (orient.mFlip[t])
            std::swap(tris[t][1], tris[t][2]);
    });

    // --- phase 4: holes -----------------------------------------------------------------
    std::vector<RepairHole> holes;
    std::vector<std::int64_t> fill_component;  // per fill triangle
    std::vector<std::int64_t> fill_hole;       // per fill triangle
    if (rOptions.mFillHoles) {
        RepairHoles found =
            repair_trace_holes(tris, edges, orient.mComponent, xyz.size(), rOptions.mMaxHoleEdges);
        result.mNumHolesDetected = found.mDetected;
        result.mNumHolesSkipped = found.mSkipped;
        result.mNumHolesFilled = static_cast<std::int64_t>(found.mFilled.size());
        holes = std::move(found.mFilled);
        for (std::size_t h = 0; h < holes.size(); ++h) {
            const std::vector<std::int64_t>& loop = holes[h].mLoop;
            Vec3 c{0.0, 0.0, 0.0};
            for (const std::int64_t v : loop)
                c = detail::vec3_add(c, xyz[static_cast<std::size_t>(v)]);
            c = detail::vec3_scale(c, 1.0 / static_cast<double>(loop.size()));
            const std::int64_t cid = static_cast<std::int64_t>(xyz.size());
            xyz.push_back(c);
            for (std::size_t i = 0; i < loop.size(); ++i) {
                tris.push_back({loop[i], loop[(i + 1) % loop.size()], cid});
                fill_component.push_back(holes[h].mComponent);
                fill_hole.push_back(static_cast<std::int64_t>(h));
            }
        }
        result.mNumFacesAdded = static_cast<std::int64_t>(tris.size() - ntri_in);
    }
    const std::size_t n_out = xyz.size();
    result.mNumPointsAdded = static_cast<std::int64_t>(n_out - n);
    const std::size_t ntri_out = tris.size();

    // --- phase 5: outward ----------------------------------------------------------------
    if (rOptions.mFixOrientation && rOptions.mOrientOutward && result.mNumComponents > 0) {
        const RepairEdges all_edges = repair_build_edges(tris);
        const std::size_t ncomp = orient.mSize.size();
        auto comp_of = [&](std::size_t t) {
            return t < ntri_in ? orient.mComponent[t] : fill_component[t - ntri_in];
        };
        std::vector<std::uint8_t> closed(ncomp, 1);
        for (std::size_t t = 0; t < ntri_out; ++t) {
            const std::int64_t c = comp_of(t);
            if (c < 0)
                continue;
            for (std::size_t e = 0; e < 3; ++e) {
                const std::int64_t edge = all_edges.mEdgeOf[t * 3 + e];
                if (edge < 0 || all_edges.Count(static_cast<std::size_t>(edge)) != 2)
                    closed[static_cast<std::size_t>(c)] = 0;
            }
        }
        std::vector<std::vector<std::int64_t>> members(ncomp);
        for (std::size_t t = 0; t < ntri_out; ++t) {
            const std::int64_t c = comp_of(t);
            if (c >= 0)
                members[static_cast<std::size_t>(c)].push_back(static_cast<std::int64_t>(t));
        }
        for (std::size_t c = 0; c < ncomp; ++c) {
            if (!closed[c] || orient.mUnorientable[c])
                continue;
            if (!(repair_signed_volume(tris, members[c], xyz) < 0.0))
                continue;
            ++result.mNumOrientedOutward;
            for (const std::int64_t t : members[c]) {
                std::swap(tris[static_cast<std::size_t>(t)][1],
                          tris[static_cast<std::size_t>(t)][2]);
                if (static_cast<std::size_t>(t) < ntri_in)
                    orient.mFlip[static_cast<std::size_t>(t)] ^= 1;
            }
        }
    }
    for (std::size_t t = 0; t < ntri_in; ++t)
        result.mNumFlipped += orient.mFlip[t];

    // --- phase 6: emit ---------------------------------------------------------------------
    Mesh& out = result.mMesh;
    {
        const NDArray& points = simp.Points();
        NDArray pts = NDArray::Uninit(points.Dtype(), {n_out, dim});
        detail::dispatch_dtype(points.Dtype(), [&]<class T>() {
            T* dst = pts.As<T>();
            parallel_for_bw(n_out, [&](std::size_t i) {
                for (std::size_t d = 0; d < dim; ++d)
                    dst[i * dim + d] = static_cast<T>(xyz[i][d]);
            });
        });
        out.AssignPoints(std::move(pts));
    }
    const bool has_parent = simp.HasCellData("convert:parent_cell");
    {
        std::vector<std::size_t> in_cells;
        for (const auto cb : base.CellRange())
            in_cells.push_back(cb.NumCells());
        std::size_t bi = 0;
        for (const auto cb : simp.CellRange()) {
            const std::size_t nc = cb.NumCells();
            if (block_is_surface[bi]) {
                NDArray block = NDArray::Uninit(DType::Int64, {nc, 3});
                std::int64_t* dst = block.As<std::int64_t>();
                const std::size_t tb = block_tri_base[bi];
                for (std::size_t c = 0; c < nc; ++c)
                    for (std::size_t k = 0; k < 3; ++k)
                        dst[c * 3 + k] = tris[tb + c][k];
                out.AddCellBlock(cell_type_name(CellType::Triangle), std::move(block));
            } else {
                out.AddCellBlock(std::string(cb.Type()), detail::data_owned_copy(cb.Conn()));
            }
            const NDArray* p_parent =
                has_parent ? &simp.CellData("convert:parent_cell", bi) : nullptr;
            std::vector<std::int64_t> cell_map(bi < in_cells.size() ? in_cells[bi] : nc, -1);
            for (std::size_t c = 0; c < nc; ++c) {
                const std::int64_t parent = p_parent != nullptr ? detail::read_int(*p_parent, c)
                                                                : static_cast<std::int64_t>(c);
                if (parent >= 0 && static_cast<std::size_t>(parent) < cell_map.size() &&
                    cell_map[static_cast<std::size_t>(parent)] < 0)
                    cell_map[static_cast<std::size_t>(parent)] = static_cast<std::int64_t>(c);
            }
            NDArray cm = NDArray::Uninit(DType::Int64, {cell_map.size()});
            std::memcpy(cm.Data(), cell_map.data(), cell_map.size() * sizeof(std::int64_t));
            result.mCellMaps.push_back(std::move(cm));
            ++bi;
        }
    }
    const std::size_t n_fill = ntri_out - ntri_in;
    if (n_fill > 0) {
        NDArray block = NDArray::Uninit(DType::Int64, {n_fill, 3});
        std::int64_t* dst = block.As<std::int64_t>();
        for (std::size_t c = 0; c < n_fill; ++c)
            for (std::size_t k = 0; k < 3; ++k)
                dst[c * 3 + k] = tris[ntri_in + c][k];
        out.AddCellBlock(cell_type_name(CellType::Triangle), std::move(block));
    }

    // cell_data: the input blocks' own rows (simplexify already replicated a
    // parent's row to its triangles); the fill block gets NaN / 0.
    for (const std::string& name : simp.CellDataNames()) {
        if (name == "convert:parent_cell")
            continue;
        const std::size_t ndata = simp.CellDataNumBlocks(name);
        std::vector<NDArray> blocks;
        blocks.reserve(ndata + 1);
        for (std::size_t b = 0; b < ndata; ++b)
            blocks.push_back(detail::data_owned_copy(simp.CellData(name, b)));
        if (n_fill > 0 && ndata == nblocks && ndata > 0) {
            const NDArray& a = simp.CellData(name, 0);
            std::vector<std::size_t> shape = a.Shape();
            if (shape.empty())
                shape.push_back(0);
            shape[0] = n_fill;
            NDArray fill(a.Dtype(), shape);
            if (detail::is_float_dtype(a.Dtype()))
                for (std::size_t i = 0; i < fill.Size(); ++i)
                    detail::write_double(fill, i, std::numeric_limits<double>::quiet_NaN());
            blocks.push_back(std::move(fill));
        }
        out.AddCellData(name, std::move(blocks));
    }

    // point_data: originals verbatim, copies from their source row, centroids
    // the mean of their loop's rows (dtype preserved).
    for (const std::string& name : simp.PointDataNames()) {
        const NDArray& a = simp.PointData(name);
        if (detail::rows(a) != n || n == 0) {
            out.AddPointData(name, detail::data_owned_copy(a));
            continue;
        }
        const std::size_t ncomp = a.Size() / n;
        std::vector<std::size_t> shape = a.Shape();
        shape[0] = n_out;
        NDArray b = NDArray::Uninit(a.Dtype(), std::move(shape));
        std::memcpy(b.Data(), a.Data(), a.Nbytes());
        const std::size_t row_bytes = a.Nbytes() / n;
        for (std::size_t k = 0; k < n_copies; ++k)
            std::memcpy(b.Data() + (n + k) * row_bytes,
                        a.Data() + static_cast<std::size_t>(parent_of_new[k]) * row_bytes,
                        row_bytes);
        auto src_row = [&](std::int64_t id) {
            return static_cast<std::size_t>(id) < n
                       ? static_cast<std::size_t>(id)
                       : static_cast<std::size_t>(parent_of_new[static_cast<std::size_t>(id) - n]);
        };
        for (std::size_t h = 0; h < holes.size(); ++h) {
            const std::vector<std::int64_t>& loop = holes[h].mLoop;
            const double inv = 1.0 / static_cast<double>(loop.size());
            for (std::size_t c = 0; c < ncomp; ++c) {
                double sum = 0.0;
                for (const std::int64_t v : loop)
                    sum += detail::read_double(a, src_row(v) * ncomp + c);
                detail::write_double(b, (n + n_copies + h) * ncomp + c, sum * inv);
            }
        }
        out.AddPointData(name, std::move(b));
    }
    for (const std::string& name : simp.FieldDataNames())
        out.AddFieldData(name, detail::data_owned_copy(simp.FieldData(name)));

    if (rOptions.mRecordProvenance) {
        NDArray pp = NDArray::Uninit(DType::Int64, {n_out});
        std::int64_t* dst = pp.As<std::int64_t>();
        for (std::size_t i = 0; i < n; ++i)
            dst[i] = static_cast<std::int64_t>(i);
        for (std::size_t k = 0; k < n_copies; ++k)
            dst[n + k] = parent_of_new[k];
        for (std::size_t h = 0; h < holes.size(); ++h)
            dst[n + n_copies + h] = -1;
        out.AddPointData(kRepairParentPointName, std::move(pp));
        std::vector<NDArray> blocks;
        for (std::size_t b = 0; b < nblocks; ++b) {
            const std::size_t nc = simp.Cells(b).NumCells();
            NDArray a = NDArray::Uninit(DType::Int64, {nc});
            std::int64_t* d = a.As<std::int64_t>();
            for (std::size_t c = 0; c < nc; ++c)
                d[c] = -1;
            blocks.push_back(std::move(a));
        }
        if (n_fill > 0) {
            NDArray a = NDArray::Uninit(DType::Int64, {n_fill});
            std::int64_t* d = a.As<std::int64_t>();
            for (std::size_t c = 0; c < n_fill; ++c)
                d[c] = fill_hole[c];
            blocks.push_back(std::move(a));
        }
        out.AddCellData(kRepairHoleName, std::move(blocks));
    }

    // Regions: FirstChild through the triangulation, points through the weld
    // map; Side regions drop by name. Then a split copy joins its source's
    // Point regions.
    {
        detail::RegionRemap rmap;
        rmap.pPointMap = &result.mPointMap;
        rmap.mCellMapKind = detail::CellMapKind::FirstChild;
        rmap.pCellMaps = &result.mCellMaps;
        rmap.mOpName = "repair";
        detail::remap_regions(rMesh, out, rmap);
    }
    if (n_copies > 0) {
        std::vector<meshioplusplus::Region> augmented;
        for (std::size_t i = 0; i < out.NumRegions(); ++i) {
            const meshioplusplus::Region& r = out.Region(i);
            if (r.mKind != RegionKind::Point)
                continue;
            std::vector<std::int64_t> entries(r.mEntries.Size());
            for (std::size_t e = 0; e < entries.size(); ++e)
                entries[e] = detail::read_int(r.mEntries, e);
            std::vector<std::int64_t> extra;
            for (std::size_t k = 0; k < n_copies; ++k)
                if (std::binary_search(entries.begin(), entries.end(), parent_of_new[k]))
                    extra.push_back(static_cast<std::int64_t>(n + k));
            if (extra.empty())
                continue;
            entries.insert(entries.end(), extra.begin(), extra.end());
            meshioplusplus::Region nr = r;
            nr.mEntries = NDArray::Uninit(DType::Int64, {entries.size()});
            std::memcpy(nr.mEntries.Data(), entries.data(), entries.size() * sizeof(std::int64_t));
            augmented.push_back(std::move(nr));
        }
        for (meshioplusplus::Region& r : augmented)
            out.AddRegion(std::move(r));
    }
    for (std::size_t i = 0; i < rMesh.NumPropertySets(); ++i)
        out.AddPropertySet(rMesh.GetPropertySet(i));

    result.mQualityAfter = detail::soup_quality(detail::build_triangle_soup(out, ""));
    return result;
}

}  // namespace meshioplusplus
