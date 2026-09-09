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
// Shrinkwrap: one projection of every (selected) source point onto the target
// surface, through the same nearest-triangle search the signed distance uses.
// See operations/shrinkwrap.hpp for the contract and the one deliberate
// divergence from upstream (the feature pseudonormal).

// System includes
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Project includes
#include "meshioplusplus/operations/shrinkwrap.hpp"
#include "meshioplusplus/detail/data_ops.hpp"
#include "meshioplusplus/detail/geometry.hpp"
#include "meshioplusplus/detail/surface_distance.hpp"
#include "meshioplusplus/detail/value_io.hpp"
#include "meshioplusplus/log.hpp"
#include "meshioplusplus/operations/data_common.hpp"
#include "meshioplusplus/parallel.hpp"

namespace meshioplusplus {
namespace {

using detail::Vec3;

constexpr const char* kSwrapPrefix = "meshio++: shrinkwrap: ";

// Write the leading `dim` columns of a flat (n, 3) double buffer back out,
// preserving the source dtype (smooth.cpp's write-back, same shape).
NDArray swrap_write_coords(const NDArray& rPoints, const std::vector<Vec3>& rXyz, std::size_t n,
                           std::size_t dim) {
    NDArray out = NDArray::Uninit(rPoints.Dtype(), {n, dim});
    if (n == 0 || dim == 0)
        return out;
    detail::dispatch_dtype(rPoints.Dtype(), [&]<class T>() {
        T* dst = out.As<T>();
        parallel_for_bw(n, [&](std::size_t i) {
            for (std::size_t d = 0; d < dim && d < 3; ++d)
                dst[i * dim + d] = static_cast<T>(rXyz[i][d]);
        });
    });
    return out;
}

// The per-point weight: 1 everywhere, or the named array's value (a float
// array blends, an integer/bool one selects).
std::vector<double> swrap_weights(const Mesh& rMesh, const std::string& rName, std::size_t n) {
    std::vector<double> w(n, 1.0);
    if (rName.empty())
        return w;
    if (!rMesh.HasPointData(rName))
        throw std::invalid_argument(std::string(kSwrapPrefix) +
                                    data_unknown_key_message(rMesh, DataLocation::Point, rName));
    const NDArray& a = rMesh.PointData(rName);
    if (detail::rows(a) != n || (n > 0 && a.Size() != n))
        throw std::invalid_argument(std::string(kSwrapPrefix) + "weights array '" + rName +
                                    "' must be a scalar per point (" + std::to_string(n) +
                                    " rows, 1 component)");
    const bool is_float = detail::is_float_dtype(a.Dtype());
    parallel_for_bw(n, [&](std::size_t i) {
        w[i] = is_float ? detail::read_double(a, i) : (detail::read_int(a, i) != 0 ? 1.0 : 0.0);
    });
    return w;
}

}  // namespace

ShrinkwrapResult shrinkwrap(const Mesh& rMesh, const Mesh& rTarget,
                            const ShrinkwrapOptions& rOptions) {
    const std::size_t n = rMesh.NumPoints();
    const std::size_t dim = rMesh.PointDim();
    if (dim < 2 || dim > 3)
        throw std::invalid_argument(std::string(kSwrapPrefix) + "points must be 2-D or 3-D");

    // --- the target: soup, quality, accelerator ------------------------------
    detail::TriangleSoup soup;
    try {
        soup = detail::build_triangle_soup(rTarget, rOptions.mTargetRegion);
    } catch (const std::invalid_argument& e) {
        std::string what = e.what();
        const std::string kernel = "meshio++: surface distance: ";
        if (what.rfind(kernel, 0) == 0)
            what = what.substr(kernel.size());
        throw std::invalid_argument(std::string(kSwrapPrefix) + "target: " + what);
    }
    if (soup.NumTriangles() == 0)
        throw std::invalid_argument(std::string(kSwrapPrefix) +
                                    "target: the surface has no triangles to project onto");

    ShrinkwrapResult result;
    result.mQuality = detail::soup_quality(soup);
    if (!result.mQuality.mWatertight)
        log::warn(
            "{}the target is not watertight: {} boundary edge(s), {} non-manifold edge(s), {} "
            "inconsistently wound pair(s), {} degenerate triangle(s) -- a nonzero offset may "
            "point to different sides near the defects",
            kSwrapPrefix, result.mQuality.mBoundaryEdges, result.mQuality.mNonManifoldEdges,
            result.mQuality.mInconsistentPairs, result.mQuality.mDegenerateTriangles);

    SurfaceDistanceOptions sd_opts;
    sd_opts.mWeight = rOptions.mNormalWeight;
    sd_opts.mGridCellSize = rOptions.mGridCellSize;
    const detail::DistanceQuery query = detail::build_distance_query(soup, sd_opts);

    // --- the source: coordinates and the selection ---------------------------
    const std::vector<double> w = swrap_weights(rMesh, rOptions.mWeights, n);
    std::vector<Vec3> xyz(n);
    {
        const NDArray& points = rMesh.Points();
        parallel_for_bw(n, [&](std::size_t i) {
            xyz[i] = detail::read_point(points, dim, static_cast<std::int64_t>(i));
        });
    }
    std::vector<std::size_t> moving;  // serial, ascending: the query order
    moving.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        if (w[i] != 0.0)
            moving.push_back(i);
    result.mNumSkipped = static_cast<std::int64_t>(n - moving.size());

    std::vector<Vec3> qpts(moving.size());
    parallel_for_bw(moving.size(), [&](std::size_t k) { qpts[k] = xyz[moving[k]]; });
    const std::vector<detail::SurfaceProjection> hits =
        detail::query_surface_projections(query, qpts);

    // --- the move ------------------------------------------------------------
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> distance(n, nan);
    std::vector<std::int64_t> closest(n, -1);
    std::vector<std::uint8_t> missed(moving.size(), 0);
    std::vector<double> disp(moving.size(), 0.0);
    const double maxd = rOptions.mMaxDistance;
    const double offset = rOptions.mOffset;
    parallel_for(moving.size(), [&](std::size_t k) {
        const std::size_t i = moving[k];
        const detail::SurfaceProjection& h = hits[k];
        const Vec3 x = xyz[i];
        if (!h.mFound) {
            missed[k] = 1;
            return;
        }
        distance[i] = h.mDistance;
        closest[i] = h.mSourceCell;
        if (maxd > 0.0 && h.mDistance > maxd) {
            missed[k] = 1;
            return;
        }
        Vec3 target = h.mPoint;
        if (offset != 0.0) {
            const double nn = detail::vec3_norm(h.mNormal);
            if (!(nn > 0.0)) {
                missed[k] = 1;
                return;
            }
            // Order is load-bearing for the numpy twin: offset / |n| first,
            // then scale, then add.
            target = detail::vec3_add(target, detail::vec3_scale(h.mNormal, offset / nn));
        }
        const Vec3 moved =
            detail::vec3_add(x, detail::vec3_scale(detail::vec3_sub(target, x), w[i]));
        xyz[i] = moved;
        disp[k] = detail::vec3_norm(detail::vec3_sub(moved, x));
    });

    // Serial folds, ascending.
    for (std::size_t k = 0; k < moving.size(); ++k) {
        if (missed[k])
            ++result.mNumMissed;
        else
            ++result.mNumProjected;
        if (disp[k] > result.mMaxDisplacement)
            result.mMaxDisplacement = disp[k];
    }

    // --- output ---------------------------------------------------------------
    result.mMesh = detail::clone_mesh(rMesh);
    result.mMesh.AssignPoints(swrap_write_coords(rMesh.Points(), xyz, n, dim));
    if (rOptions.mRecordDistance) {
        NDArray a(DType::Float64, {n});
        double* dst = a.As<double>();
        for (std::size_t i = 0; i < n; ++i)
            dst[i] = distance[i];
        result.mMesh.AddPointData(kShrinkwrapDistanceName, std::move(a));
    }
    if (rOptions.mRecordClosestCell) {
        NDArray a(DType::Int64, {n});
        std::int64_t* dst = a.As<std::int64_t>();
        for (std::size_t i = 0; i < n; ++i)
            dst[i] = closest[i];
        result.mMesh.AddPointData(kShrinkwrapClosestCellName, std::move(a));
    }
    return result;
}

}  // namespace meshioplusplus
