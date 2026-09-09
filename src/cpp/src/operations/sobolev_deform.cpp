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
// Sobolev deformation: a screened-Poisson filter of a displacement field over
// the mesh's own P1 operators, solved matrix-free by Jacobi-preconditioned
// conjugate gradients. See operations/sobolev_deform.hpp for the contract and
// for what is matched to upstream on purpose.

// System includes
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Project includes
#include "meshioplusplus/operations/sobolev_deform.hpp"
#include "meshioplusplus/cell_type.hpp"
#include "meshioplusplus/detail/data_ops.hpp"
#include "meshioplusplus/detail/value_io.hpp"
#include "meshioplusplus/log.hpp"
#include "meshioplusplus/operations/data_common.hpp"
#include "meshioplusplus/parallel.hpp"

namespace meshioplusplus {
namespace {

constexpr const char* kSoboPrefix = "meshio++: sobolev_deform: ";

/// Fixed chunk for the inner products: a constant, never derived from the
/// thread count, so the fold order is the same on every machine.
constexpr std::size_t kSoboChunk = 4096;

/// The top-dimensional simplices, flattened: `mConn` is `(ncells, D+1)`.
struct SoboCells {
    int mDim = 0;
    std::size_t mNumCells = 0;
    std::vector<std::int64_t> mConn;
};

/// Reject, by name, everything outside the simplex scope, and gather the
/// top-dimensional cells.
SoboCells sobo_gather_cells(const Mesh& rMesh) {
    int top = -1;
    for (const auto cb : rMesh.CellRange()) {
        if (cb.IsPolyhedron())
            throw std::invalid_argument(
                std::string(kSoboPrefix) +
                "mesh contains a polyhedron cell block; the P1 assembly needs simplices "
                "(run convert_cells(simplexify) first)");
        const int d = cell_type_dimension(cell_type_from_name(std::string(cb.Type())));
        top = std::max(top, d);
    }
    if (top < 1)
        throw std::invalid_argument(std::string(kSoboPrefix) +
                                    "mesh has no cells of dimension 1, 2 or 3 to assemble on");

    SoboCells out;
    out.mDim = top;
    const std::size_t nv = static_cast<std::size_t>(top) + 1;
    for (const auto cb : rMesh.CellRange()) {
        const std::string type(cb.Type());
        const CellType ct = cell_type_from_name(type);
        if (cell_type_dimension(ct) != top)
            continue;  // lower-dimensional blocks ride along untouched
        const bool simplex =
            ct == CellType::Line || ct == CellType::Triangle || ct == CellType::Tetra;
        if (!simplex) {
            const bool quadratic =
                ct == CellType::Line3 || ct == CellType::Triangle6 || ct == CellType::Tetra10;
            throw std::invalid_argument(
                std::string(kSoboPrefix) + "cell block '" + type + "' is not a linear simplex; " +
                (quadratic ? "linearize the mesh first" : "run convert_cells(simplexify) first"));
        }
        if (cb.IsRagged())
            throw std::invalid_argument(std::string(kSoboPrefix) + "cell block '" + type +
                                        "' is ragged");
        const std::size_t nc = cb.NumCells();
        const NDArray& conn = cb.Conn();
        const std::size_t base = out.mConn.size();
        out.mConn.resize(base + nc * nv);
        for (std::size_t c = 0; c < nc * nv; ++c)
            out.mConn[base + c] = detail::read_int(conn, c);
        out.mNumCells += nc;
    }
    return out;
}

/// Vertex -> (cell, local corner) incidence, CSR, each row in ascending cell
/// order (a counting sort over pairs generated in ascending (cell, corner)).
struct SoboStar {
    std::vector<std::size_t> mOffsets;
    std::vector<std::int64_t> mCell;
    std::vector<std::uint8_t> mLocal;
};

SoboStar sobo_build_star(const SoboCells& rCells, std::size_t n) {
    const std::size_t nv = static_cast<std::size_t>(rCells.mDim) + 1;
    SoboStar s;
    s.mOffsets.assign(n + 1, 0);
    for (std::size_t c = 0; c < rCells.mNumCells; ++c)
        for (std::size_t k = 0; k < nv; ++k) {
            const std::int64_t v = rCells.mConn[c * nv + k];
            if (v < 0 || static_cast<std::size_t>(v) >= n)
                throw std::invalid_argument(std::string(kSoboPrefix) + "cell " + std::to_string(c) +
                                            " references point " + std::to_string(v) +
                                            " but the mesh has " + std::to_string(n) + " points");
            ++s.mOffsets[static_cast<std::size_t>(v) + 1];
        }
    for (std::size_t i = 0; i < n; ++i)
        s.mOffsets[i + 1] += s.mOffsets[i];
    s.mCell.resize(s.mOffsets[n]);
    s.mLocal.resize(s.mOffsets[n]);
    std::vector<std::size_t> fill(s.mOffsets.begin(), s.mOffsets.end() - 1);
    for (std::size_t c = 0; c < rCells.mNumCells; ++c)
        for (std::size_t k = 0; k < nv; ++k) {
            const std::size_t v = static_cast<std::size_t>(rCells.mConn[c * nv + k]);
            s.mCell[fill[v]] = static_cast<std::int64_t>(c);
            s.mLocal[fill[v]] = static_cast<std::uint8_t>(k);
            ++fill[v];
        }
    return s;
}

/// The boundary points of the top-dimensional cells: every point on a facet
/// used by exactly one cell. Facets of a D-simplex are its D+1 sub-simplices
/// omitting one corner; a sorted, -1-padded key is what makes two cells agree.
std::vector<std::uint8_t> sobo_boundary_points(const SoboCells& rCells, std::size_t n) {
    const std::size_t nv = static_cast<std::size_t>(rCells.mDim) + 1;
    using Key = std::array<std::int64_t, 3>;
    std::vector<Key> keys;
    keys.reserve(rCells.mNumCells * nv);
    for (std::size_t c = 0; c < rCells.mNumCells; ++c)
        for (std::size_t omit = 0; omit < nv; ++omit) {
            Key key{-1, -1, -1};
            std::size_t w = 0;
            for (std::size_t k = 0; k < nv; ++k)
                if (k != omit)
                    key[w++] = rCells.mConn[c * nv + k];
            std::sort(key.begin(), key.begin() + static_cast<std::ptrdiff_t>(w));
            keys.push_back(key);
        }
    std::sort(keys.begin(), keys.end());
    std::vector<std::uint8_t> boundary(n, 0);
    for (std::size_t i = 0; i < keys.size();) {
        std::size_t j = i + 1;
        while (j < keys.size() && keys[j] == keys[i])
            ++j;
        if (j - i == 1)
            for (const std::int64_t v : keys[i])
                if (v >= 0)
                    boundary[static_cast<std::size_t>(v)] = 1;
        i = j;
    }
    return boundary;
}

/// Per-cell local stiffness `K_loc = |c| B G^-1 B^T` (row-major, `nv x nv`)
/// and measure `|c| = sqrt(det G) / D!`, from the edge Gram matrix
/// `G = E E^T`, `E_k = x_k - x_0`. The closed forms are written out per
/// dimension so the operation order is fixed. Throws on a degenerate cell.
void sobo_assemble(const SoboCells& rCells, const std::vector<double>& rXyz, std::size_t dim,
                   std::vector<double>& rKloc, std::vector<double>& rMeasure) {
    const int D = rCells.mDim;
    const std::size_t nv = static_cast<std::size_t>(D) + 1;
    rKloc.assign(rCells.mNumCells * nv * nv, 0.0);
    rMeasure.assign(rCells.mNumCells, 0.0);
    std::vector<std::uint8_t> bad(rCells.mNumCells, 0);

    parallel_for(rCells.mNumCells, [&](std::size_t c) {
        // Edge vectors relative to corner 0, in the mesh's own ambient dim.
        double E[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        const std::size_t v0 = static_cast<std::size_t>(rCells.mConn[c * nv]);
        for (int k = 1; k <= D; ++k) {
            const std::size_t vk = static_cast<std::size_t>(rCells.mConn[c * nv + k]);
            for (std::size_t a = 0; a < dim; ++a)
                E[k - 1][a] = rXyz[vk * dim + a] - rXyz[v0 * dim + a];
        }
        // Gram matrix and its inverse, closed form per D.
        double G[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        for (int i = 0; i < D; ++i)
            for (int j = 0; j < D; ++j) {
                double s = 0.0;
                for (std::size_t a = 0; a < dim; ++a)
                    s += E[i][a] * E[j][a];
                G[i][j] = s;
            }
        double det = 0.0;
        double Gi[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        double factorial = 1.0;
        if (D == 1) {
            det = G[0][0];
            if (!(std::isfinite(det) && det > 0.0)) {
                bad[c] = 1;
                return;
            }
            Gi[0][0] = 1.0 / det;
        } else if (D == 2) {
            det = G[0][0] * G[1][1] - G[0][1] * G[1][0];
            if (!(std::isfinite(det) && det > 0.0)) {
                bad[c] = 1;
                return;
            }
            Gi[0][0] = G[1][1] / det;
            Gi[0][1] = -G[0][1] / det;
            Gi[1][0] = -G[1][0] / det;
            Gi[1][1] = G[0][0] / det;
            factorial = 2.0;
        } else {
            const double c00 = G[1][1] * G[2][2] - G[1][2] * G[2][1];
            const double c01 = -(G[1][0] * G[2][2] - G[1][2] * G[2][0]);
            const double c02 = G[1][0] * G[2][1] - G[1][1] * G[2][0];
            det = G[0][0] * c00 + G[0][1] * c01 + G[0][2] * c02;
            if (!(std::isfinite(det) && det > 0.0)) {
                bad[c] = 1;
                return;
            }
            const double c10 = -(G[0][1] * G[2][2] - G[0][2] * G[2][1]);
            const double c11 = G[0][0] * G[2][2] - G[0][2] * G[2][0];
            const double c12 = -(G[0][0] * G[2][1] - G[0][1] * G[2][0]);
            const double c20 = G[0][1] * G[1][2] - G[0][2] * G[1][1];
            const double c21 = -(G[0][0] * G[1][2] - G[0][2] * G[1][0]);
            const double c22 = G[0][0] * G[1][1] - G[0][1] * G[1][0];
            // inverse = adjugate / det; adjugate = transpose of cofactors.
            Gi[0][0] = c00 / det;
            Gi[0][1] = c10 / det;
            Gi[0][2] = c20 / det;
            Gi[1][0] = c01 / det;
            Gi[1][1] = c11 / det;
            Gi[1][2] = c21 / det;
            Gi[2][0] = c02 / det;
            Gi[2][1] = c12 / det;
            Gi[2][2] = c22 / det;
            factorial = 6.0;
        }
        const double measure = std::sqrt(det) / factorial;
        rMeasure[c] = measure;
        // B = [-1^T ; I_D]  ((D+1) x D): B[0][a] = -1, B[i][a] = (i-1 == a).
        // K[i][j] = |c| * sum_ab B[i][a] Gi[a][b] B[j][b].
        double* K = &rKloc[c * nv * nv];
        for (std::size_t i = 0; i < nv; ++i)
            for (std::size_t j = 0; j < nv; ++j) {
                double s = 0.0;
                for (int a = 0; a < D; ++a) {
                    const double bi = i == 0 ? -1.0 : (static_cast<int>(i) - 1 == a ? 1.0 : 0.0);
                    if (bi == 0.0)
                        continue;
                    for (int b = 0; b < D; ++b) {
                        const double bj =
                            j == 0 ? -1.0 : (static_cast<int>(j) - 1 == b ? 1.0 : 0.0);
                        if (bj == 0.0)
                            continue;
                        s += bi * Gi[a][b] * bj;
                    }
                }
                K[i * nv + j] = measure * s;
            }
    });
    for (std::size_t c = 0; c < rCells.mNumCells; ++c)
        if (bad[c])
            throw std::invalid_argument(
                std::string(kSoboPrefix) + "top-dimensional cell " + std::to_string(c) +
                " is degenerate (non-positive Gram determinant); run clean(drop_degenerate) first");
}

/// Inner product of two `(n, dim)` flat vectors: fixed-chunk parallel
/// partials folded serially, the `accumulate_stats` idiom.
double sobo_dot(const std::vector<double>& rA, const std::vector<double>& rB, std::size_t n,
                std::size_t dim) {
    const std::size_t nchunks = (n + kSoboChunk - 1) / kSoboChunk;
    std::vector<double> partial(nchunks, 0.0);
    parallel_for(
        nchunks,
        [&](std::size_t ci) {
            const std::size_t lo = ci * kSoboChunk;
            const std::size_t hi = std::min(n, lo + kSoboChunk);
            double s = 0.0;
            for (std::size_t i = lo; i < hi; ++i)
                for (std::size_t d = 0; d < dim; ++d)
                    s += rA[i * dim + d] * rB[i * dim + d];
            partial[ci] = s;
        },
        1);
    double total = 0.0;
    for (const double p : partial)
        total += p;
    return total;
}

}  // namespace

SobolevResult sobolev_deform(const Mesh& rMesh, const SobolevOptions& rOptions) {
    const std::size_t n = rMesh.NumPoints();
    const std::size_t dim = rMesh.PointDim();
    if (dim < 1 || dim > 3)
        throw std::invalid_argument(std::string(kSoboPrefix) + "points must be 1-D, 2-D or 3-D");
    if (rOptions.mArrayName.empty())
        throw std::invalid_argument(std::string(kSoboPrefix) +
                                    "a displacement array name is required");
    if (!rMesh.HasPointData(rOptions.mArrayName))
        throw std::invalid_argument(
            std::string(kSoboPrefix) +
            data_unknown_key_message(rMesh, DataLocation::Point, rOptions.mArrayName));
    if (!(rOptions.mLengthScale >= 0.0) || !std::isfinite(rOptions.mLengthScale))
        throw std::invalid_argument(std::string(kSoboPrefix) + "length_scale must be >= 0");
    if (rOptions.mMaxIterations < 0)
        throw std::invalid_argument(std::string(kSoboPrefix) + "max_iterations must be >= 0");
    if (!(rOptions.mTolerance > 0.0))
        throw std::invalid_argument(std::string(kSoboPrefix) + "tolerance must be > 0");
    if (!rOptions.mFixedPoints.empty() && rOptions.mFixedPoints.size() != n)
        throw std::invalid_argument(std::string(kSoboPrefix) + "fixed_points has " +
                                    std::to_string(rOptions.mFixedPoints.size()) +
                                    " entries but the mesh has " + std::to_string(n) + " points");

    // --- the raw displacement d, (n, dim) ---------------------------------------
    const NDArray& darr = rMesh.PointData(rOptions.mArrayName);
    const std::size_t drows = detail::rows(darr);
    const std::size_t dcomp = drows == 0 ? 0 : darr.Size() / drows;
    if (drows != n || !(dcomp == dim || (dim == 2 && dcomp == 3)))
        throw std::invalid_argument(std::string(kSoboPrefix) + "displacement array '" +
                                    rOptions.mArrayName + "' must be (" + std::to_string(n) + ", " +
                                    std::to_string(dim) + ")" +
                                    (dim == 2 ? " or (n, 3) with the z column ignored" : ""));
    if (dcomp != dim)
        log::warn("{}'{}' has 3 components on a 2-D mesh; the z column is ignored", kSoboPrefix,
                  rOptions.mArrayName);
    std::vector<double> d(n * dim, 0.0);
    parallel_for_bw(n, [&](std::size_t i) {
        for (std::size_t k = 0; k < dim; ++k)
            d[i * dim + k] = detail::read_double(darr, i * dcomp + k);
    });

    // --- the cells, the star, the pins -----------------------------------------
    const SoboCells cells = sobo_gather_cells(rMesh);
    const SoboStar star = sobo_build_star(cells, n);
    std::vector<std::uint8_t> fixed(n, 0);
    if (!rOptions.mFixedPoints.empty())
        for (std::size_t i = 0; i < n; ++i)
            fixed[i] = rOptions.mFixedPoints[i] ? 1 : 0;
    if (!rOptions.mFixedPointsArray.empty()) {
        if (!rMesh.HasPointData(rOptions.mFixedPointsArray))
            throw std::invalid_argument(
                std::string(kSoboPrefix) +
                data_unknown_key_message(rMesh, DataLocation::Point, rOptions.mFixedPointsArray));
        const NDArray& a = rMesh.PointData(rOptions.mFixedPointsArray);
        if (detail::rows(a) != n || (n > 0 && a.Size() != n))
            throw std::invalid_argument(std::string(kSoboPrefix) + "fixed_points array '" +
                                        rOptions.mFixedPointsArray +
                                        "' must be a scalar per point");
        for (std::size_t i = 0; i < n; ++i)
            if (detail::read_double(a, i) != 0.0)
                fixed[i] = 1;
    }
    if (rOptions.mFixBoundary) {
        const std::vector<std::uint8_t> boundary = sobo_boundary_points(cells, n);
        for (std::size_t i = 0; i < n; ++i)
            if (boundary[i])
                fixed[i] = 1;
    }

    SobolevResult result;
    for (std::size_t i = 0; i < n; ++i) {
        if (fixed[i])
            ++result.mNumFixed;
        if (star.mOffsets[i + 1] == star.mOffsets[i])
            ++result.mNumIsolated;
    }

    // --- the filtered displacement u -------------------------------------------
    std::vector<double> u(n * dim, 0.0);
    const double l2 = rOptions.mLengthScale * rOptions.mLengthScale;
    if (rOptions.mLengthScale == 0.0) {
        parallel_for_bw(n, [&](std::size_t i) {
            if (fixed[i])
                return;
            for (std::size_t k = 0; k < dim; ++k)
                u[i * dim + k] = d[i * dim + k];
        });
    } else {
        // Coordinates as a flat (n, dim) double buffer.
        std::vector<double> xyz(n * dim, 0.0);
        {
            const NDArray& points = rMesh.Points();
            parallel_for_bw(n, [&](std::size_t i) {
                for (std::size_t k = 0; k < dim; ++k)
                    xyz[i * dim + k] = detail::read_double(points, i * dim + k);
            });
        }
        std::vector<double> kloc;
        std::vector<double> measure;
        sobo_assemble(cells, xyz, dim, kloc, measure);
        const std::size_t nv = static_cast<std::size_t>(cells.mDim) + 1;

        // Lumped mass and stiffness diagonal, gathered per vertex in ascending
        // cell order; then the uniform mean mass.
        std::vector<double> lumped(n, 0.0);
        std::vector<double> diagk(n, 0.0);
        parallel_for(n, [&](std::size_t i) {
            double m = 0.0;
            double dk = 0.0;
            for (std::size_t s = star.mOffsets[i]; s < star.mOffsets[i + 1]; ++s) {
                const std::size_t c = static_cast<std::size_t>(star.mCell[s]);
                const std::size_t k = star.mLocal[s];
                m += measure[c] / static_cast<double>(nv);
                dk += kloc[c * nv * nv + k * nv + k];
            }
            lumped[i] = m;
            diagk[i] = dk;
        });
        double mass_sum = 0.0;
        std::int64_t mass_count = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (lumped[i] > 0.0) {
                mass_sum += lumped[i];
                ++mass_count;
            }
        const double mbar = mass_count > 0 ? mass_sum / static_cast<double>(mass_count) : 1.0;

        // The operator, gather form: y_i = mbar x_i + l^2 sum_star K[k][j] x_j
        // over FREE j; identity on a fixed row. One thread per vertex, fixed
        // order, no scatter.
        auto apply = [&](const std::vector<double>& rX, std::vector<double>& rY) {
            parallel_for(n, [&](std::size_t i) {
                if (fixed[i]) {
                    for (std::size_t kk = 0; kk < dim; ++kk)
                        rY[i * dim + kk] = rX[i * dim + kk];
                    return;
                }
                double acc[3] = {0.0, 0.0, 0.0};
                for (std::size_t kk = 0; kk < dim; ++kk)
                    acc[kk] = mbar * rX[i * dim + kk];
                for (std::size_t s = star.mOffsets[i]; s < star.mOffsets[i + 1]; ++s) {
                    const std::size_t c = static_cast<std::size_t>(star.mCell[s]);
                    const std::size_t k = star.mLocal[s];
                    const double* K = &kloc[c * nv * nv + k * nv];
                    for (std::size_t j = 0; j < nv; ++j) {
                        const std::size_t vj = static_cast<std::size_t>(cells.mConn[c * nv + j]);
                        if (fixed[vj])
                            continue;
                        const double w = l2 * K[j];
                        for (std::size_t kk = 0; kk < dim; ++kk)
                            acc[kk] += w * rX[vj * dim + kk];
                    }
                }
                for (std::size_t kk = 0; kk < dim; ++kk)
                    rY[i * dim + kk] = acc[kk];
            });
        };

        const double tiny = std::numeric_limits<double>::min();
        std::vector<double> b(n * dim, 0.0);
        std::vector<double> diag(n, 1.0);
        parallel_for_bw(n, [&](std::size_t i) {
            if (fixed[i])
                return;
            for (std::size_t kk = 0; kk < dim; ++kk)
                b[i * dim + kk] = mbar * d[i * dim + kk];
            const double dg = mbar + l2 * diagk[i];
            diag[i] = dg > tiny ? dg : tiny;
        });
        const double b2 = sobo_dot(b, b, n, dim);

        if (b2 > 0.0) {
            // Initial guess: the raw displacement at the free points.
            std::vector<double> x(n * dim, 0.0);
            parallel_for_bw(n, [&](std::size_t i) {
                if (fixed[i])
                    return;
                for (std::size_t kk = 0; kk < dim; ++kk)
                    x[i * dim + kk] = d[i * dim + kk];
            });
            std::vector<double> r(n * dim), z(n * dim), p(n * dim), q(n * dim);
            apply(x, q);
            parallel_for_bw(n, [&](std::size_t i) {
                for (std::size_t kk = 0; kk < dim; ++kk) {
                    r[i * dim + kk] = b[i * dim + kk] - q[i * dim + kk];
                    z[i * dim + kk] = r[i * dim + kk] / diag[i];
                    p[i * dim + kk] = z[i * dim + kk];
                }
            });
            double rz = sobo_dot(r, z, n, dim);
            double r2 = sobo_dot(r, r, n, dim);
            const double thr = rOptions.mTolerance * rOptions.mTolerance * (b2 > tiny ? b2 : tiny);
            bool breakdown = false;
            std::int64_t it = 0;
            while (r2 > thr && it < rOptions.mMaxIterations) {
                apply(p, q);
                const double pq = sobo_dot(p, q, n, dim);
                if (!(std::isfinite(rz) && std::isfinite(pq) && pq > tiny)) {
                    breakdown = true;
                    break;
                }
                const double alpha = rz / pq;
                parallel_for_bw(n, [&](std::size_t i) {
                    for (std::size_t kk = 0; kk < dim; ++kk) {
                        x[i * dim + kk] += alpha * p[i * dim + kk];
                        r[i * dim + kk] -= alpha * q[i * dim + kk];
                        z[i * dim + kk] = r[i * dim + kk] / diag[i];
                    }
                });
                const double rz_new = sobo_dot(r, z, n, dim);
                r2 = sobo_dot(r, r, n, dim);
                ++it;
                if (!std::isfinite(rz_new)) {
                    breakdown = true;
                    break;
                }
                const double beta = rz_new / (rz > tiny ? rz : tiny);
                parallel_for_bw(n, [&](std::size_t i) {
                    for (std::size_t kk = 0; kk < dim; ++kk)
                        p[i * dim + kk] = z[i * dim + kk] + beta * p[i * dim + kk];
                });
                rz = rz_new;
            }
            result.mNumIterations = it;
            result.mResidual = std::sqrt(r2 / b2);
            result.mConverged = (r2 <= thr) && !breakdown;
            if (!result.mConverged)
                log::warn(
                    "{}conjugate gradients did not converge in {} iteration(s) (relative "
                    "residual {}); the last iterate is returned -- raise max_iterations or "
                    "lower length_scale",
                    kSoboPrefix, it, result.mResidual);
            parallel_for_bw(n, [&](std::size_t i) {
                if (fixed[i])
                    return;
                for (std::size_t kk = 0; kk < dim; ++kk)
                    u[i * dim + kk] = x[i * dim + kk];
            });
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::size_t kk = 0; kk < dim; ++kk)
            s += u[i * dim + kk] * u[i * dim + kk];
        const double mag = std::sqrt(s);
        if (mag > result.mMaxDisplacement)
            result.mMaxDisplacement = mag;
    }

    // --- output ---------------------------------------------------------------------
    result.mMesh = detail::clone_mesh(rMesh);
    {
        const NDArray& points = rMesh.Points();
        NDArray moved = NDArray::Uninit(points.Dtype(), {n, dim});
        detail::dispatch_dtype(points.Dtype(), [&]<class T>() {
            T* dst = moved.As<T>();
            parallel_for_bw(n, [&](std::size_t i) {
                for (std::size_t kk = 0; kk < dim; ++kk)
                    dst[i * dim + kk] =
                        static_cast<T>(detail::read_double(points, i * dim + kk) + u[i * dim + kk]);
            });
        });
        result.mMesh.AssignPoints(std::move(moved));
    }
    if (rOptions.mRecordFiltered) {
        NDArray a = NDArray::Uninit(DType::Float64, {n, dim});
        double* dst = a.As<double>();
        for (std::size_t i = 0; i < n * dim; ++i)
            dst[i] = u[i];
        result.mMesh.AddPointData(kSobolevDisplacementName, std::move(a));
    }
    return result;
}

}  // namespace meshioplusplus
