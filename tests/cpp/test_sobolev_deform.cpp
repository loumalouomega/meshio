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
// Tests for sobolev_deform. The correctness oracle is an INDEPENDENT dense
// reference: the P1 stiffness assembled by the cotangent formula (triangles)
// or 1/h (segments) -- a different formula than the Gram-matrix one the
// operation uses -- and a dense Gaussian elimination, so a sign or ordering
// slip in either the assembly or the matrix-free operator shows up here.

// System includes
#include <cmath>
#include <cstdint>
#include <vector>

// External includes
#include <gtest/gtest.h>

// Project includes
#include "meshioplusplus/operations/convert_cells.hpp"
#include "meshioplusplus/operations/sobolev_deform.hpp"
#include "meshioplusplus/operations/voxelize.hpp"
#include "mesh_fixtures.hpp"

using namespace meshioplusplus;

namespace {

using mt::plane_grid;

std::vector<double> coords(const Mesh& rMesh) {
    const NDArray& p = rMesh.Points();
    std::vector<double> out(p.Size());
    for (std::size_t i = 0; i < p.Size(); ++i)
        out[i] = detail::read_double(p, i);
    return out;
}

/// A polyline of N segments along x in 2-D, spacing h.
Mesh polyline(int N, double h) {
    std::vector<std::vector<double>> p;
    for (int i = 0; i <= N; ++i)
        p.push_back({h * i, 0.0});
    std::vector<std::vector<std::int64_t>> c;
    for (int i = 0; i < N; ++i)
        c.push_back({i, i + 1});
    return mt::make_mesh(p, "line", c);
}

/// A tetrahedral cube: an N^3 hex grid simplexified.
Mesh tet_cube(int N) {
    const Mesh hex = grid({N, N, N}, {0.0, 0.0, 0.0}, {1.0 / N, 1.0 / N, 1.0 / N});
    return convert_cells(hex, {ConvertCellsMode::Simplexify, false}).mMesh;
}

/// Attach a displacement array built from a per-point rule.
template <class F>
void set_displacement(Mesh& rMesh, const char* pName, F&& rF) {
    const std::size_t n = rMesh.NumPoints();
    const std::size_t dim = rMesh.PointDim();
    const std::vector<double> x = coords(rMesh);
    std::vector<double> d(n * dim);
    for (std::size_t i = 0; i < n; ++i) {
        const std::vector<double> v = rF(i, &x[i * dim]);
        for (std::size_t k = 0; k < dim; ++k)
            d[i * dim + k] = v[k];
    }
    rMesh.AddPointData(pName, mt::data_array(d, dim));
}

/// Dense reference solve of (mbar I + l^2 K) u = mbar d over the FREE points,
/// with K from the cotangent (triangles) / 1/h (segments) formula and mbar
/// the mean positive lumped mass. Returns u (n x dim), zero at fixed points.
std::vector<double> dense_reference(const Mesh& rMesh, const std::vector<double>& rD,
                                    double LengthScale, const std::vector<std::uint8_t>& rFixed) {
    const std::size_t n = rMesh.NumPoints();
    const std::size_t dim = rMesh.PointDim();
    const std::vector<double> x = coords(rMesh);
    std::vector<double> K(n * n, 0.0);
    std::vector<double> lumped(n, 0.0);
    auto pt = [&](std::int64_t v, std::size_t k) {
        return x[static_cast<std::size_t>(v) * dim + k];
    };
    for (const auto cb : rMesh.CellRange()) {
        const std::string type(cb.Type());
        const NDArray& conn = cb.Conn();
        if (type == "line") {
            for (std::size_t c = 0; c < cb.NumCells(); ++c) {
                const std::int64_t a = detail::read_int(conn, c * 2),
                                   b = detail::read_int(conn, c * 2 + 1);
                double h2 = 0.0;
                for (std::size_t k = 0; k < dim; ++k)
                    h2 += (pt(b, k) - pt(a, k)) * (pt(b, k) - pt(a, k));
                const double h = std::sqrt(h2);
                const double w = 1.0 / h;
                K[a * n + a] += w;
                K[b * n + b] += w;
                K[a * n + b] -= w;
                K[b * n + a] -= w;
                lumped[a] += h / 2.0;
                lumped[b] += h / 2.0;
            }
        } else if (type == "triangle") {
            for (std::size_t c = 0; c < cb.NumCells(); ++c) {
                std::int64_t v[3];
                for (int k = 0; k < 3; ++k)
                    v[k] = detail::read_int(conn, c * 3 + k);
                // Area and the three cotangents.
                double e[3][3] = {};
                for (int k = 0; k < 3; ++k) {
                    const std::int64_t p = v[(k + 1) % 3], q = v[(k + 2) % 3];
                    for (std::size_t a = 0; a < dim; ++a)
                        e[k][a] = pt(q, a) - pt(p, a);  // edge opposite corner k
                }
                auto dot = [&](const double* a, const double* b) {
                    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
                };
                const double cx = e[0][1] * e[1][2] - e[0][2] * e[1][1];
                const double cy = e[0][2] * e[1][0] - e[0][0] * e[1][2];
                const double cz = e[0][0] * e[1][1] - e[0][1] * e[1][0];
                const double twice_area = std::sqrt(cx * cx + cy * cy + cz * cz);
                for (int k = 0; k < 3; ++k) {
                    // cot of corner k = dot(-e_{k+1}, e_{k+2}) / |cross| with the
                    // two edges emanating from corner k.
                    const double* a = e[(k + 1) % 3];
                    const double* b = e[(k + 2) % 3];
                    const double cot = -dot(a, b) / twice_area;
                    const std::int64_t p = v[(k + 1) % 3], q = v[(k + 2) % 3];
                    const double w = 0.5 * cot;  // half per triangle
                    K[p * n + p] += w;
                    K[q * n + q] += w;
                    K[p * n + q] -= w;
                    K[q * n + p] -= w;
                }
                for (int k = 0; k < 3; ++k)
                    lumped[v[k]] += 0.5 * twice_area / 3.0;
            }
        }
    }
    double msum = 0.0;
    int mcount = 0;
    for (std::size_t i = 0; i < n; ++i)
        if (lumped[i] > 0.0) {
            msum += lumped[i];
            ++mcount;
        }
    const double mbar = msum / mcount;
    const double l2 = LengthScale * LengthScale;
    // A = mbar I + l2 K with Dirichlet rows/cols replaced by identity.
    std::vector<double> A(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A[i * n + j] = (i == j ? mbar : 0.0) + l2 * K[i * n + j];
    for (std::size_t i = 0; i < n; ++i)
        if (rFixed[i]) {
            for (std::size_t j = 0; j < n; ++j)
                A[i * n + j] = A[j * n + i] = 0.0;
            A[i * n + i] = 1.0;
        }
    std::vector<double> u(n * dim, 0.0);
    for (std::size_t k = 0; k < dim; ++k) {
        std::vector<double> M = A;
        std::vector<double> rhs(n);
        for (std::size_t i = 0; i < n; ++i)
            rhs[i] = rFixed[i] ? 0.0 : mbar * rD[i * dim + k];
        // Gaussian elimination with partial pivoting.
        for (std::size_t col = 0; col < n; ++col) {
            std::size_t piv = col;
            for (std::size_t r = col + 1; r < n; ++r)
                if (std::fabs(M[r * n + col]) > std::fabs(M[piv * n + col]))
                    piv = r;
            if (piv != col) {
                for (std::size_t j = 0; j < n; ++j)
                    std::swap(M[col * n + j], M[piv * n + j]);
                std::swap(rhs[col], rhs[piv]);
            }
            for (std::size_t r = col + 1; r < n; ++r) {
                const double f = M[r * n + col] / M[col * n + col];
                if (f == 0.0)
                    continue;
                for (std::size_t j = col; j < n; ++j)
                    M[r * n + j] -= f * M[col * n + j];
                rhs[r] -= f * rhs[col];
            }
        }
        for (std::size_t ii = n; ii-- > 0;) {
            double s = rhs[ii];
            for (std::size_t j = ii + 1; j < n; ++j)
                s -= M[ii * n + j] * u[j * dim + k];
            u[ii * dim + k] = s / M[ii * n + ii];
        }
    }
    return u;
}

std::vector<double> displacement_of(const Mesh& rIn, const Mesh& rOut) {
    const std::vector<double> a = coords(rIn), b = coords(rOut);
    std::vector<double> u(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        u[i] = b[i] - a[i];
    return u;
}

}  // namespace

TEST(SobolevDeform, AConstantDisplacementIsPreservedExactlyWithNothingFixed) {
    Mesh m = plane_grid(6);
    set_displacement(
        m, "d", [](std::size_t, const double*) { return std::vector<double>{0.3, -0.2, 0.7}; });
    SobolevOptions o;
    o.mArrayName = "d";
    o.mLengthScale = 2.0;
    const SobolevResult r = sobolev_deform(m, o);
    EXPECT_TRUE(r.mConverged);
    EXPECT_EQ(r.mNumIterations, 0);  // the initial guess IS the answer
    EXPECT_EQ(r.mNumFixed, 0);
    // Exactly x + d, entry for entry: the initial guess is the answer and the
    // write-back is one addition.
    const std::vector<double> a = coords(m), b = coords(r.mMesh);
    const double dconst[3] = {0.3, -0.2, 0.7};
    for (std::size_t i = 0; i < m.NumPoints(); ++i)
        for (std::size_t k = 0; k < 3; ++k)
            EXPECT_EQ(b[i * 3 + k], a[i * 3 + k] + dconst[k]);
}

TEST(SobolevDeform, LengthScaleZeroAppliesTheRawDisplacementBitExactly) {
    Mesh m = plane_grid(4);
    set_displacement(m, "d", [](std::size_t i, const double*) {
        return std::vector<double>{0.01 * i, -0.02 * i, 0.5 * ((i % 2) ? 1.0 : -1.0)};
    });
    SobolevOptions o;
    o.mArrayName = "d";
    o.mLengthScale = 0.0;
    const SobolevResult r = sobolev_deform(m, o);
    EXPECT_EQ(r.mNumIterations, 0);
    const std::vector<double> a = coords(m), b = coords(r.mMesh);
    const double* d = m.PointData("d").As<double>();
    for (std::size_t i = 0; i < a.size(); ++i)
        EXPECT_EQ(b[i], a[i] + d[i]);
    EXPECT_EQ(r.mMesh.Points().Dtype(), m.Points().Dtype());
}

TEST(SobolevDeform, FixedPointsDoNotMove) {
    Mesh m = plane_grid(5);
    const std::size_t n = m.NumPoints();
    set_displacement(m, "d", [](std::size_t, const double* x) {
        return std::vector<double>{0.0, 0.0, 0.2 * x[0]};
    });
    std::vector<std::int32_t> pin(n, 0);
    pin[7] = 1;
    m.AddPointData("pin", mt::int_data_array(pin));
    SobolevOptions o;
    o.mArrayName = "d";
    o.mLengthScale = 1.0;
    o.mFixedPoints.assign(n, 0);
    o.mFixedPoints[3] = 1;
    o.mFixedPointsArray = "pin";
    o.mFixBoundary = true;
    const SobolevResult r = sobolev_deform(m, o);
    EXPECT_TRUE(r.mConverged);
    const std::vector<double> a = coords(m), b = coords(r.mMesh);
    // The grid's boundary is the 20 rim points; 3 is on it, 7 is interior.
    EXPECT_EQ(r.mNumFixed, 21);
    for (std::size_t i = 0; i < n; ++i) {
        const bool rim =
            a[i * 3] == 0.0 || a[i * 3] == 5.0 || a[i * 3 + 1] == 0.0 || a[i * 3 + 1] == 5.0;
        if (rim || i == 3 || i == 7)
            for (std::size_t k = 0; k < 3; ++k)
                EXPECT_EQ(a[i * 3 + k], b[i * 3 + k]) << "point " << i;
    }
    // A mis-sized mask is refused by name.
    o.mFixedPoints.assign(n + 1, 0);
    EXPECT_THROW(sobolev_deform(m, o), std::invalid_argument);
}

TEST(SobolevDeform, MatchesADenseReferenceOnTrianglesAndSegments) {
    // Triangles in 3-D.
    {
        Mesh m = plane_grid(5);
        const std::size_t n = m.NumPoints();
        set_displacement(m, "d", [](std::size_t i, const double* x) {
            return std::vector<double>{0.1 * std::sin(1.7 * x[0]), 0.05 * (i % 3),
                                       0.3 * ((i % 2) ? 1.0 : -0.5)};
        });
        std::vector<std::uint8_t> fixed(n, 0);
        fixed[0] = fixed[n - 1] = 1;
        SobolevOptions o;
        o.mArrayName = "d";
        o.mLengthScale = 1.5;
        o.mFixedPoints = fixed;
        o.mTolerance = 1e-13;
        const SobolevResult r = sobolev_deform(m, o);
        ASSERT_TRUE(r.mConverged);
        EXPECT_GT(r.mNumIterations, 1);
        const std::vector<double> u = displacement_of(m, r.mMesh);
        const double* d = m.PointData("d").As<double>();
        const std::vector<double> ref =
            dense_reference(m, std::vector<double>(d, d + n * 3), 1.5, fixed);
        for (std::size_t i = 0; i < u.size(); ++i)
            EXPECT_NEAR(u[i], ref[i], 1e-9) << "entry " << i;
    }
    // Segments in 2-D: K is the 1/h tridiagonal.
    {
        Mesh m = polyline(12, 0.25);
        const std::size_t n = m.NumPoints();
        set_displacement(m, "d", [](std::size_t i, const double*) {
            return std::vector<double>{0.0, (i % 2) ? 0.2 : -0.2};
        });
        std::vector<std::uint8_t> fixed(n, 0);
        SobolevOptions o;
        o.mArrayName = "d";
        o.mLengthScale = 0.5;
        o.mTolerance = 1e-13;
        const SobolevResult r = sobolev_deform(m, o);
        ASSERT_TRUE(r.mConverged);
        const std::vector<double> u = displacement_of(m, r.mMesh);
        const double* d = m.PointData("d").As<double>();
        const std::vector<double> ref =
            dense_reference(m, std::vector<double>(d, d + n * 2), 0.5, fixed);
        for (std::size_t i = 0; i < u.size(); ++i)
            EXPECT_NEAR(u[i], ref[i], 1e-9) << "entry " << i;
    }
}

// The filter property: at l = 2h a checkerboard is damped far more than a
// linear field (which the natural Neumann boundary nearly preserves), and
// disabling the l^2 K term (the sabotage) leaves both at zero change.
TEST(SobolevDeform, DampsHighFrequenciesMoreThanLowOnes) {
    // Takes the mesh by const ref: the KRATOS backend's Mesh is not
    // copy-constructible.
    auto rel_change = [](const Mesh& m, const char* pName) {
        SobolevOptions o;
        o.mArrayName = pName;
        o.mLengthScale = 2.0;
        const SobolevResult r = sobolev_deform(m, o);
        EXPECT_TRUE(r.mConverged);
        const std::vector<double> u = displacement_of(m, r.mMesh);
        const double* d = m.PointData(pName).As<double>();
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < u.size(); ++i) {
            num += (u[i] - d[i]) * (u[i] - d[i]);
            den += d[i] * d[i];
        }
        return std::sqrt(num / den);
    };
    Mesh checker = plane_grid(10);
    set_displacement(checker, "d", [](std::size_t, const double* x) {
        const int i = static_cast<int>(x[0]), j = static_cast<int>(x[1]);
        return std::vector<double>{0.0, 0.0, ((i + j) % 2) ? 0.1 : -0.1};
    });
    Mesh linear = plane_grid(10);
    set_displacement(linear, "d", [](std::size_t, const double* x) {
        return std::vector<double>{0.0, 0.0, 0.02 * x[0]};
    });
    const double c = rel_change(checker, "d");
    const double l = rel_change(linear, "d");
    // The checkerboard is all but annihilated; the linear field is mostly
    // kept (what it loses is the Neumann boundary's flux term, O(l^2 grad)).
    EXPECT_GT(c, 0.9);
    EXPECT_LT(l, 0.2);
    EXPECT_GT(c, 4.0 * l);
}

TEST(SobolevDeform, RunsOnEveryDimensionAndRecordsTheFilteredField) {
    // Tet volume.
    Mesh vol = tet_cube(3);
    set_displacement(vol, "d", [](std::size_t i, const double* x) {
        return std::vector<double>{0.02 * ((i % 2) ? 1.0 : -1.0), 0.0, 0.01 * x[2]};
    });
    SobolevOptions o;
    o.mArrayName = "d";
    o.mLengthScale = 0.5;
    o.mRecordFiltered = true;
    const SobolevResult r = sobolev_deform(vol, o);
    EXPECT_TRUE(r.mConverged);
    EXPECT_GT(r.mNumIterations, 0);
    EXPECT_EQ(r.mNumIsolated, 0);
    ASSERT_TRUE(r.mMesh.HasPointData(kSobolevDisplacementName));
    const NDArray& f = r.mMesh.PointData(kSobolevDisplacementName);
    EXPECT_EQ(f.Shape()[1], 3u);
    const std::vector<double> u = displacement_of(vol, r.mMesh);
    const double* fd = f.As<double>();
    for (std::size_t i = 0; i < u.size(); ++i)
        EXPECT_NEAR(u[i], fd[i], 1e-14);
    EXPECT_GT(r.mMaxDisplacement, 0.0);
    EXPECT_EQ(std::string(r.mMesh.Cells(0).Type()), "tetra");
    // The z-column-ignored form on a 2-D mesh.
    Mesh line = polyline(4, 1.0);
    line.AddPointData("d3", mt::data_array(std::vector<double>(5 * 3, 0.1), 3));
    o.mArrayName = "d3";
    const SobolevResult r2 = sobolev_deform(line, o);
    EXPECT_EQ(r2.mMesh.PointDim(), 2u);
}

TEST(SobolevDeform, IsolatedPointsReceiveTheRawDisplacement) {
    Mesh m = mt::make_mesh({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {5, 5, 5}}, "triangle", {{0, 1, 2}});
    set_displacement(m, "d", [](std::size_t i, const double*) {
        return std::vector<double>{0.0, 0.0, i == 3 ? 0.9 : 0.1};
    });
    SobolevOptions o;
    o.mArrayName = "d";
    o.mLengthScale = 1.0;
    const SobolevResult r = sobolev_deform(m, o);
    EXPECT_EQ(r.mNumIsolated, 1);
    const std::vector<double> u = displacement_of(m, r.mMesh);
    EXPECT_NEAR(u[3 * 3 + 2], 0.9, 1e-12);
}

TEST(SobolevDeform, ReportsNonConvergenceInsteadOfThrowing) {
    Mesh checker = plane_grid(10);
    set_displacement(checker, "d", [](std::size_t, const double* x) {
        const int i = static_cast<int>(x[0]), j = static_cast<int>(x[1]);
        return std::vector<double>{0.0, 0.0, ((i + j) % 2) ? 0.1 : -0.1};
    });
    SobolevOptions o;
    o.mArrayName = "d";
    o.mLengthScale = 3.0;
    o.mMaxIterations = 1;
    const SobolevResult r = sobolev_deform(checker, o);
    EXPECT_FALSE(r.mConverged);
    EXPECT_EQ(r.mNumIterations, 1);
    EXPECT_GT(r.mResidual, 0.0);
    for (const double v : coords(r.mMesh))
        EXPECT_TRUE(std::isfinite(v));
}

TEST(SobolevDeform, RefusesNonSimplicesByName) {
    auto expect_msg = [](Mesh m, const char* pNeedle) {
        m.AddPointData("d", mt::data_array(std::vector<double>(m.NumPoints() * 3, 0.0), 3));
        SobolevOptions o;
        o.mArrayName = "d";
        o.mLengthScale = 1.0;
        try {
            sobolev_deform(m, o);
            FAIL() << "expected a throw";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find(pNeedle), std::string::npos) << e.what();
        }
    };
    expect_msg(mt::quad_mesh(), "convert_cells(simplexify)");
    expect_msg(mt::hex_mesh(), "convert_cells(simplexify)");
    expect_msg(mt::tet10_mesh(), "linearize");
    // Missing array, wrong shape.
    Mesh m = mt::tri_mesh();
    SobolevOptions o;
    o.mArrayName = "missing";
    EXPECT_THROW(sobolev_deform(m, o), std::invalid_argument);
    m.AddPointData("bad", mt::data_array(std::vector<double>(m.NumPoints(), 0.0)));
    o.mArrayName = "bad";
    EXPECT_THROW(sobolev_deform(m, o), std::invalid_argument);
}

TEST(SobolevDeform, DataRegionsAndDtypeSurvive) {
    Mesh m = convert_cells(mt::data_mesh(), {ConvertCellsMode::Simplexify, false}).mMesh;
    m.AddPointData("d", mt::data_array(std::vector<double>(m.NumPoints() * 3, 0.05), 3));
    SobolevOptions o;
    o.mArrayName = "d";
    o.mLengthScale = 0.3;
    const SobolevResult r = sobolev_deform(m, o);
    EXPECT_EQ(r.mMesh.Points().Dtype(), m.Points().Dtype());
    EXPECT_EQ(r.mMesh.PointDataNames(), m.PointDataNames());
    EXPECT_EQ(r.mMesh.CellDataNames(), m.CellDataNames());
    EXPECT_EQ(r.mMesh.NumRegions(), m.NumRegions());
    EXPECT_EQ(r.mMesh.NumCellBlocks(), m.NumCellBlocks());
}
