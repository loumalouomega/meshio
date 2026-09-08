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
// Affine transform of point coordinates (translate / scale / rotate / general
// 4x4 matrix / unit scaling). Points change; connectivity and data are carried
// through (vector/tensor point_data optionally rotated). Built entirely through
// the uniform mesh API so it compiles under every mesh backend. See
// operations/transform.hpp for the contract.

// System includes
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Project includes
#include "meshioplusplus/operations/transform.hpp"
#include "meshioplusplus/detail/value_io.hpp"
#include "meshioplusplus/log.hpp"
#include "meshioplusplus/parallel.hpp"

namespace meshioplusplus {

namespace {

// A deep copy of an NDArray that always owns its buffer (the source may be a
// view over foreign memory, e.g. numpy memory on the write path).
NDArray transform_owned_copy(const NDArray& rArr) {
    NDArray c = rArr;
    c.MakeOwned();
    return c;
}

// Determinant of the 3x3 linear block of a row-major 4x4 matrix.
double transform_linear_det(const std::array<double, 16>& rM) {
    const double a = rM[0], b = rM[1], c = rM[2];
    const double d = rM[4], e = rM[5], f = rM[6];
    const double g = rM[8], h = rM[9], i = rM[10];
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

// Multiply the row-major 4x4 `rA` by `rB` (result = A * B).
std::array<double, 16> transform_matmul(const std::array<double, 16>& rA,
                                        const std::array<double, 16>& rB) {
    std::array<double, 16> out = {};
    for (std::size_t r = 0; r < 4; ++r)
        for (std::size_t c = 0; c < 4; ++c) {
            double s = 0.0;
            for (std::size_t k = 0; k < 4; ++k)
                s += rA[r * 4 + k] * rB[k * 4 + c];
            out[r * 4 + c] = s;
        }
    return out;
}

// Transform the point coordinates into a fresh array of the same dtype.
NDArray transform_apply_points(const NDArray& rPoints, std::size_t n, std::size_t dim,
                               const std::array<double, 16>& rM) {
    NDArray out = NDArray::Uninit(rPoints.Dtype(), {n, dim});
    if (n == 0 || dim == 0)
        return out;
    detail::dispatch_dtype(rPoints.Dtype(), [&]<class T>() {
        const T* src = rPoints.As<T>();
        T* dst = out.As<T>();
        parallel_for_bw(n, [&](std::size_t i) {
            const std::size_t base = i * dim;
            const double x = dim > 0 ? static_cast<double>(src[base + 0]) : 0.0;
            const double y = dim > 1 ? static_cast<double>(src[base + 1]) : 0.0;
            const double z = dim > 2 ? static_cast<double>(src[base + 2]) : 0.0;
            const double nx = rM[0] * x + rM[1] * y + rM[2] * z + rM[3];
            const double ny = rM[4] * x + rM[5] * y + rM[6] * z + rM[7];
            const double nz = rM[8] * x + rM[9] * y + rM[10] * z + rM[11];
            if (dim > 0)
                dst[base + 0] = static_cast<T>(nx);
            if (dim > 1)
                dst[base + 1] = static_cast<T>(ny);
            if (dim > 2)
                dst[base + 2] = static_cast<T>(nz);
        });
    });
    return out;
}

// Rotate vector (cols == 3) or tensor (cols == 9, row-major 3x3) rows of a
// float point_data array by the 3x3 linear block `rR` (v' = R v, A' = R A R^T).
// Returns a fresh array of the same dtype. Only Float32/Float64 arrays qualify.
NDArray transform_rotate_point_data(const NDArray& rArr, std::size_t n, std::size_t cols,
                                    const std::array<double, 9>& rR) {
    NDArray out = transform_owned_copy(rArr);
    detail::dispatch_dtype(rArr.Dtype(), [&]<class T>() {
        T* p = out.As<T>();
        if (cols == 3) {
            parallel_for_bw(n, [&](std::size_t i) {
                const std::size_t b = i * 3;
                const double v0 = static_cast<double>(p[b + 0]);
                const double v1 = static_cast<double>(p[b + 1]);
                const double v2 = static_cast<double>(p[b + 2]);
                p[b + 0] = static_cast<T>(rR[0] * v0 + rR[1] * v1 + rR[2] * v2);
                p[b + 1] = static_cast<T>(rR[3] * v0 + rR[4] * v1 + rR[5] * v2);
                p[b + 2] = static_cast<T>(rR[6] * v0 + rR[7] * v1 + rR[8] * v2);
            });
        } else {  // cols == 9 : A' = R A R^T
            parallel_for_bw(n, [&](std::size_t i) {
                const std::size_t b = i * 9;
                double a[9];
                for (std::size_t k = 0; k < 9; ++k)
                    a[k] = static_cast<double>(p[b + k]);
                // tmp = R * A
                double tmp[9];
                for (std::size_t r = 0; r < 3; ++r)
                    for (std::size_t c = 0; c < 3; ++c)
                        tmp[r * 3 + c] = rR[r * 3 + 0] * a[0 * 3 + c] +
                                         rR[r * 3 + 1] * a[1 * 3 + c] +
                                         rR[r * 3 + 2] * a[2 * 3 + c];
                // out = tmp * R^T
                for (std::size_t r = 0; r < 3; ++r)
                    for (std::size_t c = 0; c < 3; ++c)
                        p[b + r * 3 + c] = static_cast<T>(tmp[r * 3 + 0] * rR[c * 3 + 0] +
                                                          tmp[r * 3 + 1] * rR[c * 3 + 1] +
                                                          tmp[r * 3 + 2] * rR[c * 3 + 2]);
            });
        }
    });
    return out;
}

bool transform_is_float(DType dt) {
    return dt == DType::Float32 || dt == DType::Float64;
}

}  // namespace

AffineTransform transform_translation(double dx, double dy, double dz) {
    AffineTransform t;
    t.mMatrix[3] = dx;
    t.mMatrix[7] = dy;
    t.mMatrix[11] = dz;
    return t;
}

AffineTransform transform_scale(double sx, double sy, double sz) {
    AffineTransform t;
    t.mMatrix[0] = sx;
    t.mMatrix[5] = sy;
    t.mMatrix[10] = sz;
    return t;
}

AffineTransform transform_units(double factor) {
    return transform_scale(factor, factor, factor);
}

AffineTransform transform_rotation(double ax, double ay, double az, double angle_rad) {
    const double len = std::sqrt(ax * ax + ay * ay + az * az);
    if (len < 1e-300)
        throw std::invalid_argument("transform: rotation axis is zero-length");
    const double x = ax / len, y = ay / len, z = az / len;
    const double c = std::cos(angle_rad), s = std::sin(angle_rad), t = 1.0 - c;
    AffineTransform out;
    auto& m = out.mMatrix;
    m[0] = t * x * x + c;
    m[1] = t * x * y - s * z;
    m[2] = t * x * z + s * y;
    m[4] = t * x * y + s * z;
    m[5] = t * y * y + c;
    m[6] = t * y * z - s * x;
    m[8] = t * x * z - s * y;
    m[9] = t * y * z + s * x;
    m[10] = t * z * z + c;
    return out;
}

AffineTransform transform_from_matrix(const double* pMatrix16) {
    AffineTransform out;
    for (std::size_t i = 0; i < 16; ++i)
        out.mMatrix[i] = pMatrix16[i];
    return out;
}

AffineTransform transform_compose(const AffineTransform& rSecond, const AffineTransform& rFirst) {
    AffineTransform out;
    out.mMatrix = transform_matmul(rSecond.mMatrix, rFirst.mMatrix);
    return out;
}

Mesh transform(const Mesh& rMesh, const AffineTransform& rXform, bool rotate_vector_data) {
    const std::array<double, 16>& M = rXform.mMatrix;
    if (transform_linear_det(M) < 0.0)
        log::warn(
            "transform: orientation-reversing (negative determinant); cell orientation "
            "may be flipped");

    Mesh out;

    // Points (transformed, dtype preserved).
    const std::size_t n = rMesh.NumPoints();
    const std::size_t dim = rMesh.PointDim();
    out.AssignPoints(transform_apply_points(rMesh.Points(), n, dim, M));

    // Cells (clone; rectangular / polygon / polyhedron).
    for (const auto cb : rMesh.CellRange()) {
        if (cb.IsPolyhedron()) {
            std::vector<std::vector<std::vector<std::int64_t>>> cells(cb.NumCells());
            for (std::size_t c = 0; c < cb.NumCells(); ++c) {
                cells[c].resize(cb.NumFaces(c));
                for (std::size_t f = 0; f < cb.NumFaces(c); ++f) {
                    auto face = cb.Face(c, f);
                    cells[c][f].assign(face.first, face.first + face.second);
                }
            }
            out.AddPolyhedronBlock(std::string(cb.Type()), std::move(cells));
        } else if (cb.IsRagged()) {
            std::vector<std::vector<std::int64_t>> rows(cb.NumCells());
            for (std::size_t c = 0; c < cb.NumCells(); ++c)
                rows[c].assign(cb.Row(c), cb.Row(c) + cb.RowSize(c));
            out.AddPolygonBlock(std::string(cb.Type()), std::move(rows));
        } else {
            out.AddCellBlock(std::string(cb.Type()), transform_owned_copy(cb.Conn()));
        }
    }

    // Point data (optionally rotate vector/tensor float arrays).
    const std::array<double, 9> R = {M[0], M[1], M[2], M[4], M[5], M[6], M[8], M[9], M[10]};
    for (const std::string& name : rMesh.PointDataNames()) {
        const NDArray& a = rMesh.PointData(name);
        const std::size_t cols = n > 0 ? a.Size() / n : 0;
        if (rotate_vector_data && transform_is_float(a.Dtype()) && n > 0 &&
            (cols == 3 || cols == 9))
            out.AddPointData(name, transform_rotate_point_data(a, n, cols, R));
        else
            out.AddPointData(name, transform_owned_copy(a));
    }

    // Cell data (optionally rotated too -- a vector living on a cell rotates
    // exactly as one living on a point does; the block's row count is its own
    // cell count, so each block is widened independently).
    std::vector<std::size_t> cell_counts;
    for (const auto cb : rMesh.CellRange())
        cell_counts.push_back(cb.NumCells());
    for (const std::string& name : rMesh.CellDataNames()) {
        std::vector<NDArray> blocks;
        for (std::size_t b = 0; b < rMesh.CellDataNumBlocks(name); ++b) {
            const NDArray& a = rMesh.CellData(name, b);
            const std::size_t rows = b < cell_counts.size() ? cell_counts[b] : 0;
            const std::size_t cols = rows > 0 ? a.Size() / rows : 0;
            if (rotate_vector_data && transform_is_float(a.Dtype()) && rows > 0 &&
                (cols == 3 || cols == 9))
                blocks.push_back(transform_rotate_point_data(a, rows, cols, R));
            else
                blocks.push_back(transform_owned_copy(a));
        }
        out.AddCellData(name, std::move(blocks));
    }
    for (const std::string& name : rMesh.FieldDataNames())
        out.AddFieldData(name, transform_owned_copy(rMesh.FieldData(name)));

    // Named regions pass through verbatim: a transform moves coordinates and
    // renumbers nothing, so every point, cell and facet index still names the
    // same entity.
    for (std::size_t i = 0; i < rMesh.NumRegions(); ++i)
        out.AddRegion(rMesh.Region(i));
    // Property sets ride along too: these operations preserve the mesh's shape,
    // so dropping a deck's material data here would be new lossiness. They are
    // keyed by id, not by entity index, so there is nothing to remap.
    for (std::size_t i = 0; i < rMesh.NumPropertySets(); ++i)
        out.AddPropertySet(rMesh.GetPropertySet(i));

    return out;
}

}  // namespace meshioplusplus
