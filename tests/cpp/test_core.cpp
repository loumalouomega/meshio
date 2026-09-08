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

// System includes
#include <cstdint>

// External includes
#include <gtest/gtest.h>

// Project includes
#include "meshioplusplus/exceptions.hpp"
#include "meshioplusplus/mesh.hpp"
#include "meshioplusplus/ndarray.hpp"

using meshioplusplus::DType;
using meshioplusplus::Mesh;
using meshioplusplus::NDArray;

TEST(NDArray, OwningConstructionAndAccess) {
    NDArray a(DType::Float64, {2, 3});
    EXPECT_EQ(a.Ndim(), 2u);
    EXPECT_EQ(a.Size(), 6u);
    EXPECT_EQ(a.Nbytes(), 6u * 8u);
    EXPECT_FALSE(a.IsView());
    // zero-initialised
    for (std::size_t i = 0; i < a.Size(); ++i)
        EXPECT_EQ(a.As<double>()[i], 0.0);
    a.As<double>()[5] = 3.5;
    EXPECT_EQ(a.As<double>()[5], 3.5);
}

TEST(NDArray, IntegerDtypeSizes) {
    EXPECT_EQ(NDArray(DType::Int32, {4}).Nbytes(), 16u);
    EXPECT_EQ(NDArray(DType::Int64, {4}).Nbytes(), 32u);
    EXPECT_EQ(NDArray(DType::UInt8, {4}).Nbytes(), 4u);
    EXPECT_EQ(meshioplusplus::dtype_size(DType::Float32), 4u);
}

TEST(NDArray, Reshape) {
    NDArray a(DType::Int32, {2, 3});
    a.Reshape({3, 2});
    EXPECT_EQ(a.Shape()[0], 3u);
    EXPECT_EQ(a.Shape()[1], 2u);
    // inconsistent reshape is ignored
    a.Reshape({5, 5});
    EXPECT_EQ(a.Size(), 6u);
}

TEST(NDArray, ViewBecomesOwnedCopy) {
    std::vector<std::int64_t> buf = {1, 2, 3, 4};
    NDArray v = NDArray::MakeView(DType::Int64, {4}, reinterpret_cast<std::byte*>(buf.data()));
    EXPECT_TRUE(v.IsView());
    EXPECT_EQ(v.As<std::int64_t>()[2], 3);
    v.MakeOwned();
    EXPECT_FALSE(v.IsView());
    buf[2] = 99;  // mutating the original no longer affects the owned copy
    EXPECT_EQ(v.As<std::int64_t>()[2], 3);
}

// --------------------------------------------------------------------------- //
// Rank-0 (scalar) arrays. An empty shape means two things -- a scalar, and a
// default-constructed "absent" array -- and `Size()` separates them by whether
// a buffer exists. Both halves are load-bearing: `PropertyValue::IsText` and
// KratosMesh's `ConcatKind` sentinel read `Size() == 0` to mean absent, while
// every byte-copy path needs a scalar to count as one element or a 0-d value
// crossing from numpy is dropped on the first clone.
// --------------------------------------------------------------------------- //
TEST(NDArray, ScalarHoldsOneElement) {
    NDArray a(DType::Float64, {});
    EXPECT_EQ(a.Ndim(), 0u);
    EXPECT_EQ(a.Size(), 1u);
    EXPECT_EQ(a.Nbytes(), 8u);
    ASSERT_NE(a.Data(), nullptr);
    EXPECT_EQ(a.As<double>()[0], 0.0);  // the ctor still zero-fills
    a.As<double>()[0] = 12345.678;
    EXPECT_EQ(a.As<double>()[0], 12345.678);
}

TEST(NDArray, DefaultConstructedHoldsNothing) {
    // The "absent" state, and the reason Size() cannot simply return the empty
    // product: two callers read this as a sentinel.
    NDArray a;
    EXPECT_EQ(a.Ndim(), 0u);
    EXPECT_EQ(a.Size(), 0u);
    EXPECT_EQ(a.Nbytes(), 0u);
    EXPECT_EQ(a.Data(), nullptr);
}

TEST(NDArray, ScalarViewSurvivesMakeOwned) {
    // The exact path a 0-d numpy value takes into a cloned mesh: view in,
    // MakeOwned, value out. It used to copy zero bytes and leave a null
    // pointer behind, which pybind11 then turned into a fresh uninitialized
    // numpy scalar.
    double buf = 12345.678;
    NDArray v = NDArray::MakeView(DType::Float64, {}, reinterpret_cast<std::byte*>(&buf));
    EXPECT_TRUE(v.IsView());
    EXPECT_EQ(v.Size(), 1u);
    v.MakeOwned();
    EXPECT_FALSE(v.IsView());
    ASSERT_NE(v.Data(), nullptr);
    buf = -1.0;  // mutating the original no longer affects the owned copy
    EXPECT_EQ(v.As<double>()[0], 12345.678);
}

TEST(NDArray, ScalarReshapesToAndFromLengthOne) {
    NDArray a(DType::Int64, {1});
    a.As<std::int64_t>()[0] = 7;
    a.Reshape({});  // one element either way
    EXPECT_EQ(a.Ndim(), 0u);
    EXPECT_EQ(a.As<std::int64_t>()[0], 7);
    a.Reshape({1});
    EXPECT_EQ(a.Ndim(), 1u);
    EXPECT_EQ(a.As<std::int64_t>()[0], 7);
    a.Reshape({2});  // inconsistent, ignored
    EXPECT_EQ(a.Ndim(), 1u);
    EXPECT_EQ(a.Shape()[0], 1u);
}

TEST(NDArray, ScalarCopiesItsValue) {
    NDArray a(DType::Float64, {});
    a.As<double>()[0] = 12345.678;
    NDArray b = a;  // implicit copy ctor -> OwnedBuf deep copy
    a.As<double>()[0] = -1.0;
    EXPECT_EQ(b.As<double>()[0], 12345.678);
}

TEST(Mesh, CountsAndCellBlock) {
    Mesh m;
    m.AssignPoints(NDArray(DType::Float64, {5, 3}));
    EXPECT_EQ(m.NumPoints(), 5u);
    EXPECT_EQ(m.PointDim(), 3u);
    m.AddCellBlock("tetra", NDArray(DType::Int64, {2, 4}));
    EXPECT_EQ(m.NumCellBlocks(), 1u);
    EXPECT_EQ(m.Cells(0).Type(), "tetra");
    EXPECT_EQ(m.Cells(0).NumCells(), 2u);
    EXPECT_EQ(m.Cells(0).NodesPerCell(), 4u);
    Mesh empty;
    EXPECT_EQ(empty.NumPoints(), 0u);
    EXPECT_EQ(empty.NumCellBlocks(), 0u);
}

TEST(Mesh, RaggedCellBlocks) {
    Mesh m;
    m.AssignPoints(NDArray(DType::Float64, {8, 3}));

    // 1-level ragged (jagged polygon): cells with varying node counts.
    m.AddPolygonBlock("polygon", {{0, 1, 2}, {1, 2, 3, 4, 5}});
    const auto poly = m.Cells(0);
    EXPECT_TRUE(poly.IsRagged());
    EXPECT_FALSE(poly.IsPolyhedron());
    EXPECT_EQ(poly.NumCells(), 2u);
    EXPECT_EQ(poly.RowSize(1), 5u);
    EXPECT_EQ(poly.Row(0)[2], 2);

    // 2-level ragged (polyhedron): each cell is a list of faces.
    m.AddPolyhedronBlock("polyhedron4", {
                                            {{1, 2, 5}, {1, 2, 7}, {1, 5, 7}, {2, 5, 7}},
                                            {{2, 5, 6}, {2, 6, 7}, {2, 5, 7}, {5, 6, 7}},
                                        });
    const auto poh = m.Cells(1);
    EXPECT_TRUE(poh.IsRagged());
    EXPECT_TRUE(poh.IsPolyhedron());
    EXPECT_EQ(poh.NumCells(), 2u);
    EXPECT_EQ(poh.NumFaces(0), 4u);  // 4 faces
    const auto face = poh.Face(1, 3);
    EXPECT_EQ(face.second, 3u);
    EXPECT_EQ(face.first[0], 5);

    // A rectangular block is not ragged.
    m.AddCellBlock("triangle", NDArray(DType::Int64, {3, 3}));
    EXPECT_FALSE(m.Cells(2).IsRagged());
    EXPECT_EQ(m.Cells(2).NumCells(), 3u);
}

TEST(Exceptions, ReadWriteErrorMessages) {
    try {
        throw meshioplusplus::ReadError("boom-read");
    } catch (const meshioplusplus::ReadError& e) {
        EXPECT_STREQ(e.what(), "boom-read");
    }
    try {
        throw meshioplusplus::WriteError("boom-write");
    } catch (const std::runtime_error& e) {  // both derive from runtime_error
        EXPECT_STREQ(e.what(), "boom-write");
    }
    EXPECT_THROW(throw meshioplusplus::ReadError("x"), meshioplusplus::ReadError);
    EXPECT_THROW(throw meshioplusplus::WriteError("x"), meshioplusplus::WriteError);
}
