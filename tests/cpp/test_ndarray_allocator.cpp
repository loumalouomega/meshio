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

// The set_buffer_allocator hook: every owning NDArray buffer is allocated
// through the installed BufferAllocator (the doc/gpu.md pinned-memory
// enabler), each buffer keeps its allocator alive via its own shared_ptr
// (free-after-uninstall must still route to the recorded allocator), content
// is byte-identical to the default heap path, and the default path never
// touches the hook.

// System includes
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

// External includes
#include <gtest/gtest.h>

// Project includes
#include "mesh_fixtures.hpp"
#include "meshioplusplus/formats/vtu.hpp"
#include "meshioplusplus/ndarray.hpp"

namespace {

using meshioplusplus::BufferAllocator;
using meshioplusplus::DType;
using meshioplusplus::NDArray;
using meshioplusplus::set_buffer_allocator;

struct Counters {
    std::size_t allocs = 0;
    std::size_t frees = 0;
    std::size_t bytes_alloc = 0;
    std::size_t bytes_free = 0;
};

void* counting_alloc(std::size_t bytes, void* pUser) {
    Counters* c = static_cast<Counters*>(pUser);
    c->allocs += 1;
    c->bytes_alloc += bytes;
    return ::operator new(bytes);
}

void counting_free(void* pPtr, std::size_t bytes, void* pUser) {
    Counters* c = static_cast<Counters*>(pUser);
    c->frees += 1;
    c->bytes_free += bytes;
    ::operator delete(pPtr);
}

// Installs a counting allocator for the enclosing scope; always uninstalls.
class ScopedCountingAllocator {
public:
    explicit ScopedCountingAllocator(Counters& rCounters) {
        auto a = std::make_shared<BufferAllocator>();
        a->alloc = &counting_alloc;
        a->free = &counting_free;
        a->pUser = &rCounters;
        set_buffer_allocator(std::move(a));
    }
    ~ScopedCountingAllocator() { set_buffer_allocator(nullptr); }
};

}  // namespace

TEST(NDArrayAllocator, OwningAllocationsRouteThroughTheHook) {
    Counters c;
    {
        ScopedCountingAllocator guard(c);
        NDArray zeroed(DType::Float64, {8, 3});  // zeroing ctor
        EXPECT_EQ(c.allocs, 1u);
        EXPECT_EQ(c.bytes_alloc, 8u * 3u * 8u);
        for (std::size_t i = 0; i < 24; ++i)
            EXPECT_EQ(zeroed.As<double>()[i], 0.0);

        NDArray un = NDArray::Uninit(DType::Int64, {5});  // uninitialized ctor
        EXPECT_EQ(c.allocs, 2u);

        double ext[6] = {1, 2, 3, 4, 5, 6};
        NDArray view = NDArray::MakeView(DType::Float64, {3, 2}, reinterpret_cast<std::byte*>(ext));
        EXPECT_EQ(c.allocs, 2u);  // views allocate nothing
        view.MakeOwned();
        EXPECT_EQ(c.allocs, 3u);
        EXPECT_EQ(std::memcmp(view.Data(), ext, sizeof(ext)), 0);

        NDArray copy = zeroed;  // copies allocate through the current hook
        EXPECT_EQ(c.allocs, 4u);
        EXPECT_EQ(std::memcmp(copy.Data(), zeroed.Data(), copy.Nbytes()), 0);
    }
    EXPECT_EQ(c.allocs, c.frees);
    EXPECT_EQ(c.bytes_alloc, c.bytes_free);
}

TEST(NDArrayAllocator, AScalarAllocatesOneElementAndAnAbsentArrayNone) {
    // The allocation is what separates a rank-0 scalar from a default-
    // constructed array: they share an empty shape and nothing else tells them
    // apart, so `Size()` reads the buffer. Counted here rather than inferred.
    Counters c;
    {
        ScopedCountingAllocator guard(c);
        NDArray scalar(DType::Float64, {});
        EXPECT_EQ(c.allocs, 1u);
        EXPECT_EQ(c.bytes_alloc, 8u);
        EXPECT_EQ(scalar.Size(), 1u);

        NDArray absent;  // allocates nothing, and says so
        EXPECT_EQ(c.allocs, 1u);
        EXPECT_EQ(absent.Size(), 0u);

        double ext = 12345.678;
        NDArray view = NDArray::MakeView(DType::Float64, {}, reinterpret_cast<std::byte*>(&ext));
        EXPECT_EQ(c.allocs, 1u);  // views allocate nothing
        view.MakeOwned();         // ... until they are made owning
        EXPECT_EQ(c.allocs, 2u);
        EXPECT_EQ(c.bytes_alloc, 16u);
        EXPECT_EQ(view.As<double>()[0], 12345.678);
    }
    EXPECT_EQ(c.allocs, c.frees);
    EXPECT_EQ(c.bytes_alloc, c.bytes_free);
}

TEST(NDArrayAllocator, FreeAfterUninstallStillRoutesToTheRecordedAllocator) {
    Counters c;
    NDArray survivor;
    {
        ScopedCountingAllocator guard(c);
        survivor = NDArray::Uninit(DType::Float64, {100});
        EXPECT_EQ(c.allocs, 1u);
    }
    // Hook uninstalled; the buffer's own shared_ptr keeps the deleter alive.
    EXPECT_EQ(c.frees, 0u);
    survivor = NDArray();
    EXPECT_EQ(c.frees, 1u);
    EXPECT_EQ(c.bytes_free, 100u * 8u);
}

TEST(NDArrayAllocator, DefaultPathNeverTouchesAnUninstalledHook) {
    Counters c;
    {
        ScopedCountingAllocator guard(c);
    }  // installed and immediately removed
    NDArray a(DType::Float64, {16});
    NDArray b = a;
    (void)b;
    EXPECT_EQ(c.allocs, 0u);
    EXPECT_EQ(c.frees, 0u);
}

TEST(NDArrayAllocator, FullReaderPathIsByteIdenticalUnderTheHook) {
    // Round-trip a mesh with data: the read with the hook installed must
    // produce byte-identical arrays to the default-heap read.
    const meshioplusplus::Mesh mesh = mt::data_mesh();
    const std::string path = mt::temp_path(".vtu");
    meshioplusplus::write_vtu(path, mesh, /*binary=*/true, /*zlib=*/false);

    const meshioplusplus::Mesh plain = meshioplusplus::read_vtu(path);

    Counters c;
    {
        ScopedCountingAllocator guard(c);
        const meshioplusplus::Mesh hooked = meshioplusplus::read_vtu(path);
        EXPECT_GT(c.allocs, 0u);
        ASSERT_EQ(hooked.NumPoints(), plain.NumPoints());
        ASSERT_EQ(hooked.Points().Nbytes(), plain.Points().Nbytes());
        EXPECT_EQ(
            std::memcmp(hooked.Points().Data(), plain.Points().Data(), plain.Points().Nbytes()), 0);
        ASSERT_EQ(hooked.NumCellBlocks(), plain.NumCellBlocks());
        for (std::size_t b = 0; b < plain.NumCellBlocks(); ++b) {
            const auto pb = plain.Cells(b);
            const auto hb = hooked.Cells(b);
            ASSERT_EQ(hb.NumCells(), pb.NumCells());
            EXPECT_EQ(std::memcmp(hb.Conn().Data(), pb.Conn().Data(), pb.Conn().Nbytes()), 0);
        }
        for (const std::string& name : plain.PointDataNames()) {
            const NDArray& pa = plain.PointData(name);
            const NDArray& ha = hooked.PointData(name);
            ASSERT_EQ(ha.Nbytes(), pa.Nbytes());
            EXPECT_EQ(std::memcmp(ha.Data(), pa.Data(), pa.Nbytes()), 0);
        }
    }
    // `hooked` was destroyed inside the scope: everything must balance.
    EXPECT_EQ(c.allocs, c.frees);
    EXPECT_EQ(c.bytes_alloc, c.bytes_free);
    std::remove(path.c_str());
}
