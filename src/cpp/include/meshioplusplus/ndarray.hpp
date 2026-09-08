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
 * @file ndarray.hpp
 * @brief `NDArray`: a minimal typed, n-dimensional, contiguous (row-major)
 * array — the storage primitive of `meshioplusplus::Mesh`.
 *
 * `NDArray` is used for points, cell connectivity, and every point/cell/field
 * data array. It either *owns* its buffer (the common case: data produced by
 * a reader) or is a non-owning *view* over externally-owned memory (used to
 * wrap a numpy buffer zero-copy on the write path — see `py_to_mesh` in
 * `bindings/python/np_conversions.hpp`). The binding layer converts between
 * `NDArray` and numpy at the I/O boundary: owning buffers are moved into a
 * capsule backing a writeable numpy array on read, and numpy buffers are
 * wrapped as views (no copy) on write. `Dtype()` records the element type
 * with an internal `DType` enum rather than a template parameter, so
 * `NDArray` can be stored uniformly (e.g. in `Mesh::mCellData`) regardless of
 * the numpy dtype it came from; `As<T>()` reinterprets the raw buffer as `T`
 * once the caller has determined (or asserted) the appropriate type.
 */

// System includes
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace meshioplusplus {

/**
 * @brief A caller-supplied allocator for `NDArray`'s *owning* buffers.
 *
 * Plain C-style callbacks (no `std::function`) so a hook can be installed
 * from any binding layer. `alloc` must return a buffer of at least `bytes`
 * bytes (uninitialized) or `nullptr` on failure (surfaced as
 * `std::bad_alloc`); `free` receives the same pointer, byte count, and
 * `pUser` back. This is the GPU-pipeline enabler recorded in `doc/gpu.md`:
 * install a pinned-memory (e.g. CUDA page-locked) allocator and every array
 * a reader produces lands directly in pinned memory, removing the staging
 * copy of a later host->device transfer.
 */
struct BufferAllocator {
    void* (*alloc)(std::size_t bytes, void* pUser);
    void (*free)(void* pPtr, std::size_t bytes, void* pUser);
    void* pUser;
};

namespace detail {
inline std::mutex& buffer_allocator_mutex() {
    static std::mutex m;
    return m;
}
inline std::shared_ptr<const BufferAllocator>& buffer_allocator_storage() {
    static std::shared_ptr<const BufferAllocator> a;
    return a;
}
/** @brief The currently installed hook (may be null = default heap). */
inline std::shared_ptr<const BufferAllocator> current_buffer_allocator() {
    std::lock_guard<std::mutex> lk(buffer_allocator_mutex());
    return buffer_allocator_storage();
}
}  // namespace detail

/**
 * @brief Installs (or, with `nullptr`, removes) the process-global allocator
 * used for every subsequently created *owning* `NDArray` buffer.
 *
 * Consulted only at allocation time: each buffer keeps its own
 * `shared_ptr` reference to the allocator it was born with, so buffers
 * outlive any later `set_buffer_allocator` call structurally — the hook's
 * `free` stays reachable until the last buffer allocated through it is
 * destroyed. Consequences worth knowing: uninstalling never orphans live
 * buffers, and a *copy* of an array allocates through the allocator current
 * at copy time (pinned-ness does not propagate through copies). Views
 * (`MakeView`) are unaffected — they own nothing.
 * @param pAllocator The hook to install, or `nullptr` to restore the default
 *                   heap (`::operator new`/`delete`).
 */
inline void set_buffer_allocator(std::shared_ptr<const BufferAllocator> pAllocator) {
    std::lock_guard<std::mutex> lk(detail::buffer_allocator_mutex());
    detail::buffer_allocator_storage() = std::move(pAllocator);
}

namespace detail {
/**
 * @brief `NDArray`'s owning byte buffer: `(pointer, size, allocator ref)`.
 *
 * Replaces the former `std::vector`-based buffer so each buffer can carry
 * the `BufferAllocator` it was allocated through (the `shared_ptr` is what
 * makes the deleter outlive install/uninstall windows). Value semantics are
 * preserved: copying deep-copies through the *currently installed* hook,
 * moving steals the pointer + allocator reference. Memory is always left
 * uninitialized on allocation (the old `NoInitAllocator` behaviour —
 * `NDArray`'s zeroing constructor memsets explicitly); `resize` is
 * allocate-exact, since `NDArray` only ever sizes a buffer once.
 *
 * @note `data`/`size`/`resize` keep their `std::vector` spellings (rather
 * than the PascalCase convention) so `NDArray`'s call sites are unchanged.
 */
class OwnedBuf {
public:
    OwnedBuf() = default;
    ~OwnedBuf() { FreeBuf(); }
    OwnedBuf(const OwnedBuf& rOther) { CopyFrom(rOther); }
    OwnedBuf& operator=(const OwnedBuf& rOther) {
        if (this != &rOther) {
            FreeBuf();
            CopyFrom(rOther);
        }
        return *this;
    }
    OwnedBuf(OwnedBuf&& rOther) noexcept
        : mPtr(rOther.mPtr), mSize(rOther.mSize), mAllocator(std::move(rOther.mAllocator)) {
        rOther.mPtr = nullptr;
        rOther.mSize = 0;
    }
    OwnedBuf& operator=(OwnedBuf&& rOther) noexcept {
        if (this != &rOther) {
            FreeBuf();
            mPtr = rOther.mPtr;
            mSize = rOther.mSize;
            mAllocator = std::move(rOther.mAllocator);
            rOther.mPtr = nullptr;
            rOther.mSize = 0;
        }
        return *this;
    }
    void resize(std::size_t n) {
        if (n == mSize)
            return;
        FreeBuf();
        if (n == 0)
            return;
        mAllocator = current_buffer_allocator();
        void* p = (mAllocator && mAllocator->alloc) ? mAllocator->alloc(n, mAllocator->pUser)
                                                    : ::operator new(n);
        if (p == nullptr)
            throw std::bad_alloc{};
        mPtr = static_cast<std::byte*>(p);
        mSize = n;
    }
    std::byte* data() noexcept { return mPtr; }
    const std::byte* data() const noexcept { return mPtr; }
    std::size_t size() const noexcept { return mSize; }

private:
    void FreeBuf() noexcept {
        if (mPtr != nullptr) {
            if (mAllocator && mAllocator->free)
                mAllocator->free(mPtr, mSize, mAllocator->pUser);
            else
                ::operator delete(mPtr);
        }
        mPtr = nullptr;
        mSize = 0;
        mAllocator.reset();
    }
    void CopyFrom(const OwnedBuf& rOther) {
        resize(rOther.mSize);
        if (rOther.mSize != 0)
            std::memcpy(mPtr, rOther.mPtr, rOther.mSize);
    }

    std::byte* mPtr = nullptr;
    std::size_t mSize = 0;
    std::shared_ptr<const BufferAllocator> mAllocator;
};
}  // namespace detail

/**
 * @brief Scalar element type of an `NDArray`, mirroring the numpy dtypes the
 * binding layer converts to/from.
 */
enum class DType {
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
};

/**
 * @brief Size in bytes of one element of the given dtype.
 * @param dt The dtype to query.
 * @return 1, 2, 4, or 8, matching the C++ scalar type `dt` represents.
 */
inline std::size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32:
            return 4;
        case DType::Float64:
            return 8;
        case DType::Int8:
        case DType::UInt8:
            return 1;
        case DType::Int16:
        case DType::UInt16:
            return 2;
        case DType::Int32:
        case DType::UInt32:
            return 4;
        case DType::Int64:
        case DType::UInt64:
            return 8;
    }
    return 0;
}

/**
 * @brief numpy dtype string (kind + itemsize) for a `DType`, e.g. `"f8"`, `"i4"`.
 * @param dt The dtype to convert.
 * @return A numpy-style struct format code understood by `numpy.dtype(...)`.
 */
inline const char* dtype_numpy_str(DType dt) {
    switch (dt) {
        case DType::Float32:
            return "f4";
        case DType::Float64:
            return "f8";
        case DType::Int8:
            return "i1";
        case DType::Int16:
            return "i2";
        case DType::Int32:
            return "i4";
        case DType::Int64:
            return "i8";
        case DType::UInt8:
            return "u1";
        case DType::UInt16:
            return "u2";
        case DType::UInt32:
            return "u4";
        case DType::UInt64:
            return "u8";
    }
    return "f8";
}

/**
 * @brief A minimal typed, n-dimensional, row-major contiguous array.
 *
 * `NDArray` is either *owning* (holds its own `detail::OwnedBuf`, freed on
 * destruction — allocated through the `set_buffer_allocator` hook when one
 * is installed) or a non-owning *view* over externally-managed memory
 * (`mView != nullptr`); `IsView()` distinguishes the two, and `Data()`
 * transparently returns whichever buffer is active. Views exist so the
 * write path can wrap a numpy array's memory directly (see
 * `bindings/python/np_conversions.hpp`'s `py_to_mesh`) without copying it into a
 * C++-owned buffer; `MakeOwned()` is the escape hatch for turning a view
 * into an owning copy when a buffer must outlive the memory it points to.
 * There is no reference counting: a view's caller is responsible for
 * keeping the underlying memory alive for the `NDArray`'s lifetime.
 */
class NDArray {
public:
    NDArray() = default;

    /**
     * @brief Constructs an owning array with a zero-initialized buffer.
     * @param dt Element dtype.
     * @param shape Row-major dimensions; total element count is their product.
     */
    NDArray(DType dt, std::vector<std::size_t> shape) : mDtype(dt), mShape(std::move(shape)) {
        // ShapeCount, not Nbytes(): no buffer exists yet, and `Size()` reports
        // 0 for a rank-0 array until one does (see its doc comment). Asking it
        // here would allocate nothing for a scalar and then hand out a null
        // pointer for its single element.
        const std::size_t nb = ShapeCount(mShape) * dtype_size(mDtype);
        mOwned.resize(nb);                  // uninitialised (OwnedBuf)
        std::memset(mOwned.data(), 0, nb);  // explicit zero-fill
    }

    /**
     * @brief Constructs an owning array whose buffer is left *uninitialized*.
     *
     * Only safe for callers that immediately overwrite every byte — typical
     * uses are reader outputs (the whole buffer is about to be filled from
     * the parsed file) and cell-block reconstruction (e.g.
     * `detail::reconstruct_cells` in `vtk_cells.hpp`). Skips both the extra
     * allocator zero-fill and, more importantly, the cold first-touch page
     * faults a `memset` would otherwise incur on a fresh large allocation —
     * the same optimization numpy applies to its own `calloc`-avoidance path.
     * Prefer the two-argument constructor whenever the buffer might not be
     * fully overwritten.
     *
     * @param dt Element dtype.
     * @param shape Row-major dimensions; total element count is their product.
     * @return A new owning, uninitialized `NDArray`.
     */
    static NDArray Uninit(DType dt, std::vector<std::size_t> shape) {
        NDArray a;
        a.mDtype = dt;
        a.mShape = std::move(shape);
        a.mOwned.resize(ShapeCount(a.mShape) * dtype_size(a.mDtype));  // no memset
        return a;
    }

    /**
     * @brief Constructs a non-owning view over externally-owned row-major memory.
     *
     * Used to wrap a numpy array's buffer directly at the write boundary
     * (zero-copy): the C++ writer reads through `pPtr` but never frees it.
     * @param dt Element dtype of the memory at `pPtr`.
     * @param shape Row-major dimensions describing how to interpret `pPtr`.
     * @param pPtr Pointer to caller-owned memory; the caller must keep it
     *            alive for at least the lifetime of the returned `NDArray`
     *            (and of any `NDArray` copies/moves derived from it that
     *            remain a view).
     * @return A new non-owning `NDArray` view.
     */
    static NDArray MakeView(DType dt, std::vector<std::size_t> shape, std::byte* pPtr) {
        NDArray a;
        a.mDtype = dt;
        a.mShape = std::move(shape);
        a.mView = pPtr;
        return a;
    }

    DType Dtype() const { return mDtype; }
    const std::vector<std::size_t>& Shape() const { return mShape; }
    std::size_t Ndim() const { return mShape.size(); }
    /** @brief Whether this array is a non-owning view (vs. owning its buffer). */
    bool IsView() const { return mView != nullptr; }

    /**
     * @brief The number of elements the buffer holds: the product of `Shape()`.
     *
     * An empty shape is the one case needing a word, because it means two
     * different things. A **rank-0 array is a scalar** and holds exactly one
     * element -- the empty product, and numpy's own `np.array(1.0).size`. A
     * **default-constructed** `NDArray` also has an empty shape and holds
     * nothing at all. Nothing else separates them: `mDtype` defaults to a
     * valid `Float64` and `DType` has no null enumerator.
     *
     * The buffer is what tells them apart, and stating the rule that way makes
     * it general rather than a special case: *`Size()` counts what the buffer
     * actually holds*, which is equally true of every non-empty shape (a
     * `{0, 3}` array allocates nothing and counts nothing).
     *
     * Callers depend on both halves. `PropertyValue::IsText`
     * (`properties.hpp`) and `KratosMesh`'s `ConcatKind` sentinel read
     * `Size() == 0` to mean "absent"; every byte-copy path -- `MakeOwned`,
     * `Nbytes`, `data_owned_copy` and the per-operation copies built on it --
     * needs a scalar to count as one element, or a 0-d value crossing from
     * numpy is silently dropped on the first clone.
     */
    std::size_t Size() const {
        if (mShape.empty())
            return Data() == nullptr ? 0 : 1;
        return ShapeCount(mShape);
    }
    /** @brief Total buffer size in bytes: `Size() * dtype_size(Dtype())`. */
    std::size_t Nbytes() const { return Size() * dtype_size(mDtype); }

    /** @brief Raw pointer to the active buffer (owned or view), for writing. */
    std::byte* Data() { return mView ? mView : mOwned.data(); }
    /** @brief Raw pointer to the active buffer (owned or view), read-only. */
    const std::byte* Data() const { return mView ? mView : mOwned.data(); }

    /**
     * @brief Changes the logical shape in place without touching the buffer.
     *
     * A no-op if the new shape's element count doesn't match the current
     * one (the mismatched reshape is silently ignored rather than throwing).
     * @param new_shape The desired row-major dimensions.
     */
    void Reshape(std::vector<std::size_t> new_shape) {
        // ShapeCount, so a one-element array moves freely between `{1}` and
        // `{}` (a scalar); an array holding nothing still refuses both.
        if (ShapeCount(new_shape) != Size())
            return;  // ignore inconsistent reshape
        mShape = std::move(new_shape);
    }

    /**
     * @brief Turns a view into an owning copy in place; a no-op if already owning.
     *
     * Copies the viewed memory into a freshly-allocated owned buffer and
     * clears the view pointer. Used before handing a buffer's lifetime over
     * to Python via a capsule (`mesh_to_py`), where the destination `NDArray`
     * must actually own the memory it hands off.
     */
    void MakeOwned() {
        if (mView == nullptr)
            return;
        const std::size_t nb = Nbytes();
        mOwned.resize(nb);  // uninitialised; fully overwritten by the memcpy below
        std::memcpy(mOwned.data(), mView, nb);
        mView = nullptr;
    }

    /**
     * @brief Reinterprets the raw buffer as a `T*`. No dtype check is performed
     * — the caller must ensure `T` matches `Dtype()`.
     * @tparam T The scalar type to view the buffer as.
     * @return Pointer to the first element, typed as `T`.
     */
    template <typename T>
    T* As() {
        return reinterpret_cast<T*>(Data());
    }
    /** @brief `const` overload of `As()`. */
    template <typename T>
    const T* As() const {
        return reinterpret_cast<const T*>(Data());
    }

private:
    /**
     * @brief The element count a shape implies: the product of its extents.
     *
     * One for an EMPTY shape, by construction -- `std::accumulate`'s init
     * value over an empty range -- which is the right answer for a rank-0
     * scalar. `Size()` is what decides whether an empty shape is a scalar at
     * all; this only does the arithmetic, so the constructors can size a
     * buffer before one exists.
     */
    static std::size_t ShapeCount(const std::vector<std::size_t>& rShape) {
        return std::accumulate(rShape.begin(), rShape.end(), std::size_t{1},
                               std::multiplies<std::size_t>());
    }

    DType mDtype = DType::Float64;
    std::vector<std::size_t> mShape;
    detail::OwnedBuf mOwned;
    std::byte* mView = nullptr;
};

}  // namespace meshioplusplus
