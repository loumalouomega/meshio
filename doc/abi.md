# C++ ABI compatibility

This page is the authoritative answer to one question: **when does a change to meshio++ break a C++ consumer that was already compiled?**

It exists because the honest answer is narrower than "every release". `find_package(... X.Y.Z EXACT ...)` keys on the release version, which moves whenever *anything* in the project does — a Python-side fix, a new format, a doc change. v9.3.0's entire installed-header delta was one new `inline constexpr` string in `formats/exodus.hpp`, which cannot affect a compiled consumer at all, yet the version rule still demanded that every C++ consumer re-pin and rebuild. [`MESHIOPLUSPLUS_ABI_VERSION`](#the-abi-version) is the finer signal.

This page covers `COMPONENTS CXX` only. The **C API is a different contract** and is unaffected by everything here: it exposes no header-defined types, its option structs grow only into their `reserved` tails, and it keeps `SOVERSION 0` — pin the major and see [the C API page](/c_api).

## The criterion

**Scope is every header under `src/cpp/include/meshioplusplus/`.** All of them are installed, and all of them are compiled by consumers, so all of them participate. This is deliberately *not* a curated list of "the important headers": such a list would have been wrong. The releases that really did break ABI touched `formats/mdpa.hpp` (`MdpaInfo`), `formats/xdmf_time_series.hpp`, `kratos_bridge.hpp` and `mesh_api.hpp` — none of which look like "the mesh types" — while every `operations/*.hpp` options and result struct (`PartitionOptions`, `SmoothOptions`, `DecimateOptions`, `QualityReport`, …) crosses the boundary by value and merely happens not to have changed yet.

`src/cpp/src/**.cpp` never matters: a consumer does not compile it.

What matters is *what the change does*, not which file it lives in.

### Tier A — layout. Bump the ABI version.

Anything that changes the shape of a type a consumer can name:

- adding, removing, reordering or retyping a **data member**;
- adding or changing **base classes** or **virtual functions**;
- changing **alignment**, or an enum's **underlying type**;
- changing a **template's parameter list**.

The consumer and the library then disagree about where the fields of an argument are. Nothing diagnoses this at run time; it presents as unrelated memory corruption.

`tests/cpp/test_abi_layout.cpp` catches this tier **mechanically** — a committed snapshot of `sizeof`/`alignof` for every boundary type, per backend, checked by `static_assert`.

### Tier B — ODR. Bump the ABI version.

Changing the **body** of an existing `inline` function, a template, or a **default argument**.

This is the tier people miss, so it is worth being precise: both the library and the consumer emit their own copy of an inline function, with vague linkage, and the linker keeps exactly one of them — arbitrarily. If the two copies differ, the program may run the *old* body in a place the new one was intended, or vice versa, with no diagnostic. Formally it is an ODR violation and the behaviour is undefined.

So yes: **an inline function body change counts, even though no type changed size.** Nothing mechanical can catch it (a new inline function and an edited one look the same textually), which is why `tools/check-abi-version.sh` demands a recorded human judgement instead of guessing.

### Tier C — additive. Do **not** bump.

- a **new** inline function, constexpr variable, type, free-function declaration, or header;
- an **appended** enumerator.

Nothing that already existed changes definition, so an already-compiled consumer's translation unit is byte-for-byte what it was. It simply does not have the new name. v9.3.0's `kExodusAttributePrefix` is exactly this.

Record these in [`doc/abi_reviews.md`](abi_reviews.md), which is what lets the CI gate pass while keeping the reasoning on the record. The entry is keyed on the **release** and must name **every** header that changed: the gate reads both columns, so a review of an earlier release does not carry a later one, and a header you leave off is a red build naming it.

#### The one asymmetry: appended enumerators

Appending to `CellType` (`enum class CellType : std::uint16_t` — fixed underlying type, so the width never moves) is safe in the direction that matters: **older headers + newer library**. The reverse — a consumer compiled against *newer* headers running against an *older* library — can hand the library an enumerator it has never heard of. The guard below enforces exact agreement, so this cannot arise silently; it is documented because the tiers would otherwise seem to promise more than they do.

### What is out of scope entirely

Same-toolchain is a **precondition**, not something meshio++ can check. A consumer built with a different compiler, standard library, `_GLIBCXX_USE_CXX11_ABI` setting, or with flags that change standard-library layout is incompatible no matter what this page says, and no sentinel symbol can detect it.

## The ABI version

`MESHIOPLUSPLUS_ABI_VERSION` lives in **one place**, `src/cpp/include/meshioplusplus/abi_version.hpp`. `CMakeLists.txt` parses it back out of that header rather than keeping a copy, so the number in a compiled object and the number in the CMake package can never disagree.

| ABI | releases | what moved |
| --- | --- | --- |
| 1 | v9.0.0 | baseline |
| 2 | v9.1.0 | `GeometricalEntity` gained a member; `ModelPart`, `MdpaInfo`, `PropertySet`, `ReadOptions`, `XdmfTimeSeriesWriter`, `kratos_bridge.hpp` |
| 3 | v9.2.0 – v9.4.1 | `KratosMesh`, `MeshioMesh`, `NativeMesh`, `PropertySet`, `mesh_api.hpp`, `XdmfTimeSeriesWriter` |
| 4 | v9.5.0 | `RefineOptions` gained the selective-refinement fields (a cell list, a region name, a `cell_data` predicate, the closure mode and the level flag) |
| 5 | v9.9.0 – v9.19.0 | `MedInfo` gained four members (`mFieldUnits`, `mStepMeta`, `mFieldTimeValues`, `mSkippedConstructs`) for the lenient-read surface |
| 6 | v9.20.0 – v10.0.0 | `OpenFoamInfo` gained `mPatchTypes`, so the OpenFOAM writer can round-trip a patch's `type` |
| 7 | v10.1.0 | `RefineOptions` gained `mRecordHierarchy`, the persistent `refine:cell_id`/`refine:parent_id` parent/child hierarchy |
| 8 | v10.11.0 | `RemeshOptions` gained `mGradation`/`mPreserveBoundary`, `RemeshResult` gained `mNumNonManifoldVertices` |
| 9 | v10.12.0 | `RemeshOptions` gained `mMaxAnisotropy`, `RemeshMetric` gained `Anisotropic` |
| 10 | v10.13.0 | `SmoothMethod` gained an explicit `: std::uint8_t` underlying type — a genuine 4-byte-to-1-byte narrowing, and Tier A independently of the `Odt` enumerator that rode in on it |
| 11 | v10.17.0 – v10.34.0 | `MeshMetadata` gained `mProvenance`/`mProvenanceRecognised` (256 → 288 bytes), so `read_metadata` can report the block a file carries |
| 12 | v10.35.0 | **Tier B, and the first entry here that is not a layout change**: `NDArray::Size()`'s inline body. It reported 0 for a rank-0 array, so `Nbytes()` was 0 and every clone silently dropped a 0-d scalar's single element; it now counts what the buffer holds. `sizeof(NDArray)` is unchanged at 72, which is exactly why the layout snapshot could not see this and the version had to move by hand |

It reaches consumers three ways:

1. **The CMake package** — `MESHIOPLUSPLUS_ABI_VERSION` is set by `meshioplusplusConfig.cmake` beside the existing `MESHIOPLUSPLUS_MESH_BACKEND` / `MESHIOPLUSPLUS_WITH_*` introspection.
2. **`SOVERSION`** — the C++ variants install as `libmeshioplusplus_core_<backend>.so.<abi>`, so the *dynamic linker* refuses to load an incompatible library into an already-linked binary, with no cooperation from anyone's build system. (The C library keeps `SOVERSION 0`; its contract is unchanged.)
3. **A link-time sentinel** — see below.

## The guard: a mismatch fails at link time

`detail/abi_version_check.hpp` plants, in every translation unit that includes `mesh.hpp`, a reference to `meshioplusplus::detail::abi_version_is_<N>()`, where `N` is the ABI version *that TU's headers* declare. The library defines exactly one such symbol. Disagree, and the link fails naming the version the consumer expected:

```
undefined reference to `meshioplusplus::detail::abi_version_is_99()'
```

instead of corrupting memory at run time. It is the twin of `detail/mesh_backend_check.hpp`, deliberately a **separate symbol** so that a consumer with the right backend and stale headers is told about the headers rather than misleadingly about the backend. Cost is one relocation per TU per axis and no static initializer.

Three properties worth knowing:

- **`gnu::used` is load-bearing.** An unused `inline` variable has vague linkage and is emitted lazily, so without it no relocation reaches the object file and the guard is *silently inert*. That was the backend guard's state through v9.1.0. CI asserts the reference really is present (`nm -uC`) rather than assuming it.
- **MSVC + a shared build is a gap.** The MSVC arm is `#pragma detect_mismatch`, whose `/FAILIFMISMATCH` records live in `.obj` files and are not reliably carried through a DLL import library. Static MSVC builds and every GNU/Clang configuration are covered. On that one configuration the `SOVERSION`-derived DLL name and the version pin are what remain.
- **Opt out with `MESHIOPLUSPLUS_NO_ABI_VERSION_CHECK`**, mirroring `MESHIOPLUSPLUS_NO_BACKEND_LINK_CHECK`. `MESHIOPLUSPLUS_ABI_VERSION` itself is deliberately *not* `-D`-overridable: a knob to change it would be a knob to silence the guard while still miscompiling.

### The one false positive

Introducing the guard in v9.4.0 means v9.4.0 headers reference `abi_version_is_3`, which a v9.2.0 or v9.3.0 library does not define — so that pairing fails to link even though it is genuinely compatible by the criterion above. This is a one-time cost at the boundary, and it fails *closed*, which is the direction to fail in.

## Keeping the number honest

A hand-bumped integer has the same "someone forgets" failure mode as the prose rule it replaces. Two gates, covering different tiers:

| gate | catches | how |
| --- | --- | --- |
| `tests/cpp/test_abi_layout.cpp` | Tier A | `static_assert`s on `sizeof`/`alignof`, per backend, in the `cpp-tests` matrix. Objective; no judgement. |
| `tools/check-abi-version.sh` | Tier B | Diffs installed headers against the previous release tag. Headers changed + ABI unchanged + no [review entry](abi_reviews.md) *for this release naming those headers* = red build. |

Neither is sufficient alone: the first cannot see an inline body change, and the second cannot tell an edited inline function from a new one.

The emphasis in that second row is load-bearing, and was a bug fixed in v9.4.1: matching a review row on the ABI number alone passes vacuously as soon as one such row exists, so the lookup is keyed on the release version *and* on the set of headers the row accounts for. `tests/python/test_check_abi_version.py` asserts the gate actually fires — the same lesson the backend guard learned in v9.1.0, when it shipped inert for a full release because nothing tested that it did.

## What to pin, in practice

See [the C++ API page](/cpp_api#versioning-what-to-pin) for the full guidance. In short:

```cmake
# Finer, and true: pin what actually constrains you.
find_package(meshioplusplus CONFIG REQUIRED COMPONENTS CXX)
if(NOT MESHIOPLUSPLUS_ABI_VERSION EQUAL 6)
  message(FATAL_ERROR "meshio++ ABI 6 required, found ${MESHIOPLUSPLUS_ABI_VERSION}")
endif()
```

```cmake
# Conservative, and still fully supported: pin the release.
find_package(meshioplusplus 9.12.0 EXACT CONFIG REQUIRED COMPONENTS CXX)
```

Both are correct. The second is stricter than it needs to be, and that is a legitimate choice — it is one line, it needs no reasoning about tiers, and re-pinning is a small price for never thinking about this page again.
