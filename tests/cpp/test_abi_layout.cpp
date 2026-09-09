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
/**
 * @file test_abi_layout.cpp
 * @brief The layout half of the ABI gate: a committed snapshot of the size and
 * alignment of every type that crosses the installed-library boundary.
 *
 * `doc/abi.md` splits ABI-relevant changes into three tiers. This file makes
 * **Tier A** (layout) fail mechanically and objectively: add, remove, reorder or
 * retype a data member of any type below and this test stops compiling, naming
 * the type. The reviewer then either reverts it or bumps
 * `MESHIOPLUSPLUS_ABI_VERSION` in `abi_version.hpp` and updates the number here.
 *
 * It is deliberately only half the gate. **Tier B** -- editing the body of an
 * existing inline function or template -- changes no size at all and is
 * invisible here; `tools/check-abi-version.sh` is what catches that, by diffing
 * the installed headers against the previous release tag. Neither tool alone is
 * sufficient, which is why both exist.
 *
 * ### Why the numbers are pinned on one configuration only
 *
 * `sizeof(std::string)` is 32 with libstdc++ and 24 with libc++; `std::vector`
 * and `std::unordered_map` differ likewise, and MSVC differs from both. Hard
 * numbers are therefore only meaningful against a named reference
 * configuration -- Linux x86_64 + libstdc++ + LP64, which is what CI's
 * `cpp-tests` matrix runs on every backend. Everywhere else the test reports the
 * layout and skips rather than failing for a reason that has nothing to do with
 * meshio++. That is not a loophole: the gate runs on every PR in CI, and a
 * developer on macOS seeing a skip has lost nothing, whereas a hard failure
 * there would train people to ignore this file.
 *
 * A snapshot mismatch is NOT automatically a bug -- it is a question. "Did you
 * mean to change the ABI?" The answer is often yes.
 */

// System includes
#include <cstddef>
#include <string>

// External includes
#include <gtest/gtest.h>

// Project includes
#include "meshioplusplus/abi_version.hpp"
#include "meshioplusplus/detail/provenance.hpp"
#include "meshioplusplus/cell_type.hpp"
#include "meshioplusplus/mesh.hpp"
#include "meshioplusplus/ndarray.hpp"
#include "meshioplusplus/properties.hpp"
#include "meshioplusplus/read_options.hpp"
#include "meshioplusplus/region.hpp"
#include "meshioplusplus/write_options.hpp"
#include "meshioplusplus/formats/gmsh.hpp"
#include "meshioplusplus/formats/mdpa.hpp"
#include "meshioplusplus/formats/openfoam.hpp"
#include "meshioplusplus/operations/pipeline.hpp"
#include "meshioplusplus/operations/refine.hpp"
#include "meshioplusplus/operations/remesh.hpp"
#include "meshioplusplus/operations/remesh_volume.hpp"
#include "meshioplusplus/operations/optimize_volume.hpp"
#include "meshioplusplus/operations/curvature.hpp"
#include "meshioplusplus/operations/repair.hpp"
#include "meshioplusplus/operations/shrinkwrap.hpp"
#include "meshioplusplus/operations/sobolev_deform.hpp"
#include "meshioplusplus/operations/smooth.hpp"
#include "meshioplusplus/operations/voxelize.hpp"
#include "meshioplusplus/detail/grid_lattice.hpp"
#include "meshioplusplus/operations/sequence.hpp"

namespace {

// The reference configuration the pinned numbers below describe. Anything else
// (libc++, MSVC, 32-bit, non-x86_64) gets the reporting path instead.
#if defined(__linux__) && defined(__x86_64__) && defined(__GLIBCXX__) && !defined(_MSC_VER)
#define MIO_ABI_LAYOUT_PINNED 1
#else
#define MIO_ABI_LAYOUT_PINNED 0
#endif

#if MIO_ABI_LAYOUT_PINNED

// One macro so a failure names the type and both numbers rather than pointing at
// an anonymous static_assert. sizeof AND alignof: a member reordering that
// happens to preserve size can still change alignment, and a change to an
// enum's underlying type changes alignment while size may survive.
#define MIO_ABI_LAYOUT(Type, Size, Align)                                             \
    static_assert(sizeof(Type) == (Size),                                             \
                  "meshio++ ABI: sizeof(" #Type                                       \
                  ") changed. This is a Tier A layout "                               \
                  "break (doc/abi.md). Bump MESHIOPLUSPLUS_ABI_VERSION in "           \
                  "abi_version.hpp and update this snapshot, or revert the member."); \
    static_assert(alignof(Type) == (Align),                                           \
                  "meshio++ ABI: alignof(" #Type                                      \
                  ") changed. This is a Tier A layout "                               \
                  "break (doc/abi.md). Bump MESHIOPLUSPLUS_ABI_VERSION in "           \
                  "abi_version.hpp and update this snapshot, or revert the member.")

// --- Backend-independent boundary types -------------------------------------
// Every one of these is passed by value or reference through an exported
// function, so a consumer and the library must agree on all of them regardless
// of which backend either was built with.

MIO_ABI_LAYOUT(meshioplusplus::NDArray, 72, 8);
MIO_ABI_LAYOUT(meshioplusplus::Region, 120, 8);
MIO_ABI_LAYOUT(meshioplusplus::ReadOptions, 56, 8);
MIO_ABI_LAYOUT(meshioplusplus::WriteOptions, 48, 8);
MIO_ABI_LAYOUT(meshioplusplus::PropertyValue, 144, 8);
MIO_ABI_LAYOUT(meshioplusplus::PropertySet, 32, 8);
MIO_ABI_LAYOUT(meshioplusplus::MeshMetadata, 288, 8);

// ProvenanceNote/ProvenanceRecord (v10.16.0, detail/provenance.hpp) -- pinned
// from the release that introduces them, the RefineOptions/RemeshOptions
// "pin in advance" lesson: both are new structs a caller can construct and
// pass across the ABI (ProvenanceScope's constructor takes a ProvenanceRecord
// by value), so a later member addition here is exactly the kind of Tier A
// change this file exists to catch mechanically.
MIO_ABI_LAYOUT(meshioplusplus::detail::ProvenanceNote, 64, 8);
MIO_ABI_LAYOUT(meshioplusplus::detail::ProvenanceRecord, 272, 8);

// The format side-channel structs. NONE of these was pinned before v9.20.0,
// which is exactly what made growing one *look* free: `MedInfo` gained four
// members in v9.9.0 and `OpenFoamInfo` gains one here, and in both cases the
// layout snapshot -- the Tier A guard -- had nothing to say. They travel by
// reference through exported `read_*`/`write_*` functions, so they are as much
// a boundary type as `ReadOptions` is.
//
// Only the three unconditional ones are here: `MedInfo` and `ExodusInfo` exist
// only in an HDF5 / netCDF build, so pinning them would make this snapshot say
// different things in different configurations -- which is the one property a
// layout snapshot must not have.
MIO_ABI_LAYOUT(meshioplusplus::OpenFoamInfo, 96, 8);
MIO_ABI_LAYOUT(meshioplusplus::GmshInfo, 24, 8);
MIO_ABI_LAYOUT(meshioplusplus::MdpaInfo, 72, 8);

// The pipeline and sequence aggregates. `run_pipeline(const Pipeline&)` and
// `run_sequence_pipeline(const SequencePipeline&)` are exported, and both
// aggregates embed their Input/Output structs BY VALUE -- so adding a member to
// `PipelineInput` silently shifts `Pipeline::mSteps` and `mOutput` under a
// consumer compiled against older headers, which is exactly the Tier A break
// this snapshot exists to make impossible to do by accident. These four went
// unpinned from v9.11.0 (when `operations/pipeline.hpp` shipped) until v9.12.0,
// which is what made growing them *look* free; that gap is what motivated
// adding a whole new header for the sequence types instead.
MIO_ABI_LAYOUT(meshioplusplus::PipelineStep, 80, 8);
MIO_ABI_LAYOUT(meshioplusplus::PipelineInput, 120, 8);
MIO_ABI_LAYOUT(meshioplusplus::PipelineOutput, 112, 8);
MIO_ABI_LAYOUT(meshioplusplus::Pipeline, 264, 8);
MIO_ABI_LAYOUT(meshioplusplus::SequenceEntry, 56, 8);
MIO_ABI_LAYOUT(meshioplusplus::SequenceInput, 176, 8);
MIO_ABI_LAYOUT(meshioplusplus::SequenceOutput, 112, 8);
MIO_ABI_LAYOUT(meshioplusplus::SequencePipeline, 336, 8);

// The distance/voxelization aggregates, pinned from the release that introduced
// them (v9.24.0) rather than after the fact -- the pipeline lesson above, applied
// in advance. Both `SdfOptions` and `VoxelOptions` embed `SurfaceDistanceOptions`
// BY VALUE, so growing that one member struct would shift everything after it in
// two outer structs at once. `operations/sdf.hpp` ships the octree fields
// populated-but-reserved for exactly this reason: adding them later would have
// been the same silent Tier A break.
//
// The matching *Result* structs are deliberately NOT pinned: each embeds a
// `Mesh`, whose size is per-backend, so a single number could not describe them
// and three would only restate the `Mesh` lines below.
MIO_ABI_LAYOUT(meshioplusplus::SurfaceQuality, 40, 8);
MIO_ABI_LAYOUT(meshioplusplus::SurfaceDistanceOptions, 80, 8);
MIO_ABI_LAYOUT(meshioplusplus::SdfOptions, 248, 8);
MIO_ABI_LAYOUT(meshioplusplus::VoxelOptions, 216, 8);
MIO_ABI_LAYOUT(meshioplusplus::detail::LatticeSpec, 72, 8);

// RefineOptions is passed by const-ref through the exported `refine()`,
// exactly like SdfOptions/VoxelOptions above -- but unlike them it went
// unpinned from v9.0.0 (when this file was introduced) all the way through
// the v9.23.0/v9.24.0 rows that grew it, which is precisely the "unpinned
// aggregate looks free to grow" gap the pipeline/sequence comment above
// warns about. Closed here rather than after a third field is added to it.
MIO_ABI_LAYOUT(meshioplusplus::RefineOptions, 120, 8);

// RemeshOptions is passed by const-ref through the exported `remesh()`,
// the same RefineOptions shape -- pinned from the release that first grew
// it (v10.11.0's mGradation/mPreserveBoundary) rather than left unpinned
// the way RefineOptions was for over twenty releases. RemeshResult is
// deliberately NOT pinned, for the same reason as every other *Result
// struct noted above: it embeds a `Mesh`, whose size is per-backend.
MIO_ABI_LAYOUT(meshioplusplus::RemeshOptions, 64, 8);

// RemeshVolumeOptions is passed by const-ref through the exported
// `remesh_volume()`, and embeds `SurfaceDistanceOptions` BY VALUE exactly as
// SdfOptions/VoxelOptions do above -- pinned from the release that
// introduces it (v10.13.0) rather than after the fact, the RefineOptions
// lesson applied in advance a second time. RemeshVolumeResult is
// deliberately NOT pinned, embedding a `Mesh` like every other *Result.
MIO_ABI_LAYOUT(meshioplusplus::RemeshVolumeOptions, 224, 8);

// OptimizeVolumeOptions is passed by const-ref through the exported
// `optimize_volume()` -- pinned from the release that introduces it, the
// RefineOptions/RemeshOptions "pin in advance" lesson. OptimizeVolumeResult is
// deliberately NOT pinned, embedding a `Mesh` like every other *Result.
MIO_ABI_LAYOUT(meshioplusplus::OptimizeVolumeOptions, 40, 8);

// CurvatureOptions is passed by const-ref through the exported
// `compute_curvature()`, and is pinned from the release that introduces it --
// the RefineOptions/RemeshOptions "pin in advance" lesson (a struct that goes
// unpinned "looks free" to grow, then does). CurvatureResult is deliberately
// NOT pinned, embedding a `Mesh` like every other *Result.
MIO_ABI_LAYOUT(meshioplusplus::CurvatureOptions, 40, 8);

// The v10.38.0 trio, pinned from the release that introduces them for the
// same reason. RepairOptions: five bools, an int64 and a double (24/8).
// ShrinkwrapOptions: two doubles, two strings, an enum with NO fixed
// underlying type (4 bytes -- giving SdfPseudonormalWeight one would shrink
// this and the pin would say so), two bools and a double (96/8).
// SobolevOptions: a string, a double, a vector, a string, two bools, an int
// and a double (112/8). Every *Result is unpinned, embedding a `Mesh`.
MIO_ABI_LAYOUT(meshioplusplus::RepairOptions, 24, 8);
MIO_ABI_LAYOUT(meshioplusplus::ShrinkwrapOptions, 96, 8);
MIO_ABI_LAYOUT(meshioplusplus::SobolevOptions, 112, 8);

// SmoothOptions is passed by const-ref through the exported `smooth()`, and
// is pinned here for the FIRST time -- not because it grew a member (it
// didn't), but because `SmoothMethod` (its first member) gained an explicit
// `: std::uint8_t` underlying type in the same release, which doc/abi.md's
// own Tier A rule names explicitly ("changing an enum's underlying type").
// A scoped enum with no explicit underlying type defaults to `int`
// [dcl.enum], so this is a genuine 4-byte -> 1-byte narrowing of
// `SmoothMethod` alone, independent of whether padding happens to leave
// `sizeof(SmoothOptions)` unchanged -- exactly the RefineOptions/
// RemeshOptions "pin from the release that first touches it" lesson,
// applied here to a change that is easy to mistake for Tier C (an appended
// enumerator, `Odt`) when it is actually Tier A (the underlying-type
// change that made the appendage narrow-safe going forward). SmoothResult
// is deliberately NOT pinned, for the same reason as every other *Result
// struct above: it embeds a `Mesh`, whose size is per-backend.
MIO_ABI_LAYOUT(meshioplusplus::SmoothOptions, 80, 8);

// CellType is stored inside cell blocks on the NATIVE and KRATOS backends, so
// its width is structural, not cosmetic. Appending an enumerator is fine (and
// deliberately not caught here); widening the underlying type is not.
static_assert(sizeof(meshioplusplus::CellType) == 2,
              "meshio++ ABI: CellType's underlying type changed width. Appending "
              "enumerators is safe and expected; changing `: std::uint16_t` is a "
              "Tier A break (doc/abi.md).");

// RemeshMetric mirrors the same reasoning as CellType above: appending
// `Anisotropic` (v10.12.0) is fine and deliberately not caught here; the
// primary guard lives in remesh.hpp itself (checked by every consumer that
// compiles it, not just this suite) -- this mirror exists for the same
// discoverability CellType's own entry has here.
static_assert(sizeof(meshioplusplus::RemeshMetric) == 1,
              "meshio++ ABI: RemeshMetric's underlying type changed width. "
              "Appending enumerators is safe and expected; widening "
              "`: std::uint8_t` is a Tier A break (doc/abi.md).");

// SmoothMethod mirrors the same reasoning: appending `Odt` (v10.13.0) is
// fine and deliberately not caught here; the primary guard lives in
// smooth.hpp itself (checked by every consumer that compiles it), this
// mirror exists for the same discoverability CellType/RemeshMetric have.
static_assert(sizeof(meshioplusplus::SmoothMethod) == 1,
              "meshio++ ABI: SmoothMethod's underlying type changed width. "
              "Appending enumerators is safe and expected; widening "
              "`: std::uint8_t` is a Tier A break (doc/abi.md).");

// --- The mesh itself, which IS the backend ----------------------------------
// Each backend gets its own line because Mesh is a different type per backend;
// only the one this TU was compiled for is checked, and the cpp-tests matrix
// runs this file once per backend so all three are covered across CI.
#if defined(MESHIOPLUSPLUS_MESH_BACKEND_NATIVE)
MIO_ABI_LAYOUT(meshioplusplus::Mesh, 560, 8);
#elif defined(MESHIOPLUSPLUS_MESH_BACKEND_KRATOS)
MIO_ABI_LAYOUT(meshioplusplus::Mesh, 680, 8);
#else
MIO_ABI_LAYOUT(meshioplusplus::Mesh, 392, 8);
#endif

#endif  // MIO_ABI_LAYOUT_PINNED

/// Reports one type's layout, so a mismatch on an unpinned platform is
/// actionable rather than merely skipped.
template <class T>
void report(const char* name) {
    std::printf("  %-42s sizeof=%3zu alignof=%2zu\n", name, sizeof(T), alignof(T));
}

}  // namespace

TEST(AbiLayout, SnapshotIsPinnedOnTheReferenceConfiguration) {
    std::printf("meshio++ ABI version %d, layout on this platform:\n", MESHIOPLUSPLUS_ABI_VERSION);
    report<meshioplusplus::NDArray>("NDArray");
    report<meshioplusplus::Region>("Region");
    report<meshioplusplus::ReadOptions>("ReadOptions");
    report<meshioplusplus::WriteOptions>("WriteOptions");
    report<meshioplusplus::PropertyValue>("PropertyValue");
    report<meshioplusplus::PropertySet>("PropertySet");
    report<meshioplusplus::MeshMetadata>("MeshMetadata");
    report<meshioplusplus::OpenFoamInfo>("OpenFoamInfo");
    report<meshioplusplus::GmshInfo>("GmshInfo");
    report<meshioplusplus::MdpaInfo>("MdpaInfo");
    report<meshioplusplus::PipelineStep>("PipelineStep");
    report<meshioplusplus::PipelineInput>("PipelineInput");
    report<meshioplusplus::PipelineOutput>("PipelineOutput");
    report<meshioplusplus::Pipeline>("Pipeline");
    report<meshioplusplus::SequenceEntry>("SequenceEntry");
    report<meshioplusplus::SequenceInput>("SequenceInput");
    report<meshioplusplus::SequenceOutput>("SequenceOutput");
    report<meshioplusplus::SequencePipeline>("SequencePipeline");
    report<meshioplusplus::Mesh>("Mesh (this backend)");

#if MIO_ABI_LAYOUT_PINNED
    // The static_asserts above already ran at compile time; reaching here means
    // the snapshot held. Nothing further to assert -- this exists so the test
    // reports rather than silently passing.
    SUCCEED() << "layout snapshot pinned for ABI version " << MESHIOPLUSPLUS_ABI_VERSION;
#else
    GTEST_SKIP() << "layout numbers are pinned only on Linux/x86_64/libstdc++ "
                    "(sizeof(std::string) alone differs across standard libraries); "
                    "CI's cpp-tests matrix is that configuration.";
#endif
}

TEST(AbiLayout, GuardSymbolIsReferencedByThisTranslationUnit) {
    // The v9.1.0 lesson in executable form: the backend guard shipped inert for a
    // full release because an unused `inline` variable is emitted lazily, so
    // nothing ever forced the symbol to resolve. This TU links, and it includes
    // mesh.hpp, so both sentinels must have resolved for this binary to exist at
    // all -- taking their addresses here makes that dependency explicit rather
    // than incidental.
    EXPECT_NE(&meshioplusplus::detail::MESHIOPLUSPLUS_ABI_SYM(MESHIOPLUSPLUS_ABI_VERSION), nullptr);
    EXPECT_NE(&meshioplusplus::detail::MESHIOPLUSPLUS_BACKEND_SYM(MESHIOPLUSPLUS_ACTIVE_BACKEND),
              nullptr);
}
