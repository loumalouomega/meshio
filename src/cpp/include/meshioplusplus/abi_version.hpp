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
 * @file abi_version.hpp
 * @brief `MESHIOPLUSPLUS_ABI_VERSION` -- the C++ API's binary-compatibility
 * counter, deliberately **not** the release version.
 *
 * The release version (`project(... VERSION ...)`) moves on every release. This
 * one moves only when the installed headers change in a way that makes an
 * already-compiled consumer incompatible with a freshly built library. Those are
 * different questions, and conflating them is what forced every C++ consumer to
 * re-pin and fully rebuild for releases that provably could not affect them --
 * v9.3.0's entire installed-header delta, for instance, was one new
 * `inline constexpr` string in `formats/exodus.hpp`.
 *
 * `doc/abi.md` owns the criterion in full. In short, scope is **every** header
 * under `src/cpp/include/meshioplusplus/` (all of them are installed and
 * compiled by consumers -- not a curated subset), and a bump is required for:
 *
 *  - a **layout** change to any type a consumer can name (data members, bases,
 *    virtuals, alignment, an enum's underlying type, template parameter lists);
 *  - an **ODR** change: editing the *body* of an existing inline function,
 *    template or default argument. Both sides emit a vague-linkage copy and the
 *    linker keeps one arbitrarily, so the program can run the older body where
 *    the newer was intended.
 *
 * and NOT for purely additive changes -- a new inline function, constexpr
 * variable, type, declaration or header, or an appended enumerator. Nothing
 * existing changes definition there, so an already-compiled consumer's
 * translation unit is byte-identical to what it was.
 *
 * Retroactive numbering of the 9.x line (see `doc/abi.md` for the diffs this was
 * derived from):
 *
 *  | ABI | releases           | what moved                                     |
 *  |-----|--------------------|------------------------------------------------|
 *  | 1   | v9.0.0             | (baseline)                                     |
 *  | 2   | v9.1.0             | `GeometricalEntity`, `ModelPart`, `MdpaInfo`, … |
 *  | 3   | v9.2.0 .. v9.4.1   | `KratosMesh`, `PropertySet`, `NativeMesh`, …   |
 *  | 4   | v9.5.0 .. v9.8.0   | `RefineOptions` gained selection/closure fields |
 *  | 5   | v9.9.0 .. v9.19.0  | `MedInfo` gained four lenient-read fields       |
 *  | 6   | v9.20.0 .. v10.0.0 | `OpenFoamInfo` gained `mPatchTypes`             |
 *  | 7   | v10.1.0            | `RefineOptions` gained `mRecordHierarchy`       |
 *  | 8   | v10.11.0           | `RemeshOptions` gained `mGradation`/`mPreserveBoundary`, `RemeshResult` gained `mNumNonManifoldVertices` |
 *  | 9   | v10.12.0           | `RemeshOptions` gained `mMaxAnisotropy`; `RemeshMetric` gained `Anisotropic` |
 *  | 10  | v10.13.0           | `SmoothMethod` gained an explicit `: std::uint8_t` underlying type (previously the scoped-enum default `int`) plus `Odt`; `RemeshVolumeOptions`/`RemeshVolumeResult` are new (Tier C, riding along) |
 *  | 11  | v10.17.0 .. v10.34.0 | `MeshMetadata` gained `mProvenance`/`mProvenanceRecognised` (256 -> 288 bytes) for provenance read-back |
 *  | 12  | v10.35.0           | **Tier B, not a layout change**: `NDArray::Size()`'s inline body. It reported 0 for a rank-0 array, so `Nbytes()` was 0 and every clone dropped a 0-d scalar's single element -- now it counts what the buffer holds. `sizeof(NDArray)` is unchanged at 72 |
 *
 * ### This is the ONE place the number is written
 *
 * `CMakeLists.txt` parses it back out of this file with `file(STRINGS ... REGEX)`
 * rather than keeping a second copy, and feeds it to the installed libraries'
 * `SOVERSION` and to `MESHIOPLUSPLUS_ABI_VERSION` in the generated
 * `meshioplusplusConfig.cmake`. A header rather than a CMake variable because a
 * consumer that never runs CMake -- pkg-config, a hand-written makefile -- must
 * still be able to read it.
 *
 * Deliberately **not** overridable with `-D`: `detail/abi_version_check.hpp`
 * turns this value into a link-time requirement, and a knob to change it would be
 * a knob to silence the guard while still miscompiling. `tools/check-abi-version.sh`
 * and the mismatch test in CI both work by patching a *copy* of the installed
 * headers for exactly this reason. `MESHIOPLUSPLUS_NO_ABI_VERSION_CHECK` is the
 * supported opt-out.
 */

#define MESHIOPLUSPLUS_ABI_VERSION 12
