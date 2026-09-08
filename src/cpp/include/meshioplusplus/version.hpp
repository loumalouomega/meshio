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
 * @file version.hpp
 * @brief The **release** version, as compile-time macros, so a consumer can
 * feature-detect with the preprocessor.
 *
 * ### Which version am I looking at?
 *
 * meshio++ answers three different questions with three different mechanisms,
 * and they are not interchangeable:
 *
 *  | question | mechanism |
 *  |----------|-----------|
 *  | what did I **compile against**? | the macros here (and `MIO_VERSION_*` in the C header) |
 *  | what am I **running against**?  | `mio_version()`, a runtime call into the linked library |
 *  | are my headers **binary-compatible** with that library? | `MESHIOPLUSPLUS_ABI_VERSION` plus
 * the link-time sentinel in `detail/abi_version_check.hpp` |
 *
 * With a shared library the first two genuinely can differ, which is the whole
 * reason both exist. Use these macros to decide what to *compile*, and
 * `mio_version()` to report what a user is actually running.
 *
 * ### Feature detection
 *
 * @code
 * #include <meshioplusplus/version.hpp>
 *
 * #if MESHIOPLUSPLUS_VERSION_AT_LEAST(9, 5, 0)
 *     options.mCells = {12, 13};      // selective refinement, new in 9.5.0
 * #endif
 * @endcode
 *
 * The encoded `MESHIOPLUSPLUS_VERSION` integer is `major * 10000 + minor * 100 +
 * patch`, so it orders exactly like the release does and can be compared
 * directly when the macro is not expressive enough.
 *
 * ### Why hand-written rather than generated
 *
 * The same reason `abi_version.hpp` is: a consumer that never runs CMake -- the
 * [single-header amalgamation](/single_header), pkg-config, a hand-written
 * makefile -- must still be able to read it, and a `configure_file`d header
 * living in the build tree would reach none of them. `CMakeLists.txt` therefore
 * *parses* this file and hard-fails at configure time if it disagrees with
 * `project(... VERSION ...)`, so the duplication cannot silently drift; the C
 * header's `MIO_VERSION_*` twins are pinned by a `static_assert` in
 * `bindings/c/c_api.cpp`.
 *
 * @note Bumping the release means editing this file too -- see the "Version
 * bumps" section of `CLAUDE.md`. Forgetting is a configure-time error, not a
 * wrong answer.
 */

/// Major component of the release version.
#define MESHIOPLUSPLUS_VERSION_MAJOR 10
/// Minor component of the release version.
#define MESHIOPLUSPLUS_VERSION_MINOR 30
/// Patch component of the release version.
#define MESHIOPLUSPLUS_VERSION_PATCH 0

/// The release version as one ordered integer: `major*10000 + minor*100 + patch`.
#define MESHIOPLUSPLUS_VERSION                                                   \
    (MESHIOPLUSPLUS_VERSION_MAJOR * 10000 + MESHIOPLUSPLUS_VERSION_MINOR * 100 + \
     MESHIOPLUSPLUS_VERSION_PATCH)

/// The release version as a string literal, e.g. `"9.6.0"`.
#define MESHIOPLUSPLUS_VERSION_STRING "10.30.0"

/// Whether the headers being compiled against are at least `major.minor.patch`.
#define MESHIOPLUSPLUS_VERSION_AT_LEAST(major, minor, patch) \
    (MESHIOPLUSPLUS_VERSION >= ((major) * 10000 + (minor) * 100 + (patch)))

/// Whether the headers being compiled against are older than `major.minor.patch`.
#define MESHIOPLUSPLUS_VERSION_BEFORE(major, minor, patch) \
    (!MESHIOPLUSPLUS_VERSION_AT_LEAST(major, minor, patch))
