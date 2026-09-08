cmake_minimum_required(VERSION 3.15...3.30)

# VERSION must track pyproject.toml's `version`, conanfile.py's `version`,
# packages/vcpkg/meshioplusplus/vcpkg.json's `version`, src/wasm/package.json's
# `version`, the "../wasm" entry in src/viewer/package-lock.json,
# bindings/julia/MeshioPlusPlus/Project.toml's `version` and
# bindings/r/meshioplusplus/DESCRIPTION's `Version:` (bump all eight together
# on a release) -- it feeds mio_version(), the shared-library VERSION
# properties, and the find_package/pkg-config metadata of the C API.
project(
  meshioplusplus_core
  VERSION 10.36.0
  LANGUAGES C CXX
  DESCRIPTION "C++ core for the meshio++ mesh I/O library")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# The C++ ABI version, PARSED OUT OF THE HEADER rather than duplicated here.
#
# src/cpp/include/meshioplusplus/abi_version.hpp is the single source of truth
# (a header, so a consumer who never runs CMake can read it too), and this is
# the only reader. Keeping a second copy in this file is precisely the drift
# this arrangement exists to prevent: the number is baked into every compiled
# TU by detail/abi_version_check.hpp, so a CMake copy that disagreed would put
# a wrong SOVERSION on a library whose symbols say otherwise.
#
# Unlike PROJECT_VERSION this moves only when the installed headers stop being
# binary-compatible -- see doc/abi.md for the criterion.
# ---------------------------------------------------------------------------
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/include/meshioplusplus/abi_version.hpp"
     _mio_abi_version_line REGEX "^#define[ \t]+MESHIOPLUSPLUS_ABI_VERSION[ \t]+[0-9]+")
if(NOT _mio_abi_version_line MATCHES "MESHIOPLUSPLUS_ABI_VERSION[ \t]+([0-9]+)")
  message(FATAL_ERROR
    "meshio++: could not parse MESHIOPLUSPLUS_ABI_VERSION out of "
    "src/cpp/include/meshioplusplus/abi_version.hpp")
endif()
set(MESHIOPLUSPLUS_ABI_VERSION ${CMAKE_MATCH_1})
unset(_mio_abi_version_line)
message(STATUS "meshio++: C++ ABI version: ${MESHIOPLUSPLUS_ABI_VERSION} "
               "(release ${PROJECT_VERSION})")

# ---------------------------------------------------------------------------
# The RELEASE version is repeated as preprocessor macros in two installed
# headers, so a consumer can feature-detect without running CMake (the
# single-header amalgamation, pkg-config, a hand-written makefile). They are
# hand-written for exactly the reason abi_version.hpp is, and verified here so
# the duplication cannot silently drift: a bump that misses one is a
# configure-time error rather than a wrong answer at run time. The C header's
# MIO_VERSION_* twins are pinned to the C++ ones by a static_assert in
# bindings/c/c_api.cpp.
# ---------------------------------------------------------------------------
function(_mio_check_version_header path prefix)
  foreach(_part MAJOR MINOR PATCH)
    file(STRINGS "${path}" _line REGEX "^#define[ \t]+${prefix}_VERSION_${_part}[ \t]+[0-9]+")
    if(NOT _line MATCHES "${prefix}_VERSION_${_part}[ \t]+([0-9]+)")
      message(FATAL_ERROR "meshio++: could not parse ${prefix}_VERSION_${_part} out of ${path}")
    endif()
    set(_got_${_part} ${CMAKE_MATCH_1})
  endforeach()
  set(_got "${_got_MAJOR}.${_got_MINOR}.${_got_PATCH}")
  if(NOT _got VERSION_EQUAL PROJECT_VERSION)
    message(FATAL_ERROR
      "meshio++: ${path} says ${prefix} version ${_got}, but project() says "
      "${PROJECT_VERSION}. Bump every version file together -- see the "
      "\"Version bumps\" section of CLAUDE.md.")
  endif()
endfunction()

_mio_check_version_header(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/include/meshioplusplus/version.hpp" MESHIOPLUSPLUS)
_mio_check_version_header(
  "${CMAKE_CURRENT_SOURCE_DIR}/bindings/c/include/meshioplusplus/meshioplusplus.h" MIO)

# Hoisted to the top (it used to be included inside the C API block): the C++
# core's usage requirements are collected with an $<INSTALL_INTERFACE:> pointing
# at CMAKE_INSTALL_INCLUDEDIR long before that block is reached. Idempotent.
include(GNUInstallDirs)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Off by default: only portable release artifacts (manylinux wheel builds --
# paired with a newer-than-image GCC, e.g. gcc-toolset-13 on manylinux_2_28 --
# and the standalone CLI binary, see cli.yml) need this, to keep the binary's
# GLIBCXX/CXXABI symbol requirements within the target policy's floor (Linux)
# or avoid a vcredist dependency (MSVC static CRT). Applied below to whichever
# of _core / meshioplusplus_cli is actually being built.
option(MESHIOPLUSPLUS_STATIC_RUNTIME
  "Statically link the C++ runtime (libgcc/libstdc++ on GNU, static CRT on MSVC)" OFF)

# Threaded WebAssembly variant. Only meaningful under Emscripten: it turns on
# Wasm threads (pthreads = Web Workers + SharedArrayBuffer) so the OpenMP
# parallel backend actually runs multi-threaded in the browser/Node, and names
# the artifact meshioplusplus_wasm_mt to sit beside the sequential one. It is
# consumed only when the serving document is cross-origin isolated (COOP/COEP);
# the loader (src/wasm/src/index.mjs) auto-selects it, else falls back to the
# sequential build -- so the two artifacts ship together. build/configure-wasm.sh
# pairs it with -DMESHIOPLUSPLUS_PARALLEL_BACKEND=OPENMP in its own build tree.
option(MESHIOPLUSPLUS_WASM_THREADS
  "Emscripten only: build the threaded (pthreads/OpenMP) wasm variant" OFF)

# The pybind11 extension is the default consumer of meshioplusplus_core_obj,
# but non-Python consumers (the GoogleTest suite, the Emscripten/WASM target
# below) must be configurable without ever locating a Python interpreter or
# pybind11 -- e.g. under emcmake, where neither exists for the wasm32 target.
option(MESHIOPLUSPLUS_BUILD_PYTHON "Build the pybind11 _core extension" ON)
if(MESHIOPLUSPLUS_BUILD_PYTHON)
  # scikit-build-core provides the right Python; locate the module-only component.
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
endif()

# ZLIB is optional too: without it the VTU zlib compression path falls back to
# Python (whose zlib is always available in the stdlib). This keeps Windows CI
# and wheels buildable with no system libraries.
option(MESHIOPLUSPLUS_WITH_ZLIB "Build the C++ VTU zlib compression path" ON)

# OFF by default: an ordinary find_package(ZLIB) links whatever the system
# provides, which is normally the *shared* libz -- fine for a regular install,
# but wrong for the one dependency-free portable artifact this project ships
# (cli.yml's statically-linked release CLI binaries, wheels.yml's wheels),
# where a dynamic libz dependency defeats the entire premise. When ON, a
# pinned, SHA256-checked zlib release is fetched and built as a static lib
# instead, mirroring the MESHIOPLUSPLUS_BUILD_TESTS GoogleTest FetchContent
# below (and build-wasm-deps.sh's pin discipline for HDF5/netCDF/cgnslib).
option(MESHIOPLUSPLUS_ZLIB_STATIC
  "Vendor and statically link a pinned zlib instead of find_package(ZLIB) (for a dependency-free portable artifact: cli.yml/wheels.yml)"
  OFF)

if(MESHIOPLUSPLUS_WITH_ZLIB AND NOT EMSCRIPTEN)
  if(MESHIOPLUSPLUS_ZLIB_STATIC)
    include(FetchContent)
    FetchContent_Declare(
      meshioplusplus_zlib
      URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
      URL_HASH SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23)
    # zlib's own CMakeLists.txt always defines both `zlib` (shared) and
    # `zlibstatic` (static) with no option to skip the shared one -- only
    # zlibstatic is linked below. SKIP_INSTALL_ALL keeps its own
    # headers/lib/pkgconfig out of `cmake --install` (relevant for the wheel
    # build, which installs the whole tree); ZLIB_BUILD_EXAMPLES=OFF skips its
    # example/minigzip binaries; CMAKE_POSITION_INDEPENDENT_CODE matches the
    # PIE default most modern toolchains link executables with.
    set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)
    set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    FetchContent_MakeAvailable(meshioplusplus_zlib)
    add_library(ZLIB::ZLIB ALIAS zlibstatic)
    set(ZLIB_FOUND TRUE)
    # Same /MT-vs-/MD rule the comment near meshioplusplus_core_obj's own
    # MSVC_RUNTIME_LIBRARY property documents: every object feeding a
    # static-runtime final link must be compiled with the same CRT setting,
    # or the link fails with unresolved __imp_-prefixed CRT externals (here,
    # zlib's gzlib.c/gzread.c/gzwrite.c POSIX-name wrappers). zlib's own
    # CMakeLists.txt never sets this, so without it zlibstatic stays on the
    # MSVC default (/MD) even when MESHIOPLUSPLUS_STATIC_RUNTIME switches
    # every other target to /MT -- exactly what broke the static-runtime CLI
    # link (cli.yml's Windows leg).
    if(MESHIOPLUSPLUS_STATIC_RUNTIME AND MSVC)
      foreach(_zlib_target zlib zlibstatic)
        if(TARGET ${_zlib_target})
          set_property(TARGET ${_zlib_target} PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        endif()
      endforeach()
    endif()
  else()
    find_package(ZLIB QUIET)
  endif()
endif()
# Under Emscripten there is no system zlib to find_package() -- the bundled
# port (-sUSE_ZLIB=1) supplies both <zlib.h> and the implementation, but only
# once that flag reaches the *compiler* invocation (not just the linker), so
# it's applied directly to meshioplusplus_core_obj below instead of via
# find_package/ZLIB::ZLIB.

# Additional VTK XML block-compression codecs. Both default OFF: zlib is the
# only codec a portable build can assume, and it stays the write default, so a
# pure build reads and writes exactly what it always did. Never probed under
# Emscripten -- there is no -sUSE_ZSTD/-sUSE_LZ4 port, and a host library must
# not be picked up for a wasm target.
option(MESHIOPLUSPLUS_WITH_ZSTD "Build the VTK XML zstd compression path" OFF)
option(MESHIOPLUSPLUS_WITH_LZ4 "Build the VTK XML lz4 compression path" OFF)
if(MESHIOPLUSPLUS_WITH_ZSTD AND NOT EMSCRIPTEN)
  find_package(zstd QUIET)
  if(NOT zstd_FOUND)
    find_path(ZSTD_INCLUDE_DIR zstd.h)
    find_library(ZSTD_LIBRARY NAMES zstd libzstd)
  endif()
endif()
if(MESHIOPLUSPLUS_WITH_LZ4 AND NOT EMSCRIPTEN)
  find_package(lz4 QUIET)
  if(NOT lz4_FOUND)
    find_path(LZ4_INCLUDE_DIR lz4.h)
    find_library(LZ4_LIBRARY NAMES lz4 liblz4)
  endif()
endif()

# KaHIP (MIT), the optional quality backend of the partition operation. OFF by
# default so the default build stays dependency-free; the SFC method is the
# always-available fallback. Bring-your-own install located via the in-repo
# cmake/FindKaHIP.cmake (KAHIP_ROOT prefix / pkg-config) -- never vendored,
# never auto-downloaded. Only the serial kaffpa interface is linked (no ParHIP,
# hence no MPI). Never probed under Emscripten -- no port exists, and a host
# library must not be picked up for a wasm target.
option(MESHIOPLUSPLUS_WITH_KAHIP "Build the KaHIP partition backend" OFF)
if(MESHIOPLUSPLUS_WITH_KAHIP AND NOT EMSCRIPTEN)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  find_package(KaHIP QUIET)
endif()

# The official CGNS library (cgnslib / the CGNS Mid-Level Library). OFF by
# default and bring-your-own, exactly like KaHIP: never vendored, never
# downloaded. It is an ADDITIVE backend -- the hand-rolled ADF-over-HDF5 CGNS
# reader and writer are unchanged and remain the default -- that buys two
# things the raw-HDF5 path fundamentally cannot have: files in the ADF
# container (a raw-HDF5 reader can never open one) and NGON_n/NFACE_n
# polyhedral sections, whose CGNS 3.x-vs-4.0 ElementStartOffset split the MLL
# absorbs. Point CGNS_ROOT at an install; cgnslib ships its own CMake config
# (targets CGNS::cgns_shared / CGNS::cgns_static), so no Find module is needed
# and the exported package stays relocatable.
option(MESHIOPLUSPLUS_WITH_CGNSLIB
       "Use the official CGNS library for CGNS (ADF containers + NGON_n/NFACE_n)" OFF)
if(MESHIOPLUSPLUS_WITH_CGNSLIB)
  find_package(CGNS CONFIG QUIET)
  if(NOT TARGET CGNS::cgns_shared AND NOT TARGET CGNS::cgns_static AND CGNS_ROOT)
    # Some installs predate the config package; fall back to a plain probe.
    find_path(CGNS_INCLUDE_DIR cgnslib.h HINTS "${CGNS_ROOT}/include")
    find_library(CGNS_LIBRARY NAMES cgns HINTS "${CGNS_ROOT}/lib")
  endif()
endif()

# Optional heavy dependencies. When absent, the corresponding format sources
# compile to empty translation units (#ifdef-guarded) and the Python
# implementations serve as the runtime fallback.
option(MESHIOPLUSPLUS_WITH_HDF5 "Build the HDF5-backed formats (CGNS, HMF, H5M, MED, XDMF-HDF)" ON)
option(MESHIOPLUSPLUS_WITH_NETCDF "Build the netCDF-backed formats (Exodus)" ON)

if(MESHIOPLUSPLUS_WITH_HDF5)
  if(EMSCRIPTEN)
    # There is only ever one HDF5 for this target: the static wasm32 build
    # produced by build/build-wasm-deps.sh, which build/configure-wasm.sh puts
    # on CMAKE_FIND_ROOT_PATH. It is serial by construction, so the
    # MPI-flavour retry below has nothing to find and the MPI probe nothing to
    # guard against -- both are skipped rather than left to mis-fire. Module
    # mode is kept (not the installed hdf5-config.cmake) so the
    # HDF5_INCLUDE_DIRS/HDF5_C_LIBRARIES plumbing further down stays identical
    # for every platform.
    set(HDF5_USE_STATIC_LIBRARIES ON)
    find_package(HDF5 QUIET COMPONENTS C)
  else()
    find_package(HDF5 QUIET COMPONENTS C)
    if(NOT HDF5_FOUND)
      # Some distros ship only an MPI-flavoured HDF5 (e.g. Debian's
      # libhdf5-openmpi-dev); FindHDF5 prefers serial by default, so retry.
      set(HDF5_PREFER_PARALLEL ON)
      find_package(HDF5 QUIET COMPONENTS C)
    endif()
    # A parallel HDF5 needs mpi.h even for serial use of the API.
    if(HDF5_FOUND AND HDF5_IS_PARALLEL)
      find_package(MPI QUIET COMPONENTS C)
      if(NOT MPI_C_FOUND)
        set(HDF5_FOUND FALSE)  # unusable without MPI headers -> Python fallback
      endif()
    endif()
  endif()
endif()

# gidpost's OPTIONAL HDF5 flavour needs the HDF5 *high-level* library
# (H5LTset_attribute_string and friends, in libhdf5_hl) in addition to the
# core C API every other HDF5-backed format here already links -- and unlike
# those formats, gidpost's own hdf5c.c calls H5LT* directly. This is
# deliberately a SEPARATE, best-effort probe rather than widening the
# find_package(HDF5 COMPONENTS C) call above to add HL: requesting an
# unavailable component there can flip the whole-package HDF5_FOUND false,
# which would silently disable CGNS/HMF/H5M/MED/XDMF-HDF for every user on a
# system that has hdf5_hl missing -- a blast radius far beyond gidpost's own
# HDF5 flavour, which is what MESHIOPLUSPLUS_HAS_GIDPOST_HDF5 exists to gate
# instead (Block A, below, auto-disables just that one flavour when this
# probe comes up empty).
if(HDF5_FOUND AND NOT EMSCRIPTEN)
  set(_mio_hdf5_hl_hints "${HDF5_LIBRARY_DIRS}")
  if(HDF5_C_LIBRARIES)
    list(GET HDF5_C_LIBRARIES 0 _mio_hdf5_first_lib)
    get_filename_component(_mio_hdf5_lib_dir "${_mio_hdf5_first_lib}" DIRECTORY)
    list(APPEND _mio_hdf5_hl_hints "${_mio_hdf5_lib_dir}")
  endif()
  # No preceding set(MESHIOPLUSPLUS_HDF5_HL_LIBRARY ...): a plain set() of the
  # SAME name as a find_library() cache variable shadows the cache entry for
  # every later read in this directory scope, even though find_library()
  # still (successfully) populates the cache -- reproduced in isolation, not
  # a theoretical concern. NOTFOUND (find_library's own default when nothing
  # is set yet) is what makes the plain `if(... AND MESHIOPLUSPLUS_HDF5_HL_LIBRARY)`
  # checks below work correctly with no separate not-found variable needed.
  find_library(MESHIOPLUSPLUS_HDF5_HL_LIBRARY NAMES hdf5_hl HINTS ${_mio_hdf5_hl_hints})
endif()
if(MESHIOPLUSPLUS_WITH_NETCDF)
  # Also look in ~/.local for a user-built netcdf-c (no-sudo installs).
  find_package(netCDF CONFIG QUIET HINTS $ENV{HOME}/.local/lib/cmake/netCDF)
  if(NOT netCDF_FOUND)
    find_library(NETCDF_LIBRARY netcdf PATHS $ENV{HOME}/.local/lib)
    find_path(NETCDF_INCLUDE_DIR netcdf.h PATHS $ENV{HOME}/.local/include)
  endif()
endif()

# Polyscope (MIT), the native viewer behind the CLI's `view`/`screenshot`
# verbs. OFF by default and, unlike every other optional dependency here,
# attached to the **CLI target only** -- never to meshioplusplus_core_obj. The
# core, the C API, the Fortran module and the wasm build therefore cannot see
# it at all, which is what keeps "the core is dependency-free" literally true
# rather than true-by-default: Polyscope pulls in OpenGL, GLFW and X11.
#
# Vendored as a git submodule at src/cpp/third_party/polyscope (it is in no package
# manager and its own docs offer only add_subdirectory/FetchContent). It has
# nested submodules, so `git submodule update --init --recursive` is required;
# with a plain --init the EXISTS guard below simply leaves the viewer out.
option(MESHIOPLUSPLUS_WITH_POLYSCOPE
       "Build the CLI's native viewer (needs OpenGL/GLFW; see src/cpp/third_party/polyscope)"
       OFF)

# --------------------------------------------------------------------------
# MESHIOPLUSPLUS_WITH_GIDPOST: the GiD postprocess writer, on a hardcopy of
# CIMNE's gidpost 2.14 (src/cpp/third_party/gidpost, BSD-2-Clause-Views).
#
# Hard-gated on zlib, not by preference: gidpostInt.h includes <zlib.h>
# UNCONDITIONALLY and the binary flavour is always deflated, so zlib is a
# build prerequisite for the whole library, not for one of its three modes.
#
# NOT a FATAL_ERROR when zlib is missing, deliberately. Several existing CI
# legs build with -DMESHIOPLUSPLUS_WITH_ZLIB=OFF on purpose (the statically-
# linked release CLI binaries and the Windows wheels), and a configure-time
# hard error would break all of them. This repo reserves FATAL_ERROR for a
# contradiction the *user authored* (MESHIOPLUSPLUS_WITH_CGNSLIB without
# MESHIOPLUSPLUS_WITH_HDF5, below); a default-ON option meeting an absent
# optional dependency is the KaHIP/cgnslib "requested but not found -- will
# raise by name" family instead. The no-silent-downgrade contract is kept at
# RUN time: write_gid() always exists and throws naming both flags (the
# partition_kahip_parts contract), and `gid` stays registered in
# registry_writers()/registry_extension_defaults() so `.post.msh` can never
# quietly resolve to gmsh. The *explicit* contradiction is caught earlier, in
# build/configure.sh and build/configure-wasm.sh, where --with-gidpost is a
# flag the user actually typed.
# --------------------------------------------------------------------------
option(MESHIOPLUSPLUS_WITH_GIDPOST
       "Build the GiD postprocess writer (vendored gidpost; requires zlib)" ON)

set(_mio_gidpost_dir "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/third_party/gidpost/source")
set(_mio_gidpost_enabled OFF)
set(_mio_gidpost_hdf5 OFF)
set(_mio_gidpost_sources "")

if(MESHIOPLUSPLUS_WITH_GIDPOST)
  if(NOT EXISTS "${_mio_gidpost_dir}/gidpost.c")
    message(STATUS "meshio++: gidpost sources not present - the 'gid' writer will raise")
  elseif(NOT MESHIOPLUSPLUS_WITH_ZLIB)
    message(STATUS "meshio++: gidpost needs zlib (MESHIOPLUSPLUS_WITH_ZLIB=OFF) - "
                   "the 'gid' writer will raise by name")
  elseif(NOT (EMSCRIPTEN OR ZLIB_FOUND))
    message(STATUS "meshio++: gidpost needs zlib (not found) - "
                   "the 'gid' writer will raise by name")
  else()
    set(_mio_gidpost_enabled ON)
    set(_mio_gidpost_sources
        "${_mio_gidpost_dir}/gidpost.c"
        "${_mio_gidpost_dir}/gidpostFILES.c"
        "${_mio_gidpost_dir}/gidpostHash.c"
        "${_mio_gidpost_dir}/gidpostInt.c"
        "${_mio_gidpost_dir}/hashtab.c"
        "${_mio_gidpost_dir}/lookupa.c"
        "${_mio_gidpost_dir}/recycle.c")

    # The HDF5 flavour rides MESHIOPLUSPLUS_WITH_HDF5, but ALSO needs the HDF5
    # high-level library specifically (gidpost's hdf5c.c calls H5LT* directly,
    # unlike every other HDF5-backed format here) -- see the
    # MESHIOPLUSPLUS_HDF5_HL_LIBRARY probe above. Two files are added AND a
    # macro is defined -- both are required: gidpostHDF5.c does not self-gate
    # (its own `#ifdef ENABLE_HDF5` is commented out) while the header it
    # implements IS gated, so compiling it without ENABLE_HDF5 fails. Upstream
    # source/CMakeLists.txt makes the same file list conditional.
    if(MESHIOPLUSPLUS_WITH_HDF5 AND HDF5_FOUND AND MESHIOPLUSPLUS_HDF5_HL_LIBRARY)
      set(_mio_gidpost_hdf5 ON)
      list(APPEND _mio_gidpost_sources
           "${_mio_gidpost_dir}/gidpostHDF5.c"
           "${_mio_gidpost_dir}/hdf5c.c")
    elseif(MESHIOPLUSPLUS_WITH_HDF5 AND HDF5_FOUND)
      message(STATUS "meshio++: gidpost's HDF5 flavour needs libhdf5_hl (not found) - "
                     "the 'gid' writer will still build, minus GidMode::Hdf5")
    endif()

    # NDEBUG unconditionally, in every configuration:
    #  * GiD_PostInit() printf()s "Debug version: ..." to stdout when NDEBUG is
    #    absent (gidpost.c), which would corrupt CLI stdout in a Debug build;
    #  * gidpost validates its own state machine with assert() throughout
    #    (gidpostFILES.c, gidpostInt.c). An abort() inside a Python extension
    #    or the WASM module kills the host instead of raising, which
    #    contradicts this project's errors-are-exceptions contract. We
    #    validate every argument before the call and check every return code,
    #    so the asserts buy us nothing and cost us the process.
    # ENABLE_HDF5 is set per-source, never target-wide: it is a foreign macro
    # name and _mio_core_defs is applied PUBLIC.
    set(_mio_gidpost_src_defs NDEBUG)
    if(_mio_gidpost_hdf5)
      list(APPEND _mio_gidpost_src_defs ENABLE_HDF5)
    endif()
    set_source_files_properties(${_mio_gidpost_sources}
      PROPERTIES COMPILE_DEFINITIONS "${_mio_gidpost_src_defs}")

    message(STATUS "meshio++: GiD postprocess writer enabled "
                   "(gidpost 2.14, HDF5 flavour: ${_mio_gidpost_hdf5})")
  endif()
endif()

# --------------------------------------------------------------------------
# meshioplusplus_core_obj: the pybind11-free C++ core (format readers/writers + pugixml).
# It is compiled once and shared by the Python extension (`_core`) and, when
# MESHIOPLUSPLUS_BUILD_TESTS is on, the standalone GoogleTest binary. All include
# dirs / feature flags / optional-library links are attached PUBLIC here so
# both consumers inherit them.
# --------------------------------------------------------------------------
file(GLOB_RECURSE MESHIOPLUSPLUS_CORE_SOURCES
    src/cpp/src/*.cpp
    src/cpp/third_party/pugixml/pugixml.cpp)

# Vendored gidpost's C sources join the core source list rather than a target
# of their own: that is what makes them inherit Emscripten's -sUSE_ZLIB=1 from
# _mio_core_copts (without it there is no <zlib.h> at all on the wasm leg), and
# it sidesteps install(EXPORT)'s refusal to export a target whose PRIVATE
# static-library dependency is unexported (see meshioplusplus_add_core_variant
# below).
list(APPEND MESHIOPLUSPLUS_CORE_SOURCES ${_mio_gidpost_sources})

add_library(meshioplusplus_core_obj OBJECT ${MESHIOPLUSPLUS_CORE_SOURCES})
set_target_properties(meshioplusplus_core_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --------------------------------------------------------------------------
# Usage-requirement accumulators for the C++ core.
#
# Every optional-dependency probe below appends here instead of calling
# target_*(meshioplusplus_core_obj PUBLIC ...) directly, so the identical set can
# be applied to more than one target by meshioplusplus_configure_core_target():
# the in-tree meshioplusplus_core_obj (Python `_core`, WASM, CLI, gtest -- exactly
# as before) and, when MESHIOPLUSPLUS_INSTALL_CPP=ON, one installable
# meshioplusplus_core_<backend> library per mesh backend.
#
# This indirection is what makes the C++ install work at all. The C API
# deliberately keeps the core's usage requirements OUT of its exported interface
# (its installed surface is the C header alone -- see the PRIVATE link comment
# further down), but a find_package(... COMPONENTS CXX) consumer compiles against
# the real headers and therefore needs the include dirs, the backend/parallel
# macros and the HDF5/netCDF/... link deps to reach it. Re-declaring them on the
# exported library is the only way to get them there: install(EXPORT) refuses a
# target whose INTERFACE_LINK_LIBRARIES names an unexported target, so the object
# library can be neither exported nor PUBLIC-linked.
#
# _mio_core_priv_incs is the exception that must NEVER reach an exported target:
# the vendored pugixml/Eigen include dirs are build-only (no installed header
# includes either) and Eigen's is a git-submodule path that does not exist on a
# consumer's machine at all.
# --------------------------------------------------------------------------
set(_mio_core_defs "")
set(_mio_core_incs "")
set(_mio_core_priv_incs "")
set(_mio_core_libs "")
set(_mio_core_copts "")
set(_mio_core_lopts "")

# Several operations (interpolate's barycentric weighted sum, merge/partition's
# weighted accumulations, ...) are pinned byte-identical against a pure-Python/
# numpy twin, which never fuses a multiply-add into one rounding step. x86_64
# has no FMA instruction unless a target flag (-mfma/-march=...) opts in, so
# this was silently never an issue on the Linux/Windows CI legs -- but AArch64
# (e.g. GitHub's arm64 macos-latest runners) has FMA in baseline hardware, and
# Clang's default -ffp-contract=on permits fusing `acc = acc + w * v` within a
# single expression, changing the rounding by up to 1 ulp relative to numpy's
# separately-rounded multiply and add (this is exactly what broke
# tests/python/test_interpolate.py::test_cpp_matches_python on macOS). Disable
# contraction outright so this class of expression rounds identically on every
# target; there is no MSVC equivalent needed since /fp:precise (the default)
# never contracts.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  list(APPEND _mio_core_copts -ffp-contract=off)
endif()

# On MSVC, /MT vs /MD is a *compile*-time flag baked into each .obj (unlike
# GNU's -static-libgcc/-static-libstdc++, which is link-only) -- every object
# library feeding a static-runtime final link must be compiled with the same
# setting, or the link fails with unresolved __imp_-prefixed CRT externals
# (nearbyint/isnan/ecvt_s/... -- the dynamic-CRT objects expect an import lib
# that a static-CRT link doesn't provide). Never applied when building the
# Python extension: pybind11 modules must always match the interpreter's own
# (dynamic) CRT, and Windows wheels never set MESHIOPLUSPLUS_STATIC_RUNTIME.
if(MESHIOPLUSPLUS_STATIC_RUNTIME AND MSVC AND NOT MESHIOPLUSPLUS_BUILD_PYTHON)
  set_property(TARGET meshioplusplus_core_obj PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
list(APPEND _mio_core_incs
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/include>
     $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# pugixml is vendored and reached by four format .cpp files only -- no installed
# header names a pugi:: type (`grep -r 'pugi::' src/cpp/include` is empty), so
# this is PRIVATE. A PUBLIC entry would put the vendored copy on every consumer's
# include path once the C++ API is installed, colliding with a consumer that
# vendors its own pugixml (Kratos does).
list(APPEND _mio_core_priv_incs
     ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/third_party/pugixml)

if(EMSCRIPTEN)
  # Emscripten disables JS-catchable C++ exceptions by default (an uncaught
  # exception calls abort() instead of unwinding into a JS Error) -- readers/
  # writers here throw ReadError/WriteError routinely (e.g. on an unsupported
  # construct), and bindings/wasm/js_bindings.cpp relies on those surfacing as
  # normal catchable JS exceptions. -fwasm-exceptions enables the native Wasm
  # exception-handling proposal (supported by all current browsers/Node),
  # which is faster than the legacy JS-longjmp emulation and needs no
  # separate -sDISABLE_EXCEPTION_CATCHING= linker flag. Applied PUBLIC so both
  # the compile step (here) and whatever links these objects inherit it.
  list(APPEND _mio_core_copts "-fwasm-exceptions")
  list(APPEND _mio_core_lopts "-fwasm-exceptions")
endif()

# --------------------------------------------------------------------------
# gcov/lcov instrumentation for the coverage CI job. OFF by default, so a
# normal `pip install` never pays for it. Applied PUBLIC so every consumer of
# these objects (_core, meshioplusplus_tests) is instrumented and emits into
# the same .gcno/.gcda set -- that is what lets one lcov capture cover the C++
# core as exercised by BOTH the GoogleTest binary and the pytest suite.
# -fprofile-update=atomic is required: meshioplusplus::parallel_for runs the
# instrumented loops on several threads, and the default non-atomic counter
# updates race and silently under-count.
# --------------------------------------------------------------------------
option(MESHIOPLUSPLUS_COVERAGE "Instrument the C++ core for gcov/lcov coverage" OFF)
if(MESHIOPLUSPLUS_COVERAGE)
  if(NOT (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
    message(FATAL_ERROR "meshio++: MESHIOPLUSPLUS_COVERAGE needs GCC or Clang, got ${CMAKE_CXX_COMPILER_ID}")
  endif()
  message(STATUS "meshio++: coverage instrumentation ON (--coverage, -O0)")
  list(APPEND _mio_core_copts --coverage -O0 -g -fprofile-update=atomic)
  list(APPEND _mio_core_lopts --coverage)
endif()

if(MESHIOPLUSPLUS_WITH_ZLIB AND EMSCRIPTEN)
  message(STATUS "meshio++: VTU zlib compression enabled (Emscripten -sUSE_ZLIB=1 port)")
  list(APPEND _mio_core_defs MESHIOPLUSPLUS_HAS_ZLIB)
  list(APPEND _mio_core_copts "-sUSE_ZLIB=1")
  list(APPEND _mio_core_lopts "-sUSE_ZLIB=1")
elseif(MESHIOPLUSPLUS_WITH_ZLIB AND ZLIB_FOUND)
  message(STATUS "meshio++: VTU zlib compression enabled")
  list(APPEND _mio_core_defs MESHIOPLUSPLUS_HAS_ZLIB)
  list(APPEND _mio_core_libs ZLIB::ZLIB)
else()
  message(STATUS "meshio++: zlib not found/disabled - VTU zlib falls back to Python")
endif()

if(MESHIOPLUSPLUS_WITH_ZSTD AND (zstd_FOUND OR ZSTD_LIBRARY))
  message(STATUS "meshio++: VTK XML zstd compression enabled")
  list(APPEND _mio_core_defs MESHIOPLUSPLUS_HAS_ZSTD)
  if(zstd_FOUND)
    list(APPEND _mio_core_libs zstd::libzstd_shared)
  else()
    list(APPEND _mio_core_incs ${ZSTD_INCLUDE_DIR})
    list(APPEND _mio_core_libs ${ZSTD_LIBRARY})
  endif()
elseif(MESHIOPLUSPLUS_WITH_ZSTD)
  message(STATUS "meshio++: zstd requested but not found - zstd files fall back to Python")
endif()

if(MESHIOPLUSPLUS_WITH_LZ4 AND (lz4_FOUND OR LZ4_LIBRARY))
  message(STATUS "meshio++: VTK XML lz4 compression enabled")
  list(APPEND _mio_core_defs MESHIOPLUSPLUS_HAS_LZ4)
  if(lz4_FOUND)
    list(APPEND _mio_core_libs LZ4::lz4_shared)
  else()
    list(APPEND _mio_core_incs ${LZ4_INCLUDE_DIR})
    list(APPEND _mio_core_libs ${LZ4_LIBRARY})
  endif()
elseif(MESHIOPLUSPLUS_WITH_LZ4)
  message(STATUS "meshio++: lz4 requested but not found - lz4 files fall back to Python")
endif()

if(MESHIOPLUSPLUS_WITH_KAHIP AND KAHIP_FOUND)
  message(STATUS "meshio++: KaHIP partition backend enabled (${KAHIP_LIBRARIES})")
  list(APPEND _mio_core_defs MESHIOPLUSPLUS_HAS_KAHIP)
  list(APPEND _mio_core_libs kahip::kahip)
elseif(MESHIOPLUSPLUS_WITH_KAHIP)
  message(STATUS "meshio++: KaHIP requested but not found (set KAHIP_ROOT) - "
                 "partition method 'kahip' will raise")
endif()

if(MESHIOPLUSPLUS_WITH_CGNSLIB AND NOT MESHIOPLUSPLUS_WITH_HDF5)
  message(FATAL_ERROR
          "meshio++: MESHIOPLUSPLUS_WITH_CGNSLIB needs MESHIOPLUSPLUS_WITH_HDF5=ON - "
          "the cgnslib backend augments the HDF5-backed CGNS format, it does not replace it.")
endif()
if(MESHIOPLUSPLUS_WITH_CGNSLIB AND TARGET CGNS::cgns_shared)
  message(STATUS "meshio++: cgnslib CGNS backend enabled (shared)")
  list(APPEND _mio_core_defs MESHIOPLUSPLUS_HAS_CGNSLIB)
  list(APPEND _mio_core_libs CGNS::cgns_shared)
elseif(MESHIOPLUSPLUS_WITH_CGNSLIB AND TARGET CGNS::cgns_static)
  message(STATUS "meshio++: cgnslib CGNS backend enabled (static)")
  list(APPEND _mio_core_defs MESHIOPLUSPLUS_HAS_CGNSLIB)
  list(APPEND _mio_core_libs CGNS::cgns_static)
elseif(MESHIOPLUSPLUS_WITH_CGNSLIB AND CGNS_LIBRARY AND CGNS_INCLUDE_DIR)
  message(STATUS "meshio++: cgnslib CGNS backend enabled (${CGNS_LIBRARY})")
  list(APPEND _mio_core_defs MESHIOPLUSPLUS_HAS_CGNSLIB)
  list(APPEND _mio_core_incs ${CGNS_INCLUDE_DIR})
  list(APPEND _mio_core_libs ${CGNS_LIBRARY})
elseif(MESHIOPLUSPLUS_WITH_CGNSLIB)
  message(STATUS "meshio++: cgnslib requested but not found (set CGNS_ROOT) - "
                 "ADF containers and NGON_n/NFACE_n sections will raise by name")
endif()

if(MESHIOPLUSPLUS_WITH_HDF5 AND HDF5_FOUND)
  message(STATUS "meshio++: HDF5 formats enabled (HDF5 ${HDF5_VERSION})")
  list(APPEND _mio_core_defs MESHIOPLUSPLUS_HAS_HDF5)
  # Prefer the imported target: it makes the exported package RELOCATABLE.
  # Linking ${HDF5_C_LIBRARIES} writes absolute library paths straight into
  # meshioplusplusTargets.cmake, which then only resolves on a machine whose
  # HDF5 sits exactly where this build's did. HDF5::HDF5 instead records a
  # dependency the consumer re-finds locally (see the find_dependency in
  # meshioplusplusConfig.cmake.in). FindHDF5 has provided it since CMake 3.19;
  # older CMake falls back to the historical raw-path behaviour.
  if(TARGET HDF5::HDF5)
    list(APPEND _mio_core_libs HDF5::HDF5)
    set(_mio_hdf5_imported TRUE)
  else()
    list(APPEND _mio_core_incs ${HDF5_INCLUDE_DIRS})
    list(APPEND _mio_core_libs ${HDF5_C_LIBRARIES})
  endif()
  if(HDF5_IS_PARALLEL AND MPI_C_FOUND)
    list(APPEND _mio_core_incs ${MPI_C_INCLUDE_DIRS})
    list(APPEND _mio_core_libs MPI::MPI_C)
    set(_mio_hdf5_needs_mpi TRUE)
  endif()
else()
  message(STATUS "meshio++: HDF5 not found/disabled - HDF5 formats fall back to Python")
endif()

if(MESHIOPLUSPLUS_WITH_NETCDF AND (netCDF_FOUND OR NETCDF_LIBRARY))
  message(STATUS "meshio++: netCDF formats enabled")
  list(APPEND _mio_core_defs MESHIOPLUSPLUS_HAS_NETCDF)
  if(netCDF_FOUND)
    list(APPEND _mio_core_libs netCDF::netcdf)
  else()
    list(APPEND _mio_core_incs ${NETCDF_INCLUDE_DIR})
    list(APPEND _mio_core_libs ${NETCDF_LIBRARY})
  endif()
else()
  message(STATUS "meshio++: netCDF not found/disabled - Exodus falls back to Python")
endif()

# Eigen (header-only, vendored as a git submodule at src/cpp/third_party/eigen):
# used for the MED Fortran<->C transpose. Optional -- when the submodule is not
# checked out (e.g. an sdist without submodules) the code falls back to the
# hand-written transpose loop. Run `git submodule update --init` to enable it.
option(MESHIOPLUSPLUS_WITH_EIGEN "Use vendored Eigen for the MED transpose" ON)
if(MESHIOPLUSPLUS_WITH_EIGEN AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/third_party/eigen/Eigen/Dense")
  message(STATUS "meshio++: Eigen enabled (MED transpose)")
  list(APPEND _mio_core_defs MESHIOPLUSPLUS_HAS_EIGEN)
  # PRIVATE, like pugixml: Eigen is reached only by src/cpp/src/formats/med.cpp
  # and no installed header names an Eigen type. It matters more here than for
  # pugixml -- this is a git-submodule path inside the source tree, so exporting
  # it would bake a directory that simply does not exist on a consumer's machine
  # into meshioplusplusTargets.cmake and break find_package() outright.
  list(APPEND _mio_core_priv_incs
       ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/third_party/eigen)
else()
  message(STATUS "meshio++: Eigen not found/disabled - MED transpose uses the plain loop")
endif()

# nlohmann/json (header-only, vendored as a git submodule at
# src/cpp/third_party/json, pinned to v3.12.0): used only by the JSON pipeline
# front-end (operations/pipeline.cpp's parse_pipeline_*). Optional -- when the
# submodule is not checked out (e.g. an sdist without submodules) the typed
# pipeline API still compiles and only the JSON-parsing entry points throw,
# naming this option. Run `git submodule update --init` to enable it.
option(MESHIOPLUSPLUS_WITH_JSON "Build the JSON pipeline parser (vendored nlohmann/json)" ON)
if(MESHIOPLUSPLUS_WITH_JSON AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/third_party/json/single_include/nlohmann/json.hpp")
  message(STATUS "meshio++: nlohmann/json enabled (pipeline settings parser)")
  list(APPEND _mio_core_defs MESHIOPLUSPLUS_HAS_JSON)
  # PRIVATE for exactly Eigen's reason above: a submodule path inside the
  # source tree must never leak into meshioplusplusTargets.cmake, and no
  # installed header names an nlohmann type (pipeline.cpp only).
  list(APPEND _mio_core_priv_incs
       ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/third_party/json/single_include)
else()
  message(STATUS "meshio++: nlohmann/json not found/disabled - pipeline JSON parsing raises")
endif()

# gidpost's include dir is PRIVATE for pugixml's exact reason above: only
# src/cpp/src/formats/gid.cpp names a GiD_ type, no installed header does, and
# a PUBLIC entry would put the vendored copy on the include path of a
# consumer that vendors its own gidpost -- Kratos does, at
# external_libraries/gidpost. The MACRO is PUBLIC, as with HAS_JSON/HAS_EIGEN.
if(_mio_gidpost_enabled)
  list(APPEND _mio_core_defs MESHIOPLUSPLUS_HAS_GIDPOST)
  if(_mio_gidpost_hdf5)
    list(APPEND _mio_core_defs MESHIOPLUSPLUS_HAS_GIDPOST_HDF5)
    # hdf5c.c calls H5LT* directly (see the MESHIOPLUSPLUS_HDF5_HL_LIBRARY
    # probe above) -- the core C API alone (HDF5::HDF5 / HDF5_C_LIBRARIES,
    # linked below for the other HDF5-backed formats) does not export those
    # symbols.
    list(APPEND _mio_core_libs "${MESHIOPLUSPLUS_HDF5_HL_LIBRARY}")
  endif()
  list(APPEND _mio_core_priv_incs "${_mio_gidpost_dir}")
endif()

# --------------------------------------------------------------------------
# Parallel backend for meshioplusplus::parallel_for (src/cpp/include/meshioplusplus/parallel.hpp).
# Selected at configure time; exactly one MESHIOPLUSPLUS_PARALLEL_* macro is defined:
#   AUTO   - (default) OpenMP if available, else STL, else SEQ. Prefers OpenMP
#            because it is portable and needs no TBB; STL without TBB would
#            silently run sequentially.
#   SEQ    - sequential (always available)
#   STL    - C++17 std::execution::par. Built into MSVC's STL; on
#            libstdc++ it requires TBB - probed below, with an automatic
#            fallback to SEQ (warning) when unusable (e.g. Apple libc++).
#   OPENMP - #pragma omp parallel for (hard error if OpenMP is missing)
#   TBB    - tbb::parallel_for        (hard error if TBB is missing)
#   KOKKOS - Kokkos::parallel_for on the HOST execution space (hard error if
#            Kokkos is missing). Bring-your-own like KaHIP: never picked by
#            AUTO, located via find_package(Kokkos CONFIG) — point Kokkos_DIR
#            or CMAKE_PREFIX_PATH at an installed Kokkos. Host-space only:
#            loop bodies capture host pointers, so device execution is served
#            by the DLPack/CuPy handoff (doc/gpu.md), not by this backend.
# Adding a new backend (HPX, ...) = a branch here + one #elif block in
# src/cpp/include/meshioplusplus/parallel.hpp.
# --------------------------------------------------------------------------
set(MESHIOPLUSPLUS_PARALLEL_BACKEND "AUTO" CACHE STRING
    "Parallel backend for meshioplusplus::parallel_for: AUTO, SEQ, STL, OPENMP, TBB or KOKKOS")
set_property(CACHE MESHIOPLUSPLUS_PARALLEL_BACKEND PROPERTY STRINGS AUTO SEQ STL OPENMP TBB KOKKOS)
string(TOUPPER "${MESHIOPLUSPLUS_PARALLEL_BACKEND}" _meshioplusplus_parallel)

# AUTO: prefer OpenMP (portable: libgomp on manylinux, MSVC built-in, libomp on
# macOS; needs no TBB), then the STL(+TBB) path, else the sequential backend.
if(_meshioplusplus_parallel STREQUAL "AUTO")
  find_package(OpenMP QUIET COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    set(_meshioplusplus_parallel "OPENMP")
  else()
    set(_meshioplusplus_parallel "STL")
  endif()
endif()

if(_meshioplusplus_parallel STREQUAL "STL")
  if(MSVC)
    list(APPEND _mio_core_defs MESHIOPLUSPLUS_PARALLEL_STL)
  else()
    find_package(TBB CONFIG QUIET)
    include(CheckCXXSourceCompiles)
    if(TBB_FOUND)
      set(CMAKE_REQUIRED_LIBRARIES TBB::tbb)
    endif()
    check_cxx_source_compiles(
      "
      #include <algorithm>
      #include <execution>
      #include <vector>
      int main() {
        std::vector<int> v(4);
        std::for_each(std::execution::par, v.begin(), v.end(), [](int& x) { x = 1; });
        return v[0] - 1;
      }"
      MESHIOPLUSPLUS_HAS_PSTL)
    unset(CMAKE_REQUIRED_LIBRARIES)
    if(MESHIOPLUSPLUS_HAS_PSTL)
      list(APPEND _mio_core_defs MESHIOPLUSPLUS_PARALLEL_STL)
      if(TBB_FOUND)
        list(APPEND _mio_core_libs TBB::tbb)
        set(_mio_parallel_needs_tbb TRUE)
      endif()
    else()
      message(WARNING
        "meshio++: the STL parallel backend is unusable on this toolchain "
        "(libstdc++ needs TBB installed; Apple libc++ has no parallel STL) - "
        "falling back to the sequential backend. Install TBB or configure with "
        "-DMESHIOPLUSPLUS_PARALLEL_BACKEND=OPENMP.")
      set(_meshioplusplus_parallel "SEQ")
      list(APPEND _mio_core_defs MESHIOPLUSPLUS_PARALLEL_SEQ)
    endif()
  endif()
elseif(_meshioplusplus_parallel STREQUAL "OPENMP")
  list(APPEND _mio_core_defs MESHIOPLUSPLUS_PARALLEL_OPENMP)
  if(EMSCRIPTEN)
    # The Emscripten toolchain ships no OpenMP CMake package to find_package();
    # its OpenMP is enabled by -fopenmp on top of Wasm threads, which -pthread
    # turns on (Web Workers + SharedArrayBuffer + a shared-memory build). Both
    # flags are whole-translation-unit properties, so they go PUBLIC on the core
    # object library (every format/operation TU) and on whatever links it (the
    # wasm binding). This branch is only reached in the dedicated threaded wasm
    # tree (MESHIOPLUSPLUS_WASM_THREADS=ON); the Python/gtest/C-API trees
    # configure with SEQ or a native backend, so -pthread never leaks into them.
    if(NOT MESHIOPLUSPLUS_WASM_THREADS)
      message(FATAL_ERROR
        "meshio++: the OPENMP parallel backend under Emscripten requires "
        "-DMESHIOPLUSPLUS_WASM_THREADS=ON (it needs Wasm threads/pthreads).")
    endif()
    list(APPEND _mio_core_copts "-fopenmp" "-pthread")
    list(APPEND _mio_core_lopts "-fopenmp" "-pthread")
  else()
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    list(APPEND _mio_core_libs OpenMP::OpenMP_CXX)
  endif()
elseif(_meshioplusplus_parallel STREQUAL "TBB")
  find_package(TBB CONFIG REQUIRED)
  list(APPEND _mio_core_defs MESHIOPLUSPLUS_PARALLEL_TBB)
  list(APPEND _mio_core_libs TBB::tbb)
  set(_mio_parallel_needs_tbb TRUE)
elseif(_meshioplusplus_parallel STREQUAL "KOKKOS")
  if(EMSCRIPTEN)
    message(FATAL_ERROR
      "meshio++: the KOKKOS parallel backend is not supported under Emscripten (use SEQ)")
  endif()
  find_package(Kokkos REQUIRED CONFIG)
  list(APPEND _mio_core_defs MESHIOPLUSPLUS_PARALLEL_KOKKOS)
  list(APPEND _mio_core_libs Kokkos::kokkos)
elseif(_meshioplusplus_parallel STREQUAL "SEQ")
  list(APPEND _mio_core_defs MESHIOPLUSPLUS_PARALLEL_SEQ)
else()
  message(FATAL_ERROR
    "meshio++: unknown MESHIOPLUSPLUS_PARALLEL_BACKEND '${MESHIOPLUSPLUS_PARALLEL_BACKEND}' "
    "(use SEQ, STL, OPENMP, TBB or KOKKOS)")
endif()
message(STATUS "meshio++: parallel backend: ${_meshioplusplus_parallel}")

# ---------------------------------------------------------------------------
# In-memory mesh backend (src/cpp/include/meshioplusplus/mesh.hpp). Exactly one
# MESHIOPLUSPLUS_MESH_BACKEND_* macro is defined; all format code is written
# against the uniform mesh API (mesh_api.hpp) so any backend compiles:
#   MESHIO - (default) the meshio-mirroring Mesh/CellBlock over dtype-erased
#            NDArrays. REQUIRED when MESHIOPLUSPLUS_BUILD_PYTHON=ON (the
#            zero-copy numpy boundary in bindings/python/np_conversions.hpp is
#            written against it).
#   NATIVE - canonical statically-typed storage (Float64 points, Int64
#            connectivity, CellType enum, CSR ragged blocks). The fastest
#            pure-C++ consumer surface; the WebAssembly build uses it.
#   KRATOS - a Kratos-Multiphysics-style ModelPart (Nodes/Elements/
#            Conditions/SubModelParts) behind the same API, for near-costless
#            exchange with Kratos (see kratos_bridge.hpp).
# ---------------------------------------------------------------------------
set(MESHIOPLUSPLUS_MESH_BACKEND "MESHIO" CACHE STRING
    "In-memory mesh backend: MESHIO, NATIVE or KRATOS")
set_property(CACHE MESHIOPLUSPLUS_MESH_BACKEND PROPERTY STRINGS MESHIO NATIVE KRATOS)
string(TOUPPER "${MESHIOPLUSPLUS_MESH_BACKEND}" _meshioplusplus_mesh_backend)

if(MESHIOPLUSPLUS_BUILD_PYTHON AND NOT _meshioplusplus_mesh_backend STREQUAL "MESHIO")
  message(FATAL_ERROR
    "meshio++: the pybind11 extension (MESHIOPLUSPLUS_BUILD_PYTHON=ON) requires "
    "MESHIOPLUSPLUS_MESH_BACKEND=MESHIO (got '${MESHIOPLUSPLUS_MESH_BACKEND}'). "
    "Configure with -DMESHIOPLUSPLUS_BUILD_PYTHON=OFF to use the NATIVE/KRATOS backends.")
endif()

if(NOT _meshioplusplus_mesh_backend MATCHES "^(MESHIO|NATIVE|KRATOS)$")
  message(FATAL_ERROR
    "meshio++: unknown MESHIOPLUSPLUS_MESH_BACKEND '${MESHIOPLUSPLUS_MESH_BACKEND}' "
    "(use MESHIO, NATIVE or KRATOS)")
endif()
message(STATUS "meshio++: mesh backend: ${_meshioplusplus_mesh_backend}")

# Deliberately NOT in _mio_core_defs: the backend macro is the one usage
# requirement that differs per target. meshioplusplus_core_obj gets the single
# backend this configure selected (as it always has); each installable
# meshioplusplus_core_<backend> variant gets its own.
option(MESHIOPLUSPLUS_NO_STD_SPAN
       "Omit NativeMesh::ConnSpan()/<span> (set when the consumer's Boost uBLAS \
collides with <span>, e.g. Kratos)" OFF)
if(MESHIOPLUSPLUS_NO_STD_SPAN)
  message(STATUS "meshio++: <span>/ConnSpan() omitted (MESHIOPLUSPLUS_NO_STD_SPAN)")
  list(APPEND _mio_core_defs MESHIOPLUSPLUS_NO_STD_SPAN)
endif()

# --------------------------------------------------------------------------
# Applies the accumulated usage requirements above to one or more targets.
# ARGV0 is the mesh backend (MESHIO|NATIVE|KRATOS) to bake in; the remaining
# arguments are target names. Everything is attached PUBLIC, which for the
# installable variants is what lands it in INTERFACE_* and therefore in
# meshioplusplusTargets.cmake -- a consumer then compiles its own translation
# units with the same backend/parallel macros and the same include dirs the
# library was built with, which is what makes the ABI agree by construction.
# --------------------------------------------------------------------------
function(meshioplusplus_configure_core_target backend)
  foreach(_t IN LISTS ARGN)
    target_compile_features(${_t} PUBLIC cxx_std_20)
    target_compile_definitions(${_t} PUBLIC
                               MESHIOPLUSPLUS_MESH_BACKEND_${backend} ${_mio_core_defs})
    if(_mio_core_incs)
      target_include_directories(${_t} PUBLIC ${_mio_core_incs})
    endif()
    if(_mio_core_priv_incs)
      target_include_directories(${_t} PRIVATE ${_mio_core_priv_incs})
    endif()
    if(_mio_core_libs)
      target_link_libraries(${_t} PUBLIC ${_mio_core_libs})
    endif()
    if(_mio_core_copts)
      target_compile_options(${_t} PUBLIC ${_mio_core_copts})
    endif()
    if(_mio_core_lopts)
      target_link_options(${_t} PUBLIC ${_mio_core_lopts})
    endif()
  endforeach()
endfunction()

# The in-tree core, unchanged in behaviour: same single backend, same PUBLIC
# usage requirements every existing consumer (_core, wasm, CLI, gtest) inherits.
meshioplusplus_configure_core_target(${_meshioplusplus_mesh_backend} meshioplusplus_core_obj)

# ---------------------------------------------------------------------------
# MESHIOPLUSPLUS_INSTALL_CPP: the installable, find_package()-able C++ API.
#
# OFF by default, so `pip install .`, cibuildwheel and every existing CI leg pay
# nothing for it. When ON, one library per entry of
# MESHIOPLUSPLUS_INSTALL_CPP_BACKENDS is built and installed side by side:
#
#     meshioplusplus::core_meshio   meshioplusplus::core_native   meshioplusplus::core_kratos
#
# plus `meshioplusplus::core`, an alias for whichever one matches
# MESHIOPLUSPLUS_MESH_BACKEND. Building all three from a single prefix is what
# lets one installed meshio++ serve consumers that disagree about the backend --
# a Kratos application wants core_kratos, a plain C++ tool wants core_native --
# instead of forcing each of them to build and install their own copy.
#
# The headers are backend-agnostic source (the backend is a compile-time macro),
# so they install exactly once no matter how many variants are built. The cost is
# compile time: each variant is a full, independent compile of the core.
#
# meshioplusplus_core_obj -- what the Python extension, the wasm module, the CLI,
# the gtest binary and the C API all link -- is deliberately NOT one of these. It
# keeps its single backend and its default visibility, so none of those consumers
# are perturbed by this option at all.
# ---------------------------------------------------------------------------
option(MESHIOPLUSPLUS_INSTALL_CPP
       "Install the full C++ API (meshioplusplus::core*) for find_package(... COMPONENTS CXX)"
       OFF)
set(MESHIOPLUSPLUS_INSTALL_CPP_BACKENDS "MESHIO;NATIVE;KRATOS" CACHE STRING
    "Mesh backends built+installed as meshioplusplus::core_<backend> when MESHIOPLUSPLUS_INSTALL_CPP=ON")

function(meshioplusplus_add_core_variant backend)
  string(TOLOWER "${backend}" _lc)
  set(_lib meshioplusplus_core_${_lc})

  # Sources compiled straight into the library rather than through an OBJECT
  # library. The core OBJECT library exists so the Python extension and the gtest
  # binary can share one compile; a variant has exactly one consumer, so there is
  # nothing to share -- and a STATIC library records even PRIVATE dependencies in
  # INTERFACE_LINK_LIBRARIES as $<LINK_ONLY:...> (a static archive does not link
  # them itself), which would drag the object library into the export set and
  # make install(EXPORT) fail.
  #
  # STATIC or SHARED per BUILD_SHARED_LIBS.
  add_library(${_lib} ${MESHIOPLUSPLUS_CORE_SOURCES})
  meshioplusplus_configure_core_target(${backend} ${_lib})
  set_target_properties(${_lib} PROPERTIES
    VERSION ${PROJECT_VERSION}
    # Tracks the ABI counter, NOT the release version and no longer a flat 0.
    # This is what makes the number enforced rather than advisory: the SONAME
    # becomes libmeshioplusplus_core_<backend>.so.<abi>, so the dynamic linker
    # itself refuses to load an incompatible library into a binary built against
    # an older one -- no cooperation from the consumer's build system required.
    # A flat 0 said "this filename promises nothing", which was true but useless.
    #
    # The C API's libmeshioplusplus deliberately does NOT follow: its contract is
    # SOVERSION 0 + append-only option structs + pin-the-major, and it is
    # unaffected by header layout changes because no C consumer compiles them.
    SOVERSION ${MESHIOPLUSPLUS_ABI_VERSION}
    POSITION_INDEPENDENT_CODE ON
    # Hidden by default: MESHIOPLUSPLUS_API (export.hpp) is what opts a symbol
    # back in, and a missing annotation then fails a shared consumer's link
    # loudly rather than silently exporting the entire core.
    CXX_VISIBILITY_PRESET hidden
    # Same reason, for the vendored gidpost C sources: without this, a shared
    # core_kratos exports every GiD_* symbol and collides with the consumer's
    # own vendored gidpost (Kratos vendors one at external_libraries/gidpost).
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    EXPORT_NAME core_${_lc})
  if(MESHIOPLUSPLUS_STATIC_RUNTIME AND MSVC)
    set_property(TARGET ${_lib} PROPERTY
                 MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()

  get_target_property(_type ${_lib} TYPE)
  if(_type STREQUAL "SHARED_LIBRARY")
    # PUBLIC so the library's own TUs see it (dllexport, together with the
    # PRIVATE _BUILDING below) and consumers see it too (dllimport).
    target_compile_definitions(${_lib} PUBLIC MESHIOPLUSPLUS_CORE_SHARED)
    target_compile_definitions(${_lib} PRIVATE MESHIOPLUSPLUS_CORE_BUILDING)
  endif()

  add_library(meshioplusplus::core_${_lc} ALIAS ${_lib})
  install(TARGETS ${_lib} EXPORT meshioplusplusTargets
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()

if(MESHIOPLUSPLUS_INSTALL_CPP)
  include(CMakePackageConfigHelpers)

  # All 112 headers, subdirectory structure preserved. detail/ is included
  # deliberately: the public headers include it transitively, so an install
  # without it does not compile.
  install(DIRECTORY src/cpp/include/meshioplusplus/
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/meshioplusplus
          FILES_MATCHING PATTERN "*.hpp")

  foreach(_backend IN LISTS MESHIOPLUSPLUS_INSTALL_CPP_BACKENDS)
    string(TOUPPER "${_backend}" _backend)
    if(NOT _backend MATCHES "^(MESHIO|NATIVE|KRATOS)$")
      message(FATAL_ERROR
        "meshio++: unknown backend '${_backend}' in MESHIOPLUSPLUS_INSTALL_CPP_BACKENDS "
        "(use MESHIO, NATIVE and/or KRATOS)")
    endif()
    meshioplusplus_add_core_variant(${_backend})
  endforeach()

  # meshioplusplus::core is "the default backend's variant". The installed
  # config re-creates it the same way (an INTERFACE target forwarding to the
  # variant), so `find_package(... COMPONENTS CXX)` + link meshioplusplus::core
  # means the same thing in-tree and out.
  string(TOLOWER "${_meshioplusplus_mesh_backend}" _mio_default_backend_lc)
  if(_meshioplusplus_mesh_backend IN_LIST MESHIOPLUSPLUS_INSTALL_CPP_BACKENDS)
    add_library(meshioplusplus::core ALIAS meshioplusplus_core_${_mio_default_backend_lc})
  else()
    message(WARNING
      "meshio++: MESHIOPLUSPLUS_MESH_BACKEND=${_meshioplusplus_mesh_backend} is not in "
      "MESHIOPLUSPLUS_INSTALL_CPP_BACKENDS=${MESHIOPLUSPLUS_INSTALL_CPP_BACKENDS}; "
      "meshioplusplus::core will not be defined -- link an explicit "
      "meshioplusplus::core_<backend> target instead.")
    set(_mio_default_backend_lc "")
  endif()

  # Shipped so a consumer with no FindKaHIP.cmake of its own can still resolve
  # the find_dependency(KaHIP) the generated config emits for a KaHIP build.
  install(FILES cmake/FindKaHIP.cmake
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/meshioplusplus)
  message(STATUS
    "meshio++: installing the C++ API (backends: ${MESHIOPLUSPLUS_INSTALL_CPP_BACKENDS})")
endif()

# ---------------------------------------------------------------------------
# Packaging metadata shared by both installable surfaces (the C API and the C++
# API). Derived from _mio_core_defs rather than from the raw *_FOUND variables so
# it reports what the build actually compiled in: `-DMESHIOPLUSPLUS_WITH_HDF5=OFF`
# on a machine that happens to have HDF5 now correctly reports "no HDF5".
# ---------------------------------------------------------------------------
foreach(_f ZLIB ZSTD LZ4 KAHIP HDF5 NETCDF EIGEN JSON CGNSLIB)
  if("MESHIOPLUSPLUS_HAS_${_f}" IN_LIST _mio_core_defs)
    set(_mio_has_${_f} TRUE)
  else()
    set(_mio_has_${_f} FALSE)
  endif()
endforeach()

# Normalized to literal TRUE/FALSE because these are substituted into if()
# conditions in the generated config: an unset variable would expand to nothing
# and make `if(X AND )` a hard CMake error at the consumer's find_package().
foreach(_v _mio_hdf5_needs_mpi _mio_parallel_needs_tbb _mio_hdf5_imported)
  if(${_v})
    set(${_v} TRUE)
  else()
    set(${_v} FALSE)
  endif()
endforeach()
# Which discovery mode each dep was found through -- the generated config must
# only find_dependency() a package that genuinely has a config/module to find
# (the manual find_library() fallbacks below have none).
foreach(_pair "zstd_FOUND:_mio_zstd_config" "lz4_FOUND:_mio_lz4_config"
              "netCDF_FOUND:_mio_netcdf_config" "HDF5_IS_PARALLEL:_mio_hdf5_parallel")
  string(REPLACE ":" ";" _pair "${_pair}")
  list(GET _pair 0 _src)
  list(GET _pair 1 _dst)
  if(${_src})
    set(${_dst} TRUE)
  else()
    set(${_dst} FALSE)
  endif()
endforeach()

# Libs.private for both .pc files: whatever optional deps this configure linked
# (a static-link aid -- `pkg-config --static --libs`).
set(MIO_PC_LIBS_PRIVATE "")
if(_mio_has_HDF5)
  string(APPEND MIO_PC_LIBS_PRIVATE " -lhdf5")
endif()
if(_mio_has_NETCDF)
  string(APPEND MIO_PC_LIBS_PRIVATE " -lnetcdf")
endif()
if(_mio_has_ZLIB)
  string(APPEND MIO_PC_LIBS_PRIVATE " -lz")
endif()
if(_mio_has_ZSTD)
  string(APPEND MIO_PC_LIBS_PRIVATE " -lzstd")
endif()
if(_mio_has_LZ4)
  string(APPEND MIO_PC_LIBS_PRIVATE " -llz4")
endif()
if(_mio_has_KAHIP)
  string(APPEND MIO_PC_LIBS_PRIVATE " -lkahip")
endif()

if(MESHIOPLUSPLUS_BUILD_PYTHON)
  # The pybind11 extension: bindings + the shared core object library.
  file(GLOB MESHIOPLUSPLUS_BINDING_SOURCES bindings/python/*.cpp)
  pybind11_add_module(_core ${MESHIOPLUSPLUS_BINDING_SOURCES})
  target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bindings/python)
  target_link_libraries(_core PRIVATE meshioplusplus_core_obj)

  # LDFLAGS env-var seeding of CMAKE_MODULE_LINKER_FLAGS was tried first and
  # silently didn't apply to this MODULE target, hence targeting it directly.
  if(MESHIOPLUSPLUS_STATIC_RUNTIME AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(_core PRIVATE -static-libgcc -static-libstdc++)
  endif()

  # Land the extension next to the pure-Python package inside the wheel. The
  # destination is relative to the wheel's platlib, so it only makes sense when
  # scikit-build-core is driving the install (it defines SKBUILD); on a plain
  # `cmake --install` -- normal since MESHIOPLUSPLUS_INSTALL_CPP made installing
  # a Python-enabled tree a reasonable thing to do -- the unguarded rule dumped
  # a stray <prefix>/meshioplusplus/_core*.so at the prefix root.
  if(DEFINED SKBUILD)
    install(TARGETS _core DESTINATION meshioplusplus)
  endif()
endif()

# --------------------------------------------------------------------------
# WebAssembly build (Emscripten + embind), producing the @meshioplusplus/wasm
# npm package's native artifact. EMSCRIPTEN is set automatically by CMake when
# configured through emcmake; MESHIOPLUSPLUS_BUILD_WASM is the explicit,
# documented opt-in on top of that (see build/configure-wasm.sh). This is
# entirely independent of MESHIOPLUSPLUS_BUILD_PYTHON -- a wasm configure runs
# with -DMESHIOPLUSPLUS_BUILD_PYTHON=OFF since no Python/pybind11 exists for
# the wasm32 target, and links the same meshioplusplus_core_obj the Python
# extension and GoogleTest suite already share.
# --------------------------------------------------------------------------
option(MESHIOPLUSPLUS_BUILD_WASM "Build the Emscripten/embind JS bindings" ${EMSCRIPTEN})
if(MESHIOPLUSPLUS_BUILD_WASM)
  if(NOT EMSCRIPTEN)
    message(FATAL_ERROR
      "meshio++: MESHIOPLUSPLUS_BUILD_WASM requires the Emscripten toolchain "
      "(configure with emcmake, e.g. via build/configure-wasm.sh).")
  endif()
  file(GLOB MESHIOPLUSPLUS_JS_BINDING_SOURCES bindings/wasm/*.cpp)
  add_executable(meshioplusplus_wasm ${MESHIOPLUSPLUS_JS_BINDING_SOURCES})
  target_include_directories(meshioplusplus_wasm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bindings/wasm)
  target_link_libraries(meshioplusplus_wasm PRIVATE meshioplusplus_core_obj)
  # -sUSE_ZLIB=1 is not repeated here: meshioplusplus_core_obj already applies
  # it PUBLIC (both compile and link) when MESHIOPLUSPLUS_WITH_ZLIB is on, and
  # this target inherits it transitively via target_link_libraries above.
  # STACK_SIZE: Emscripten's default wasm stack is 64 KiB, which HDF5 and
  # netCDF-4 overrun -- their call chains are deep and their frames large (name
  # buffers, B-tree nodes, dataspace scratch). The overrun is SILENT: the stack
  # grows downward into the static data segment, so the first symptom is
  # unrelated global state going bad. Concretely, writing one Exodus file
  # clobbered libc++'s locale facets, and every subsequent `istream >> number`
  # in the module -- i.e. every ASCII reader, gmsh/obj/off/vtk included --
  # trapped with "null function or function signature mismatch". 4 MiB is
  # comfortably above what the deepest observed path needs (the minimum that
  # survived a netCDF-4 write was 128 KiB) and is only an address-space
  # reservation inside the growable heap.
  #
  # STACK_OVERFLOW_CHECK turns a future recurrence back into a loud abort
  # rather than silent corruption of whatever happens to sit below the stack.
  # It costs a canary check on stack allocation, which is the right trade for a
  # library whose whole job is not corrupting the user's data.
  target_link_options(
    meshioplusplus_wasm PRIVATE
    "--bind"
    "-sMODULARIZE=1"
    "-sEXPORT_ES6=1"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sSTACK_SIZE=4MB"
    "-sSTACK_OVERFLOW_CHECK=1"
    "-sFORCE_FILESYSTEM=1"
    "-sEXPORTED_RUNTIME_METHODS=['FS']")
  # Threaded variant: a distinct artifact (meshioplusplus_wasm_mt) built in its
  # own tree with the OpenMP backend (which added -pthread PUBLIC on the core
  # object library above). A pool of one worker per logical core is pre-spawned
  # so an `omp parallel` region on the module's main thread has threads ready
  # (pthread_create cannot synchronously spin up a worker on the browser main
  # thread); POOL_SIZE_STRICT=0 still lets the pool grow on demand for deeper
  # nesting. The pool size is a runtime JS expression: `navigator` exists in
  # browsers and Web Workers but only in Node >= 21, and this package supports
  # Node >= 18 -- so guard with globalThis.navigator (undefined -> falsy ->
  # fall back to 8; Node then grows the pool on demand anyway because STRICT=0).
  # Written without spaces/quotes so it survives CMake -> Ninja -> em++ as one
  # argument. Consumed only when the page is cross-origin isolated; the JS
  # loader auto-falls-back to the sequential build otherwise.
  if(MESHIOPLUSPLUS_WASM_THREADS)
    target_link_options(
      meshioplusplus_wasm PRIVATE
      "-pthread"
      "-sPTHREAD_POOL_SIZE=globalThis.navigator?navigator.hardwareConcurrency:8"
      "-sPTHREAD_POOL_SIZE_STRICT=0")
    set_target_properties(meshioplusplus_wasm PROPERTIES OUTPUT_NAME "meshioplusplus_wasm_mt"
                                                          SUFFIX ".mjs")
  else()
    set_target_properties(meshioplusplus_wasm PROPERTIES OUTPUT_NAME "meshioplusplus_wasm"
                                                          SUFFIX ".mjs")
  endif()
endif()

# --------------------------------------------------------------------------
# C API (bindings/c/): the installable `libmeshioplusplus` shared library +
# the pure-C header, the third flat binding over the same core (alongside
# WASM). Written against the uniform mesh API only, so it builds under every
# MESHIOPLUSPLUS_MESH_BACKEND. The object-library split lets the GoogleTest
# binary link the C API's objects directly (no RPATH/shared-lib coupling in
# the per-backend test legs); the shared lib is what gets installed/exported.
# --------------------------------------------------------------------------
option(MESHIOPLUSPLUS_BUILD_C_API "Build the installable libmeshioplusplus C API" OFF)
option(MESHIOPLUSPLUS_BUILD_FORTRAN "Build the Fortran module (implies MESHIOPLUSPLUS_BUILD_C_API)" OFF)
if(MESHIOPLUSPLUS_BUILD_FORTRAN AND NOT MESHIOPLUSPLUS_BUILD_C_API)
  message(STATUS "meshio++: MESHIOPLUSPLUS_BUILD_FORTRAN=ON force-enables MESHIOPLUSPLUS_BUILD_C_API")
  set(MESHIOPLUSPLUS_BUILD_C_API ON)
endif()

if(MESHIOPLUSPLUS_BUILD_C_API)
  include(GNUInstallDirs)  # before any use of CMAKE_INSTALL_*DIR below
  include(CMakePackageConfigHelpers)

  add_library(meshioplusplus_c_obj OBJECT bindings/c/c_api.cpp)
  set_target_properties(meshioplusplus_c_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(meshioplusplus_c_obj
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bindings/c/include>)
  # MIO_SHARED/MIO_BUILDING drive the MIO_API dllexport/dllimport macro on
  # Windows; consumers of the installed library get MIO_SHARED from the
  # INTERFACE definition on the shared lib below.
  target_compile_definitions(meshioplusplus_c_obj
    PRIVATE MIO_BUILDING MIO_SHARED MIO_VERSION_STRING="${PROJECT_VERSION}")
  target_link_libraries(meshioplusplus_c_obj PUBLIC meshioplusplus_core_obj)

  # PRIVATE link: the object files are embedded, but the C++ core's usage
  # requirements (in-tree include dirs, backend macros, HDF5/... link deps)
  # stay out of the exported interface -- the installed surface is the C
  # header alone. Both object libraries must be linked DIRECTLY: CMake embeds
  # only directly-linked OBJECT libraries' objects (a transitive one, like
  # core_obj via c_obj, would contribute usage requirements but no objects,
  # leaving the .so with undefined registry/format symbols).
  add_library(meshioplusplus SHARED)
  target_link_libraries(meshioplusplus PRIVATE meshioplusplus_c_obj meshioplusplus_core_obj)
  target_include_directories(meshioplusplus
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bindings/c/include>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_compile_definitions(meshioplusplus INTERFACE MIO_SHARED)
  set_target_properties(meshioplusplus PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0)  # C ABI declared unstable pre-1.0 of the C API

  install(TARGETS meshioplusplus EXPORT meshioplusplusTargets
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES bindings/c/include/meshioplusplus/meshioplusplus.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/meshioplusplus)
  # NOTE: the export set, the package config and the version file are written
  # once, after this block, because MESHIOPLUSPLUS_INSTALL_CPP feeds the same
  # meshioplusplusTargets export -- either option alone must still produce a
  # complete, usable find_package().

  # pkg-config, the lingua franca of HPC build systems. MIO_PC_LIBS_PRIVATE is
  # computed once above and shared with the C++ .pc file.
  configure_file(cmake/meshioplusplus.pc.in
                 ${CMAKE_CURRENT_BINARY_DIR}/meshioplusplus.pc @ONLY)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/meshioplusplus.pc
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()

# ---------------------------------------------------------------------------
# The export set + package config, written once for whichever of the two
# installable surfaces was enabled. Both feed the same `meshioplusplusTargets`
# export, so one find_package(meshioplusplus) serves both APIs and
# `COMPONENTS C` / `COMPONENTS CXX` select between them.
# ---------------------------------------------------------------------------
if(MESHIOPLUSPLUS_BUILD_C_API OR MESHIOPLUSPLUS_INSTALL_CPP)
  include(CMakePackageConfigHelpers)

  install(EXPORT meshioplusplusTargets NAMESPACE meshioplusplus::
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/meshioplusplus)
  configure_package_config_file(cmake/meshioplusplusConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/meshioplusplusConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/meshioplusplus)
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/meshioplusplusConfigVersion.cmake
    COMPATIBILITY SameMajorVersion)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/meshioplusplusConfig.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/meshioplusplusConfigVersion.cmake
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/meshioplusplus)
endif()

# A second pkg-config file for the C++ API, deliberately NOT an extension of the
# C one: `pkg-config --cflags meshioplusplus` must keep working for a plain-C
# consumer compiling with -std=c99 (doc/examples/c_api_example.c does exactly
# that in CI), so it can never grow -std=c++20 or the backend macro.
if(MESHIOPLUSPLUS_INSTALL_CPP AND _mio_default_backend_lc)
  set(MIO_CXX_PC_BACKEND "${_meshioplusplus_mesh_backend}")
  set(MIO_CXX_PC_LIB "meshioplusplus_core_${_mio_default_backend_lc}")
  set(MIO_CXX_PC_CFLAGS_EXTRA "")
  if(MESHIOPLUSPLUS_NO_STD_SPAN)
    string(APPEND MIO_CXX_PC_CFLAGS_EXTRA " -DMESHIOPLUSPLUS_NO_STD_SPAN")
  endif()
  configure_file(cmake/meshioplusplus-cxx.pc.in
                 ${CMAKE_CURRENT_BINARY_DIR}/meshioplusplus-cxx.pc @ONLY)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/meshioplusplus-cxx.pc
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()

# --------------------------------------------------------------------------
# Fortran module (bindings/fortran/): a modern OO Fortran 2008 interface
# (`type(mio_mesh)` with type-bound procedures) layered on the C API via
# ISO_C_BINDING. The .f90 source is installed alongside the compiled .mod
# because .mod files are compiler-(major-version-)specific -- consumers on a
# different compiler recompile the module from source (the HDF5 approach).
# --------------------------------------------------------------------------
if(MESHIOPLUSPLUS_BUILD_FORTRAN)
  enable_language(Fortran)
  add_library(meshioplusplus_fortran SHARED bindings/fortran/meshioplusplus.f90)
  target_link_libraries(meshioplusplus_fortran PUBLIC meshioplusplus)
  set_target_properties(meshioplusplus_fortran PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran_modules)
  target_include_directories(meshioplusplus_fortran INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/fortran_modules>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/meshioplusplus/fortran>)

  install(TARGETS meshioplusplus_fortran EXPORT meshioplusplusTargets
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fortran_modules/meshioplusplus.mod
                bindings/fortran/meshioplusplus.f90
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/meshioplusplus/fortran)
endif()

# --------------------------------------------------------------------------
# Optional standalone C++ tests (GoogleTest via CTest). OFF by default so that
# `pip install .` / cibuildwheel never build them. Enable with
# `-DMESHIO_BUILD_TESTS=ON` for a direct CMake configure.
# --------------------------------------------------------------------------
option(MESHIOPLUSPLUS_BUILD_TESTS "Build the C++ GoogleTest suite" OFF)
if(MESHIOPLUSPLUS_BUILD_TESTS)
  include(CTest)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  # Keep `cmake --install` of a tests-enabled tree from installing gtest
  # next to the C API (relevant since MESHIOPLUSPLUS_BUILD_C_API).
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  file(GLOB MESHIOPLUSPLUS_TEST_SOURCES tests/cpp/*.cpp)
  if(NOT MESHIOPLUSPLUS_BUILD_C_API)
    list(REMOVE_ITEM MESHIOPLUSPLUS_TEST_SOURCES
         ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/test_c_api.cpp)
  endif()
  add_executable(meshioplusplus_tests ${MESHIOPLUSPLUS_TEST_SOURCES})
  target_include_directories(meshioplusplus_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp)
  # The Polyscope payload builder is CLI-side but deliberately Polyscope-free,
  # so its invariants are tested here in the default build -- no OpenGL needed.
  target_sources(meshioplusplus_tests PRIVATE src/cpp/cli/view_payload.cpp)
  target_link_libraries(meshioplusplus_tests PRIVATE meshioplusplus_core_obj GTest::gtest_main)
  if(MESHIOPLUSPLUS_BUILD_C_API)
    # Link the C API's objects directly (not the shared lib): the gtest binary
    # then needs no install RPATH and the per-backend CI legs stay one-target.
    target_link_libraries(meshioplusplus_tests PRIVATE meshioplusplus_c_obj)
  endif()

  include(GoogleTest)
  gtest_discover_tests(meshioplusplus_tests)

  if(MESHIOPLUSPLUS_BUILD_FORTRAN)
    # Plain Fortran program, nonzero exit on any failed check; argv[1] is a
    # path prefix for the files it writes.
    add_executable(meshioplusplus_fortran_test tests/fortran/test_fortran_api.f90)
    target_link_libraries(meshioplusplus_fortran_test PRIVATE meshioplusplus_fortran)
    add_test(NAME fortran_api
             COMMAND meshioplusplus_fortran_test ${CMAKE_CURRENT_BINARY_DIR}/fortran_test_out)
  endif()
endif()

# --------------------------------------------------------------------------
# C++ backend benchmark (src/cpp/benchmark/bench_backends.cpp): one binary per
# mesh backend (the backend is a compile-time choice); benchmark/
# bench_backends.sh builds all three variants and collates the CSV output.
# No external benchmark framework -- std::chrono, warmup + median-of-N.
# --------------------------------------------------------------------------
option(MESHIOPLUSPLUS_BUILD_BENCHMARKS "Build the C++ mesh-backend benchmark binary" OFF)
if(MESHIOPLUSPLUS_BUILD_BENCHMARKS)
  add_executable(meshioplusplus_bench src/cpp/benchmark/bench_backends.cpp)
  target_link_libraries(meshioplusplus_bench PRIVATE meshioplusplus_core_obj)
endif()

# --------------------------------------------------------------------------
# Native command-line binary (src/cpp/cli/main.cpp): a Python-free CLI over
# meshioplusplus_core_obj, mirroring the Python console script's verbs
# (convert/info/ascii/binary/compress/decompress/quality/extract-surface/
# reorder/diff). Installed executable name is `meshioplusplus`. Opt-in like the
# other executables; works under any mesh/parallel backend and optional-dep set.
# --------------------------------------------------------------------------
option(MESHIOPLUSPLUS_BUILD_CLI "Build the native command-line binary" OFF)
if(MESHIOPLUSPLUS_BUILD_CLI)
  add_executable(meshioplusplus_cli src/cpp/cli/main.cpp)
  target_sources(meshioplusplus_cli PRIVATE src/cpp/cli/view_payload.cpp)
  target_link_libraries(meshioplusplus_cli PRIVATE meshioplusplus_core_obj)
  target_compile_definitions(meshioplusplus_cli
    PRIVATE MESHIOPLUSPLUS_CLI_VERSION="${PROJECT_VERSION}")

  # The viewer half. PRIVATE on purpose -- nothing else in the project may
  # acquire an OpenGL dependency through it. EXCLUDE_FROM_ALL keeps Polyscope's
  # own examples and tests out of the build.
  if(MESHIOPLUSPLUS_WITH_POLYSCOPE AND NOT EMSCRIPTEN)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/third_party/polyscope/CMakeLists.txt")
      message(STATUS "meshio++: Polyscope viewer enabled (CLI only)")
      add_subdirectory(src/cpp/third_party/polyscope EXCLUDE_FROM_ALL)
      target_sources(meshioplusplus_cli PRIVATE src/cpp/cli/polyscope_view.cpp)
      target_link_libraries(meshioplusplus_cli PRIVATE polyscope)
      target_compile_definitions(meshioplusplus_cli PRIVATE MESHIOPLUSPLUS_HAS_POLYSCOPE)
    else()
      message(STATUS
        "meshio++: Polyscope requested but src/cpp/third_party/polyscope is empty - "
        "run `git submodule update --init --recursive`; view/screenshot will raise")
    endif()
  endif()
  set_target_properties(meshioplusplus_cli PROPERTIES OUTPUT_NAME meshioplusplus)
  if(MESHIOPLUSPLUS_STATIC_RUNTIME)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      target_link_options(meshioplusplus_cli PRIVATE -static-libgcc -static-libstdc++)
    elseif(MSVC)
      # CMP0091 (NEW via the top-of-file cmake_minimum_required range) makes this
      # property authoritative regardless of when the target was created: static
      # CRT (/MT, /MTd for Debug) so the binary needs no vcredist at runtime.
      set_property(TARGET meshioplusplus_cli PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
  endif()
  include(GNUInstallDirs)
  install(TARGETS meshioplusplus_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
