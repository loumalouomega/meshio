# Conan 2.x recipe for the meshio++ C API (libmeshioplusplus).
#
# This packages the installable C API only (-DMESHIOPLUSPLUS_BUILD_C_API=ON,
# -DMESHIOPLUSPLUS_BUILD_PYTHON=OFF): the same relocatable CMake config-package
# (`meshioplusplus::meshioplusplus`) + pkg-config file the standalone
# `cmake --install` produces. The Python wheel is published to PyPI separately
# (see pyproject.toml / .github/workflows/wheels.yml) and is unrelated to this.
#
# `version` must track pyproject.toml's `version` and CMakeLists.txt's
# `project(... VERSION ...)` -- bump all three together on a release.
#
# Notes on optional dependencies:
#   * pugixml is vendored in-tree (src/cpp/third_party/pugixml) -- no requirement.
#   * gidpost (the GiD postprocess writer's backend) is likewise vendored
#     in-tree (src/cpp/third_party/gidpost) and IS exported below -- unlike
#     Eigen/nlohmann-json, it is a hardcopy, not a submodule, so it is present
#     in a source export and with_gidpost defaults True. It is hard-gated on
#     zlib (gidpostInt.h includes <zlib.h> unconditionally): with with_zlib
#     off, the CMake option auto-disables and the `gid` writer raises by name
#     rather than failing the build.
#   * Eigen is a git submodule (src/cpp/third_party/eigen), absent from a source
#     export, so `with_eigen` defaults False and the MED transpose uses the
#     hand-written fallback loop. Enabling it would need a small CMake change to
#     accept an external Eigen3::Eigen (tracked as a follow-up).
#   * HDF5/netCDF/zlib are PRIVATE to the shared library (the installed surface
#     is the C header alone), but they must be present at build and run time.

import os

from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy


class MeshioplusplusConan(ConanFile):
    name = "meshioplusplus"
    version = "10.38.0"
    license = "MIT"
    description = "C++ core for the meshio++ mesh I/O library (installable C API)"
    homepage = "https://github.com/loumalouomega/meshioplusplus"
    url = "https://github.com/loumalouomega/meshioplusplus"
    topics = ("mesh", "fem", "file-formats", "scientific-computing", "hpc")

    settings = "os", "compiler", "build_type", "arch"
    # The C API shared library is hardcoded SHARED in CMake today, so there is
    # no `shared` option (v1). A static build is a documented follow-up.
    options = {
        "fPIC": [True, False],
        "with_hdf5": [True, False],
        "with_netcdf": [True, False],
        "with_zlib": [True, False],
        "with_zstd": [True, False],
        "with_lz4": [True, False],
        "with_kahip": [True, False],
        "with_cgnslib": [True, False],
        "with_gidpost": [True, False],
        "with_eigen": [True, False],
        "with_json": [True, False],
        "fortran": [True, False],
        # The full C++ API (meshioplusplus::core*) alongside the C API. Off by
        # default so the package stays the small C-API-only artifact it was.
        "with_cxx_api": [True, False],
        # Which mesh backends to build+install as meshioplusplus::core_<backend>
        # when with_cxx_api is on. Comma-separated; translated to a CMake list.
        "cxx_api_backends": ["ANY"],
    }
    default_options = {
        "fPIC": True,
        "with_hdf5": True,
        "with_netcdf": True,
        "with_zlib": True,
        # Off by default, unlike zlib/hdf5/netcdf: zlib stays the write default,
        # so a package without these reads and writes exactly what it always did.
        "with_zstd": False,
        "with_lz4": False,
        # KaHIP is not on ConanCenter: no requirement is added -- the consumer
        # supplies an install and points KAHIP_ROOT at it (find_package prefix,
        # same policy as the CMake build). Off by default.
        "with_kahip": False,
        "with_cgnslib": False,
        # Unlike with_eigen/with_json (submodules, off by default), gidpost is
        # a vendored hardcopy that IS in exports_sources, so this defaults True
        # like with_hdf5/with_netcdf/with_zlib -- consistent with the CMake
        # option's own ON default.
        "with_gidpost": True,
        "with_eigen": False,  # submodule not in a source export -> fallback transpose
        # Same reason as with_eigen: the nlohmann/json submodule is not in a
        # source export, so the pipeline JSON entry points raise by name.
        "with_json": False,
        "fortran": False,
        "with_cxx_api": False,
        # All three, so one package serves consumers that disagree about the
        # backend (a Kratos app wants kratos, a plain C++ tool wants native).
        # Trim it to cut build time: -o cxx_api_backends=KRATOS.
        "cxx_api_backends": "MESHIO,NATIVE,KRATOS",
    }

    # Everything the C API build needs. pugixml is inside src/cpp/third_party; the
    # Eigen and nlohmann/json submodules are deliberately not exported
    # (with_eigen/with_json default off).
    exports_sources = (
        "CMakeLists.txt",
        "LICENSE",
        "src/cpp/include/*",
        "src/cpp/src/*",
        "src/cpp/third_party/pugixml/*",
        "src/cpp/third_party/gidpost/*",
        "bindings/c/*",
        "bindings/fortran/*",
        "cmake/*",
    )

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        # Header-only / C consumer surface: no C++ settings leak from deps.
        self.settings.rm_safe("compiler.cppstd")

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        if self.options.with_zlib:
            self.requires("zlib/[>=1.2.11 <2]")
        if self.options.with_zstd:
            self.requires("zstd/[>=1.5 <2]")
        if self.options.with_lz4:
            self.requires("lz4/[>=1.9 <2]")
        if self.options.with_netcdf:
            self.requires("netcdf/[>=4.8 <5]")
        if self.options.with_hdf5:
            # netcdf transitively pins an exact, older hdf5 (netcdf/4.8.1 ->
            # hdf5/1.14.3) than the top of our floating range, which Conan
            # reports as a hard version conflict. When both are on, pin our
            # direct hdf5 to netcdf's version so the graph resolves and we still
            # link hdf5 ourselves; otherwise float within the compatible range.
            # (The `packages` CI now runs on every recipe change, so a future
            # netcdf bumping its hdf5 pin is caught here rather than downstream.)
            if self.options.with_netcdf:
                self.requires("hdf5/1.14.3")
            else:
                self.requires("hdf5/[>=1.14 <2]")

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.cache_variables["MESHIOPLUSPLUS_BUILD_C_API"] = True
        tc.cache_variables["MESHIOPLUSPLUS_BUILD_PYTHON"] = False
        tc.cache_variables["MESHIOPLUSPLUS_BUILD_FORTRAN"] = bool(self.options.fortran)
        tc.cache_variables["MESHIOPLUSPLUS_WITH_HDF5"] = bool(self.options.with_hdf5)
        tc.cache_variables["MESHIOPLUSPLUS_WITH_NETCDF"] = bool(
            self.options.with_netcdf
        )
        tc.cache_variables["MESHIOPLUSPLUS_WITH_ZLIB"] = bool(self.options.with_zlib)
        tc.cache_variables["MESHIOPLUSPLUS_WITH_ZSTD"] = bool(self.options.with_zstd)
        tc.cache_variables["MESHIOPLUSPLUS_WITH_LZ4"] = bool(self.options.with_lz4)
        tc.cache_variables["MESHIOPLUSPLUS_WITH_KAHIP"] = bool(self.options.with_kahip)
        # cgnslib is not on ConanCenter, so like KaHIP it is bring-your-own:
        # this only flips the CMake flag and the consumer supplies CGNS_ROOT.
        tc.cache_variables["MESHIOPLUSPLUS_WITH_CGNSLIB"] = bool(
            self.options.with_cgnslib
        )
        tc.cache_variables["MESHIOPLUSPLUS_WITH_GIDPOST"] = bool(
            self.options.with_gidpost
        )
        tc.cache_variables["MESHIOPLUSPLUS_WITH_EIGEN"] = bool(self.options.with_eigen)
        tc.cache_variables["MESHIOPLUSPLUS_WITH_JSON"] = bool(self.options.with_json)
        tc.cache_variables["MESHIOPLUSPLUS_INSTALL_CPP"] = bool(
            self.options.with_cxx_api
        )
        if self.options.with_cxx_api:
            # CMake wants a ;-list; the option is comma-separated so it survives
            # a Conan command line without quoting games.
            backends = str(self.options.cxx_api_backends).replace(",", ";")
            tc.cache_variables["MESHIOPLUSPLUS_INSTALL_CPP_BACKENDS"] = backends
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        copy(
            self,
            "LICENSE",
            src=self.source_folder,
            dst=os.path.join(self.package_folder, "licenses"),
        )

    def package_info(self):
        # Match the installed find_package config so downstream
        # find_package(meshioplusplus) + meshioplusplus::meshioplusplus resolve.
        self.cpp_info.set_property("cmake_file_name", "meshioplusplus")
        self.cpp_info.set_property("pkg_config_name", "meshioplusplus")

        if not self.options.with_cxx_api:
            # C-API-only package: the flat, pre-existing shape. Components would
            # be a gratuitous break for every current consumer.
            self.cpp_info.libs = ["meshioplusplus"]
            self.cpp_info.set_property(
                "cmake_target_name", "meshioplusplus::meshioplusplus"
            )
            if self.options.fortran:
                self.cpp_info.libs.insert(0, "meshioplusplus_fortran")
            return

        # With the C++ API there is more than one library in the package, so
        # Conan needs components (cpp_info.libs and components are mutually
        # exclusive). Names mirror the CMake targets exactly.
        c = self.cpp_info.components["c_api"]
        c.libs = ["meshioplusplus"]
        c.set_property("cmake_target_name", "meshioplusplus::meshioplusplus")
        c.set_property("pkg_config_name", "meshioplusplus")

        if self.options.fortran:
            f = self.cpp_info.components["fortran"]
            f.libs = ["meshioplusplus_fortran"]
            f.requires = ["c_api"]
            f.set_property(
                "cmake_target_name", "meshioplusplus::meshioplusplus_fortran"
            )

        # Unlike the C API -- whose heavy deps are PRIVATE to its shared object --
        # the C++ libraries propagate theirs, because a consumer compiles the real
        # headers and (statically) links the real dependency graph.
        cxx_requires = []
        if self.options.with_zlib:
            cxx_requires.append("zlib::zlib")
        if self.options.with_zstd:
            cxx_requires.append("zstd::zstd")
        if self.options.with_lz4:
            cxx_requires.append("lz4::lz4")
        if self.options.with_hdf5:
            cxx_requires.append("hdf5::hdf5")
        if self.options.with_netcdf:
            cxx_requires.append("netcdf::netcdf")

        backends = [
            b.strip().upper()
            for b in str(self.options.cxx_api_backends).split(",")
            if b.strip()
        ]
        for backend in backends:
            lc = backend.lower()
            comp = self.cpp_info.components[f"core_{lc}"]
            comp.libs = [f"meshioplusplus_core_{lc}"]
            comp.requires = list(cxx_requires)
            comp.set_property("cmake_target_name", f"meshioplusplus::core_{lc}")
            # The backend macro is part of the ABI: a consumer compiled without
            # it disagrees with this library about what meshioplusplus::Mesh is.
            comp.defines = [f"MESHIOPLUSPLUS_MESH_BACKEND_{backend}"]

        # meshioplusplus::core == the default backend, carrying no library of its
        # own (it forwards), which is how the CMake install expresses it too.
        default_backend = "MESHIO"
        if default_backend in backends:
            core = self.cpp_info.components["core"]
            core.requires = [f"core_{default_backend.lower()}"]
            core.set_property("cmake_target_name", "meshioplusplus::core")
            core.set_property("pkg_config_name", "meshioplusplus-cxx")
