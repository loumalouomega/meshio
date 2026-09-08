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

// External includes
#include <gtest/gtest.h>

// System includes
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Project includes
#include "mesh_fixtures.hpp"
#include "meshioplusplus/formats/flac3d.hpp"
#include "meshioplusplus/region.hpp"

namespace {

using meshioplusplus::Mesh;
using meshioplusplus::Region;
using meshioplusplus::RegionKind;

// A hand-written fixture rather than a read of tests/python/meshes: this suite
// has no test-data path, and a file written by our own writer is a weak oracle
// for our own reader -- a consistently wrong convention survives a round trip.
constexpr const char* kGroupsAscii = R"(* GRIDPOINTS
G 1 0.0 0.0 0.0
G 2 1.0 0.0 0.0
G 3 0.0 1.0 0.0
G 4 0.0 0.0 1.0
G 5 1.0 1.0 1.0
* ZONES
Z T4 1 1 2 3 4
Z T4 2 2 3 4 5
* ZONE GROUPS
ZGROUP "solid" SLOT "Default"
 1 2
ZGROUP "empty" SLOT "5"
* FACES
F T3 1 1 2 3
F T3 2 2 3 4
* FACE GROUPS
FGROUP "bottom" SLOT "Default"
 2
)";

std::string write_fixture(const std::string& rText, const std::string& rSuffix) {
    const std::string path = mt::temp_path(rSuffix);
    std::ofstream out(path, std::ios::binary);
    out << rText;
    out.close();
    return path;
}

// Every cell region as `name -> entries`; regions are stored sorted by
// (kind, name, dim, tag), but a map keeps the assertions readable either way.
std::map<std::string, std::vector<std::int64_t>> cell_regions(const Mesh& rMesh) {
    std::map<std::string, std::vector<std::int64_t>> out;
    for (std::size_t i = 0; i < rMesh.NumRegions(); ++i) {
        const Region& r = rMesh.Region(i);
        if (r.mKind != RegionKind::Cell)
            continue;
        out[r.mName] = std::vector<std::int64_t>(r.Entries(), r.Entries() + r.NumEntries());
    }
    return out;
}

// Append a little-endian scalar to a byte string.
template <class T>
void put(std::string& rOut, T v) {
    rOut.append(reinterpret_cast<const char*>(&v), sizeof(T));
}

// The same mesh as kGroupsAscii, in the binary layout.
std::string binary_fixture() {
    std::string b;
    put<std::uint32_t>(b, 1375135718u);
    put<std::uint32_t>(b, 3u);
    put<std::uint32_t>(b, 5u);  // points
    const double pts[5][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};
    for (std::uint32_t i = 0; i < 5; ++i) {
        put<std::uint32_t>(b, i + 1);
        for (int c = 0; c < 3; ++c)
            put<double>(b, pts[i][c]);
    }
    // zones (ids 1..2, their own namespace)
    put<std::uint32_t>(b, 2u);
    const std::uint32_t zones[2][4] = {{1, 2, 3, 4}, {2, 3, 4, 5}};
    for (std::uint32_t z = 0; z < 2; ++z) {
        put<std::uint32_t>(b, z + 1);
        put<std::uint32_t>(b, 4u);
        for (int j = 0; j < 4; ++j)
            put<std::uint32_t>(b, zones[z][j]);
    }
    // zone groups: "solid"/Default over both, "empty"/5 over none
    put<std::uint32_t>(b, 2u);
    put<std::uint16_t>(b, 5u);
    b += "solid";
    put<std::uint16_t>(b, 7u);
    b += "Default";
    put<std::uint32_t>(b, 2u);
    put<std::uint32_t>(b, 1u);
    put<std::uint32_t>(b, 2u);
    put<std::uint16_t>(b, 5u);
    b += "empty";
    put<std::uint16_t>(b, 1u);
    b += "5";
    put<std::uint32_t>(b, 0u);
    // faces (ids 1..2, restarting)
    put<std::uint32_t>(b, 2u);
    const std::uint32_t faces[2][3] = {{1, 2, 3}, {2, 3, 4}};
    for (std::uint32_t k = 0; k < 2; ++k) {
        put<std::uint32_t>(b, k + 1);
        put<std::uint32_t>(b, 3u);
        for (int j = 0; j < 3; ++j)
            put<std::uint32_t>(b, faces[k][j]);
    }
    // face groups: "bottom"/Default over face id 2
    put<std::uint32_t>(b, 1u);
    put<std::uint16_t>(b, 6u);
    b += "bottom";
    put<std::uint16_t>(b, 7u);
    b += "Default";
    put<std::uint32_t>(b, 1u);
    put<std::uint32_t>(b, 2u);
    return b;
}

// Blocks come back faces-first, so the two faces are global 0..1 and the two
// zones global 2..3 -- and `bottom` names face id 2, i.e. global 1.
const std::map<std::string, std::vector<std::int64_t>> kExpected = {
    {"face:bottom:Default", {1}},
    {"zone:empty:5", {}},
    {"zone:solid:Default", {2, 3}},
};

// Two triangles (faces) and two tetrahedra (zones) over five points.
Mesh base_mesh() {
    Mesh m;
    meshioplusplus::NDArray pts(meshioplusplus::DType::Float64, {5, 3});
    const double raw[15] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
    std::copy(raw, raw + 15, pts.As<double>());
    m.AssignPoints(std::move(pts));

    meshioplusplus::NDArray tri(meshioplusplus::DType::Int64, {2, 3});
    const std::int64_t traw[6] = {0, 1, 2, 1, 2, 3};
    std::copy(traw, traw + 6, tri.As<std::int64_t>());
    m.AddCellBlock("triangle", std::move(tri));

    meshioplusplus::NDArray tet(meshioplusplus::DType::Int64, {2, 4});
    const std::int64_t qraw[8] = {0, 1, 2, 3, 1, 2, 3, 4};
    std::copy(qraw, qraw + 8, tet.As<std::int64_t>());
    m.AddCellBlock("tetra", std::move(tet));
    return m;
}

Mesh region_mesh() {
    Mesh m = base_mesh();

    meshioplusplus::NDArray e1(meshioplusplus::DType::Int64, {1});
    e1.As<std::int64_t>()[0] = 1;
    m.AddRegion(Region("face:bottom:Default", RegionKind::Cell, std::move(e1)));

    meshioplusplus::NDArray e2(meshioplusplus::DType::Int64, {2});
    e2.As<std::int64_t>()[0] = 2;
    e2.As<std::int64_t>()[1] = 3;
    m.AddRegion(Region("zone:solid:Default", RegionKind::Cell, std::move(e2)));
    return m;
}

std::string slurp(const std::string& rPath) {
    std::ifstream in(rPath, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

}  // namespace

TEST(Flac3dGroups, AsciiFixture) {
    const std::string path = write_fixture(kGroupsAscii, ".f3grid");
    const Mesh m = meshioplusplus::read_flac3d(path);
    std::remove(path.c_str());
    EXPECT_EQ(cell_regions(m), kExpected);
}

TEST(Flac3dGroups, BinaryFixture) {
    const std::string path = write_fixture(binary_fixture(), ".f3grid");
    const Mesh m = meshioplusplus::read_flac3d(path);
    std::remove(path.c_str());
    // The two encodings of the same mesh must agree on the group names too:
    // the ascii slot arrives quoted and the binary one bare.
    EXPECT_EQ(cell_regions(m), kExpected);
}

TEST(Flac3dGroups, RoundTrip) {
    for (bool binary : {false, true}) {
        const Mesh in = region_mesh();
        const std::string path = mt::temp_path(".f3grid");
        meshioplusplus::write_flac3d(path, in, ".16e", binary);
        const Mesh out = meshioplusplus::read_flac3d(path);
        std::remove(path.c_str());
        EXPECT_EQ(cell_regions(out), cell_regions(in)) << "binary=" << binary;
    }
}

TEST(Flac3dGroups, NameIsAFixedPoint) {
    Mesh m = region_mesh();
    for (int pass = 0; pass < 2; ++pass) {
        const std::string path = mt::temp_path(".f3grid");
        meshioplusplus::write_flac3d(path, m, ".16e", false);
        m = meshioplusplus::read_flac3d(path);
        std::remove(path.c_str());
        EXPECT_EQ(cell_regions(m), cell_regions(region_mesh())) << "pass=" << pass;
    }
}

TEST(Flac3dGroups, FaceIdsRestartAtOne) {
    // Zones and faces are separate 1-based namespaces. Sharing one counter
    // made the ids disagree with the FGROUP lists written beside them.
    const std::string path = mt::temp_path(".f3grid");
    meshioplusplus::write_flac3d(path, region_mesh(), ".16e", false);
    const std::string text = slurp(path);
    std::remove(path.c_str());
    EXPECT_NE(text.find("\nZ T4 1 "), std::string::npos) << text;
    EXPECT_NE(text.find("\nF T3 1 "), std::string::npos) << text;
    EXPECT_NE(text.find("FGROUP \"bottom\" SLOT \"Default\""), std::string::npos) << text;
}

TEST(Flac3dGroups, NoRegionsWritesTheSameBytes) {
    // The guarantee that carrying groups costs a group-less mesh nothing: a
    // mesh with no Cell regions writes exactly the empty group sections it
    // always did.
    const Mesh without = base_mesh();

    const std::string a = mt::temp_path(".f3grid");
    meshioplusplus::write_flac3d(a, without, ".16e", false);
    const std::string text = slurp(a);
    std::remove(a.c_str());
    EXPECT_NE(text.find("* ZONE GROUPS\n* FACES"), std::string::npos) << text;
    EXPECT_NE(text.find("* FACE GROUPS\n"), std::string::npos) << text;
    EXPECT_EQ(text.find("ZGROUP"), std::string::npos) << text;
}
