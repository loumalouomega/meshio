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
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Project includes
#include "meshioplusplus/formats/flac3d.hpp"
#include "meshioplusplus/detail/cell_index.hpp"
#include "meshioplusplus/detail/value_io.hpp"
#include "meshioplusplus/detail/provenance.hpp"
#include "meshioplusplus/exceptions.hpp"
#include "meshioplusplus/log.hpp"
#include "meshioplusplus/parallel.hpp"
#include "meshioplusplus/region.hpp"

namespace meshioplusplus {

namespace {

// meshio type -> simplified FLAC3D base type (zone = 3D, face = 2D), or "".
std::string zone_key(const std::string& rT) {
    static const std::unordered_map<std::string, std::string> m = {{"tetra", "tetra"},
                                                                   {"tetra10", "tetra"},
                                                                   {"pyramid", "pyramid"},
                                                                   {"pyramid13", "pyramid"},
                                                                   {"wedge", "wedge"},
                                                                   {"wedge12", "wedge"},
                                                                   {"wedge15", "wedge"},
                                                                   {"wedge18", "wedge"},
                                                                   {"hexahedron", "hexahedron"},
                                                                   {"hexahedron20", "hexahedron"},
                                                                   {"hexahedron24", "hexahedron"},
                                                                   {"hexahedron27", "hexahedron"}};
    auto it = m.find(rT);
    return it == m.end() ? std::string() : it->second;
}
std::string face_key(const std::string& rT) {
    static const std::unordered_map<std::string, std::string> m = {
        {"triangle", "triangle"}, {"triangle6", "triangle"}, {"triangle7", "triangle"},
        {"quad", "quad"},         {"quad8", "quad"},         {"quad9", "quad"}};
    auto it = m.find(rT);
    return it == m.end() ? std::string() : it->second;
}

const std::unordered_map<int, std::string>& numnodes_type(int dim) {
    static const std::unordered_map<int, std::string> z = {
        {4, "tetra"}, {5, "pyramid"}, {6, "wedge"}, {8, "hexahedron"}};
    static const std::unordered_map<int, std::string> fc = {{3, "triangle"}, {4, "quad"}};
    return dim == 3 ? z : fc;
}

const char* flac3d_type(const std::string& rKey) {
    if (rKey == "triangle")
        return "T3";
    if (rKey == "quad")
        return "Q4";
    if (rKey == "tetra")
        return "T4";
    if (rKey == "pyramid")
        return "P5";
    if (rKey == "wedge")
        return "W6";
    return "B8";  // hexahedron
}

const std::vector<int>& f2m_order(const std::string& rKey) {
    static const std::unordered_map<std::string, std::vector<int>> m = {
        {"triangle", {0, 1, 2}},       {"quad", {0, 1, 2, 3}},
        {"tetra", {0, 1, 2, 3}},       {"pyramid", {0, 1, 4, 2, 3}},
        {"wedge", {0, 1, 3, 2, 4, 5}}, {"hexahedron", {0, 1, 4, 2, 3, 6, 7, 5}}};
    return m.at(rKey);
}
const std::vector<int>& m2f_order(const std::string& rKey) {
    static const std::unordered_map<std::string, std::vector<int>> m = {
        {"triangle", {0, 1, 2}},       {"quad", {0, 1, 2, 3}},
        {"tetra", {0, 1, 2, 3}},       {"pyramid", {0, 1, 3, 4, 2}},
        {"wedge", {0, 1, 3, 2, 4, 5}}, {"hexahedron", {0, 1, 3, 4, 2, 7, 5, 6}}};
    return m.at(rKey);
}
const std::vector<int>& m2f_order2(const std::string& rKey) {
    static const std::unordered_map<std::string, std::vector<int>> m = {
        {"tetra", {0, 2, 1, 3}},
        {"pyramid", {0, 3, 1, 4, 2}},
        {"wedge", {0, 2, 3, 1, 5, 4}},
        {"hexahedron", {0, 3, 1, 4, 2, 5, 7, 6}}};
    return m.at(rKey);
}

// little-endian binary scalar I/O (host assumed little-endian)
std::uint32_t ru32(std::istream& rIn) {
    std::uint32_t v;
    rIn.read(reinterpret_cast<char*>(&v), 4);
    if (rIn.gcount() != 4)
        throw ReadError("FLAC3D: unexpected end of file");
    return v;
}
double rf64(std::istream& rIn) {
    double v;
    rIn.read(reinterpret_cast<char*>(&v), 8);
    if (rIn.gcount() != 8)
        throw ReadError("FLAC3D: unexpected end of file");
    return v;
}
void wu32(std::ostream& rOs, std::uint32_t v) {
    rOs.write(reinterpret_cast<const char*>(&v), 4);
}
void wf64(std::ostream& rOs, double v) {
    rOs.write(reinterpret_cast<const char*>(&v), 8);
}

// Accumulating raw cell block: meshio node order will be applied later.
struct Flac3dRawBlock {
    std::string mType;                             // meshio type
    std::vector<std::vector<std::int64_t>> mRows;  // 0-based point indices
};

void add_cell(std::vector<Flac3dRawBlock>& rBlocks, const std::string& rType,
              std::vector<std::int64_t>&& cell) {
    if (rBlocks.empty() || rBlocks.back().mType != rType)
        rBlocks.push_back(Flac3dRawBlock{rType, {}});
    rBlocks.back().mRows.push_back(std::move(cell));
}

// The slot a group is written into when its name does not name one; the twin
// of `_flac3d.DEFAULT_SLOT`.
const char* const kFlac3dDefaultSlot = "Default";

// One ZGROUP/FGROUP as it appears in the file: a (namespace, name, slot)
// triple plus the file's own cell ids. Zone and face ids are SEPARATE 1-based
// namespaces, which is why `mIsFace` has to be carried rather than inferred.
struct Flac3dGroup {
    bool mIsFace = false;
    std::string mName;
    std::string mSlot;
    std::vector<std::int64_t> mIds;
};

std::uint16_t flac3d_ru16(std::istream& rIn) {
    std::uint16_t v;
    rIn.read(reinterpret_cast<char*>(&v), 2);
    if (rIn.gcount() != 2)
        throw ReadError("FLAC3D: unexpected end of file");
    return v;
}
void flac3d_wu16(std::ostream& rOs, std::uint16_t v) {
    rOs.write(reinterpret_cast<const char*>(&v), 2);
}

// A length-prefixed (uint16) string, the binary group section's name/slot.
std::string flac3d_read_str(std::istream& rIn) {
    const std::uint16_t n = flac3d_ru16(rIn);
    std::string out(n, '\0');
    if (n != 0) {
        rIn.read(&out[0], n);
        if (rIn.gcount() != static_cast<std::streamsize>(n))
            throw ReadError("FLAC3D: unexpected end of file");
    }
    return out;
}

std::string flac3d_trim(const std::string& rS) {
    std::size_t b = rS.find_first_not_of(" \t\r\n");
    if (b == std::string::npos)
        return std::string();
    std::size_t e = rS.find_last_not_of(" \t\r\n");
    return rS.substr(b, e - b + 1);
}

// The ascii slot is the raw remainder of the header line and may or may not be
// quoted (`SLOT "Default"` and `SLOT 5` are both real); the binary one is a
// bare length-prefixed string. Stripping here is what makes the two readers
// agree on a group's name for the same mesh -- the Python twin does the same
// in `_strip_quotes`.
std::string flac3d_unquote(const std::string& rS) {
    if (rS.size() >= 2 && rS.front() == rS.back() && (rS.front() == '"' || rS.front() == '\''))
        return rS.substr(1, rS.size() - 2);
    return rS;
}

// `ZGROUP "name" SLOT "slot"` / `FGROUP 'name' SLOT 5`.
Flac3dGroup flac3d_parse_group_header(const std::string& rLine, bool face) {
    const std::size_t q1 = rLine.find_first_of("'\"");
    if (q1 == std::string::npos)
        throw ReadError("FLAC3D: malformed group header: " + rLine);
    const std::size_t q2 = rLine.find(rLine[q1], q1 + 1);
    if (q2 == std::string::npos)
        throw ReadError("FLAC3D: malformed group header: " + rLine);

    Flac3dGroup g;
    g.mIsFace = face;
    g.mName = rLine.substr(q1 + 1, q2 - q1 - 1);

    const std::string rest = rLine.substr(q2 + 1);
    const std::size_t kw = rest.find("SLOT");
    if (kw == std::string::npos)
        throw ReadError("FLAC3D: expected SLOT in group header: " + rLine);
    g.mSlot = flac3d_unquote(flac3d_trim(rest.substr(kw + 4)));
    return g;
}

// The reader's `<zone|face>:<name>:<slot>` key, inverted. Split on the LAST
// colon because a group name may contain one while a slot may not; a name in
// any other shape keeps its whole self and takes the default slot. Keep this
// in step with `_flac3d._decompose_group_name` -- a drift here silently breaks
// the round trip's idempotence.
std::pair<std::string, std::string> flac3d_decompose_group_name(const std::string& rLabel,
                                                                const char* pFlag) {
    const std::string prefix = std::string(pFlag) + ":";
    const std::string rest =
        rLabel.compare(0, prefix.size(), prefix) == 0 ? rLabel.substr(prefix.size()) : rLabel;
    const std::size_t colon = rest.rfind(':');
    if (colon == std::string::npos)
        return {rest, kFlac3dDefaultSlot};
    return {rest.substr(0, colon), rest.substr(colon + 1)};
}

std::vector<std::string> flac3d_split_ws(const std::string& rS) {
    std::vector<std::string> out;
    std::istringstream iss(rS);
    std::string t;
    while (iss >> t)
        out.push_back(t);
    return out;
}

}  // namespace

Mesh read_flac3d(const std::string& rPath) {
    // Sniff binary (a null byte in the first 8 bytes).
    bool binary = false;
    {
        std::ifstream sniff(rPath, std::ios::binary);
        if (!sniff)
            throw ReadError("Could not open file: " + rPath);
        char block[8] = {0};
        sniff.read(block, 8);
        std::streamsize got = sniff.gcount();
        for (std::streamsize i = 0; i < got; ++i)
            if (block[i] == '\0') {
                binary = true;
                break;
            }
    }

    std::vector<double> points;                                // flat xyz
    std::unordered_map<std::int64_t, std::int64_t> point_ids;  // file id -> index
    std::vector<Flac3dRawBlock> z_blocks, f_blocks;
    std::vector<std::int64_t> z_ids, f_ids;
    std::vector<Flac3dGroup> groups;

    if (binary) {
        std::ifstream in(rPath, std::ios::binary);
        char hdr[8];
        in.read(hdr, 8);  // unknown header
        std::uint32_t num_nodes = ru32(in);
        points.reserve(num_nodes * 3);
        for (std::uint32_t i = 0; i < num_nodes; ++i) {
            std::uint32_t pid = ru32(in);
            double x = rf64(in), y = rf64(in), z = rf64(in);
            point_ids[pid] = static_cast<std::int64_t>(i);
            points.push_back(x);
            points.push_back(y);
            points.push_back(z);
        }
        for (int fi = 0; fi < 2; ++fi) {
            int dim = (fi == 0) ? 3 : 2;
            std::vector<Flac3dRawBlock>& blocks = (fi == 0) ? z_blocks : f_blocks;
            std::vector<std::int64_t>& ids = (fi == 0) ? z_ids : f_ids;
            std::uint32_t num_cells = ru32(in);
            const auto& tmap = numnodes_type(dim);
            for (std::uint32_t k = 0; k < num_cells; ++k) {
                std::uint32_t cid = ru32(in);
                std::uint32_t nv = ru32(in);
                std::vector<std::int64_t> cell(nv);
                for (std::uint32_t j = 0; j < nv; ++j)
                    cell[j] = point_ids.at(ru32(in));
                if (nv == 7)
                    cell.push_back(cell.back());
                auto it = tmap.find(static_cast<int>(cell.size()));
                if (it == tmap.end())
                    throw ReadError("FLAC3D: bad cell node count");
                ids.push_back(cid);
                add_cell(blocks, it->second, std::move(cell));
            }
            // Group section: uint32 count, then per group a uint16-prefixed
            // name, a uint16-prefixed slot, and a uint32-counted id list.
            const std::uint32_t num_groups = ru32(in);
            for (std::uint32_t g = 0; g < num_groups; ++g) {
                Flac3dGroup grp;
                grp.mIsFace = (fi == 1);
                grp.mName = flac3d_read_str(in);
                grp.mSlot = flac3d_read_str(in);
                const std::uint32_t n = ru32(in);
                grp.mIds.resize(n);
                for (std::uint32_t j = 0; j < n; ++j)
                    grp.mIds[j] = static_cast<std::int64_t>(ru32(in));
                groups.push_back(std::move(grp));
            }
        }
    } else {
        std::ifstream in(rPath, std::ios::binary);
        std::string line;
        // Index of the group whose id list the following lines belong to
        // (`npos` = none). A group header is followed by whitespace-separated
        // id lines until anything that is not one -- a comment, a new group, a
        // cell record, a blank line or EOF.
        std::size_t active = std::string::npos;
        while (std::getline(in, line)) {
            std::vector<std::string> s = flac3d_split_ws(line);
            if (s.empty()) {
                active = std::string::npos;
                continue;
            }
            if (s[0] == "ZGROUP" || s[0] == "FGROUP") {
                groups.push_back(flac3d_parse_group_header(line, s[0] == "FGROUP"));
                active = groups.size() - 1;
                continue;
            }
            if (active != std::string::npos && s[0][0] != '*' && s[0] != "G" && s[0] != "Z" &&
                s[0] != "F") {
                for (const std::string& t : s)
                    groups[active].mIds.push_back(std::strtoll(t.c_str(), nullptr, 10));
                continue;
            }
            active = std::string::npos;
            if (s[0] == "G") {
                std::int64_t pid = std::strtoll(s[1].c_str(), nullptr, 10);
                point_ids[pid] = static_cast<std::int64_t>(points.size() / 3);
                for (std::size_t j = 2; j < s.size(); ++j)
                    points.push_back(std::strtod(s[j].c_str(), nullptr));
            } else if (s[0] == "Z" || s[0] == "F") {
                int dim = (s[0] == "Z") ? 3 : 2;
                std::int64_t cid = std::strtoll(s[2].c_str(), nullptr, 10);
                bool is_b7 = (s[1] == "B7");
                std::vector<std::int64_t> cell;
                for (std::size_t j = 3; j < s.size(); ++j)
                    cell.push_back(point_ids.at(std::strtoll(s[j].c_str(), nullptr, 10)));
                if (is_b7)
                    cell.push_back(cell.back());
                const auto& tmap = numnodes_type(dim);
                auto it = tmap.find(static_cast<int>(cell.size()));
                if (it == tmap.end())
                    throw ReadError("FLAC3D: bad cell node count");
                if (dim == 3) {
                    z_ids.push_back(cid);
                    add_cell(z_blocks, it->second, std::move(cell));
                } else {
                    f_ids.push_back(cid);
                    add_cell(f_blocks, it->second, std::move(cell));
                }
            }
            // other lines (comments starting with '*') are ignored
        }
    }

    // Assemble: faces first, then zones (matching the Python reader).
    Mesh mesh;
    const std::int64_t npoints = static_cast<std::int64_t>(points.size() / 3);
    NDArray pts(DType::Float64, {static_cast<std::size_t>(npoints), 3});
    std::memcpy(pts.Data(), points.data(), points.size() * sizeof(double));
    mesh.AssignPoints(std::move(pts));

    std::vector<std::size_t> block_sizes;
    auto emit = [&](std::vector<Flac3dRawBlock>& blocks) {
        for (auto& b : blocks) {
            const std::vector<int>& ord =
                f2m_order(zone_key(b.mType).empty() ? face_key(b.mType) : zone_key(b.mType));
            std::size_t n = b.mRows.size();
            std::size_t k = ord.size();
            NDArray data(DType::Int64, {n, k});
            std::int64_t* dp = data.As<std::int64_t>();
            for (std::size_t r = 0; r < n; ++r)
                for (std::size_t j = 0; j < k; ++j)
                    dp[r * k + j] = b.mRows[r][ord[j]];
            mesh.AddCellBlock(b.mType, std::move(data));
            block_sizes.push_back(n);
        }
    };
    emit(f_blocks);
    emit(z_blocks);

    // ZGROUP/FGROUP -> one `RegionKind::Cell` region each.
    //
    // `emit` walks f_blocks then z_blocks in read order, so the global
    // (block-major) index of the i-th face read is exactly `i` and of the i-th
    // zone `f_ids.size() + i` -- the same invariant the cell_ids block below
    // already relies on. Zone and face ids live in separate 1-based namespaces,
    // hence two maps.
    if (!groups.empty()) {
        std::unordered_map<std::int64_t, std::int64_t> f_map, z_map;
        f_map.reserve(f_ids.size());
        z_map.reserve(z_ids.size());
        for (std::size_t i = 0; i < f_ids.size(); ++i)
            f_map[f_ids[i]] = static_cast<std::int64_t>(i);
        const std::int64_t z_base = static_cast<std::int64_t>(f_ids.size());
        for (std::size_t i = 0; i < z_ids.size(); ++i)
            z_map[z_ids[i]] = z_base + static_cast<std::int64_t>(i);

        for (const Flac3dGroup& g : groups) {
            const auto& map = g.mIsFace ? f_map : z_map;
            std::vector<std::int64_t> ent;
            ent.reserve(g.mIds.size());
            std::size_t dropped = 0;
            for (std::int64_t id : g.mIds) {
                auto it = map.find(id);
                if (it == map.end()) {
                    ++dropped;  // an id the file never defined -- never guessed at
                    continue;
                }
                ent.push_back(it->second);
            }
            if (dropped != 0)
                log::warn(
                    "FLAC3D: group '{}' names {} cell id(s) the file does not define; "
                    "ignored",
                    g.mName, dropped);
            NDArray entries(DType::Int64, {ent.size()});
            std::copy(ent.begin(), ent.end(), entries.As<std::int64_t>());
            // An empty group is still carried: the name is information (the
            // same rule detail/region_remap.hpp applies to every operation).
            mesh.AddRegion(
                Region(std::string(g.mIsFace ? "face:" : "zone:") + g.mName + ":" + g.mSlot,
                       RegionKind::Cell, std::move(entries)));
        }
    }

    // Global cell ids -> cell_data["cell_ids"], split per block.
    if (mesh.NumCellBlocks() != 0) {
        std::int64_t z_offset = static_cast<std::int64_t>(f_ids.size());
        std::vector<std::int64_t> all_ids;
        all_ids.reserve(f_ids.size() + z_ids.size());
        for (auto v : f_ids)
            all_ids.push_back(v);
        for (auto v : z_ids)
            all_ids.push_back(v + z_offset);

        std::vector<NDArray> id_blocks;
        std::size_t off = 0;
        for (std::size_t sz : block_sizes) {
            NDArray a(DType::Int64, {sz});
            for (std::size_t r = 0; r < sz; ++r)
                a.As<std::int64_t>()[r] = all_ids[off + r];
            off += sz;
            id_blocks.push_back(std::move(a));
        }
        mesh.AddCellData("cell_ids", std::move(id_blocks));
    }

    return mesh;
}

namespace {

// One group as it is about to be written: name, slot and this category's own
// 1-based cell ids.
struct Flac3dGroupOut {
    std::string mName;
    std::string mSlot;
    std::vector<std::uint32_t> mIds;
};

// Collect the mesh's `RegionKind::Cell` regions for one FLAC3D category.
//
// `rIdx` is that category's block list *in write order*, so the running
// counter here reproduces `_write_cells`' own `gid` exactly. A region whose
// name explicitly names the other category is skipped; one that names this
// category is emitted even when empty, so a file read from disk is a fixed
// point. Anything else is placed by its members alone, which is how a region
// carried in from another format can land in both sections at once.
std::vector<Flac3dGroupOut> flac3d_groups_for(const Mesh& rMesh,
                                              const std::vector<std::size_t>& rIdx,
                                              const char* pFlag, const char* pOther) {
    const std::vector<std::int64_t> bases = detail::block_bases(rMesh);

    std::unordered_map<std::size_t, std::int64_t> local_base;
    std::int64_t gid = 0;
    for (std::size_t b : rIdx) {
        local_base[b] = gid;
        gid += static_cast<std::int64_t>(rMesh.Cells(b).NumCells());
    }

    const std::string own_prefix = std::string(pFlag) + ":";
    const std::string other_prefix = std::string(pOther) + ":";

    std::vector<Flac3dGroupOut> out;
    for (std::size_t i = 0; i < rMesh.NumRegions(); ++i) {
        const Region& r = rMesh.Region(i);
        if (r.mKind != RegionKind::Cell)
            continue;
        if (r.mName.compare(0, other_prefix.size(), other_prefix) == 0)
            continue;

        std::vector<std::uint32_t> ids;
        const std::int64_t* e = r.Entries();
        for (std::size_t k = 0; k < r.NumEntries(); ++k) {
            const auto [b, row] = detail::global_to_block_row(bases, e[k]);
            auto it = local_base.find(b);
            if (b == static_cast<std::size_t>(-1) || it == local_base.end())
                continue;  // a cell of the other category, or out of range
            ids.push_back(
                static_cast<std::uint32_t>(it->second + static_cast<std::int64_t>(row) + 1));
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

        const bool named_for_us = r.mName.compare(0, own_prefix.size(), own_prefix) == 0;
        if (ids.empty() && !named_for_us)
            continue;

        const auto [name, slot] = flac3d_decompose_group_name(r.mName, pFlag);
        out.push_back(Flac3dGroupOut{name, slot, std::move(ids)});
    }
    return out;
}

// Group section emitters. The ascii layout -- 20 ids per line, each preceded
// by a space -- byte-matches `_flac3d._write_table`, and the binary one is the
// reader's own (uint16-prefixed name, uint16-prefixed slot, uint32-counted id
// list), so a file is readable whichever engine produced it.
void flac3d_write_groups_ascii(std::ostream& rOs, const std::vector<Flac3dGroupOut>& rGroups,
                               const char* pSection) {
    rOs << "* " << pSection << " GROUPS\n";
    const std::string kw = std::string(pSection) == "ZONE" ? "ZGROUP" : "FGROUP";
    for (const Flac3dGroupOut& g : rGroups) {
        rOs << kw << " \"" << g.mName << "\" SLOT \"" << g.mSlot << "\"\n";
        for (std::size_t k = 0; k < g.mIds.size(); ++k) {
            rOs << ' ' << g.mIds[k];
            if ((k + 1) % 20 == 0 || k + 1 == g.mIds.size())
                rOs << '\n';
        }
    }
}

void flac3d_write_groups_binary(std::ostream& rOs, const std::vector<Flac3dGroupOut>& rGroups) {
    wu32(rOs, static_cast<std::uint32_t>(rGroups.size()));
    for (const Flac3dGroupOut& g : rGroups) {
        flac3d_wu16(rOs, static_cast<std::uint16_t>(g.mName.size()));
        rOs.write(g.mName.data(), static_cast<std::streamsize>(g.mName.size()));
        flac3d_wu16(rOs, static_cast<std::uint16_t>(g.mSlot.size()));
        rOs.write(g.mSlot.data(), static_cast<std::streamsize>(g.mSlot.size()));
        wu32(rOs, static_cast<std::uint32_t>(g.mIds.size()));
        for (std::uint32_t v : g.mIds)
            wu32(rOs, v);
    }
}

// Reorder one zone cell to FLAC3D order, choosing the right-handed permutation
// via the scalar triple product of the first four ordered corners.
std::vector<std::int64_t> zone_cell_flac3d(const NDArray& rPoints, const NDArray& rData,
                                           std::size_t row, const std::string& rKey) {
    const std::vector<int>& o1 = m2f_order(rKey);
    const std::vector<int>& o2 = m2f_order2(rKey);
    const std::size_t ncols = detail::cols(rData);

    auto node = [&](int local) -> std::int64_t {
        return detail::read_int(rData, row * ncols + local);
    };
    auto coord = [&](std::int64_t p, int c) -> double {
        return detail::read_double(rPoints, static_cast<std::size_t>(p) * 3 + c);
    };

    // first four corners in FLAC3D order
    std::int64_t c0 = node(o1[0]), c1 = node(o1[1]), c2 = node(o1[2]), c3 = node(o1[3]);
    double a[3], b[3], c[3];
    for (int i = 0; i < 3; ++i) {
        a[i] = coord(c1, i) - coord(c0, i);
        b[i] = coord(c2, i) - coord(c0, i);
        c[i] = coord(c3, i) - coord(c0, i);
    }
    double cross0 = b[1] * c[2] - b[2] * c[1];
    double cross1 = b[2] * c[0] - b[0] * c[2];
    double cross2 = b[0] * c[1] - b[1] * c[0];
    double det = a[0] * cross0 + a[1] * cross1 + a[2] * cross2;

    const std::vector<int>& ord = (det > 0) ? o1 : o2;
    std::vector<std::int64_t> out(ord.size());
    for (std::size_t j = 0; j < ord.size(); ++j)
        out[j] = node(ord[j]);
    return out;
}

}  // namespace

void write_flac3d(const std::string& rPath, const Mesh& rMesh, const std::string& rFloatFmt,
                  bool binary) {
    // Split blocks by FLAC3D category.
    std::vector<std::size_t> zone_idx, face_idx;
    for (std::size_t i = 0; i < rMesh.NumCellBlocks(); ++i) {
        if (!zone_key(rMesh.Cells(i).Type()).empty())
            zone_idx.push_back(i);
        else if (!face_key(rMesh.Cells(i).Type()).empty())
            face_idx.push_back(i);
    }

    std::ofstream f(rPath, std::ios::binary);
    if (!f)
        throw WriteError("Could not open file for writing: " + rPath);

    const std::size_t npts = rMesh.NumPoints();
    const std::size_t pdim = rMesh.PointDim();
    const NDArray& points = rMesh.Points();

    const std::vector<Flac3dGroupOut> zgroups = flac3d_groups_for(rMesh, zone_idx, "zone", "face");
    const std::vector<Flac3dGroupOut> fgroups = flac3d_groups_for(rMesh, face_idx, "face", "zone");

    if (binary) {
        wu32(f, 1375135718u);
        wu32(f, 3u);
        // points
        wu32(f, static_cast<std::uint32_t>(npts));
        for (std::size_t i = 0; i < npts; ++i) {
            wu32(f, static_cast<std::uint32_t>(i + 1));
            for (int c = 0; c < 3; ++c)
                wf64(f,
                     c < static_cast<int>(pdim) ? detail::read_double(points, i * pdim + c) : 0.0);
        }
        // ZONES and FACES are numbered independently in a FLAC3D file -- both
        // start at 1 -- so each section gets its own counter. Sharing one made
        // the ids disagree with the group lists written beside them.
        std::uint32_t gid = 0;
        // zones
        std::uint32_t nz = 0;
        for (auto i : zone_idx)
            nz += static_cast<std::uint32_t>(rMesh.Cells(i).NumCells());
        wu32(f, nz);
        for (auto i : zone_idx) {
            const auto cb = rMesh.Cells(i);
            const NDArray& conn = cb.Conn();
            std::string key = zone_key(cb.Type());
            std::size_t n = cb.NumCells();
            // Right-handed reorder per row is independent -> compute in
            // parallel, then stream sequentially.
            std::vector<std::vector<std::int64_t>> zcells(n);
            parallel_for(n, [&](std::size_t r) {
                zcells[r] = zone_cell_flac3d(points, conn, r, key);
            });
            for (std::size_t r = 0; r < n; ++r) {
                const auto& cell = zcells[r];
                wu32(f, ++gid);
                wu32(f, static_cast<std::uint32_t>(cell.size()));
                for (auto v : cell)
                    wu32(f, static_cast<std::uint32_t>(v + 1));
            }
        }
        flac3d_write_groups_binary(f, zgroups);
        // faces
        gid = 0;
        std::uint32_t nf = 0;
        for (auto i : face_idx)
            nf += static_cast<std::uint32_t>(rMesh.Cells(i).NumCells());
        wu32(f, nf);
        for (auto i : face_idx) {
            const auto cb = rMesh.Cells(i);
            const NDArray& conn = cb.Conn();
            std::string key = face_key(cb.Type());
            const std::vector<int>& ord = m2f_order(key);
            std::size_t n = cb.NumCells();
            std::size_t ncols = detail::cols(conn);
            for (std::size_t r = 0; r < n; ++r) {
                wu32(f, ++gid);
                wu32(f, static_cast<std::uint32_t>(ord.size()));
                for (int local : ord)
                    wu32(f,
                         static_cast<std::uint32_t>(detail::read_int(conn, r * ncols + local) + 1));
            }
        }
        flac3d_write_groups_binary(f, fgroups);
        return;
    }

    // ASCII
    f << detail::provenance_render_lines(detail::SlotTier::Block, "* ");
    f << "* GRIDPOINTS\n";
    char buf[64];
    for (std::size_t i = 0; i < npts; ++i) {
        // `setw(8)` matches the Python reference writer's `"G\t{:8}\t"`, which
        // is what makes the two engines' ascii output byte-identical; the field
        // is whitespace-tokenized on read, so the padding carries no meaning.
        f << "G\t" << std::setw(8) << (i + 1) << std::setw(0) << "\t";
        for (int c = 0; c < 3; ++c) {
            double v =
                c < static_cast<int>(pdim) ? detail::read_double(points, i * pdim + c) : 0.0;
            std::snprintf(buf, sizeof(buf), ("%" + rFloatFmt).c_str(), v);
            f << buf << (c == 2 ? '\n' : '\t');
        }
    }

    std::int64_t gid = 0;
    f << "* ZONES\n";
    for (auto i : zone_idx) {
        const auto cb = rMesh.Cells(i);
        const NDArray& conn = cb.Conn();
        std::string key = zone_key(cb.Type());
        const char* abbr = flac3d_type(key);
        std::size_t n = cb.NumCells();
        // Right-handed reorder per row is independent -> compute in parallel,
        // then stream sequentially.
        std::vector<std::vector<std::int64_t>> zcells(n);
        parallel_for(n,
                     [&](std::size_t r) { zcells[r] = zone_cell_flac3d(points, conn, r, key); });
        for (std::size_t r = 0; r < n; ++r) {
            f << "Z " << abbr << " " << (++gid);
            for (auto v : zcells[r])
                f << " " << (v + 1);
            f << "\n";
        }
    }
    flac3d_write_groups_ascii(f, zgroups, "ZONE");

    gid = 0;
    f << "* FACES\n";
    for (auto i : face_idx) {
        const auto cb = rMesh.Cells(i);
        const NDArray& conn = cb.Conn();
        std::string key = face_key(cb.Type());
        const char* abbr = flac3d_type(key);
        const std::vector<int>& ord = m2f_order(key);
        std::size_t n = cb.NumCells();
        std::size_t ncols = detail::cols(conn);
        for (std::size_t r = 0; r < n; ++r) {
            f << "F " << abbr << " " << (++gid);
            for (int local : ord)
                f << " " << (detail::read_int(conn, r * ncols + local) + 1);
            f << "\n";
        }
    }
    flac3d_write_groups_ascii(f, fgroups, "FACE");
}

}  // namespace meshioplusplus
