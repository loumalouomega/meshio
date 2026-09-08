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
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// External includes
#ifdef MESHIOPLUSPLUS_HAS_JSON
#include <nlohmann/json.hpp>  // IWYU pragma: keep
#endif

// Project includes
#include "meshioplusplus/operations/pipeline.hpp"
#include "meshioplusplus/detail/provenance.hpp"
#include "meshioplusplus/exceptions.hpp"
#include "meshioplusplus/registry.hpp"
#include "meshioplusplus/skin.hpp"
#include "meshioplusplus/operations/agglomerate.hpp"
#include "meshioplusplus/operations/clean.hpp"
#include "meshioplusplus/operations/convert_cells.hpp"
#include "meshioplusplus/operations/curvature.hpp"
#include "meshioplusplus/operations/crop.hpp"
#include "meshioplusplus/operations/data_average.hpp"
#include "meshioplusplus/operations/data_calc.hpp"
#include "meshioplusplus/operations/data_common.hpp"
#include "meshioplusplus/operations/data_condition.hpp"
#include "meshioplusplus/operations/data_manage.hpp"
#include "meshioplusplus/operations/decimate.hpp"
#include "meshioplusplus/operations/decimate_volume.hpp"
#include "meshioplusplus/operations/error.hpp"
#include "meshioplusplus/operations/gradient.hpp"
#include "meshioplusplus/operations/hessian.hpp"
#include "meshioplusplus/operations/isosurface.hpp"
#include "meshioplusplus/operations/sdf.hpp"
#include "meshioplusplus/operations/voxelize.hpp"
#include "meshioplusplus/operations/partition.hpp"
#include "meshioplusplus/operations/quality.hpp"
#include "meshioplusplus/operations/refine.hpp"
#include "meshioplusplus/operations/reorder.hpp"
#include "meshioplusplus/operations/sequence.hpp"
#include "meshioplusplus/operations/slice.hpp"
#include "meshioplusplus/operations/smooth.hpp"
#include "meshioplusplus/operations/sniff.hpp"
#include "meshioplusplus/operations/subdivide.hpp"
#include "meshioplusplus/operations/surface.hpp"
#include "meshioplusplus/operations/transform.hpp"
#include "meshioplusplus/operations/remesh.hpp"
#include "meshioplusplus/operations/remesh_volume.hpp"
#include "meshioplusplus/operations/optimize_volume.hpp"

namespace meshioplusplus {

namespace {

// --------------------------------------------------------------------------
// Typed parameter access. Every getter throws std::invalid_argument naming the
// step and the key on a type mismatch -- a mis-typed settings entry must fail
// by name, never coerce silently (beyond the one deliberate int->double
// widening, which mirrors JSON's single number type).
// --------------------------------------------------------------------------

std::string pipe_err(const PipelineStep& rStep, const std::string& rWhat) {
    return "meshio++: pipeline: step '" + rStep.mOp + "': " + rWhat;
}

const PipelineValue* pipe_find(const PipelineStep& rStep, const char* pKey) {
    auto it = rStep.mParams.find(pKey);
    return it == rStep.mParams.end() ? nullptr : &it->second;
}

/// True when the value is an empty array of any element type. An empty JSON
/// array carries no element type, so the parser stores it arbitrarily and the
/// typed getters must all accept it as "empty of my type".
bool pipe_is_empty_array(const PipelineValue& rValue) {
    if (const auto* d = std::get_if<std::vector<double>>(&rValue))
        return d->empty();
    if (const auto* i = std::get_if<std::vector<std::int64_t>>(&rValue))
        return i->empty();
    if (const auto* s = std::get_if<std::vector<std::string>>(&rValue))
        return s->empty();
    return false;
}

double pipe_number(const PipelineStep& rStep, const char* pKey, double fallback) {
    const PipelineValue* v = pipe_find(rStep, pKey);
    if (!v)
        return fallback;
    if (const auto* d = std::get_if<double>(v))
        return *d;
    if (const auto* i = std::get_if<std::int64_t>(v))
        return static_cast<double>(*i);
    throw std::invalid_argument(
        pipe_err(rStep, std::string("parameter '") + pKey + "' must be a number"));
}

bool pipe_flag(const PipelineStep& rStep, const char* pKey, bool fallback) {
    const PipelineValue* v = pipe_find(rStep, pKey);
    if (!v)
        return fallback;
    if (const auto* b = std::get_if<bool>(v))
        return *b;
    throw std::invalid_argument(
        pipe_err(rStep, std::string("parameter '") + pKey + "' must be a boolean"));
}

std::string pipe_text(const PipelineStep& rStep, const char* pKey, const char* pFallback) {
    const PipelineValue* v = pipe_find(rStep, pKey);
    if (!v)
        return pFallback;
    if (const auto* s = std::get_if<std::string>(v))
        return *s;
    throw std::invalid_argument(
        pipe_err(rStep, std::string("parameter '") + pKey + "' must be a string"));
}

/// Numeric array (int entries widened to double). Empty when the key is absent.
std::vector<double> pipe_dvec(const PipelineStep& rStep, const char* pKey) {
    const PipelineValue* v = pipe_find(rStep, pKey);
    if (!v)
        return {};
    if (pipe_is_empty_array(*v))
        return {};
    if (const auto* d = std::get_if<std::vector<double>>(v))
        return *d;
    if (const auto* i = std::get_if<std::vector<std::int64_t>>(v)) {
        std::vector<double> out(i->begin(), i->end());
        return out;
    }
    throw std::invalid_argument(
        pipe_err(rStep, std::string("parameter '") + pKey + "' must be an array of numbers"));
}

/// Integer array. Doubles are accepted only when integral (JSON has one
/// number type, so `[0, 1.0]` is a legitimate spelling of an index list).
std::vector<std::int64_t> pipe_ivec(const PipelineStep& rStep, const char* pKey) {
    const PipelineValue* v = pipe_find(rStep, pKey);
    if (!v)
        return {};
    if (pipe_is_empty_array(*v))
        return {};
    if (const auto* i = std::get_if<std::vector<std::int64_t>>(v))
        return *i;
    if (const auto* d = std::get_if<std::vector<double>>(v)) {
        std::vector<std::int64_t> out;
        out.reserve(d->size());
        for (double x : *d) {
            if (x != std::floor(x))
                throw std::invalid_argument(
                    pipe_err(rStep, std::string("parameter '") + pKey + "' must hold integers"));
            out.push_back(static_cast<std::int64_t>(x));
        }
        return out;
    }
    throw std::invalid_argument(
        pipe_err(rStep, std::string("parameter '") + pKey + "' must be an array of integers"));
}

std::vector<std::string> pipe_svec(const PipelineStep& rStep, const char* pKey) {
    const PipelineValue* v = pipe_find(rStep, pKey);
    if (!v)
        return {};
    if (pipe_is_empty_array(*v))
        return {};
    if (const auto* s = std::get_if<std::vector<std::string>>(v))
        return *s;
    throw std::invalid_argument(
        pipe_err(rStep, std::string("parameter '") + pKey + "' must be an array of strings"));
}

/// A required 3-vector (`Point`, `Normal`, `Translate`, ...).
void pipe_vec3(const PipelineStep& rStep, const char* pKey, double* pOut) {
    std::vector<double> v = pipe_dvec(rStep, pKey);
    if (v.size() != 3)
        throw std::invalid_argument(
            pipe_err(rStep, std::string("parameter '") + pKey + "' must be an array of 3 numbers"));
    for (int i = 0; i < 3; ++i)
        pOut[i] = v[static_cast<std::size_t>(i)];
}

std::string pipe_trim(const std::string& rText) {
    std::size_t b = rText.find_first_not_of(" \t");
    if (b == std::string::npos)
        return "";
    std::size_t e = rText.find_last_not_of(" \t");
    return rText.substr(b, e - b + 1);
}

/// Cells across every block. The uniform mesh API counts per block only.
std::size_t pipe_total_cells(const Mesh& rMesh) {
    std::size_t n = 0;
    for (const auto& block : rMesh.CellRange())
        n += block.NumCells();
    return n;
}

// --------------------------------------------------------------------------
// The step vocabulary. One table, consumed by validate_pipeline_step and by
// the error messages -- the dispatcher itself reads known keys only, so a
// caller-side converter (the WASM binding) may copy every key it was given
// and rely on validation to reject the unknown ones.
// --------------------------------------------------------------------------

struct PipeOpSpec {
    const char* mName;
    std::vector<const char*> mKeys;
};

const std::vector<PipeOpSpec>& pipe_op_table() {
    static const std::vector<PipeOpSpec> table = {
        {"Quality", {}},
        {"Clean", {"Weld", "Atol", "RemoveOrphans", "DropDegenerate", "DropDuplicateCells"}},
        {"Smooth",
         {"Method", "Iterations", "Lambda", "Mu", "FixBoundary", "PreserveFeatures", "FeatureAngle",
          "GuardInversion"}},
        {"Refine",
         {"Levels", "Cells", "Region", "Array", "Compare", "Value", "Closure", "RecordLevels",
          "RecordHierarchy"}},
        {"Decimate",
         {"Ratio", "TargetFaces", "MaxError", "Placement", "PreserveBoundary", "PreserveFeatures",
          "FeatureAngle"}},
        {"DecimateVolume",
         {"Ratio", "TargetCells", "MaxError", "Placement", "PreserveBoundary", "PreserveFeatures",
          "FeatureAngle"}},
        {"Partition", {"Nparts", "Method", "Imbalance", "Mode", "Seed", "WeightsKey"}},
        {"Slice", {"Point", "Normal", "RecordParentIds"}},
        {"Section", {"Point", "Normal", "RecordParentIds"}},  // alias of Slice
        {"Gradient", {"Array", "Operator", "Method", "Location", "Output", "Component"}},
        {"Hessian", {"Array", "Method", "Location", "Output"}},
        {"Curvature",
         {"Mean", "Gaussian", "DualArea", "IncludeBoundary", "RecordArea",
          "RecordPrincipal", "Region"}},
        {"EstimateError", {"Array", "Method", "Marking", "MarkingValue", "Output", "Marked"}},
        {"Remesh",
         {"NumClusters", "Subdivide", "SubsampleRatio", "MaxSubdivide", "MaxIterations",
          "MaxRepairPasses", "Metric", "Gradation", "PreserveBoundary", "MaxAnisotropy"}},
        {"RemeshVolume",
         {"Resolution", "CellSize", "Bounds", "Padding", "PaddingRelative", "MaxCells", "MaxTets",
          "WarpFraction", "Sign", "WatertightCheck"}},
        {"OptimizeVolume",
         {"MaxIterations", "Relocate", "Flip", "PreserveBoundary", "MinImprovement"}},
        {"Isosurface", {"Array", "Isovalue", "Isovalues", "Component", "RecordParentIds"}},
        {"Voxelize",
         {"Resolution", "CellSize", "Bounds", "Padding", "PaddingRelative", "Fill",
          "AttachOccupancy", "MaxCells", "Sign"}},
        {"ComputeSdf",
         {"Structure", "Resolution", "CellSize", "Bounds", "Padding", "PaddingRelative",
          "RootResolution", "MaxDepth", "BandCells", "RecordLevels", "MaxCells", "Sign", "Location",
          "Band"}},
        {"Transform",
         {"Translate", "Scale", "RotateAxis", "RotateDegrees", "Matrix", "ScaleUnits",
          "RotateData"}},
        {"ConvertCells", {"Mode", "RecordParentIds"}},
        {"Subdivide", {"RecordParentIds"}},
        {"Agglomerate", {"TargetGroupSize"}},
        {"Crop", {"Bbox", "Point", "Normal", "Where", "Compare", "Value", "Mode", "RecordIds"}},
        {"ExtractSurface", {"RecordParentIds"}},
        {"ExtractSkin", {"Linearize"}},
        {"Reorder", {"Method"}},
        {"DataDrop", {"Point", "Cell", "Field", "IgnoreMissing"}},
        {"DataKeep", {"Point", "Cell", "Field"}},
        {"DataRename", {"Point", "Cell", "Field"}},
        {"DataCalc", {"Expr", "Location", "Overwrite"}},
        {"DataCondition",
         {"Mode", "Location", "Names", "Scope", "Lo", "Hi", "NanPolicy", "NanReplacement",
          "Suffix"}},
        {"ToCell", {"Names"}},
        {"ToPoint", {"Names", "Weight"}},
    };
    return table;
}

const PipeOpSpec* pipe_find_op(const std::string& rOp) {
    for (const auto& spec : pipe_op_table())
        if (rOp == spec.mName)
            return &spec;
    return nullptr;
}

/// The single-mesh-chain scope rule: these exist as operations but cannot be a
/// step, and the error must say where to go instead.
const char* pipe_excluded_hint(const std::string& rOp) {
    if (rOp == "Merge")
        return "'Merge' needs several input meshes and is not a pipeline step; "
               "use the `merge` CLI verb";
    if (rOp == "Interpolate")
        return "'Interpolate' needs a second (source) mesh and is not a pipeline "
               "step; use the `interpolate` CLI verb";
    if (rOp == "Split")
        return "'Split' produces several output meshes and is not a pipeline "
               "step; use the `split` CLI verb";
    if (rOp == "Diff")
        return "'Diff' compares two meshes and is not a pipeline step; use the "
               "`diff` CLI verb";
    if (rOp == "UndoGreen")
        return "'UndoGreen' needs a second (coarse) mesh and is not a pipeline "
               "step; use the `undo-green` CLI verb";
    if (rOp == "ConservativeInterpolate")
        return "'ConservativeInterpolate' needs a second (source) mesh and is not "
               "a pipeline step; use the `conservative-interpolate` CLI verb";
    return nullptr;
}

/// Renders one pipeline step as `Op(Key=value, Key=value, ...)` for the
/// provenance operation chain -- `mParams` is a `std::map`, so key order is
/// already deterministic without a separate sort here.
std::string pipe_render_op(const PipelineStep& rStep) {
    std::ostringstream out;
    out << rStep.mOp << "(";
    bool first = true;
    for (const auto& [key, value] : rStep.mParams) {
        if (!first)
            out << ", ";
        first = false;
        out << key << "=";
        std::visit(
            [&](const auto& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, bool>) {
                    out << (v ? "true" : "false");
                } else if constexpr (std::is_same_v<T, std::string>) {
                    out << v;
                } else if constexpr (std::is_same_v<T, std::vector<double>> ||
                                     std::is_same_v<T, std::vector<std::int64_t>> ||
                                     std::is_same_v<T, std::vector<std::string>>) {
                    out << "[";
                    for (std::size_t i = 0; i < v.size(); ++i) {
                        if (i)
                            out << ", ";
                        out << v[i];
                    }
                    out << "]";
                } else {
                    out << v;
                }
            },
            value);
    }
    out << ")";
    return out.str();
}

void pipe_push_step(PipelineReport& rReport, const PipelineStep& rStep,
                    std::vector<std::pair<std::string, double>> counters = {}) {
    rReport.mSteps.push_back(PipelineStepReport{rStep.mOp, std::move(counters)});
    // A no-op when no provenance scope is open (the common case); when one
    // is, this is what lets a written file's "Operation:" lines match the
    // pipeline's own report instead of drifting from it.
    detail::provenance_add_operation(pipe_render_op(rStep));
}

// --------------------------------------------------------------------------
// Per-op appliers. The nine ops the browser viewer's pipeline already had are
// transcriptions of the former wasm `apply_one_op` -- defaults, counters and
// warning strings identical (bindings/wasm/js_bindings.cpp now dispatches
// through here, so drift is structurally impossible).
// --------------------------------------------------------------------------

Mesh pipe_apply_transform(Mesh mesh, const PipelineStep& rStep, PipelineReport& rReport) {
    AffineTransform xf;
    int given = 0;
    if (pipe_find(rStep, "Translate")) {
        double t[3];
        pipe_vec3(rStep, "Translate", t);
        xf = transform_translation(t[0], t[1], t[2]);
        ++given;
    }
    if (pipe_find(rStep, "Scale")) {
        std::vector<double> v = pipe_dvec(rStep, "Scale");
        // A bare number is the uniform-scale spelling.
        if (v.empty())
            v.assign(1, pipe_number(rStep, "Scale", 1.0));
        if (v.size() == 1)
            xf = transform_scale(v[0], v[0], v[0]);
        else if (v.size() == 3)
            xf = transform_scale(v[0], v[1], v[2]);
        else
            throw std::invalid_argument(
                pipe_err(rStep, "'Scale' expects one number or an array of 3"));
        ++given;
    }
    if (pipe_find(rStep, "RotateAxis") || pipe_find(rStep, "RotateDegrees")) {
        double axis[3];
        pipe_vec3(rStep, "RotateAxis", axis);
        if (!pipe_find(rStep, "RotateDegrees"))
            throw std::invalid_argument(pipe_err(rStep, "'RotateAxis' requires 'RotateDegrees'"));
        const double deg = pipe_number(rStep, "RotateDegrees", 0.0);
        xf = transform_rotation(axis[0], axis[1], axis[2], deg * 3.14159265358979323846 / 180.0);
        ++given;
    }
    if (pipe_find(rStep, "Matrix")) {
        std::vector<double> v = pipe_dvec(rStep, "Matrix");
        if (v.size() != 16)
            throw std::invalid_argument(
                pipe_err(rStep, "'Matrix' expects 16 values (row-major 4x4)"));
        xf = transform_from_matrix(v.data());
        ++given;
    }
    if (pipe_find(rStep, "ScaleUnits")) {
        xf = transform_units(pipe_number(rStep, "ScaleUnits", 1.0));
        ++given;
    }
    if (given != 1)
        throw std::invalid_argument(
            pipe_err(rStep,
                     "give exactly one of 'Translate'/'Scale'/'RotateAxis'+'RotateDegrees'/"
                     "'Matrix'/'ScaleUnits'"));
    Mesh out = transform(mesh, xf, pipe_flag(rStep, "RotateData", false));
    pipe_push_step(rReport, rStep);
    return out;
}

Mesh pipe_apply_data_manage(Mesh mesh, const PipelineStep& rStep, PipelineReport& rReport) {
    // DataDrop / DataKeep / DataRename share the location-keyed shape.
    struct Loc {
        const char* mKey;
        DataLocation mLocation;
    };
    static const Loc locations[] = {{"Point", DataLocation::Point},
                                    {"Cell", DataLocation::Cell},
                                    {"Field", DataLocation::Field}};
    if (rStep.mOp == "DataDrop") {
        const bool ignore_missing = pipe_flag(rStep, "IgnoreMissing", false);
        for (const Loc& loc : locations)
            if (pipe_find(rStep, loc.mKey))
                mesh = data_drop(mesh, loc.mLocation, pipe_svec(rStep, loc.mKey), ignore_missing);
    } else if (rStep.mOp == "DataKeep") {
        // Keep affects only the locations it mentions (the data_manage rule) --
        // an absent key leaves that location alone, an empty list drops it all.
        for (const Loc& loc : locations)
            if (pipe_find(rStep, loc.mKey))
                mesh = data_keep(mesh, loc.mLocation, pipe_svec(rStep, loc.mKey),
                                 /*ignore_missing=*/false);
    } else {  // DataRename
        for (const Loc& loc : locations) {
            for (const std::string& entry : pipe_svec(rStep, loc.mKey)) {
                // "OLD:NEW" splits on the LAST colon: data names routinely
                // carry colons (`gmsh:physical`) -- the CLI's documented rule.
                std::size_t pos = entry.rfind(':');
                if (pos == std::string::npos || pos == 0 || pos + 1 == entry.size())
                    throw std::invalid_argument(
                        pipe_err(rStep, "rename entry '" + entry + "' must be 'OLD:NEW'"));
                mesh =
                    data_rename(mesh, loc.mLocation, entry.substr(0, pos), entry.substr(pos + 1));
            }
        }
    }
    pipe_push_step(rReport, rStep);
    return mesh;
}

}  // namespace

std::vector<std::pair<std::string, std::vector<std::string>>> pipeline_op_table() {
    std::vector<std::pair<std::string, std::vector<std::string>>> out;
    for (const auto& spec : pipe_op_table()) {
        std::vector<std::string> keys(spec.mKeys.begin(), spec.mKeys.end());
        out.emplace_back(spec.mName, std::move(keys));
    }
    return out;
}

void validate_pipeline_step(const PipelineStep& rStep) {
    const PipeOpSpec* spec = pipe_find_op(rStep.mOp);
    if (!spec) {
        if (const char* hint = pipe_excluded_hint(rStep.mOp))
            throw std::invalid_argument("meshio++: pipeline: " + std::string(hint));
        std::string known;
        for (const auto& s : pipe_op_table()) {
            if (!known.empty())
                known += ", ";
            known += s.mName;
        }
        throw std::invalid_argument("meshio++: pipeline: unknown operation '" + rStep.mOp +
                                    "' (known: " + known + ")");
    }
    for (const auto& kv : rStep.mParams) {
        const bool known = std::any_of(spec->mKeys.begin(), spec->mKeys.end(),
                                       [&](const char* k) { return kv.first == k; });
        if (!known) {
            std::string keys;
            for (const char* k : spec->mKeys) {
                if (!keys.empty())
                    keys += ", ";
                keys += k;
            }
            throw std::invalid_argument(pipe_err(
                rStep,
                "unknown parameter '" + kv.first + "'" +
                    (keys.empty() ? " (the op takes no parameters)" : " (known: " + keys + ")")));
        }
    }
}

Mesh apply_pipeline_step(Mesh mesh, const PipelineStep& rStep, PipelineReport& rReport) {
    const std::string& op = rStep.mOp;

    if (op == "Quality") {
        Mesh out = attach_quality(mesh);
        pipe_push_step(rReport, rStep);
        return out;
    }
    if (op == "Clean") {
        CleanOptions opts;
        opts.weld = pipe_flag(rStep, "Weld", false);
        opts.atol = pipe_number(rStep, "Atol", 1e-8);
        opts.remove_orphans = pipe_flag(rStep, "RemoveOrphans", true);
        opts.drop_degenerate = pipe_flag(rStep, "DropDegenerate", true);
        opts.drop_duplicate_cells = pipe_flag(rStep, "DropDuplicateCells", true);
        auto result = clean(mesh, opts);
        pipe_push_step(
            rReport, rStep,
            {{"PointsWelded", static_cast<double>(result.mPointsWelded)},
             {"PointsRemovedOrphan", static_cast<double>(result.mPointsRemovedOrphan)},
             {"CellsDroppedDegenerate", static_cast<double>(result.mCellsDroppedDegenerate)},
             {"CellsDroppedDuplicate", static_cast<double>(result.mCellsDroppedDuplicate)}});
        return std::move(result.mMesh);
    }
    if (op == "Smooth") {
        SmoothOptions opts;
        opts.mMethod = smooth_method_from_name(pipe_text(rStep, "Method", "taubin"));
        opts.mIterations = static_cast<int>(pipe_number(rStep, "Iterations", 10));
        opts.mLambda = pipe_number(rStep, "Lambda", -1.0);
        opts.mMu = pipe_number(rStep, "Mu", -0.34);
        opts.mFixBoundary = pipe_flag(rStep, "FixBoundary", true);
        opts.mPreserveFeatures = pipe_flag(rStep, "PreserveFeatures", true);
        opts.mFeatureAngleDeg = pipe_number(rStep, "FeatureAngle", 30.0);
        opts.mGuardInversion = pipe_flag(rStep, "GuardInversion", true);
        auto result = smooth(mesh, opts);
        pipe_push_step(rReport, rStep,
                       {{"NumNodesMoved", static_cast<double>(result.mNumNodesMoved)},
                        {"MaxDisplacement", result.mMaxDisplacement},
                        {"NumSkippedInversion", static_cast<double>(result.mNumSkippedInversion)}});
        return std::move(result.mMesh);
    }
    if (op == "Refine") {
        RefineOptions opts;
        opts.mLevels = static_cast<int>(pipe_number(rStep, "Levels", 1));
        opts.mRecordLevels = pipe_flag(rStep, "RecordLevels", false);
        opts.mRecordHierarchy = pipe_flag(rStep, "RecordHierarchy", false);
        opts.mClosure = refine_closure_from_name(pipe_text(rStep, "Closure", ""));
        opts.mCells = pipe_ivec(rStep, "Cells");
        opts.mRegion = pipe_text(rStep, "Region", "");
        opts.mPredicateArray = pipe_text(rStep, "Array", "");
        if (!opts.mPredicateArray.empty()) {
            // `Compare`, not `Op`: `Op` is the step's own discriminant.
            opts.mPredicateOp = refine_compare_from_name(pipe_text(rStep, "Compare", "<"));
            opts.mPredicateValue = pipe_number(rStep, "Value", 0.0);
        }
        auto result = refine(mesh, opts);
        pipe_push_step(rReport, rStep);
        return std::move(result.mMesh);
    }
    if (op == "Decimate") {
        DecimateOptions opts;
        opts.mTargetRatio = pipe_number(rStep, "Ratio", -1.0);
        const double faces = pipe_number(rStep, "TargetFaces", -1.0);
        opts.mTargetFaces =
            faces < 0.0 ? static_cast<std::int64_t>(-1) : static_cast<std::int64_t>(faces);
        opts.mMaxError = pipe_number(rStep, "MaxError", -1.0);
        if (opts.mTargetRatio < 0.0 && opts.mTargetFaces < 0 && opts.mMaxError < 0.0)
            opts.mTargetRatio = 0.5;  // a usable pipeline-chip default
        opts.mPlacement = decimate_placement_from_name(pipe_text(rStep, "Placement", "optimal"));
        opts.mPreserveBoundary = pipe_flag(rStep, "PreserveBoundary", true);
        opts.mPreserveFeatures = pipe_flag(rStep, "PreserveFeatures", true);
        opts.mFeatureAngleDeg = pipe_number(rStep, "FeatureAngle", 30.0);
        auto result = decimate(mesh, opts);
        pipe_push_step(rReport, rStep,
                       {{"FacesRemoved", static_cast<double>(result.mFacesRemoved)},
                        {"CollapsesRejected", static_cast<double>(result.mCollapsesRejected)}});
        return std::move(result.mMesh);
    }
    if (op == "DecimateVolume") {
        DecimateVolumeOptions opts;
        opts.mTargetRatio = pipe_number(rStep, "Ratio", -1.0);
        const double cells = pipe_number(rStep, "TargetCells", -1.0);
        opts.mTargetCells =
            cells < 0.0 ? static_cast<std::int64_t>(-1) : static_cast<std::int64_t>(cells);
        opts.mMaxError = pipe_number(rStep, "MaxError", -1.0);
        if (opts.mTargetRatio < 0.0 && opts.mTargetCells < 0 && opts.mMaxError < 0.0)
            opts.mTargetRatio = 0.5;  // a usable pipeline-chip default
        opts.mPlacement = decimate_placement_from_name(pipe_text(rStep, "Placement", "optimal"));
        opts.mPreserveBoundary = pipe_flag(rStep, "PreserveBoundary", false);
        opts.mPreserveFeatures = pipe_flag(rStep, "PreserveFeatures", true);
        opts.mFeatureAngleDeg = pipe_number(rStep, "FeatureAngle", 30.0);
        auto result = decimate_volume(mesh, opts);
        pipe_push_step(rReport, rStep,
                       {{"TetsRemoved", static_cast<double>(result.mTetsRemoved)},
                        {"CollapsesRejected", static_cast<double>(result.mCollapsesRejected)}});
        return std::move(result.mMesh);
    }
    if (op == "Partition") {
        PartitionOptions opts;
        opts.mNParts = static_cast<int>(pipe_number(rStep, "Nparts", 2));
        opts.mMethod = partition_method_from_name(pipe_text(rStep, "Method", "auto"));
        opts.mImbalance = pipe_number(rStep, "Imbalance", 0.03);
        opts.mMode = partition_mode_from_name(pipe_text(rStep, "Mode", ""));
        opts.mSeed = static_cast<int>(pipe_number(rStep, "Seed", 0));
        opts.mWeightsKey = pipe_text(rStep, "WeightsKey", "");
        // Attach the assignment as cell data rather than splitting into
        // pieces: a pipeline chains one mesh, and colour-by-part is what the
        // browser viewer's partition chip always meant.
        auto labels = partition_labels(mesh, opts);
        mesh.AddCellData("partition:part", std::move(labels));
        pipe_push_step(rReport, rStep, {{"Nparts", pipe_number(rStep, "Nparts", 2)}});
        return mesh;
    }
    if (op == "Slice" || op == "Section") {
        // The planar cross-section: slice() intersects the mesh with the plane
        // and returns a surface (triangle/quad) one dimension lower.
        SliceOptions opts;
        pipe_vec3(rStep, "Point", opts.mOrigin.data());
        pipe_vec3(rStep, "Normal", opts.mNormal.data());
        opts.mRecordParentIds = pipe_flag(rStep, "RecordParentIds", false);
        Mesh section = slice(mesh, opts);
        pipe_push_step(rReport, rStep,
                       {{"SectionFaces", static_cast<double>(pipe_total_cells(section))}});
        if (pipe_total_cells(section) == 0)
            rReport.mWarnings.push_back(
                "the section is empty; the plane misses the mesh -- move or flip it");
        return section;
    }
    if (op == "Gradient") {
        // A pure data step: geometry is untouched, so the pipeline carries the
        // mesh straight through with one new array attached.
        GradientOptions opts;
        opts.mArrayName = pipe_text(rStep, "Array", "");
        opts.mOperator = gradient_operator_from_name(pipe_text(rStep, "Operator", ""));
        opts.mMethod = gradient_method_from_name(pipe_text(rStep, "Method", ""));
        opts.mLocation = data_location_from_name(pipe_text(rStep, "Location", "cell"));
        opts.mOutputName = pipe_text(rStep, "Output", "");
        const int gcomp = static_cast<int>(pipe_number(rStep, "Component", -1.0));
        if (gcomp >= 0)
            opts.mComponent = gcomp;
        opts.mOverwrite = true;
        GradientResult gr = gradient(mesh, opts);
        pipe_push_step(rReport, rStep,
                       {{"NumSkipped", static_cast<double>(gr.mNumSkipped)},
                        {"NumFallback", static_cast<double>(gr.mNumFallback)}});
        if (gr.mNumSkipped > 0)
            rReport.mWarnings.push_back("gradient: " + std::to_string(gr.mNumSkipped) +
                                        " cell(s) could not be differentiated and are NaN");
        return std::move(gr.mMesh);
    }
    if (op == "Hessian") {
        // A pure data step: geometry is untouched, so the pipeline carries the
        // mesh straight through with one new (n,9) array attached.
        HessianOptions opts;
        opts.mArrayName = pipe_text(rStep, "Array", "");
        opts.mMethod = gradient_method_from_name(pipe_text(rStep, "Method", ""));
        opts.mLocation = data_location_from_name(pipe_text(rStep, "Location", "cell"));
        opts.mOutputName = pipe_text(rStep, "Output", "");
        opts.mOverwrite = true;
        HessianResult hr = hessian(mesh, opts);
        pipe_push_step(rReport, rStep,
                       {{"NumSkipped", static_cast<double>(hr.mNumSkipped)},
                        {"NumFallback", static_cast<double>(hr.mNumFallback)}});
        if (hr.mNumSkipped > 0)
            rReport.mWarnings.push_back("hessian: " + std::to_string(hr.mNumSkipped) +
                                        " cell(s) could not be evaluated and are NaN");
        return std::move(hr.mMesh);
    }
    if (op == "Curvature") {
        // A pure data step: geometry is untouched, so the pipeline carries the
        // mesh straight through with the curvature arrays attached.
        CurvatureOptions opts;
        opts.mMean = pipe_flag(rStep, "Mean", true);
        opts.mGaussian = pipe_flag(rStep, "Gaussian", true);
        opts.mDualArea = curvature_dual_area_from_name(pipe_text(rStep, "DualArea", "mixed-voronoi"));
        opts.mIncludeBoundary = pipe_flag(rStep, "IncludeBoundary", false);
        opts.mRecordArea = pipe_flag(rStep, "RecordArea", false);
        opts.mRecordPrincipal = pipe_flag(rStep, "RecordPrincipal", false);
        opts.mRegion = pipe_text(rStep, "Region", "");
        CurvatureResult cr = compute_curvature(mesh, opts);
        pipe_push_step(rReport, rStep,
                       {{"NumBoundary", static_cast<double>(cr.mNumBoundary)},
                        {"NumIsolated", static_cast<double>(cr.mNumIsolated)},
                        {"NumDegenerate", static_cast<double>(cr.mNumDegenerate)},
                        {"TotalAngleDefect", cr.mTotalAngleDefect}});
        // H's sign comes from the surface's own winding, so a mesh whose facets
        // disagree about which side is out yields sign-flipped patches with no
        // error raised. Say so rather than letting it pass silently.
        if (cr.mQuality.mInconsistentPairs > 0)
            rReport.mWarnings.push_back(
                "curvature: " + std::to_string(cr.mQuality.mInconsistentPairs) +
                " edge pair(s) wind the same way, so the sign of 'curvature:mean' is not "
                "trustworthy");
        return std::move(cr.mMesh);
    }
    if (op == "EstimateError") {
        // A pure data step: geometry is untouched, so the pipeline carries the
        // mesh straight through with the indicator (and, if requested, the
        // marking) array attached.
        ErrorOptions opts;
        opts.mArrayName = pipe_text(rStep, "Array", "");
        opts.mMethod = error_method_from_name(pipe_text(rStep, "Method", ""));
        opts.mMarking = error_marking_from_name(pipe_text(rStep, "Marking", ""));
        opts.mMarkingValue = pipe_number(rStep, "MarkingValue", 0.0);
        opts.mOutputName = pipe_text(rStep, "Output", "");
        opts.mMarkedName = pipe_text(rStep, "Marked", "");
        opts.mOverwrite = true;
        ErrorResult er = estimate_error(mesh, opts);
        pipe_push_step(rReport, rStep,
                       {{"GlobalError", er.mGlobalError},
                        {"NumSkipped", static_cast<double>(er.mNumSkipped)},
                        {"NumMarked", static_cast<double>(er.mNumMarked)}});
        if (er.mNumSkipped > 0)
            rReport.mWarnings.push_back("estimate_error: " + std::to_string(er.mNumSkipped) +
                                        " cell(s) could not be evaluated and are NaN");
        return std::move(er.mMesh);
    }
    if (op == "Remesh") {
        // Unlike every other step, the output has NO correspondence to the
        // input: new points, new connectivity. point_data/cell_data/regions
        // are dropped (remesh's own contract), field_data carries through.
        RemeshOptions opts;
        opts.mNumClusters = static_cast<std::int64_t>(pipe_number(rStep, "NumClusters", 0));
        opts.mSubdivide = static_cast<int>(pipe_number(rStep, "Subdivide", -1));
        opts.mSubsampleRatio = pipe_number(rStep, "SubsampleRatio", 10.0);
        opts.mMaxSubdivide = static_cast<int>(pipe_number(rStep, "MaxSubdivide", 4));
        opts.mMaxIterations = static_cast<int>(pipe_number(rStep, "MaxIterations", 100));
        opts.mMaxRepairPasses = static_cast<int>(pipe_number(rStep, "MaxRepairPasses", 10));
        opts.mMetric = remesh_metric_from_name(pipe_text(rStep, "Metric", "isotropic"));
        opts.mGradation = pipe_number(rStep, "Gradation", 0.0);
        opts.mPreserveBoundary = pipe_flag(rStep, "PreserveBoundary", true);
        opts.mMaxAnisotropy = pipe_number(rStep, "MaxAnisotropy", kRemeshDefaultMaxAnisotropy);
        RemeshResult rr = remesh(mesh, opts);
        pipe_push_step(
            rReport, rStep,
            {{"NumClusters", static_cast<double>(rr.mNumClusters)},
             {"NumIterations", static_cast<double>(rr.mNumIterations)},
             {"SubdivideApplied", static_cast<double>(rr.mSubdivideApplied)},
             {"NumIsolatedClusters", static_cast<double>(rr.mNumIsolatedClusters)},
             {"NumNonManifoldVertices", static_cast<double>(rr.mNumNonManifoldVertices)}});
        if (rr.mNumIsolatedClusters > 0 || rr.mNumNonManifoldVertices > 0)
            rReport.mWarnings.push_back(
                "remesh: " + std::to_string(rr.mNumIsolatedClusters) +
                " isolated cluster(s), " + std::to_string(rr.mNumNonManifoldVertices) +
                " non-manifold vertex/vertices could not be repaired; output may be "
                "non-manifold near them");
        return std::move(rr.mMesh);
    }
    if (op == "RemeshVolume") {
        // Remesh's volumetric sibling: same no-correspondence-with-the-input
        // contract, same lattice-sizing vocabulary as Voxelize/ComputeSdf.
        RemeshVolumeOptions opts;
        const std::vector<std::int64_t> resolution = pipe_ivec(rStep, "Resolution");
        if (resolution.size() == 3)
            opts.mResolution =
                std::array<std::int64_t, 3>{{resolution[0], resolution[1], resolution[2]}};
        if (pipe_find(rStep, "CellSize") != nullptr)
            opts.mCellSize = pipe_number(rStep, "CellSize", 0.0);
        const std::vector<double> bounds = pipe_dvec(rStep, "Bounds");
        if (bounds.size() == 6)
            opts.mBounds = std::array<double, 6>{
                {bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]}};
        opts.mPadding = pipe_number(rStep, "Padding", 0.0);
        opts.mPaddingRelative = pipe_number(rStep, "PaddingRelative", 0.1);
        opts.mMaxCells = static_cast<std::int64_t>(pipe_number(rStep, "MaxCells", 20000000.0));
        opts.mMaxTets = static_cast<std::int64_t>(pipe_number(rStep, "MaxTets", 20000000.0));
        opts.mWarpFraction = pipe_number(rStep, "WarpFraction", 0.35);
        opts.mDistance.mSign = sdf_sign_from_name(pipe_text(rStep, "Sign", "pseudonormal"));
        opts.mDistance.mWatertightCheck =
            sdf_watertight_check_from_name(pipe_text(rStep, "WatertightCheck", "warn"));
        RemeshVolumeResult rv = remesh_volume(mesh, opts);
        pipe_push_step(rReport, rStep,
                       {{"NumTets", static_cast<double>(rv.mNumTets)},
                        {"NumVerticesWarped", static_cast<double>(rv.mNumVerticesWarped)},
                        {"NumTetsRejected", static_cast<double>(rv.mNumTetsRejected)},
                        {"NumNonManifoldEdges", static_cast<double>(rv.mNumNonManifoldEdges)}});
        if (rv.mNumNonManifoldEdges > 0)
            rReport.mWarnings.push_back(
                "remesh_volume: " + std::to_string(rv.mNumNonManifoldEdges) +
                " non-manifold boundary edge(s), from warping (see doc/remesh_volume.md)");
        return std::move(rv.mMesh);
    }
    if (op == "OptimizeVolume") {
        // ODT remeshing: relocate vertices AND flip connectivity (2-3/3-2). The
        // point set is invariant, so point_data/field_data and Point regions
        // carry; cell_data + Cell/Side regions are dropped (a flip has no cell
        // map).
        OptimizeVolumeOptions opts;
        opts.mMaxIterations = static_cast<int>(pipe_number(rStep, "MaxIterations", 10.0));
        opts.mRelocate = pipe_flag(rStep, "Relocate", true);
        opts.mFlip = pipe_flag(rStep, "Flip", true);
        opts.mPreserveBoundary = pipe_flag(rStep, "PreserveBoundary", true);
        opts.mMinImprovement = pipe_number(rStep, "MinImprovement", 1e-6);
        OptimizeVolumeResult ov = optimize_volume(mesh, opts);
        pipe_push_step(rReport, rStep,
                       {{"NumFlips", static_cast<double>(ov.mNumFlips)},
                        {"Num23Flips", static_cast<double>(ov.mNum23Flips)},
                        {"Num32Flips", static_cast<double>(ov.mNum32Flips)},
                        {"NumVerticesMoved", static_cast<double>(ov.mNumVerticesMoved)},
                        {"NumTets", static_cast<double>(ov.mNumTets)},
                        {"MinQualityBefore", ov.mMinQualityBefore},
                        {"MinQualityAfter", ov.mMinQualityAfter}});
        return std::move(ov.mMesh);
    }
    if (op == "Voxelize") {
        // A regular grid around the mesh. Unlike every other step this one does
        // not transform its input's geometry -- it replaces it -- which is
        // exactly why it is useful as a pipeline step: read a skin, voxelize it,
        // write a grid.
        VoxelOptions opts;
        const std::vector<std::int64_t> resolution = pipe_ivec(rStep, "Resolution");
        if (resolution.size() == 3)
            opts.mResolution =
                std::array<std::int64_t, 3>{{resolution[0], resolution[1], resolution[2]}};
        if (pipe_find(rStep, "CellSize") != nullptr)
            opts.mCellSize = pipe_number(rStep, "CellSize", 0.0);
        const std::vector<double> bounds = pipe_dvec(rStep, "Bounds");
        if (bounds.size() == 6)
            opts.mBounds = std::array<double, 6>{
                {bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]}};
        opts.mPadding = pipe_number(rStep, "Padding", 0.0);
        opts.mPaddingRelative = pipe_number(rStep, "PaddingRelative", 0.0);
        opts.mFill = voxel_fill_from_name(pipe_text(rStep, "Fill", "all"));
        opts.mAttachOccupancy = pipe_flag(rStep, "AttachOccupancy", false);
        opts.mMaxCells = static_cast<std::int64_t>(pipe_number(rStep, "MaxCells", 20000000.0));
        opts.mDistance.mSign = sdf_sign_from_name(pipe_text(rStep, "Sign", "pseudonormal"));
        VoxelResult r = voxelize(mesh, opts);
        pipe_push_step(rReport, rStep, {{"NumOccupied", static_cast<double>(r.mNumOccupied)}});
        return std::move(r.mMesh);
    }

    if (op == "ComputeSdf") {
        // Voxelize's sibling: it likewise REPLACES the input's geometry, taking
        // a surface in and handing a filled grid back, which is exactly the
        // shape a single-mesh chain wants.
        SdfOptions opts;
        opts.mStructure = sdf_structure_from_name(pipe_text(rStep, "Structure", "voxel"));
        const std::vector<std::int64_t> resolution = pipe_ivec(rStep, "Resolution");
        if (resolution.size() == 3)
            opts.mResolution =
                std::array<std::int64_t, 3>{{resolution[0], resolution[1], resolution[2]}};
        if (pipe_find(rStep, "CellSize") != nullptr)
            opts.mCellSize = pipe_number(rStep, "CellSize", 0.0);
        const std::vector<double> bounds = pipe_dvec(rStep, "Bounds");
        if (bounds.size() == 6)
            opts.mBounds = std::array<double, 6>{
                {bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]}};
        opts.mPadding = pipe_number(rStep, "Padding", 0.0);
        opts.mPaddingRelative = pipe_number(rStep, "PaddingRelative", 0.1);
        opts.mRootResolution = static_cast<std::int64_t>(pipe_number(rStep, "RootResolution", 8.0));
        opts.mMaxDepth = static_cast<std::int64_t>(pipe_number(rStep, "MaxDepth", 4.0));
        opts.mBandCells = pipe_number(rStep, "BandCells", 1.0);
        opts.mRecordLevels = pipe_flag(rStep, "RecordLevels", true);
        opts.mMaxCells = static_cast<std::int64_t>(pipe_number(rStep, "MaxCells", 20000000.0));
        opts.mDistance.mSign = sdf_sign_from_name(pipe_text(rStep, "Sign", "pseudonormal"));
        opts.mDistance.mLocation = sdf_location_from_name(pipe_text(rStep, "Location", "corner"));
        opts.mDistance.mBand = pipe_number(rStep, "Band", 0.0);
        SdfResult r = compute_sdf(mesh, opts);
        pipe_push_step(rReport, rStep,
                       {{"MaxDepth", static_cast<double>(r.mMaxDepth)},
                        {"NumBanded", static_cast<double>(r.mNumBanded)}});
        return std::move(r.mMesh);
    }

    if (op == "Isosurface") {
        // The level set of a scalar point_data field -- slice's data-driven
        // sibling, a surface one topological dimension lower.
        IsosurfaceOptions opts;
        opts.mArrayName = pipe_text(rStep, "Array", "");
        opts.mIsovalues = pipe_dvec(rStep, "Isovalues");
        if (opts.mIsovalues.empty())
            opts.mIsovalues.push_back(pipe_number(rStep, "Isovalue", 0.0));
        const int comp = static_cast<int>(pipe_number(rStep, "Component", -1.0));
        if (comp >= 0)
            opts.mComponent = comp;
        opts.mRecordParentIds = pipe_flag(rStep, "RecordParentIds", false);
        Mesh contour = isosurface(mesh, opts);
        pipe_push_step(rReport, rStep,
                       {{"ContourCells", static_cast<double>(pipe_total_cells(contour))}});
        if (pipe_total_cells(contour) == 0)
            rReport.mWarnings.push_back(
                "the contour is empty; the isovalue lies outside the field's range");
        return contour;
    }
    if (op == "Transform")
        return pipe_apply_transform(std::move(mesh), rStep, rReport);
    if (op == "ConvertCells") {
        ConvertCellsOptions opts;
        opts.mMode = convert_cells_mode_from_name(pipe_text(rStep, "Mode", ""));
        opts.mRecordParentIds = pipe_flag(rStep, "RecordParentIds", false);
        auto result = convert_cells(mesh, opts);
        pipe_push_step(rReport, rStep);
        return std::move(result.mMesh);
    }
    if (op == "Subdivide") {
        SubdivideOptions opts;
        opts.mRecordParentIds = pipe_flag(rStep, "RecordParentIds", false);
        auto result = subdivide(mesh, opts);
        pipe_push_step(rReport, rStep);
        return std::move(result.mMesh);
    }
    if (op == "Agglomerate") {
        AgglomerateOptions opts;
        opts.mTargetGroupSize =
            static_cast<std::size_t>(pipe_number(rStep, "TargetGroupSize", 8.0));
        auto result = agglomerate(mesh, opts);
        pipe_push_step(rReport, rStep);
        return std::move(result.mMesh);
    }
    if (op == "Crop") {
        const CropMode mode =
            pipe_text(rStep, "Mode", "all") == "any" ? CropMode::Any : CropMode::All;
        const bool record = pipe_flag(rStep, "RecordIds", false);
        const bool has_bbox = pipe_find(rStep, "Bbox") != nullptr;
        const bool has_plane = pipe_find(rStep, "Point") || pipe_find(rStep, "Normal");
        const bool has_where = pipe_find(rStep, "Where") != nullptr;
        if (static_cast<int>(has_bbox) + static_cast<int>(has_plane) +
                static_cast<int>(has_where) !=
            1)
            throw std::invalid_argument(
                pipe_err(rStep, "give one of 'Bbox', 'Point'+'Normal' or 'Where'"));
        if (has_where && pipe_find(rStep, "Mode") != nullptr)
            throw std::invalid_argument(
                pipe_err(rStep,
                         "'Mode' applies to 'Bbox' and 'Point'+'Normal', which test "
                         "points; a 'Where' predicate is already one value per cell "
                         "and has nothing to reduce"));
        CropResult result = [&] {
            if (has_where)
                return crop_predicate(mesh, pipe_text(rStep, "Where", ""),
                                      refine_compare_from_name(pipe_text(rStep, "Compare", "<")),
                                      pipe_number(rStep, "Value", 0.0), record);
            if (has_bbox) {
                std::vector<double> b = pipe_dvec(rStep, "Bbox");
                if (b.size() != 6)
                    throw std::invalid_argument(
                        pipe_err(rStep, "'Bbox' must be [xmin, ymin, zmin, xmax, ymax, zmax]"));
                const double lo[3] = {b[0], b[1], b[2]};
                const double hi[3] = {b[3], b[4], b[5]};
                return crop_bbox(mesh, lo, hi, mode, record);
            }
            double point[3];
            double normal[3];
            pipe_vec3(rStep, "Point", point);
            pipe_vec3(rStep, "Normal", normal);
            return crop_halfspace(mesh, point, normal, mode, record);
        }();
        pipe_push_step(rReport, rStep,
                       {{"CellsKept", static_cast<double>(pipe_total_cells(result.mMesh))}});
        return std::move(result.mMesh);
    }
    if (op == "ExtractSurface") {
        Mesh out = extract_surface(mesh, pipe_flag(rStep, "RecordParentIds", false));
        pipe_push_step(rReport, rStep);
        return out;
    }
    if (op == "ExtractSkin") {
        Mesh out = extract_skin(mesh, pipe_flag(rStep, "Linearize", false));
        pipe_push_step(rReport, rStep);
        return out;
    }
    if (op == "Reorder") {
        auto result = reorder(mesh, reorder_method_from_name(pipe_text(rStep, "Method", "")));
        pipe_push_step(rReport, rStep);
        return std::move(result.mMesh);
    }
    if (op == "DataDrop" || op == "DataKeep" || op == "DataRename")
        return pipe_apply_data_manage(std::move(mesh), rStep, rReport);
    if (op == "DataCalc") {
        const std::string spec = pipe_text(rStep, "Expr", "");
        // "NAME = EXPR" splits on the FIRST '=' (the CLI's documented rule):
        // the expression side may itself contain none, and the name side may
        // carry any character but '='.
        const std::size_t pos = spec.find('=');
        if (pos == std::string::npos)
            throw std::invalid_argument(pipe_err(rStep, "'Expr' must be 'NAME = EXPRESSION'"));
        DataCalcOptions opts;
        opts.output = pipe_trim(spec.substr(0, pos));
        opts.location = data_location_from_name(pipe_text(rStep, "Location", "point"));
        opts.overwrite = pipe_flag(rStep, "Overwrite", false);
        if (opts.output.empty())
            throw std::invalid_argument(pipe_err(rStep, "'Expr' has an empty output name"));
        Mesh out = data_calc(mesh, pipe_trim(spec.substr(pos + 1)), opts);
        pipe_push_step(rReport, rStep);
        return out;
    }
    if (op == "DataCondition") {
        DataConditionOptions opts;
        opts.mode = condition_mode_from_name(pipe_text(rStep, "Mode", ""));
        opts.location = data_location_from_name(pipe_text(rStep, "Location", "point"));
        opts.names = pipe_svec(rStep, "Names");
        opts.scope = condition_scope_from_name(pipe_text(rStep, "Scope", ""));
        opts.lo = pipe_number(rStep, "Lo", 0.0);
        opts.hi = pipe_number(rStep, "Hi", 1.0);
        opts.nan_policy = nan_policy_from_name(pipe_text(rStep, "NanPolicy", ""));
        opts.nan_replacement = pipe_number(rStep, "NanReplacement", 0.0);
        opts.suffix = pipe_text(rStep, "Suffix", "");
        Mesh out = data_condition(mesh, opts);
        pipe_push_step(rReport, rStep);
        return out;
    }
    if (op == "ToCell" || op == "ToPoint") {
        DataAverageOptions opts;
        opts.names = pipe_svec(rStep, "Names");
        Mesh out;
        if (op == "ToCell") {
            out = point_data_to_cell_data(mesh, opts);
        } else {
            opts.weight = cell_point_weight_from_name(pipe_text(rStep, "Weight", ""));
            out = cell_data_to_point_data(mesh, opts);
        }
        pipe_push_step(rReport, rStep);
        return out;
    }

    // validate_pipeline_step() rejects unknown ops with a better message; this
    // is the backstop for a caller that skipped validation.
    if (const char* hint = pipe_excluded_hint(op))
        throw std::invalid_argument("meshio++: pipeline: " + std::string(hint));
    throw std::invalid_argument("meshio++: pipeline: unknown operation '" + op + "'");
}

Mesh run_pipeline_steps(Mesh mesh, const std::vector<PipelineStep>& rSteps,
                        PipelineReport& rReport) {
    for (const PipelineStep& step : rSteps) {
        validate_pipeline_step(step);
        mesh = apply_pipeline_step(std::move(mesh), step, rReport);
    }
    return mesh;
}

PipelineReport run_pipeline(const Pipeline& rPipeline) {
    if (rPipeline.mVersion != 1)
        throw std::invalid_argument("meshio++: pipeline: unsupported Version " +
                                    std::to_string(rPipeline.mVersion) + " (this build knows 1)");
    if (rPipeline.mInput.mPath.empty())
        throw std::invalid_argument("meshio++: pipeline: Input.Path is required");
    if (rPipeline.mOutput.mPath.empty())
        throw std::invalid_argument("meshio++: pipeline: Output.Path is required");

    // Validate the whole chain before the (possibly expensive) read: a typo in
    // step 7 must not cost reading a 10 GB mesh first.
    for (const PipelineStep& step : rPipeline.mSteps)
        validate_pipeline_step(step);

    // Read: resolve_format with the sniff_format fallback, the read-path rule
    // everywhere (the CLI, mio_read, the wasm read_mesh).
    std::string rfmt;
    try {
        rfmt = resolve_format(rPipeline.mInput.mPath, rPipeline.mInput.mFormat);
    } catch (const ReadError&) {
        rfmt = sniff_format(rPipeline.mInput.mPath);
        if (rfmt.empty())
            throw;
    }
    if (!registry_readers().count(rfmt) && !registry_reader_supports_options(rfmt)) {
        std::string hint;
        if (const char* dep = registry_compiled_out(rfmt))
            hint = " (format '" + rfmt + "' is not available in this build; requires " +
                   std::string(dep) + ")";
        throw ReadError("meshio++: pipeline: no reader for format '" + rfmt + "'" + hint);
    }

    // A no-op unless the caller opened a provenance scope around this call
    // (see detail/provenance.hpp) -- the pipeline never opens one itself,
    // consistent with the record being opt-in everywhere.
    detail::provenance_set_source(rPipeline.mInput.mPath, rfmt);

    PipelineReport report;
    Mesh mesh = registry_read(rPipeline.mInput.mPath, rfmt, rPipeline.mInput.mOptions);
    mesh = run_pipeline_steps(std::move(mesh), rPipeline.mSteps, report);

    const std::string out_fmt = resolve_format(rPipeline.mOutput.mPath, rPipeline.mOutput.mFormat);
    const WriteOptions& wopts = rPipeline.mOutput.mOptions;
    const char* encoding_name = wopts.mEncoding == WriteEncoding::Ascii    ? "ascii"
                                : wopts.mEncoding == WriteEncoding::Binary ? "binary"
                                                                           : "";
    const char* codec_name = wopts.mCodecSet ? vtk_codec_name(wopts.mCodec) : "";
    detail::provenance_set_target(out_fmt, encoding_name, codec_name, wopts.mFloatFormat);

    // Write through the parameterized single owner: an Output option the
    // format cannot honour is an error, never silently ignored.
    registry_write_ex(rPipeline.mOutput.mPath, mesh, rPipeline.mOutput.mFormat,
                      rPipeline.mOutput.mOptions);
    return report;
}

WriteEncoding pipeline_encoding_from_name(const std::string& rName) {
    if (rName.empty() || rName == "default")
        return WriteEncoding::Default;
    if (rName == "ascii")
        return WriteEncoding::Ascii;
    if (rName == "binary")
        return WriteEncoding::Binary;
    throw std::invalid_argument(
        "meshio++: pipeline: Output.Encoding must be 'ascii' or "
        "'binary', not '" +
        rName + "'");
}

detail::VtkCodec pipeline_codec_from_name(const std::string& rName) {
    if (rName == "none")
        return detail::VtkCodec::None;
    if (rName == "zlib")
        return detail::VtkCodec::Zlib;
    if (rName == "lz4")
        return detail::VtkCodec::LZ4;
    if (rName == "zstd")
        return detail::VtkCodec::ZSTD;
    throw std::invalid_argument(
        "meshio++: pipeline: Output.Codec must be 'none', 'zlib', "
        "'lz4' or 'zstd', not '" +
        rName + "'");
}

MmapMode pipeline_mmap_from_name(const std::string& rName) {
    if (rName.empty() || rName == "auto")
        return MmapMode::Auto;
    if (rName == "on")
        return MmapMode::On;
    if (rName == "off")
        return MmapMode::Off;
    throw std::invalid_argument(
        "meshio++: pipeline: Input.Options.Mmap must be 'auto', "
        "'on' or 'off', not '" +
        rName + "'");
}

// --------------------------------------------------------------------------
// JSON front-end. Everything below is a thin, strict mapping from a
// settings.json document onto the typed model above; the vendored
// nlohmann/json never appears outside this section (or this file).
// --------------------------------------------------------------------------

bool pipeline_has_json() noexcept {
#ifdef MESHIOPLUSPLUS_HAS_JSON
    return true;
#else
    return false;
#endif
}

#ifndef MESHIOPLUSPLUS_HAS_JSON

namespace {
[[noreturn]] void pipe_no_json() {
    throw std::runtime_error(
        "meshio++: pipeline: this build has no JSON parser; rebuild with "
        "-DMESHIOPLUSPLUS_WITH_JSON=ON (after `git submodule update --init "
        "src/cpp/third_party/json`)");
}
}  // namespace

Pipeline parse_pipeline_json(const std::string&) {
    pipe_no_json();
}
Pipeline parse_pipeline_file(const std::string&) {
    pipe_no_json();
}
PipelineReport run_pipeline_json(const std::string&) {
    pipe_no_json();
}
PipelineReport run_pipeline_file(const std::string&) {
    pipe_no_json();
}

// The sequence document shares this parser, and therefore this guard: the
// typed sequence driver in operations/sequence.cpp compiles either way, but a
// settings document cannot be read without a JSON parser.
SequencePipeline parse_sequence_json(const std::string&) {
    pipe_no_json();
}
SequencePipeline parse_sequence_file(const std::string&) {
    pipe_no_json();
}
PipelineReport run_sequence_json(const std::string&) {
    pipe_no_json();
}
PipelineReport run_sequence_file(const std::string&) {
    pipe_no_json();
}

#else  // MESHIOPLUSPLUS_HAS_JSON

namespace {

using pipe_json = nlohmann::json;

[[noreturn]] void pipe_schema_error(const std::string& rWhat) {
    throw std::invalid_argument("meshio++: pipeline: " + rWhat);
}

/// Strict object-key check: every present key must be in `rAllowed`.
void pipe_check_keys(const pipe_json& rObject, const char* pWhere,
                     std::initializer_list<const char*> rAllowed) {
    for (const auto& item : rObject.items()) {
        const bool known = std::any_of(rAllowed.begin(), rAllowed.end(),
                                       [&](const char* k) { return item.key() == k; });
        if (!known) {
            std::string keys;
            for (const char* k : rAllowed) {
                if (!keys.empty())
                    keys += ", ";
                keys += k;
            }
            pipe_schema_error("unknown key '" + item.key() + "' in " + pWhere + " (known: " + keys +
                              ")");
        }
    }
}

const pipe_json* pipe_get(const pipe_json& rObject, const char* pKey) {
    auto it = rObject.find(pKey);
    return it == rObject.end() ? nullptr : &*it;
}

std::string pipe_get_string(const pipe_json& rObject, const char* pKey, const char* pWhere,
                            bool required = false) {
    const pipe_json* v = pipe_get(rObject, pKey);
    if (!v) {
        if (required)
            pipe_schema_error(std::string(pWhere) + "." + pKey + " is required");
        return "";
    }
    if (!v->is_string())
        pipe_schema_error(std::string(pWhere) + "." + pKey + " must be a string");
    return v->get<std::string>();
}

bool pipe_get_bool(const pipe_json& rObject, const char* pKey, const char* pWhere, bool fallback) {
    const pipe_json* v = pipe_get(rObject, pKey);
    if (!v)
        return fallback;
    if (!v->is_boolean())
        pipe_schema_error(std::string(pWhere) + "." + pKey + " must be a boolean");
    return v->get<bool>();
}

/// One step-parameter value. Arrays must be homogeneous; nested objects have
/// no place in the flat step vocabulary.
PipelineValue pipe_value_from_json(const pipe_json& rValue, const std::string& rContext) {
    if (rValue.is_boolean())
        return rValue.get<bool>();
    if (rValue.is_number_integer() || rValue.is_number_unsigned())
        return rValue.get<std::int64_t>();
    if (rValue.is_number_float())
        return rValue.get<double>();
    if (rValue.is_string())
        return rValue.get<std::string>();
    if (rValue.is_array()) {
        bool all_int = true, all_num = true, all_str = true;
        for (const auto& e : rValue) {
            all_int = all_int && (e.is_number_integer() || e.is_number_unsigned());
            all_num = all_num && e.is_number();
            all_str = all_str && e.is_string();
        }
        // An empty array carries no element type; the typed getters accept any
        // empty vector alternative as "empty of my type".
        if (rValue.empty())
            return std::vector<std::string>{};
        if (all_int)
            return rValue.get<std::vector<std::int64_t>>();
        if (all_num)
            return rValue.get<std::vector<double>>();
        if (all_str)
            return rValue.get<std::vector<std::string>>();
        pipe_schema_error(rContext + " must be a homogeneous array of numbers or strings");
    }
    pipe_schema_error(rContext +
                      " has an unsupported value type (objects and null are "
                      "not step parameters)");
}

/// A parsed document, plus whether it used any of the sequence-only keys.
///
/// There is ONE parser: it always builds a `SequencePipeline`, and
/// `parse_pipeline_json` projects that down to the single-file `Pipeline`,
/// refusing by name if a sequence key was present. Two parsers would drift on
/// the shared two thirds (Version/Operations/Output), and a
/// `parse_pipeline_json` that merely *ignored* `Mode` would silently truncate a
/// transient run to its first step -- exactly the failure this feature exists
/// to prevent.
struct PipeDocument {
    SequencePipeline mSeq;
    bool mSequenceKeys = false;
};

SequenceInput pipe_sequence_input_from_json(const pipe_json& rInput, bool& rSequenceKeys) {
    if (!rInput.is_object())
        pipe_schema_error("Input must be an object");
    pipe_check_keys(rInput, "Input",
                    {"Path", "Format", "Options", "Pattern", "Paths", "Times", "TimeFrom"});
    SequenceInput input;

    const pipe_json* path = pipe_get(rInput, "Path");
    const pipe_json* pattern = pipe_get(rInput, "Pattern");
    const pipe_json* paths = pipe_get(rInput, "Paths");
    const int given = (path ? 1 : 0) + (pattern ? 1 : 0) + (paths ? 1 : 0);
    if (given == 0)
        pipe_schema_error("Input.Path is required (or Input.Pattern / Input.Paths)");
    if (given > 1)
        pipe_schema_error(
            "Input names more than one source; set exactly one of Path, Pattern or Paths");
    if (path) {
        input.mPaths = {pipe_get_string(rInput, "Path", "Input", /*required=*/true)};
    } else if (pattern) {
        rSequenceKeys = true;
        input.mPattern = pipe_get_string(rInput, "Pattern", "Input", /*required=*/true);
        if (input.mPattern.empty())
            pipe_schema_error("Input.Pattern must not be empty");
    } else {
        rSequenceKeys = true;
        if (!paths->is_array() || paths->empty())
            pipe_schema_error("Input.Paths must be a non-empty array of strings");
        for (const auto& e : *paths) {
            if (!e.is_string())
                pipe_schema_error("Input.Paths must be a non-empty array of strings");
            input.mPaths.push_back(e.get<std::string>());
        }
        // An explicit list is a stated order, so it is not re-sorted; the
        // caller asked for these files in this order.
        input.mSortExplicit = false;
    }

    if (const pipe_json* times = pipe_get(rInput, "Times")) {
        rSequenceKeys = true;
        if (!times->is_array())
            pipe_schema_error("Input.Times must be an array of numbers");
        for (const auto& e : *times) {
            if (!e.is_number())
                pipe_schema_error("Input.Times must be an array of numbers");
            input.mTimes.push_back(e.get<double>());
        }
    }
    if (pipe_get(rInput, "TimeFrom")) {
        rSequenceKeys = true;
        input.mTimeFrom = sequence_time_from_name(pipe_get_string(rInput, "TimeFrom", "Input"));
    }

    input.mFormat = pipe_get_string(rInput, "Format", "Input");
    if (const pipe_json* opts = pipe_get(rInput, "Options")) {
        if (!opts->is_object())
            pipe_schema_error("Input.Options must be an object");
        pipe_check_keys(*opts, "Input.Options",
                        {"PointsOnly", "DataArrays", "TimeStep", "Lenient", "Mmap"});
        input.mOptions.mPointsOnly = pipe_get_bool(*opts, "PointsOnly", "Input.Options", false);
        input.mOptions.mLenient = pipe_get_bool(*opts, "Lenient", "Input.Options", false);
        if (const pipe_json* v = pipe_get(*opts, "TimeStep")) {
            if (!v->is_number_integer() && !v->is_number_unsigned())
                pipe_schema_error("Input.Options.TimeStep must be an integer");
            input.mOptions.mTimeStep = v->get<int>();
        }
        if (const pipe_json* v = pipe_get(*opts, "DataArrays")) {
            if (!v->is_array())
                pipe_schema_error("Input.Options.DataArrays must be an array of strings");
            std::vector<std::string> names;
            for (const auto& e : *v) {
                if (!e.is_string())
                    pipe_schema_error("Input.Options.DataArrays must be an array of strings");
                names.push_back(e.get<std::string>());
            }
            input.mOptions.mDataArrays = std::move(names);
        }
        input.mOptions.mMmap =
            pipeline_mmap_from_name(pipe_get_string(*opts, "Mmap", "Input.Options"));
    }
    return input;
}

SequenceOutput pipe_sequence_output_from_json(const pipe_json& rOutput) {
    if (!rOutput.is_object())
        pipe_schema_error("Output must be an object");
    pipe_check_keys(rOutput, "Output", {"Path", "Format", "Encoding", "Codec", "FloatFormat"});
    SequenceOutput output;
    output.mPath = pipe_get_string(rOutput, "Path", "Output", /*required=*/true);
    output.mFormat = pipe_get_string(rOutput, "Format", "Output");
    output.mOptions.mEncoding =
        pipeline_encoding_from_name(pipe_get_string(rOutput, "Encoding", "Output"));
    const std::string codec = pipe_get_string(rOutput, "Codec", "Output");
    if (!codec.empty()) {
        output.mOptions.mCodec = pipeline_codec_from_name(codec);
        output.mOptions.mCodecSet = true;
    }
    output.mOptions.mFloatFormat = pipe_get_string(rOutput, "FloatFormat", "Output");
    return output;
}

PipeDocument pipe_document_from_json(const std::string& rText) {
    pipe_json doc;
    try {
        doc = pipe_json::parse(rText);
    } catch (const pipe_json::parse_error& e) {
        throw std::invalid_argument(std::string("meshio++: pipeline: invalid JSON: ") + e.what());
    }
    if (!doc.is_object())
        pipe_schema_error("the settings document must be a JSON object");
    pipe_check_keys(doc, "the settings document",
                    {"Version", "Input", "Operations", "Output", "Mode", "Parallel", "Workers"});

    PipeDocument parsed;
    SequencePipeline& pipeline = parsed.mSeq;
    if (const pipe_json* v = pipe_get(doc, "Version")) {
        if (!v->is_number_integer() && !v->is_number_unsigned())
            pipe_schema_error("Version must be an integer");
        pipeline.mVersion = v->get<int>();
        if (pipeline.mVersion != 1)
            pipe_schema_error("unsupported Version " + std::to_string(pipeline.mVersion) +
                              " (this build knows 1)");
    }

    if (pipe_get(doc, "Mode")) {
        parsed.mSequenceKeys = true;
        pipeline.mMode =
            sequence_mode_from_name(pipe_get_string(doc, "Mode", "the settings document"));
    }
    if (pipe_get(doc, "Parallel")) {
        parsed.mSequenceKeys = true;
        pipeline.mParallel = pipe_get_bool(doc, "Parallel", "the settings document", false);
    }
    if (const pipe_json* v = pipe_get(doc, "Workers")) {
        parsed.mSequenceKeys = true;
        if (!v->is_number_integer() && !v->is_number_unsigned())
            pipe_schema_error("Workers must be an integer");
        pipeline.mWorkers = v->get<int>();
        if (pipeline.mWorkers < 0)
            pipe_schema_error("Workers must not be negative (0 means one per core)");
    }

    const pipe_json* input = pipe_get(doc, "Input");
    if (!input)
        pipe_schema_error("Input is required");
    pipeline.mInput = pipe_sequence_input_from_json(*input, parsed.mSequenceKeys);

    const pipe_json* output = pipe_get(doc, "Output");
    if (!output)
        pipe_schema_error("Output is required");
    pipeline.mOutput = pipe_sequence_output_from_json(*output);

    if (const pipe_json* ops = pipe_get(doc, "Operations")) {
        if (!ops->is_array())
            pipe_schema_error("Operations must be an array of steps");
        for (std::size_t i = 0; i < ops->size(); ++i) {
            const pipe_json& spec = (*ops)[i];
            const std::string where = "Operations[" + std::to_string(i) + "]";
            if (!spec.is_object())
                pipe_schema_error(where + " must be an object");
            PipelineStep step;
            step.mOp = pipe_get_string(spec, "Op", where.c_str(), /*required=*/true);
            for (const auto& item : spec.items()) {
                if (item.key() == "Op")
                    continue;
                step.mParams.emplace(item.key(),
                                     pipe_value_from_json(item.value(), where + "." + item.key()));
            }
            validate_pipeline_step(step);
            pipeline.mSteps.push_back(std::move(step));
        }
    }
    return parsed;
}

std::string pipe_read_file(const std::string& rPath) {
    std::ifstream in(rPath, std::ios::binary);
    if (!in)
        throw ReadError("meshio++: pipeline: cannot open settings file '" + rPath + "'");
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

/// Project a parsed document down to the single-file model, refusing by name
/// if it used a sequence key. Ignoring one instead would quietly run a
/// transient document as its first step.
Pipeline pipe_project_single(const PipeDocument& rParsed) {
    if (rParsed.mSequenceKeys)
        pipe_schema_error(
            "this document uses sequence keys (Mode / Input.Pattern / Input.Paths / "
            "Input.Times / Input.TimeFrom / Parallel / Workers); run it with "
            "run_sequence_file / run_sequence_json (the CLI's `pipeline` verb and Python's "
            "run_pipeline route there automatically)");
    Pipeline out;
    out.mVersion = rParsed.mSeq.mVersion;
    out.mInput.mPath = rParsed.mSeq.mInput.mPaths.empty() ? "" : rParsed.mSeq.mInput.mPaths[0];
    out.mInput.mFormat = rParsed.mSeq.mInput.mFormat;
    out.mInput.mOptions = rParsed.mSeq.mInput.mOptions;
    out.mSteps = rParsed.mSeq.mSteps;
    out.mOutput.mPath = rParsed.mSeq.mOutput.mPath;
    out.mOutput.mFormat = rParsed.mSeq.mOutput.mFormat;
    out.mOutput.mOptions = rParsed.mSeq.mOutput.mOptions;
    return out;
}

}  // namespace

Pipeline parse_pipeline_json(const std::string& rText) {
    return pipe_project_single(pipe_document_from_json(rText));
}

Pipeline parse_pipeline_file(const std::string& rPath) {
    return parse_pipeline_json(pipe_read_file(rPath));
}

PipelineReport run_pipeline_json(const std::string& rText) {
    return run_pipeline(parse_pipeline_json(rText));
}

PipelineReport run_pipeline_file(const std::string& rPath) {
    return run_pipeline(parse_pipeline_file(rPath));
}

SequencePipeline parse_sequence_json(const std::string& rText) {
    return pipe_document_from_json(rText).mSeq;
}

SequencePipeline parse_sequence_file(const std::string& rPath) {
    return parse_sequence_json(pipe_read_file(rPath));
}

PipelineReport run_sequence_json(const std::string& rText) {
    const PipeDocument parsed = pipe_document_from_json(rText);
    // A document using no sequence key, naming a plain output path AND a
    // single-step input IS a single-file run, and takes the physically
    // unchanged v9.11.0 path -- so every existing settings.json still produces
    // byte-identical output, right down to which function wrote it. A
    // multi-step input is routed to the driver even without a sequence key, so
    // it refuses by name rather than quietly writing step 0.
    if (!parsed.mSequenceKeys &&
        !sequence_input_needs_driver(parsed.mSeq.mInput, parsed.mSeq.mOutput))
        return run_pipeline(pipe_project_single(parsed));
    return run_sequence_pipeline(parsed.mSeq);
}

PipelineReport run_sequence_file(const std::string& rPath) {
    return run_sequence_json(pipe_read_file(rPath));
}

#endif  // MESHIOPLUSPLUS_HAS_JSON

}  // namespace meshioplusplus
