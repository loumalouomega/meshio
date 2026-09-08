/* The operations layer: computations *on* a mesh, not file formats.
 *
 * Ownership rule for the opaque C results (split, partition, reorder, refine,
 * decimate, convert_cells): read the maps and counters first, then TAKE the
 * mesh(es) out of the result, then free it. Every mesh handed to R therefore
 * owns its handle and cannot dangle when the result goes away. This mirrors
 * the Julia binding exactly.
 */

#include "mio_r.h"

static SEXP quality_list(const mio_surface_quality *q); /* defined below; used earlier */

#include <string.h>

/* --- surface / skin / quality ------------------------------------------- */

SEXP R_mio_extract_surface(SEXP mesh, SEXP record_parent_ids) {
    mio_mesh *out = mio_extract_surface(mio_r_mesh(mesh),
                                        mio_r_bool(record_parent_ids, "record_parent_ids"));
    if (out == NULL) mio_r_fail("extract_surface");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_extract_skin(SEXP mesh, SEXP linearize) {
    mio_mesh *out = mio_extract_skin(mio_r_mesh(mesh), mio_r_bool(linearize, "linearize"));
    if (out == NULL) mio_r_fail("extract_skin");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_attach_quality(SEXP mesh) {
    mio_mesh *out = mio_attach_quality(mio_r_mesh(mesh));
    if (out == NULL) mio_r_fail("attach_quality");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_quality_counts(SEXP mesh) {
    int64_t nc = 0, ni = 0, nd = 0;
    mio_r_check(mio_quality_counts(mio_r_mesh(mesh), &nc, &ni, &nd), "quality_counts");
    SEXP a = PROTECT(Rf_ScalarReal((double)nc));
    SEXP b = PROTECT(Rf_ScalarReal((double)ni));
    SEXP c = PROTECT(Rf_ScalarReal((double)nd));
    const char *names[] = {"num_cells", "num_inverted", "num_degenerate"};
    SEXP values[] = {a, b, c};
    SEXP out = PROTECT(mio_r_named_list(3, names, values));
    UNPROTECT(4);
    return out;
}

/* --- geometry-preserving transforms ------------------------------------- */

SEXP R_mio_transform(SEXP mesh, SEXP matrix, SEXP rotate_vector_data) {
    SEXP m = PROTECT(Rf_coerceVector(matrix, REALSXP));
    if (Rf_xlength(m) != 16) {
        UNPROTECT(1);
        Rf_error("`matrix` must be a 4x4 affine matrix (16 values)");
    }
    /* The C API takes the matrix ROW-major, while an R matrix is column-major,
     * so this one is transposed on the way out. That is correct here and only
     * here: a 4x4 affine matrix is a mathematical object, not a mesh array. */
    double rowmajor[16];
    const double *src = REAL(m);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            rowmajor[i * 4 + j] = src[j * 4 + i];
        }
    }
    mio_mesh *out = mio_transform(mio_r_mesh(mesh), rowmajor,
                                  mio_r_bool(rotate_vector_data, "rotate_vector_data"));
    UNPROTECT(1);
    if (out == NULL) mio_r_fail("transform");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_clean(SEXP mesh, SEXP weld, SEXP atol, SEXP remove_orphans, SEXP drop_degenerate,
                 SEXP drop_duplicate_cells) {
    int64_t welded = 0, orphan = 0, degen = 0, dup = 0;
    mio_mesh *out = mio_clean(mio_r_mesh(mesh), mio_r_bool(weld, "weld"),
                              mio_r_double(atol, "atol"),
                              mio_r_bool(remove_orphans, "remove_orphans"),
                              mio_r_bool(drop_degenerate, "drop_degenerate"),
                              mio_r_bool(drop_duplicate_cells, "drop_duplicate_cells"),
                              &welded, &orphan, &degen, &dup);
    if (out == NULL) mio_r_fail("clean");
    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    SEXP a = PROTECT(Rf_ScalarReal((double)welded));
    SEXP b = PROTECT(Rf_ScalarReal((double)orphan));
    SEXP c = PROTECT(Rf_ScalarReal((double)degen));
    SEXP d = PROTECT(Rf_ScalarReal((double)dup));
    const char *names[] = {"mesh", "points_welded", "points_removed_orphan",
                           "cells_dropped_degenerate", "cells_dropped_duplicate"};
    SEXP values[] = {mo, a, b, c, d};
    SEXP res = PROTECT(mio_r_named_list(5, names, values));
    UNPROTECT(6);
    return res;
}

SEXP R_mio_smooth(SEXP mesh, SEXP method, SEXP iterations, SEXP lambda, SEXP mu,
                  SEXP fix_boundary, SEXP preserve_features, SEXP feature_angle,
                  SEXP guard_inversion) {
    int64_t moved = 0, skipped = 0;
    double disp = 0.0;
    mio_mesh *out =
        mio_smooth(mio_r_mesh(mesh), mio_r_opt_string(method),
                   mio_r_int(iterations, "iterations"), mio_r_double(lambda, "lambda"),
                   mio_r_double(mu, "mu"), mio_r_bool(fix_boundary, "fix_boundary"),
                   mio_r_bool(preserve_features, "preserve_features"),
                   mio_r_double(feature_angle, "feature_angle"),
                   mio_r_bool(guard_inversion, "guard_inversion"), &moved, &disp, &skipped);
    if (out == NULL) mio_r_fail("smooth");
    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    SEXP a = PROTECT(Rf_ScalarReal((double)moved));
    SEXP b = PROTECT(Rf_ScalarReal(disp));
    SEXP c = PROTECT(Rf_ScalarReal((double)skipped));
    const char *names[] = {"mesh", "num_nodes_moved", "max_displacement",
                           "num_skipped_inversion"};
    SEXP values[] = {mo, a, b, c};
    SEXP res = PROTECT(mio_r_named_list(4, names, values));
    UNPROTECT(5);
    return res;
}

SEXP R_mio_optimize_volume(SEXP mesh, SEXP max_iterations, SEXP relocate, SEXP flip,
                           SEXP preserve_boundary, SEXP min_improvement) {
    int64_t nflips = 0, n23 = 0, n32 = 0, nmoved = 0, ntets = 0;
    double qbefore = 0.0, qafter = 0.0;
    mio_mesh *out = mio_optimize_volume(
        mio_r_mesh(mesh), mio_r_int(max_iterations, "max_iterations"),
        mio_r_bool(relocate, "relocate"), mio_r_bool(flip, "flip"),
        mio_r_bool(preserve_boundary, "preserve_boundary"),
        mio_r_double(min_improvement, "min_improvement"), &nflips, &n23, &n32, &nmoved, &ntets,
        &qbefore, &qafter);
    if (out == NULL) mio_r_fail("optimize_volume");
    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    SEXP v0 = PROTECT(Rf_ScalarReal((double)nflips));
    SEXP v1 = PROTECT(Rf_ScalarReal((double)n23));
    SEXP v2 = PROTECT(Rf_ScalarReal((double)n32));
    SEXP v3 = PROTECT(Rf_ScalarReal((double)nmoved));
    SEXP v4 = PROTECT(Rf_ScalarReal((double)ntets));
    SEXP v5 = PROTECT(Rf_ScalarReal(qbefore));
    SEXP v6 = PROTECT(Rf_ScalarReal(qafter));
    const char *names[] = {"mesh",     "num_flips",          "num_23_flips",
                           "num_32_flips", "num_vertices_moved", "num_tets",
                           "min_quality_before", "min_quality_after"};
    SEXP values[] = {mo, v0, v1, v2, v3, v4, v5, v6};
    SEXP res = PROTECT(mio_r_named_list(8, names, values));
    UNPROTECT(9);
    return res;
}

/* --- subsetting --------------------------------------------------------- */

SEXP R_mio_crop_bbox(SEXP mesh, SEXP lo, SEXP hi, SEXP mode, SEXP record_ids) {
    double l[3], h[3];
    mio_r_vec3(lo, l, "lo");
    mio_r_vec3(hi, h, "hi");
    mio_mesh *out = mio_crop_bbox(mio_r_mesh(mesh), l, h, mio_r_int(mode, "mode"),
                                  mio_r_bool(record_ids, "record_ids"));
    if (out == NULL) mio_r_fail("crop_bbox");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_crop_predicate(SEXP mesh, SEXP array, SEXP compare, SEXP value,
                          SEXP record_ids) {
    mio_mesh *out = mio_crop_predicate(mio_r_mesh(mesh), mio_r_string(array, "array"),
                                       mio_r_int(compare, "compare"),
                                       mio_r_double(value, "value"),
                                       mio_r_bool(record_ids, "record_ids"));
    if (out == NULL) mio_r_fail("crop_predicate");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_crop_plane(SEXP mesh, SEXP point, SEXP normal, SEXP mode, SEXP record_ids) {
    double p[3], n[3];
    mio_r_vec3(point, p, "point");
    mio_r_vec3(normal, n, "normal");
    mio_mesh *out = mio_crop_plane(mio_r_mesh(mesh), p, n, mio_r_int(mode, "mode"),
                                   mio_r_bool(record_ids, "record_ids"));
    if (out == NULL) mio_r_fail("crop_plane");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_slice(SEXP mesh, SEXP origin, SEXP normal, SEXP record_parent_ids) {
    double o[3], n[3];
    mio_r_vec3(origin, o, "origin");
    mio_r_vec3(normal, n, "normal");
    mio_mesh *out = mio_slice(mio_r_mesh(mesh), o, n,
                              mio_r_bool(record_parent_ids, "record_parent_ids"));
    if (out == NULL) mio_r_fail("slice");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_isosurface(SEXP mesh, SEXP array, SEXP isovalues, SEXP component,
                      SEXP record_parent_ids) {
    SEXP v = PROTECT(Rf_coerceVector(isovalues, REALSXP));
    R_xlen_t n = Rf_xlength(v);
    if (n < 1) {
        UNPROTECT(1);
        Rf_error("at least one isovalue is required");
    }
    mio_mesh *out = mio_isosurface(mio_r_mesh(mesh), mio_r_string(array, "array"), REAL(v),
                                   (int)n, mio_r_int(component, "component"),
                                   mio_r_bool(record_parent_ids, "record_parent_ids"));
    UNPROTECT(1);
    if (out == NULL) mio_r_fail("isosurface");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_gradient(SEXP mesh, SEXP array, SEXP op, SEXP method, SEXP location, SEXP output,
                    SEXP component, SEXP overwrite) {
    int64_t skipped = 0, fallback = 0;
    mio_mesh *out = mio_gradient(mio_r_mesh(mesh), mio_r_string(array, "array"),
                                 mio_r_opt_string(op), mio_r_opt_string(method),
                                 mio_r_opt_string(location), mio_r_opt_string(output),
                                 mio_r_int(component, "component"),
                                 mio_r_bool(overwrite, "overwrite"), &skipped, &fallback);
    if (out == NULL) mio_r_fail("gradient");
    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    /* R has no native int64, so the counters come back as doubles (exact well
       past any plausible cell count). */
    SEXP a = PROTECT(Rf_ScalarReal((double)skipped));
    SEXP b = PROTECT(Rf_ScalarReal((double)fallback));
    const char *names[] = {"mesh", "num_skipped", "num_fallback"};
    SEXP values[] = {mo, a, b};
    SEXP res = PROTECT(mio_r_named_list(3, names, values));
    UNPROTECT(4);
    return res;
}

SEXP R_mio_hessian(SEXP mesh, SEXP array, SEXP method, SEXP location, SEXP output,
                   SEXP overwrite) {
    int64_t skipped = 0, fallback = 0;
    mio_mesh *out = mio_hessian(mio_r_mesh(mesh), mio_r_string(array, "array"),
                                mio_r_opt_string(method), mio_r_opt_string(location),
                                mio_r_opt_string(output), mio_r_bool(overwrite, "overwrite"),
                                &skipped, &fallback);
    if (out == NULL) mio_r_fail("hessian");
    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    /* R has no native int64, so the counters come back as doubles -- the
       mio_gradient wrapper's rule. */
    SEXP a = PROTECT(Rf_ScalarReal((double)skipped));
    SEXP b = PROTECT(Rf_ScalarReal((double)fallback));
    const char *names[] = {"mesh", "num_skipped", "num_fallback"};
    SEXP values[] = {mo, a, b};
    SEXP res = PROTECT(mio_r_named_list(3, names, values));
    UNPROTECT(4);
    return res;
}

SEXP R_mio_estimate_error(SEXP mesh, SEXP array, SEXP method, SEXP marking, SEXP marking_value,
                          SEXP output, SEXP marked, SEXP overwrite) {
    double global_error = 0.0;
    int64_t skipped = 0, nmarked = 0;
    mio_mesh *out = mio_estimate_error(
        mio_r_mesh(mesh), mio_r_string(array, "array"), mio_r_opt_string(method),
        mio_r_opt_string(marking), mio_r_double(marking_value, "marking_value"),
        mio_r_opt_string(output), mio_r_opt_string(marked),
        mio_r_bool(overwrite, "overwrite"), &global_error, &skipped, &nmarked);
    if (out == NULL) mio_r_fail("estimate_error");
    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    /* R has no native int64, so the counters come back as doubles (exact well
       past any plausible cell count) -- the mio_gradient wrapper's rule. */
    SEXP ge = PROTECT(Rf_ScalarReal(global_error));
    SEXP sk = PROTECT(Rf_ScalarReal((double)skipped));
    SEXP mk = PROTECT(Rf_ScalarReal((double)nmarked));
    const char *names[] = {"mesh", "global_error", "num_skipped", "num_marked"};
    SEXP values[] = {mo, ge, sk, mk};
    SEXP res = PROTECT(mio_r_named_list(4, names, values));
    UNPROTECT(5);
    return res;
}

/* Goes through mio_remesh_ex/mio_remesh_opts rather than the flat mio_remesh
   -- the R_mio_refine/mio_refine_ex precedent above, needed because
   mio_remesh is a flat C function with no room to grow (it already changed
   once). Unlike Julia/Fortran, R links the real C header, so mio_remesh_opts
   is used directly -- no hand-mirrored struct to keep in sync. */
SEXP R_mio_remesh(SEXP mesh, SEXP num_clusters, SEXP subdivide, SEXP subsample_ratio,
                  SEXP max_subdivide, SEXP max_iterations, SEXP max_repair_passes, SEXP metric,
                  SEXP gradation, SEXP preserve_boundary, SEXP max_anisotropy) {
    mio_remesh_opts opts;
    mio_remesh_report report;
    mio_mesh *out;

    mio_remesh_opts_init(&opts);
    opts.num_clusters = (int64_t)mio_r_int(num_clusters, "num_clusters");
    opts.subdivide = (int32_t)mio_r_int(subdivide, "subdivide");
    opts.max_subdivide = (int32_t)mio_r_int(max_subdivide, "max_subdivide");
    opts.subsample_ratio = mio_r_double(subsample_ratio, "subsample_ratio");
    opts.max_iterations = (int32_t)mio_r_int(max_iterations, "max_iterations");
    opts.max_repair_passes = (int32_t)mio_r_int(max_repair_passes, "max_repair_passes");
    opts.metric = mio_r_opt_string(metric);
    opts.gradation = mio_r_double(gradation, "gradation");
    opts.preserve_boundary = mio_r_bool(preserve_boundary, "preserve_boundary") ? 1 : 0;
    opts.max_anisotropy = mio_r_double(max_anisotropy, "max_anisotropy");

    out = mio_remesh_ex(mio_r_mesh(mesh), &opts, &report);
    if (out == NULL) mio_r_fail("remesh");
    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    /* R has no native int64, so the counters come back as doubles -- the
       mio_gradient wrapper's rule. */
    SEXP nc = PROTECT(Rf_ScalarReal((double)report.num_clusters));
    SEXP ni = PROTECT(Rf_ScalarReal((double)report.num_iterations));
    SEXP sa = PROTECT(Rf_ScalarReal((double)report.subdivide_applied));
    SEXP iso = PROTECT(Rf_ScalarReal((double)report.num_isolated_clusters));
    SEXP nm = PROTECT(Rf_ScalarReal((double)report.num_non_manifold_vertices));
    const char *names[] = {"mesh", "num_clusters", "num_iterations", "subdivide_applied",
                           "num_isolated_clusters", "num_non_manifold_vertices"};
    SEXP values[] = {mo, nc, ni, sa, iso, nm};
    SEXP res = PROTECT(mio_r_named_list(6, names, values));
    UNPROTECT(7);
    return res;
}

SEXP R_mio_compute_curvature(SEXP mesh, SEXP mean, SEXP gaussian, SEXP dual_area,
                             SEXP include_boundary, SEXP record_area, SEXP record_principal,
                             SEXP region) {
    mio_curvature_opts opts;
    mio_curvature_report report;
    mio_mesh *out;

    mio_curvature_opts_init(&opts);
    opts.mean = mio_r_bool(mean, "mean") ? 1 : 0;
    opts.gaussian = mio_r_bool(gaussian, "gaussian") ? 1 : 0;
    const char *da = mio_r_opt_string(dual_area);
    if (da != NULL && strcmp(da, "barycentric") == 0) {
        opts.dual_area = MIO_CURVATURE_BARYCENTRIC;
    } else if (da != NULL && strcmp(da, "mixed-voronoi") != 0) {
        Rf_error("meshio++: curvature: unknown dual area '%s' "
                 "(expected 'mixed-voronoi' or 'barycentric')", da);
    }
    opts.include_boundary = mio_r_bool(include_boundary, "include_boundary") ? 1 : 0;
    opts.record_area = mio_r_bool(record_area, "record_area") ? 1 : 0;
    opts.record_principal = mio_r_bool(record_principal, "record_principal") ? 1 : 0;
    opts.region = mio_r_opt_string(region);

    out = mio_compute_curvature(mio_r_mesh(mesh), &opts, &report);
    if (out == NULL) mio_r_fail("curvature");
    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    SEXP q = PROTECT(quality_list(&report.quality));
    /* R has no native int64, so the counters come back as doubles -- the
       mio_gradient wrapper's rule. */
    SEXP nb = PROTECT(Rf_ScalarReal((double)report.num_boundary));
    SEXP ni = PROTECT(Rf_ScalarReal((double)report.num_isolated));
    SEXP nd = PROTECT(Rf_ScalarReal((double)report.num_degenerate));
    SEXP tad = PROTECT(Rf_ScalarReal(report.total_angle_defect));
    const char *names[] = {"mesh", "quality", "num_boundary", "num_isolated",
                           "num_degenerate", "total_angle_defect"};
    SEXP values[] = {mo, q, nb, ni, nd, tad};
    SEXP res = PROTECT(mio_r_named_list(6, names, values));
    UNPROTECT(7);
    return res;
}

/* --- combining / comparing ---------------------------------------------- */

SEXP R_mio_merge(SEXP meshes, SEXP weld, SEXP atol, SEXP source_tag, SEXP data_policy,
                 SEXP drop_duplicate_cells) {
    if (TYPEOF(meshes) != VECSXP || Rf_length(meshes) < 1) {
        Rf_error("`meshes` must be a non-empty list of mio_mesh objects");
    }
    R_xlen_t n = Rf_length(meshes);
    SEXP buf = PROTECT(Rf_allocVector(RAWSXP, (R_xlen_t)(n * sizeof(mio_mesh *))));
    const mio_mesh **handles = (const mio_mesh **)RAW(buf);
    for (R_xlen_t i = 0; i < n; ++i) {
        handles[i] = mio_r_mesh(VECTOR_ELT(meshes, i));
    }
    mio_mesh *out = mio_merge(handles, (int64_t)n, mio_r_bool(weld, "weld"),
                              mio_r_double(atol, "atol"),
                              mio_r_bool(source_tag, "source_tag"),
                              mio_r_int(data_policy, "data_policy"),
                              mio_r_bool(drop_duplicate_cells, "drop_duplicate_cells"));
    UNPROTECT(1);
    if (out == NULL) mio_r_fail("merge");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_interpolate(SEXP source, SEXP target, SEXP method, SEXP arrays, SEXP extrapolate,
                       SEXP default_value, SEXP on_conflict) {
    SEXP shelter;
    int64_t count = 0;
    const char *const *names = mio_r_names(arrays, &count, &shelter);
    mio_mesh *out = mio_interpolate(mio_r_mesh(source), mio_r_mesh(target),
                                    mio_r_opt_string(method), names, count,
                                    mio_r_bool(extrapolate, "extrapolate"),
                                    mio_r_double(default_value, "default_value"),
                                    mio_r_opt_string(on_conflict));
    UNPROTECT(1); /* shelter */
    if (out == NULL) mio_r_fail("interpolate");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_conservative_interpolate(SEXP source, SEXP target, SEXP arrays, SEXP default_value,
                                    SEXP on_conflict) {
    SEXP shelter;
    int64_t count = 0;
    const char *const *names = mio_r_names(arrays, &count, &shelter);
    mio_mesh *out = mio_conservative_interpolate(mio_r_mesh(source), mio_r_mesh(target), names,
                                                 count, mio_r_double(default_value, "default_value"),
                                                 mio_r_opt_string(on_conflict));
    UNPROTECT(1); /* shelter */
    if (out == NULL) mio_r_fail("conservative_interpolate");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_undo_green(SEXP coarse, SEXP fine) {
    int64_t ngroups = 0, nremoved = 0;
    mio_mesh *out =
        mio_undo_green(mio_r_mesh(coarse), mio_r_mesh(fine), &ngroups, &nremoved);
    if (out == NULL) mio_r_fail("undo_green");
    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    SEXP a = PROTECT(Rf_ScalarReal((double)ngroups));
    SEXP b = PROTECT(Rf_ScalarReal((double)nremoved));
    const char *names[] = {"mesh", "num_groups_undone", "num_cells_removed"};
    SEXP values[] = {mo, a, b};
    SEXP res = PROTECT(mio_r_named_list(3, names, values));
    UNPROTECT(4);
    return res;
}

SEXP R_mio_meshes_equal(SEXP a, SEXP b, SEXP atol, SEXP rtol, SEXP unordered) {
    int equal = 0;
    mio_r_check(mio_meshes_equal(mio_r_mesh(a), mio_r_mesh(b), mio_r_double(atol, "atol"),
                                 mio_r_double(rtol, "rtol"),
                                 mio_r_bool(unordered, "unordered"), &equal),
                "meshes_equal");
    return Rf_ScalarLogical(equal != 0);
}

SEXP R_mio_diff(SEXP a, SEXP b, SEXP atol, SEXP rtol, SEXP unordered) {
    mio_diff_result *result = NULL;
    mio_r_check(mio_diff(mio_r_mesh(a), mio_r_mesh(b), mio_r_double(atol, "atol"),
                         mio_r_double(rtol, "rtol"), mio_r_bool(unordered, "unordered"),
                         &result),
                "diff");

    static const char *const kVerdicts[] = {"identical", "equal_within_tolerance",
                                            "different"};
    int v = (int)mio_diff_result_verdict(result);
    if (v < 0 || v > 2) v = 2;

    double mabs = 0.0, mrel = 0.0;
    int64_t worst = -1, nex = 0;
    if (mio_diff_result_point_summary(result, &mabs, &mrel, &worst, &nex) != MIO_OK) {
        mio_diff_result_free(result);
        mio_r_fail("diff point summary");
    }

    int64_t nb = mio_diff_result_num_block_diffs(result);
    if (nb < 0) nb = 0;
    SEXP tm = PROTECT(Rf_allocVector(LGLSXP, (R_xlen_t)nb));
    SEXP cm = PROTECT(Rf_allocVector(LGLSXP, (R_xlen_t)nb));
    SEXP cc = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)nb));
    for (int64_t i = 0; i < nb; ++i) {
        int t = 0, c = 0;
        int64_t k = 0;
        if (mio_diff_result_block(result, i, &t, &c, &k) != MIO_OK) {
            mio_diff_result_free(result);
            UNPROTECT(3);
            mio_r_fail("diff block");
        }
        LOGICAL(tm)[i] = t != 0;
        LOGICAL(cm)[i] = c != 0;
        REAL(cc)[i] = (double)k;
    }
    mio_diff_result_free(result);

    SEXP verdict = PROTECT(Rf_mkString(kVerdicts[v]));
    SEXP sa = PROTECT(Rf_ScalarReal(mabs));
    SEXP sr = PROTECT(Rf_ScalarReal(mrel));
    /* worst_index is a flat index into the coordinate buffer: 1-based here,
     * with the C API's -1 "no offender" becoming 0. */
    SEXP sw = PROTECT(Rf_ScalarReal(worst < 0 ? 0.0 : (double)(worst + 1)));
    SEXP sn = PROTECT(Rf_ScalarReal((double)nex));

    const char *pnames[] = {"max_abs_error", "max_rel_error", "worst_index",
                            "num_exceeding"};
    SEXP pvalues[] = {sa, sr, sw, sn};
    SEXP pts = PROTECT(mio_r_named_list(4, pnames, pvalues));

    const char *bnames[] = {"type_mismatch", "count_mismatch", "conn_mismatch_count"};
    SEXP bvalues[] = {tm, cm, cc};
    SEXP blocks = PROTECT(mio_r_named_list(3, bnames, bvalues));

    const char *names[] = {"verdict", "points", "blocks"};
    SEXP values[] = {verdict, pts, blocks};
    SEXP out = PROTECT(mio_r_named_list(3, names, values));
    UNPROTECT(11); /* tm cm cc verdict sa sr sw sn pts blocks out */
    return out;
}

/* --- statistics --------------------------------------------------------- */

SEXP R_mio_stats(SEXP mesh) {
    mio_stats_report r;
    memset(&r, 0, sizeof(r));
    mio_r_check(mio_stats(mio_r_mesh(mesh), &r), "stats");

    SEXP np = PROTECT(Rf_ScalarReal((double)r.num_points));
    SEXP nc = PROTECT(Rf_ScalarReal((double)r.num_cells));
    SEXP bmin = PROTECT(Rf_allocVector(REALSXP, 3));
    SEXP bmax = PROTECT(Rf_allocVector(REALSXP, 3));
    SEXP ext = PROTECT(Rf_allocVector(REALSXP, 3));
    SEXP cen = PROTECT(Rf_allocVector(REALSXP, 3));
    memcpy(REAL(bmin), r.bbox_min, 3 * sizeof(double));
    memcpy(REAL(bmax), r.bbox_max, 3 * sizeof(double));
    memcpy(REAL(ext), r.extent, 3 * sizeof(double));
    memcpy(REAL(cen), r.centroid, 3 * sizeof(double));
    SEXP ta = PROTECT(Rf_ScalarReal(r.total_area));
    SEXP sv = PROTECT(Rf_ScalarReal(r.signed_volume));
    SEXP uv = PROTECT(Rf_ScalarReal(r.unsigned_volume));
    SEXP ninv = PROTECT(Rf_ScalarReal((double)r.num_inverted));

    const char *names[] = {"num_points", "num_cells",       "bbox_min",        "bbox_max",
                           "extent",     "centroid",        "total_area",      "signed_volume",
                           "unsigned_volume", "num_inverted"};
    SEXP values[] = {np, nc, bmin, bmax, ext, cen, ta, sv, uv, ninv};
    SEXP out = PROTECT(mio_r_named_list(10, names, values));
    UNPROTECT(11);
    return out;
}

SEXP R_mio_compute_bandwidth(SEXP mesh) {
    int64_t bw = mio_compute_bandwidth(mio_r_mesh(mesh));
    if (bw < 0) mio_r_fail("compute_bandwidth");
    return Rf_ScalarReal((double)bw);
}

/* --- opaque-result operations ------------------------------------------- */

SEXP R_mio_reorder(SEXP mesh, SEXP method) {
    mio_reorder_result *r = mio_reorder(mio_r_mesh(mesh), mio_r_string(method, "method"));
    if (r == NULL) mio_r_fail("reorder");

    const void *data = NULL;
    mio_dtype dt;
    int64_t n = 0;
    if (mio_reorder_result_node_perm(r, &data, &dt, &n) != MIO_OK) {
        mio_reorder_result_free(r);
        mio_r_fail("reorder node_perm");
    }
    SEXP node_perm = PROTECT(mio_r_shift_map((const int64_t *)data, n));

    int64_t nperms = mio_reorder_result_num_cell_perms(r);
    if (nperms < 0) nperms = 0;
    SEXP cell_perms = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)nperms));
    for (int64_t b = 0; b < nperms; ++b) {
        const void *d = NULL;
        int64_t len = 0;
        if (mio_reorder_result_cell_perm(r, b, &d, &dt, &len) != MIO_OK) {
            mio_reorder_result_free(r);
            UNPROTECT(2);
            mio_r_fail("reorder cell_perm");
        }
        SEXP p = PROTECT(mio_r_shift_map((const int64_t *)d, len));
        SET_VECTOR_ELT(cell_perms, (R_xlen_t)b, p);
        UNPROTECT(1);
    }

    mio_mesh *out = mio_reorder_result_take_mesh(r);
    mio_reorder_result_free(r);
    if (out == NULL) {
        UNPROTECT(2);
        mio_r_fail("reorder take_mesh");
    }
    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    const char *names[] = {"mesh", "node_perm", "cell_perms"};
    SEXP values[] = {mo, node_perm, cell_perms};
    SEXP res = PROTECT(mio_r_named_list(3, names, values));
    UNPROTECT(4);
    return res;
}

typedef struct {
    const mio_split_result *result;
    int64_t index;
} split_ctx;

static int64_t split_key_getter(void *c, char *buf, int64_t buflen) {
    split_ctx *x = (split_ctx *)c;
    return mio_split_result_key(x->result, x->index, buf, buflen);
}

SEXP R_mio_split(SEXP mesh, SEXP by, SEXP tag_name) {
    mio_split_result *r = mio_split(mio_r_mesh(mesh), mio_r_string(by, "by"),
                                    mio_r_opt_string(tag_name));
    if (r == NULL) mio_r_fail("split");
    int64_t n = mio_split_result_count(r);
    if (n < 0) {
        mio_split_result_free(r);
        mio_r_fail("split count");
    }
    SEXP pieces = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)n));
    SEXP keys = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t)n));
    for (int64_t i = 0; i < n; ++i) {
        split_ctx ctx = {r, i};
        SEXP k = PROTECT(mio_r_getstring(split_key_getter, &ctx, "split key"));
        SET_STRING_ELT(keys, (R_xlen_t)i, STRING_ELT(k, 0));
        UNPROTECT(1);
        /* TAKE, never borrow: the piece must outlive the result handle. */
        mio_mesh *piece = mio_split_result_take_mesh(r, i);
        if (piece == NULL) {
            mio_split_result_free(r);
            UNPROTECT(2);
            mio_r_fail("split take_mesh");
        }
        SEXP p = PROTECT(mio_r_wrap_mesh(piece));
        SET_VECTOR_ELT(pieces, (R_xlen_t)i, p);
        UNPROTECT(1);
    }
    mio_split_result_free(r);
    Rf_setAttrib(pieces, R_NamesSymbol, keys);
    UNPROTECT(2);
    return pieces;
}

/* convert_cells / refine / decimate share a result shape: one mesh, a point
 * map, and one cell map per input block. `kind` selects which family of C
 * getters to use. */
static SEXP result_maps(void *r, int kind, SEXP *point_map, SEXP *cell_maps) {
    const void *data = NULL;
    mio_dtype dt;
    int64_t n = 0;
    mio_status st;
    if (kind == 0) {
        st = mio_convert_cells_result_point_map((mio_convert_cells_result *)r, &data, &dt, &n);
    } else if (kind == 1) {
        st = mio_refine_result_point_map((mio_refine_result *)r, &data, &dt, &n);
    } else {
        st = mio_decimate_result_point_map((mio_decimate_result *)r, &data, &dt, &n);
    }
    if (st != MIO_OK) return NULL;
    *point_map = PROTECT(mio_r_shift_map((const int64_t *)data, n));

    int64_t nmaps;
    if (kind == 0) {
        nmaps = mio_convert_cells_result_num_cell_maps((mio_convert_cells_result *)r);
    } else if (kind == 1) {
        nmaps = mio_refine_result_num_cell_maps((mio_refine_result *)r);
    } else {
        nmaps = mio_decimate_result_num_cell_maps((mio_decimate_result *)r);
    }
    if (nmaps < 0) nmaps = 0;
    *cell_maps = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)nmaps));
    for (int64_t b = 0; b < nmaps; ++b) {
        const void *d = NULL;
        int64_t len = 0;
        if (kind == 0) {
            st = mio_convert_cells_result_cell_map((mio_convert_cells_result *)r, b, &d, &dt,
                                                   &len);
        } else if (kind == 1) {
            st = mio_refine_result_cell_map((mio_refine_result *)r, b, &d, &dt, &len);
        } else {
            st = mio_decimate_result_cell_map((mio_decimate_result *)r, b, &d, &dt, &len);
        }
        if (st != MIO_OK) {
            UNPROTECT(2);
            return NULL;
        }
        SEXP p = PROTECT(mio_r_shift_map((const int64_t *)d, len));
        SET_VECTOR_ELT(*cell_maps, (R_xlen_t)b, p);
        UNPROTECT(1);
    }
    return *cell_maps; /* non-NULL marker; two PROTECTs are outstanding */
}

/* subdivide's cell maps only -- unlike convert_cells/refine/decimate, there
 * is no point map to read first (subdivide never prunes or renumbers a
 * point), so this does not share result_maps() above. Leaves one PROTECT
 * outstanding on success (the returned list), matching result_maps()'s own
 * convention of leaving its caller to UNPROTECT. */
static SEXP subdivide_cell_maps(mio_subdivide_result *r) {
    int64_t nmaps = mio_subdivide_result_num_cell_maps(r);
    if (nmaps < 0) nmaps = 0;
    SEXP cell_maps = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)nmaps));
    for (int64_t b = 0; b < nmaps; ++b) {
        const void *d = NULL;
        mio_dtype dt;
        int64_t len = 0;
        mio_status st = mio_subdivide_result_cell_map(r, b, &d, &dt, &len);
        if (st != MIO_OK) {
            UNPROTECT(1);
            return NULL;
        }
        SEXP p = PROTECT(mio_r_shift_map((const int64_t *)d, len));
        SET_VECTOR_ELT(cell_maps, (R_xlen_t)b, p);
        UNPROTECT(1);
    }
    return cell_maps; /* one PROTECT outstanding */
}

SEXP R_mio_subdivide(SEXP mesh, SEXP record_parent_ids) {
    mio_subdivide_result *r =
        mio_subdivide(mio_r_mesh(mesh), mio_r_bool(record_parent_ids, "record_parent_ids"));
    if (r == NULL) mio_r_fail("subdivide");
    SEXP cm = subdivide_cell_maps(r);
    if (cm == NULL) {
        mio_subdivide_result_free(r);
        mio_r_fail("subdivide maps");
    }
    mio_mesh *out = mio_subdivide_result_take_mesh(r);
    mio_subdivide_result_free(r);
    if (out == NULL) {
        UNPROTECT(1);
        mio_r_fail("subdivide take_mesh");
    }
    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    const char *names[] = {"mesh", "cell_maps"};
    SEXP values[] = {mo, cm};
    SEXP res = PROTECT(mio_r_named_list(2, names, values));
    UNPROTECT(3);
    return res;
}

/* agglomerate's cell map is a single FLAT array (unlike subdivide's
 * per-block cell_maps), so it needs no per-block loop at all -- just one
 * mio_r_shift_map() call. */
SEXP R_mio_agglomerate(SEXP mesh, SEXP target_group_size) {
    mio_agglomerate_result *r =
        mio_agglomerate(mio_r_mesh(mesh), mio_r_int64(target_group_size, "target_group_size"));
    if (r == NULL) mio_r_fail("agglomerate");
    const void *d = NULL;
    mio_dtype dt;
    int64_t len = 0;
    mio_status st = mio_agglomerate_result_cell_map(r, &d, &dt, &len);
    if (st != MIO_OK) {
        mio_agglomerate_result_free(r);
        mio_r_fail("agglomerate cell_map");
    }
    SEXP cm = PROTECT(mio_r_shift_map((const int64_t *)d, len));
    mio_mesh *out = mio_agglomerate_result_take_mesh(r);
    mio_agglomerate_result_free(r);
    if (out == NULL) {
        UNPROTECT(1);
        mio_r_fail("agglomerate take_mesh");
    }
    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    const char *names[] = {"mesh", "cell_map"};
    SEXP values[] = {mo, cm};
    SEXP res = PROTECT(mio_r_named_list(2, names, values));
    UNPROTECT(3);
    return res;
}

SEXP R_mio_convert_cells(SEXP mesh, SEXP mode, SEXP record_parent_ids) {
    mio_convert_cells_result *r =
        mio_convert_cells(mio_r_mesh(mesh), mio_r_string(mode, "mode"),
                          mio_r_bool(record_parent_ids, "record_parent_ids"));
    if (r == NULL) mio_r_fail("convert_cells");
    SEXP pm = R_NilValue, cm = R_NilValue;
    if (result_maps(r, 0, &pm, &cm) == NULL) {
        mio_convert_cells_result_free(r);
        mio_r_fail("convert_cells maps");
    }
    mio_mesh *out = mio_convert_cells_result_take_mesh(r);
    mio_convert_cells_result_free(r);
    if (out == NULL) {
        UNPROTECT(2);
        mio_r_fail("convert_cells take_mesh");
    }
    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    const char *names[] = {"mesh", "point_map", "cell_maps"};
    SEXP values[] = {mo, pm, cm};
    SEXP res = PROTECT(mio_r_named_list(3, names, values));
    UNPROTECT(4);
    return res;
}

/* The mio_refine_closure / mio_refine_compare code for a name, or -1. */
static int refine_closure_code(const char *name) {
    if (name == NULL || name[0] == '\0') return MIO_REFINE_CLOSURE_REDGREEN;
    if (strcmp(name, "redgreen") == 0 || strcmp(name, "red-green") == 0 ||
        strcmp(name, "green") == 0)
        return MIO_REFINE_CLOSURE_REDGREEN;
    if (strcmp(name, "propagate") == 0 || strcmp(name, "red") == 0)
        return MIO_REFINE_CLOSURE_PROPAGATE;
    if (strcmp(name, "balanced") == 0 || strcmp(name, "2:1") == 0)
        return MIO_REFINE_CLOSURE_BALANCED;
    return -1;
}

static int refine_compare_code(const char *name) {
    if (name == NULL || name[0] == '\0') return MIO_REFINE_LT;
    if (strcmp(name, "<") == 0 || strcmp(name, "lt") == 0) return MIO_REFINE_LT;
    if (strcmp(name, "<=") == 0 || strcmp(name, "le") == 0) return MIO_REFINE_LE;
    if (strcmp(name, ">") == 0 || strcmp(name, "gt") == 0) return MIO_REFINE_GT;
    if (strcmp(name, ">=") == 0 || strcmp(name, "ge") == 0) return MIO_REFINE_GE;
    if (strcmp(name, "==") == 0 || strcmp(name, "=") == 0 || strcmp(name, "eq") == 0)
        return MIO_REFINE_EQ;
    if (strcmp(name, "!=") == 0 || strcmp(name, "ne") == 0) return MIO_REFINE_NE;
    return -1;
}

SEXP R_mio_refine(SEXP mesh, SEXP levels, SEXP record_parent_ids, SEXP cells, SEXP region,
                  SEXP where_array, SEXP where_op, SEXP where_value, SEXP closure,
                  SEXP record_levels, SEXP record_hierarchy) {
    mio_refine_opts opts;
    int64_t *ids = NULL;
    R_xlen_t n_ids = 0;
    mio_refine_result *r = NULL;

    mio_refine_opts_init(&opts);
    opts.levels = (int32_t)mio_r_int(levels, "levels");
    opts.record_parent_ids = mio_r_bool(record_parent_ids, "record_parent_ids");
    opts.record_levels = mio_r_bool(record_levels, "record_levels");
    opts.record_hierarchy = mio_r_bool(record_hierarchy, "record_hierarchy");
    opts.region = mio_r_opt_string(region);
    opts.predicate_array = mio_r_opt_string(where_array);
    if (where_value != R_NilValue && Rf_length(where_value) > 0)
        opts.predicate_value = Rf_asReal(where_value);
    opts.closure = refine_closure_code(mio_r_opt_string(closure));
    if (opts.closure < 0) Rf_error("meshio++: refine: unknown closure");
    opts.predicate_op = refine_compare_code(mio_r_opt_string(where_op));
    if (opts.predicate_op < 0) Rf_error("meshio++: refine: unknown comparison");

    if (cells != R_NilValue && Rf_length(cells) > 0) {
        /* R has no native int64, so cell indices arrive as double (exact to
         * 2^53) or integer; either way they are 1-based on this side. */
        n_ids = Rf_length(cells);
        ids = (int64_t *)R_alloc((size_t)n_ids, sizeof(int64_t));
        for (R_xlen_t i = 0; i < n_ids; ++i) {
            double v = (TYPEOF(cells) == REALSXP) ? REAL(cells)[i] : (double)INTEGER(cells)[i];
            ids[i] = (int64_t)v - 1;
        }
        opts.cells = ids;
        opts.num_cells = (int64_t)n_ids;
    }

    r = mio_refine_ex(mio_r_mesh(mesh), &opts);
    if (r == NULL) mio_r_fail("refine");
    SEXP pm = R_NilValue, cm = R_NilValue;
    if (result_maps(r, 1, &pm, &cm) == NULL) {
        mio_refine_result_free(r);
        mio_r_fail("refine maps");
    }
    mio_mesh *out = mio_refine_result_take_mesh(r);
    mio_refine_result_free(r);
    if (out == NULL) {
        UNPROTECT(2);
        mio_r_fail("refine take_mesh");
    }
    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    const char *names[] = {"mesh", "point_map", "cell_maps"};
    SEXP values[] = {mo, pm, cm};
    SEXP res = PROTECT(mio_r_named_list(3, names, values));
    UNPROTECT(4);
    return res;
}

SEXP R_mio_decimate(SEXP mesh, SEXP target_ratio, SEXP target_faces, SEXP max_error,
                    SEXP placement, SEXP preserve_boundary, SEXP preserve_features,
                    SEXP feature_angle) {
    mio_decimate_result *r =
        mio_decimate(mio_r_mesh(mesh), mio_r_double(target_ratio, "target_ratio"),
                     mio_r_int64(target_faces, "target_faces"),
                     mio_r_double(max_error, "max_error"), mio_r_opt_string(placement),
                     mio_r_bool(preserve_boundary, "preserve_boundary"),
                     mio_r_bool(preserve_features, "preserve_features"),
                     mio_r_double(feature_angle, "feature_angle"));
    if (r == NULL) mio_r_fail("decimate");
    SEXP pm = R_NilValue, cm = R_NilValue;
    if (result_maps(r, 2, &pm, &cm) == NULL) {
        mio_decimate_result_free(r);
        mio_r_fail("decimate maps");
    }
    SEXP fr = PROTECT(Rf_ScalarReal((double)mio_decimate_result_faces_removed(r)));
    SEXP pr = PROTECT(Rf_ScalarReal((double)mio_decimate_result_points_removed(r)));
    SEXP cr = PROTECT(Rf_ScalarReal((double)mio_decimate_result_collapses_rejected(r)));
    SEXP me = PROTECT(Rf_ScalarReal(mio_decimate_result_max_error_applied(r)));
    mio_mesh *out = mio_decimate_result_take_mesh(r);
    mio_decimate_result_free(r);
    if (out == NULL) {
        UNPROTECT(6);
        mio_r_fail("decimate take_mesh");
    }
    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    const char *names[] = {"mesh",           "point_map",          "cell_maps",
                           "faces_removed",  "points_removed",     "collapses_rejected",
                           "max_error_applied"};
    SEXP values[] = {mo, pm, cm, fr, pr, cr, me};
    SEXP res = PROTECT(mio_r_named_list(7, names, values));
    UNPROTECT(8);
    return res;
}

SEXP R_mio_partition(SEXP mesh, SEXP nparts, SEXP method, SEXP imbalance, SEXP mode,
                     SEXP seed, SEXP record_ids, SEXP ghost_layers, SEXP weights_key) {
    mio_partition_result *r =
        mio_partition(mio_r_mesh(mesh), mio_r_int(nparts, "nparts"),
                      mio_r_opt_string(method), mio_r_double(imbalance, "imbalance"),
                      mio_r_opt_string(mode), mio_r_int(seed, "seed"),
                      mio_r_bool(record_ids, "record_ids"),
                      mio_r_int(ghost_layers, "ghost_layers"), mio_r_opt_string(weights_key));
    if (r == NULL) mio_r_fail("partition");

    int64_t n = mio_partition_result_num_pieces(r);
    int64_t nmaps = mio_partition_result_num_cell_maps(r);
    if (n < 0) n = 0;
    if (nmaps < 0) nmaps = 0;

    SEXP out = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)n));
    for (int64_t i = 0; i < n; ++i) {
        const void *d = NULL;
        mio_dtype dt;
        int64_t len = 0;
        if (mio_partition_result_point_map(r, i, &d, &dt, &len) != MIO_OK) {
            mio_partition_result_free(r);
            UNPROTECT(1);
            mio_r_fail("partition point_map");
        }
        SEXP pm = PROTECT(mio_r_shift_map((const int64_t *)d, len));
        SEXP cms = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)nmaps));
        for (int64_t b = 0; b < nmaps; ++b) {
            const void *d2 = NULL;
            int64_t l2 = 0;
            if (mio_partition_result_cell_map(r, i, b, &d2, &dt, &l2) != MIO_OK) {
                mio_partition_result_free(r);
                UNPROTECT(2); /* pm, cms */
                mio_r_fail("partition cell_map");
            }
            SEXP p = PROTECT(mio_r_shift_map((const int64_t *)d2, l2));
            SET_VECTOR_ELT(cms, (R_xlen_t)b, p);
            UNPROTECT(1);
        }
        SEXP pid = PROTECT(Rf_ScalarInteger(mio_partition_result_part_id(r, i)));
        mio_mesh *piece = mio_partition_result_take_mesh(r, i);
        if (piece == NULL) {
            mio_partition_result_free(r);
            UNPROTECT(4);
            mio_r_fail("partition take_mesh");
        }
        SEXP mo = PROTECT(mio_r_wrap_mesh(piece));
        const char *names[] = {"part_id", "mesh", "point_map", "cell_maps"};
        SEXP values[] = {pid, mo, pm, cms};
        SEXP item = PROTECT(mio_r_named_list(4, names, values));
        SET_VECTOR_ELT(out, (R_xlen_t)i, item);
        UNPROTECT(5);
    }
    mio_partition_result_free(r);
    UNPROTECT(1);
    return out;
}

SEXP R_mio_partition_labels(SEXP mesh, SEXP nparts, SEXP method, SEXP imbalance, SEXP mode,
                            SEXP seed, SEXP weights_key, SEXP num_cells) {
    int64_t n = mio_r_int64(num_cells, "num_cells");
    SEXP buf = PROTECT(Rf_allocVector(RAWSXP, (R_xlen_t)((n > 0 ? n : 1) * sizeof(int64_t))));
    int64_t *labels = (int64_t *)RAW(buf);
    mio_status st = mio_partition_labels(
        mio_r_mesh(mesh), mio_r_int(nparts, "nparts"), mio_r_opt_string(method),
        mio_r_double(imbalance, "imbalance"), mio_r_opt_string(mode), mio_r_int(seed, "seed"),
        mio_r_opt_string(weights_key), labels, n);
    if (st != MIO_OK) {
        UNPROTECT(1);
        mio_r_fail("partition_labels");
    }
    SEXP out = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)n));
    /* Part IDs, not indices: deliberately NOT shifted to 1-based. */
    for (int64_t i = 0; i < n; ++i) REAL(out)[i] = (double)labels[i];
    UNPROTECT(2);
    return out;
}

/* --- data operations ---------------------------------------------------- */

SEXP R_mio_data_drop(SEXP mesh, SEXP location, SEXP names, SEXP ignore_missing) {
    SEXP shelter;
    int64_t count = 0;
    const char *const *nm = mio_r_names(names, &count, &shelter);
    mio_mesh *out = mio_data_drop(mio_r_mesh(mesh),
                                  (mio_data_location)mio_r_int(location, "location"), nm,
                                  count, mio_r_bool(ignore_missing, "ignore_missing"));
    UNPROTECT(1);
    if (out == NULL) mio_r_fail("data_drop");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_data_keep(SEXP mesh, SEXP location, SEXP names, SEXP ignore_missing) {
    SEXP shelter;
    int64_t count = 0;
    const char *const *nm = mio_r_names(names, &count, &shelter);
    mio_mesh *out = mio_data_keep(mio_r_mesh(mesh),
                                  (mio_data_location)mio_r_int(location, "location"), nm,
                                  count, mio_r_bool(ignore_missing, "ignore_missing"));
    UNPROTECT(1);
    if (out == NULL) mio_r_fail("data_keep");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_data_rename(SEXP mesh, SEXP location, SEXP from, SEXP to) {
    mio_mesh *out = mio_data_rename(mio_r_mesh(mesh),
                                    (mio_data_location)mio_r_int(location, "location"),
                                    mio_r_string(from, "from"), mio_r_string(to, "to"));
    if (out == NULL) mio_r_fail("data_rename");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_data_point_to_cell(SEXP mesh, SEXP names, SEXP suffix) {
    SEXP shelter;
    int64_t count = 0;
    const char *const *nm = mio_r_names(names, &count, &shelter);
    mio_mesh *out = mio_data_point_to_cell(mio_r_mesh(mesh), nm, count,
                                           mio_r_opt_string(suffix));
    UNPROTECT(1);
    if (out == NULL) mio_r_fail("data_point_to_cell");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_data_cell_to_point(SEXP mesh, SEXP names, SEXP weight, SEXP suffix) {
    SEXP shelter;
    int64_t count = 0;
    const char *const *nm = mio_r_names(names, &count, &shelter);
    mio_mesh *out = mio_data_cell_to_point(
        mio_r_mesh(mesh), nm, count, (mio_cell_point_weight)mio_r_int(weight, "weight"),
        mio_r_opt_string(suffix));
    UNPROTECT(1);
    if (out == NULL) mio_r_fail("data_cell_to_point");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_data_calc(SEXP mesh, SEXP expression, SEXP location, SEXP output_name,
                     SEXP overwrite) {
    mio_mesh *out = mio_data_calc(mio_r_mesh(mesh), mio_r_string(expression, "expression"),
                                  (mio_data_location)mio_r_int(location, "location"),
                                  mio_r_string(output_name, "output_name"),
                                  mio_r_bool(overwrite, "overwrite"));
    if (out == NULL) mio_r_fail("data_calc");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_data_condition(SEXP mesh, SEXP location, SEXP names, SEXP mode, SEXP lo, SEXP hi,
                          SEXP scope, SEXP nan_policy, SEXP nan_replacement, SEXP suffix) {
    SEXP shelter;
    int64_t count = 0;
    const char *const *nm = mio_r_names(names, &count, &shelter);
    mio_mesh *out = mio_data_condition(
        mio_r_mesh(mesh), (mio_data_location)mio_r_int(location, "location"), nm, count,
        (mio_condition_mode)mio_r_int(mode, "mode"), mio_r_double(lo, "lo"),
        mio_r_double(hi, "hi"), (mio_condition_scope)mio_r_int(scope, "scope"),
        (mio_nan_policy)mio_r_int(nan_policy, "nan_policy"),
        mio_r_double(nan_replacement, "nan_replacement"), mio_r_opt_string(suffix));
    UNPROTECT(1);
    if (out == NULL) mio_r_fail("data_condition");
    return mio_r_wrap_mesh(out);
}

typedef struct {
    const mio_data_info *info;
    int64_t index;
} info_ctx;

static int64_t info_name_getter(void *c, char *buf, int64_t buflen) {
    info_ctx *x = (info_ctx *)c;
    return mio_data_info_name(x->info, x->index, buf, buflen);
}

SEXP R_mio_data_info(SEXP mesh) {
    mio_data_info *info =
        (mio_data_info *)mio_r_check_ptr(mio_data_info_create(mio_r_mesh(mesh)), "data_info");
    int64_t n = mio_data_info_count(info);
    if (n < 0) {
        mio_data_info_free(info);
        mio_r_fail("data_info count");
    }
    static const char *const kLocations[] = {"point", "cell", "field"};

    SEXP out = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)n));
    for (int64_t i = 0; i < n; ++i) {
        info_ctx ctx = {info, i};
        SEXP name = PROTECT(mio_r_getstring(info_name_getter, &ctx, "data_info name"));
        mio_data_array_info e;
        memset(&e, 0, sizeof(e));
        if (mio_data_info_entry(info, i, &e) != MIO_OK) {
            mio_data_info_free(info);
            UNPROTECT(1); /* name */
            mio_r_fail("data_info entry");
        }
        SEXP loc = PROTECT(Rf_mkString(
            (e.location >= 0 && e.location <= 2) ? kLocations[e.location] : "unknown"));
        SEXP dtype = PROTECT(Rf_mkString(mio_r_dtype_name((mio_dtype)e.dtype)));
        SEXP nb = PROTECT(Rf_ScalarReal((double)e.num_blocks));
        SEXP ne = PROTECT(Rf_ScalarReal((double)e.num_entries));
        SEXP ncomp = PROTECT(Rf_ScalarReal((double)e.num_components));
        SEXP nv = PROTECT(Rf_ScalarReal((double)e.num_values));
        SEXP mn = PROTECT(Rf_ScalarReal(e.min));
        SEXP mx = PROTECT(Rf_ScalarReal(e.max));
        SEXP mean = PROTECT(Rf_ScalarReal(e.mean));
        SEXP nnan = PROTECT(Rf_ScalarReal((double)e.num_nan));
        SEXP ninf = PROTECT(Rf_ScalarReal((double)e.num_inf));
        SEXP nfin = PROTECT(Rf_ScalarReal((double)e.num_finite));
        SEXP incons = PROTECT(Rf_ScalarLogical(e.inconsistent_blocks != 0));

        int64_t nc = e.num_components > 0 ? e.num_components : 0;
        SEXP comps = PROTECT(Rf_allocMatrix(REALSXP, (int)nc, 3));
        for (int64_t c = 0; c < nc; ++c) {
            double lo = 0.0, hi = 0.0, mu = 0.0;
            if (mio_data_info_component(info, i, c, &lo, &hi, &mu) != MIO_OK) {
                mio_data_info_free(info);
                UNPROTECT(15); /* this entry's 15 values */
                mio_r_fail("data_info component");
            }
            REAL(comps)[c] = lo;
            REAL(comps)[nc + c] = hi;
            REAL(comps)[2 * nc + c] = mu;
        }
        SEXP cdimnames = PROTECT(Rf_allocVector(VECSXP, 2));
        SEXP ccols = PROTECT(Rf_allocVector(STRSXP, 3));
        SET_STRING_ELT(ccols, 0, Rf_mkChar("min"));
        SET_STRING_ELT(ccols, 1, Rf_mkChar("max"));
        SET_STRING_ELT(ccols, 2, Rf_mkChar("mean"));
        SET_VECTOR_ELT(cdimnames, 0, R_NilValue);
        SET_VECTOR_ELT(cdimnames, 1, ccols);
        Rf_setAttrib(comps, R_DimNamesSymbol, cdimnames);
        UNPROTECT(2); /* cdimnames, ccols */

        const char *names[] = {"name",        "location",   "dtype",     "num_blocks",
                               "num_entries", "num_components", "num_values", "min",
                               "max",         "mean",       "num_nan",   "num_inf",
                               "num_finite",  "inconsistent_blocks", "components"};
        SEXP values[] = {name, loc,  dtype, nb,   ne,     ncomp, nv, mn,
                         mx,   mean, nnan,  ninf, nfin,   incons, comps};
        SEXP item = PROTECT(mio_r_named_list(15, names, values));
        SET_VECTOR_ELT(out, (R_xlen_t)i, item);
        UNPROTECT(16); /* the 15 values + item */
    }
    mio_data_info_free(info);
    UNPROTECT(1);
    return out;
}

typedef struct {
    const mio_data_integrate *result;
    int64_t index;
} integrate_name_ctx;

typedef struct {
    const mio_data_integrate *result;
    int64_t index;
    int64_t region;
} integrate_region_name_ctx;

static int64_t integrate_name_getter(void *c, char *buf, int64_t buflen) {
    integrate_name_ctx *x = (integrate_name_ctx *)c;
    return mio_data_integrate_name(x->result, x->index, buf, buflen);
}

static int64_t integrate_region_name_getter(void *c, char *buf, int64_t buflen) {
    integrate_region_name_ctx *x = (integrate_region_name_ctx *)c;
    return mio_data_integrate_region_name(x->result, x->index, x->region, buf, buflen);
}

/* Build the (num_components x 4) total/mean/domain_measure/num_nan matrix for
 * one array's whole-mesh entry (region < 0) or one of its named Cell
 * regions (region >= 0). Frees `result` and raises via mio_r_fail (which
 * longjmps, unwinding the protection stack) on any C API failure -- the
 * caller never sees a NULL back. Returns unprotected, like mio_r_named_list;
 * the caller must PROTECT it immediately. */
static SEXP integrate_components_matrix(mio_data_integrate *result, int64_t index, int64_t region,
                                        int64_t num_components) {
    int64_t nc = num_components > 0 ? num_components : 0;
    SEXP m = PROTECT(Rf_allocMatrix(REALSXP, (int)nc, 4));
    for (int64_t c = 0; c < nc; ++c) {
        double total = 0.0, mean = 0.0, dm = 0.0;
        int64_t nan = 0;
        mio_status s = (region < 0) ? mio_data_integrate_component(result, index, c, &total,
                                                                    &mean, &dm, &nan)
                                    : mio_data_integrate_region_component(
                                          result, index, region, c, &total, &mean, &dm, &nan);
        if (s != MIO_OK) {
            mio_data_integrate_free(result);
            UNPROTECT(1); /* m */
            mio_r_fail("data_integrate component");
        }
        REAL(m)[c] = total;
        REAL(m)[nc + c] = mean;
        REAL(m)[2 * nc + c] = dm;
        REAL(m)[3 * nc + c] = (double)nan;
    }
    SEXP dimnames = PROTECT(Rf_allocVector(VECSXP, 2));
    SEXP cols = PROTECT(Rf_allocVector(STRSXP, 4));
    SET_STRING_ELT(cols, 0, Rf_mkChar("total"));
    SET_STRING_ELT(cols, 1, Rf_mkChar("mean"));
    SET_STRING_ELT(cols, 2, Rf_mkChar("domain_measure"));
    SET_STRING_ELT(cols, 3, Rf_mkChar("num_nan"));
    SET_VECTOR_ELT(dimnames, 0, R_NilValue);
    SET_VECTOR_ELT(dimnames, 1, cols);
    Rf_setAttrib(m, R_DimNamesSymbol, dimnames);
    UNPROTECT(3); /* dimnames, cols, m */
    return m;
}

/* Cell-measure-weighted total/mean of one or more cell_data arrays --
 * gradient's integration counterpart. `names` NULL means every cell_data
 * array. Reported for the whole mesh (`domain`) and independently for every
 * named Cell region (`regions`) -- a cell in two regions contributes fully
 * to both, one in none contributes to neither. A point_data-only name fails,
 * naming data_point_to_cell as the fix. See doc/field_integration.md. */
SEXP R_mio_data_integrate(SEXP mesh, SEXP names) {
    SEXP shelter;
    int64_t ncount = 0;
    const char *const *nm = mio_r_names(names, &ncount, &shelter);
    mio_data_integrate *result = (mio_data_integrate *)mio_r_check_ptr(
        mio_data_integrate_create(mio_r_mesh(mesh), nm, ncount), "data_integrate");
    UNPROTECT(1); /* shelter */

    int64_t n = mio_data_integrate_count(result);
    if (n < 0) {
        mio_data_integrate_free(result);
        mio_r_fail("data_integrate count");
    }

    SEXP out = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)n));
    for (int64_t i = 0; i < n; ++i) {
        integrate_name_ctx nctx = {result, i};
        SEXP name =
            PROTECT(mio_r_getstring(integrate_name_getter, &nctx, "data_integrate name"));

        mio_field_integral_info e;
        memset(&e, 0, sizeof(e));
        if (mio_data_integrate_entry(result, i, &e) != MIO_OK) {
            mio_data_integrate_free(result);
            UNPROTECT(2); /* name, out */
            mio_r_fail("data_integrate entry");
        }
        SEXP ncomp = PROTECT(Rf_ScalarReal((double)e.num_components));
        SEXP dcells = PROTECT(Rf_ScalarReal((double)e.num_cells));
        SEXP dskip = PROTECT(Rf_ScalarReal((double)e.num_skipped));
        SEXP dcomps = PROTECT(integrate_components_matrix(result, i, -1, e.num_components));

        const char *domain_names[] = {"num_cells", "num_skipped", "components"};
        SEXP domain_values[] = {dcells, dskip, dcomps};
        SEXP domain = PROTECT(mio_r_named_list(3, domain_names, domain_values));

        int64_t nregions = mio_data_integrate_region_count(result, i);
        if (nregions < 0) {
            mio_data_integrate_free(result);
            UNPROTECT(7); /* name, ncomp, dcells, dskip, dcomps, domain, out */
            mio_r_fail("data_integrate region count");
        }

        SEXP regions = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)nregions));
        for (int64_t r = 0; r < nregions; ++r) {
            integrate_region_name_ctx rctx = {result, i, r};
            SEXP rname = PROTECT(mio_r_getstring(integrate_region_name_getter, &rctx,
                                                 "data_integrate region name"));

            mio_field_integral_info re;
            memset(&re, 0, sizeof(re));
            if (mio_data_integrate_region_entry(result, i, r, &re) != MIO_OK) {
                mio_data_integrate_free(result);
                UNPROTECT(9); /* rname + the 7 above + regions */
                mio_r_fail("data_integrate region entry");
            }
            SEXP rcells = PROTECT(Rf_ScalarReal((double)re.num_cells));
            SEXP rskip = PROTECT(Rf_ScalarReal((double)re.num_skipped));
            SEXP rcomps = PROTECT(integrate_components_matrix(result, i, r, re.num_components));

            const char *region_names[] = {"name", "num_cells", "num_skipped", "components"};
            SEXP region_values[] = {rname, rcells, rskip, rcomps};
            SEXP region_item = PROTECT(mio_r_named_list(4, region_names, region_values));
            SET_VECTOR_ELT(regions, (R_xlen_t)r, region_item);
            UNPROTECT(5); /* rname, rcells, rskip, rcomps, region_item */
        }

        const char *array_names[] = {"name", "num_components", "domain", "regions"};
        SEXP array_values[] = {name, ncomp, domain, regions};
        SEXP item = PROTECT(mio_r_named_list(4, array_names, array_values));
        SET_VECTOR_ELT(out, (R_xlen_t)i, item);
        UNPROTECT(8); /* name, ncomp, dcells, dskip, dcomps, domain, regions, item */
    }
    mio_data_integrate_free(result);
    UNPROTECT(1); /* out */
    return out;
}

/* --- settings pipeline (v9.11.0) ---------------------------------------- */

SEXP R_mio_pipeline_run_file(SEXP settings_path) {
    mio_r_check(mio_pipeline_run_file(mio_r_string(settings_path, "settings_path")),
                "pipeline_run_file");
    return R_NilValue;
}

SEXP R_mio_pipeline_run_json(SEXP json_text) {
    mio_r_check(mio_pipeline_run_json(mio_r_string(json_text, "json_text")),
                "pipeline_run_json");
    return R_NilValue;
}

SEXP R_mio_pipeline_has_json(void) {
    return Rf_ScalarLogical(mio_pipeline_has_json() != 0);
}

/* --- regular grids and signed distance ---------------------------------- */

/* R has no int64, so the counts come back as doubles (exact well past any
 * plausible cell count) -- the convention every other counter here follows. */

static void fill_sdf_opts(mio_sdf_opts *o, SEXP sign, SEXP location, SEXP band,
                          SEXP record_inside, SEXP watertight_check) {
    mio_sdf_opts_init(o);
    o->sign = (int32_t)mio_r_int(sign, "sign");
    o->location = (int32_t)mio_r_int(location, "location");
    o->band = mio_r_double(band, "band");
    o->record_inside = mio_r_bool(record_inside, "record_inside");
    o->watertight_check = (int32_t)mio_r_int(watertight_check, "watertight_check");
}

static SEXP quality_list(const mio_surface_quality *q) {
    SEXP a = PROTECT(Rf_ScalarReal((double)q->boundary_edges));
    SEXP b = PROTECT(Rf_ScalarReal((double)q->non_manifold_edges));
    SEXP c = PROTECT(Rf_ScalarReal((double)q->inconsistent_pairs));
    SEXP d = PROTECT(Rf_ScalarReal((double)q->degenerate_triangles));
    SEXP w = PROTECT(Rf_ScalarLogical(q->watertight != 0));
    const char *names[] = {"boundary_edges", "non_manifold_edges", "inconsistent_pairs",
                           "degenerate_triangles", "watertight"};
    SEXP values[] = {a, b, c, d, w};
    SEXP out = PROTECT(mio_r_named_list(5, names, values));
    UNPROTECT(6);
    return out;
}

SEXP R_mio_grid(SEXP dims, SEXP origin, SEXP spacing, SEXP max_cells) {
    SEXP d = PROTECT(Rf_coerceVector(dims, REALSXP));
    if (Rf_xlength(d) != 3) {
        UNPROTECT(1);
        Rf_error("`dims` must have exactly 3 elements");
    }
    int64_t cdims[3];
    for (int i = 0; i < 3; ++i) cdims[i] = (int64_t)REAL(d)[i];
    UNPROTECT(1);

    double o[3], s[3];
    mio_r_vec3(origin, o, "origin");
    mio_r_vec3(spacing, s, "spacing");
    mio_mesh *out = mio_grid(cdims, o, s, mio_r_int64(max_cells, "max_cells"));
    if (out == NULL) mio_r_fail("grid");
    return mio_r_wrap_mesh(out);
}

SEXP R_mio_voxelize(SEXP mesh, SEXP resolution, SEXP cell_size, SEXP bounds, SEXP padding,
                    SEXP padding_relative, SEXP fill, SEXP sign, SEXP attach_occupancy,
                    SEXP max_cells, SEXP watertight_check) {
    mio_voxel_opts opts;
    mio_voxel_opts_init(&opts);

    /* The buffers the option pointers reference must outlive the call. */
    int64_t res[3];
    double bnd[6];
    if (Rf_xlength(resolution) > 0) {
        SEXP r = PROTECT(Rf_coerceVector(resolution, REALSXP));
        if (Rf_xlength(r) != 3) {
            UNPROTECT(1);
            Rf_error("`resolution` must have exactly 3 elements");
        }
        for (int i = 0; i < 3; ++i) res[i] = (int64_t)REAL(r)[i];
        UNPROTECT(1);
        opts.resolution = res;
    }
    if (Rf_xlength(bounds) > 0) {
        SEXP b = PROTECT(Rf_coerceVector(bounds, REALSXP));
        if (Rf_xlength(b) != 6) {
            UNPROTECT(1);
            Rf_error("`bounds` must have exactly 6 elements");
        }
        memcpy(bnd, REAL(b), 6 * sizeof(double));
        UNPROTECT(1);
        opts.bounds = bnd;
    }
    opts.cell_size = mio_r_double(cell_size, "cell_size");
    opts.padding = mio_r_double(padding, "padding");
    opts.padding_relative = mio_r_double(padding_relative, "padding_relative");
    opts.fill = (int32_t)mio_r_int(fill, "fill");
    opts.sign = (int32_t)mio_r_int(sign, "sign");
    opts.attach_occupancy = mio_r_bool(attach_occupancy, "attach_occupancy");
    opts.max_cells = mio_r_int64(max_cells, "max_cells");
    opts.watertight_check = (int32_t)mio_r_int(watertight_check, "watertight_check");

    int64_t cdims[3] = {0, 0, 0}, occupied = 0;
    double origin[3] = {0, 0, 0}, spacing[3] = {0, 0, 0};
    mio_mesh *out = mio_voxelize(mio_r_mesh(mesh), &opts, cdims, origin, spacing, &occupied);
    if (out == NULL) mio_r_fail("voxelize");

    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    SEXP sd = PROTECT(Rf_allocVector(REALSXP, 3));
    for (int i = 0; i < 3; ++i) REAL(sd)[i] = (double)cdims[i];
    SEXP so = PROTECT(Rf_allocVector(REALSXP, 3));
    memcpy(REAL(so), origin, 3 * sizeof(double));
    SEXP ss = PROTECT(Rf_allocVector(REALSXP, 3));
    memcpy(REAL(ss), spacing, 3 * sizeof(double));
    SEXP no = PROTECT(Rf_ScalarReal((double)occupied));
    const char *names[] = {"mesh", "dims", "origin", "spacing", "num_occupied"};
    SEXP values[] = {mo, sd, so, ss, no};
    SEXP res_list = PROTECT(mio_r_named_list(5, names, values));
    UNPROTECT(6);
    return res_list;
}

SEXP R_mio_compute_sdf(SEXP mesh, SEXP structure, SEXP resolution, SEXP cell_size,
                       SEXP bounds, SEXP padding, SEXP padding_relative,
                       SEXP root_resolution, SEXP max_depth, SEXP band_cells,
                       SEXP record_levels, SEXP max_cells, SEXP sign, SEXP location,
                       SEXP band, SEXP watertight_check) {
    mio_compute_sdf_opts opts;
    mio_compute_sdf_opts_init(&opts);

    /* The buffers the option pointers reference must outlive the call. */
    int64_t res[3];
    double bnd[6];
    if (Rf_xlength(resolution) > 0) {
        SEXP r = PROTECT(Rf_coerceVector(resolution, REALSXP));
        if (Rf_xlength(r) != 3) {
            UNPROTECT(1);
            Rf_error("`resolution` must have exactly 3 elements");
        }
        for (int i = 0; i < 3; ++i) res[i] = (int64_t)REAL(r)[i];
        UNPROTECT(1);
        opts.resolution = res;
    }
    if (Rf_xlength(bounds) > 0) {
        SEXP b = PROTECT(Rf_coerceVector(bounds, REALSXP));
        if (Rf_xlength(b) != 6) {
            UNPROTECT(1);
            Rf_error("`bounds` must have exactly 6 elements");
        }
        memcpy(bnd, REAL(b), 6 * sizeof(double));
        UNPROTECT(1);
        opts.bounds = bnd;
    }
    opts.structure = (int32_t)mio_r_int(structure, "structure");
    opts.cell_size = mio_r_double(cell_size, "cell_size");
    opts.padding = mio_r_double(padding, "padding");
    opts.padding_relative = mio_r_double(padding_relative, "padding_relative");
    opts.root_resolution = mio_r_int64(root_resolution, "root_resolution");
    opts.max_depth = mio_r_int64(max_depth, "max_depth");
    opts.band_cells = mio_r_double(band_cells, "band_cells");
    opts.record_levels = mio_r_bool(record_levels, "record_levels");
    opts.max_cells = mio_r_int64(max_cells, "max_cells");
    fill_sdf_opts(&opts.distance, sign, location, band, Rf_ScalarLogical(0),
                  watertight_check);

    int64_t cdims[3] = {0, 0, 0}, depth = 0, banded = 0;
    double origin[3] = {0, 0, 0}, spacing[3] = {0, 0, 0};
    mio_surface_quality q;
    memset(&q, 0, sizeof(q));
    mio_mesh *out = mio_compute_sdf(mio_r_mesh(mesh), &opts, cdims, origin, spacing, &depth,
                                    &banded, &q);
    if (out == NULL) mio_r_fail("compute_sdf");

    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    SEXP sd = PROTECT(Rf_allocVector(REALSXP, 3));
    for (int i = 0; i < 3; ++i) REAL(sd)[i] = (double)cdims[i];
    SEXP so = PROTECT(Rf_allocVector(REALSXP, 3));
    memcpy(REAL(so), origin, 3 * sizeof(double));
    SEXP ss = PROTECT(Rf_allocVector(REALSXP, 3));
    memcpy(REAL(ss), spacing, 3 * sizeof(double));
    SEXP dp = PROTECT(Rf_ScalarReal((double)depth));
    SEXP nb = PROTECT(Rf_ScalarReal((double)banded));
    SEXP ql = PROTECT(quality_list(&q));
    const char *names[] = {"mesh", "dims", "origin", "spacing", "max_depth", "num_banded",
                           "quality"};
    SEXP values[] = {mo, sd, so, ss, dp, nb, ql};
    SEXP res_list = PROTECT(mio_r_named_list(7, names, values));
    UNPROTECT(8);
    return res_list;
}

/* Retetrahedralize a volume mesh (or closed surface) by isosurface stuffing
   -- remesh's volumetric sibling. Same "R links the real C header, no
   hand-mirrored struct" note as R_mio_remesh above; `distance` is set
   directly rather than through fill_sdf_opts, whose location/band/
   record_inside fields remesh_volume never reads. */
SEXP R_mio_remesh_volume(SEXP mesh, SEXP resolution, SEXP cell_size, SEXP bounds,
                         SEXP padding, SEXP padding_relative, SEXP max_cells, SEXP max_tets,
                         SEXP warp_fraction, SEXP sign, SEXP watertight_check,
                         SEXP surface_region, SEXP grid_cell_size, SEXP max_winding_work) {
    mio_remesh_volume_opts opts;
    mio_remesh_volume_report report;
    mio_mesh *out;

    mio_remesh_volume_opts_init(&opts);

    /* The buffers the option pointers reference must outlive the call, the
       R_mio_compute_sdf idiom above. */
    int64_t res[3];
    double bnd[6];
    if (Rf_xlength(resolution) > 0) {
        SEXP r = PROTECT(Rf_coerceVector(resolution, REALSXP));
        if (Rf_xlength(r) != 3) {
            UNPROTECT(1);
            Rf_error("`resolution` must have exactly 3 elements");
        }
        for (int i = 0; i < 3; ++i) res[i] = (int64_t)REAL(r)[i];
        UNPROTECT(1);
        opts.resolution = res;
    }
    if (Rf_xlength(bounds) > 0) {
        SEXP b = PROTECT(Rf_coerceVector(bounds, REALSXP));
        if (Rf_xlength(b) != 6) {
            UNPROTECT(1);
            Rf_error("`bounds` must have exactly 6 elements");
        }
        memcpy(bnd, REAL(b), 6 * sizeof(double));
        UNPROTECT(1);
        opts.bounds = bnd;
    }
    opts.cell_size = mio_r_double(cell_size, "cell_size");
    opts.padding = mio_r_double(padding, "padding");
    opts.padding_relative = mio_r_double(padding_relative, "padding_relative");
    opts.max_cells = mio_r_int64(max_cells, "max_cells");
    opts.max_tets = mio_r_int64(max_tets, "max_tets");
    opts.warp_fraction = mio_r_double(warp_fraction, "warp_fraction");
    opts.distance.sign = (int32_t)mio_r_int(sign, "sign");
    opts.distance.watertight_check = (int32_t)mio_r_int(watertight_check, "watertight_check");
    opts.distance.surface_region = mio_r_opt_string(surface_region);
    opts.distance.grid_cell_size = mio_r_double(grid_cell_size, "grid_cell_size");
    opts.distance.max_winding_work = mio_r_double(max_winding_work, "max_winding_work");

    out = mio_remesh_volume_ex(mio_r_mesh(mesh), &opts, &report);
    if (out == NULL) mio_r_fail("remesh_volume");
    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    SEXP nt = PROTECT(Rf_ScalarReal((double)report.num_tets));
    SEXP nw = PROTECT(Rf_ScalarReal((double)report.num_vertices_warped));
    SEXP nr = PROTECT(Rf_ScalarReal((double)report.num_tets_rejected));
    SEXP nm = PROTECT(Rf_ScalarReal((double)report.num_non_manifold_edges));
    SEXP ql = PROTECT(quality_list(&report.input_quality));
    const char *names[] = {"mesh", "num_tets", "num_vertices_warped", "num_tets_rejected",
                           "num_non_manifold_edges", "input_quality"};
    SEXP values[] = {mo, nt, nw, nr, nm, ql};
    SEXP res_list = PROTECT(mio_r_named_list(6, names, values));
    UNPROTECT(7);
    return res_list;
}


SEXP R_mio_surface_watertight_check(SEXP mesh) {
    mio_surface_quality q;
    memset(&q, 0, sizeof(q));
    mio_r_check(mio_surface_watertight_check(mio_r_mesh(mesh), &q), "surface_watertight_check");
    return quality_list(&q);
}

SEXP R_mio_sample_distance(SEXP mesh, SEXP points, SEXP sign, SEXP band,
                           SEXP watertight_check) {
    if (!Rf_isMatrix(points)) Rf_error("`points` must be a (3 x n) matrix");
    SEXP p = PROTECT(Rf_coerceVector(points, REALSXP));
    SEXP dim = Rf_getAttrib(points, R_DimSymbol);
    if (INTEGER(dim)[0] != 3) {
        UNPROTECT(1);
        Rf_error("`points` must be a (3 x n) matrix");
    }
    int64_t n = INTEGER(dim)[1];

    mio_sdf_opts opts;
    fill_sdf_opts(&opts, sign, Rf_ScalarInteger(0), band, Rf_ScalarLogical(0),
                  watertight_check);

    SEXP out = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)n));
    /* Column-major R == row-major C, so the buffer goes across as it stands. */
    mio_status st = mio_sample_distance(mio_r_mesh(mesh), REAL(p), n, &opts, REAL(out));
    if (st != MIO_OK) {
        UNPROTECT(2);
        mio_r_fail("sample_distance");
    }
    UNPROTECT(2);
    return out;
}

SEXP R_mio_distance_to_surface(SEXP mesh, SEXP surface, SEXP sign, SEXP location, SEXP band,
                               SEXP record_inside, SEXP watertight_check) {
    mio_sdf_opts opts;
    fill_sdf_opts(&opts, sign, location, band, record_inside, watertight_check);

    int64_t banded = 0;
    mio_surface_quality q;
    memset(&q, 0, sizeof(q));
    mio_mesh *out =
        mio_distance_to_surface(mio_r_mesh(mesh), mio_r_mesh(surface), &opts, &banded, &q);
    if (out == NULL) mio_r_fail("distance_to_surface");

    SEXP mo = PROTECT(mio_r_wrap_mesh(out));
    SEXP nb = PROTECT(Rf_ScalarReal((double)banded));
    SEXP ql = PROTECT(quality_list(&q));
    const char *names[] = {"mesh", "num_banded", "quality"};
    SEXP values[] = {mo, nb, ql};
    SEXP res = PROTECT(mio_r_named_list(3, names, values));
    UNPROTECT(4);
    return res;
}

/* --- Provenance (see doc/provenance.md) --------------------------------
 *
 * Rides the flat C ABI. R has no RAII, so the begin/end pair is exposed as
 * two entry points and the R-side `mio_with_provenance()` wrapper pairs them
 * with `on.exit()` -- the same shape `mio_xdmf_series()` already uses to give
 * a C handle an R-shaped lifetime. */

SEXP R_mio_provenance_begin(SEXP mode) {
    if (mio_provenance_scope_begin(Rf_asInteger(mode)) != MIO_OK)
        mio_r_fail("provenance_scope_begin");
    return R_NilValue;
}

SEXP R_mio_provenance_end(void) {
    mio_provenance_scope_end();
    return R_NilValue;
}

SEXP R_mio_provenance_note(SEXP category, SEXP detail) {
    mio_provenance_note(mio_r_string(category, "category"), mio_r_string(detail, "detail"));
    return R_NilValue;
}

SEXP R_mio_provenance_set_source(SEXP path, SEXP format) {
    mio_provenance_set_source(mio_r_string(path, "path"), mio_r_string(format, "format"));
    return R_NilValue;
}

SEXP R_mio_provenance_set_target(SEXP format, SEXP encoding, SEXP codec, SEXP float_format) {
    mio_provenance_set_target(mio_r_string(format, "format"), mio_r_string(encoding, "encoding"),
                              mio_r_string(codec, "codec"),
                              mio_r_string(float_format, "float_format"));
    return R_NilValue;
}
