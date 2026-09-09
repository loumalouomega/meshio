/* Registration of the .Call entry points.
 *
 * R_useDynamicSymbols(FALSE) plus a full CallMethodDef table is what keeps
 * `R CMD check` quiet about symbol registration, and it means R resolves each
 * entry point once at load rather than by name on every call. */

#include "mio_r.h"

#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>

/* mesh.c */
extern SEXP R_mio_version(void);
extern SEXP R_mio_mesh_backend(void);
extern SEXP R_mio_last_error(void);
extern SEXP R_mio_format_readable(SEXP);
extern SEXP R_mio_format_writable(SEXP);
extern SEXP R_mio_sniff_format(SEXP);
extern SEXP R_mio_cell_type_num_nodes(SEXP);
extern SEXP R_mio_cell_type_dimension(SEXP);
extern SEXP R_mio_reader_supports_options(SEXP);
extern SEXP R_mio_mesh_create(void);
extern SEXP R_mio_mesh_release(SEXP);
extern SEXP R_mio_mesh_is_open(SEXP);
extern SEXP R_mio_read(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_write(SEXP, SEXP, SEXP);
extern SEXP R_mio_convert(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_read_metadata(SEXP, SEXP);
extern SEXP R_mio_num_points(SEXP);
extern SEXP R_mio_point_dim(SEXP);
extern SEXP R_mio_num_cell_blocks(SEXP);
extern SEXP R_mio_cell_block_info(SEXP, SEXP);
extern SEXP R_mio_cell_block_type(SEXP, SEXP);
extern SEXP R_mio_points(SEXP);
extern SEXP R_mio_connectivity(SEXP, SEXP);
extern SEXP R_mio_connectivity_raw(SEXP, SEXP);
extern SEXP R_mio_polygon_block(SEXP, SEXP);
extern SEXP R_mio_polyhedron_block(SEXP, SEXP);
extern SEXP R_mio_add_polygon_block(SEXP, SEXP, SEXP);
extern SEXP R_mio_add_polyhedron_block(SEXP, SEXP, SEXP);
extern SEXP R_mio_point_data_names(SEXP);
extern SEXP R_mio_cell_data_names(SEXP);
extern SEXP R_mio_field_data_names(SEXP);
extern SEXP R_mio_cell_data_num_blocks(SEXP, SEXP);
extern SEXP R_mio_point_data(SEXP, SEXP);
extern SEXP R_mio_cell_data(SEXP, SEXP, SEXP);
extern SEXP R_mio_field_data(SEXP, SEXP);
extern SEXP R_mio_set_points(SEXP, SEXP);
extern SEXP R_mio_add_cell_block(SEXP, SEXP, SEXP);
extern SEXP R_mio_add_point_data(SEXP, SEXP, SEXP);
extern SEXP R_mio_append_cell_data(SEXP, SEXP, SEXP);
extern SEXP R_mio_add_field_data(SEXP, SEXP, SEXP);
extern SEXP R_mio_regions(SEXP);
extern SEXP R_mio_add_region(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

/* ops.c */
extern SEXP R_mio_extract_surface(SEXP, SEXP);
extern SEXP R_mio_extract_skin(SEXP, SEXP);
extern SEXP R_mio_attach_quality(SEXP);
extern SEXP R_mio_quality_counts(SEXP);
extern SEXP R_mio_transform(SEXP, SEXP, SEXP);
extern SEXP R_mio_clean(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_smooth(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_optimize_volume(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_crop_bbox(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_crop_plane(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_crop_predicate(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_slice(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_isosurface(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_gradient(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_hessian(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_estimate_error(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_remesh(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_compute_curvature(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_repair(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_shrinkwrap(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_sobolev_deform(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_remesh_volume(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
                                SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_grid(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_voxelize(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_compute_sdf(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
                              SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_surface_watertight_check(SEXP);
extern SEXP R_mio_sample_distance(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_distance_to_surface(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_merge(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_interpolate(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_conservative_interpolate(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_undo_green(SEXP, SEXP);
extern SEXP R_mio_meshes_equal(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_diff(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_stats(SEXP);
extern SEXP R_mio_compute_bandwidth(SEXP);
extern SEXP R_mio_reorder(SEXP, SEXP);
extern SEXP R_mio_split(SEXP, SEXP, SEXP);
extern SEXP R_mio_convert_cells(SEXP, SEXP, SEXP);
extern SEXP R_mio_subdivide(SEXP, SEXP);
extern SEXP R_mio_agglomerate(SEXP, SEXP);
extern SEXP R_mio_refine(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_decimate(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_partition(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_partition_labels(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_data_drop(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_data_keep(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_data_rename(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_data_point_to_cell(SEXP, SEXP, SEXP);
extern SEXP R_mio_data_cell_to_point(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_data_calc(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_data_condition(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_data_info(SEXP);
extern SEXP R_mio_data_integrate(SEXP, SEXP);
extern SEXP R_mio_pipeline_run_file(SEXP);
extern SEXP R_mio_sequence_open(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_sequence_open_list(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_sequence_count(SEXP);
extern SEXP R_mio_sequence_path(SEXP, SEXP);
extern SEXP R_mio_sequence_step(SEXP, SEXP);
extern SEXP R_mio_sequence_time(SEXP, SEXP);
extern SEXP R_mio_sequence_time_source(SEXP, SEXP);
extern SEXP R_mio_sequence_read(SEXP, SEXP);
extern SEXP R_mio_sequence_free(SEXP);
extern SEXP R_mio_sequence_to_timeseries(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_timeseries_to_sequence(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_sequence_pipeline_run_file(SEXP);
extern SEXP R_mio_sequence_pipeline_run_json(SEXP);
extern SEXP R_mio_pipeline_run_json(SEXP);
extern SEXP R_mio_pipeline_has_json(void);

/* xdmf_series.c */
extern SEXP R_mio_xdmf_series_create(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_mio_xdmf_series_flush(SEXP);
extern SEXP R_mio_xdmf_series_finalized(SEXP);
extern SEXP R_mio_xdmf_series_write_points_cells(SEXP, SEXP);
extern SEXP R_mio_xdmf_series_write_data(SEXP, SEXP, SEXP);
extern SEXP R_mio_xdmf_series_finalize(SEXP);
extern SEXP R_mio_xdmf_series_num_steps(SEXP);
extern SEXP R_mio_xdmf_series_release(SEXP);
extern SEXP R_mio_xdmf_series_is_open(SEXP);
extern SEXP R_mio_provenance_begin(SEXP);
extern SEXP R_mio_provenance_end(void);
extern SEXP R_mio_provenance_note(SEXP, SEXP);
extern SEXP R_mio_provenance_set_source(SEXP, SEXP);
extern SEXP R_mio_provenance_set_target(SEXP, SEXP, SEXP, SEXP);

#define CALLDEF(name, n) {#name, (DL_FUNC)&name, n}

static const R_CallMethodDef CallEntries[] = {
    CALLDEF(R_mio_version, 0),
    CALLDEF(R_mio_mesh_backend, 0),
    CALLDEF(R_mio_last_error, 0),
    CALLDEF(R_mio_format_readable, 1),
    CALLDEF(R_mio_format_writable, 1),
    CALLDEF(R_mio_sniff_format, 1),
    CALLDEF(R_mio_cell_type_num_nodes, 1),
    CALLDEF(R_mio_cell_type_dimension, 1),
    CALLDEF(R_mio_reader_supports_options, 1),
    CALLDEF(R_mio_mesh_create, 0),
    CALLDEF(R_mio_mesh_release, 1),
    CALLDEF(R_mio_mesh_is_open, 1),
    CALLDEF(R_mio_read, 8),
    CALLDEF(R_mio_write, 3),
    CALLDEF(R_mio_convert, 4),
    CALLDEF(R_mio_read_metadata, 2),
    CALLDEF(R_mio_num_points, 1),
    CALLDEF(R_mio_point_dim, 1),
    CALLDEF(R_mio_num_cell_blocks, 1),
    CALLDEF(R_mio_cell_block_info, 2),
    CALLDEF(R_mio_cell_block_type, 2),
    CALLDEF(R_mio_points, 1),
    CALLDEF(R_mio_connectivity, 2),
    CALLDEF(R_mio_connectivity_raw, 2),
    CALLDEF(R_mio_polygon_block, 2),
    CALLDEF(R_mio_polyhedron_block, 2),
    CALLDEF(R_mio_add_polygon_block, 3),
    CALLDEF(R_mio_add_polyhedron_block, 3),
    CALLDEF(R_mio_point_data_names, 1),
    CALLDEF(R_mio_cell_data_names, 1),
    CALLDEF(R_mio_field_data_names, 1),
    CALLDEF(R_mio_cell_data_num_blocks, 2),
    CALLDEF(R_mio_point_data, 2),
    CALLDEF(R_mio_cell_data, 3),
    CALLDEF(R_mio_field_data, 2),
    CALLDEF(R_mio_set_points, 2),
    CALLDEF(R_mio_add_cell_block, 3),
    CALLDEF(R_mio_add_point_data, 3),
    CALLDEF(R_mio_append_cell_data, 3),
    CALLDEF(R_mio_add_field_data, 3),
    CALLDEF(R_mio_regions, 1),
    CALLDEF(R_mio_add_region, 6),
    CALLDEF(R_mio_extract_surface, 2),
    CALLDEF(R_mio_extract_skin, 2),
    CALLDEF(R_mio_attach_quality, 1),
    CALLDEF(R_mio_quality_counts, 1),
    CALLDEF(R_mio_transform, 3),
    CALLDEF(R_mio_clean, 6),
    CALLDEF(R_mio_smooth, 9),
    CALLDEF(R_mio_optimize_volume, 6),
    CALLDEF(R_mio_crop_bbox, 5),
    CALLDEF(R_mio_crop_plane, 5),
    CALLDEF(R_mio_crop_predicate, 5),
    CALLDEF(R_mio_slice, 4),
    CALLDEF(R_mio_isosurface, 5),
    CALLDEF(R_mio_gradient, 8),
    CALLDEF(R_mio_hessian, 6),
    CALLDEF(R_mio_estimate_error, 8),
    CALLDEF(R_mio_remesh, 11),
    CALLDEF(R_mio_compute_curvature, 8),
    CALLDEF(R_mio_repair, 8),
    CALLDEF(R_mio_shrinkwrap, 9),
    CALLDEF(R_mio_sobolev_deform, 8),
    CALLDEF(R_mio_remesh_volume, 14),
    CALLDEF(R_mio_grid, 4),
    CALLDEF(R_mio_voxelize, 11),
    CALLDEF(R_mio_compute_sdf, 16),
    CALLDEF(R_mio_surface_watertight_check, 1),
    CALLDEF(R_mio_sample_distance, 5),
    CALLDEF(R_mio_distance_to_surface, 7),
    CALLDEF(R_mio_merge, 6),
    CALLDEF(R_mio_interpolate, 7),
    CALLDEF(R_mio_conservative_interpolate, 5),
    CALLDEF(R_mio_undo_green, 2),
    CALLDEF(R_mio_meshes_equal, 5),
    CALLDEF(R_mio_diff, 5),
    CALLDEF(R_mio_stats, 1),
    CALLDEF(R_mio_compute_bandwidth, 1),
    CALLDEF(R_mio_reorder, 2),
    CALLDEF(R_mio_split, 3),
    CALLDEF(R_mio_convert_cells, 3),
    CALLDEF(R_mio_subdivide, 2),
    CALLDEF(R_mio_agglomerate, 2),
    CALLDEF(R_mio_refine, 11),
    CALLDEF(R_mio_decimate, 8),
    CALLDEF(R_mio_partition, 9),
    CALLDEF(R_mio_partition_labels, 8),
    CALLDEF(R_mio_data_drop, 4),
    CALLDEF(R_mio_data_keep, 4),
    CALLDEF(R_mio_data_rename, 4),
    CALLDEF(R_mio_data_point_to_cell, 3),
    CALLDEF(R_mio_data_cell_to_point, 4),
    CALLDEF(R_mio_data_calc, 5),
    CALLDEF(R_mio_data_condition, 10),
    CALLDEF(R_mio_data_info, 1),
    CALLDEF(R_mio_data_integrate, 2),
    CALLDEF(R_mio_pipeline_run_file, 1),
    CALLDEF(R_mio_sequence_open, 5),
    CALLDEF(R_mio_sequence_open_list, 5),
    CALLDEF(R_mio_sequence_count, 1),
    CALLDEF(R_mio_sequence_path, 2),
    CALLDEF(R_mio_sequence_step, 2),
    CALLDEF(R_mio_sequence_time, 2),
    CALLDEF(R_mio_sequence_time_source, 2),
    CALLDEF(R_mio_sequence_read, 2),
    CALLDEF(R_mio_sequence_free, 1),
    CALLDEF(R_mio_sequence_to_timeseries, 4),
    CALLDEF(R_mio_timeseries_to_sequence, 4),
    CALLDEF(R_mio_sequence_pipeline_run_file, 1),
    CALLDEF(R_mio_sequence_pipeline_run_json, 1),
    CALLDEF(R_mio_pipeline_run_json, 1),
    CALLDEF(R_mio_pipeline_has_json, 0),
    CALLDEF(R_mio_xdmf_series_create, 5),
    CALLDEF(R_mio_xdmf_series_flush, 1),
    CALLDEF(R_mio_xdmf_series_finalized, 1),
    CALLDEF(R_mio_xdmf_series_write_points_cells, 2),
    CALLDEF(R_mio_xdmf_series_write_data, 3),
    CALLDEF(R_mio_xdmf_series_finalize, 1),
    CALLDEF(R_mio_xdmf_series_num_steps, 1),
    CALLDEF(R_mio_xdmf_series_release, 1),
    CALLDEF(R_mio_xdmf_series_is_open, 1),
    CALLDEF(R_mio_provenance_begin, 1),
    CALLDEF(R_mio_provenance_end, 0),
    CALLDEF(R_mio_provenance_note, 2),
    CALLDEF(R_mio_provenance_set_source, 2),
    CALLDEF(R_mio_provenance_set_target, 4),
    {NULL, NULL, 0}};

void attribute_visible R_init_meshioplusplus(DllInfo *dll) {
    /* Rf_install() interns the symbol permanently, so caching it in a global
     * needs no further protection. It tags our external pointers, which is
     * what makes a foreign pointer an error rather than a crash. */
    mio_r_mesh_tag = Rf_install("mio_mesh_handle");
    mio_r_series_tag = Rf_install("mio_xdmf_series_handle");
    mio_r_sequence_tag = Rf_install("mio_sequence_handle");
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}
