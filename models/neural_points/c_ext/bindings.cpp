#include "query_point.h"
#include "query_point_p.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("claim_occ",  claim_occ);
    m.def("map_coor2occ",  map_coor2occ);
    m.def("fill_occ2pnts",  fill_occ2pnts);
    m.def("mask_raypos",  mask_raypos);
    m.def("get_shadingloc",  get_shadingloc);
    m.def("query_along_ray",  query_neigh_along_ray_layered);
    m.def("ray_voxel_intersect", ray_voxel_intersect);
    m.def("seg_point_sample", seg_point_sample);
    m.def("seg_point_sample_nonlinear", seg_point_sample_nonlinear);
    // for query_point_p # not used.
    m.def("get_occ_vox", get_occ_vox);
    m.def("near_vox_full", near_vox_full);
    m.def("insert_vox_points", insert_vox_points);
    m.def("query_rand_along_ray", query_rand_along_ray);
    m.def("query_neigh_along_ray_layered_h", query_neigh_along_ray_layered_h);
    // m.def("query_rand_along_ray",  query_rand_along_ray);
}


