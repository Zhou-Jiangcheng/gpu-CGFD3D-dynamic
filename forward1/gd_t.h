#ifndef GD_CURV_H
#define GD_CURV_H

#include <mpi.h>

#include "constants.h"
#include "gd_info.h"

#define GD_TILE_NX 4
#define GD_TILE_NY 4
#define GD_TILE_NZ 4
/*************************************************
 * structure
 *************************************************/

typedef enum {

  GD_TYPE_CART = 1,
  GD_TYPE_VMAP = 2,
  GD_TYPE_CURV = 3

} gd_type_t;

//  grid coordinate for both cart, vmap and curv
//    to reduce duplicated functions
typedef struct {

  gd_type_t type;

  int n1, n2, n3, n4;
  int nx, ny, nz, ncmp;
  float *v4d; // allocated var

  //to avoid ref x3d at different funcs
  float *x3d; // pointer to var
  float *y3d;
  float *z3d;
  float dx;
  float dy;
  float dz;

  // min/max of this thread including ghost points
  float xmin, xmax;
  float ymin, ymax;
  float zmin, zmax;

  // min/max of this thread for points in physical region
  float xmin_phy, xmax_phy;
  float ymin_phy, ymax_phy;
  float zmin_phy, zmax_phy;

  // boundary of each cell for AABB algorithm
  float *cell_xmin;
  float *cell_xmax;
  float *cell_ymin;
  float *cell_ymax;
  float *cell_zmin;
  float *cell_zmax;
  // boundary of tiles by 4x4x4 partition for AABB algorithm
  int   tile_istart[GD_TILE_NX];
  int   tile_iend  [GD_TILE_NX];
  int   tile_jstart[GD_TILE_NY];
  int   tile_jend  [GD_TILE_NY];
  int   tile_kstart[GD_TILE_NZ];
  int   tile_kend  [GD_TILE_NZ];
  float tile_xmin[GD_TILE_NZ][GD_TILE_NY][GD_TILE_NX];
  float tile_xmax[GD_TILE_NZ][GD_TILE_NY][GD_TILE_NX];
  float tile_ymin[GD_TILE_NZ][GD_TILE_NY][GD_TILE_NX];
  float tile_ymax[GD_TILE_NZ][GD_TILE_NY][GD_TILE_NX];
  float tile_zmin[GD_TILE_NZ][GD_TILE_NY][GD_TILE_NX];
  float tile_zmax[GD_TILE_NZ][GD_TILE_NY][GD_TILE_NX];

  size_t siz_line;
  size_t siz_slice;
  size_t siz_volume;
  
  size_t siz_slice_yz; 
  size_t siz_slice_yz2; 

  size_t *cmp_pos;
  char  **cmp_name;
} gd_t;

//  for metric
typedef struct {
  int n1, n2, n3, n4;
  int nx, ny, nz, ncmp;
  float *v4d; // allocated var

  float *jac; // pointer to var
  float *xi_x;
  float *xi_y;
  float *xi_z;
  float *eta_x;
  float *eta_y;
  float *eta_z;
  float *zeta_x;
  float *zeta_y;
  float *zeta_z;

  size_t siz_line;
  size_t siz_slice;
  size_t siz_volume;

  size_t *cmp_pos;
  char  **cmp_name;
} gdcurv_metric_t;


/*************************************************
 * function prototype
 *************************************************/

void 
gd_curv_init(gdinfo_t *gdinfo, gd_t *gdcurv);

void 
gd_curv_metric_init(gdinfo_t        *gdinfo,
                    gdcurv_metric_t *metric);
void
gd_curv_metric_cal(gdinfo_t        *gdinfo,
                   gd_t        *gdcurv,
                   gdcurv_metric_t *metric);

void
gd_curv_exchange(gdinfo_t *gdinfo,
                 float *g3d,
                 int ncmp,
                 int *neighid,
                 MPI_Comm topocomm);


void
gd_curv_gen_fault(
  gd_t *gdcurv,
  gdinfo_t *gdinfo,
  int  num_of_x_points,
  float dh,
  char *in_grid_fault_nc);

void
nc_read_fault_geometry(
        float *fault_x, float *fault_y, float *fault_z, 
        char *in_grid_fault_nc, gdinfo_t *gdinfo);

void
gd_curv_metric_import(gdcurv_metric_t *metric, char *fname_coords, char *import_dir);

void
gd_curv_coord_import(gd_t *gdcurv, char *fname_coords, char *import_dir);

void
gd_curv_coord_export(
  gdinfo_t *gdinfo,
  gd_t *gdcurv,
  char *fname_coords,
  char *output_dir);


void
gd_curv_metric_export(gdinfo_t        *gdinfo,
                      gdcurv_metric_t *metric,
                      char *fname_coords,
                      char *output_dir);


void
gd_curv_set_minmax(gdinfo_t *gdinfo, gd_t *gdcurv);

int
gd_curv_coord_to_glob_indx(gdinfo_t *gdinfo,
                           gd_t *gdcurv,
                           float sx,
                           float sy,
                           float sz,
                           MPI_Comm comm,
                           int myid,
                           int   *ou_si, int *ou_sj, int *ou_sk,
                           float *ou_sx_inc, float *ou_sy_inc, float *ou_sz_inc);
__device__ int
gd_curv_coord_to_glob_indx_gpu(gdinfo_t *gdinfo,
                               gd_t *gdcurv,
                               float sx,
                               float sy,
                               float sz,
                               MPI_Comm comm,
                               int myid,
                               int *ou_si, int *ou_sj, int *ou_sk,
                               float *ou_sx_inc, float *ou_sy_inc, float *ou_sz_inc);


__host__ __device__ int
gd_curv_coord_to_local_indx(gdinfo_t *gdinfo,
                            gd_t *gd,
                            float sx, float sy, float sz,
                            int *si, int *sj, int *sk,
                            float *sx_inc, float *sy_inc, float *sz_inc);

__host__ __device__
int gd_curv_depth_to_axis(gdinfo_t *gdinfo,
                          gd_t  *gd,
                          float sx,
                          float sy,
                          float *sz,
                          MPI_Comm comm,
                          int myid);

__host__ __device__ int
gd_curv_coord2shift_sample(float sx, float sy, float sz, 
    int num_points,
    float *points_x, 
    float *points_y,
    float *points_z,
    int    nx_sample,
    int    ny_sample,
    int    nz_sample,
    float *si_shift, 
    float *sj_shift,
    float *sk_shift);

__host__ __device__ int
gd_curv_coord2index_sample(float sx, float sy, float sz, 
    int num_points,
    float *points_x, // x coord of all points
    float *points_y,
    float *points_z,
    float *points_i, // curv coord of all points
    float *points_j,
    float *points_k,
    int    nx_sample,
    int    ny_sample,
    int    nz_sample,
    float *si_curv, // interped curv coord
    float *sj_curv,
    float *sk_curv);

  int
gd_curv_coord2index_rdinterp(float sx, float sy, float sz, 
    int num_points,
    float *points_x, // x coord of all points
    float *points_y,
    float *points_z,
    float *points_i, // curv coord of all points
    float *points_j,
    float *points_k,
    float *si_curv, // interped curv coord
    float *sj_curv,
    float *sk_curv);

float
gd_coord_get_x(gd_t *gd, int i, int j, int k);

float
gd_coord_get_y(gd_t *gd, int i, int j, int k);

float
gd_coord_get_z(gd_t *gd, int i, int j, int k);

__host__ __device__
int isPointInHexahedron_c(float px,  float py,  float pz,
                        float *vx, float *vy, float *vz);

__host__ __device__
int point2face(float *hexa1d,float *point, float *p2f);

__host__ __device__
int face_normal(float (*hexa2d)[3], float *normal_unit);

int
gd_print(gd_t *gd);

#endif