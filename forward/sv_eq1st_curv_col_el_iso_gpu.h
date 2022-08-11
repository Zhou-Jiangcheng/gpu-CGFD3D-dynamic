#ifndef SV_EQ1ST_CURV_COL_EL_ISO_H
#define SV_EQ1ST_CURV_COL_EL_ISO_H

#include "fd_t.h"
#include "gd_info.h"
#include "mympi_t.h"
#include "gd_t.h"
#include "md_t.h"
#include "wav_t.h"
#include "bdry_free.h"
#include "bdry_pml.h"
#include "io_funcs.h"
#include <cuda_runtime.h>

/*************************************************
 * function prototype
 *************************************************/

void
sv_eq1st_curv_col_el_iso_onestage(
  float  *w_cur_d,
  float  *rhs_d, 
  wav_t  wav_d,
  gdinfo_t   gdinfo_d,
  gdcurv_metric_t  metric_d,
  md_t mdeliso_d,
  bdryfree_t bdryfree_d,
  bdrypml_t  bdrypml_d,
  // include different order/stentil
  fd_op_t *fdx_op,
  fd_op_t *fdy_op,
  fd_op_t *fdz_op,
  const int myid, const int verbose);

__global__ void
sv_eq1st_curv_col_el_iso_rhs_inner_gpu(
    float *  Vx , float *  Vy , float *  Vz ,
    float *  Txx, float *  Tyy, float *  Tzz,
    float *  Txz, float *  Tyz, float *  Txy,
    float * hVx , float * hVy , float * hVz ,
    float * hTxx, float * hTyy, float * hTzz,
    float * hTxz, float * hTyz, float * hTxy,
    float * xi_x, float * xi_y, float * xi_z,
    float * et_x, float * et_y, float * et_z,
    float * zt_x, float * zt_y, float * zt_z,
    float * lam3d, float * mu3d, float * slw3d,
    int ni1, int ni, int nj1, int nj, int nk1, int nk,
    size_t siz_line, size_t siz_slice,
    int idir, int jdir, int kdir,
    const int myid, const int verbose);

__global__ void
sv_eq1st_curv_col_el_iso_rhs_timg_z2_gpu(
    float *  Txx, float *  Tyy, float *  Tzz,
    float *  Txz, float *  Tyz, float *  Txy,
    float * hVx , float * hVy , float * hVz ,
    float * xi_x, float * xi_y, float * xi_z,
    float * et_x, float * et_y, float * et_z,
    float * zt_x, float * zt_y, float * zt_z,
    float * jac3d, float * slw3d,
    int ni1, int ni, int nj1, int nj, int nk1, int nk2,
    size_t siz_line, size_t siz_slice, 
    int idir, int jdir, int kdir,
    const int myid, const int verbose);

__global__ void
sv_eq1st_curv_col_el_iso_rhs_vlow_z2_gpu(
    float *  Vx , float *  Vy , float *  Vz ,
    float * hTxx, float * hTyy, float * hTzz,
    float * hTxz, float * hTyz, float * hTxy,
    float * xi_x, float * xi_y, float * xi_z,
    float * et_x, float * et_y, float * et_z,
    float * zt_x, float * zt_y, float * zt_z,
    float * lam3d, float * mu3d, float * slw3d,
    float * matVx2Vz, float * matVy2Vz,
    int ni1, int ni, int nj1, int nj, int nk1, int nk2,
    size_t siz_line, size_t siz_slice,
    int idir, int jdir, int kdir,
    const int myid, const int verbose);

void
sv_eq1st_curv_col_el_iso_rhs_cfspml(
    float *  Vx , float *  Vy , float *  Vz ,
    float *  Txx, float *  Tyy, float *  Tzz,
    float *  Txz, float *  Tyz, float *  Txy,
    float * hVx , float * hVy , float * hVz ,
    float * hTxx, float * hTyy, float * hTzz,
    float * hTxz, float * hTyz, float * hTxy,
    float * xi_x, float * xi_y, float * xi_z,
    float * et_x, float * et_y, float * et_z,
    float * zt_x, float * zt_y, float * zt_z,
    float * lam3d, float *  mu3d, float * slw3d,
    int nk2, size_t siz_line, size_t siz_slice,
    bdrypml_t bdrypml, bdryfree_t bdryfree,
    int idir, int jdir, int kdir,
    const int myid, const int verbose);

__global__ void
sv_eq1st_curv_col_el_iso_rhs_cfspml_gpu(
    int idim, int iside,
    float *  Vx , float *  Vy , float *  Vz ,
    float *  Txx, float *  Tyy, float *  Tzz,
    float *  Txz, float *  Tyz, float *  Txy,
    float * hVx , float * hVy , float * hVz ,
    float * hTxx, float * hTyy, float * hTzz,
    float * hTxz, float * hTyz, float * hTxy,
    float * xi_x, float * xi_y, float * xi_z,
    float * et_x, float * et_y, float * et_z,
    float * zt_x, float * zt_y, float * zt_z,
    float * lam3d, float *  mu3d, float * slw3d,
    int nk2, size_t siz_line, size_t siz_slice,
    bdrypml_t bdrypml, bdryfree_t bdryfree,
    int idir, int jdir, int kdir,
    const int myid, const int verbose);

__global__ void
sv_eq1st_curv_col_el_iso_dvh2dvz_gpu(gdinfo_t        gdinfo_d,
                                     gdcurv_metric_t metric_d,
                                     md_t       md_d,
                                     bdryfree_t      bdryfree_d,
                                     const int verbose);
#endif
