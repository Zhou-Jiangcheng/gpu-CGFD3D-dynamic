#ifndef SV_CURV_COL_EL_ISO_H
#define SV_CURV_COL_EL_ISO_H

#include "fd_t.h"
#include "mympi_t.h"
#include "gd_t.h"
#include "md_t.h"
#include "wav_t.h"
#include "bdry_t.h"
#include <cuda_runtime.h>

/*************************************************
 * function prototype
 *************************************************/

int
sv_curv_col_el_iso_onestage(
  float  *w_cur_d,
  float  *rhs_d, 
  wav_t  wav_d,
  gd_t   gd_d,
  fd_device_t fd_device_d,
  gd_metric_t  metric_d,
  md_t md_d,
  bdryfree_t bdryfree_d,
  bdrypml_t  bdrypml_d,
  // include different order/stentil
  fd_op_t *fdx_op,
  fd_op_t *fdy_op,
  fd_op_t *fdz_op,
  const int myid);

__global__ void
sv_curv_col_el_iso_rhs_inner_gpu(
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
    size_t siz_iy, size_t siz_iz,
    int * lfdx_shift, float * lfdx_coef,
    int * lfdy_shift, float * lfdy_coef,
    int * lfdz_shift, float * lfdz_coef,
    const int myid);

__global__ void
sv_curv_col_el_iso_rhs_timg_z2_gpu(
    float *  Txx, float *  Tyy, float *  Tzz,
    float *  Txz, float *  Tyz, float *  Txy,
    float * hVx , float * hVy , float * hVz ,
    float * xi_x, float * xi_y, float * xi_z,
    float * et_x, float * et_y, float * et_z,
    float * zt_x, float * zt_y, float * zt_z,
    float * jac3d, float * slw3d,
    int ni1, int ni, int nj1, int nj, int nk1, int nk2,
    size_t siz_iy, size_t siz_iz, 
    int fdx_len, int * fdx_indx, 
    int fdy_len, int * fdy_indx, 
    int fdz_len, int * fdz_indx, 
    int idir, int jdir, int kdir,
    const int myid);

__global__ void
sv_curv_col_el_iso_rhs_vlow_z2_gpu(
    float *  Vx , float *  Vy , float *  Vz ,
    float * hTxx, float * hTyy, float * hTzz,
    float * hTxz, float * hTyz, float * hTxy,
    float * xi_x, float * xi_y, float * xi_z,
    float * et_x, float * et_y, float * et_z,
    float * zt_x, float * zt_y, float * zt_z,
    float * lam3d, float * mu3d, float * slw3d,
    float * matVx2Vz, float * matVy2Vz,
    int ni1, int ni, int nj1, int nj, int nk1, int nk2,
    size_t siz_iy, size_t siz_iz,
    int idir, int jdir, int kdir,
    const int myid);

int
sv_curv_col_el_iso_rhs_cfspml(
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
    int nk2, size_t siz_iy, size_t siz_iz,
    int *lfdx_shift, float *lfdx_coef,
    int *lfdy_shift, float *lfdy_coef,
    int *lfdz_shift, float *lfdz_coef,
    bdrypml_t bdrypml, bdryfree_t bdryfree,
    const int myid);

__global__ void
sv_curv_col_el_iso_rhs_cfspml_gpu(
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
    int nk2, size_t siz_iy, size_t siz_iz,
    int *lfdx_shift, float *lfdx_coef,
    int *lfdy_shift, float *lfdy_coef,
    int *lfdz_shift, float *lfdz_coef,
    bdrypml_t bdrypml, bdryfree_t bdryfree,
    const int myid);

__global__ void
sv_curv_col_el_iso_dvh2dvz_gpu(gd_t    gd_d,
                               gd_metric_t metric_d,
                               md_t       md_d,
                               bdryfree_t      bdryfree_d);

#endif
