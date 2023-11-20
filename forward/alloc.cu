#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "alloc.h"
#include "cuda_common.h"

// only copy grid info
int init_gdinfo_device(gdcurv_t *gdcurv, gdcurv_t *gdcurv_d)
{
  memcpy(gdcurv_d,gdcurv,sizeof(gdcurv_t));

  return 0;
}

int init_gdcurv_device(gdcurv_t *gdcurv, gdcurv_t *gdcurv_d)
{
  size_t siz_icmp = gdcurv->siz_icmp;
  memcpy(gdcurv_d,gdcurv,sizeof(gdcurv_t));
  gdcurv_d->x3d = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gdcurv_d->y3d = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gdcurv_d->z3d = (float *) cuda_malloc(sizeof(float)*siz_icmp);

  gdcurv_d->cell_xmin = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gdcurv_d->cell_xmax = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gdcurv_d->cell_ymin = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gdcurv_d->cell_ymax = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gdcurv_d->cell_zmin = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gdcurv_d->cell_zmax = (float *) cuda_malloc(sizeof(float)*siz_icmp);

  gdcurv_d->tile_istart = (int *) cuda_malloc(sizeof(int)*GD_TILE_NX);
  gdcurv_d->tile_iend   = (int *) cuda_malloc(sizeof(int)*GD_TILE_NX);
  gdcurv_d->tile_jstart = (int *) cuda_malloc(sizeof(int)*GD_TILE_NY);
  gdcurv_d->tile_jend   = (int *) cuda_malloc(sizeof(int)*GD_TILE_NY);
  gdcurv_d->tile_kstart = (int *) cuda_malloc(sizeof(int)*GD_TILE_NZ);
  gdcurv_d->tile_kend   = (int *) cuda_malloc(sizeof(int)*GD_TILE_NZ);

  int size = GD_TILE_NX * GD_TILE_NY * GD_TILE_NZ;
  gdcurv_d->tile_xmin = (float *) cuda_malloc(sizeof(float)*size);
  gdcurv_d->tile_xmax = (float *) cuda_malloc(sizeof(float)*size);
  gdcurv_d->tile_ymin = (float *) cuda_malloc(sizeof(float)*size);
  gdcurv_d->tile_ymax = (float *) cuda_malloc(sizeof(float)*size);
  gdcurv_d->tile_zmin = (float *) cuda_malloc(sizeof(float)*size);
  gdcurv_d->tile_zmax = (float *) cuda_malloc(sizeof(float)*size);

  CUDACHECK(cudaMemcpy(gdcurv_d->x3d, gdcurv->x3d, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->y3d, gdcurv->y3d, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->z3d, gdcurv->z3d, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(gdcurv_d->cell_xmin, gdcurv->cell_xmin, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->cell_xmax, gdcurv->cell_xmax, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->cell_ymin, gdcurv->cell_ymin, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->cell_ymax, gdcurv->cell_ymax, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->cell_zmin, gdcurv->cell_zmin, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->cell_zmax, gdcurv->cell_zmax, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(gdcurv_d->tile_istart, gdcurv->tile_istart, sizeof(int)*GD_TILE_NX, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_iend, gdcurv->tile_iend, sizeof(int)*GD_TILE_NX, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_jstart, gdcurv->tile_jstart, sizeof(int)*GD_TILE_NY, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_jend, gdcurv->tile_jend, sizeof(int)*GD_TILE_NY, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_kstart, gdcurv->tile_kstart, sizeof(int)*GD_TILE_NZ, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_kend, gdcurv->tile_kend, sizeof(int)*GD_TILE_NZ, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(gdcurv_d->tile_xmin, gdcurv->tile_xmin, sizeof(float)*size, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_xmax, gdcurv->tile_xmax, sizeof(float)*size, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_ymin, gdcurv->tile_ymin, sizeof(float)*size, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_ymax, gdcurv->tile_ymax, sizeof(float)*size, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_zmin, gdcurv->tile_zmin, sizeof(float)*size, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_zmax, gdcurv->tile_zmax, sizeof(float)*size, cudaMemcpyHostToDevice));

  return 0;
}

int init_md_device(md_t *md, md_t *md_d)
{
  size_t siz_icmp = md->siz_icmp;

  memcpy(md_d,md,sizeof(md_t));
  if (md->medium_type == CONST_MEDIUM_ELASTIC_ISO)
  {
    md_d->rho    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->lambda = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->mu     = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    CUDACHECK(cudaMemcpy(md_d->rho,    md->rho,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->lambda, md->lambda, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->mu,     md->mu,     sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  }
  if (md->medium_type == CONST_MEDIUM_ELASTIC_VTI)
  {
    md_d->rho    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c11    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c33    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c55    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c66    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c13    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    CUDACHECK(cudaMemcpy(md_d->rho,    md->rho,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c11,    md->c11,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c33,    md->c33,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c55,    md->c55,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c66,    md->c66,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c13,    md->c13,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  }
  if (md->medium_type == CONST_MEDIUM_ELASTIC_ANISO)
  {
    md_d->rho    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c11    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c12    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c13    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c14    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c15    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c16    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c22    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c23    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c24    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c25    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c26    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c33    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c34    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c35    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c36    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c44    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c45    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c46    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c55    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c56    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c66    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    CUDACHECK(cudaMemcpy(md_d->rho,    md->rho,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c11,    md->c11,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c12,    md->c12,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c13,    md->c13,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c14,    md->c14,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c15,    md->c15,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c16,    md->c16,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c22,    md->c22,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c23,    md->c23,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c24,    md->c24,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c25,    md->c25,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c26,    md->c26,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c33,    md->c33,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c34,    md->c34,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c35,    md->c35,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c36,    md->c36,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c44,    md->c44,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c45,    md->c45,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c46,    md->c46,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c55,    md->c55,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c56,    md->c56,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c66,    md->c66,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  }

  return 0;
}

int init_fd_device(fd_t *fd, fd_device_t *fd_device_d)
{
  int max_len = fd->fdz_max_len; //=5 

  fd_device_d->fdx_coef_d    = (float *) cuda_malloc(sizeof(float)*max_len);
  fd_device_d->fdy_coef_d    = (float *) cuda_malloc(sizeof(float)*max_len);
  fd_device_d->fdz_coef_d    = (float *) cuda_malloc(sizeof(float)*max_len);

  fd_device_d->fdx_indx_d    = (int *) cuda_malloc(sizeof(int)*max_len);
  fd_device_d->fdy_indx_d    = (int *) cuda_malloc(sizeof(int)*max_len);
  fd_device_d->fdz_indx_d    = (int *) cuda_malloc(sizeof(int)*max_len);

  fd_device_d->fdx_shift_d    = (int *) cuda_malloc(sizeof(size_t)*max_len);
  fd_device_d->fdy_shift_d    = (int *) cuda_malloc(sizeof(size_t)*max_len);
  fd_device_d->fdz_shift_d    = (int *) cuda_malloc(sizeof(size_t)*max_len);

  return 0;
}

int init_metric_device(gd_metric_t *metric, gd_metric_t *metric_d)
{
  size_t siz_icmp = metric->siz_icmp;

  memcpy(metric_d,metric,sizeof(gd_metric_t));
  metric_d->jac     = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  metric_d->xi_x    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  metric_d->xi_y    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  metric_d->xi_z    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  metric_d->eta_x   = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  metric_d->eta_y   = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  metric_d->eta_z   = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  metric_d->zeta_x  = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  metric_d->zeta_y  = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  metric_d->zeta_z  = (float *) cuda_malloc(sizeof(float)*siz_icmp);

  CUDACHECK(cudaMemcpy(metric_d->jac,   metric->jac,     sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(metric_d->xi_x,  metric->xi_x,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(metric_d->xi_y,  metric->xi_y,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(metric_d->xi_z,  metric->xi_z,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(metric_d->eta_x, metric->eta_x,   sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(metric_d->eta_y, metric->eta_y,   sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(metric_d->eta_z, metric->eta_z,   sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(metric_d->zeta_x, metric->zeta_x, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(metric_d->zeta_y, metric->zeta_y, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(metric_d->zeta_z, metric->zeta_z, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  return 0;
}

int init_fault_coef_device(gdcurv_t *gdcurv, fault_coef_t *FC, fault_coef_t *FC_d)
{
  int ny = gdcurv->ny;
  int nz = gdcurv->nz;
  memcpy(FC_d,FC,sizeof(fault_coef_t));

  FC_d->rho_f = (float *) cuda_malloc(sizeof(float)*ny*nz*2);
  FC_d->mu_f  = (float *) cuda_malloc(sizeof(float)*ny*nz*2);
  FC_d->lam_f = (float *) cuda_malloc(sizeof(float)*ny*nz*2);

  FC_d->D21_1 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->D22_1 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->D23_1 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->D31_1 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->D32_1 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->D33_1 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);

  FC_d->D21_2 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->D22_2 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->D23_2 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->D31_2 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->D32_2 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->D33_2 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);

  FC_d->vec_n  = (float *) cuda_malloc(sizeof(float)*ny*nz*3);
  FC_d->vec_s1 = (float *) cuda_malloc(sizeof(float)*ny*nz*3);
  FC_d->vec_s2 = (float *) cuda_malloc(sizeof(float)*ny*nz*3);
  FC_d->x_et   = (float *) cuda_malloc(sizeof(float)*ny*nz);
  FC_d->y_et   = (float *) cuda_malloc(sizeof(float)*ny*nz);
  FC_d->z_et   = (float *) cuda_malloc(sizeof(float)*ny*nz);
  
  FC_d->matMin2Plus1 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->matMin2Plus2 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->matMin2Plus3 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->matMin2Plus4 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->matMin2Plus5 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);

  FC_d->matPlus2Min1 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->matPlus2Min2 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->matPlus2Min3 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->matPlus2Min4 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->matPlus2Min5 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  
  FC_d->matT1toVx_Min  = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->matVytoVx_Min  = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->matVztoVx_Min  = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->matT1toVx_Plus = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->matVytoVx_Plus = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  FC_d->matVztoVx_Plus = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
  
  FC_d->matVx2Vz1     = (float *) cuda_malloc(sizeof(float)*ny*3*3);
  FC_d->matVy2Vz1     = (float *) cuda_malloc(sizeof(float)*ny*3*3);
  FC_d->matVx2Vz2     = (float *) cuda_malloc(sizeof(float)*ny*3*3);
  FC_d->matVy2Vz2     = (float *) cuda_malloc(sizeof(float)*ny*3*3);
  FC_d->matPlus2Min1f = (float *) cuda_malloc(sizeof(float)*ny*3*3);
  FC_d->matPlus2Min2f = (float *) cuda_malloc(sizeof(float)*ny*3*3);
  FC_d->matPlus2Min3f = (float *) cuda_malloc(sizeof(float)*ny*3*3);
  FC_d->matMin2Plus1f = (float *) cuda_malloc(sizeof(float)*ny*3*3);
  FC_d->matMin2Plus2f = (float *) cuda_malloc(sizeof(float)*ny*3*3);
  FC_d->matMin2Plus3f = (float *) cuda_malloc(sizeof(float)*ny*3*3);

  FC_d->matT1toVxf_Min  = (float *) cuda_malloc(sizeof(float)*ny*3*3);
  FC_d->matVytoVxf_Min  = (float *) cuda_malloc(sizeof(float)*ny*3*3);
  FC_d->matT1toVxf_Plus = (float *) cuda_malloc(sizeof(float)*ny*3*3);
  FC_d->matVytoVxf_Plus = (float *) cuda_malloc(sizeof(float)*ny*3*3);

  CUDACHECK(cudaMemcpy(FC_d->rho_f, FC->rho_f, sizeof(float)*ny*nz*2, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->mu_f,  FC->mu_f,  sizeof(float)*ny*nz*2, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->lam_f, FC->lam_f, sizeof(float)*ny*nz*2, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(FC_d->D21_1, FC->D21_1, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->D22_1, FC->D22_1, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->D23_1, FC->D23_1, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->D31_1, FC->D31_1, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->D32_1, FC->D32_1, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->D33_1, FC->D33_1, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(FC_d->D21_2, FC->D21_2, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->D22_2, FC->D22_2, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->D23_2, FC->D23_2, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->D31_2, FC->D31_2, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->D32_2, FC->D32_2, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->D33_2, FC->D33_2, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(FC_d->vec_n,  FC->vec_n,  sizeof(float)*ny*nz*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->vec_s1, FC->vec_s1, sizeof(float)*ny*nz*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->vec_s2, FC->vec_s2, sizeof(float)*ny*nz*3, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(FC_d->x_et, FC->x_et, sizeof(float)*ny*nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->y_et, FC->y_et, sizeof(float)*ny*nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->z_et, FC->z_et, sizeof(float)*ny*nz, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(FC_d->matMin2Plus1, FC->matMin2Plus1, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matMin2Plus2, FC->matMin2Plus2, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matMin2Plus3, FC->matMin2Plus3, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matMin2Plus4, FC->matMin2Plus4, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matMin2Plus5, FC->matMin2Plus5, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(FC_d->matPlus2Min1, FC->matPlus2Min1, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matPlus2Min2, FC->matPlus2Min2, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matPlus2Min3, FC->matPlus2Min3, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matPlus2Min4, FC->matPlus2Min4, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matPlus2Min5, FC->matPlus2Min5, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(FC_d->matT1toVx_Min, FC->matT1toVx_Min, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matVytoVx_Min, FC->matVytoVx_Min, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matVztoVx_Min, FC->matVztoVx_Min, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(FC_d->matT1toVx_Plus, FC->matT1toVx_Plus, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matVytoVx_Plus, FC->matVytoVx_Plus, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matVztoVx_Plus, FC->matVztoVx_Plus, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(FC_d->matVx2Vz1, FC->matVx2Vz1, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matVy2Vz1, FC->matVy2Vz1, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matVx2Vz2, FC->matVx2Vz2, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matVy2Vz2, FC->matVy2Vz2, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(FC_d->matPlus2Min1f, FC->matPlus2Min1f, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matPlus2Min2f, FC->matPlus2Min2f, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matPlus2Min3f, FC->matPlus2Min3f, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matMin2Plus1f, FC->matMin2Plus1f, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matMin2Plus2f, FC->matMin2Plus2f, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matMin2Plus3f, FC->matMin2Plus3f, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(FC_d->matT1toVxf_Min,  FC->matT1toVxf_Min, sizeof(float)*ny*3*3,  cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matVytoVxf_Min,  FC->matVytoVxf_Min, sizeof(float)*ny*3*3,  cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matT1toVxf_Plus, FC->matT1toVxf_Plus, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(FC_d->matVytoVxf_Plus, FC->matVytoVxf_Plus, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));

  return 0;
}

int init_fault_device(gdcurv_t *gdcurv, fault_t *F, fault_t *F_d)
{
  int nj = gdcurv->nj;
  int nk = gdcurv->nk;
  memcpy(F_d,F,sizeof(fault_t));
  // for input
  F_d->T0x   = (float *) cuda_malloc(sizeof(float)*nj*nk);  // stress_init_x
  F_d->T0y   = (float *) cuda_malloc(sizeof(float)*nj*nk);  // stress_init_y
  F_d->T0z   = (float *) cuda_malloc(sizeof(float)*nj*nk);  // stress_init_z
  F_d->mu_s  = (float *) cuda_malloc(sizeof(float)*nj*nk);
  F_d->mu_d  = (float *) cuda_malloc(sizeof(float)*nj*nk);
  F_d->Dc    = (float *) cuda_malloc(sizeof(float)*nj*nk);
  F_d->C0    = (float *) cuda_malloc(sizeof(float)*nj*nk);
  // for output
  F_d->Tn         = (float *) cuda_malloc(sizeof(float)*nj*nk);
  F_d->Ts1        = (float *) cuda_malloc(sizeof(float)*nj*nk);
  F_d->Ts2        = (float *) cuda_malloc(sizeof(float)*nj*nk);
  F_d->slip       = (float *) cuda_malloc(sizeof(float)*nj*nk);
  F_d->slip1      = (float *) cuda_malloc(sizeof(float)*nj*nk); 
  F_d->slip2      = (float *) cuda_malloc(sizeof(float)*nj*nk);  
  F_d->Vs         = (float *) cuda_malloc(sizeof(float)*nj*nk);
  F_d->Vs1        = (float *) cuda_malloc(sizeof(float)*nj*nk);
  F_d->Vs2        = (float *) cuda_malloc(sizeof(float)*nj*nk);
  F_d->peak_Vs    = (float *) cuda_malloc(sizeof(float)*nj*nk);
  F_d->init_t0    = (float *) cuda_malloc(sizeof(float)*nj*nk);
  // for inner
  F_d->tTn          = (float *) cuda_malloc(sizeof(float)*nj*nk);
  F_d->tTs1         = (float *) cuda_malloc(sizeof(float)*nj*nk);
  F_d->tTs2         = (float *) cuda_malloc(sizeof(float)*nj*nk);
  F_d->united       = (int *)   cuda_malloc(sizeof(int)*nj*nk);
  F_d->faultgrid    = (int *)   cuda_malloc(sizeof(int)*nj*nk);
  F_d->rup_index_y  = (int *)   cuda_malloc(sizeof(int)*nj*nk);
  F_d->rup_index_z  = (int *)   cuda_malloc(sizeof(int)*nj*nk);
  F_d->flag_rup     = (int *)   cuda_malloc(sizeof(int)*nj*nk);
  F_d->init_t0_flag = (int *)   cuda_malloc(sizeof(int)*nj*nk);

  CUDACHECK(cudaMemcpy(F_d->T0x,  F->T0x,  sizeof(float)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->T0y,  F->T0y,  sizeof(float)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->T0z,  F->T0z,  sizeof(float)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->mu_s, F->mu_s, sizeof(float)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->mu_d, F->mu_d, sizeof(float)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->Dc,   F->Dc,   sizeof(float)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->C0,   F->C0,   sizeof(float)*nj*nk, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(F_d->Tn,    F->Tn,    sizeof(float)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->Ts1,   F->Ts1,   sizeof(float)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->Ts2,   F->Ts2,   sizeof(float)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->slip,  F->slip,  sizeof(float)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->slip1, F->slip1, sizeof(float)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->slip2, F->slip2, sizeof(float)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->Vs,    F->Vs,    sizeof(float)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->Vs1,   F->Vs1,   sizeof(float)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->Vs2,   F->Vs2,   sizeof(float)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->peak_Vs, F->peak_Vs, sizeof(float)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->init_t0, F->init_t0, sizeof(float)*nj*nk, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(F_d->tTn,  F->tTn,  sizeof(float)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->tTs1, F->tTs1, sizeof(float)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->tTs2, F->tTs2, sizeof(float)*nj*nk, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(F_d->united,       F->united,       sizeof(int)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->faultgrid,    F->faultgrid,    sizeof(int)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->rup_index_y,  F->rup_index_y,  sizeof(int)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->rup_index_z,  F->rup_index_z,  sizeof(int)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->flag_rup,     F->flag_rup,     sizeof(int)*nj*nk, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(F_d->init_t0_flag, F->init_t0_flag, sizeof(int)*nj*nk, cudaMemcpyHostToDevice));

  return 0;
}

int init_fault_wav_device(fault_wav_t *FW, fault_wav_t *FW_d)
{
  int ny = FW->ny;
  int nz = FW->nz;
  int nlevel = FW->nlevel;
  size_t siz_ilevel = FW->siz_ilevel;
  memcpy(FW_d,FW,sizeof(fault_wav_t));
  FW_d->v5d  = (float *) cuda_malloc(sizeof(float)*siz_ilevel*nlevel);
  FW_d->T1x  = (float *) cuda_malloc(sizeof(float)*7*ny*nz); 
  FW_d->T1y  = (float *) cuda_malloc(sizeof(float)*7*ny*nz); 
  FW_d->T1z  = (float *) cuda_malloc(sizeof(float)*7*ny*nz); 
  FW_d->hT1x = (float *) cuda_malloc(sizeof(float)*ny*nz); 
  FW_d->hT1y = (float *) cuda_malloc(sizeof(float)*ny*nz); 
  FW_d->hT1z = (float *) cuda_malloc(sizeof(float)*ny*nz); 
  FW_d->mT1x = (float *) cuda_malloc(sizeof(float)*ny*nz); 
  FW_d->mT1y = (float *) cuda_malloc(sizeof(float)*ny*nz); 
  FW_d->mT1z = (float *) cuda_malloc(sizeof(float)*ny*nz); 
  CUDACHECK(cudaMemset(FW_d->v5d,  0, sizeof(float)*siz_ilevel*nlevel));
  CUDACHECK(cudaMemset(FW_d->T1x,  0, sizeof(float)*7*ny*nz));
  CUDACHECK(cudaMemset(FW_d->T1y,  0, sizeof(float)*7*ny*nz));
  CUDACHECK(cudaMemset(FW_d->T1z,  0, sizeof(float)*7*ny*nz));
  CUDACHECK(cudaMemset(FW_d->hT1x, 0, sizeof(float)*ny*nz));
  CUDACHECK(cudaMemset(FW_d->hT1y, 0, sizeof(float)*ny*nz));
  CUDACHECK(cudaMemset(FW_d->hT1z, 0, sizeof(float)*ny*nz));
  CUDACHECK(cudaMemset(FW_d->mT1x, 0, sizeof(float)*ny*nz));
  CUDACHECK(cudaMemset(FW_d->mT1y, 0, sizeof(float)*ny*nz));
  CUDACHECK(cudaMemset(FW_d->mT1z, 0, sizeof(float)*ny*nz));

  return 0;
}

int init_bdryfree_device(gdcurv_t *gdcurv, bdryfree_t *bdryfree, bdryfree_t *bdryfree_d)
{
  int nx = gdcurv->nx;
  int ny = gdcurv->ny;

  memcpy(bdryfree_d,bdryfree,sizeof(bdryfree_t));
  
  if (bdryfree->is_sides_free[CONST_NDIM-1][1] == 1)
  {
    bdryfree_d->matVx2Vz2 = (float *) cuda_malloc(sizeof(float)*nx*ny*CONST_NDIM*CONST_NDIM);
    bdryfree_d->matVy2Vz2 = (float *) cuda_malloc(sizeof(float)*nx*ny*CONST_NDIM*CONST_NDIM);
    
    // use gpu calculate, not need copy 
    //CUDACHECK(cudaMemcpy(bdryfree_d->matVx2Vz2, bdryfree->matVx2Vz2, sizeof(float)*nx*ny*CONST_NDIM*CONST_NDIM, cudaMemcpyHostToDevice));
    //CUDACHECK(cudaMemcpy(bdryfree_d->matVy2Vz2, bdryfree->matVy2Vz2, sizeof(float)*nx*ny*CONST_NDIM*CONST_NDIM, cudaMemcpyHostToDevice));
  }

  return 0;
}

int init_bdrypml_device(gdcurv_t *gdcurv, bdrypml_t *bdrypml, bdrypml_t *bdrypml_d)
{
  memcpy(bdrypml_d,bdrypml,sizeof(bdrypml_t));
  // copy bdrypml
  if (bdrypml_d->is_enable_pml == 1)
  {
    for(int idim=0; idim<CONST_NDIM; idim++){
      for(int iside=0; iside<2; iside++){
        if(bdrypml_d->is_sides_pml[idim][iside] == 1){
          int npoints = bdrypml_d->num_of_layers[idim][iside] + 1;
          bdrypml_d->A[idim][iside]   = (float *) cuda_malloc(npoints * sizeof(float));
          bdrypml_d->B[idim][iside]   = (float *) cuda_malloc(npoints * sizeof(float));
          bdrypml_d->D[idim][iside]   = (float *) cuda_malloc(npoints * sizeof(float));
          CUDACHECK(cudaMemcpy(bdrypml_d->A[idim][iside],bdrypml->A[idim][iside],npoints*sizeof(float),cudaMemcpyHostToDevice));
          CUDACHECK(cudaMemcpy(bdrypml_d->B[idim][iside],bdrypml->B[idim][iside],npoints*sizeof(float),cudaMemcpyHostToDevice));
          CUDACHECK(cudaMemcpy(bdrypml_d->D[idim][iside],bdrypml->D[idim][iside],npoints*sizeof(float),cudaMemcpyHostToDevice));
          } else {
          bdrypml_d->A[idim][iside]   = NULL;
          bdrypml_d->B[idim][iside]   = NULL;
          bdrypml_d->D[idim][iside]   = NULL;
        }
      }
    }

    for(int idim=0; idim<CONST_NDIM; idim++){
      for(int iside=0; iside<2; iside++){
        bdrypml_auxvar_t *auxvar_d = &(bdrypml_d->auxvar[idim][iside]);
        if(auxvar_d->siz_icmp > 0){
          auxvar_d->var = (float *) cuda_malloc(sizeof(float)*auxvar_d->siz_ilevel*auxvar_d->nlevel); 
          CUDACHECK(cudaMemset(auxvar_d->var,0,sizeof(float)*auxvar_d->siz_ilevel*auxvar_d->nlevel));
        } else {
        auxvar_d->var = NULL;
        }
      }
    }
  }

  return 0;
}

int init_bdryexp_device(gdcurv_t *gdcurv, bdryexp_t *bdryexp, bdryexp_t *bdryexp_d)
{
  int nx = gdcurv->nx;
  int ny = gdcurv->ny;
  int nz = gdcurv->nz;

  memcpy(bdryexp_d,bdryexp,sizeof(bdryexp_t));
  // copy bdryexp
  if (bdryexp->is_enable_ablexp == 1)
  {
    bdryexp_d->ablexp_Ex = (float *) cuda_malloc(nx * sizeof(float));
    bdryexp_d->ablexp_Ey = (float *) cuda_malloc(ny * sizeof(float));
    bdryexp_d->ablexp_Ez = (float *) cuda_malloc(nz * sizeof(float));
    CUDACHECK(cudaMemcpy(bdryexp_d->ablexp_Ex,bdryexp->ablexp_Ex,nx*sizeof(float),cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(bdryexp_d->ablexp_Ey,bdryexp->ablexp_Ey,ny*sizeof(float),cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(bdryexp_d->ablexp_Ez,bdryexp->ablexp_Ez,nz*sizeof(float),cudaMemcpyHostToDevice));
  }

  return 0;
}

int init_wave_device(wav_t *wav, wav_t *wav_d)
{
  size_t siz_ilevel = wav->siz_ilevel;
  int nlevel = wav->nlevel;
  memcpy(wav_d,wav,sizeof(wav_t));
  wav_d->v5d   = (float *) cuda_malloc(sizeof(float)*siz_ilevel*nlevel);
  CUDACHECK(cudaMemset(wav_d->v5d,0,sizeof(float)*siz_ilevel*nlevel));

  return 0;
}

float *init_PGVAD_device(gdcurv_t *gdcurv)
{
  float *PG_d;
  int nx = gdcurv->nx;
  int ny = gdcurv->ny;
  PG_d = (float *) cuda_malloc(sizeof(float)*CONST_NDIM_5*nx*ny);
  CUDACHECK(cudaMemset(PG_d,0,sizeof(float)*CONST_NDIM_5*nx*ny));

  return PG_d;
}

float *init_Dis_accu_device(gdcurv_t *gdcurv)
{
  float *Dis_accu_d;
  int nx = gdcurv->nx;
  int ny = gdcurv->ny;
  Dis_accu_d = (float *) cuda_malloc(sizeof(float)*CONST_NDIM*nx*ny);
  CUDACHECK(cudaMemset(Dis_accu_d,0,sizeof(float)*CONST_NDIM*nx*ny));

  return Dis_accu_d;
}

int *init_neighid_device(int *neighid)
{
  int *neighid_d; 
  neighid_d = (int *) cuda_malloc(sizeof(int)*CONST_NDIM_2);
  CUDACHECK(cudaMemcpy(neighid_d,neighid,sizeof(int)*CONST_NDIM_2,cudaMemcpyHostToDevice));

  return neighid_d;
}

int dealloc_gdcurv_device(gdcurv_t gdcurv_d)
{
  CUDACHECK(cudaFree(gdcurv_d.x3d)); 
  CUDACHECK(cudaFree(gdcurv_d.y3d)); 
  CUDACHECK(cudaFree(gdcurv_d.z3d)); 

  CUDACHECK(cudaFree(gdcurv_d.cell_xmin)); 
  CUDACHECK(cudaFree(gdcurv_d.cell_xmax)); 
  CUDACHECK(cudaFree(gdcurv_d.cell_ymin)); 
  CUDACHECK(cudaFree(gdcurv_d.cell_ymax)); 
  CUDACHECK(cudaFree(gdcurv_d.cell_zmin)); 
  CUDACHECK(cudaFree(gdcurv_d.cell_zmax)); 
  CUDACHECK(cudaFree(gdcurv_d.tile_istart));
  CUDACHECK(cudaFree(gdcurv_d.tile_iend));  
  CUDACHECK(cudaFree(gdcurv_d.tile_jstart));
  CUDACHECK(cudaFree(gdcurv_d.tile_jend));  
  CUDACHECK(cudaFree(gdcurv_d.tile_kstart));
  CUDACHECK(cudaFree(gdcurv_d.tile_kend));  
  CUDACHECK(cudaFree(gdcurv_d.tile_xmin));
  CUDACHECK(cudaFree(gdcurv_d.tile_xmax));
  CUDACHECK(cudaFree(gdcurv_d.tile_ymin));
  CUDACHECK(cudaFree(gdcurv_d.tile_ymax));
  CUDACHECK(cudaFree(gdcurv_d.tile_zmin));
  CUDACHECK(cudaFree(gdcurv_d.tile_zmax));  

  return 0;
}

int dealloc_md_device(md_t md_d)
{
  if (md_d.medium_type == CONST_MEDIUM_ELASTIC_ISO)
  {
    CUDACHECK(cudaFree(md_d.rho   )); 
    CUDACHECK(cudaFree(md_d.lambda)); 
    CUDACHECK(cudaFree(md_d.mu    )); 
  }
  if (md_d.medium_type == CONST_MEDIUM_ELASTIC_VTI)
  {
    CUDACHECK(cudaFree(md_d.rho)); 
    CUDACHECK(cudaFree(md_d.c11)); 
    CUDACHECK(cudaFree(md_d.c33)); 
    CUDACHECK(cudaFree(md_d.c55)); 
    CUDACHECK(cudaFree(md_d.c66)); 
    CUDACHECK(cudaFree(md_d.c13)); 
  }
  if (md_d.medium_type == CONST_MEDIUM_ELASTIC_ANISO)
  {
    CUDACHECK(cudaFree(md_d.rho)); 
    CUDACHECK(cudaFree(md_d.c11)); 
    CUDACHECK(cudaFree(md_d.c12)); 
    CUDACHECK(cudaFree(md_d.c13)); 
    CUDACHECK(cudaFree(md_d.c14)); 
    CUDACHECK(cudaFree(md_d.c15)); 
    CUDACHECK(cudaFree(md_d.c16)); 
    CUDACHECK(cudaFree(md_d.c22)); 
    CUDACHECK(cudaFree(md_d.c23)); 
    CUDACHECK(cudaFree(md_d.c24)); 
    CUDACHECK(cudaFree(md_d.c25)); 
    CUDACHECK(cudaFree(md_d.c26)); 
    CUDACHECK(cudaFree(md_d.c33)); 
    CUDACHECK(cudaFree(md_d.c34)); 
    CUDACHECK(cudaFree(md_d.c35)); 
    CUDACHECK(cudaFree(md_d.c36)); 
    CUDACHECK(cudaFree(md_d.c44)); 
    CUDACHECK(cudaFree(md_d.c45)); 
    CUDACHECK(cudaFree(md_d.c46)); 
    CUDACHECK(cudaFree(md_d.c55)); 
    CUDACHECK(cudaFree(md_d.c56)); 
    CUDACHECK(cudaFree(md_d.c66)); 
  }

  return 0;
}

int dealloc_fd_device(fd_device_t fd_device_d)
{
  CUDACHECK(cudaFree(fd_device_d.fdx_coef_d));
  CUDACHECK(cudaFree(fd_device_d.fdy_coef_d));
  CUDACHECK(cudaFree(fd_device_d.fdz_coef_d));

  CUDACHECK(cudaFree(fd_device_d.fdx_indx_d));
  CUDACHECK(cudaFree(fd_device_d.fdy_indx_d));
  CUDACHECK(cudaFree(fd_device_d.fdz_indx_d));

  CUDACHECK(cudaFree(fd_device_d.fdx_shift_d));
  CUDACHECK(cudaFree(fd_device_d.fdy_shift_d));
  CUDACHECK(cudaFree(fd_device_d.fdz_shift_d));

  return 0;
}
int dealloc_metric_device(gd_metric_t metric_d)
{
  CUDACHECK(cudaFree(metric_d.jac   )); 
  CUDACHECK(cudaFree(metric_d.xi_x  )); 
  CUDACHECK(cudaFree(metric_d.xi_y  )); 
  CUDACHECK(cudaFree(metric_d.xi_z  )); 
  CUDACHECK(cudaFree(metric_d.eta_x )); 
  CUDACHECK(cudaFree(metric_d.eta_y )); 
  CUDACHECK(cudaFree(metric_d.eta_z )); 
  CUDACHECK(cudaFree(metric_d.zeta_x)); 
  CUDACHECK(cudaFree(metric_d.zeta_y)); 
  CUDACHECK(cudaFree(metric_d.zeta_z)); 
  return 0;
}

int dealloc_fault_coef_device(fault_coef_t FC_d)
{
  CUDACHECK( cudaFree(FC_d.rho_f));
  CUDACHECK( cudaFree(FC_d.mu_f ));
  CUDACHECK( cudaFree(FC_d.lam_f));

  CUDACHECK( cudaFree(FC_d.D21_1));
  CUDACHECK( cudaFree(FC_d.D22_1));
  CUDACHECK( cudaFree(FC_d.D23_1));
  CUDACHECK( cudaFree(FC_d.D31_1));
  CUDACHECK( cudaFree(FC_d.D32_1));
  CUDACHECK( cudaFree(FC_d.D33_1));

  CUDACHECK( cudaFree(FC_d.D21_2));
  CUDACHECK( cudaFree(FC_d.D22_2));
  CUDACHECK( cudaFree(FC_d.D23_2));
  CUDACHECK( cudaFree(FC_d.D31_2));
  CUDACHECK( cudaFree(FC_d.D32_2));
  CUDACHECK( cudaFree(FC_d.D33_2));

  CUDACHECK( cudaFree(FC_d.vec_n ));
  CUDACHECK( cudaFree(FC_d.vec_s1));
  CUDACHECK( cudaFree(FC_d.vec_s2));

  CUDACHECK( cudaFree(FC_d.x_et));
  CUDACHECK( cudaFree(FC_d.y_et));
  CUDACHECK( cudaFree(FC_d.z_et));

  CUDACHECK( cudaFree(FC_d.matMin2Plus1));
  CUDACHECK( cudaFree(FC_d.matMin2Plus2));
  CUDACHECK( cudaFree(FC_d.matMin2Plus3));
  CUDACHECK( cudaFree(FC_d.matMin2Plus4));
  CUDACHECK( cudaFree(FC_d.matMin2Plus5));

  CUDACHECK( cudaFree(FC_d.matPlus2Min1));
  CUDACHECK( cudaFree(FC_d.matPlus2Min2));
  CUDACHECK( cudaFree(FC_d.matPlus2Min3));
  CUDACHECK( cudaFree(FC_d.matPlus2Min4));
  CUDACHECK( cudaFree(FC_d.matPlus2Min5));

  CUDACHECK( cudaFree(FC_d.matT1toVx_Min));
  CUDACHECK( cudaFree(FC_d.matVytoVx_Min));
  CUDACHECK( cudaFree(FC_d.matVztoVx_Min));

  CUDACHECK( cudaFree(FC_d.matT1toVx_Plus));
  CUDACHECK( cudaFree(FC_d.matVytoVx_Plus));
  CUDACHECK( cudaFree(FC_d.matVztoVx_Plus));

  CUDACHECK( cudaFree(FC_d.matVx2Vz1));
  CUDACHECK( cudaFree(FC_d.matVy2Vz1));
  CUDACHECK( cudaFree(FC_d.matVx2Vz2));
  CUDACHECK( cudaFree(FC_d.matVy2Vz2));

  CUDACHECK( cudaFree(FC_d.matPlus2Min1f));
  CUDACHECK( cudaFree(FC_d.matPlus2Min2f));
  CUDACHECK( cudaFree(FC_d.matPlus2Min3f));
  CUDACHECK( cudaFree(FC_d.matMin2Plus1f));
  CUDACHECK( cudaFree(FC_d.matMin2Plus2f));
  CUDACHECK( cudaFree(FC_d.matMin2Plus3f));

  CUDACHECK( cudaFree(FC_d.matT1toVxf_Min));
  CUDACHECK( cudaFree(FC_d.matVytoVxf_Min));
  CUDACHECK( cudaFree(FC_d.matT1toVxf_Plus));
  CUDACHECK( cudaFree(FC_d.matVytoVxf_Plus));

  return 0;
}

int dealloc_fault_device(fault_t F_d)
{
  CUDACHECK( cudaFree(F_d.T0x));
  CUDACHECK( cudaFree(F_d.T0y));
  CUDACHECK( cudaFree(F_d.T0z));
  CUDACHECK( cudaFree(F_d.mu_s));
  CUDACHECK( cudaFree(F_d.mu_d));
  CUDACHECK( cudaFree(F_d.Dc));
  CUDACHECK( cudaFree(F_d.C0));

  CUDACHECK( cudaFree(F_d.Tn));
  CUDACHECK( cudaFree(F_d.Ts1));
  CUDACHECK( cudaFree(F_d.Ts2));
  CUDACHECK( cudaFree(F_d.slip));
  CUDACHECK( cudaFree(F_d.slip1));
  CUDACHECK( cudaFree(F_d.slip2));
  CUDACHECK( cudaFree(F_d.Vs));
  CUDACHECK( cudaFree(F_d.Vs1));
  CUDACHECK( cudaFree(F_d.Vs2));
  CUDACHECK( cudaFree(F_d.peak_Vs));
  CUDACHECK( cudaFree(F_d.init_t0));

  CUDACHECK( cudaFree(F_d.united));
  CUDACHECK( cudaFree(F_d.faultgrid));
  CUDACHECK( cudaFree(F_d.rup_index_y));
  CUDACHECK( cudaFree(F_d.rup_index_z));
  CUDACHECK( cudaFree(F_d.flag_rup));
  CUDACHECK( cudaFree(F_d.init_t0_flag));

  return 0;
}

int dealloc_fault_wav_device(fault_wav_t FW_d)
{

  CUDACHECK(cudaFree(FW_d.v5d));
  CUDACHECK(cudaFree(FW_d.T1x));
  CUDACHECK(cudaFree(FW_d.T1y));
  CUDACHECK(cudaFree(FW_d.T1z));
  CUDACHECK(cudaFree(FW_d.hT1x));
  CUDACHECK(cudaFree(FW_d.hT1y));
  CUDACHECK(cudaFree(FW_d.hT1z));
  CUDACHECK(cudaFree(FW_d.mT1x));
  CUDACHECK(cudaFree(FW_d.mT1y));
  CUDACHECK(cudaFree(FW_d.mT1z));

  return 0;
}

int dealloc_bdryfree_device(bdryfree_t bdryfree_d)
{
  if (bdryfree_d.is_sides_free[CONST_NDIM-1][1] == 1)
  {
    CUDACHECK(cudaFree(bdryfree_d.matVx2Vz2)); 
    CUDACHECK(cudaFree(bdryfree_d.matVy2Vz2)); 
  }
  return 0;
}

int dealloc_bdrypml_device(bdrypml_t bdrypml_d)
{
  if (bdrypml_d.is_enable_pml == 1)
  {
    for(int idim=0; idim<CONST_NDIM; idim++){
      for(int iside=0; iside<2; iside++){
        if(bdrypml_d.is_sides_pml[idim][iside] == 1){
          CUDACHECK(cudaFree(bdrypml_d.A[idim][iside])); 
          CUDACHECK(cudaFree(bdrypml_d.B[idim][iside])); 
          CUDACHECK(cudaFree(bdrypml_d.D[idim][iside])); 
        }
      }
    }  
    for(int idim=0; idim<CONST_NDIM; idim++){
      for(int iside=0; iside<2; iside++){
        bdrypml_auxvar_t *auxvar_d = &(bdrypml_d.auxvar[idim][iside]);
        if(auxvar_d->siz_icmp > 0){
          CUDACHECK(cudaFree(auxvar_d->var)); 
        }
      }
    }  
  }

  return 0;
}

int dealloc_bdryexp_device(bdryexp_t bdryexp_d)
{
  if (bdryexp_d.is_enable_ablexp == 1)
  {
    CUDACHECK(cudaFree(bdryexp_d.ablexp_Ex)); 
    CUDACHECK(cudaFree(bdryexp_d.ablexp_Ey));
    CUDACHECK(cudaFree(bdryexp_d.ablexp_Ez));
  }
  return 0;
}

int dealloc_wave_device(wav_t wav_d)
{
  CUDACHECK(cudaFree(wav_d.v5d)); 
  return 0;
}
