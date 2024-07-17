#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "alloc.h"
#include "cuda_common.h"

// only copy grid info
int init_gdinfo_device(gd_t *gd, gd_t *gd_d)
{
  memcpy(gd_d,gd,sizeof(gd_t));

  return 0;
}

int init_gd_device(gd_t *gd, gd_t *gd_d)
{
  size_t siz_icmp = gd->siz_icmp;
  memcpy(gd_d,gd,sizeof(gd_t));
  gd_d->x3d = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gd_d->y3d = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gd_d->z3d = (float *) cuda_malloc(sizeof(float)*siz_icmp);

  gd_d->cell_xmin = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gd_d->cell_xmax = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gd_d->cell_ymin = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gd_d->cell_ymax = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gd_d->cell_zmin = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gd_d->cell_zmax = (float *) cuda_malloc(sizeof(float)*siz_icmp);

  gd_d->tile_istart = (int *) cuda_malloc(sizeof(int)*GD_TILE_NX);
  gd_d->tile_iend   = (int *) cuda_malloc(sizeof(int)*GD_TILE_NX);
  gd_d->tile_jstart = (int *) cuda_malloc(sizeof(int)*GD_TILE_NY);
  gd_d->tile_jend   = (int *) cuda_malloc(sizeof(int)*GD_TILE_NY);
  gd_d->tile_kstart = (int *) cuda_malloc(sizeof(int)*GD_TILE_NZ);
  gd_d->tile_kend   = (int *) cuda_malloc(sizeof(int)*GD_TILE_NZ);

  int size = GD_TILE_NX * GD_TILE_NY * GD_TILE_NZ;
  gd_d->tile_xmin = (float *) cuda_malloc(sizeof(float)*size);
  gd_d->tile_xmax = (float *) cuda_malloc(sizeof(float)*size);
  gd_d->tile_ymin = (float *) cuda_malloc(sizeof(float)*size);
  gd_d->tile_ymax = (float *) cuda_malloc(sizeof(float)*size);
  gd_d->tile_zmin = (float *) cuda_malloc(sizeof(float)*size);
  gd_d->tile_zmax = (float *) cuda_malloc(sizeof(float)*size);

  CUDACHECK(cudaMemcpy(gd_d->x3d, gd->x3d, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gd_d->y3d, gd->y3d, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gd_d->z3d, gd->z3d, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(gd_d->cell_xmin, gd->cell_xmin, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gd_d->cell_xmax, gd->cell_xmax, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gd_d->cell_ymin, gd->cell_ymin, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gd_d->cell_ymax, gd->cell_ymax, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gd_d->cell_zmin, gd->cell_zmin, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gd_d->cell_zmax, gd->cell_zmax, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(gd_d->tile_istart, gd->tile_istart, sizeof(int)*GD_TILE_NX, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gd_d->tile_iend, gd->tile_iend, sizeof(int)*GD_TILE_NX, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gd_d->tile_jstart, gd->tile_jstart, sizeof(int)*GD_TILE_NY, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gd_d->tile_jend, gd->tile_jend, sizeof(int)*GD_TILE_NY, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gd_d->tile_kstart, gd->tile_kstart, sizeof(int)*GD_TILE_NZ, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gd_d->tile_kend, gd->tile_kend, sizeof(int)*GD_TILE_NZ, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(gd_d->tile_xmin, gd->tile_xmin, sizeof(float)*size, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gd_d->tile_xmax, gd->tile_xmax, sizeof(float)*size, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gd_d->tile_ymin, gd->tile_ymin, sizeof(float)*size, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gd_d->tile_ymax, gd->tile_ymax, sizeof(float)*size, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gd_d->tile_zmin, gd->tile_zmin, sizeof(float)*size, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gd_d->tile_zmax, gd->tile_zmax, sizeof(float)*size, cudaMemcpyHostToDevice));

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

int init_fault_coef_device(gd_t *gd, fault_coef_t *FC, fault_coef_t *FC_d)
{
  int ny = gd->ny;
  int nz = gd->nz;
  memcpy(FC_d,FC,sizeof(fault_coef_t));

  FC_d->fault_index = (int *) cuda_malloc(sizeof(int)*FC->number_fault);
  CUDACHECK(cudaMemcpy(FC_d->fault_index, FC->fault_index, sizeof(int)*FC->number_fault, cudaMemcpyHostToDevice));

  for(int id=0; id<FC->number_fault; id++)
  {
    fault_coef_one_t *thisone = FC->fault_coef_one + id;
    fault_coef_one_t *thisone_d = FC_d->fault_coef_one + id;

    thisone_d->rho_f = (float *) cuda_malloc(sizeof(float)*ny*nz*2);
    thisone_d->mu_f  = (float *) cuda_malloc(sizeof(float)*ny*nz*2);
    thisone_d->lam_f = (float *) cuda_malloc(sizeof(float)*ny*nz*2);

    thisone_d->D21_1 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->D22_1 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->D23_1 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->D31_1 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->D32_1 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->D33_1 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);

    thisone_d->D21_2 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->D22_2 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->D23_2 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->D31_2 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->D32_2 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->D33_2 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);

    thisone_d->vec_n  = (float *) cuda_malloc(sizeof(float)*ny*nz*3);
    thisone_d->vec_s1 = (float *) cuda_malloc(sizeof(float)*ny*nz*3);
    thisone_d->vec_s2 = (float *) cuda_malloc(sizeof(float)*ny*nz*3);
    thisone_d->x_et   = (float *) cuda_malloc(sizeof(float)*ny*nz);
    thisone_d->y_et   = (float *) cuda_malloc(sizeof(float)*ny*nz);
    thisone_d->z_et   = (float *) cuda_malloc(sizeof(float)*ny*nz);
    
    thisone_d->matMin2Plus1 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->matMin2Plus2 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->matMin2Plus3 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->matMin2Plus4 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->matMin2Plus5 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);

    thisone_d->matPlus2Min1 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->matPlus2Min2 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->matPlus2Min3 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->matPlus2Min4 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->matPlus2Min5 = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    
    thisone_d->matT1toVx_Min  = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->matVytoVx_Min  = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->matVztoVx_Min  = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->matT1toVx_Plus = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->matVytoVx_Plus = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    thisone_d->matVztoVx_Plus = (float *) cuda_malloc(sizeof(float)*ny*nz*3*3);
    
    thisone_d->matVx2Vz1     = (float *) cuda_malloc(sizeof(float)*ny*3*3);
    thisone_d->matVy2Vz1     = (float *) cuda_malloc(sizeof(float)*ny*3*3);
    thisone_d->matVx2Vz2     = (float *) cuda_malloc(sizeof(float)*ny*3*3);
    thisone_d->matVy2Vz2     = (float *) cuda_malloc(sizeof(float)*ny*3*3);
    thisone_d->matPlus2Min1f = (float *) cuda_malloc(sizeof(float)*ny*3*3);
    thisone_d->matPlus2Min2f = (float *) cuda_malloc(sizeof(float)*ny*3*3);
    thisone_d->matPlus2Min3f = (float *) cuda_malloc(sizeof(float)*ny*3*3);
    thisone_d->matMin2Plus1f = (float *) cuda_malloc(sizeof(float)*ny*3*3);
    thisone_d->matMin2Plus2f = (float *) cuda_malloc(sizeof(float)*ny*3*3);
    thisone_d->matMin2Plus3f = (float *) cuda_malloc(sizeof(float)*ny*3*3);

    thisone_d->matT1toVxf_Min  = (float *) cuda_malloc(sizeof(float)*ny*3*3);
    thisone_d->matVytoVxf_Min  = (float *) cuda_malloc(sizeof(float)*ny*3*3);
    thisone_d->matT1toVxf_Plus = (float *) cuda_malloc(sizeof(float)*ny*3*3);
    thisone_d->matVytoVxf_Plus = (float *) cuda_malloc(sizeof(float)*ny*3*3);

    CUDACHECK(cudaMemcpy(thisone_d->rho_f, thisone->rho_f, sizeof(float)*ny*nz*2, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->mu_f,  thisone->mu_f,  sizeof(float)*ny*nz*2, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->lam_f, thisone->lam_f, sizeof(float)*ny*nz*2, cudaMemcpyHostToDevice));

    CUDACHECK(cudaMemcpy(thisone_d->D21_1, thisone->D21_1, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->D22_1, thisone->D22_1, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->D23_1, thisone->D23_1, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->D31_1, thisone->D31_1, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->D32_1, thisone->D32_1, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->D33_1, thisone->D33_1, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));

    CUDACHECK(cudaMemcpy(thisone_d->D21_2, thisone->D21_2, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->D22_2, thisone->D22_2, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->D23_2, thisone->D23_2, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->D31_2, thisone->D31_2, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->D32_2, thisone->D32_2, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->D33_2, thisone->D33_2, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));

    CUDACHECK(cudaMemcpy(thisone_d->vec_n,  thisone->vec_n,  sizeof(float)*ny*nz*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->vec_s1, thisone->vec_s1, sizeof(float)*ny*nz*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->vec_s2, thisone->vec_s2, sizeof(float)*ny*nz*3, cudaMemcpyHostToDevice));

    CUDACHECK(cudaMemcpy(thisone_d->x_et, thisone->x_et, sizeof(float)*ny*nz, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->y_et, thisone->y_et, sizeof(float)*ny*nz, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->z_et, thisone->z_et, sizeof(float)*ny*nz, cudaMemcpyHostToDevice));

    CUDACHECK(cudaMemcpy(thisone_d->matMin2Plus1, thisone->matMin2Plus1, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matMin2Plus2, thisone->matMin2Plus2, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matMin2Plus3, thisone->matMin2Plus3, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matMin2Plus4, thisone->matMin2Plus4, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matMin2Plus5, thisone->matMin2Plus5, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));

    CUDACHECK(cudaMemcpy(thisone_d->matPlus2Min1, thisone->matPlus2Min1, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matPlus2Min2, thisone->matPlus2Min2, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matPlus2Min3, thisone->matPlus2Min3, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matPlus2Min4, thisone->matPlus2Min4, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matPlus2Min5, thisone->matPlus2Min5, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));

    CUDACHECK(cudaMemcpy(thisone_d->matT1toVx_Min, thisone->matT1toVx_Min, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matVytoVx_Min, thisone->matVytoVx_Min, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matVztoVx_Min, thisone->matVztoVx_Min, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));

    CUDACHECK(cudaMemcpy(thisone_d->matT1toVx_Plus, thisone->matT1toVx_Plus, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matVytoVx_Plus, thisone->matVytoVx_Plus, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matVztoVx_Plus, thisone->matVztoVx_Plus, sizeof(float)*ny*nz*3*3, cudaMemcpyHostToDevice));

    CUDACHECK(cudaMemcpy(thisone_d->matVx2Vz1, thisone->matVx2Vz1, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matVy2Vz1, thisone->matVy2Vz1, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matVx2Vz2, thisone->matVx2Vz2, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matVy2Vz2, thisone->matVy2Vz2, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));

    CUDACHECK(cudaMemcpy(thisone_d->matPlus2Min1f, thisone->matPlus2Min1f, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matPlus2Min2f, thisone->matPlus2Min2f, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matPlus2Min3f, thisone->matPlus2Min3f, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matMin2Plus1f, thisone->matMin2Plus1f, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matMin2Plus2f, thisone->matMin2Plus2f, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matMin2Plus3f, thisone->matMin2Plus3f, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));

    CUDACHECK(cudaMemcpy(thisone_d->matT1toVxf_Min,  thisone->matT1toVxf_Min, sizeof(float)*ny*3*3,  cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matVytoVxf_Min,  thisone->matVytoVxf_Min, sizeof(float)*ny*3*3,  cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matT1toVxf_Plus, thisone->matT1toVxf_Plus, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->matVytoVxf_Plus, thisone->matVytoVxf_Plus, sizeof(float)*ny*3*3, cudaMemcpyHostToDevice));

  }

  return 0;
}

int init_fault_device(gd_t *gd, fault_t *F, fault_t *F_d)
{
  int ny = gd->ny;
  int nz = gd->nz;
  memcpy(F_d,F,sizeof(fault_t));

  F_d->fault_index = (int *) cuda_malloc(sizeof(int)*F->number_fault);
  CUDACHECK(cudaMemcpy(F_d->fault_index, F->fault_index, sizeof(int)*F->number_fault, cudaMemcpyHostToDevice));

  for(int id=0; id<F->number_fault; id++)
  {
    fault_one_t *thisone = F->fault_one + id;
    fault_one_t *thisone_d = F_d->fault_one +id;

    // for input
    thisone_d->T0x   = (float *) cuda_malloc(sizeof(float)*ny*nz);  // stress_init_x
    thisone_d->T0y   = (float *) cuda_malloc(sizeof(float)*ny*nz);  // stress_init_y
    thisone_d->T0z   = (float *) cuda_malloc(sizeof(float)*ny*nz);  // stress_init_z
    thisone_d->mu_s  = (float *) cuda_malloc(sizeof(float)*ny*nz);
    thisone_d->mu_d  = (float *) cuda_malloc(sizeof(float)*ny*nz);
    thisone_d->Dc    = (float *) cuda_malloc(sizeof(float)*ny*nz);
    thisone_d->C0    = (float *) cuda_malloc(sizeof(float)*ny*nz);
    // for output
    thisone_d->output   = (float *) cuda_malloc(sizeof(float)*ny*nz*F->ncmp);
    thisone_d->Tn  = thisone_d->output + F->cmp_pos[0];
    thisone_d->Ts1 = thisone_d->output + F->cmp_pos[1];
    thisone_d->Ts2 = thisone_d->output + F->cmp_pos[2];
    thisone_d->Vs  = thisone_d->output + F->cmp_pos[3];
    thisone_d->Vs1 = thisone_d->output + F->cmp_pos[4];
    thisone_d->Vs2 = thisone_d->output + F->cmp_pos[5];
    thisone_d->Slip  = thisone_d->output + F->cmp_pos[6];
    thisone_d->Slip1 = thisone_d->output + F->cmp_pos[7];
    thisone_d->Slip2 = thisone_d->output + F->cmp_pos[8];
    thisone_d->peak_Vs = thisone_d->output + F->cmp_pos[9];
    thisone_d->init_t0 = thisone_d->output + F->cmp_pos[10];
    // for inner
    thisone_d->tTn          = (float *) cuda_malloc(sizeof(float)*ny*nz);
    thisone_d->tTs1         = (float *) cuda_malloc(sizeof(float)*ny*nz);
    thisone_d->tTs2         = (float *) cuda_malloc(sizeof(float)*ny*nz);
    thisone_d->united       = (int *)   cuda_malloc(sizeof(int)*ny*nz);
    thisone_d->faultgrid    = (int *)   cuda_malloc(sizeof(int)*ny*nz);
    thisone_d->rup_index_y  = (int *)   cuda_malloc(sizeof(int)*ny*nz);
    thisone_d->rup_index_z  = (int *)   cuda_malloc(sizeof(int)*ny*nz);
    thisone_d->flag_rup     = (int *)   cuda_malloc(sizeof(int)*ny*nz);
    thisone_d->init_t0_flag = (int *)   cuda_malloc(sizeof(int)*ny*nz);

    CUDACHECK(cudaMemcpy(thisone_d->T0x,  thisone->T0x,  sizeof(float)*ny*nz, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->T0y,  thisone->T0y,  sizeof(float)*ny*nz, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->T0z,  thisone->T0z,  sizeof(float)*ny*nz, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->mu_s, thisone->mu_s, sizeof(float)*ny*nz, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->mu_d, thisone->mu_d, sizeof(float)*ny*nz, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->Dc,   thisone->Dc,   sizeof(float)*ny*nz, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->C0,   thisone->C0,   sizeof(float)*ny*nz, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->Tn,   thisone->Tn,   sizeof(float)*ny*nz, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->Ts1,  thisone->Ts1,  sizeof(float)*ny*nz, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->Ts2,  thisone->Ts2,  sizeof(float)*ny*nz, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->init_t0, thisone->init_t0, sizeof(float)*ny*nz, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->united,       thisone->united,       sizeof(int)*ny*nz, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->faultgrid,    thisone->faultgrid,    sizeof(int)*ny*nz, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->rup_index_y,  thisone->rup_index_y,  sizeof(int)*ny*nz, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->rup_index_z,  thisone->rup_index_z,  sizeof(int)*ny*nz, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(thisone_d->flag_rup,     thisone->flag_rup,     sizeof(int)*ny*nz, cudaMemcpyHostToDevice));

    CUDACHECK(cudaMemset(thisone_d->Slip,    0, sizeof(float)*ny*nz));
    CUDACHECK(cudaMemset(thisone_d->Slip1,   0, sizeof(float)*ny*nz));
    CUDACHECK(cudaMemset(thisone_d->Slip2,   0, sizeof(float)*ny*nz));
    CUDACHECK(cudaMemset(thisone_d->Vs,      0, sizeof(float)*ny*nz));
    CUDACHECK(cudaMemset(thisone_d->Vs1,     0, sizeof(float)*ny*nz));
    CUDACHECK(cudaMemset(thisone_d->Vs2,     0, sizeof(float)*ny*nz));
    CUDACHECK(cudaMemset(thisone_d->peak_Vs, 0, sizeof(float)*ny*nz));
    CUDACHECK(cudaMemset(thisone_d->tTn,     0, sizeof(float)*ny*nz));
    CUDACHECK(cudaMemset(thisone_d->tTs1,    0, sizeof(float)*ny*nz));
    CUDACHECK(cudaMemset(thisone_d->tTs2,    0, sizeof(float)*ny*nz));
    CUDACHECK(cudaMemset(thisone_d->init_t0_flag, 0, sizeof(int)*ny*nz));
  }

  return 0;
}

int init_fault_wav_device(fault_wav_t *FW, fault_wav_t *FW_d)
{
  int ny = FW->ny;
  int nz = FW->nz;
  int nlevel = FW->nlevel;
  size_t siz_ilevel = FW->siz_ilevel;
  int number_fault = FW->number_fault;

  memcpy(FW_d,FW,sizeof(fault_wav_t));

  FW_d->fault_index = (int *) cuda_malloc(sizeof(int)*FW->number_fault);
  CUDACHECK(cudaMemcpy(FW_d->fault_index, FW->fault_index, sizeof(int)*FW->number_fault, cudaMemcpyHostToDevice));

  FW_d->v5d  = (float *) cuda_malloc(sizeof(float)*siz_ilevel*nlevel*number_fault);
  FW_d->T1x  = (float *) cuda_malloc(sizeof(float)*7*ny*nz*number_fault); 
  FW_d->T1y  = (float *) cuda_malloc(sizeof(float)*7*ny*nz*number_fault); 
  FW_d->T1z  = (float *) cuda_malloc(sizeof(float)*7*ny*nz*number_fault); 
  FW_d->hT1x = (float *) cuda_malloc(sizeof(float)*ny*nz*number_fault); 
  FW_d->hT1y = (float *) cuda_malloc(sizeof(float)*ny*nz*number_fault); 
  FW_d->hT1z = (float *) cuda_malloc(sizeof(float)*ny*nz*number_fault); 
  FW_d->mT1x = (float *) cuda_malloc(sizeof(float)*ny*nz*number_fault); 
  FW_d->mT1y = (float *) cuda_malloc(sizeof(float)*ny*nz*number_fault); 
  FW_d->mT1z = (float *) cuda_malloc(sizeof(float)*ny*nz*number_fault); 

  CUDACHECK(cudaMemset(FW_d->v5d,  0, sizeof(float)*siz_ilevel*nlevel*number_fault));
  CUDACHECK(cudaMemset(FW_d->T1x,  0, sizeof(float)*7*ny*nz*number_fault));
  CUDACHECK(cudaMemset(FW_d->T1y,  0, sizeof(float)*7*ny*nz*number_fault));
  CUDACHECK(cudaMemset(FW_d->T1z,  0, sizeof(float)*7*ny*nz*number_fault));
  CUDACHECK(cudaMemset(FW_d->hT1x, 0, sizeof(float)*ny*nz*number_fault));
  CUDACHECK(cudaMemset(FW_d->hT1y, 0, sizeof(float)*ny*nz*number_fault));
  CUDACHECK(cudaMemset(FW_d->hT1z, 0, sizeof(float)*ny*nz*number_fault));
  CUDACHECK(cudaMemset(FW_d->mT1x, 0, sizeof(float)*ny*nz*number_fault));
  CUDACHECK(cudaMemset(FW_d->mT1y, 0, sizeof(float)*ny*nz*number_fault));
  CUDACHECK(cudaMemset(FW_d->mT1z, 0, sizeof(float)*ny*nz*number_fault));

  return 0;
}

int init_bdryfree_device(gd_t *gd, bdryfree_t *bdryfree, bdryfree_t *bdryfree_d)
{
  int nx = gd->nx;
  int ny = gd->ny;

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

int init_bdrypml_device(gd_t *gd, bdrypml_t *bdrypml, bdrypml_t *bdrypml_d)
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

int init_bdryexp_device(gd_t *gd, bdryexp_t *bdryexp, bdryexp_t *bdryexp_d)
{
  int nx = gd->nx;
  int ny = gd->ny;
  int nz = gd->nz;

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

float *init_PGVAD_device(gd_t *gd)
{
  float *PG_d;
  int nx = gd->nx;
  int ny = gd->ny;
  PG_d = (float *) cuda_malloc(sizeof(float)*CONST_NDIM_5*nx*ny);
  CUDACHECK(cudaMemset(PG_d,0,sizeof(float)*CONST_NDIM_5*nx*ny));

  return PG_d;
}

float *init_Dis_accu_device(gd_t *gd)
{
  float *Dis_accu_d;
  int nx = gd->nx;
  int ny = gd->ny;
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

int dealloc_gd_device(gd_t gd_d)
{
  CUDACHECK(cudaFree(gd_d.x3d)); 
  CUDACHECK(cudaFree(gd_d.y3d)); 
  CUDACHECK(cudaFree(gd_d.z3d)); 

  CUDACHECK(cudaFree(gd_d.cell_xmin)); 
  CUDACHECK(cudaFree(gd_d.cell_xmax)); 
  CUDACHECK(cudaFree(gd_d.cell_ymin)); 
  CUDACHECK(cudaFree(gd_d.cell_ymax)); 
  CUDACHECK(cudaFree(gd_d.cell_zmin)); 
  CUDACHECK(cudaFree(gd_d.cell_zmax)); 
  CUDACHECK(cudaFree(gd_d.tile_istart));
  CUDACHECK(cudaFree(gd_d.tile_iend));  
  CUDACHECK(cudaFree(gd_d.tile_jstart));
  CUDACHECK(cudaFree(gd_d.tile_jend));  
  CUDACHECK(cudaFree(gd_d.tile_kstart));
  CUDACHECK(cudaFree(gd_d.tile_kend));  
  CUDACHECK(cudaFree(gd_d.tile_xmin));
  CUDACHECK(cudaFree(gd_d.tile_xmax));
  CUDACHECK(cudaFree(gd_d.tile_ymin));
  CUDACHECK(cudaFree(gd_d.tile_ymax));
  CUDACHECK(cudaFree(gd_d.tile_zmin));
  CUDACHECK(cudaFree(gd_d.tile_zmax));  

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
  CUDACHECK(cudaFree(FC_d.fault_index));
  for(int id=0; id<FC_d.number_fault; id++)
  {
    fault_coef_one_t *thisone = FC_d.fault_coef_one + id;

    CUDACHECK(cudaFree(thisone->rho_f));
    CUDACHECK(cudaFree(thisone->mu_f ));
    CUDACHECK(cudaFree(thisone->lam_f));

    CUDACHECK(cudaFree(thisone->D21_1));
    CUDACHECK(cudaFree(thisone->D22_1));
    CUDACHECK(cudaFree(thisone->D23_1));
    CUDACHECK(cudaFree(thisone->D31_1));
    CUDACHECK(cudaFree(thisone->D32_1));
    CUDACHECK(cudaFree(thisone->D33_1));

    CUDACHECK(cudaFree(thisone->D21_2));
    CUDACHECK(cudaFree(thisone->D22_2));
    CUDACHECK(cudaFree(thisone->D23_2));
    CUDACHECK(cudaFree(thisone->D31_2));
    CUDACHECK(cudaFree(thisone->D32_2));
    CUDACHECK(cudaFree(thisone->D33_2));

    CUDACHECK(cudaFree(thisone->vec_n ));
    CUDACHECK(cudaFree(thisone->vec_s1));
    CUDACHECK(cudaFree(thisone->vec_s2));

    CUDACHECK(cudaFree(thisone->x_et));
    CUDACHECK(cudaFree(thisone->y_et));
    CUDACHECK(cudaFree(thisone->z_et));

    CUDACHECK(cudaFree(thisone->matMin2Plus1));
    CUDACHECK(cudaFree(thisone->matMin2Plus2));
    CUDACHECK(cudaFree(thisone->matMin2Plus3));
    CUDACHECK(cudaFree(thisone->matMin2Plus4));
    CUDACHECK(cudaFree(thisone->matMin2Plus5));

    CUDACHECK(cudaFree(thisone->matPlus2Min1));
    CUDACHECK(cudaFree(thisone->matPlus2Min2));
    CUDACHECK(cudaFree(thisone->matPlus2Min3));
    CUDACHECK(cudaFree(thisone->matPlus2Min4));
    CUDACHECK(cudaFree(thisone->matPlus2Min5));

    CUDACHECK(cudaFree(thisone->matT1toVx_Min));
    CUDACHECK(cudaFree(thisone->matVytoVx_Min));
    CUDACHECK(cudaFree(thisone->matVztoVx_Min));

    CUDACHECK(cudaFree(thisone->matT1toVx_Plus));
    CUDACHECK(cudaFree(thisone->matVytoVx_Plus));
    CUDACHECK(cudaFree(thisone->matVztoVx_Plus));

    CUDACHECK(cudaFree(thisone->matVx2Vz1));
    CUDACHECK(cudaFree(thisone->matVy2Vz1));
    CUDACHECK(cudaFree(thisone->matVx2Vz2));
    CUDACHECK(cudaFree(thisone->matVy2Vz2));

    CUDACHECK(cudaFree(thisone->matPlus2Min1f));
    CUDACHECK(cudaFree(thisone->matPlus2Min2f));
    CUDACHECK(cudaFree(thisone->matPlus2Min3f));
    CUDACHECK(cudaFree(thisone->matMin2Plus1f));
    CUDACHECK(cudaFree(thisone->matMin2Plus2f));
    CUDACHECK(cudaFree(thisone->matMin2Plus3f));

    CUDACHECK(cudaFree(thisone->matT1toVxf_Min));
    CUDACHECK(cudaFree(thisone->matVytoVxf_Min));
    CUDACHECK(cudaFree(thisone->matT1toVxf_Plus));
    CUDACHECK(cudaFree(thisone->matVytoVxf_Plus));
  }

  return 0;
}

int dealloc_fault_device(fault_t F_d)
{

  CUDACHECK(cudaFree(F_d.fault_index));
  for(int id=0; id<F_d.number_fault; id++)
  {
    fault_one_t *thisone = F_d.fault_one + id;

    CUDACHECK(cudaFree(thisone->T0x));
    CUDACHECK(cudaFree(thisone->T0y));
    CUDACHECK(cudaFree(thisone->T0z));
    CUDACHECK(cudaFree(thisone->mu_s));
    CUDACHECK(cudaFree(thisone->mu_d));
    CUDACHECK(cudaFree(thisone->Dc));
    CUDACHECK(cudaFree(thisone->C0));

    // include 0-10 variable 
    CUDACHECK(cudaFree(thisone->output));

    CUDACHECK(cudaFree(thisone->united));
    CUDACHECK(cudaFree(thisone->faultgrid));
    CUDACHECK(cudaFree(thisone->rup_index_y));
    CUDACHECK(cudaFree(thisone->rup_index_z));
    CUDACHECK(cudaFree(thisone->flag_rup));
    CUDACHECK(cudaFree(thisone->init_t0_flag));
  }

  return 0;
}

int dealloc_fault_wav_device(fault_wav_t FW_d)
{

  CUDACHECK(cudaFree(FW_d.fault_index));
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
