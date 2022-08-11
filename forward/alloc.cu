#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "alloc.h"
#include "cuda_common.h"


int init_gdinfo_device(gdinfo_t *gdinfo, gdinfo_t *gdinfo_d)
{
  memcpy(gdinfo_d,gdinfo,sizeof(gdinfo_t));
  return 0;
}

int init_gdcurv_device(gd_t *gdcurv, gd_t *gdcurv_d)
{
  size_t siz_volume = gdcurv->siz_volume;
  memcpy(gdcurv_d,gdcurv,sizeof(gd_t));
  gdcurv_d->x3d = (float *) cuda_malloc(sizeof(float)*siz_volume);
  gdcurv_d->y3d = (float *) cuda_malloc(sizeof(float)*siz_volume);
  gdcurv_d->z3d = (float *) cuda_malloc(sizeof(float)*siz_volume);

  gdcurv_d->cell_xmin = (float *) cuda_malloc(sizeof(float)*siz_volume);
  gdcurv_d->cell_xmax = (float *) cuda_malloc(sizeof(float)*siz_volume);
  gdcurv_d->cell_ymin = (float *) cuda_malloc(sizeof(float)*siz_volume);
  gdcurv_d->cell_ymax = (float *) cuda_malloc(sizeof(float)*siz_volume);
  gdcurv_d->cell_zmin = (float *) cuda_malloc(sizeof(float)*siz_volume);
  gdcurv_d->cell_zmax = (float *) cuda_malloc(sizeof(float)*siz_volume);

  CUDACHECK(cudaMemcpy(gdcurv_d->x3d, gdcurv->x3d, sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->y3d, gdcurv->y3d, sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->z3d, gdcurv->z3d, sizeof(float)*siz_volume, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(gdcurv_d->cell_xmin, gdcurv->cell_xmin, sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->cell_xmax, gdcurv->cell_xmax, sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->cell_ymin, gdcurv->cell_ymin, sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->cell_ymax, gdcurv->cell_ymax, sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->cell_zmin, gdcurv->cell_zmin, sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->cell_zmax, gdcurv->cell_zmax, sizeof(float)*siz_volume, cudaMemcpyHostToDevice));

  return 0;
}
int init_md_device(md_t *md, md_t *md_d)
{
  size_t siz_volume = md->siz_volume;

  memcpy(md_d,md,sizeof(md_t));
  if (md->medium_type == CONST_MEDIUM_ELASTIC_ISO)
  {
    md_d->rho    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->lambda = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->mu     = (float *) cuda_malloc(sizeof(float)*siz_volume);
    CUDACHECK(cudaMemcpy(md_d->rho,    md->rho,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->lambda, md->lambda, sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->mu,     md->mu,     sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
  }
  if (md->medium_type == CONST_MEDIUM_ELASTIC_VTI)
  {
    md_d->rho    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c11    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c33    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c55    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c66    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c13    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    CUDACHECK(cudaMemcpy(md_d->rho,    md->rho,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c11,    md->c11,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c33,    md->c33,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c55,    md->c55,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c66,    md->c66,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c13,    md->c13,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
  }
  if (md->medium_type == CONST_MEDIUM_ELASTIC_ANISO)
  {
    md_d->rho    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c11    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c12    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c13    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c14    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c15    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c16    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c22    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c23    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c24    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c25    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c26    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c33    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c34    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c35    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c36    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c44    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c45    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c46    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c55    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c56    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    md_d->c66    = (float *) cuda_malloc(sizeof(float)*siz_volume);
    CUDACHECK(cudaMemcpy(md_d->rho,    md->rho,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c11,    md->c11,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c12,    md->c12,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c13,    md->c13,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c14,    md->c14,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c15,    md->c15,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c16,    md->c16,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c22,    md->c22,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c23,    md->c23,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c24,    md->c24,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c25,    md->c25,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c26,    md->c26,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c33,    md->c33,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c34,    md->c34,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c35,    md->c35,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c36,    md->c36,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c44,    md->c44,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c45,    md->c45,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c46,    md->c46,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c55,    md->c55,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c56,    md->c56,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c66,    md->c66,    sizeof(float)*siz_volume, cudaMemcpyHostToDevice));
  }

  return 0;
}

int init_metric_device(gdcurv_metric_t *metric, gdcurv_metric_t *metric_d)
{
  size_t siz_volume = metric->siz_volume;

  memcpy(metric_d,metric,sizeof(gdcurv_metric_t));
  metric_d->jac     = (float *) cuda_malloc(sizeof(float)*siz_volume);
  metric_d->xi_x    = (float *) cuda_malloc(sizeof(float)*siz_volume);
  metric_d->xi_y    = (float *) cuda_malloc(sizeof(float)*siz_volume);
  metric_d->xi_z    = (float *) cuda_malloc(sizeof(float)*siz_volume);
  metric_d->eta_x   = (float *) cuda_malloc(sizeof(float)*siz_volume);
  metric_d->eta_y   = (float *) cuda_malloc(sizeof(float)*siz_volume);
  metric_d->eta_z   = (float *) cuda_malloc(sizeof(float)*siz_volume);
  metric_d->zeta_x   = (float *) cuda_malloc(sizeof(float)*siz_volume);
  metric_d->zeta_y   = (float *) cuda_malloc(sizeof(float)*siz_volume);
  metric_d->zeta_z   = (float *) cuda_malloc(sizeof(float)*siz_volume);

  CUDACHECK( cudaMemcpy(metric_d->jac,   metric->jac,   sizeof(float)*siz_volume, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(metric_d->xi_x,  metric->xi_x,  sizeof(float)*siz_volume, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(metric_d->xi_y,  metric->xi_y,  sizeof(float)*siz_volume, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(metric_d->xi_z,  metric->xi_z,  sizeof(float)*siz_volume, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(metric_d->eta_x, metric->eta_x, sizeof(float)*siz_volume, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(metric_d->eta_y, metric->eta_y, sizeof(float)*siz_volume, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(metric_d->eta_z, metric->eta_z, sizeof(float)*siz_volume, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(metric_d->zeta_x, metric->zeta_x, sizeof(float)*siz_volume, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(metric_d->zeta_y, metric->zeta_y, sizeof(float)*siz_volume, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(metric_d->zeta_z, metric->zeta_z, sizeof(float)*siz_volume, cudaMemcpyHostToDevice) );
  return 0;
}

int init_fault_coef_device(fault_coef_t *fault_coef, fault_coef_t *fault_coef_d)
{
  memcpy(fault_d,fault,sizeof(fault_t));
  if(method == 1) //zhangzhengguo
  {


  }
  if(method == 2) //zhangwenqiang
  {


  }




  return 0;
}

int init_bdryfree_device(gdinfo_t *gdinfo, bdryfree_t *bdryfree, bdryfree_t *bdryfree_d)
{
  int nx = gdinfo->nx;
  int ny = gdinfo->ny;

  memcpy(bdryfree_d,bdryfree,sizeof(bdryfree_t));
  
  if (bdryfree->is_at_sides[CONST_NDIM-1][1] == 1)
  {
    bdryfree_d->matVx2Vz2   = (float *) cuda_malloc(sizeof(float)*nx*ny*CONST_NDIM*CONST_NDIM);
    bdryfree_d->matVy2Vz2   = (float *) cuda_malloc(sizeof(float)*nx*ny*CONST_NDIM*CONST_NDIM);

    CUDACHECK(cudaMemcpy(bdryfree_d->matVx2Vz2, bdryfree->matVx2Vz2, sizeof(float)*nx*ny*CONST_NDIM*CONST_NDIM, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(bdryfree_d->matVy2Vz2, bdryfree->matVy2Vz2, sizeof(float)*nx*ny*CONST_NDIM*CONST_NDIM, cudaMemcpyHostToDevice));
  }

  return 0;
}

int init_bdrypml_device(gdinfo_t *gdinfo, bdrypml_t *bdrypml, bdrypml_t *bdrypml_d)
{
  memcpy(bdrypml_d,bdrypml,sizeof(bdrypml_t));
  for(int idim=0; idim<CONST_NDIM; idim++){
    for(int iside=0; iside<2; iside++){
      if(bdrypml_d->is_at_sides[idim][iside] == 1){
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
      if(auxvar_d->siz_volume > 0){
        auxvar_d->var = (float *) cuda_malloc(sizeof(float)*auxvar_d->siz_ilevel*auxvar_d->nlevel); 
        CUDACHECK(cudaMemset(auxvar_d->var,0,sizeof(float)*auxvar_d->siz_ilevel*auxvar_d->nlevel));
      } else {
      auxvar_d->var = NULL;
      }
    }
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

float *init_PGVAD_device(gdinfo_t *gdinfo)
{
  float *PG_d;
  int nx = gdinfo->nx;
  int ny = gdinfo->ny;
  PG_d = (float *) cuda_malloc(sizeof(float)*CONST_NDIM_5*nx*ny);
  CUDACHECK(cudaMemset(PG_d,0,sizeof(float)*CONST_NDIM_5*nx*ny));

  return PG_d;
}

float *init_Dis_accu_device(gdinfo_t *gdinfo)
{
  float *Dis_accu_d;
  int nx = gdinfo->nx;
  int ny = gdinfo->ny;
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

int dealloc_gdcurv_device(gd_t gdcurv_d)
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

int dealloc_fd_device(fd_wav_t fd_wav_d)
{
  CUDACHECK(cudaFree(fd_wav_d.fdx_coef_d));
  CUDACHECK(cudaFree(fd_wav_d.fdy_coef_d));
  CUDACHECK(cudaFree(fd_wav_d.fdz_coef_d));
  CUDACHECK(cudaFree(fd_wav_d.fdz_coef_all_d));

  CUDACHECK(cudaFree(fd_wav_d.fdx_indx_d));
  CUDACHECK(cudaFree(fd_wav_d.fdy_indx_d));
  CUDACHECK(cudaFree(fd_wav_d.fdz_indx_d));
  CUDACHECK(cudaFree(fd_wav_d.fdz_indx_all_d));

  CUDACHECK(cudaFree(fd_wav_d.fdx_shift_d));
  CUDACHECK(cudaFree(fd_wav_d.fdy_shift_d));
  CUDACHECK(cudaFree(fd_wav_d.fdz_shift_d));
  CUDACHECK(cudaFree(fd_wav_d.fdz_shift_all_d));

  return 0;
}
int dealloc_metric_device(gdcurv_metric_t metric_d)
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

int dealloc_bdryfree_device(bdryfree_t bdryfree_d)
{
  if (bdryfree_d.is_at_sides[CONST_NDIM-1][1] == 1)
  {
    CUDACHECK(cudaFree(bdryfree_d.matVx2Vz2)); 
    CUDACHECK(cudaFree(bdryfree_d.matVy2Vz2)); 
  }
  return 0;
}

int dealloc_bdrypml_device(bdrypml_t bdrypml_d)
{
  for(int idim=0; idim<CONST_NDIM; idim++){
    for(int iside=0; iside<2; iside++){
      if(bdrypml_d.is_at_sides[idim][iside] == 1){
        CUDACHECK(cudaFree(bdrypml_d.A[idim][iside])); 
        CUDACHECK(cudaFree(bdrypml_d.B[idim][iside])); 
        CUDACHECK(cudaFree(bdrypml_d.D[idim][iside])); 
      }
    }
  }  
  for(int idim=0; idim<CONST_NDIM; idim++){
    for(int iside=0; iside<2; iside++){
      bdrypml_auxvar_t *auxvar_d = &(bdrypml_d.auxvar[idim][iside]);
      if(auxvar_d->siz_volume > 0){
        CUDACHECK(cudaFree(auxvar_d->var)); 
      }
    }
  }  
  return 0;
}

int dealloc_wave_device(wav_t wav_d)
{
  CUDACHECK(cudaFree(wav_d.v5d)); 
  return 0;
}

