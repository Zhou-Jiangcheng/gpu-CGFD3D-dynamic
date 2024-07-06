/*********************************************************************
 * fault wavefield for 3d elastic 1st-order equations
 **********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "constants.h"
#include "fdlib_mem.h"
#include "fault_wav_t.h"

int 
fault_wav_init(gdcurv_t *gdcurv,
               fault_wav_t *FW,
               int number_of_levels)
{
  int ierr = 0;
  int ny = gdcurv->ny;
  int nz = gdcurv->nz;
  FW->ny   = gdcurv->ny;
  FW->nz   = gdcurv->nz;
  FW->ncmp = 9;
  FW->nlevel = number_of_levels;
  FW->siz_slice_yz = ny * nz;
  FW->siz_slice_yz_2 = 2 * ny * nz;
  FW->siz_ilevel = 2 * ny * nz * FW->ncmp;

  // NOTE ! 
  // for gpu code, this fault wave in CPU is not necessary alloc
  
  // i0-3 i0-2 i0-1 i0 i0+1 i0+2 i0+3
  // i0 is fault plane x index
  // this is for zhang wenqiang method
  // zhang zhenguo method only need i0-1 i0 i0+1
  
  FW->T1x = (float *) fdlib_mem_calloc_1d_float(7 * ny * nz,
                        0.0, "T1x, ft_wav_el3d_1st");
  FW->T1y = (float *) fdlib_mem_calloc_1d_float(7 * ny * nz,
                        0.0, "T1y, ft_wav_el3d_1st");
  FW->T1z = (float *) fdlib_mem_calloc_1d_float(7 * ny * nz,
                        0.0, "T1z, ft_wav_el3d_1st");
  // dT/dt 1st order
  FW->hT1x = (float *) fdlib_mem_calloc_1d_float(ny * nz,
                        0.0, "hT1x, ft_wav_el3d_1st");
  FW->hT1y = (float *) fdlib_mem_calloc_1d_float(ny * nz,
                        0.0, "hT1y, ft_wav_el3d_1st");
  FW->hT1z = (float *) fdlib_mem_calloc_1d_float(ny * nz,
                        0.0, "hT1z, ft_wav_el3d_1st");

  FW->mT1x = (float *) fdlib_mem_calloc_1d_float(ny * nz,
                        0.0, "mT1x, ft_wav_el3d_1st");
  FW->mT1y = (float *) fdlib_mem_calloc_1d_float(ny * nz,
                        0.0, "mT1y, ft_wav_el3d_1st");
  FW->mT1z = (float *) fdlib_mem_calloc_1d_float(ny * nz,
                        0.0, "mT1z, ft_wav_el3d_1st");
  // vars
  // split "-" minus "+" plus 
  // Vx, Vy, Vz, T2x, T2y, T2z, T3x, T3y, T3z
  // 4 rk stages
  FW->v5d = (float *) fdlib_mem_calloc_1d_float(FW->siz_ilevel * FW->nlevel,
                        0.0, "v5d, ft_wav_el3d_1st");
  // position of each var
  size_t *cmp_pos = (size_t *) fdlib_mem_calloc_1d_sizet(
                      FW->ncmp, 0, "ft_w3d_pos, ft_wav_el3d_1st");
  // name of each var
  char **cmp_name = (char **) fdlib_mem_malloc_2l_char(
                      FW->ncmp, CONST_MAX_STRLEN, "ft_w3d_name, ft_wav_el3d_1st");
  
  // set value
  for (int icmp=0; icmp < FW->ncmp; icmp++)
  {
    cmp_pos[icmp] = icmp * 2 * ny * nz;
  }

  // set values
  int icmp = 0;

  /*
   * 0-3: Vx,Vy,Vz
   * 4-9: T2x, T2y, T2z, T3x, T3y, T3z
   */

  sprintf(cmp_name[icmp],"%s","Vx");
  FW->Vx_pos = cmp_pos[icmp];
  FW->Vx_seq = 0;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","Vy");
  FW->Vy_pos = cmp_pos[icmp];
  FW->Vy_seq = 1;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","Vz");
  FW->Vz_pos = cmp_pos[icmp];
  FW->Vz_seq = 2;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","T2x");
  FW->T2x_pos = cmp_pos[icmp];
  FW->T2x_seq = 3;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","T2y");
  FW->T2y_pos = cmp_pos[icmp];
  FW->T2y_seq = 4;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","T2z");
  FW->T2z_pos = cmp_pos[icmp];
  FW->T2z_seq = 5;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","T3x");
  FW->T3x_pos = cmp_pos[icmp];
  FW->T3x_seq = 6;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","T3y");
  FW->T3y_pos = cmp_pos[icmp];
  FW->T3y_seq = 7;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","T3z");
  FW->T3z_pos = cmp_pos[icmp];
  FW->T3z_seq = 8;
  icmp += 1;

  // set pointer
  FW->cmp_pos  = cmp_pos;
  FW->cmp_name = cmp_name;

  return ierr;
}

int 
fault_var_update(float *f_end_d, int it, float dt,  
                 gdcurv_t gdcurv_d, fault_t F, 
                 fault_coef_t FC, fault_wav_t FW)
{
  int nj  = gdcurv_d.nj;
  int nj1 = gdcurv_d.nj1;
  int nk  = gdcurv_d.nk;
  int nk1 = gdcurv_d.nk1;
  int ny  = gdcurv_d.ny;
  size_t siz_slice_yz  = gdcurv_d.siz_slice_yz;
  float *f_Vx = f_end_d + FW.Vx_pos; 
  float *f_Vy = f_end_d + FW.Vy_pos; 
  float *f_Vz = f_end_d + FW.Vz_pos; 
  {
    dim3 block(8,8);
    dim3 grid;
    grid.x = (nj + block.x - 1) / block.x;
    grid.y = (nk + block.y - 1) / block.y;
    fault_var_update_gpu<<<grid, block >>> ( f_Vx, f_Vy, f_Vz, 
                                             nj, nj1, nk, nk1, ny, 
                                             siz_slice_yz, it, dt, FC, F, FW);
  }
  return 0;
}

__global__ void
fault_var_update_gpu(float *f_Vx,float *f_Vy, float *f_Vz, 
                     int nj, int nj1, int nk, int nk1, 
                     int ny, size_t siz_slice_yz,
                     int it, float dt, fault_coef_t FC,  
                     fault_t F, fault_wav_t FW)
{
  size_t iy = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iz = blockIdx.y * blockDim.y + threadIdx.y;

  size_t iptr_t = iy + iz*nj;
  size_t iptr_f = (iy+nj1) + (iz+nk1)*ny;

  float Vs1, Vs2, Vs;
  float dVx, dVy, dVz;
  float vec_s1[3], vec_s2[3];

  if(iy<nj && iz<nk && F.united[iptr_t] == 0)
  {
    dVx = f_Vx[iptr_f + siz_slice_yz] - f_Vx[iptr_f];
    dVy = f_Vy[iptr_f + siz_slice_yz] - f_Vy[iptr_f];
    dVz = f_Vz[iptr_f + siz_slice_yz] - f_Vz[iptr_f];

    vec_s1[0] = FC.vec_s1[iptr_f*3 + 0];
    vec_s1[1] = FC.vec_s1[iptr_f*3 + 1];
    vec_s1[2] = FC.vec_s1[iptr_f*3 + 2];
    vec_s2[0] = FC.vec_s2[iptr_f*3 + 0];
    vec_s2[1] = FC.vec_s2[iptr_f*3 + 1];
    vec_s2[2] = FC.vec_s2[iptr_f*3 + 2];

    Vs1 = dVx * vec_s1[0] + dVy * vec_s1[1] + dVz * vec_s1[2];
    Vs2 = dVx * vec_s2[0] + dVy * vec_s2[1] + dVz * vec_s2[2];
    Vs  = sqrt(Vs2*Vs2 + Vs1*Vs1);
    F.Vs [iptr_t] = Vs;
    F.Vs1[iptr_t] = Vs1;
    F.Vs2[iptr_t] = Vs2; 
    F.slip[iptr_t]  += Vs  * dt; 
    F.slip1[iptr_t] += Vs1 * dt; 
    F.slip2[iptr_t] += Vs2 * dt; 

    FW.hT1x[iptr_f] = (FW.T1x[iptr_f+3*siz_slice_yz] - FW.mT1x[iptr_f])/dt;
    FW.hT1y[iptr_f] = (FW.T1y[iptr_f+3*siz_slice_yz] - FW.mT1y[iptr_f])/dt;
    FW.hT1z[iptr_f] = (FW.T1z[iptr_f+3*siz_slice_yz] - FW.mT1z[iptr_f])/dt;

    FW.mT1x[iptr_f] = FW.T1x[iptr_f+3*siz_slice_yz];
    FW.mT1y[iptr_f] = FW.T1y[iptr_f+3*siz_slice_yz];
    FW.mT1z[iptr_f] = FW.T1z[iptr_f+3*siz_slice_yz];

    if(Vs > F.peak_Vs[iptr_t]) 
    {
      F.peak_Vs[iptr_t] = Vs;
    }

    if(F.init_t0_flag[iptr_t] == 0) {
      if (Vs > 1e-3) {
        F.init_t0[iptr_t] = it * dt;;
        F.init_t0_flag[iptr_t] = 1;
        F.flag_rup[iptr_t] = 1;
      }
    }
  }
  return;
}

__global__ void
fault_stress_update_first(int nj, int nk, float coef, fault_t F)
{
  size_t iy = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iz = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iptr_t = iy + iz*nj;
  if(iy<nj && iz<nk && F.united[iptr_t]==0){
    F.Tn [iptr_t] = coef * F.tTn [iptr_t];
    F.Ts1[iptr_t] = coef * F.tTs1[iptr_t];
    F.Ts2[iptr_t] = coef * F.tTs2[iptr_t];
  }
  return;
}

__global__ void
fault_stress_update(int nj, int nk, float coef, fault_t F)
{
  size_t iy = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iz = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iptr_t = iy + iz*nj;
  if(iy<nj && iz<nk && F.united[iptr_t]==0){
    F.Tn [iptr_t] += coef * F.tTn [iptr_t];
    F.Ts1[iptr_t] += coef * F.tTs1[iptr_t];
    F.Ts2[iptr_t] += coef * F.tTs2[iptr_t];
  }
  return;
}

__global__ void
fault_wav_update(gdcurv_t gdcurv_d, int num_of_vars, 
                 float coef, fault_t F,
                 float *w_update, float *w_input1, float *w_input2)
{
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iz = blockIdx.z * blockDim.z + threadIdx.z;

  int nj = gdcurv_d.nj;
  int nk = gdcurv_d.nk;
  int nj1 = gdcurv_d.nj1;
  int nk1 = gdcurv_d.nk1;
  int ny = gdcurv_d.ny;
  size_t siz_slice_yz = gdcurv_d.siz_slice_yz;

  size_t iptr_t = iy+iz*nj;
  size_t iptr_f;
  if(ix < 2*num_of_vars && iy < nj && iz < nk && F.united[iptr_t]==0)
  {
    iptr_f = (iy+nj1) + (iz+nk1) * ny + ix * siz_slice_yz;
    w_update[iptr_f] = w_input1[iptr_f] + coef * w_input2[iptr_f];
  }
}

__global__ void
fault_wav_update_end(gdcurv_t gdcurv_d, int num_of_vars, 
                     float coef, fault_t F,
                     float *w_update, float *w_input2)
{
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iz = blockIdx.z * blockDim.z + threadIdx.z;
      
  int nj = gdcurv_d.nj;
  int nk = gdcurv_d.nk;
  int nj1 = gdcurv_d.nj1;
  int nk1 = gdcurv_d.nk1;
  int ny = gdcurv_d.ny;
  size_t siz_slice_yz = gdcurv_d.siz_slice_yz;

  size_t iptr_t = iy+iz*nj;
  size_t iptr_f;
  if(ix < 2*num_of_vars && iy < nj && iz < nk && F.united[iptr_t]==0)
  {
    iptr_f = (iy+nj1) + (iz+nk1) * ny + ix * siz_slice_yz;
    w_update[iptr_f] += coef * w_input2[iptr_f];
  }
}
