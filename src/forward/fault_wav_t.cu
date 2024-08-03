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
fault_wav_init(gd_t *gd,
               fault_wav_t *FW,
               int number_fault,
               int *fault_x_index,
               int number_of_levels)
{
  int ierr = 0;
  int ny = gd->ny;
  int nz = gd->nz;

  FW->ny   = gd->ny;
  FW->nz   = gd->nz;
  FW->ncmp = 9;
  FW->nlevel = number_of_levels;
  FW->siz_slice_yz = ny * nz;
  FW->siz_slice_yz_2 = 2 * ny * nz;
  FW->siz_ilevel = 2 * ny * nz * FW->ncmp;

  FW->number_fault = number_fault;

  FW->fault_index = (int *) malloc(sizeof(int)*number_fault);

  for(int id=0; id<number_fault; id++)
  {
    FW->fault_index[id] = fault_x_index[id];
  }

  // NOTE ! 
  // for gpu code, this fault wave in CPU is not necessary alloc
  // but we alloc
  
  // vars
  // split "-" minus "+" plus 
  // Vx, Vy, Vz, T2x, T2y, T2z, T3x, T3y, T3z
  // 4 rk stages
  FW->v5d = (float *) fdlib_mem_calloc_1d_float(FW->siz_ilevel * FW->nlevel * number_fault,
                        0.0, "v5d, ft_wav_el3d_1st");

  
  // i0-3 i0-2 i0-1 i0 i0+1 i0+2 i0+3
  // i0 is fault plane x index
  // this is for zhang wenqiang method
  // zhang zhenguo method only need i0-1 i0 i0+1

  FW->T1x = (float *) fdlib_mem_calloc_1d_float(7 * ny * nz * number_fault,
                        0.0, "T1x, ft_wav_el3d_1st");
  FW->T1y = (float *) fdlib_mem_calloc_1d_float(7 * ny * nz * number_fault,
                        0.0, "T1y, ft_wav_el3d_1st");
  FW->T1z = (float *) fdlib_mem_calloc_1d_float(7 * ny * nz * number_fault,
                        0.0, "T1z, ft_wav_el3d_1st");
  // dT/dt 1st order
  FW->hT1x = (float *) fdlib_mem_calloc_1d_float(ny * nz * number_fault,
                        0.0, "hT1x, ft_wav_el3d_1st");
  FW->hT1y = (float *) fdlib_mem_calloc_1d_float(ny * nz * number_fault,
                        0.0, "hT1y, ft_wav_el3d_1st");
  FW->hT1z = (float *) fdlib_mem_calloc_1d_float(ny * nz * number_fault,
                        0.0, "hT1z, ft_wav_el3d_1st");

  FW->mT1x = (float *) fdlib_mem_calloc_1d_float(ny * nz * number_fault,
                        0.0, "mT1x, ft_wav_el3d_1st");
  FW->mT1y = (float *) fdlib_mem_calloc_1d_float(ny * nz * number_fault,
                        0.0, "mT1y, ft_wav_el3d_1st");
  FW->mT1z = (float *) fdlib_mem_calloc_1d_float(ny * nz * number_fault,
                        0.0, "mT1z, ft_wav_el3d_1st");
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
                 gd_t gd_d, fault_t F, 
                 fault_coef_t FC, fault_wav_t FW)
{
  int nj  = gd_d.nj;
  int nj1 = gd_d.nj1;
  int nk  = gd_d.nk;
  int nk1 = gd_d.nk1;
  int ny  = gd_d.ny;
  size_t siz_slice_yz  = gd_d.siz_slice_yz;

  for(int id=0; id<FW.number_fault; id++)
  {
    float *f_T1x  = FW.T1x + id*7*siz_slice_yz;
    float *f_T1y  = FW.T1y + id*7*siz_slice_yz;
    float *f_T1z  = FW.T1z + id*7*siz_slice_yz;
    float *f_hT1x = FW.hT1x + id*siz_slice_yz;
    float *f_hT1y = FW.hT1y + id*siz_slice_yz;
    float *f_hT1z = FW.hT1z + id*siz_slice_yz;
    float *f_mT1x = FW.mT1x + id*siz_slice_yz;
    float *f_mT1y = FW.mT1y + id*siz_slice_yz;
    float *f_mT1z = FW.mT1z + id*siz_slice_yz;

    float *f_end_thisone = f_end_d + id*FW.siz_ilevel; 
    float *f_Vx = f_end_thisone + FW.Vx_pos; 
    float *f_Vy = f_end_thisone + FW.Vy_pos; 
    float *f_Vz = f_end_thisone + FW.Vz_pos; 
    {
      dim3 block(8,8);
      dim3 grid;
      grid.x = (nj + block.x - 1) / block.x;
      grid.y = (nk + block.y - 1) / block.y;
      fault_var_update_gpu<<<grid, block >>> ( f_Vx, f_Vy, f_Vz, 
                                               f_T1x, f_T1y, f_T1z,
                                               f_hT1x, f_hT1y, f_hT1z,
                                               f_mT1x, f_mT1y, f_mT1z,
                                               nj, nj1, nk, nk1, ny, 
                                               siz_slice_yz, it, dt, id, F, FC);
    }
  }

  return 0;
}

__global__ void
fault_var_update_gpu(float *f_Vx,float *f_Vy, float *f_Vz, 
                     float *f_T1x,float *f_T1y, float *f_T1z,
                     float *f_hT1x,float *f_hT1y, float *f_hT1z,
                     float *f_mT1x,float *f_mT1y, float *f_mT1z,
                     int nj, int nj1, int nk, int nk1, 
                     int ny, size_t siz_slice_yz,
                     int it, float dt, int id, 
                     fault_t F,  fault_coef_t FC)
{
  size_t iy = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iz = blockIdx.y * blockDim.y + threadIdx.y;

  fault_one_t *F_thisone = F.fault_one + id;
  fault_coef_one_t *FC_thisone = FC.fault_coef_one + id;

  size_t iptr_f = (iy+nj1) + (iz+nk1)*ny;

  float Vs1, Vs2, Vs;
  float dVx, dVy, dVz;
  float vec_s1[3], vec_s2[3];

  if(iy<nj && iz<nk && F_thisone->united[iptr_f] == 0)
  {
    dVx = f_Vx[iptr_f + siz_slice_yz] - f_Vx[iptr_f];
    dVy = f_Vy[iptr_f + siz_slice_yz] - f_Vy[iptr_f];
    dVz = f_Vz[iptr_f + siz_slice_yz] - f_Vz[iptr_f];

    vec_s1[0] = FC_thisone->vec_s1[iptr_f*3 + 0];
    vec_s1[1] = FC_thisone->vec_s1[iptr_f*3 + 1];
    vec_s1[2] = FC_thisone->vec_s1[iptr_f*3 + 2];
    vec_s2[0] = FC_thisone->vec_s2[iptr_f*3 + 0];
    vec_s2[1] = FC_thisone->vec_s2[iptr_f*3 + 1];
    vec_s2[2] = FC_thisone->vec_s2[iptr_f*3 + 2];

    Vs1 = dVx * vec_s1[0] + dVy * vec_s1[1] + dVz * vec_s1[2];
    Vs2 = dVx * vec_s2[0] + dVy * vec_s2[1] + dVz * vec_s2[2];
    Vs  = sqrt(Vs2*Vs2 + Vs1*Vs1);
    F_thisone->Vs [iptr_f] = Vs;
    F_thisone->Vs1[iptr_f] = Vs1;
    F_thisone->Vs2[iptr_f] = Vs2; 
    F_thisone->Slip[iptr_f]  += Vs  * dt; 
    F_thisone->Slip1[iptr_f] += Vs1 * dt; 
    F_thisone->Slip2[iptr_f] += Vs2 * dt; 

    f_hT1x[iptr_f] = (f_T1x[iptr_f+3*siz_slice_yz] - f_mT1x[iptr_f])/dt;
    f_hT1y[iptr_f] = (f_T1y[iptr_f+3*siz_slice_yz] - f_mT1y[iptr_f])/dt;
    f_hT1z[iptr_f] = (f_T1z[iptr_f+3*siz_slice_yz] - f_mT1z[iptr_f])/dt;

    f_mT1x[iptr_f] = f_T1x[iptr_f+3*siz_slice_yz];
    f_mT1y[iptr_f] = f_T1y[iptr_f+3*siz_slice_yz];
    f_mT1z[iptr_f] = f_T1z[iptr_f+3*siz_slice_yz];

    if(Vs > F_thisone->Peak_vs[iptr_f]) 
    {
      F_thisone->Peak_vs[iptr_f] = Vs;
    }

    if(F_thisone->init_t0_flag[iptr_f] == 0) {
      if (Vs > 1e-3) {
        F_thisone->Init_t0[iptr_f] = (it+1) * dt;;
        F_thisone->init_t0_flag[iptr_f] = 1;
        F_thisone->flag_rup[iptr_f] = 1;
      }
    }
  }
  return;
}

int
fault_var_stage_update(float coef, int istage, gd_t gd_d, fault_t F)
{
  int nj  = gd_d.nj;
  int nj1 = gd_d.nj1;
  int nk  = gd_d.nk;
  int nk1 = gd_d.nk1;
  int ny  = gd_d.ny;
  for(int id=0; id<F.number_fault; id++)
  {
    {
      dim3 block(8,8);
      dim3 grid;
      grid.x = (nj + block.x - 1) / block.x;
      grid.y = (nk + block.y - 1) / block.y;
      fault_var_stage_update_gpu <<<grid, block >>>(
                                 nj, nj1, nk, nk1, ny, coef, istage,  
                                 id, F);
    }
  }
  
  return 0;
}

__global__ void
fault_var_stage_update_gpu(int nj, int nj1, int nk, int nk1, int ny,
                           float coef, int istage, int id, fault_t F)
{
  size_t iy = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iz = blockIdx.y * blockDim.y + threadIdx.y;

  fault_one_t *F_thisone = F.fault_one + id;

  size_t iptr_f = (iy+nj1) + (iz+nk1)*ny;

  if(iy<nj && iz<nk && F_thisone->united[iptr_f]==0)
  {
    if(istage == 0)
    {
      F_thisone->Tn [iptr_f] = coef * F_thisone->tTn [iptr_f];
      F_thisone->Ts1[iptr_f] = coef * F_thisone->tTs1[iptr_f];
      F_thisone->Ts2[iptr_f] = coef * F_thisone->tTs2[iptr_f];
    } else
    {
      F_thisone->Tn [iptr_f] += coef * F_thisone->tTn [iptr_f];
      F_thisone->Ts1[iptr_f] += coef * F_thisone->tTs1[iptr_f];
      F_thisone->Ts2[iptr_f] += coef * F_thisone->tTs2[iptr_f];
    }
  }
  return;
}

__global__ void
fault_wav_update(gd_t gd_d, int num_of_vars, 
                 float coef, int id, fault_t F,
                 float *w_update, float *w_input1, float *w_input2)
{
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iz = blockIdx.z * blockDim.z + threadIdx.z;

  fault_one_t *F_thisone = F.fault_one + id;
  int nj = gd_d.nj;
  int nk = gd_d.nk;
  int nj1 = gd_d.nj1;
  int nk1 = gd_d.nk1;
  int ny = gd_d.ny;
  size_t siz_slice_yz = gd_d.siz_slice_yz;

  size_t iptr_f = (iy+nj1) + (iz+nk1) * ny;
  if(ix < 2*num_of_vars && iy < nj && iz < nk && F_thisone->united[iptr_f]==0)
  {
    iptr_f = (iy+nj1) + (iz+nk1) * ny + ix * siz_slice_yz;
    w_update[iptr_f] = w_input1[iptr_f] + coef * w_input2[iptr_f];
  }
}

__global__ void
fault_wav_update_end(gd_t gd_d, int num_of_vars, 
                     float coef, int id, fault_t F,
                     float *w_update, float *w_input2)
{
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iz = blockIdx.z * blockDim.z + threadIdx.z;
  fault_one_t *F_thisone = F.fault_one + id;
      
  int nj = gd_d.nj;
  int nk = gd_d.nk;
  int nj1 = gd_d.nj1;
  int nk1 = gd_d.nk1;
  int ny = gd_d.ny;
  size_t siz_slice_yz = gd_d.siz_slice_yz;

  size_t iptr_f = (iy+nj1) + (iz+nk1) * ny;
  if(ix < 2*num_of_vars && iy < nj && iz < nk && F_thisone->united[iptr_f]==0)
  {
    iptr_f = (iy+nj1) + (iz+nk1) * ny + ix * siz_slice_yz;
    w_update[iptr_f] += coef * w_input2[iptr_f];
  }
}

