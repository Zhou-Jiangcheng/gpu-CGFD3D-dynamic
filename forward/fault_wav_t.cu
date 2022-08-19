/*********************************************************************
 * fault wavefield for 3d elastic 1st-order equations
 **********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "constants.h"
#include "fdlib_mem.h"
#include "wav_t.h"

int 
fault_wav_init(gdinfo_t *gdinfo,
               fault_wav_t *FW,
               int number_of_levels)
{
  int ierr = 0;

  FW->ny   = gdinfo->ny;
  FW->nz   = gdinfo->nz;
  FW->ncmp = 9;
  FW->nlevel = number_of_levels;
  FW->siz_slice_yz = ny * nz;
  FW->siz_slice_yz_2 = 2 * ny * nz;
  FW->siz_ilevel = 2 * ny * nz * FW->ncmp;
  
  // i0-3 i0-2 i0-1 i0 i0+1 i0+2 i0+3
  // i0 is fault plane x index
  // this is for zhang wenqiang method
  // zhang zhenguo method only need i0-1 i0 i0+1
  FW->T11 = (float *) fdlib_mem_calloc_1d_float(7 * ny * nz,
                        0.0, "T11, ft_wav_el3d_1st");
  FW->T12 = (float *) fdlib_mem_calloc_1d_float(7 * ny * nz,
                        0.0, "T12, ft_wav_el3d_1st");
  FW->T13 = (float *) fdlib_mem_calloc_1d_float(7 * ny * nz,
                        0.0, "T13, ft_wav_el3d_1st");
  // dT/dt 1st order
  FW->hT11 = (float *) fdlib_mem_calloc_1d_float(ny * nz,
                        0.0, "hT11, ft_wav_el3d_1st");
  FW->hT12 = (float *) fdlib_mem_calloc_1d_float(ny * nz,
                        0.0, "hT12, ft_wav_el3d_1st");
  FW->hT13 = (float *) fdlib_mem_calloc_1d_float(ny * nz,
                        0.0, "hT13, ft_wav_el3d_1st");
  // vars
  // split "-" minus "+" plus 
  // Vx, Vy, Vz, T21, T22, T23, T31, T32, T33
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
   * 4-9: T21, T22, T23, T31, T32, T33
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

  sprintf(cmp_name[icmp],"%s","T21");
  FW->T21_pos = cmp_pos[icmp];
  FW->T21_seq = 3;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","T22");
  FW->T22_pos = cmp_pos[icmp];
  FW->T22_seq = 4;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","T23");
  FW->T23_pos = cmp_pos[icmp];
  FW->T23_seq = 5;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","T31");
  FW->T31_pos = cmp_pos[icmp];
  FW->T31_seq = 6;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","T32");
  FW->T32_pos = cmp_pos[icmp];
  FW->T32_seq = 7;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","T33");
  FW->T33_pos = cmp_pos[icmp];
  FW->T33_seq = 8;
  icmp += 1;

  // set pointer
  FW->cmp_pos  = cmp_pos;
  FW->cmp_name = cmp_name;

  return ierr;
}


__global__ void
failt_wav_update(size_t size, float coef, float *w_update, float *w_input1, float *w_input2)
{
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  if(ix<size){
    w_update[ix] = w_input1[ix] + coef * w_input2[ix];
  }
}

__global__ void
fault_wav_update_end(size_t size, float coef, float *w_update, float *w_input2)
{
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  if(ix<size){
    w_update[ix] = w_update[ix] + coef * w_input2[ix];
  }
}

