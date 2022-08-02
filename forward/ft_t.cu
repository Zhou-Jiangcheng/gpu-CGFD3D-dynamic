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
               fault_wav_t *V,
               int number_of_levels)
{
  int ierr = 0;

  V->nx   = gdinfo->nx;
  V->ny   = gdinfo->ny;
  V->nz   = gdinfo->nz;
  V->ncmp = 9;
  V->nlevel = number_of_levels;
  V->siz_slice_yz = ny * nz;
  
  // i0-3 i0-2 i0-1 i0 i0+1 i0+2 i0+3
  // i0 is fault plane x index
  // this is for zhang wenqaing method
  // zhang zhengguo methid only need i0-1 i0 i0+1
  V->T11 = (float *) fdlib_mem_calloc_1d_float(7 * ny * nz,
                        0.0, "T11, ft_wav_el3d_1st");
  V->T12 = (float *) fdlib_mem_calloc_1d_float(7 * ny * nz,
                        0.0, "T12, ft_wav_el3d_1st");
  V->T13 = (float *) fdlib_mem_calloc_1d_float(7 * ny * nz,
                        0.0, "T13, ft_wav_el3d_1st");
  // vars
  // split "-" minus "+" plus 
  // Vx, Vy, Vz, T21, T22, T23, T31, T32, T33
  // 4 rk stages
  V->v5d = (float *) fdlib_mem_calloc_1d_float(2 * ny * nz * V->ncmp * V->nlevel,
                        0.0, "v5d, ft_wav_el3d_1st");
  // position of each var
  size_t *cmp_pos = (size_t *) fdlib_mem_calloc_1d_sizet(
                      V->ncmp, 0, "ft_w3d_pos, ft_wav_el3d_1st");
  // name of each var
  char **cmp_name = (char **) fdlib_mem_malloc_2l_char(
                      V->ncmp, CONST_MAX_STRLEN, "ft_w3d_name, ft_wav_el3d_1st");
  
  // set value
  for (int icmp=0; icmp < V->ncmp; icmp++)
  {
    cmp_pos[icmp] = icmp * 2 * ny * nz;;
  }

  // set values
  int icmp = 0;

  /*
   * 0-3: Vx,Vy,Vz
   * 4-9: T21, T22, T23, T31, T32, T33
   */

  sprintf(cmp_name[icmp],"%s","Vx");
  V->Vx_pos = cmp_pos[icmp];
  V->Vx_seq = 0;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","Vy");
  V->Vy_pos = cmp_pos[icmp];
  V->Vy_seq = 1;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","Vz");
  V->Vz_pos = cmp_pos[icmp];
  V->Vz_seq = 2;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","T21");
  V->T21_pos = cmp_pos[icmp];
  V->T21_seq = 3;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","T22");
  V->T22_pos = cmp_pos[icmp];
  V->T22_seq = 4;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","T23");
  V->T23_pos = cmp_pos[icmp];
  V->T23_seq = 5;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","T31");
  V->T31_pos = cmp_pos[icmp];
  V->T31_seq = 6;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","T32");
  V->T32_pos = cmp_pos[icmp];
  V->T32_seq = 7;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","T33");
  V->T33_pos = cmp_pos[icmp];
  V->T33_seq = 8;
  icmp += 1;

  // set pointer
  V->cmp_pos  = cmp_pos;
  V->cmp_name = cmp_name;

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

