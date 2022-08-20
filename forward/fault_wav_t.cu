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

