#ifndef FT_EL_1ST_H
#define FT_EL_1ST_H

#include "gd_info.h"

/*************************************************
 * structure
 *************************************************/

/*
 * fault wavefield structure
 */

// fault wavefield variables elastic 1st eqn: vel + stress
typedef struct {
  float *v5d; // allocated var
  float *T1x;
  float *T1y;
  float *T1z;
  float *hT1x;
  float *hT1y;
  float *hT1z;
  int nx, ny, nz, ncmp, nlevel;

  size_t siz_slice_yz;
  size_t siz_slice_yz_2;
  size_t siz_ilevel;

  size_t *cmp_pos;
  char  **cmp_name;

  size_t Vx_pos;
  size_t Vy_pos;
  size_t Vz_pos;
  size_t T2x_pos;
  size_t T2y_pos;
  size_t T2z_pos;
  size_t T3x_pos;
  size_t T3y_pos;
  size_t T3z_pos;

  // sequential index 0-based
  size_t Vx_seq;
  size_t Vy_seq;
  size_t Vz_seq;
  size_t T2x_seq;
  size_t T2y_seq;
  size_t T2z_seq;
  size_t T3x_seq;
  size_t T3y_seq;
  size_t T3z_seq;

} fault_wav_t;

/*************************************************
 * function prototype
 *************************************************/

int 
fault_wav_init(gdinfo_t *gdinfo,
               wav_t *V,
               int number_of_levels);

__global__ void
fault_wav_update(size_t size, float coef, float *w_update, float *w_input1, float *w_input2);

__global__ void
fault_wav_update_end(size_t size, float coef, float *w_update, float *w_input2);

#endif

