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
  float *T11;
  float *T12;
  float *T13;
  int nx, ny, nz, ncmp, nlevel;

  size_t siz_slice_yz;
  size_t siz_slice_yz_2;

  size_t *cmp_pos;
  char  **cmp_name;

  size_t Vx_pos;
  size_t Vy_pos;
  size_t Vz_pos;
  size_t T21_pos;
  size_t T22_pos;
  size_t T23_pos;
  size_t T31_pos;
  size_t T32_pos;
  size_t T33_pos;

  // sequential index 0-based
  size_t Vx_seq;
  size_t Vy_seq;
  size_t Vz_seq;
  size_t T21_seq;
  size_t T22_seq;
  size_t T23_seq;
  size_t T31_seq;
  size_t T32_seq;
  size_t T33_seq;

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

