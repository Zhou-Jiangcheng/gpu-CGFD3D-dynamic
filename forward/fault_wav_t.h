#ifndef FT_EL_1ST_H
#define FT_EL_1ST_H

#include "gd_info.h"
#include "fault_info.h"
#include <cuda_runtime.h>


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
  float *mT1x;
  float *mT1y;
  float *mT1z;
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
               fault_wav_t *FW,
               int number_of_levels);

int 
fault_var_update(float *f_end_d, int it, float dt, 
                 gdinfo_t gdinfo_d, fault_t F, 
                 fault_coef_t FC, fault_wav_t FW);

__global__ void
fault_var_update_gpu(float *f_Vx,float *f_Vy, float *f_Vz, 
                     int nj, int nj1, int nk, int nk1, 
                     int ny, size_t siz_slice_yz,
                     int it, float dt, fault_coef_t FC, 
                     fault_t F, fault_wav_t FW);

__global__ void
fault_stress_update_first(size_t size, float coef, fault_t F);

__global__ void
fault_stress_update(size_t size, float coef, fault_t F);

#endif

