#ifndef FT_WAVE_H
#define FT_WAVE_H

#include "gd_t.h"
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

  int ny, nz, ncmp, nlevel;

  size_t siz_slice_yz;
  size_t siz_slice_yz_2;
  size_t siz_ilevel;

  size_t *cmp_pos;
  char  **cmp_name;

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

  int number_fault;
  int *fault_index;
} fault_wav_t;
/*************************************************
 * function prototype
 *************************************************/

int 
fault_wav_init(gd_t *gd,
               fault_wav_t *FW,
               int number_fault,
               int *fault_x_index,
               int number_of_levels);

int 
fault_var_update(float *f_end_d, int it, float dt, 
                 gd_t gd_d, fault_t F, 
                 fault_coef_t FC, fault_wav_t FW);

__global__ void
fault_var_update_gpu(float *f_Vx,float *f_Vy, float *f_Vz, 
                     float *f_T1x,float *f_T1y, float *f_T1z,
                     float *f_hT1x,float *f_hT1y, float *f_hT1z,
                     float *f_mT1x,float *f_mT1y, float *f_mT1z,
                     int nj, int nj1, int nk, int nk1, 
                     int ny, size_t siz_slice_yz,
                     int it, float dt, int id, 
                     fault_t F, fault_coef_t FC);

int
fault_var_stage_update(float coef, int istage, 
                       gd_t gd_d, fault_t F);

__global__ void
fault_var_stage_update_gpu(int nj, int nj1, int nk, int nk1, int ny,
                           float coef, int istage, 
                           int id, fault_t F);

__global__ void
fault_wav_update(gd_t gd_d, int num_of_vars, 
                 float coef, int id, fault_t F,
                 float *w_update, float *w_input1, float *w_input2);

__global__ void
fault_wav_update_end(gd_t gd_d, int num_of_vars, 
                     float coef, int id, fault_t F,
                     float *w_update, float *w_input2);

#endif
