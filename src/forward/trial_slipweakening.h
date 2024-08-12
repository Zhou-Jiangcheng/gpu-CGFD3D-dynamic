#ifndef T_SW_H
#define T_SW_H

#include "wav_t.h"
#include "fd_t.h"
#include "gd_t.h"
#include "md_t.h"
#include "fault_info.h"
#include "fault_wav_t.h"

int 
trial_slipweakening_onestage(
                  float *w_cur_d,
                  float *f_cur_d,
                  float *f_pre_d,
                  int isfree,
                  float dt,
                  gd_t gd_d,
                  gd_metric_t metric_d,
                  wav_t wav_d,
                  fault_wav_t FW,
                  fault_t F,
                  fault_coef_t FC,
                  fd_op_t *fdy_op,
                  fd_op_t *fdz_op,
                  const int myid);

__global__ void 
trial_slipweakening_gpu(
                  float *Txx,   float *Tyy,   float *Tzz,
                  float *Tyz,   float *Txz,   float *Txy,
                  float *f_T2x, float *f_T2y, float *f_T2z,
                  float *f_T3x, float *f_T3y, float *f_T3z,
                  float *f_T1x, float *f_T1y, float *f_T1z,
                  float *f_mVx, float *f_mVy, float *f_mVz,
                  float *xi_x,  float *xi_y,  float *xi_z,
                  float *jac3d, int isfree, float dt, 
                  int nj1, int nj, int nk1, int nk, int ny,
                  size_t siz_iy, size_t siz_iz, size_t siz_slice_yz, 
                  int jdir, int kdir,
                  int id, fault_t F, fault_coef_t FC);

#endif
