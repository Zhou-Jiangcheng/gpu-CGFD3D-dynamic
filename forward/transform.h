#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "gd_info.h"
#include "gd_t.h"
#include "wav_t.h"
#include "fault_info.h"
#include "fault_wav_t.h"

int 
wave2fault_onestage(float *w_cur_d, float *w_rhs_d, wav_t wav_d, 
                    float *f_cur_d, float *f_rhs_d, fault_wav_t FW, 
                    int i0, fault_t F, gd_metric_t metric_d, gdinfo_t gdinfo_d);

__global__ void 
wave2fault_gpu(float * Vx,  float * Vy,  float * Vz,
               float * Txx, float * Tyy, float * Tzz,
               float * Tyz, float * Txz, float * Txy,
               float * hTxx,float * hTyy,float * hTzz,
               float * hTyz,float * hTxz,float * hTxy,
               float * f_Vx,  float * f_Vy,  float * f_Vz,
               float * f_T2x, float * f_T2y, float * f_T2z,
               float * f_T3x, float * f_T3y, float * f_T3z,
               float * f_T1x, float * f_T1y, float * f_T1z,
               float * f_hT2x,float * f_hT2y,float * f_hT2z,
               float * f_hT3x,float * f_hT3y,float * f_hT3z,
               float * xi_x,  float * xi_y, float * xi_z,
               float * et_x,  float * et_y, float * et_z,
               float * zt_x,  float * zt_y, float * zt_z,
               float * jac3d,  int i0, int nj, int nj1, 
               int nk, int nk1, int ny, 
               size_t siz_iy, size_t siz_iz, 
               size_t siz_iz_yz, fault_t F);

int
fault2wave_onestage(float *w_cur_d, wav_t wav_d, 
                    float *f_cur_d, fault_wav_t FW, 
                    int i0, fault_t F, gd_metric_t metric_d, gdinfo_t gdinfo_d);

__global__ void 
fault2wave_gpu(float * Vx,  float * Vy,  float * Vz,
               float * Txx, float * Tyy, float * Tzz,
               float * Tyz, float * Txz, float * Txy,
               float * f_Vx,  float * f_Vy,  float * f_Vz,
               float * f_T2x, float * f_T2y, float * f_T2z,
               float * f_T3x, float * f_T3y, float * f_T3z,
               float * f_T1x, float * f_T1y, float * f_T1z,
               float * xi_x,  float * xi_y, float * xi_z,
               float * et_x,  float * et_y, float * et_z,
               float * zt_x,  float * zt_y, float * zt_z,
               float * jac3d, int i0, int nj, int nj1,
               int nk, int nk1, int ny, 
               size_t siz_iy, size_t siz_iz, 
               size_t siz_iz_yz, fault_t F);

#endif
