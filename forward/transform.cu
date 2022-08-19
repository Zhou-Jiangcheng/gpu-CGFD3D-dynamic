#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "params.h"
#include "macdrp.h"


__global__ void 
wave2fault_onestage(float *w_cur_d, float *w_rhs_d, wav_t wav_d, 
                    float *f_cur_d, float *f_rhs_d, fault_wav_t FW, 
                    int i0, fault_t F, gdcurv_metric_t metric_d, gdinfo_t gdinfo_d)
{
  int nj = gdinfo_d.nj;
  int nk = gdinfo_d.nk;
  int ny = gdinfo_d.ny;
  size_t siz_line   = gdinfo_d.siz_line;
  size_t siz_slice  = gdinfo_d.siz_slice;
  size_t siz_slice_yz = gdinfo_d.siz_slice_yz;

  // INPUT
  // local pointer get each vars
  float *Vx    = w_cur_d + wav_d.Vx_pos ;
  float *Vy    = w_cur_d + wav_d.Vy_pos ;
  float *Vz    = w_cur_d + wav_d.Vz_pos ;
  float *Txx   = w_cur_d + wav_d.Txx_pos;
  float *Tyy   = w_cur_d + wav_d.Tyy_pos;
  float *Tzz   = w_cur_d + wav_d.Tzz_pos;
  float *Txz   = w_cur_d + wav_d.Txz_pos;
  float *Tyz   = w_cur_d + wav_d.Tyz_pos;
  float *Txy   = w_cur_d + wav_d.Txy_pos;

  float *hTxx  = w_rhs_d   + wav_d.Txx_pos; 
  float *hTyy  = w_rhs_d   + wav_d.Tyy_pos; 
  float *hTzz  = w_rhs_d   + wav_d.Tzz_pos; 
  float *hTyz  = w_rhs_d   + wav_d.Tyz_pos; 
  float *hTxz  = w_rhs_d   + wav_d.Txz_pos; 
  float *hTxy  = w_rhs_d   + wav_d.Txy_pos; 

  float *xi_x  = metric_d.xi_x;
  float *xi_y  = metric_d.xi_y;
  float *xi_z  = metric_d.xi_z;
  float *et_x  = metric_d.eta_x;
  float *et_y  = metric_d.eta_y;
  float *et_z  = metric_d.eta_z;
  float *zt_x  = metric_d.zeta_x;
  float *zt_y  = metric_d.zeta_y;
  float *zt_z  = metric_d.zeta_z;
  float *jac3d = metric_d.jac;

  // OUTPUT
  // local pointer get each vars
  float *f_Vx    = f_cur_d + FW.Vx_pos ;
  float *f_Vy    = f_cur_d + FW.Vy_pos ;
  float *f_Vz    = f_cur_d + FW.Vz_pos ;
  float *f_T21   = f_cur_d + FW.T21_pos;
  float *f_T22   = f_cur_d + FW.T22_pos;
  float *f_T23   = f_cur_d + FW.T23_pos;
  float *f_T31   = f_cur_d + FW.T31_pos;
  float *f_T32   = f_cur_d + FW.T32_pos;
  float *f_T33   = f_cur_d + FW.T33_pos;

  float *f_T11   = FW.T11;
  float *f_T12   = FW.T12;
  float *f_T13   = FW.T13;

  float *f_hT21  = f_rhs_d + FW.T21_pos; 
  float *f_hT22  = f_rhs_d + FW.T22_pos; 
  float *f_hT23  = f_rhs_d + FW.T23_pos; 
  float *f_hT31  = f_rhs_d + FW.T31_pos; 
  float *f_hT32  = f_rhs_d + FW.T32_pos; 
  float *f_hT33  = f_rhs_d + FW.T33_pos; 
  {
    dim3 block(8,8);
    dim3 grid;
    grid.x = (nj+block.x-1)/block.x;
    grid.y = (nk+block.y-1)/block.y;
    wave2fault_gpu(Vx, Vy, Vy, Txx, Tyy, Tzz,
                   Txz, Tyz, Txy, hTxx, hTyy, hTzz,
                   hTxz,hTyz,hTxy,f_Vx, f_Vy, f_Vz,
                   f_T21, f_T22, f_T23,
                   f_T31, f_T32, f_T33,
                   f_T11, f_T12, f_T13,
                   f_hT21,f_hT22,f_hT23,
                   f_hT31,f_hT32,f_hT33,
                   xi_x, xi_y, xi_z,
                   et_x, et_y, et_z,
                   zt_x, zt_y, zt_z,
                   jac3d, i0, nj, nk, ny, 
                   siz_line, siz_slice,
                   siz_slice_yz, F);

  return 0;
}

__global__ void 
wave2fault_gpu(float * Vx,  float * Vy,  float * Vy,
               float * Txx, float * Tyy, float * Tzz,
               float * Txz, float * Tyz, float * Txy,
               float * hTxx,float * hTyy,float * hTzz,
               float * hTxz,float * hTyz,float * hTxy,
               float * f_Vx,  float * f_Vy,  float * f_Vz,
               float * f_T21, float * f_T22, float * f_T23,
               float * f_T31, float * f_T32, float * f_T33,
               float * f_T11, float * f_T12, float * f_T13,
               float * f_hT21,float * f_hT22,float * f_hT23,
               float * f_hT31,float * f_hT32,float * f_hT33,
               float * xi_x,  float * xi_y, float * xi_z,
               float * et_x,  float * et_y, float * et_z,
               float * zt_x,  float * zt_y, float * zt_z,
               float * jac3d,  int i0, int nj, int nk, int ny, 
               size_t siz_line, size_t siz_slice, 
               size_t siz_slice_yz, fault_t F)
{
  // it's not necessary do
  // transform
  //          wave (Txx, Tyy, ..., Tyz, Vx, ... (at i0))
  // to
  //          fault (T11, T12, T13 (at i0), T21, ..., T31, ..., Vx, ...)
  // and
  // transform
  //          wave (hTxx, hTyy, ..., hTyz)
  // to
  //          fault (hT21, ..., hT31, ...)
 
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  float jac;
  float metric[3][3], stress[3][3], traction[3][3];
  int iptr, iptr_f;

  // only united == 1 , wave transform fault
  // pml + strong boundry easy implement
  if( iy < nj && iz < nk) 
  {
    if(F.united[iy + iz * nj])
    {
      iptr = i0 + (iy+3) * siz_line + (iz+3) * siz_slice;
      metric[0][0]=xi_x[iptr];metric[0][1]=et_x[iptr];metric[0][2]=zt_x[iptr];
      metric[1][0]=xi_y[iptr];metric[1][1]=et_y[iptr];metric[1][2]=zt_y[iptr];
      metric[2][0]=xi_z[iptr];metric[2][1]=et_z[iptr];metric[2][2]=zt_z[iptr];
      jac = jac3d[iptr];

      stress[0][0]=Txx[iptr];stress[0][1]=Txy[iptr];stress[0][2]=Txz[iptr];
      stress[1][0]=Txy[iptr];stress[1][1]=Tyy[iptr];stress[1][2]=Tyz[iptr];
      stress[2][0]=Txz[iptr];stress[2][1]=Tyz[iptr];stress[2][2]=Tzz[iptr];

      fdlib_math_matmul3x3(stress, metric, traction);

      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < 3; j++)
        {
          traction[i][j] *= jac;
        }
      }
      //NOTE  T11 -3 : 3. fault T11 is medium, = 0. so 3 * siz_slice_yz   
      iptr_f = (iy+3) + (iz+3) * ny + 3 * siz_slice_yz;  
      f_T11[iptr_f] = traction[0][0];
      f_T12[iptr_f] = traction[1][0];
      f_T13[iptr_f] = traction[2][0];

      for (int m = 0; m < 2; m++)
      {
        // Split nodes => 0: minus side, 1: plus side
        iptr_f = (iy+3) + (iz+3) * ny + m * siz_slice_yz;
        f_Vx [iptr_f] = Vx[iptr];
        f_Vy [iptr_f] = Vy[iptr];
        f_Vz [iptr_f] = Vz[iptr];
        f_T21[iptr_f] = traction[0][1];
        f_T22[iptr_f] = traction[1][1];
        f_T23[iptr_f] = traction[2][1];
        f_T31[iptr_f] = traction[0][2];
        f_T32[iptr_f] = traction[1][2];
        f_T33[iptr_f] = traction[2][2];
      }

      stress[0][0]=hTxx[iptr];stress[0][1]=hTxy[iptr];stress[0][2]=hTxz[iptr];
      stress[1][0]=hTxy[iptr];stress[1][1]=hTyy[iptr];stress[1][2]=hTyz[iptr];
      stress[2][0]=hTxz[iptr];stress[2][1]=hTyz[iptr];stress[2][2]=hTzz[iptr];

      fdlib_math_matmul3x3(stress, metric, traction);

      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < 3; j++)
        {
          traction[i][j] *= jac;
        }
      }

      for (int m = 0; m < 2; m++) 
      {
        iptr_f = (iy+3) + (iz+3) * ny + m * siz_slice_yz;
        f_hT21[iptr_f] = traction[0][1];
        f_hT22[iptr_f] = traction[1][1];
        f_hT23[iptr_f] = traction[2][1];
        f_hT31[iptr_f] = traction[0][2];
        f_hT32[iptr_f] = traction[1][2];
        f_hT33[iptr_f] = traction[2][2];
      }

    } // end if of united
  } // end of loop j, k
  return;
}

int
fault2wave_onestage(float *w_cur_d, wav_t wav_d, 
                    float *f_cur_d, fault_wav_t FW, 
                    int i0, fault_t F, gdcurv_metric_t metric_d, gdinfo_t gdinfo_d)
{
  int nj = gdinfo_d.nj;
  int nk = gdinfo_d.nk;
  int ny = gdinfo_d.ny;
  int nz = gdinfo_d.nz;
  size_t siz_line   = gdinfo_d.siz_line;
  size_t siz_slice  = gdinfo_d.siz_slice;
  size_t siz_slice_yz = ny * nz;

  // OUTPUT
  float *Vx    = w_cur_d + wav_d.Vx_pos ;
  float *Vy    = w_cur_d + wav_d.Vy_pos ;
  float *Vz    = w_cur_d + wav_d.Vz_pos ;
  float *Txx   = w_cur_d + wav_d.Txx_pos;
  float *Tyy   = w_cur_d + wav_d.Tyy_pos;
  float *Tzz   = w_cur_d + wav_d.Tzz_pos;
  float *Txz   = w_cur_d + wav_d.Txz_pos;
  float *Tyz   = w_cur_d + wav_d.Tyz_pos;
  float *Txy   = w_cur_d + wav_d.Txy_pos;

  // INPUT
  float *xi_x  = metric_d.xi_x;
  float *xi_y  = metric_d.xi_y;
  float *xi_z  = metric_d.xi_z;
  float *et_x  = metric_d.eta_x;
  float *et_y  = metric_d.eta_y;
  float *et_z  = metric_d.eta_z;
  float *zt_x  = metric_d.zeta_x;
  float *zt_y  = metric_d.zeta_y;
  float *zt_z  = metric_d.zeta_z;
  float *jac3d = metric_d.jac;

  float *f_Vx    = f_cur_d + FW.Vx_pos ;
  float *f_Vy    = f_cur_d + FW.Vy_pos ;
  float *f_Vz    = f_cur_d + FW.Vz_pos ;
  float *f_T21   = f_cur_d + FW.T21_pos;
  float *f_T22   = f_cur_d + FW.T22_pos;
  float *f_T23   = f_cur_d + FW.T23_pos;
  float *f_T31   = f_cur_d + FW.T31_pos;
  float *f_T32   = f_cur_d + FW.T32_pos;
  float *f_T33   = f_cur_d + FW.T33_pos;

  float *f_T11   = FW.T11;
  float *f_T12   = FW.T12;
  float *f_T13   = FW.T13;
  {
    dim3 block(8,8);
    dim3 grid;
    grid.x = (nj+block.x-1)/block.x;
    grid.y = (nk+block.y-1)/block.y;
    fault2wave_gpu(Vx, Vy, Vy, Txx, Tyy, Tzz,
                   Txz, Tyz, Txy, f_Vx, f_Vy, f_Vz,
                   f_T21, f_T22, f_T23, 
                   f_T31, f_T32, f_T33,
                   f_T11, f_T12, f_T13,
                   xi_x, xi_y, xi_z,
                   et_x, et_y, et_z,
                   zt_x, zt_y, zt_z,
                   jac3d, i0, nj, nk, ny, 
                   siz_line, siz_slice, 
                   siz_slice_yz, F);

  return 0;
}

__global__ void 
fault2wave_gpu(float * Vx,  float * Vy,  float * Vy,
               float * Txx, float * Tyy, float * Tzz,
               float * Txz, float * Tyz, float * Txy,
               float * f_Vx,  float * f_Vy,  float * f_Vz,
               float * f_T21, float * f_T22, float * f_T23,
               float * f_T31, float * f_T32, float * f_T33,
               float * f_T11, float * f_T12, float * f_T13,
               float * xi_x,  float * xi_y, float * xi_z,
               float * et_x,  float * et_y, float * et_z,
               float * zt_x,  float * zt_y, float * zt_z,
               float *jac3d, int i0, int nj, int nk, int ny, 
               size_t siz_line, size_t siz_slice, 
               size_t siz_slice_yz, fault_t F)
{
  // it's necessary for wave output
  // transform
  //          fault (T11 (at i0), ...
  //                 T21, ..., F->T31, ...)
  // to
  //          wave (Txx, Tyy, ..., Tyz (at i0)
  // and
  // transform
  //          fault (Vx_f, ..., F->Vz_f)
  // to
  //          wave (Vx, ..., Vz (at i0))
  int iy = blockIdx.x * blockDim.x + threadIdx.x;
  int iz = blockIdx.y * blockDim.y + threadIdx.y;

  float jac;
  float metric[3][3], stress[3][3], traction[3][3];
  size_t iptr, iptr_f, iptr_t;
  
  // only united == 0 , fault transform wave
  if( iy < nj && iz < nk)
  { 
    if(F.united[iy + iz * nj]) return;

    iptr = i0 + (iy+3) * siz_line + (iz+3) * siz_slice;
    metric[0][0]=xi_x[iptr];metric[0][1]=et_x[iptr];metric[0][2]=zt_x[iptr];
    metric[1][0]=xi_y[iptr];metric[1][1]=et_y[iptr];metric[1][2]=zt_y[iptr];
    metric[2][0]=xi_z[iptr];metric[2][1]=et_z[iptr];metric[2][2]=zt_z[iptr];
    jac = 1.0/jac3d[iptr];

    //NOTE  T11 -3 : 3. fault T11 is medium, = 0. so 3 * siz_slice_yz
    iptr_f = (iy+3) + (iz+3) * ny + 3 * siz_slice_yz;  
    traction[0][0] = f_T11[iptr_f];
    traction[1][0] = f_T12[iptr_f];
    traction[2][0] = f_T13[iptr_f];
    iptr_f = (iy+3) + (iz+3) * ny;
    traction[0][1] = (f_T21[iptr_f] + f_T21[iptr_f + siz_slice_yz]) * 0.5; // T21
    traction[1][1] = (f_T22[iptr_f] + f_T22[iptr_f + siz_slice_yz]) * 0.5; // T22
    traction[2][1] = (f_T23[iptr_f] + f_T23[iptr_f + siz_slice_yz]) * 0.5; // T23
    traction[0][2] = (f_T31[iptr_f] + f_T31[iptr_f + siz_slice_yz]) * 0.5; // T31
    traction[1][2] = (f_T32[iptr_f] + f_T32[iptr_f + siz_slice_yz]) * 0.5; // T32
    traction[2][2] = (f_T33[iptr_f] + f_T33[iptr_f + siz_slice_yz]) * 0.5; // T33

    fdlib_math_invert3x3(metric);
    fdlib_math_matmul3x3(traction, metric, stress);

    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        stress[i][j] *= jac;
      }
    }

    Vx [iptr] = (f_Vx[iptr_f] + f_Vx[iptr_f + siz_slice_yz]) * 0.5;
    Vy [iptr] = (f_Vy[iptr_f] + f_Vy[iptr_f + siz_slice_yz]) * 0.5;
    Vz [iptr] = (f_Vz[iptr_f] + f_Vz[iptr_f + siz_slice_yz]) * 0.5;
    Txx[iptr] = stress[0][0];
    Tyy[iptr] = stress[1][1];
    Tzz[iptr] = stress[2][2];
    Txy[iptr] = (stress[0][1] + stress[1][0]) * 0.5;
    Txz[iptr] = (stress[0][2] + stress[2][0]) * 0.5;
    Tyz[iptr] = (stress[1][2] + stress[2][1]) * 0.5;

  } // if not united

  return;
}

