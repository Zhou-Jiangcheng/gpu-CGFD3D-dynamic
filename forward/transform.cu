#include <stdio.h>
#include <stdlib.h>
#include "transform.h"
#include "fdlib_math.h"


int 
wave2fault_onestage(float *w_cur_d, float *w_rhs_d, wav_t wav_d, 
                    float *f_cur_d, float *f_rhs_d, fault_wav_t FW, 
                    int i0, fault_t F, gd_metric_t metric_d, gdinfo_t gdinfo_d)
{
  int nj = gdinfo_d.nj;
  int nk = gdinfo_d.nk;
  int ny = gdinfo_d.ny;
  int nj1 = gdinfo_d.nj1;
  int nk1 = gdinfo_d.nk1;
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
  float *Tyz   = w_cur_d + wav_d.Tyz_pos;
  float *Txz   = w_cur_d + wav_d.Txz_pos;
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
  float *f_T2x   = f_cur_d + FW.T2x_pos;
  float *f_T2y   = f_cur_d + FW.T2y_pos;
  float *f_T2z   = f_cur_d + FW.T2z_pos;
  float *f_T3x   = f_cur_d + FW.T3x_pos;
  float *f_T3y   = f_cur_d + FW.T3y_pos;
  float *f_T3z   = f_cur_d + FW.T3z_pos;

  float *f_T1x   = FW.T1x;
  float *f_T1y   = FW.T1y;
  float *f_T1z   = FW.T1z;

  float *f_hT2x  = f_rhs_d + FW.T2x_pos; 
  float *f_hT2y  = f_rhs_d + FW.T2y_pos; 
  float *f_hT2z  = f_rhs_d + FW.T2z_pos; 
  float *f_hT3x  = f_rhs_d + FW.T3x_pos; 
  float *f_hT3y  = f_rhs_d + FW.T3y_pos; 
  float *f_hT3z  = f_rhs_d + FW.T3z_pos; 
  {
    dim3 block(8,8);
    dim3 grid;
    grid.x = (nj+block.x-1)/block.x;
    grid.y = (nk+block.y-1)/block.y;
    wave2fault_gpu <<<grid, block>>>(Vx, Vy, Vz, Txx, Tyy, Tzz,
                                     Tyz, Txz, Txy, hTxx, hTyy, hTzz,
                                     hTyz,hTxz,hTxy,f_Vx, f_Vy, f_Vz,
                                     f_T2x, f_T2y, f_T2z,
                                     f_T3x, f_T3y, f_T3z,
                                     f_T1x, f_T1y, f_T1z,
                                     f_hT2x,f_hT2y,f_hT2z,
                                     f_hT3x,f_hT3y,f_hT3z,
                                     xi_x, xi_y, xi_z,
                                     et_x, et_y, et_z,
                                     zt_x, zt_y, zt_z,
                                     jac3d, i0, nj, nj1, nk, nk1, ny, 
                                     siz_line, siz_slice,
                                     siz_slice_yz, F);
  }
  return 0;
}

__global__ void 
wave2fault_gpu(float * Vx,    float * Vy,    float * Vz,
               float * Txx,   float * Tyy,   float * Tzz,
               float * Tyz,   float * Txz,   float * Txy,
               float * hTxx,  float * hTyy,  float * hTzz,
               float * hTyz,  float * hTxz,  float * hTxy,
               float * f_Vx,  float * f_Vy,  float * f_Vz,
               float * f_T2x, float * f_T2y, float * f_T2z,
               float * f_T3x, float * f_T3y, float * f_T3z,
               float * f_T1x, float * f_T1y, float * f_T1z,
               float * f_hT2x,float * f_hT2y,float * f_hT2z,
               float * f_hT3x,float * f_hT3y,float * f_hT3z,
               float * xi_x,  float * xi_y,  float * xi_z,
               float * et_x,  float * et_y,  float * et_z,
               float * zt_x,  float * zt_y,  float * zt_z,
               float * jac3d, int i0, int nj, int nj1, 
               int nk, int nk1, int ny, 
               size_t siz_line, size_t siz_slice, 
               size_t siz_slice_yz, fault_t F)
{
  // it's not necessary do
  // transform
  //          wave (Txx, Tyy, ..., Tyz, Vx, ... (at i0))
  // to
  //          fault (T1x, T1y, T1z (at i0), T2x, ..., T3x, ..., Vx, ...)
  // and
  // transform
  //          wave (hTxx, hTyy, ..., hTyz)
  // to
  //          fault (hT2x, ..., hT3x, ...)
 
  size_t iy = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iz = blockIdx.y * blockDim.y + threadIdx.y;
  float jac;
  float metric[3][3], stress[3][3], traction[3][3];
  size_t iptr, iptr_f;

  // only united == 1 , wave transform fault
  // pml + strong boundry easy implement
  if( iy < nj && iz < nk && F.united[iy + iz * nj] == 1) 
  {
    iptr = i0 + (iy+nj1) * siz_line + (iz+nk1) * siz_slice;
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
    //NOTE  T1x -3 : 3. fault T1x is medium, = 0. so 3 * siz_slice_yz   
    iptr_f = (iy+nj1) + (iz+nk1) * ny + 3 * siz_slice_yz;  
    f_T1x[iptr_f] = traction[0][0];
    f_T1y[iptr_f] = traction[1][0];
    f_T1z[iptr_f] = traction[2][0];

    for (int m = 0; m < 2; m++)
    {
      // Split nodes => 0: minus side, 1: plus side
      iptr_f = (iy+nj1) + (iz+nk1) * ny + m * siz_slice_yz;
      f_Vx [iptr_f] = Vx[iptr];
      f_Vy [iptr_f] = Vy[iptr];
      f_Vz [iptr_f] = Vz[iptr];
      f_T2x[iptr_f] = traction[0][1];
      f_T2y[iptr_f] = traction[1][1];
      f_T2z[iptr_f] = traction[2][1];
      f_T3x[iptr_f] = traction[0][2];
      f_T3y[iptr_f] = traction[1][2];
      f_T3z[iptr_f] = traction[2][2];
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
      iptr_f = (iy+nj1) + (iz+nk1) * ny + m * siz_slice_yz;
      f_hT2x[iptr_f] = traction[0][1];
      f_hT2y[iptr_f] = traction[1][1];
      f_hT2z[iptr_f] = traction[2][1];
      f_hT3x[iptr_f] = traction[0][2];
      f_hT3y[iptr_f] = traction[1][2];
      f_hT3z[iptr_f] = traction[2][2];
    }
  } // end of loop j, k
  return;
}

int
fault2wave_onestage(float *w_cur_d, wav_t wav_d, 
                    float *f_cur_d, fault_wav_t FW, 
                    int i0, fault_t F, gd_metric_t metric_d, gdinfo_t gdinfo_d)
{
  int nj = gdinfo_d.nj;
  int nk = gdinfo_d.nk;
  int ny = gdinfo_d.ny;
  int nz = gdinfo_d.nz;
  int nj1 = gdinfo_d.nj1;
  int nk1 = gdinfo_d.nk1;
  size_t siz_line   = gdinfo_d.siz_line;
  size_t siz_slice  = gdinfo_d.siz_slice;
  size_t siz_slice_yz = gdinfo_d.siz_slice_yz;

  // OUTPUT
  float *Vx    = w_cur_d + wav_d.Vx_pos ;
  float *Vy    = w_cur_d + wav_d.Vy_pos ;
  float *Vz    = w_cur_d + wav_d.Vz_pos ;
  float *Txx   = w_cur_d + wav_d.Txx_pos;
  float *Tyy   = w_cur_d + wav_d.Tyy_pos;
  float *Tzz   = w_cur_d + wav_d.Tzz_pos;
  float *Tyz   = w_cur_d + wav_d.Tyz_pos;
  float *Txz   = w_cur_d + wav_d.Txz_pos;
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
  float *f_T2x   = f_cur_d + FW.T2x_pos;
  float *f_T2y   = f_cur_d + FW.T2y_pos;
  float *f_T2z   = f_cur_d + FW.T2z_pos;
  float *f_T3x   = f_cur_d + FW.T3x_pos;
  float *f_T3y   = f_cur_d + FW.T3y_pos;
  float *f_T3z   = f_cur_d + FW.T3z_pos;

  float *f_T1x   = FW.T1x;
  float *f_T1y   = FW.T1y;
  float *f_T1z   = FW.T1z;
  {
    dim3 block(8,8);
    dim3 grid;
    grid.x = (nj+block.x-1)/block.x;
    grid.y = (nk+block.y-1)/block.y;
    fault2wave_gpu <<<grid, block>>>(Vx, Vy, Vz, Txx, Tyy, Tzz,
                                     Tyz, Txz, Txy, f_Vx, f_Vy, f_Vz,
                                     f_T2x, f_T2y, f_T2z, 
                                     f_T3x, f_T3y, f_T3z,
                                     f_T1x, f_T1y, f_T1z,
                                     xi_x, xi_y, xi_z,
                                     et_x, et_y, et_z,
                                     zt_x, zt_y, zt_z,
                                     jac3d, i0, nj, nj1, nk, nk1, ny, 
                                     siz_line, siz_slice, 
                                     siz_slice_yz, F);
  }

  return 0;
}

__global__ void 
fault2wave_gpu(float * Vx,  float * Vy,  float * Vz,
               float * Txx, float * Tyy, float * Tzz,
               float * Txz, float * Tyz, float * Txy,
               float * f_Vx,  float * f_Vy,  float * f_Vz,
               float * f_T2x, float * f_T2y, float * f_T2z,
               float * f_T3x, float * f_T3y, float * f_T3z,
               float * f_T1x, float * f_T1y, float * f_T1z,
               float * xi_x,  float * xi_y, float * xi_z,
               float * et_x,  float * et_y, float * et_z,
               float * zt_x,  float * zt_y, float * zt_z,
               float *jac3d, int i0, int nj, int nj1,
               int nk, int nk1,int ny, 
               size_t siz_line, size_t siz_slice, 
               size_t siz_slice_yz, fault_t F)
{
  // it's necessary for wave output
  // transform
  //          fault (T1x (at i0), ...
  //                 T2x, ..., F->T3x, ...)
  // to
  //          wave (Txx, Tyy, ..., Tyz (at i0)
  // and
  // transform
  //          fault (Vx_f, ..., F->Vz_f)
  // to
  //          wave (Vx, ..., Vz (at i0))
  size_t iy = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iz = blockIdx.y * blockDim.y + threadIdx.y;

  float jac;
  float metric[3][3], stress[3][3], traction[3][3];
  size_t iptr, iptr_f;
  
  // only united == 0 , fault transform wave
  if( iy < nj && iz < nk && F.united[iy + iz * nj] == 0)
  { 
    iptr = i0 + (iy+nj1) * siz_line + (iz+nk1) * siz_slice;
    metric[0][0]=xi_x[iptr];metric[0][1]=et_x[iptr];metric[0][2]=zt_x[iptr];
    metric[1][0]=xi_y[iptr];metric[1][1]=et_y[iptr];metric[1][2]=zt_y[iptr];
    metric[2][0]=xi_z[iptr];metric[2][1]=et_z[iptr];metric[2][2]=zt_z[iptr];
    jac = 1.0/jac3d[iptr];

    //NOTE  T1x -3 : 3. fault T1x is medium, = 0. so 3 * siz_slice_yz
    iptr_f = (iy+nj1) + (iz+nk1) * ny + 3 * siz_slice_yz;  
    traction[0][0] = f_T1x[iptr_f];
    traction[1][0] = f_T1y[iptr_f];
    traction[2][0] = f_T1z[iptr_f];
    iptr_f = (iy+nj1) + (iz+nk1) * ny;
    traction[0][1] = (f_T2x[iptr_f] + f_T2x[iptr_f + siz_slice_yz]) * 0.5; // T2x
    traction[1][1] = (f_T2y[iptr_f] + f_T2y[iptr_f + siz_slice_yz]) * 0.5; // T2y
    traction[2][1] = (f_T2z[iptr_f] + f_T2z[iptr_f + siz_slice_yz]) * 0.5; // T2z
    traction[0][2] = (f_T3x[iptr_f] + f_T3x[iptr_f + siz_slice_yz]) * 0.5; // T3x
    traction[1][2] = (f_T3y[iptr_f] + f_T3y[iptr_f + siz_slice_yz]) * 0.5; // T3y
    traction[2][2] = (f_T3z[iptr_f] + f_T3z[iptr_f + siz_slice_yz]) * 0.5; // T3z

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

