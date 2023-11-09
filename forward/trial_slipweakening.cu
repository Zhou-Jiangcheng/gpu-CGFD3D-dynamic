#include <stdio.h>
#include <stdlib.h>
#include "trial_slipweakening.h"
#include "fdlib_math.h"

int 
trial_slipweakening_onestage(
                  float *w_cur_d,
                  float *f_cur_d,
                  float *f_pre_d,
                  int i0,
                  int isfree,
                  float dt,
                  gdinfo_t gdinfo_d,
                  gd_metric_t metric_d,
                  wav_t wav_d,
                  fault_wav_t FW,
                  fault_t F,
                  fault_coef_t FC,
                  fd_op_t *fdy_op,
                  fd_op_t *fdz_op,
                  const int myid, const int verbose)
{
  // local pointer get each vars
  float *Txx   = w_cur_d + wav_d.Txx_pos;
  float *Tyy   = w_cur_d + wav_d.Tyy_pos;
  float *Tzz   = w_cur_d + wav_d.Tzz_pos;
  float *Tyz   = w_cur_d + wav_d.Tyz_pos;
  float *Txz   = w_cur_d + wav_d.Txz_pos;
  float *Txy   = w_cur_d + wav_d.Txy_pos;

  float *f_Vx    = f_cur_d + FW.Vx_pos ;
  float *f_Vy    = f_cur_d + FW.Vy_pos ;
  float *f_Vz    = f_cur_d + FW.Vz_pos ;
  float *f_T2x   = f_cur_d + FW.T2x_pos;
  float *f_T2y   = f_cur_d + FW.T2y_pos;
  float *f_T2z   = f_cur_d + FW.T2z_pos;
  float *f_T3x   = f_cur_d + FW.T3x_pos;
  float *f_T3y   = f_cur_d + FW.T3y_pos;
  float *f_T3z   = f_cur_d + FW.T3z_pos;

  float *f_T1x = FW.T1x;
  float *f_T1y = FW.T1y;
  float *f_T1z = FW.T1z;

  float *f_mVx  = f_pre_d + FW.Vx_pos;
  float *f_mVy  = f_pre_d + FW.Vy_pos;
  float *f_mVz  = f_pre_d + FW.Vz_pos;

  float *xi_x  = metric_d.xi_x;
  float *xi_y  = metric_d.xi_y;
  float *xi_z  = metric_d.xi_z;
  float *jac3d = metric_d.jac;

  int nj1 = gdinfo_d.nj1;
  int nk1 = gdinfo_d.nk1;
  int nj  = gdinfo_d.nj;
  int nk  = gdinfo_d.nk;
  int ny  = gdinfo_d.ny;

  size_t siz_iy  = gdinfo_d.siz_iy;
  size_t siz_iz  = gdinfo_d.siz_iz;
  size_t siz_iz_yz = gdinfo_d.siz_iz_yz;

  int jdir = fdy_op->dir;
  int kdir = fdz_op->dir;
  
  {
    dim3 block(8,8);
    dim3 grid;
    grid.x = (nj+block.x-1)/block.x;
    grid.y = (nk+block.y-1)/block.y;
    trial_slipweakening_gpu <<<grid, block>>>(
                         Txx, Tyy, Tzz, Tyz, Txz, Txy, 
                         f_T2x, f_T2y, f_T2z,
                         f_T3x, f_T3y, f_T3z,
                         f_T1x, f_T1y, f_T1z,
                         f_mVx, f_mVy, f_mVz,
                         xi_x,  xi_y,  xi_z,
                         jac3d, i0, isfree, dt, 
                         nj1, nj, nk1, nk, ny, siz_iy, 
                         siz_iz, siz_iz_yz, 
                         jdir, kdir, F, FC);
  }
 
  return 0;
}

__global__ void 
trial_slipweakening_gpu(
    float *Txx,   float *Tyy,   float *Tzz,
    float *Tyz,   float *Txz,   float *Txy,
    float *f_T2x, float *f_T2y, float *f_T2z,
    float *f_T3x, float *f_T3y, float *f_T3z,
    float *f_T1x, float *f_T1y, float *f_T1z,
    float *f_mVx, float *f_mVy, float *f_mVz,
    float *xi_x,  float *xi_y,  float *xi_z,
    float *jac3d, int i0, int isfree, float dt, 
    int nj1, int nj, int nk1, int nk, int ny,
    size_t siz_iy, size_t siz_iz, 
    size_t siz_iz_yz, int jdir, int kdir,
    fault_t F, fault_coef_t FC)
{
  size_t iy = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iz = blockIdx.y * blockDim.y + threadIdx.y;

  size_t iptr, iptr_f, iptr_t;
  float xix, xiy, xiz;
  float jac;
  float vec_n0;
  float jacvec;
  float rho;
  float Mrho[2], Rx[2], Ry[2], Rz[2];
  float T1x, T1y, T1z;
  float DyT2x, DyT2y, DyT2z;
  float DzT3x, DzT3y, DzT3z;
  float vecT3x[7];
  float vecT3y[7];
  float vecT3z[7];

  float *T2x_ptr;
  float *T2y_ptr;
  float *T2z_ptr;
  float *T3x_ptr;
  float *T3y_ptr;
  float *T3z_ptr;

  iptr_t = iy + iz * nj;
  if ( iy < nj && iz < nk && F.united[iptr_t] == 0)
  { 
    for (int m = 0; m < 2; m++)
    {
      // m = 0 -> minus
      // m = 1 -> plus
      // T1x T1y T1z
      // 0 1 2 3 4 5 6 index 0,1,2 minus, 3 fault, 4,5,6 plus
      iptr_f = (iy+nj1) + (iz+nk1) * ny; 
      iptr = i0 + (iy+nj1) * siz_iy + (iz+nk1) * siz_iz;
      jac = jac3d[iptr];
      rho = FC.rho_f[iptr_f + m * siz_iz_yz];
      // dh = 1, so omit dh in formula
      Mrho[m] = 0.5*jac*rho;

      // fault traction image method by zhang wenqiang 
      for (int l = 1; l <= 3; l++)
      {
        iptr = (i0+(2*m-1)*l) + (iy+nj1) * siz_iy + (iz+nk1) * siz_iz;
        iptr_f = (iy+nj1) + (iz+nk1) * ny; 
        xix = xi_x[iptr];
        xiy = xi_y[iptr];
        xiz = xi_z[iptr];
        jac = jac3d[iptr];
        T1x = jac*(xix * Txx[iptr] + xiy * Txy[iptr] + xiz * Txz[iptr]);
        T1y = jac*(xix * Txy[iptr] + xiy * Tyy[iptr] + xiz * Tyz[iptr]);
        T1z = jac*(xix * Txz[iptr] + xiy * Tyz[iptr] + xiz * Tzz[iptr]);
        f_T1x[(3+(2*m-1)*l)*siz_iz_yz + iptr_f] = T1x;
        f_T1y[(3+(2*m-1)*l)*siz_iz_yz + iptr_f] = T1y;
        f_T1z[(3+(2*m-1)*l)*siz_iz_yz + iptr_f] = T1z;
      }

      iptr_f = (iy+nj1) + (iz+nk1) * ny; 
      T2x_ptr = f_T2x + iptr_f + m*siz_iz_yz;
      T2y_ptr = f_T2y + iptr_f + m*siz_iz_yz;
      T2z_ptr = f_T2z + iptr_f + m*siz_iz_yz;
      // fault point use short stencil
      // due to slip change sharp
      if(F.rup_index_y[iptr_t] == 1)
      {
        M_FD_SHIFT_PTR_MAC22(DyT2x, T2x_ptr, 1, jdir);
        M_FD_SHIFT_PTR_MAC22(DyT2y, T2y_ptr, 1, jdir);
        M_FD_SHIFT_PTR_MAC22(DyT2z, T2z_ptr, 1, jdir);
      }
      if(F.rup_index_y[iptr_t] == 0)
      {
        M_FD_SHIFT_PTR_MACDRP(DyT2x, T2x_ptr, 1, jdir);
        M_FD_SHIFT_PTR_MACDRP(DyT2y, T2y_ptr, 1, jdir);
        M_FD_SHIFT_PTR_MACDRP(DyT2z, T2z_ptr, 1, jdir);
      }

      iptr_f = (iy+nj1) + (iz+nk1) * ny; 
      T3x_ptr = f_T3x + iptr_f + m*siz_iz_yz;
      T3y_ptr = f_T3y + iptr_f + m*siz_iz_yz;
      T3z_ptr = f_T3z + iptr_f + m*siz_iz_yz;
      if(F.rup_index_z[iptr_t] == 1)
      {
        M_FD_SHIFT_PTR_MAC22(DzT3x, T3x_ptr, ny, kdir);
        M_FD_SHIFT_PTR_MAC22(DzT3y, T3y_ptr, ny, kdir);
        M_FD_SHIFT_PTR_MAC22(DzT3z, T3z_ptr, ny, kdir);
      }
      if(F.rup_index_z[iptr_t] == 0)
      {
        M_FD_SHIFT_PTR_MACDRP(DzT3x, T3x_ptr, ny, kdir);
        M_FD_SHIFT_PTR_MACDRP(DzT3y, T3y_ptr, ny, kdir);
        M_FD_SHIFT_PTR_MACDRP(DzT3z, T3z_ptr, ny, kdir);
      }
      int km = nk - (iz+1); // index distance between current point and surface
      int n_free = km+3;    // index of free surface in vecT[]; 
      if(isfree == 1 && km<=3)
      {
        for (int l=-3; l<=3 ; l++)
        {
          iptr_f = (iy+nj1) + (iz+nk1+l) * ny + m*siz_iz_yz;
          vecT3x[l+3] = f_T3x[iptr_f];
          vecT3y[l+3] = f_T3y[iptr_f];
          vecT3z[l+3] = f_T3z[iptr_f];
        }

        vecT3x[n_free] = 0.0;
        vecT3y[n_free] = 0.0;
        vecT3z[n_free] = 0.0;
        for (int l = n_free+1; l<7; l++)
        {
          vecT3x[l] = -vecT3x[2*n_free-l];
          vecT3y[l] = -vecT3y[2*n_free-l];
          vecT3z[l] = -vecT3z[2*n_free-l];
        }

        M_FD_VEC(DzT3x, vecT3x+3, kdir);
        M_FD_VEC(DzT3y, vecT3y+3, kdir);
        M_FD_VEC(DzT3z, vecT3z+3, kdir);
      } 
      
      iptr_f = (iy+nj1) + (iz+nk1) * ny;
      if (m == 0){ // "-" side
        Rx[m] =
          a_1 * f_T1x[2*siz_iz_yz+iptr_f] +
          a_2 * f_T1x[1*siz_iz_yz+iptr_f] +
          a_3 * f_T1x[0*siz_iz_yz+iptr_f] ;
        Ry[m] =
          a_1 * f_T1y[2*siz_iz_yz+iptr_f] +
          a_2 * f_T1y[1*siz_iz_yz+iptr_f] +
          a_3 * f_T1y[0*siz_iz_yz+iptr_f] ;
        Rz[m] =
          a_1 * f_T1z[2*siz_iz_yz+iptr_f] +
          a_2 * f_T1z[1*siz_iz_yz+iptr_f] +
          a_3 * f_T1z[0*siz_iz_yz+iptr_f] ;
      }else{ // "+" side
        Rx[m] =
          a_1 * f_T1x[4*siz_iz_yz+iptr_f] +
          a_2 * f_T1x[5*siz_iz_yz+iptr_f] +
          a_3 * f_T1x[6*siz_iz_yz+iptr_f] ;
        Ry[m] =
          a_1 * f_T1y[4*siz_iz_yz+iptr_f] +
          a_2 * f_T1y[5*siz_iz_yz+iptr_f] +
          a_3 * f_T1y[6*siz_iz_yz+iptr_f] ;
        Rz[m] =
          a_1 * f_T1z[4*siz_iz_yz+iptr_f] +
          a_2 * f_T1z[5*siz_iz_yz+iptr_f] +
          a_3 * f_T1z[6*siz_iz_yz+iptr_f] ;
      }
      // dh = 1, so omit dh in formula
      Rx[m] = 0.5*((2*m-1)*Rx[m] + (DyT2x + DzT3x));
      Ry[m] = 0.5*((2*m-1)*Ry[m] + (DyT2y + DzT3y));
      Rz[m] = 0.5*((2*m-1)*Rz[m] + (DyT2z + DzT3z));
    } // end m

    // dv = (V+) - (V-)
    iptr_f = (iy+nj1) + (iz+nk1) * ny;
    float dVx = f_mVx[iptr_f + siz_iz_yz] - f_mVx[iptr_f];
    float dVy = f_mVy[iptr_f + siz_iz_yz] - f_mVy[iptr_f];
    float dVz = f_mVz[iptr_f + siz_iz_yz] - f_mVz[iptr_f];

    float Trial[3];       // stress variation
    float Trial_local[3]; // + init background stress
    float Trial_s[3];     // shear stress

    Trial[0] = 2.0*(Mrho[0]*Mrho[1]*dVx/dt + Mrho[0]*Rx[1] - Mrho[1]*Rx[0])/(a_0*(Mrho[0]+Mrho[1]));
    Trial[1] = 2.0*(Mrho[0]*Mrho[1]*dVy/dt + Mrho[0]*Ry[1] - Mrho[1]*Ry[0])/(a_0*(Mrho[0]+Mrho[1]));
    Trial[2] = 2.0*(Mrho[0]*Mrho[1]*dVz/dt + Mrho[0]*Rz[1] - Mrho[1]*Rz[0])/(a_0*(Mrho[0]+Mrho[1]));

    float vec_n [3];
    float vec_s1[3];
    float vec_s2[3];
    float vec_n0;

    iptr_f = (iy+nj1) + (iz+nk1) * ny;
    vec_s1[0] = FC.vec_s1[iptr_f * 3 + 0];
    vec_s1[1] = FC.vec_s1[iptr_f * 3 + 1];
    vec_s1[2] = FC.vec_s1[iptr_f * 3 + 2];
    vec_s2[0] = FC.vec_s2[iptr_f * 3 + 0];
    vec_s2[1] = FC.vec_s2[iptr_f * 3 + 1];
    vec_s2[2] = FC.vec_s2[iptr_f * 3 + 2];

    iptr = i0 + (iy+nj1) * siz_iy + (iz+nk1) * siz_iz;
    vec_n[0] = xi_x[iptr];
    vec_n[1] = xi_y[iptr];
    vec_n[2] = xi_z[iptr];
    vec_n0 = fdlib_math_norm3(vec_n);

    jacvec = jac3d[iptr] * vec_n0;
    for (int i=0; i<3; i++)
    {
        vec_n[i] /= vec_n0;
    }
    
    iptr_t = iy + iz * nj; 
    Trial_local[0] = Trial[0]/jacvec + F.T0x[iptr_t];
    Trial_local[1] = Trial[1]/jacvec + F.T0y[iptr_t];
    Trial_local[2] = Trial[2]/jacvec + F.T0z[iptr_t];

    float Trial_n0 = fdlib_math_dot_product(Trial_local, vec_n);
    // Ts = T - n * Tn
    Trial_s[0] = Trial_local[0] - vec_n[0]*Trial_n0;
    Trial_s[1] = Trial_local[1] - vec_n[1]*Trial_n0;
    Trial_s[2] = Trial_local[2] - vec_n[2]*Trial_n0;

    float Trial_s0;
    Trial_s0 = fdlib_math_norm3(Trial_s);

    float Tau_n = Trial_n0;
    float Tau_s = Trial_s0;

    int ifchange = 0; // false
    if(Trial_n0 >= 0.0){
      // fault can not open
      Tau_n = 0.0;
      ifchange = 1;
    }else{
      Tau_n = Trial_n0;
    }

    float mu_s = F.mu_s[iptr_t];
    float mu_d = F.mu_d[iptr_t];
    float slip = F.slip[iptr_t];
    float Dc = F.Dc[iptr_t];
    float C0 = F.C0[iptr_t];
    float friction;

    // slip weakening
    if(slip <= Dc){
      friction = mu_s - (mu_s - mu_d) * slip / Dc;
    }else{
      friction = mu_d;
    }

    float Tau_c = -friction * Tau_n + C0;

    if(Trial_s0 >= Tau_c){
      Tau_s = Tau_c; // can not exceed shear strengh!
      F.flag_rup[iptr_t] = 1;
      ifchange = 1;
    }else{
      Tau_s = Trial_s0;
      F.flag_rup[iptr_t] = 0;
    }

    float Tau[3];
    if(ifchange == 1){
      // to avoid divide by 0, 1e-1 is a small value compared to stress
      if(fabs(Trial_s0) < 1e-1){
        Tau[0] = Tau_n * vec_n[0];
        Tau[1] = Tau_n * vec_n[1];
        Tau[2] = Tau_n * vec_n[2];
      }else{
        Tau[0] = Tau_n * vec_n[0] + (Tau_s/Trial_s0) * Trial_s[0];
        Tau[1] = Tau_n * vec_n[1] + (Tau_s/Trial_s0) * Trial_s[1];
        Tau[2] = Tau_n * vec_n[2] + (Tau_s/Trial_s0) * Trial_s[2];
      }
    }else{
      Tau[0] = Trial_local[0];
      Tau[1] = Trial_local[1];
      Tau[2] = Trial_local[2];
    }

    iptr_f = (iy+nj1) + (iz+nk1) * ny;
    if(ifchange == 1){
      f_T1x[iptr_f+3*siz_iz_yz] = (Tau[0] - F.T0x[iptr_t])*jacvec;
      f_T1y[iptr_f+3*siz_iz_yz] = (Tau[1] - F.T0y[iptr_t])*jacvec;
      f_T1z[iptr_f+3*siz_iz_yz] = (Tau[2] - F.T0z[iptr_t])*jacvec;
    }else{
      f_T1x[iptr_f+3*siz_iz_yz] = Trial[0];
      f_T1y[iptr_f+3*siz_iz_yz] = Trial[1];
      f_T1z[iptr_f+3*siz_iz_yz] = Trial[2];
    }

    F.tTs1[iptr_t] = fdlib_math_dot_product(Tau, vec_s1);
    F.tTs2[iptr_t] = fdlib_math_dot_product(Tau, vec_s2);
    F.tTn [iptr_t] = Tau_n;

  } 
  return;
}
