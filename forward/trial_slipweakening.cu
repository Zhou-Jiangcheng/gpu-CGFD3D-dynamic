#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "params.h"
#include "macdrp.h"

__global__ void 
trial_sw_gpu(float *w_cur_d,
             float *f_cur_d,
             float *f_pre_d,
             gdinfo_t gdinfo_d,
             gdcurv_metric_t metric_d,
             wav_t wav_d,
             fault_wav_t f_wav_d,
             fault_t f_d,
             fault_coef_t f_coef_d)
{
  int iy = blockIdx.x * blockDim.x + threadIdx.x;
  int iz = blockIdx.y * blockDim.y + threadIdx.y;

  int ni = gdinfo_d.ni;
  int nj = gdinfo_d.nj;
  int nk = gdinfo_d.nk;
  int nx = gdinfo_d.nx;
  int ny = gdinfo_d.ny;
  int nz = gdinfo_d.nz;
  int npoint_x = gdinfo_d.npoint_x;
  int i0  = npoint_x/2 + 3; //fault local index with ghost

  size_t iptr, iptr1, iptr2, iptr_f, iptr_t;
  size_t siz_line   = gdinfo_d.siz_line;
  size_t siz_slice  = gdinfo_d.siz_slice;
  size_t siz_volume = gdinfo_d.siz_volume;
  size_t siz_slice_yz = gdinfo_d.siz_slice_yz;
  float xix, xiy, xiz;
  float jac;
  float vec_n0;
  float jacvec;
  float rho;
  float Mrho[2], Rx[2], Ry[2], Rz[2];
  float T11, T12, T13;
  float DyT21, DyT22, DyT23;
  float DzT31, DzT32, DzT33;
  float vecT31[7];
  float vecT32[7];
  float vecT33[7];

  float *xi_x  = metric_d.xi_x;
  float *xi_y  = metric_d.xi_y;
  float *xi_z  = metric_d.xi_z;
  float *jac3d = metric_d.jac;

  float *Txx   = w_cur_d + wav_d.Txx_pos;
  float *Tyy   = w_cur_d + wav_d.Tyy_pos;
  float *Tzz   = w_cur_d + wav_d.Tzz_pos;
  float *Txz   = w_cur_d + wav_d.Txz_pos;
  float *Tyz   = w_cur_d + wav_d.Tyz_pos;
  float *Txy   = w_cur_d + wav_d.Txy_pos;

  float *f_Vx    = f_cur_d + f_wav_d.Vx_pos ;
  float *f_Vy    = f_cur_d + f_wav_d.Vy_pos ;
  float *f_Vz    = f_cur_d + f_wav_d.Vz_pos ;
  float *f_T21   = f_cur_d + f_wav_d.T21_pos;
  float *f_T22   = f_cur_d + f_wav_d.T22_pos;
  float *f_T23   = f_cur_d + f_wav_d.T23_pos;
  float *f_T31   = f_cur_d + f_wav_d.T31_pos;
  float *f_T32   = f_cur_d + f_wav_d.T32_pos;
  float *f_T33   = f_cur_d + f_wav_d.T33_pos;

  float *f_mVx  = f_pre_d + f_wav_d.Vx_pos;
  float *f_mVy  = f_pre_d + f_wav_d.Vy_pos;
  float *f_mVz  = f_pre_d + f_wav_d.Vz_pos;

  if ( iy < nj && iz < nk && f_d.united[iy + iz * nj] == 0)
  { 
    for (int m = 0; m < 2; m++)
    {
      // m = 0 -> minus
      // m = 1 -> plus
      // T11 T12 T13
      // 0 1 2 3 4 5 6 index 0,1,2 minus, 3 fault, 4,5,6 plus
      iptr = i0 + (iy+3) * siz_line + (iz+3) * siz_slice;
      xix = xi_x [iptr];
      xiy = xi_y [iptr];
      xiz = xi_z [iptr];
      jac = jac3d[iptr];
      rho = f_d.rho_f[(iy+3) + (iz+3) * ny + m * ny * nz];
      Mrho[m] = 0.5*jac*rho;

      iptr_f = (iy+3) + (iz+3) * ny; 
      iptr_t = iy + iz * nj; 
      // fault traction image method by zhang wenqiang 
      for (int l = 1; l <= 3; l++)
      {
        iptr1 = (i0+(2*m-1)*l) + (iy+3) * siz_line + (iz+3) * siz_slice;
        xix = xi_x[iptr1];
        xiy = xi_y[iptr1];
        xiz = xi_z[iptr1];
        jac = jac3d[iptr1];
        T11 = jac*(xix * Txx[iptr1] + xiy * Txy[iptr1] + xiz * Txz[iptr1]);
        T12 = jac*(xix * Txy[iptr1] + xiy * Tyy[iptr1] + xiz * Tyz[iptr1]);
        T13 = jac*(xix * Txz[iptr1] + xiy * Tyz[iptr1] + xiz * Tzz[iptr1]);
        f_wav_d.T11[(3+(2*m-1)*l)*siz_slice_yz + iptr_f] = T11;
        f_wav_d.T12[(3+(2*m-1)*l)*siz_slice_yz + iptr_f] = T12;
        f_wav_d.T13[(3+(2*m-1)*l)*siz_slice_yz + iptr_f] = T13;
      }

      float *T21_ptr = f_T21 + m*siz_slice_yz + iptr_f;
      float *T22_ptr = f_T22 + m*siz_slice_yz + iptr_f;
      float *T23_ptr = f_T23 + m*siz_slice_yz + iptr_f;
      // fault point use short stencil
      // due to slip change sharp
      if(f_d.rup_index_y[iptr_t] == 1)
      {
        M_FD_SHIFT_PTR_MAC22(DyT21, T21_ptr, 1, jdir);
        M_FD_SHIFT_PTR_MAC22(DyT22, T22_ptr, 1, jdir);
        M_FD_SHIFT_PTR_MAC22(DyT23, T23_ptr, 1, jdir);
      }
      if(f_d.rup_index_y[iptr_t] == 0)
      {
        M_FD_SHIFT_PTR_MACDRP(DyT21, T21_ptr, 1, jdir);
        M_FD_SHIFT_PTR_MACDRP(DyT22, T22_ptr, 1, jdir);
        M_FD_SHIFT_PTR_MACDRP(DyT23, T23_ptr, 1, jdir);
      }

      float *T31_ptr = f_T31 + m*siz_slice_yz + iptr_f;
      float *T32_ptr = f_T32 + m*siz_slice_yz + iptr_f;
      float *T33_ptr = f_T33 + m*siz_slice_yz + iptr_f;
      if(f_d.rup_index_z[iptr_t] == 1)
      {
        M_FD_SHIFT_PTR_MAC22(DzT31, T31_ptr, ny, kdir);
        M_FD_SHIFT_PTR_MAC22(DzT32, T32_ptr, ny, kdir);
        M_FD_SHIFT_PTR_MAC22(DzT33, T33_ptr, ny, kdir);
      }
      if(f_d.rup_index_z[iptr_t] == 0)
      {
        M_FD_SHIFT_PTR_MACDRP(DzT31, T31_ptr, ny, kdir);
        M_FD_SHIFT_PTR_MACDRP(DzT32, T32_ptr, ny, kdir);
        M_FD_SHIFT_PTR_MACDRP(DzT33, T33_ptr, ny, kdir);
      }
      int km = nk - (iz+1); //index distance between current point and surface
      int n_free = km+3; // index of free surface in vecT[]; 
      if(bdryfree_d.is_at_sides[2][1] == 1 && km<=3)
      {
        for (int l=-3; l<=3 ; l++)
        {
          iptr2 = (iy+3) + (iz+3+l) * ny;
          vecT31[l+3] = f_T31[iptr2 + m*siz_slice_yz];
          vecT32[l+3] = f_T32[iptr2 + m*siz_slice_yz];
          vecT33[l+3] = f_T33[iptr2 + m*siz_slice_yz];
        }

        vecT31[n_free] = 0.0;
        vecT32[n_free] = 0.0;
        vecT33[n_free] = 0.0;
        for (int l = n_free+1; l<7; l++)
        {
          vecT31[l] = -vecT31[2*n_free-l];
          vecT32[l] = -vecT32[2*n_free-l];
          vecT33[l] = -vecT33[2*n_free-l];
        }

        M_FD_VEC(DzT31, vecT31+3, kdir);
        M_FD_VEC(DzT32, vecT32+3, kdir);
        M_FD_VEC(DzT33, vecT33+3, kdir);
      } 
      
      if (m == 0){ // "-" side
        Rx[m] =
          a_1 * f_wav_d.T11[2*siz_slice_yz+iptr_f] +
          a_2 * f_wav_d.T11[1*siz_slice_yz+iptr_f] +
          a_3 * f_wav_d.T11[0*siz_slice_yz+iptr_f] ;
        Ry[m] =
          a_1 * f_wav_d.T12[2*siz_slice_yz+iptr_f] +
          a_2 * f_wav_d.T12[1*siz_slice_yz+iptr_f] +
          a_3 * f_wav_d.T12[0*siz_slice_yz+iptr_f] ;
        Rz[m] =
          a_1 * f_wav_d.T13[2*siz_slice_yz+iptr_f] +
          a_2 * f_wav_d.T13[1*siz_slice_yz+iptr_f] +
          a_3 * f_wav_d.T13[0*siz_slice_yz+iptr_f] ;
      }else{ // "+" side
        Rx[m] =
          a_1 * f_wav_d.T11[4*siz_slice_yz+iptr_f] +
          a_2 * f_wav_d.T11[5*siz_slice_yz+iptr_f] +
          a_3 * f_wav_d.T11[6*siz_slice_yz+iptr_f] ;
        Ry[m] =
          a_1 * f_wav_d.T12[4*siz_slice_yz+iptr_f] +
          a_2 * f_wav_d.T12[5*siz_slice_yz+iptr_f] +
          a_3 * f_wav_d.T12[6*siz_slice_yz+iptr_f] ;
        Rz[m] =
          a_1 * f_wav_d.T13[4*siz_slice_yz+iptr_f] +
          a_2 * f_wav_d.T13[5*siz_slice_yz+iptr_f] +
          a_3 * f_wav_d.T13[6*siz_slice_yz+iptr_f] ;
      }
      // dh = 1, so omit dh in formula
      Rx[m] = 0.5*( (2*m-1)*Rx[m] + (DyT21 + DzT31));
      Ry[m] = 0.5*( (2*m-1)*Ry[m] + (DyT22 + DzT32));
      Rz[m] = 0.5*( (2*m-1)*Rz[m] + (DyT23 + DzT33));
    } // end m

    // dv = (V+) - (V-)
    float dVx = f_mVx[iptr_f + siz_slice_yz] - f_mVx[iptr_f];
    float dVy = f_mVy[iptr_f + siz_slice_yz] - f_mVy[iptr_f];
    float dVz = f_mVz[iptr_f + siz_slice_yz] - f_mVz[iptr_f];

    float Trial[3];       // stress variation
    float Trial_local[3]; // + init background stress
    float Trial_s[3];     // shear stress

    Trial[0] = (Mrho[0]*Mrho[1]*dVx/dt + Mrho[0]*Rx[1] - Mrho[1]*Rx[0])/(a_0*(Mrho[0]+Mrho[1]))*2.0;
    Trial[1] = (Mrho[0]*Mrho[1]*dVy/dt + Mrho[0]*Ry[1] - Mrho[1]*Ry[0])/(a_0*(Mrho[0]+Mrho[1]))*2.0;
    Trial[2] = (Mrho[0]*Mrho[1]*dVz/dt + Mrho[0]*Rz[1] - Mrho[1]*Rz[0])/(a_0*(Mrho[0]+Mrho[1]))*2.0;

    float vec_n [3];
    float vec_s1[3];
    float vec_s2[3];
    float vec_n0;

    vec_s1[0] = f_coef_d.vec_s1[iptr_f * 3 + 0];
    vec_s1[1] = f_coef_d.vec_s1[iptr_f * 3 + 1];
    vec_s1[2] = f_coef_d.vec_s1[iptr_f * 3 + 2];
    vec_s2[0] = f_coef_d.vec_s2[iptr_f * 3 + 0];
    vec_s2[1] = f_coef_d.vec_s2[iptr_f * 3 + 1];
    vec_s2[2] = f_coef_d.vec_s2[iptr_f * 3 + 2];

    vec_n[0] = xi_x[iptr];
    vec_n[1] = xi_y[iptr];
    vec_n[2] = xi_z[iptr];
    vec_n0 = fdlib_math_norm3(vec_n);

    jacvec = jac3d[iptr] * vec_n0;

    for (int i=0; i<3; i++)
    {
        vec_n[i] /= vec_n0;
    }

    Trial_local[0] = Trial[0]/jacvec + F.T0x[iptr_t];
    Trial_local[1] = Trial[1]/jacvec + F.T0y[iptr_t];
    Trial_local[2] = Trial[2]/jacvec + F.T0z[iptr_t];

    float Trial_n = fdlib_math_dot_product(Trial_local, vec_n);
    // Ts = T - n * Tn
    Trial_s[0] = Trial_local[0] - vec_n[0]*Trial_n;
    Trial_s[1] = Trial_local[1] - vec_n[1]*Trial_n;
    Trial_s[2] = Trial_local[2] - vec_n[2]*Trial_n;


    float Trial_s0;
    Trial_s0 = fdlib_math_norm3(Trial_s);

    float Tau_n = Trial_n;
    float Tau_s = Trial_s0;

    float ifchange = 0; // false
    if(Trial_n >= 0.0){
      // fault can not open
      Tau_n = 0.0;
      ifchange = 1;
    }else{
      Tau_n = Trial_n;
    }

    float mu_s = f_d.mu_s[iptr_t];
    float mu_d = f_d.mu_d[iptr_t];
    float slip = f_d.slip[iptr_t];
    float Dc = f_d.Dc[iptr_t];
    float C0 = f_d.C0[iptr_t];
    float friction;

    // slip weakening
    if(slip <= Dc){
      friction = mu_s - (mu_s - mu_d) * slip / Dc;
    }else{
      friction = mu_d;
    }

    //real_t C0 = 0;
    float Tau_c = -friction * Tau_n + C0;

    if(Trial_s0 >= Tau_c){
      Tau_s = Tau_c; // can not exceed shear strengh!
      f_d.flag_rup[iptr_t] = 1;
      ifchange = 1;
    }else{
      Tau_s = Trial_s0;
      f_d.flag_rup[iptr_t] = 0;
    }

    float Tau[3];
    if(ifchange){
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

    if(ifchange){
      f_wav_d.T11[iptr_f+3*siz_slice_yz] = (Tau[0] - f_d.T0x[iptr_t])*jacvec;
      f_wav_d.T12[iptr_f+3*siz_slice_yz] = (Tau[1] - f_d.T0y[iptr_t])*jacvec;
      f_wav_d.T13[iptr_f+3*siz_slice_yz] = (Tau[2] - f_d.T0z[iptr_t])*jacvec;
    }else{
      f_wav_d.T11[iptr_f+3*siz_slice_yz] = Trial[0];
      f_wav_d.T12[iptr_f+3*siz_slice_yz] = Trial[1];
      f_wav_d.T13[iptr_f+3*siz_slice_yz] = Trial[2];
    }

    f_d.tTs1[iptr_t] = fdlib_math_dot_product(Tau, vec_s1);
    f_d.tTs2[iptr_t] = fdlib_math_dot_product(Tau, vec_s2);
    f_d.tTn [iptr_t] = Tau_n;

    if(f_d.init_t0_flag[iptr_t] == 0) {
      if (f_d.hslip[iptr_t] > 1e-3) {
        f_d.init_t0[iptr_t] = it * dt;
        f_d.init_t0_flag[iptr_t] = 1;
        f_d.flag_rup[iptr_t] = 1;
      }
    }
  } 
  return;
}
