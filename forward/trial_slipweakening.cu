#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "params.h"
#include "macdrp.h"

__global__ void 
trial_sw_gpu(
    )
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  int ni = gdinfo_d.ni;
  int nj = gdinfo_d.nj;
  int nk = gdinfo_d.nk;
  int nx = gdinfo_d.nx;
  int ny = gdinfo_d.ny;
  int nz = gdinfo_d.nz;
  int npoint_x = gdinfo_d.npoint_x;
  int i0  = (npoint_x/2+3); //fault index with ghost 

  size_t siz_line   = gdinfo_d.siz_line;
  size_t siz_slice  = gdinfo_d.siz_slice;
  size_t siz_volume = gdinfo_d.siz_volume;
  size_t siz_slice_yz = ny * nz;
  float xix, xiy, xiz;
  float jac;
  float vec_n0;
  float jacvec;
  float rho;
  float Mrho[2], Rx[2], Ry[2], Rz[2];
  float T11, T12, T13;
  int iptr, iptr_f, iptr_t;
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

  float *f_mVx  = F_d.mW + 0 * (2*siz_slice_yz);
  float *f_mVy  = F_d.mW + 1 * (2*siz_slice_yz);
  float *f_mVz  = F_d.mW + 2 * (2*siz_slice_yz);

  float *f_tVx  = F_d.tW + 0 * (2*siz_slice_yz);
  float *f_tVy  = F_d.tW + 1 * (2*siz_slice_yz);
  float *f_tVz  = F_d.tW + 2 * (2*siz_slice_yz);

  if ( j < nj && k < nk && fault.united[j + k * nj] == 0)
  { 
    for (int m = 0; m < 2; m++)
    {
      // m = 0 -> minus
      // m = 1 -> plus
      iptr = i0 + (j+3) * siz_line + (k+3) * siz_slice; 
      xix = xi_x [iptr];
      xiy = xi_y [iptr];
      xiz = xi_z [iptr];
      jac = jac3d[iptr];
      rho = f.rho_f[(j+3) + (k+3) * ny + m * ny * nz];

      iptr_f = (j+3) + (k+3) * ny; 
      for (int l = 1; l <= 3; l++)
      {
        iptr = (i0+(2*m-1)*l) + (j+3) * siz_line + (k+3) * siz_slice;
        xix = xi_x[iptr];
        xiy = xi_y[iptr];
        xiz = xi_z[iptr];
        jac = jac3d[iptr];
        T11 = jac*(xix * Txx[iptr] + xiy * Txy[iptr] + xiz * Txz[iptr]);
        T12 = jac*(xix * Txy[iptr] + xiy * Tyy[iptr] + xiz * Tyz[iptr]);
        T13 = jac*(xix * Txz[iptr] + xiy * Tyz[iptr] + xiz * Tzz[iptr]);
        f_wav_d.T11[(3+(2*m-1)*l)*siz_slice_yz + iptr_f] = T11;
        f_wav_d.T12[(3+(2*m-1)*l)*siz_slice_yz + iptr_f] = T12;
        f_wav_d.T13[(3+(2*m-1)*l)*siz_slice_yz + iptr_f] = T13;
      }

      float *T21 = f_T21 + m*siz_slice_yz;
      float *T22 = f_T22 + m*siz_slice_yz;
      float *T23 = f_T23 + m*siz_slice_yz;
      // fault point use short stencil
      // due to slip change sharp
      if(fault.rup_index_y[j + k * nj] == 1)
      {
        M_FD_SHIFT(DyT21, T21, iptr_f, fdy_len, lfdy_shift, lfdy_coef, n_fd);
        M_FD_SHIFT(DyT22, T21, iptr_f, fdy_len, lfdy_shift, lfdy_coef, n_fd);
        M_FD_SHIFT(DyT23, T21, iptr_f, fdy_len, lfdy_shift, lfdy_coef, n_fd);
      }
      if(fault.rup_index_y[j + k * nj] == 0)
      {
        M_FD_SHIFT(DyT21, T21, iptr_f, fdy_len, lfdy_shift, lfdy_coef, n_fd);
        M_FD_SHIFT(DyT22, T21, iptr_f, fdy_len, lfdy_shift, lfdy_coef, n_fd);
        M_FD_SHIFT(DyT23, T21, iptr_f, fdy_len, lfdy_shift, lfdy_coef, n_fd);
      }

      for (int l = -3; l <=3 ; l++){
        iptr_f = (j+3) + (k+3+l) * ny;
        vecT31[l+3] = f_T31[iptr_f + m*nyz];
        vecT32[l+3] = f_T32[iptr_f + m*nyz];
        vecT33[l+3] = f_T33[iptr_f + m*nyz];
      }

      int km = nk - k; // nk2-3, nk2-2, nk2-1 ==> (3, 2, 1)
      if(par.freenode && km<=3){
        vecT31[km+2] = 0.0;
        vecT32[km+2] = 0.0;
        vecT33[km+2] = 0.0;
        for (int l = km+3; l<7; l++){
          vecT31[l] = -vecT31[2*(km+2)-l];
          vecT32[l] = -vecT32[2*(km+2)-l];
          vecT33[l] = -vecT33[2*(km+2)-l];
        }
      } // end par.freenode
      if(f.rup_index_z[j + k * nj] % 7){
        DzT31 = vec_L22(vecT31, 3, FlagZ) * rDH;
        DzT32 = vec_L22(vecT32, 3, FlagZ) * rDH;
        DzT33 = vec_L22(vecT33, 3, FlagZ) * rDH;
      }else{
        DzT31 = vec_L(vecT31, 3, FlagZ) * rDH;
        DzT32 = vec_L(vecT32, 3, FlagZ) * rDH;
        DzT33 = vec_L(vecT33, 3, FlagZ) * rDH;
      }

      if (m == 0){ // "-" side
        Rx[m] =
          a_1 * f.T11[2*nyz+pos_f] +
          a_2 * f.T11[1*nyz+pos_f] +
          a_3 * f.T11[0*nyz+pos_f] ;
        Ry[m] =
          a_1 * f.T12[2*nyz+pos_f] +
          a_2 * f.T12[1*nyz+pos_f] +
          a_3 * f.T12[0*nyz+pos_f] ;
        Rz[m] =
          a_1 * f.T13[2*nyz+pos_f] +
          a_2 * f.T13[1*nyz+pos_f] +
          a_3 * f.T13[0*nyz+pos_f] ;
      }else{ // "+" side
        Rx[m] =
          a_1 * f.T11[4*nyz+pos_f] +
          a_2 * f.T11[5*nyz+pos_f] +
          a_3 * f.T11[6*nyz+pos_f] ;
        Ry[m] =
          a_1 * f.T12[4*nyz+pos_f] +
          a_2 * f.T12[5*nyz+pos_f] +
          a_3 * f.T12[6*nyz+pos_f] ;
        Rz[m] =
          a_1 * f.T13[4*nyz+pos_f] +
          a_2 * f.T13[5*nyz+pos_f] +
          a_3 * f.T13[6*nyz+pos_f] ;
      }

      Rx[m] = 0.5f*( (2*m-1)*Rx[m] + (DyT21 + DzT31)*DH )*DH2;
      Ry[m] = 0.5f*( (2*m-1)*Ry[m] + (DyT22 + DzT32)*DH )*DH2;
      Rz[m] = 0.5f*( (2*m-1)*Rz[m] + (DyT23 + DzT33)*DH )*DH2;

      Mrho[m] = 0.5f*jac*rho*DH*DH2;

    } // end m

    real_t t;
    if(irk == 0){
      t = it*DT;
    }else if (irk == 1 || irk == 2){
      t = (it+0.5)*DT;
    }else{
      t = (it+1)*DT;
    }
    pos = j1 + k1 * ny;
    real_t dVx = f_mVx[pos + nyz] - f_mVx[pos];
    real_t dVy = f_mVy[pos + nyz] - f_mVy[pos];
    real_t dVz = f_mVz[pos + nyz] - f_mVz[pos];

    real_t Trial[3]; // stress variation
    real_t Trial_local[3]; // + init background stress
    real_t Trial_s[3]; // shear stress

    real_t dt1 = DT;

    real_t a0p,a0m;
    if(FlagX==FWD){
      a0p = a_0pF;
      a0m = a_0mF;
    }else{
      a0p = a_0pB;
      a0m = a_0mB;
    }
    Trial[0] = -(Mrho[0]*Mrho[1]*dVx/dt1 + Mrho[0]*Rx[1] - Mrho[1]*Rx[0])/(DH2*(Mrho[0]*a0p-a0m*Mrho[1]))*2.0f;
    Trial[1] = -(Mrho[0]*Mrho[1]*dVy/dt1 + Mrho[0]*Ry[1] - Mrho[1]*Ry[0])/(DH2*(Mrho[0]*a0p-a0m*Mrho[1]))*2.0f;
    Trial[2] = -(Mrho[0]*Mrho[1]*dVz/dt1 + Mrho[0]*Rz[1] - Mrho[1]*Rz[0])/(DH2*(Mrho[0]*a0p-a0m*Mrho[1]))*2.0f;

    real_t vec_n [3];
    real_t vec_s1[3];
    real_t vec_s2[3];


    pos = (j1 + k1 * ny) * 3;

    vec_s1[0] = f.vec_s1[pos + 0];
    vec_s1[1] = f.vec_s1[pos + 1];
    vec_s1[2] = f.vec_s1[pos + 2];
    vec_s2[0] = f.vec_s2[pos + 0];
    vec_s2[1] = f.vec_s2[pos + 1];
    vec_s2[2] = f.vec_s2[pos + 2];

    pos = j1 + k1 * ny + i0 * ny * nz;
    vec_n[0] = XIX[pos];//M[pos + 0];
    vec_n[1] = XIY[pos];//M[pos + 1];
    vec_n[2] = XIZ[pos];//M[pos + 2];
    vec_n0 = norm3(vec_n);

    jacvec = JAC[pos] * vec_n0;

    for (int ii = 0; ii < 3; ++ii)
    {
        vec_n[ii] /= vec_n0;
    }

    pos = j + k * nj;
    Trial_local[0] = Trial[0]/jacvec + f.str_init_x[pos];
    Trial_local[1] = Trial[1]/jacvec + f.str_init_y[pos];
    Trial_local[2] = Trial[2]/jacvec + f.str_init_z[pos];

    real_t Trial_n = dot_product(Trial_local, vec_n);
    // Ts = T - n * Tn
    Trial_s[0] = Trial_local[0] - vec_n[0]*Trial_n;
    Trial_s[1] = Trial_local[1] - vec_n[1]*Trial_n;
    Trial_s[2] = Trial_local[2] - vec_n[2]*Trial_n;


    real_t Trial_s0;
    Trial_s0 = norm3(Trial_s);

    real_t Tau_n = Trial_n;
    real_t Tau_s = Trial_s0;

    real_t ifchange = 0; // false
    if(Trial_n >= 0.0){
      // fault can not open
      Tau_n = 0.0;
      ifchange = 1;
    }else{
      Tau_n = Trial_n;
    }



    pos = j + k * nj;
    real_t mu_s1 = f.str_peak[pos];
    real_t slip = f.slip[pos];
    real_t friction;
    real_t Dc = f.Dc[pos];
    real_t mu_d = f.mu_d[pos];

    // slip weakening
    if(slip <= Dc){
      friction = mu_s1 - (mu_s1 - mu_d) * slip / Dc;
    }else{
      friction = mu_d;
    }

    //real_t C0 = 0;
    real_t Tau_c = -friction * Tau_n + f.C0[pos];

    if(Trial_s0 >= Tau_c){
      Tau_s = Tau_c; // can not exceed shear strengh!
      //f.flag_rup[j*nk + k] = 1;
      f.flag_rup[j + k * nj] = 1;
      ifchange = 1;
    }else{
      Tau_s = Trial_s0;
      //f.flag_rup[j*nk + k] = 0;
      f.flag_rup[j + k * nj] = 0;
    }

    real_t Tau[3];
    if(ifchange){
      // to avoid divide by 0, 1e-1 is a small value compared to stress
      if(fabsf(Trial_s0) < 1e-1){
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


    pos1 = j1 + k1 * ny + 3*nyz;
    pos  = j + k * nj;

    if(ifchange){
      f.T11[pos1] = (Tau[0] - f.str_init_x[pos])*jacvec;
      f.T12[pos1] = (Tau[1] - f.str_init_y[pos])*jacvec;
      f.T13[pos1] = (Tau[2] - f.str_init_z[pos])*jacvec;
    }else{
      f.T11[pos1] = Trial[0];
      f.T12[pos1] = Trial[1];
      f.T13[pos1] = Trial[2];
    }

    real_t hT11, hT12, hT13;
    real_t viscosity = par.viscosity * DT;

    real_t DT2 = DT;
    if(irk == 0){
      DT2 = 0*DT;
    }else if(irk == 1){
      DT2 = 0.5*DT;
    }else if(irk == 2){
      DT2 = 0.5*DT;
    }else if(irk == 3){
      DT2 = 1.0*DT;
    }

    if(irk==0){
      hT11 = f.hT11[j1+k1*ny];
      hT12 = f.hT12[j1+k1*ny];
      hT13 = f.hT13[j1+k1*ny];
    }else{
      // update
      hT11 = (f.T11[pos1] - f.mT11[pos1])/DT2;
      hT12 = (f.T12[pos1] - f.mT12[pos1])/DT2;
      hT13 = (f.T13[pos1] - f.mT13[pos1])/DT2;
    }
    hT11 = (f.T11[pos1] - f.mT11[pos1])/DT;
    hT12 = (f.T12[pos1] - f.mT12[pos1])/DT;
    hT13 = (f.T13[pos1] - f.mT13[pos1])/DT;

#ifndef DxV_hT1
    if(irk==3){
    f.hT11[j1+k1*ny] = hT11;
    f.hT12[j1+k1*ny] = hT12;
    f.hT13[j1+k1*ny] = hT13;
    }
#endif

    f.T11[pos1] += viscosity * hT11;
    f.T12[pos1] += viscosity * hT12;
    f.T13[pos1] += viscosity * hT13;

    pos = j + k * nj;
    f.tTs1[pos] = dot_product(Tau, vec_s1);
    f.tTs2[pos] = dot_product(Tau, vec_s2);
    f.tTn [pos] = Tau_n;

    pos = j + k * nj;
    if(!f.init_t0_flag[pos]) {
      if (f.hslip[pos] > 1e-3) {
        f.init_t0[pos] = it * DT;
        f.init_t0_flag[pos] = 1;
        f.flag_rup[pos] = 1;
      }
    }
  } // end j k
  return;
}

