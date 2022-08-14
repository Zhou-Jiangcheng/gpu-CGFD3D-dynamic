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
             gdcurv_metric_t  metric_d,
             wav_t  wav_d,
             fault_wav_t  f_wav_d)
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

  int iptr, iptr_f, iptr_t;
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

  if ( iy < nj && iz < nk && fault.united[iy + iz * nj] == 0)
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
      rho = f.rho_f[(iy+3) + (iz+3) * ny + m * ny * nz];

      iptr_f = (iy+3) + (iz+3) * ny; 
      // fault traction image method by zhang wenqiang 
      for (int l = 1; l <= 3; l++)
      {
        iptr = (i0+(2*m-1)*l) + (iy+3) * siz_line + (iz+3) * siz_slice;
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

      float *T21_ptr = f_T21 + m*siz_slice_yz + iptr_f;
      float *T22_ptr = f_T22 + m*siz_slice_yz + iptr_f;
      float *T23_ptr = f_T23 + m*siz_slice_yz + iptr_f;
      // fault point use short stencil
      // due to slip change sharp
      if(fault.rup_index_y[iy + iz * nj] == 1)
      {
        M_FD_SHIFT_PTR_MAC22(DyT21, T21_ptr, 1, jdir);
        M_FD_SHIFT_PTR_MAC22(DyT22, T22_ptr, 1, jdir);
        M_FD_SHIFT_PTR_MAC22(DyT23, T23_ptr, 1, jdir);
      }
      if(fault.rup_index_y[iy + iz * nj] == 0)
      {
        M_FD_SHIFT_PTR_MACDRP(DyT21, T21_ptr, 1, jdir);
        M_FD_SHIFT_PTR_MACDRP(DyT22, T22_ptr, 1, jdir);
        M_FD_SHIFT_PTR_MACDRP(DyT23, T23_ptr, 1, jdir);
      }

      float *T31_ptr = f_T31 + m*siz_slice_yz + iptr_f;
      float *T32_ptr = f_T32 + m*siz_slice_yz + iptr_f;
      float *T33_ptr = f_T33 + m*siz_slice_yz + iptr_f;
      if(fault.rup_index_z[iy + iz * nj] == 1)
      {
        M_FD_SHIFT_PTR_MAC22(DzT31, T31_ptr, ny, kdir);
        M_FD_SHIFT_PTR_MAC22(DzT32, T32_ptr, ny, kdir);
        M_FD_SHIFT_PTR_MAC22(DzT33, T33_ptr, ny, kdir);
      }
      if(fault.rup_index_z[iy + iz * nj] == 0)
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
          iptr_t = (iy+3) + (iz+3+l) * ny;
          vecT31[l+3] = f_T31[iptr_t + m*siz_slice_yz];
          vecT32[l+3] = f_T32[iptr_t + m*siz_slice_yz];
          vecT33[l+3] = f_T33[iptr_t + m*siz_slice_yz];
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
      } 
      if(kdir == 1)
      {
        // -1 ~ 3, vecT pointer is +2
        M_FD_NOINDEX(DzT31, vecT31+2, kdir);
        M_FD_NOINDEX(DzT32, vecT32+2, kdir);
        M_FD_NOINDEX(DzT33, vecT33+2, kdir);
      }
      if(kdir == 0)
      {
        M_FD_NOINDEX(DzT31, vecT31, kdir);
        M_FD_NOINDEX(DzT32, vecT32, kdir);
        M_FD_NOINDEX(DzT33, vecT33, kdir);
      }
      
      float a_1 = 1.541764761036000;
      float a_2 = -0.333411808829999;
      float a_3 = 0.0416862855405999;
      if (m == 0){ // "-" side
        Rx[m] =
          a_1 * f.T11[2*siz_slice_yz+iptr_f] +
          a_2 * f.T11[1*siz_slice_yz+iptr_f] +
          a_3 * f.T11[0*siz_slice_yz+iptr_f] ;
        Ry[m] =
          a_1 * f.T12[2*siz_slice_yz+iptr_f] +
          a_2 * f.T12[1*siz_slice_yz+iptr_f] +
          a_3 * f.T12[0*siz_slice_yz+iptr_f] ;
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

      Mrho[m] = 0.5*jac*rho;

    } // end m

    // dv = (V+) - (V-)
    float dVx = f_mVx[iptr_f + siz_slice_yz] - f_mVx[iptr_f];
    float dVy = f_mVy[iptr_f + siz_slice_yz] - f_mVy[iptr_f];
    float dVz = f_mVz[iptr_f + siz_slice_yz] - f_mVz[iptr_f];

    float Trial[3];       // stress variation
    float Trial_local[3]; // + init background stress
    float Trial_s[3];     // shear stress

    float  a_0 = 1.250039237746600;
    Trial[0] = (Mrho[0]*Mrho[1]*dVx/dt + Mrho[0]*Rx[1] - Mrho[1]*Rx[0])/(a_0*(Mrho[0]+Mrho[1]))*2.0;
    Trial[1] = (Mrho[0]*Mrho[1]*dVy/dt + Mrho[0]*Ry[1] - Mrho[1]*Ry[0])/(a_0*(Mrho[0]+Mrho[1]))*2.0;
    Trial[2] = (Mrho[0]*Mrho[1]*dVz/dt + Mrho[0]*Rz[1] - Mrho[1]*Rz[0])/(a_0*(Mrho[0]+Mrho[1]))*2.0;

    float vec_n [3];
    float vec_s1[3];
    float vec_s2[3];


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

