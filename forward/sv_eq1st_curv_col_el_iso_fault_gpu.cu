#include <stdio.h>
#include <stdlib.h>

int 
sv_eq1st_curv_col_el_iso_fault_onestage(
                         float *w_cur_d,
                         float *w_rhs_d, 
                         float *f_cur_d,
                         float *f_rhs_d, 
                         int i0,
                         int isfree,
                         wav_t  wav_d,
                         fault_wav_t FW,
                         gdinfo_t  gdinfo_d,
                         gdcurv_metric_t metric_d,
                         md_t md_d,
                         bdryfree_t bdryfree_d,
                         fd_op_t *fdx_op,
                         fd_op_t *fdy_op,
                         fd_op_t *fdz_op,
                         const int myid, const int verbose)
{
  // local pointer get each vars
  float *Vx  = w_cur_d + wav_d.Vx_pos;
  float *Vy  = w_cur_d + wav_d.Vy_pos;
  float *Vz  = w_cur_d + wav_d.Vz_pos;
  float *Txx = w_cur_d + wav_d.Txx_pos;
  float *Tyy = w_cur_d + wav_d.Tyy_pos;
  float *Tzz = w_cur_d + wav_d.Tzz_pos;
  float *Txz = w_cur_d + wav_d.Txz_pos;
  float *Tyz = w_cur_d + wav_d.Tyz_pos;
  float *Txy = w_cur_d + wav_d.Txy_pos;

  float *hVx  = w_rhs_d + wav_d.Vx_pos;
  float *hVy  = w_rhs_d + wav_d.Vy_pos;
  float *hVz  = w_rhs_d + wav_d.Vz_pos;
  float *hTxx = w_rhs_d + wav_d.Txx_pos;
  float *hTyy = w_rhs_d + wav_d.Tyy_pos;
  float *hTzz = w_rhs_d + wav_d.Tzz_pos;
  float *hTxz = w_rhs_d + wav_d.Txz_pos;
  float *hTyz = w_rhs_d + wav_d.Tyz_pos;
  float *hTxy = w_rhs_d + wav_d.Txy_pos;

  float *f_Vx  = f_cur_d + FW.Vx_pos;
  float *f_Vy  = f_cur_d + FW.Vy_pos;
  float *f_Vz  = f_cur_d + FW.Vz_pos;
  float *f_T21 = f_cur_d + FW.T21_pos;
  float *f_T22 = f_cur_d + FW.T22_pos;
  float *f_T23 = f_cur_d + FW.T23_pos;
  float *f_T31 = f_cur_d + FW.T31_pos;
  float *f_T32 = f_cur_d + FW.T32_pos;
  float *f_T33 = f_cur_d + FW.T33_pos;

  float *f_hVx  = f_rhs_d + FW.Vx_pos;
  float *f_hVy  = f_rhs_d + FW.Vy_pos;
  float *f_hVz  = f_rhs_d + FW.Vz_pos;
  float *f_hT21 = f_rhs_d + FW.T21_pos;
  float *f_hT22 = f_rhs_d + FW.T22_pos;
  float *f_hT23 = f_rhs_d + FW.T23_pos;
  float *f_hT31 = f_rhs_d + FW.T31_pos;
  float *f_hT32 = f_rhs_d + FW.T32_pos;
  float *f_hT33 = f_rhs_d + FW.T33_pos;

  float *f_T11 = FW.T11;
  float *f_T12 = FW.T12;
  float *f_T13 = FW.T13;

  int nj = gdinfo_d.nj;
  int nk = gdinfo_d.nk;
  int ny = gdinfo_d.ny;

  size_t siz_line = gdinfo_d.siz_line;
  size_t siz_slice = gdinfo_d.siz_slice;
  size_t siz_slice_yz = gdinfo_d.siz_slice_yz;

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

  float *lam3d = md_d.lambda;
  float * mu3d = md_d.mu;
  float *slw3d = md_d.rho;

  int idir = fdx_op->dir;
  int jdir = fdy_op->dir;
  int kdir = fdz_op->dir;
  float *matVx2Vz = bdryfree_d.matVx2Vz;
  float *matVy2Vz = bdryfree_d.matVy2Vz;
  {
    dim3 block(8,8);
    dim3 grid;
    grid.x = (nj+block.x-1)/block.x;
    grid.y = (nk+block.y-1)/block.y;
    sv_eq1st_curv_col_el_iso_rhs_fault_velo_gpu <<<grid, block>>>(
                                   Txx, Tyy, Tzz, Txz, Tyz, Txy, hVx, hVy, hVz, 
                                   f_T21, f_T22, f_T23, f_T31, f_T32, f_T33,
                                   f_hVx, f_hVy, f_hVz, f_T11, f_T12, f_T13,
                                   xi_x,  xi_y, xi_z, et_x,  et_y, et_z, zt_x,  zt_y, zt_z,
                                   jac3d, slw3d, fault_t f_d, isfree, i0,
                                   nj1, nj, nk1, nk, ny, siz_line, siz_slice, siz_slice_yz,
                                   idir, jdir, kdir)
    CUDACHECK( cudaDeviceSynchronize() );
  }
  {
    dim3 block(8,8);
    dim3 grid;
    grid.x = (nj+block.x-1)/block.x;
    grid.y = (nk+block.y-1)/block.y;
    if(idir == 1) 
    {
      sv_eq1st_curv_col_el_iso_rhs_fault_stress_F_gpu <<<grid, block>>>(
                                   Vx, Vy, Vz, hTxx, hTyy, hTzz, hTxz, hTyz, hTxy,
                                   f_Vx, f_Vy, f_Vz,f_hT21, f_hT22, f_hT23,
                                   f_hT31, f_hT32, f_hT33, f_T11, f_T12, f_T13,
                                   xi_x, xi_y, xi_z, et_x, et_y, et_z, zt_x, zt_y, zt_z,
                                   lam3d, mu3d, slw3d, f_d, matVx2Vz, matVy2Vz,
                                   isfree, i0, nj1, nj, nk1, nk, ny, 
                                   siz_line, siz_slice, siz_slice_yz,
                                   idir, jdir, kdir)
    }
    if(idir == 0) 
    {
      sv_eq1st_curv_col_el_iso_rhs_fault_stress_B_gpu <<<grid, block>>>(
                                   Vx, Vy, Vz, hTxx, hTyy, hTzz, hTxz, hTyz, hTxy,
                                   f_Vx, f_Vy, f_Vz,f_hT21, f_hT22, f_hT23,
                                   f_hT31, f_hT32, f_hT33, f_T11, f_T12, f_T13,
                                   xi_x, xi_y, xi_z, et_x, et_y, et_z, zt_x, zt_y, zt_z,
                                   lam3d, mu3d, slw3d, f_d, matVx2Vz, matVy2Vz,
                                   isfree, i0, nj1, nj, nk1, nk, ny, 
                                   siz_line, siz_slice, siz_slice_yz,
                                   idir, jdir, kdir)
    }
    CUDACHECK( cudaDeviceSynchronize() );
  }

  return 0;
}

__global__
void sv_eq1st_curv_col_el_iso_rhs_fault_velo_gpu(
                       float * Txx, float * Tyy, float * Tzz,
                       float * Txz, float * Tyz, float * Txy,
                       float * hVx, float * hVy, float * hVz,
                       float * f_T21, float * f_T22, float * f_T23,
                       float * f_T31, float * f_T32, float * f_T33,
                       float * f_hVx, float * f_hVy, float * f_hVz,
                       float * f_T11, float * f_T12, float * f_T13,
                       float * xi_x,  float * xi_y, float * xi_z,
                       float * et_x,  float * et_y, float * et_z,
                       float * zt_x,  float * zt_y, float * zt_z,
                       float * jac3d, float * slw3d, 
                       fault_t f_d, int isfree, int i0,
                       int nj1, int nj, int nk1, int nk, int ny, 
                       size_t siz_line, size_t siz_slice, size_t siz_slice_yz,
                       int idir, int jdir, int kdir)
{
  size_t iy = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iz = blockIdx.y * blockDim.y + threadIdx.y;

  float rrhojac;
  size_t iptr, iptr_f;
  float *T21_ptr;
  float *T22_ptr;
  float *T23_ptr;
  float *T31_ptr;
  float *T32_ptr;
  float *T33_ptr;


  float vecT11[7], vecT12[7], vecT13[7];
  float vecT21[7], vecT22[7], vecT23[7];
  float vecT31[7], vecT32[7], vecT33[7];

  if (iy < nj && iz < nk ) 
  { 
    if(f_d.united[iy + iz * nj]) return;
    int km = nk - (iz+1); 
    int n_free = km+3;
    for (int i = i0-3; i <= i0+3; i++)
    {
      int n = i0 - i; 
      if(n==0) continue; // skip Split nodes
      for (int l = -3; l <= 3; l++)
      {
        iptr =(i+l) + (iy+nj1) * siz_line + (iz+nk1) * siz_slice;
        vecT11[l+3] = jac3d[iptr]*(xi_x[iptr]*Txx[iptr] + xi_y[iptr]*Txy[iptr] + xi_z[iptr]*Txz[iptr]);
        vecT12[l+3] = jac3d[iptr]*(xi_x[iptr]*Txy[iptr] + xi_y[iptr]*Tyy[iptr] + xi_z[iptr]*Tyz[iptr]);
        vecT13[l+3] = jac3d[iptr]*(xi_x[iptr]*Txz[iptr] + xi_y[iptr]*Tyz[iptr] + xi_z[iptr]*Tzz[iptr]);

        iptr = i + (iy+nj1+l) * siz_line + (iz+nk1) * siz_slice;
        vecT21[l+3] = jac3d[iptr]*(et_x[iptr]*Txx[iptr] + et_y[iptr]*Txy[iptr] + et_z[iptr]*Txz[iptr]);
        vecT22[l+3] = jac3d[iptr]*(et_x[iptr]*Txy[iptr] + et_y[iptr]*Tyy[iptr] + et_z[iptr]*Tyz[iptr]);
        vecT23[l+3] = jac3d[iptr]*(et_x[iptr]*Txz[iptr] + et_y[iptr]*Tyz[iptr] + et_z[iptr]*Tzz[iptr]);

        iptr = i + (iy+nj1) *siz_line + (iz+nk1+l) * siz_slice;
        vecT31[l+3] = jac3d[iptr]*(zt_x[iptr]*Txx[iptr] + zt_y[iptr]*Txy[iptr] + zt_z[iptr]*Txz[iptr]);
        vecT32[l+3] = jac3d[iptr]*(zt_x[iptr]*Txy[iptr] + zt_y[iptr]*Tyy[iptr] + zt_z[iptr]*Tyz[iptr]);
        vecT33[l+3] = jac3d[iptr]*(zt_x[iptr]*Txz[iptr] + zt_y[iptr]*Tyz[iptr] + zt_z[iptr]*Tzz[iptr]);
      }

      iptr_f = (iy+nj1) + (iz+nk1) * ny + 3 * siz_slice_yz; 
      vecT11[n+3] = f_T11[iptr_f]; // fault T11
      vecT12[n+3] = f_T12[iptr_f]; // fault T12
      vecT13[n+3] = f_T13[iptr_f]; // fault T13

      //  TractionImg
      if (n==2) { // i0-2
        vecT11[6] = 2.0*vecT11[5] - vecT11[4];
        vecT12[6] = 2.0*vecT12[5] - vecT12[4];
        vecT13[6] = 2.0*vecT13[5] - vecT13[4];
      }
      if (n==1) { // i0-1
        vecT11[5] = 2.0*vecT11[4] - vecT11[3];
        vecT12[5] = 2.0*vecT12[4] - vecT12[3];
        vecT13[5] = 2.0*vecT13[4] - vecT13[3];
        vecT11[6] = 2.0*vecT11[4] - vecT11[2];
        vecT12[6] = 2.0*vecT12[4] - vecT12[2];
        vecT13[6] = 2.0*vecT13[4] - vecT13[2];
      }
      if (n==-1) { // i0+1
        vecT11[0] = 2.0*vecT11[2] - vecT11[4];
        vecT12[0] = 2.0*vecT12[2] - vecT12[4];
        vecT13[0] = 2.0*vecT13[2] - vecT13[4];
        vecT11[1] = 2.0*vecT11[2] - vecT11[3];
        vecT12[1] = 2.0*vecT12[2] - vecT12[3];
        vecT13[1] = 2.0*vecT13[2] - vecT13[3];
      }
      if (n==-2) { // i0+2
        vecT11[0] = 2.0*vecT11[1] - vecT11[2];
        vecT12[0] = 2.0*vecT12[1] - vecT12[2];
        vecT13[0] = 2.0*vecT13[1] - vecT13[2];
      }

      if(isfree == 1 && km<=3)
      {
        vecT31[n_free] = 0.0;
        vecT32[n_free] = 0.0;
        vecT33[n_free] = 0.0;
        for (int l = n_free+1; l<7; l++){
          vecT31[l] = -vecT31[2*n_free-l];
          vecT32[l] = -vecT32[2*n_free-l];
          vecT33[l] = -vecT33[2*n_free-l];
        }
      }

      M_FD_VEC(DxT11, vecT11+3, idir);
      M_FD_VEC(DxT12, vecT12+3, idir);
      M_FD_VEC(DxT13, vecT13+3, idir);
      M_FD_VEC(DyT21, vecT21+3, jdir);
      M_FD_VEC(DyT22, vecT22+3, jdir);
      M_FD_VEC(DyT23, vecT23+3, jdir);
      M_FD_VEC(DzT31, vecT31+3, kdir);
      M_FD_VEC(DzT32, vecT32+3, kdir);
      M_FD_VEC(DzT33, vecT33+3, kdir);

      iptr = i + (iy+nj1) * siz_line + (iz+nk1) * siz_slice;
      rrhojac = slw3d[iptr] / jac3d[iptr];

      hVx[iptr] = (DxT11+DyT21+DzT31)*rrhojac;
      hVy[iptr] = (DxT12+DyT22+DzT32)*rrhojac;
      hVz[iptr] = (DxT13+DyT23+DzT33)*rrhojac;
    } // end of loop i

    // update velocity at the fault plane
    // 0 for minus side on the fault
    // 1 for plus  side on the fault
    for (int m = 0; m < 2; m++)
    {
      iptr_f = (iy+nj1) + (iz+nk1) * ny
      if(m==0){ // "-" side
        DxT11 =     a_0*f_T11[iptr_f+3*siz_slice_yz] 
                  - a_1*f_T11[iptr_f+2*siz_slice_yz] 
                  - a_2*f_T11[iptr_f+1*siz_slice_yz] 
                  - a_3*f_T11[iptr_f+0*siz_slice_yz];

        DxT12 =     a_0*f_T12[iptr_f+3*siz_slice_yz]
                  - a_1*f_T12[iptr_f+2*siz_slice_yz]
                  - a_2*f_T12[iptr_f+1*siz_slice_yz]
                  - a_3*f_T12[iptr_f+0*siz_slice_yz];

        DxT13 =     a_0*f_T13[iptr_f+3*siz_slice_yz]
                  - a_1*f_T13[iptr_f+2*siz_slice_yz]
                  - a_2*f_T13[iptr_f+1*siz_slice_yz]
                  - a_3*f_T13[iptr_f+0*siz_slice_yz];
      }else{ // "+" side
        DxT11 =   - a_0*f_T11[iptr_f+3*siz_slice_yz]
                  + a_1*f_T11[iptr_f+4*siz_slice_yz]
                  + a_2*f_T11[iptr_f+5*siz_slice_yz]
                  + a_3*f_T11[iptr_f+6*siz_slice_yz];

        DxT12 =   - a_0*f_T12[iptr_f+3*siz_slice_yz]
                  + a_1*f_T12[iptr_f+4*siz_slice_yz]
                  + a_2*f_T12[iptr_f+5*siz_slice_yz]
                  + a_3*f_T12[iptr_f+6*siz_slice_yz];

        DxT13 =   - a_0*f_T13[iptr_f+3*siz_slice_yz]
                  + a_1*f_T13[iptr_f+4*siz_slice_yz]
                  + a_2*f_T13[iptr_f+5*siz_slice_yz]
                  + a_3*f_T13[iptr_f+6*siz_slice_yz];
      }
      T21_ptr = f_T21 + m*siz_slice_yz + iptr_f;
      T22_ptr = f_T22 + m*siz_slice_yz + iptr_f;
      T23_ptr = f_T23 + m*siz_slice_yz + iptr_f;
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

      T31_ptr = f_T31 + m*siz_slice_yz + iptr_f;
      T32_ptr = f_T32 + m*siz_slice_yz + iptr_f;
      T33_ptr = f_T33 + m*siz_slice_yz + iptr_f;
      if(f_d.rup_index_z[iy+iz*nj] == 1)
      {
        M_FD_SHIFT_PTR_MAC22(DzT31, T31_ptr, ny, kdir);
        M_FD_SHIFT_PTR_MAC22(DzT32, T32_ptr, ny, kdir);
        M_FD_SHIFT_PTR_MAC22(DzT33, T33_ptr, ny, kdir);
      }
      if(f_d.rup_index_z[iy+iz*nj] == 0)
      {
        M_FD_SHIFT_PTR_MACDRP(DzT31, T31_ptr, ny, kdir);
        M_FD_SHIFT_PTR_MACDRP(DzT32, T32_ptr, ny, kdir);
        M_FD_SHIFT_PTR_MACDRP(DzT33, T33_ptr, ny, kdir);
      }
      if(isfree == 1 && km<=3)
      {
        for (int l=-3; l<=3; l++)
        {
          iptr_f = (iy+3) + (iz+3+l) * ny + m * siz_slice_yz;
          vecT31[l+3] = f_T31[iptr_f];
          vecT32[l+3] = f_T32[iptr_f];
          vecT33[l+3] = f_T33[iptr_f];
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

      iptr = i0 + (iy+nj1) * siz_line + (iz+nk1) * siz_slice;
      rrhojac = slw3d[iptr] / jac3d[iptr];

      iptr_f = (iy+nj1) + (iz+nk1) * ny + m * siz_slice_yz; 
      f_hVx[iptr_f] = (DxT11+DyT21+DzT31)*rrhojac;
      f_hVy[iptr_f] = (DxT12+DyT22+DzT32)*rrhojac;
      f_hVz[iptr_f] = (DxT13+DyT23+DzT33)*rrhojac;
    } 
  } 
  return;
}

__global__
void sv_eq1st_curv_col_el_iso_rhs_fault_stress_F_gpu(
                       float * Vx, float * Vy, float * Vz,
                       float * hTxx, float * hTyy, float * hTzz,
                       float * hTxz, float * hTyz, float * hTxy,
                       float * f_Vx, float * f_Vy, float * f_Vz,
                       float * f_hT21, float * f_hT22, float * f_hT23,
                       float * f_hT31, float * f_hT32, float * f_hT33,
                       float * f_T11, float * f_T12, float * f_T13,
                       float * xi_x,  float * xi_y, float * xi_z,
                       float * et_x,  float * et_y, float * et_z,
                       float * zt_x,  float * zt_y, float * zt_z,
                       float * lam3d, float * mu3d, float * slw3d, 
                       fault_t f_d, float *matVx2Vz, float *matVy2Vz,
                       int isfree, int i0,
                       int nj1, int nj, int nk1, int nk, int ny, 
                       size_t siz_line, size_t siz_slice, size_t siz_slice_yz,
                       int idir, int jdir, int kdir)
{
  int iy = blockIdx.x * blockDim.x + threadIdx.x;
  int iz = blockIdx.y * blockDim.y + threadIdx.y;

  float *Vx_ptr;
  float *Vy_ptr;
  float *Vz_ptr;
  float *f_Vx_ptr;
  float *f_Vy_ptr;
  float *f_Vz_ptr;
  float xix, xiy, xiz;
  float etx, ety, etz;
  float ztx, zty, ztz;
  float mu, lam, lam2mu;

  float DxVx[8],DxVy[8],DxVz[8];
  float DyVx[8],DyVy[8],DyVz[8];
  float DzVx[8],DzVy[8],DzVz[8];
  float mat1[3][3], mat2[3][3], mat3[3][3];
  float vec_3[3], vec_5[5];
  float vec1[3], vec2[3], vec3[3];
  float vecg1[3], vecg2[3], vecg3[3];
  float dtT1[3];
  float dxV1[3], dyV1[3], dzV1[3];
  float dxV2[3], dyV2[3], dzV2[3];
  float out1[3], out2[3], out3[3], out4[3], out5[3];

  float matT1toVxm[3][3];
  float matVytoVxm[3][3];
  float matVztoVxm[3][3];
  float matT1toVxp[3][3];
  float matVytoVxp[3][3];
  float matVztoVxp[3][3];

  float matT1toVxfm[3][3];
  float matVytoVxfm[3][3];
  float matT1toVxfp[3][3];
  float matVytoVxfp[3][3];

  float matPlus2Min1[3][3];
  float matPlus2Min2[3][3];
  float matPlus2Min3[3][3];
  float matPlus2Min4[3][3];
  float matPlus2Min5[3][3];

  float matPlus2Min1f[3][3], matPlus2Min2f[3][3], matPlus2Min3f[3][3];
  float matVx2Vz1[3][3], matVy2Vz1[3][3];
  float matVx2Vz2[3][3], matVy2Vz2[3][3];

  int idx = ((iy+3) + (iz+3) * ny)*3*3;
  if (iy < nj && iz < nk ) { 
    // non united
    if(F.united[iy + iz * nj]) return;
    int km = nk - (iz+1); 
    int n_free = km + 3;
    //
    //     Update Stress (in Zhang's Thesis, 2014)
    // ---V-----V-----V-----0-----0-----V-----V-----V---  (grid point)
    //    G     F     E     D-    D+    C     B     A     (grid name in thesis)
    //    i0-3  i0-2  i0-1  i0-0  i0+0  i0+1  i0+2  i0+3  (3D grid index)
    //    0     1     2     3     4     5     6     7     (vec grid index)
    //    -3    -2    -1    -0    +0    1     2     3     (offset from fault)

    // point A
    iptr = (i0+3) + (iy+nj1) * siz_line + (iz+nk1) *siz_slice;
    Vx_ptr = Vx + iptr;
    Vy_ptr = Vy + iptr;
    Vz_ptr = Vz + iptr;
    MACDRP_F(DxVx[7],Vx_ptr,1);
    MACDRP_F(DxVy[7],Vy_ptr,1);
    MACDRP_F(DxVz[7],Vz_ptr,1);

    // point B
    iptr = (i0+2) + (iy+nj1) * siz_line + (iz+nk1) *siz_slice;
    Vx_ptr = Vx + iptr;
    Vy_ptr = Vy + iptr;
    Vz_ptr = Vz + iptr;
    MAC24_F(DxVx[6],Vx_ptr,1);
    MAC24_F(DxVy[6],Vy_ptr,1);
    MAC24_F(DxVz[6],Vz_ptr,1);

    // point C
    iptr = (i0+1) + (iy+nj1) * siz_line + (iz+nk1) *siz_slice;
    Vx_ptr = Vx + iptr;
    Vy_ptr = Vy + iptr;
    Vz_ptr = Vz + iptr;
    MAC22_F(DxVx[5],Vx_ptr,1);
    MAC22_F(DxVy[5],Vy_ptr,1);
    MAC22_F(DxVz[5],Vz_ptr,1);

    //fault split point D+ D-
    if(f_d.rup_index_y[iy + iz * nj] == 1)
    {
      iptr_f = (iy+nj1) + (iz+nk1) * ny + 1 * siz_slice_yz;
      f_Vx_ptr = f_Vx + iptr_f;
      f_Vy_ptr = f_Vy + iptr_f;
      f_Vz_ptr = f_Vz + iptr_f;
      MAC22_F(DyVx[4],f_Vx_ptr,siz_line);
      MAC22_F(DyVy[4],f_Vy_ptr,siz_line);
      MAC22_F(DyVz[4],f_Vz_ptr,siz_line);
      iptr_f = (iy+nj1) + (iz+nk1) * ny + 0 * siz_slice_yz;
      f_Vx_ptr = f_Vx + iptr_f;
      f_Vy_ptr = f_Vy + iptr_f;
      f_Vz_ptr = f_Vz + iptr_f;
      MAC22_F(DyVx[3],f_Vx_ptr,siz_line);
      MAC22_F(DyVy[3],f_Vy_ptr,siz_line);
      MAC22_F(DyVz[3],f_Vz_ptr,siz_line);
    }
    if(f_d.rup_index_y[iy + iz * nj] == 0)
    {
      iptr_f = (iy+nj1) + (iz+nk1) * ny + 1 * siz_slice_yz;
      f_Vx_ptr = f_Vx + iptr_f;
      f_Vy_ptr = f_Vy + iptr_f;
      f_Vz_ptr = f_Vz + iptr_f;
      MACDRP_F(DyVx[4],f_Vx_ptr,siz_line);
      MACDRP_F(DyVy[4],f_Vy_ptr,siz_line);
      MACDRP_F(DyVz[4],f_Vz_ptr,siz_line);
      iptr_f = (iy+nj1) + (iz+nk1) * ny + 0 * siz_slice_yz;
      f_Vx_ptr = f_Vx + iptr_f;
      f_Vy_ptr = f_Vy + iptr_f;
      f_Vz_ptr = f_Vz + iptr_f;
      MACDRP_F(DyVx[3],f_Vx_ptr,siz_line);
      MACDRP_F(DyVy[3],f_Vy_ptr,siz_line);
      MACDRP_F(DyVz[3],f_Vz_ptr,siz_line);
    }
    if(f_d.rup_index_z[iy + iz * nj] == 1){
      iptr_f = (iy+3) + (iz+3) * ny + 1 * siz_slice_yz;
      f_Vx_ptr = f_Vx + iptr_f;
      f_Vy_ptr = f_Vy + iptr_f;
      f_Vz_ptr = f_Vz + iptr_f;
      MAC22_F(DzVx[4],f_Vx_ptr,siz_slice);
      MAC22_F(DzVy[4],f_Vy_ptr,siz_slice);
      MAC22_F(DzVz[4],f_Vz_ptr,siz_slice);
      iptr_f = (iy+3) + (iz+3) * ny + 0 * siz_slice_yz;
      f_Vx_ptr = f_Vx + iptr_f;
      f_Vy_ptr = f_Vy + iptr_f;
      f_Vz_ptr = f_Vz + iptr_f;
      MAC22_F(DzVx[3],f_Vx_ptr,siz_slice);
      MAC22_F(DzVy[3],f_Vy_ptr,siz_slice);
      MAC22_F(DzVz[3],f_Vz_ptr,siz_slice);
    }
    if(f_d.rup_index_z[iy + iz * nj] == 0){
      iptr_f = (iy+3) + (iz+3) * ny + 1 * siz_slice_yz;
      f_Vx_ptr = f_Vx + iptr_f;
      f_Vy_ptr = f_Vy + iptr_f;
      f_Vz_ptr = f_Vz + iptr_f;
      MACDRP_F(DzVx[4],f_Vx_ptr,siz_slice);
      MACDRP_F(DzVy[4],f_Vy_ptr,siz_slice);
      MACDRP_F(DzVz[4],f_Vz_ptr,siz_slice);
      iptr_f = (iy+3) + (iz+3) * ny + 0 * siz_slice_yz;
      f_Vx_ptr = f_Vx + iptr_f;
      f_Vy_ptr = f_Vy + iptr_f;
      f_Vz_ptr = f_Vz + iptr_f;
      MACDRP_F(DzVx[3],f_Vx_ptr,siz_slice);
      MACDRP_F(DzVy[3],f_Vy_ptr,siz_slice);
      MACDRP_F(DzVz[3],f_Vz_ptr,siz_slice);
    }

    for (int ii = 0; ii < 3; ii++){
      for (int jj = 0; jj < 3; jj++){
        matPlus2Min1[ii][jj] = FC.matPlus2Min1[idx + ii*3 + jj];
        matPlus2Min2[ii][jj] = FC.matPlus2Min2[idx + ii*3 + jj];
        matPlus2Min3[ii][jj] = FC.matPlus2Min3[idx + ii*3 + jj];
        matPlus2Min4[ii][jj] = FC.matPlus2Min4[idx + ii*3 + jj];
        matPlus2Min5[ii][jj] = FC.matPlus2Min5[idx + ii*3 + jj];

        matT1toVxm[ii][jj] = FC.matT1toVxm[idx + ii*3 + jj];
        matVytoVxm[ii][jj] = FC.matVytoVxm[idx + ii*3 + jj];
        matVztoVxm[ii][jj] = FC.matVztoVxm[idx + ii*3 + jj];
        matT1toVxp[ii][jj] = FC.matT1toVxp[idx + ii*3 + jj];
        matVytoVxp[ii][jj] = FC.matVytoVxp[idx + ii*3 + jj];
        matVztoVxp[ii][jj] = FC.matVztoVxp[idx + ii*3 + jj];
      }
    }
    if(method == 1)
    {
      iptr = (i0+1) + (iy+nj1) * siz_line + (iz+nk1) * siz_slice; 
      iptr_f = (iy+nj1) + (iz+nk1) * ny + 1 * ny * nz;
      DxVx[4] = (Vx[iptr] - f_Vx[iptr_f]);
      DxVy[4] = (Vy[iptr] - f_Vy[iptr_f]);
      DxVz[4] = (Vz[iptr] - f_Vz[iptr_f]); 

                         dyV1[0] = DyVx[3]; dzV1[0] = DzVx[3];
                         dyV1[1] = DyVy[3]; dzV1[1] = DzVy[3];
                         dyV1[2] = DyVz[3]; dzV1[2] = DzVz[3];

      dxV2[0] = DxVx[4]; dyV2[0] = DyVx[4]; dzV2[0] = DzVx[4];
      dxV2[1] = DxVy[4]; dyV2[1] = DyVy[4]; dzV2[1] = DzVy[4];
      dxV2[2] = DxVz[4]; dyV2[2] = DyVz[4]; dzV2[2] = DzVz[4];

      fdlib_math_matmul3x1(matPlus2Min1, dxV2, out1);
      fdlib_math_matmul3x1(matPlus2Min2, dyV2, out2);
      fdlib_math_matmul3x1(matPlus2Min3, dzV2, out3);
      fdlib_math_matmul3x1(matPlus2Min4, dyV1, out4);
      fdlib_math_matmul3x1(matPlus2Min5, dzV1, out5);

      DxVx[3] = out1[0] + out2[0] + out3[0] - out4[0] - out5[0];
      DxVy[3] = out1[1] + out2[1] + out3[1] - out4[1] - out5[1];
      DxVz[3] = out1[2] + out2[2] + out3[2] - out4[2] - out5[2];
    }
    if(method == 2)
    {
      if(F.faultgrid[iy+iz*nj]){
        dtT1[0] = F.hT11[(iy+3)+(iz+3)*ny];   // inner use 1st order
        dtT1[1] = F.hT12[(iy+3)+(iz+3)*ny];
        dtT1[2] = F.hT13[(iy+3)+(iz+3)*ny];
      }else{          
        dtT1[0] = 0.0;  // boundary use 0 order
        dtT1[1] = 0.0;
        dtT1[2] = 0.0;
      }

      // Plus side --------------------------------------------
      dyV2[0] = DyVx[4];
      dyV2[1] = DyVy[4];
      dyV2[2] = DyVz[4];
      dzV2[0] = DzVx[4];
      dzV2[1] = DzVy[4];
      dzV2[2] = DzVz[4];

      fdlib_math_matmul3x1(matT1toVxp, dtT1, out1);
      fdlib_math_matmul3x1(matVytoVxp, dyV2, out2);
      fdlib_math_matmul3x1(matVztoVxp, dzV2, out3);

      DxVx[4] = out1[0] - out2[0] - out3[0];
      DxVy[4] = out1[1] - out2[1] - out3[1];
      DxVz[4] = out1[2] - out2[2] - out3[2];


      // Minus side --------------------------------------------
      dyV1[0] = DyVx[3];
      dyV1[1] = DyVy[3];
      dyV1[2] = DyVz[3];
      dzV1[0] = DzVx[3];
      dzV1[1] = DzVy[3];
      dzV1[2] = DzVz[3];

      fdlib_math_matmul3x1(matT1toVxm, dtT1, out1);
      fdlib_math_matmul3x1(matVytoVxm, dyV1, out2);
      fdlib_math_matmul3x1(matVztoVxm, dzV1, out3);

      DxVx[3] = out1[0] - out2[0] - out3[0];
      DxVy[3] = out1[1] - out2[1] - out3[1];
      DxVz[3] = out1[2] - out2[2] - out3[2];
    }
    // recalculate free surface point
    if(isfree == 1 && km==0) 
    {
      for (int ii = 0; ii < 3; ii++){
        for (int jj = 0; jj < 3; jj++){
          matVx2Vz1    [ii][jj] = FC.matVx2Vz1    [(iy+3)*9 + ii*3 + jj];
          matVx2Vz2    [ii][jj] = FC.matVx2Vz2    [(iy+3)*9 + ii*3 + jj];
          matVy2Vz1    [ii][jj] = FC.matVy2Vz1    [(iy+3)*9 + ii*3 + jj];
          matVy2Vz2    [ii][jj] = FC.matVy2Vz2    [(iy+3)*9 + ii*3 + jj];
          matPlus2Min1_f[ii][jj] = FC.matPlus2Min1f[(iy+3)*9 + ii*3 + jj];
          matPlus2Min2_f[ii][jj] = FC.matPlus2Min2f[(iy+3)*9 + ii*3 + jj];
          matPlus2Min3_f[ii][jj] = FC.matPlus2Min3f[(iy+3)*9 + ii*3 + jj];
          matT1toVxfm[ii][jj] = FC.matT1toVxfm[(iy+3)*9 + ii*3 + jj];
          matVytoVxfm[ii][jj] = FC.matVytoVxfm[(iy+3)*9 + ii*3 + jj];
          matT1toVxfp[ii][jj] = FC.matT1toVxfp[(iy+3)*9 + ii*3 + jj];
          matVytoVxfp[ii][jj] = FC.matVytoVxfp[(iy+3)*9 + ii*3 + jj];
        }
      }

      if(method == 1) 
      {

                           dyV1[0] = DyVx[3];
                           dyV1[1] = DyVy[3];
                           dyV1[2] = DyVz[3];

        dxV2[0] = DxVx[4]; dyV2[0] = DyVx[4];
        dxV2[1] = DxVy[4]; dyV2[1] = DyVy[4];
        dxV2[2] = DxVz[4]; dyV2[2] = DyVz[4];

        fdlib_math_matmul3x1(matPlus2Min1f, dxV2, out1);
        fdlib_math_matmul3x1(matPlus2Min2f, dyV2, out2);
        fdlib_math_matmul3x1(matPlus2Min3f, dyV1, out3);

        DxVx[3] = out1[0] + out2[0] - out3[0];
        DxVy[3] = out1[1] + out2[1] - out3[1];
        DxVz[3] = out1[2] + out2[2] - out3[2];
      }
      if(method == 2)
      {
        if(F.faultgrid[iy+iz*nj]){
          dtT1[0] = F.hT11[j1+k1*ny];
          dtT1[1] = F.hT12[j1+k1*ny];
          dtT1[2] = F.hT13[j1+k1*ny];
        }else{
          dtT1[0] = 0.0;
          dtT1[1] = 0.0;
          dtT1[2] = 0.0;
        }

        // plus
        dyV2[0] = DyVx[4];
        dyV2[1] = DyVy[4];
        dyV2[2] = DyVz[4];

        fdlib_math_matmul3x1(matT1toVxfp, dtT1, out1);
        fdlib_math_matmul3x1(matVytoVxfp, dyV2, out2);

        DxVx[4] = out1[0] - out2[0];
        DxVy[4] = out1[1] - out2[1];
        DxVz[4] = out1[2] - out2[2];
        // minus
        dyV1[0] = DyVx[3];
        dyV1[1] = DyVy[3];
        dyV1[2] = DyVz[3];

        fdlib_math_matmul3x1(matT1toVxfm, dtT1, out1);
        fdlib_math_matmul3x1(matVytoVxfm, dyV1, out2);

        DxVx[3] = out1[0] - out2[0];
        DxVy[3] = out1[1] - out2[1];
        DxVz[3] = out1[2] - out2[2];
      }
      // plus
      dxV2[0] = DxVx[4];
      dxV2[1] = DxVy[4];
      dxV2[2] = DxVz[4];

      fdlib_math_matmul3x1(matVx2Vz2, dxV2, out1);
      fdlib_math_matmul3x1(matVy2Vz2, dyV2, out2);

      DzVx[4] = out1[0] + out2[0];
      DzVy[4] = out1[1] + out2[1];
      DzVz[4] = out1[2] + out2[2];
      // minus
      dxV1[0] = DxVx[3];
      dxV1[1] = DxVy[3];
      dxV1[2] = DxVz[3];

      fdlib_math_matmul3x1(matVx2Vz1, dxV1, out1);
      fdlib_math_matmul3x1(matVy2Vz1, dyV1, out2);

      DzVx[3] = out1[0] + out2[0];
      DzVy[3] = out1[1] + out2[1];
      DzVz[3] = out1[2] + out2[2];
    } // end of isfree and km == 0

    if(isfree == 1 && (km==1 || km==2) )
    {
      if(km==1){
        iptr_f = (iy+3) + (iz+3) * ny + 1 * siz_slice_yz;
        f_Vx_ptr = f_Vx + iptr_f;
        f_Vy_ptr = f_Vy + iptr_f;
        f_Vz_ptr = f_Vz + iptr_f;
        MAC22_F(DzVx[4], f_Vx_ptr, siz_slice);
        MAC22_F(DzVy[4], f_Vy_ptr, siz_slice);
        MAC22_F(DzVz[4], f_Vz_ptr, siz_slice);
        iptr_f = (iy+3) + (iz+3) * ny + 0 * siz_slice_yz;
        f_Vx_ptr = f_Vx + iptr_f;
        f_Vy_ptr = f_Vy + iptr_f;
        f_Vz_ptr = f_Vz + iptr_f;
        MAC22_F(DzVx[3], f_Vx_ptr, siz_slice);
        MAC22_F(DzVy[3], f_Vy_ptr, siz_slice);
        MAC22_F(DzVz[3], f_Vz_ptr, siz_slice);
      }
      if(km==2){
        iptr_f = (iy+3) + (iz+3) * ny + 1 * siz_slice_yz;
        f_Vx_ptr = f_Vx + iptr_f;
        f_Vy_ptr = f_Vy + iptr_f;
        f_Vz_ptr = f_Vz + iptr_f;
        MAC24_F(DzVx[4], f_Vx_ptr, siz_slice);
        MAC24_F(DzVy[4], f_Vy_ptr, siz_slice);
        MAC24_F(DzVz[4], f_Vz_ptr, siz_slice);
        iptr_f = (iy+3) + (iz+3) * ny + 0 * siz_slice_yz;
        f_Vx_ptr = f_Vx + iptr_f;
        f_Vy_ptr = f_Vy + iptr_f;
        f_Vz_ptr = f_Vz + iptr_f;
        MAC24_F(DzVx[3], f_Vx_ptr, siz_slice);
        MAC24_F(DzVy[3], f_Vy_ptr, siz_slice);
        MAC24_F(DzVz[3], f_Vz_ptr, siz_slice);
      }

      //idx = ((iy+3) + (iz+3) * ny)*3*3;
      for (int ii = 0; ii < 3; ii++){
        for (int jj = 0; jj < 3; jj++){
          matPlus2Min1[ii][jj] = FC.matPlus2Min1[idx + ii*3 + jj];
          matPlus2Min2[ii][jj] = FC.matPlus2Min2[idx + ii*3 + jj];
          matPlus2Min3[ii][jj] = FC.matPlus2Min3[idx + ii*3 + jj];
          matPlus2Min4[ii][jj] = FC.matPlus2Min4[idx + ii*3 + jj];
          matPlus2Min5[ii][jj] = FC.matPlus2Min5[idx + ii*3 + jj];

          matT1toVxm[ii][jj] = FC.matT1toVxm[idx + ii*3 + jj];
          matVytoVxm[ii][jj] = FC.matVytoVxm[idx + ii*3 + jj];
          matVztoVxm[ii][jj] = FC.matVztoVxm[idx + ii*3 + jj];
          matT1toVxp[ii][jj] = FC.matT1toVxp[idx + ii*3 + jj];
          matVytoVxp[ii][jj] = FC.matVytoVxp[idx + ii*3 + jj];
          matVztoVxp[ii][jj] = FC.matVztoVxp[idx + ii*3 + jj];
        }
      }
      if(method == 1)
      {
                           dyV1[0] = DyVx[3]; dzV1[0] = DzVx[3];
                           dyV1[1] = DyVy[3]; dzV1[1] = DzVy[3];
                           dyV1[2] = DyVz[3]; dzV1[2] = DzVz[3];

        dxV2[0] = DxVx[4]; dyV2[0] = DyVx[4]; dzV2[0] = DzVx[4];
        dxV2[1] = DxVy[4]; dyV2[1] = DyVy[4]; dzV2[1] = DzVy[4];
        dxV2[2] = DxVz[4]; dyV2[2] = DyVz[4]; dzV2[2] = DzVz[4];

        fdlib_math_matmul3x1(matPlus2Min1, dxV2, out1);
        fdlib_math_matmul3x1(matPlus2Min2, dyV2, out2);
        fdlib_math_matmul3x1(matPlus2Min3, dzV2, out3);
        fdlib_math_matmul3x1(matPlus2Min4, dyV1, out4);
        fdlib_math_matmul3x1(matPlus2Min5, dzV1, out5);

        DxVx[3] = out1[0] + out2[0] + out3[0] - out4[0] - out5[0];
        DxVy[3] = out1[1] + out2[1] + out3[1] - out4[1] - out5[1];
        DxVz[3] = out1[2] + out2[2] + out3[2] - out4[2] - out5[2];
      }
      if(method == 2)
      {
        if(F.faultgrid[iy+iz*nj]){
          dtT1[0] = F.hT11[(iy+3)+(iz+3)*ny];
          dtT1[1] = F.hT12[(iy+3)+(iz+3)*ny];
          dtT1[2] = F.hT13[(iy+3)+(iz+3)*ny];
        }else{
          dtT1[0] = 0.0;
          dtT1[1] = 0.0;
          dtT1[2] = 0.0;
        }

        // Plus side --------------------------------------------
        dyV2[0] = DyVx[4];
        dyV2[1] = DyVy[4];
        dyV2[2] = DyVz[4];
        dzV2[0] = DzVx[4];
        dzV2[1] = DzVy[4];
        dzV2[2] = DzVz[4];

        fdlib_math_matmul3x1(matT1toVxp, dtT1, out1);
        fdlib_math_matmul3x1(matVytoVxp, dyV2, out2);
        fdlib_math_matmul3x1(matVztoVxp, dzV2, out3);

        DxVx[4] = out1[0] - out2[0] - out3[0];
        DxVy[4] = out1[1] - out2[1] - out3[1];
        DxVz[4] = out1[2] - out2[2] - out3[2];

        // Minus side --------------------------------------------
        dyV1[0] = DyVx[3];
        dyV1[1] = DyVy[3];
        dyV1[2] = DyVz[3];
        dzV1[0] = DzVx[3];
        dzV1[1] = DzVy[3];
        dzV1[2] = DzVz[3];

        fdlib_math_matmul3x1(matT1toVxm, dtT1, out1);
        fdlib_math_matmul3x1(matVytoVxm, dyV1, out2);
        fdlib_math_matmul3x1(matVztoVxm, dzV1, out3);

        DxVx[3] = out1[0] - out2[0] - out3[0];
        DxVy[3] = out1[1] - out2[1] - out3[1];
        DxVz[3] = out1[2] - out2[2] - out3[2];
      }
    } // isfree and km==1 or km==2

    // calculate f_hT2, f_hT3 on the Plus side 
    vec1[0] = DxVx[4]; vec1[1] = DxVy[4]; vec1[2] = DxVz[4];
    vec2[0] = DyVx[4]; vec2[1] = DyVy[4]; vec2[2] = DyVz[4];
    vec3[0] = DzVx[4]; vec3[1] = DzVy[4]; vec3[2] = DzVz[4];

    //idx = ((iy+3) + (iz+3) * ny)*3*3;
    for (int ii = 0; ii < 3; ii++){
      for (int jj = 0; jj < 3; jj++){
        mat1[ii][jj] = FC.D21_2[idx + ii*3 + jj];
        mat2[ii][jj] = FC.D22_2[idx + ii*3 + jj];
        mat3[ii][jj] = FC.D23_2[idx + ii*3 + jj];
      }
    }

    fdlib_math_matmul3x1(mat1, vec1, vecg1);
    fdlib_math_matmul3x1(mat2, vec2, vecg2);
    fdlib_math_matmul3x1(mat3, vec3, vecg3);

    iptr_f = (iy+nj1) + (iz+nk1) * ny + 1 * siz_slice_yz;
    f_hT21[iptr_f] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT22[iptr_f] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT23[iptr_f] = vecg1[2] + vecg2[2] + vecg3[2];

    //idx = ((iy+3) + (iz+3) * ny)*3*3;
    for (int ii = 0; ii < 3; ii++){
      for (int jj = 0; jj < 3; jj++){
        mat1[ii][jj] = FC.D31_2[idx + ii*3 + jj];
        mat2[ii][jj] = FC.D32_2[idx + ii*3 + jj];
        mat3[ii][jj] = FC.D33_2[idx + ii*3 + jj];
      }
    }

    fdlib_math_matmul3x1(mat1, vec1, vecg1);
    fdlib_math_matmul3x1(mat2, vec2, vecg2);
    fdlib_math_matmul3x1(mat3, vec3, vecg3);

    f_hT31[iptr_f] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT32[iptr_f] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT33[iptr_f] = vecg1[2] + vecg2[2] + vecg3[2];

    // calculate f_hT2, f_hT3 on the Minus side
    vec1[0] = DxVx[3]; vec1[1] = DxVy[3]; vec1[2] = DxVz[3];
    vec2[0] = DyVx[3]; vec2[1] = DyVy[3]; vec2[2] = DyVz[3];
    vec3[0] = DzVx[3]; vec3[1] = DzVy[3]; vec3[2] = DzVz[3];

    //idx = ((iy+3) + (iz+3) * ny)*3*3;
    for (int ii = 0; ii < 3; ii++){
      for (int jj = 0; jj < 3; jj++){
        mat1[ii][jj] = FC.D21_1[idx + ii*3 + jj];
        mat2[ii][jj] = FC.D22_1[idx + ii*3 + jj];
        mat3[ii][jj] = FC.D23_1[idx + ii*3 + jj];
      }
    }

    fdlib_math_matmul3x1(mat1, vec1, vecg1);
    fdlib_math_matmul3x1(mat2, vec2, vecg2);
    fdlib_math_matmul3x1(mat3, vec3, vecg3);

    iptr_f = (iy+nj1) + (iz+nk1) * ny + 0 * siz_slice_yz;
    f_hT21[iptr_f] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT22[iptr_f] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT23[iptr_f] = vecg1[2] + vecg2[2] + vecg3[2];

    //idx = ((iy+3) + (iz+3) * ny)*3*3;
    for (int ii = 0; ii < 3; ii++){
      for (int jj = 0; jj < 3; jj++){
        mat1[ii][jj] = FC.D31_1[idx + ii*3 + jj];
        mat2[ii][jj] = FC.D32_1[idx + ii*3 + jj];
        mat3[ii][jj] = FC.D33_1[idx + ii*3 + jj];
      }
    }

    fdlib_math_matmul3x1(mat1, vec1, vecg1);
    fdlib_math_matmul3x1(mat2, vec2, vecg2);
    fdlib_math_matmul3x1(mat3, vec3, vecg3);

    f_hT31[iptr_f] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT32[iptr_f] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT33[iptr_f] = vecg1[2] + vecg2[2] + vecg3[2];

    // point E
    iptr = (i0-1) + (iy+3) *siz_line + (iz+3) * siz_slice;
    iptr_f = (iy+nj1) + (iz+nk1) * ny + 0 * siz_slice_yz;
    DxVx[2] = (f_Vx[iptr_f] - Vx[iptr]); 
    DxVy[2] = (f_Vy[iptr_f] - Vy[iptr]); 
    DxVz[2] = (f_Vz[iptr_f] - Vz[iptr]); 

    // point F
    iptr = (i0-2) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_3[0] = Vx[iptr];
    iptr = (i0-1) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_3[1] = Vx[iptr];
    iptr_t = (iy+3) + (iz+3) * ny + 0 * siz_slice_yz;       vec_3[2] = f_Vx[iptr_t];
    VEC_24_F(DxVx[1], vec_3);

    iptr = (i0-2) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_3[0] = Vy[iptr];
    iptr = (i0-1) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_3[1] = Vy[iptr];
    iptr_t = (iy+3) + (iz+3) * ny + 0 * siz_slice_yz;       vec_3[2] = f_Vy[iptr_t];
    VEC_24_F(DxVy[1], vec_3);

    iptr = (i0-2) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_3[0] = Vz[iptr];
    iptr = (i0-1) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_3[1] = Vz[iptr];
    iptr_t = (iy+3) + (iz+3) * ny + 0 * siz_slice_yz;       vec_3[2] = f_Vz[iptr_t];
    VEC_24_F(DxVz[1], vec_3);

    // point G
    iptr = (i0-4) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[0] = Vx[iptr];
    iptr = (i0-3) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[1] = Vx[iptr];
    iptr = (i0-2) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[2] = Vx[iptr];
    iptr = (i0-1) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[3] = Vx[iptr];
    iptr_f = (iy+3) + (iz+3) * ny + 0 * siz_slice_yz;       vec_5[4] = f_Vx[iptr_f];
    VEC_DRP_F(DxVx[0], vec_5);

    iptr = (i0-4) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[0] = Vy[iptr];
    iptr = (i0-3) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[1] = Vy[iptr];
    iptr = (i0-2) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[2] = Vy[iptr];
    iptr = (i0-1) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[3] = Vy[iptr];
    iptr_f = (iy+3) + (iz+3) * ny + 0 * siz_slice_yz;       vec_5[4] = f_Vy[iptr_f];
    VEC_DRP_F(DxVy[0], vec_5);

    iptr = (i0-4) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[0] = Vz[iptr];
    iptr = (i0-3) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[1] = Vz[iptr];
    iptr = (i0-2) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[2] = Vz[iptr];
    iptr = (i0-1) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[3] = Vz[iptr];
    iptr_f = (iy+3) + (iz+3) * ny + 0 * siz_slice_yz;       vec_5[4] = f_Vz[iptr_f];
    VEC_DRP_F(DxVz[0], vec_5);

    for (int n = 0; n < 8 ; n++)
    {
      // n =  0  1  2  3  4  5  6  7
      // m = -3 -2 -1 -0 +0 +1 +2 +3
      int m;
      if(n < 4){
        m = n-3;
      }else{
        m = n-4;
      }
      if(m==0) continue; // do not update i0
      i = i0+m;

      iptr = i + (iy+3) * siz_line + (iz+3) * siz_slice;
      Vx_ptr = Vx + iptr;
      Vy_ptr = Vy + iptr;
      Vz_ptr = Vz + iptr;
      M_FD_SHIFT_PTR_MACDRP(DyVx[n],Vx_ptr,siz_line,jdir);
      M_FD_SHIFT_PTR_MACDRP(DyVy[n],Vy_ptr,siz_line,jdir);
      M_FD_SHIFT_PTR_MACDRP(DyVz[n],Vz_ptr,siz_line,jdir);

      M_FD_SHIFT_PTR_MACDRP(DzVx[n],Vx_ptr,siz_slice,kdir);
      M_FD_SHIFT_PTR_MACDRP(DzVy[n],Vy_ptr,siz_slice,kdir);
      M_FD_SHIFT_PTR_MACDRP(DzVz[n],Vz_ptr,siz_slice,kdir);

      if(is_free==1 && km==0){
        DzVx[n] = matVx2Vz[idx + 3*0 + 0] * DxVx[n]
                + matVx2Vz[idx + 3*0 + 1] * DxVy[n]
                + matVx2Vz[idx + 3*0 + 2] * DxVz[n]
                + matVy2Vz[idx + 3*0 + 0] * DyVx[n]
                + matVy2Vz[idx + 3*0 + 1] * DyVy[n]
                + matVy2Vz[idx + 3*0 + 2] * DyVz[n];

        DzVy[n] = matVx2Vz[idx + 3*1 + 0] * DxVx[n]
                + matVx2Vz[idx + 3*1 + 1] * DxVy[n]
                + matVx2Vz[idx + 3*1 + 2] * DxVz[n]
                + matVy2Vz[idx + 3*1 + 0] * DyVx[n]
                + matVy2Vz[idx + 3*1 + 1] * DyVy[n]
                + matVy2Vz[idx + 3*1 + 2] * DyVz[n];

        DzVz[n] = matVx2Vz[idx + 3*2 + 0] * DxVx[n]
                + matVx2Vz[idx + 3*2 + 1] * DxVy[n]
                + matVx2Vz[idx + 3*2 + 2] * DxVz[n]
                + matVy2Vz[idx + 3*2 + 0] * DyVx[n]
                + matVy2Vz[idx + 3*2 + 1] * DyVy[n]
                + matVy2Vz[idx + 3*2 + 2] * DyVz[n] ;
      } 
      if(is_free==1 && km==1){
        M_FD_SHIFT_PTR_MAC22(DzVx[n],Vx_ptr,siz_slice,kdir);
        M_FD_SHIFT_PTR_MAC22(DzVy[n],Vy_ptr,siz_slice,kdir);
        M_FD_SHIFT_PTR_MAC22(DzVz[n],Vz_ptr,siz_slice,kdir);
      }
      if(is_free==1 && km==2){
        M_FD_SHIFT_PTR_MAC24(DzVx[n],Vx_ptr,siz_slice,kdir);
        M_FD_SHIFT_PTR_MAC24(DzVy[n],Vy_ptr,siz_slice,kdir);
        M_FD_SHIFT_PTR_MAC24(DzVz[n],Vz_ptr,siz_slice,kdir);
      }

      // iptr = i + (iy+3) * siz_line + (iz+3) * siz_slice;
      lam = lam3d[iptr]; mu = mu3d[iptr];
      lam2mu  = lam + 2.0*mu;
      xix = xi_x[iptr]; xiy = xi_y[iptr]; xiz = xi_z[iptr];
      etx = et_x[iptr]; ety = et_y[iptr]; etz = et_z[iptr];
      ztx = zt_x[iptr]; zty = zt_y[iptr]; ztz = zt_z[iptr];

      hTxx[iptr] =   lam2mu * ( xix*DxVx[n] + etx*DyVx[n] + ztx*DzVx[n])
                   + lam    * ( xiy*DxVy[n] + ety*DyVy[n] + zty*DzVy[n]
                               +xiz*DxVz[n] + etz*DyVz[n] + ztz*DzVz[n]);

      hTyy[iptr] =    lam2mu * ( xiy*DxVy[n] + ety*DyVy[n] + zty*DzVy[n])
                    + lam    * ( xix*DxVx[n] + etx*DyVx[n] + ztx*DzVx[n]
                                +xiz*DxVz[n] + etz*DyVz[n] + ztz*DzVz[n]);

      hTzz[iptr] =    lam2mu * ( xiz*DxVz[n] + etz*DyVz[n] + ztz*DzVz[n])
                    + lam    * ( xix*DxVx[n] + etx*DyVx[n] + ztx*DzVx[n]
                                +xiy*DxVy[n] + ety*DyVy[n] + zty*DzVy[n]);

      hTxy[iptr] =  mu *(
                    xiy*DxVx[n] + xix*DxVy[n] +
                    ety*DyVx[n] + etx*DyVy[n] +
                    zty*DzVx[n] + ztx*DzVy[n] );

      hTxz[iptr] =  mu *(
                    xiz*DxVx[n] + xix*DxVz[n] +
                    etz*DyVx[n] + etx*DyVz[n] +
                    ztz*DzVx[n] + ztx*DzVz[n] ); 

      hTyz[iptr] =  mu *(
                    xiz*DxVy[n] + xiy*DxVz[n] +
                    etz*DyVy[n] + ety*DyVz[n] +
                    ztz*DzVy[n] + zty*DzVz[n] );
    } 
  } 
  return;
}

__global__
void sv_eq1st_curv_col_el_iso_rhs_fault_stress_B_gpu(
                       float * Vx, float * Vy, float * Vz,
                       float * hTxx, float * hTyy, float * hTzz,
                       float * hTxz, float * hTyz, float * hTxy,
                       float * f_Vx, float * f_Vy, float * f_Vz,
                       float * f_hT21, float * f_hT22, float * f_hT23,
                       float * f_hT31, float * f_hT32, float * f_hT33,
                       float * f_T11, float * f_T12, float * f_T13,
                       float * xi_x,  float * xi_y, float * xi_z,
                       float * et_x,  float * et_y, float * et_z,
                       float * zt_x,  float * zt_y, float * zt_z,
                       float * lam3d, float * mu3d, float * slw3d, 
                       fault_t f_d, float *matVx2Vz, float *matVy2Vz,
                       int isfree, int i0,
                       int nj1, int nj, int nk1, int nk, int ny, 
                       size_t siz_line, size_t siz_slice, size_t siz_slice_yz,
                       int idir, int jdir, int kdir)
{
  int iy = blockIdx.x * blockDim.x + threadIdx.x;
  int iz = blockIdx.y * blockDim.y + threadIdx.y;

  float *Vx_ptr;
  float *Vy_ptr;
  float *Vz_ptr;
  float *f_Vx_ptr;
  float *f_Vy_ptr;
  float *f_Vz_ptr;
  float xix, xiy, xiz;
  float etx, ety, etz;
  float ztx, zty, ztz;
  float mu, lam, lam2mu;

  float DxVx[8],DxVy[8],DxVz[8];
  float DyVx[8],DyVy[8],DyVz[8];
  float DzVx[8],DzVy[8],DzVz[8];
  float mat1[3][3], mat2[3][3], mat3[3][3];
  float vec_3[3], vec_5[5];
  float vec1[3], vec2[3], vec3[3];
  float vecg1[3], vecg2[3], vecg3[3];
  float dtT1[3];
  float dxV1[3], dyV1[3], dzV1[3];
  float dxV2[3], dyV2[3], dzV2[3];
  float out1[3], out2[3], out3[3], out4[3], out5[3];

  float matT1toVxm[3][3];
  float matVytoVxm[3][3];
  float matVztoVxm[3][3];
  float matT1toVxp[3][3];
  float matVytoVxp[3][3];
  float matVztoVxp[3][3];

  float matT1toVxfm[3][3];
  float matVytoVxfm[3][3];
  float matT1toVxfp[3][3];
  float matVytoVxfp[3][3];

  float matMin2Plus1[3][3];
  float matMin2Plus2[3][3];
  float matMin2Plus3[3][3];
  float matMin2Plus4[3][3];
  float matMin2Plus5[3][3];

  float matMin2Plus1f[3][3], matMin2Plus2f[3][3], matMin2Plus3f[3][3];
  float matVx2Vz1[3][3], matVy2Vz1[3][3];
  float matVx2Vz2[3][3], matVy2Vz2[3][3];

  int idx = ((iy+3) + (iz+3) * ny)*3*3;
  if (iy < nj && iz < nk ) { 
    // non united
    if(F.united[iy + iz * nj]) return;
    int km = nk - (iz+1); 
    int n_free = km + 3;
    //
    //     Update Stress (in Zhang's Thesis, 2014)
    // ---V-----V-----V-----0-----0-----V-----V-----V---  (grid point)
    //    G     F     E     D-    D+    C     B     A     (grid name in thesis)
    //    i0-3  i0-2  i0-1  i0-0  i0+0  i0+1  i0+2  i0+3  (3D grid index)
    //    0     1     2     3     4     5     6     7     (vec grid index)
    //    -3    -2    -1    -0    +0    1     2     3     (offset from fault)

    // point G
    iptr = (i0-3) + (iy+nj1) * siz_line + (iz+nk1) *siz_slice;
    Vx_ptr = Vx + iptr;
    Vy_ptr = Vy + iptr;
    Vz_ptr = Vz + iptr;
    MACDRP_B(DxVx[0],Vx_ptr,1);
    MACDRP_B(DxVy[0],Vy_ptr,1);
    MACDRP_B(DxVz[0],Vz_ptr,1);

    // point F
    iptr = (i0-2) + (iy+nj1) * siz_line + (iz+nk1) *siz_slice;
    Vx_ptr = Vx + iptr;
    Vy_ptr = Vy + iptr;
    Vz_ptr = Vz + iptr;
    MAC24_B(DxVx[1],Vx_ptr,1);
    MAC24_B(DxVy[1],Vy_ptr,1);
    MAC24_B(DxVz[1],Vz_ptr,1);

    // point E
    iptr = (i0-1) + (iy+nj1) * siz_line + (iz+nk1) *siz_slice;
    Vx_ptr = Vx + iptr;
    Vy_ptr = Vy + iptr;
    Vz_ptr = Vz + iptr;
    MAC22_B(DxVx[2],Vx_ptr,1);
    MAC22_B(DxVy[2],Vy_ptr,1);
    MAC22_B(DxVz[2],Vz_ptr,1);

    //fault split point D- D+
    if(F.rup_index_y[iy + iz * nj] == 1)
    {
      iptr_f = (iy+nj1) + (iz+nk1) * ny + 0 * siz_slice_yz;
      f_Vx_ptr = f_Vx + iptr_f;
      f_Vy_ptr = f_Vy + iptr_f;
      f_Vz_ptr = f_Vz + iptr_f;
      MAC22_B(DyVx[3],f_Vx_ptr,siz_line);
      MAC22_B(DyVy[3],f_Vy_ptr,siz_line);
      MAC22_B(DyVz[3],f_Vz_ptr,siz_line);
      iptr_f = (iy+nj1) + (iz+nk1) * ny + 1 * siz_slice_yz;
      f_Vx_ptr = f_Vx + iptr_f;
      f_Vy_ptr = f_Vy + iptr_f;
      f_Vz_ptr = f_Vz + iptr_f;
      MAC22_B(DyVx[4],f_Vx_ptr,siz_line);
      MAC22_B(DyVy[4],f_Vy_ptr,siz_line);
      MAC22_B(DyVz[4],f_Vz_ptr,siz_line);
    }
    if(F.rup_index_y[iy + iz * nj] == 0)
    {
      iptr_f = (iy+nj1) + (iz+nk1) * ny + 0 * siz_slice_yz;
      f_Vx_ptr = f_Vx + iptr_f;
      f_Vy_ptr = f_Vy + iptr_f;
      f_Vz_ptr = f_Vz + iptr_f;
      MACDRP_B(DyVx[3],f_Vx_ptr,siz_line);
      MACDRP_B(DyVy[3],f_Vy_ptr,siz_line);
      MACDRP_B(DyVz[3],f_Vz_ptr,siz_line);
      iptr_f = (iy+nj1) + (iz+nk1) * ny + 1 * siz_slice_yz;
      f_Vx_ptr = f_Vx + iptr_f;
      f_Vy_ptr = f_Vy + iptr_f;
      f_Vz_ptr = f_Vz + iptr_f;
      MACDRP_B(DyVx[4],f_Vx_ptr,siz_line);
      MACDRP_B(DyVy[4],f_Vy_ptr,siz_line);
      MACDRP_B(DyVz[4],f_Vz_ptr,siz_line);
    }
    if(F.rup_index_z[iy + iz * nj] == 1){
      iptr_f = (iy+3) + (iz+3) * ny + 1 * siz_slice_yz;
      f_Vx_ptr = f_Vx + iptr_f;
      f_Vy_ptr = f_Vy + iptr_f;
      f_Vz_ptr = f_Vz + iptr_f;
      MAC22_B(DzVx[4],f_Vx_ptr,siz_slice);
      MAC22_B(DzVy[4],f_Vy_ptr,siz_slice);
      MAC22_B(DzVz[4],f_Vz_ptr,siz_slice);
      iptr_f = (iy+3) + (iz+3) * ny + 0 * siz_slice_yz;
      f_Vx_ptr = f_Vx + iptr_f;
      f_Vy_ptr = f_Vy + iptr_f;
      f_Vz_ptr = f_Vz + iptr_f;
      MAC22_B(DzVx[3],f_Vx_ptr,siz_slice);
      MAC22_B(DzVy[3],f_Vy_ptr,siz_slice);
      MAC22_B(DzVz[3],f_Vz_ptr,siz_slice);
    }
    if(F.rup_index_z[iy + iz * nj] == 0){
      iptr_f = (iy+3) + (iz+3) * ny + 0 * siz_slice_yz;
      f_Vx_ptr = f_Vx + iptr_f;
      f_Vy_ptr = f_Vy + iptr_f;
      f_Vz_ptr = f_Vz + iptr_f;
      MACDRP_B(DzVx[3],f_Vx_ptr,siz_slice);
      MACDRP_B(DzVy[3],f_Vy_ptr,siz_slice);
      MACDRP_B(DzVz[3],f_Vz_ptr,siz_slice);
      iptr_f = (iy+3) + (iz+3) * ny + 1 * siz_slice_yz;
      f_Vx_ptr = f_Vx + iptr_f;
      f_Vy_ptr = f_Vy + iptr_f;
      f_Vz_ptr = f_Vz + iptr_f;
      MACDRP_B(DzVx[4],f_Vx_ptr,siz_slice);
      MACDRP_B(DzVy[4],f_Vy_ptr,siz_slice);
      MACDRP_B(DzVz[4],f_Vz_ptr,siz_slice);
    }

    for (int ii = 0; ii < 3; ii++){
      for (int jj = 0; jj < 3; jj++){
        matMin2Plus1[ii][jj] = FC.matMin2Plus1[idx + ii*3 + jj];
        matMin2Plus2[ii][jj] = FC.matMin2Plus2[idx + ii*3 + jj];
        matMin2Plus3[ii][jj] = FC.matMin2Plus3[idx + ii*3 + jj];
        matMin2Plus4[ii][jj] = FC.matMin2Plus4[idx + ii*3 + jj];
        matMin2Plus5[ii][jj] = FC.matMin2Plus5[idx + ii*3 + jj];

        matT1toVxm[ii][jj] = FC.matT1toVxm[idx + ii*3 + jj];
        matVytoVxm[ii][jj] = FC.matVytoVxm[idx + ii*3 + jj];
        matVztoVxm[ii][jj] = FC.matVztoVxm[idx + ii*3 + jj];
        matT1toVxp[ii][jj] = FC.matT1toVxp[idx + ii*3 + jj];
        matVytoVxp[ii][jj] = FC.matVytoVxp[idx + ii*3 + jj];
        matVztoVxp[ii][jj] = FC.matVztoVxp[idx + ii*3 + jj];
      }
    }
    if(method == 1)
    {
      iptr = (i0-1) + (iy+nj1) * siz_line + (iz+nk1) * siz_slice; 
      iptr_f = (iy+nj1) + (iz+nk1) * ny + 0 * siz_slice_yz;
      DxVx[3] = (f_Vx[iptr_f] - Vx[iptr]);
      DxVy[3] = (f_Vy[iptr_f] - Vy[iptr]);
      DxVz[3] = (f_Vz[iptr_f] - Vz[iptr]); 

      dxV1[0] = DxVx[3]; dyV1[0] = DyVx[3]; dzV1[0] = DzVx[3];
      dxV1[1] = DxVy[3]; dyV1[1] = DyVy[3]; dzV1[1] = DzVy[3];
      dxV1[2] = DxVz[3]; dyV1[2] = DyVz[3]; dzV1[2] = DzVz[3];

                         dyV2[0] = DyVx[4]; dzV2[0] = DzVx[4];
                         dyV2[1] = DyVy[4]; dzV2[1] = DzVy[4];
                         dyV2[2] = DyVz[4]; dzV2[2] = DzVz[4];

      fdlib_math_matmul3x1(matMin2Plus1, dxV1, out1);
      fdlib_math_matmul3x1(matMin2Plus2, dyV1, out2);
      fdlib_math_matmul3x1(matMin2Plus3, dzV1, out3);
      fdlib_math_matmul3x1(matMin2Plus4, dyV2, out4);
      fdlib_math_matmul3x1(matMin2Plus5, dzV2, out5);

      DxVx[4] = out1[0] + out2[0] + out3[0] - out4[0] - out5[0];
      DxVy[4] = out1[1] + out2[1] + out3[1] - out4[1] - out5[1];
      DxVz[4] = out1[2] + out2[2] + out3[2] - out4[2] - out5[2];
    }
    if(method == 2)
    {
      if(F.faultgrid[iy+iz*nj]){
        dtT1[0] = F.hT11[(iy+3)+(iz+3)*ny];
        dtT1[1] = F.hT12[(iy+3)+(iz+3)*ny];
        dtT1[2] = F.hT13[(iy+3)+(iz+3)*ny];
      }else{
        dtT1[0] = 0.0;
        dtT1[1] = 0.0;
        dtT1[2] = 0.0;
      }

      // Minus side --------------------------------------------
      dyV1[0] = DyVx[3];
      dyV1[1] = DyVy[3];
      dyV1[2] = DyVz[3];
      dzV1[0] = DzVx[3];
      dzV1[1] = DzVy[3];
      dzV1[2] = DzVz[3];

      fdlib_math_matmul3x1(matT1toVxm, dtT1, out1);
      fdlib_math_matmul3x1(matVytoVxm, dyV1, out2);
      fdlib_math_matmul3x1(matVztoVxm, dzV1, out3);

      DxVx[3] = out1[0] - out2[0] - out3[0];
      DxVy[3] = out1[1] - out2[1] - out3[1];
      DxVz[3] = out1[2] - out2[2] - out3[2];

      // Plus side --------------------------------------------
      dyV2[0] = DyVx[4];
      dyV2[1] = DyVy[4];
      dyV2[2] = DyVz[4];
      dzV2[0] = DzVx[4];
      dzV2[1] = DzVy[4];
      dzV2[2] = DzVz[4];

      fdlib_math_matmul3x1(matT1toVxp, dtT1, out1);
      fdlib_math_matmul3x1(matVytoVxp, dyV2, out2);
      fdlib_math_matmul3x1(matVztoVxp, dzV2, out3);

      DxVx[4] = out1[0] - out2[0] - out3[0];
      DxVy[4] = out1[1] - out2[1] - out3[1];
      DxVz[4] = out1[2] - out2[2] - out3[2];
    }
    // recalculate free surface point
    if(isfree == 1 && km==0) 
    {
      for (int ii = 0; ii < 3; ii++){
        for (int jj = 0; jj < 3; jj++){
          matVx2Vz1    [ii][jj] = FC.matVx2Vz1    [(iy+3)*9 + ii*3 + jj];
          matVx2Vz2    [ii][jj] = FC.matVx2Vz2    [(iy+3)*9 + ii*3 + jj];
          matVy2Vz1    [ii][jj] = FC.matVy2Vz1    [(iy+3)*9 + ii*3 + jj];
          matVy2Vz2    [ii][jj] = FC.matVy2Vz2    [(iy+3)*9 + ii*3 + jj];
          matMin2Plus1_f[ii][jj] = FC.matMin2Plus1f[(iy+3)*9 + ii*3 + jj];
          matMin2Plus2_f[ii][jj] = FC.matMin2Plus2f[(iy+3)*9 + ii*3 + jj];
          matMin2Plus3_f[ii][jj] = FC.matMin2Plus3f[(iy+3)*9 + ii*3 + jj];
          matT1toVxfm[ii][jj] = FC.matT1toVxfm[(iy+3)*9 + ii*3 + jj];
          matVytoVxfm[ii][jj] = FC.matVytoVxfm[(iy+3)*9 + ii*3 + jj];
          matT1toVxfp[ii][jj] = FC.matT1toVxfp[(iy+3)*9 + ii*3 + jj];
          matVytoVxfp[ii][jj] = FC.matVytoVxfp[(iy+3)*9 + ii*3 + jj];
        }
      }

      if(method == 1) 
      {

        dxV1[0] = DxVx[3]; dyV1[0] = DyVx[3];
        dxV1[1] = DxVy[3]; dyV1[1] = DyVy[3];
        dxV1[2] = DxVz[3]; dyV1[2] = DyVz[3];

                           dyV2[0] = DyVx[4];
                           dyV2[1] = DyVy[4];
                           dyV2[2] = DyVz[4];

        fdlib_math_matmul3x1(matMin2Plusf, dxV1, out1);
        fdlib_math_matmul3x1(matMin2Plusf, dyV1, out2);
        fdlib_math_matmul3x1(matMin2Plusf, dyV2, out3);

        DxVx[4] = out1[0] + out2[0] - out3[0];
        DxVy[4] = out1[1] + out2[1] - out3[1];
        DxVz[4] = out1[2] + out2[2] - out3[2];
      }
      if(method == 2)
      {
        if(F.faultgrid[iy+iz*nj]){
          dtT1[0] = F.hT11[j1+k1*ny];
          dtT1[1] = F.hT12[j1+k1*ny];
          dtT1[2] = F.hT13[j1+k1*ny];
        }else{
          dtT1[0] = 0.0;
          dtT1[1] = 0.0;
          dtT1[2] = 0.0;
        }
        // minus
        dyV1[0] = DyVx[3];
        dyV1[1] = DyVy[3];
        dyV1[2] = DyVz[3];

        fdlib_math_matmul3x1(matT1toVxfm, dtT1, out1);
        fdlib_math_matmul3x1(matVytoVxfm, dyV1, out2);

        DxVx[3] = out1[0] - out2[0];
        DxVy[3] = out1[1] - out2[1];
        DxVz[3] = out1[2] - out2[2];
        // plus
        dyV2[0] = DyVx[4];
        dyV2[1] = DyVy[4];
        dyV2[2] = DyVz[4];

        fdlib_math_matmul3x1(matT1toVxfp, dtT1, out1);
        fdlib_math_matmul3x1(matVytoVxfp, dyV2, out2);

        DxVx[4] = out1[0] - out2[0];
        DxVy[4] = out1[1] - out2[1];
        DxVz[4] = out1[2] - out2[2];
      }
      // minus
      dxV1[0] = DxVx[3];
      dxV1[1] = DxVy[3];
      dxV1[2] = DxVz[3];

      fdlib_math_matmul3x1(matVx2Vz1, dxV1, out1);
      fdlib_math_matmul3x1(matVy2Vz1, dyV1, out2);

      DzVx[3] = out1[0] + out2[0];
      DzVy[3] = out1[1] + out2[1];
      DzVz[3] = out1[2] + out2[2];
      // plus
      dxV2[0] = DxVx[4];
      dxV2[1] = DxVy[4];
      dxV2[2] = DxVz[4];

      fdlib_math_matmul3x1(matVx2Vz2, dxV2, out1);
      fdlib_math_matmul3x1(matVy2Vz2, dyV2, out2);

      DzVx[4] = out1[0] + out2[0];
      DzVy[4] = out1[1] + out2[1];
      DzVz[4] = out1[2] + out2[2];
    } // end of isfree and km == 0

    if(isfree == 1 && (km==1 || km==2) )
    {
      if(km==1){
        iptr_f = (iy+3) + (iz+3) * ny + 0 * siz_slice_yz;
        f_Vx_ptr = f_Vx + iptr_f;
        f_Vy_ptr = f_Vy + iptr_f;
        f_Vz_ptr = f_Vz + iptr_f;
        MAC22_B(DzVx[3], f_Vx_ptr, siz_slice);
        MAC22_B(DzVy[3], f_Vy_ptr, siz_slice);
        MAC22_B(DzVz[3], f_Vz_ptr, siz_slice);
        iptr_f = (iy+3) + (iz+3) * ny + 1 * siz_slice_yz;
        f_Vx_ptr = f_Vx + iptr_f;
        f_Vy_ptr = f_Vy + iptr_f;
        f_Vz_ptr = f_Vz + iptr_f;
        MAC22_B(DzVx[4], f_Vx_ptr, siz_slice);
        MAC22_B(DzVy[4], f_Vy_ptr, siz_slice);
        MAC22_B(DzVz[4], f_Vz_ptr, siz_slice);
      }
      if(km==2){
        iptr_f = (iy+3) + (iz+3) * ny + 0 * siz_slice_yz;
        f_Vx_ptr = f_Vx + iptr_f;
        f_Vy_ptr = f_Vy + iptr_f;
        f_Vz_ptr = f_Vz + iptr_f;
        MAC24_B(DzVx[3], f_Vx_ptr, siz_slice);
        MAC24_B(DzVy[3], f_Vy_ptr, siz_slice);
        MAC24_B(DzVz[3], f_Vz_ptr, siz_slice);
        iptr_f = (iy+3) + (iz+3) * ny + 1 * siz_slice_yz;
        f_Vx_ptr = f_Vx + iptr_f;
        f_Vy_ptr = f_Vy + iptr_f;
        f_Vz_ptr = f_Vz + iptr_f;
        MAC24_B(DzVx[4], f_Vx_ptr, siz_slice);
        MAC24_B(DzVy[4], f_Vy_ptr, siz_slice);
        MAC24_B(DzVz[4], f_Vz_ptr, siz_slice);
      }

      //idx = ((iy+3) + (iz+3) * ny)*3*3;
      for (int ii = 0; ii < 3; ii++){
        for (int jj = 0; jj < 3; jj++){
          matMin2Plus1[ii][jj] = FC.matMin2Plus1[idx + ii*3 + jj];
          matMin2Plus2[ii][jj] = FC.matMin2Plus2[idx + ii*3 + jj];
          matMin2Plus3[ii][jj] = FC.matMin2Plus3[idx + ii*3 + jj];
          matMin2Plus4[ii][jj] = FC.matMin2Plus4[idx + ii*3 + jj];
          matMin2Plus5[ii][jj] = FC.matMin2Plus5[idx + ii*3 + jj];

          matT1toVxm[ii][jj] = FC.matT1toVxm[idx + ii*3 + jj];
          matVytoVxm[ii][jj] = FC.matVytoVxm[idx + ii*3 + jj];
          matVztoVxm[ii][jj] = FC.matVztoVxm[idx + ii*3 + jj];
          matT1toVxp[ii][jj] = FC.matT1toVxp[idx + ii*3 + jj];
          matVytoVxp[ii][jj] = FC.matVytoVxp[idx + ii*3 + jj];
          matVztoVxp[ii][jj] = FC.matVztoVxp[idx + ii*3 + jj];
        }
      }
      if(method == 1)
      {
         dxV1[0] = DxVx[3]; dyV1[0] = DyVx[3]; dzV1[0] = DzVx[3];
         dxV1[1] = DxVy[3]; dyV1[1] = DyVy[3]; dzV1[1] = DzVy[3];
         dxV1[2] = DxVz[3]; dyV1[2] = DyVz[3]; dzV1[2] = DzVz[3];

                            dyV2[0] = DyVx[4]; dzV2[0] = DzVx[4];
                            dyV2[1] = DyVy[4]; dzV2[1] = DzVy[4];
                            dyV2[2] = DyVz[4]; dzV2[2] = DzVz[4];

        fdlib_math_matmul3x1(matMin2Plus1, dxV1, out1);
        fdlib_math_matmul3x1(matMin2Plus2, dyV1, out2);
        fdlib_math_matmul3x1(matMin2Plus3, dzV1, out3);
        fdlib_math_matmul3x1(matMin2Plus4, dyV2, out4);
        fdlib_math_matmul3x1(matMin2Plus5, dzV2, out5);

        DxVx[4] = out1[0] + out2[0] + out3[0] - out4[0] - out5[0];
        DxVy[4] = out1[1] + out2[1] + out3[1] - out4[1] - out5[1];
        DxVz[4] = out1[2] + out2[2] + out3[2] - out4[2] - out5[2];
      }
      if(method == 2)
      {
        if(F.faultgrid[iy+iz*nj]){
          dtT1[0] = F.hT11[(iy+3)+(iz+3)*ny];
          dtT1[1] = F.hT12[(iy+3)+(iz+3)*ny];
          dtT1[2] = F.hT13[(iy+3)+(iz+3)*ny];
        }else{
          dtT1[0] = 0.0;
          dtT1[1] = 0.0;
          dtT1[2] = 0.0;
        }

        // Minus side --------------------------------------------
        dyV1[0] = DyVx[3];
        dyV1[1] = DyVy[3];
        dyV1[2] = DyVz[3];
        dzV1[0] = DzVx[3];
        dzV1[1] = DzVy[3];
        dzV1[2] = DzVz[3];

        fdlib_math_matmul3x1(matT1toVxm, dtT1, out1);
        fdlib_math_matmul3x1(matVytoVxm, dyV1, out2);
        fdlib_math_matmul3x1(matVztoVxm, dzV1, out3);

        DxVx[3] = out1[0] - out2[0] - out3[0];
        DxVy[3] = out1[1] - out2[1] - out3[1];
        DxVz[3] = out1[2] - out2[2] - out3[2];

        // Plus side --------------------------------------------
        dyV2[0] = DyVx[4];
        dyV2[1] = DyVy[4];
        dyV2[2] = DyVz[4];
        dzV2[0] = DzVx[4];
        dzV2[1] = DzVy[4];
        dzV2[2] = DzVz[4];

        fdlib_math_matmul3x1(matT1toVxp, dtT1, out1);
        fdlib_math_matmul3x1(matVytoVxp, dyV2, out2);
        fdlib_math_matmul3x1(matVztoVxp, dzV2, out3);

        DxVx[4] = out1[0] - out2[0] - out3[0];
        DxVy[4] = out1[1] - out2[1] - out3[1];
        DxVz[4] = out1[2] - out2[2] - out3[2];

      }
    } // isfree and km==1 or km==2

    // calculate f_hT2, f_hT3 on the Minus side
    vec1[0] = DxVx[3]; vec1[1] = DxVy[3]; vec1[2] = DxVz[3];
    vec2[0] = DyVx[3]; vec2[1] = DyVy[3]; vec2[2] = DyVz[3];
    vec3[0] = DzVx[3]; vec3[1] = DzVy[3]; vec3[2] = DzVz[3];

    //idx = ((iy+3) + (iz+3) * ny)*3*3;
    for (int ii = 0; ii < 3; ii++){
      for (int jj = 0; jj < 3; jj++){
        mat1[ii][jj] = FC.D21_1[idx + ii*3 + jj];
        mat2[ii][jj] = FC.D22_1[idx + ii*3 + jj];
        mat3[ii][jj] = FC.D23_1[idx + ii*3 + jj];
      }
    }

    fdlib_math_matmul3x1(mat1, vec1, vecg1);
    fdlib_math_matmul3x1(mat2, vec2, vecg2);
    fdlib_math_matmul3x1(mat3, vec3, vecg3);

    iptr_f = (iy+nj1) + (iz+nk1) * ny + 0 * siz_slice_yz;
    f_hT21[iptr_f] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT22[iptr_f] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT23[iptr_f] = vecg1[2] + vecg2[2] + vecg3[2];

    //idx = ((iy+3) + (iz+3) * ny)*3*3;
    for (int ii = 0; ii < 3; ii++){
      for (int jj = 0; jj < 3; jj++){
        mat1[ii][jj] = FC.D31_1[idx + ii*3 + jj];
        mat2[ii][jj] = FC.D32_1[idx + ii*3 + jj];
        mat3[ii][jj] = FC.D33_1[idx + ii*3 + jj];
      }
    }

    fdlib_math_matmul3x1(mat1, vec1, vecg1);
    fdlib_math_matmul3x1(mat2, vec2, vecg2);
    fdlib_math_matmul3x1(mat3, vec3, vecg3);

    f_hT31[iptr_f] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT32[iptr_f] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT33[iptr_f] = vecg1[2] + vecg2[2] + vecg3[2];

    // calculate f_hT2, f_hT3 on the Plus side 
    vec1[0] = DxVx[4]; vec1[1] = DxVy[4]; vec1[2] = DxVz[4];
    vec2[0] = DyVx[4]; vec2[1] = DyVy[4]; vec2[2] = DyVz[4];
    vec3[0] = DzVx[4]; vec3[1] = DzVy[4]; vec3[2] = DzVz[4];

    //idx = ((iy+3) + (iz+3) * ny)*3*3;
    for (int ii = 0; ii < 3; ii++){
      for (int jj = 0; jj < 3; jj++){
        mat1[ii][jj] = FC.D21_2[idx + ii*3 + jj];
        mat2[ii][jj] = FC.D22_2[idx + ii*3 + jj];
        mat3[ii][jj] = FC.D23_2[idx + ii*3 + jj];
      }
    }

    fdlib_math_matmul3x1(mat1, vec1, vecg1);
    fdlib_math_matmul3x1(mat2, vec2, vecg2);
    fdlib_math_matmul3x1(mat3, vec3, vecg3);

    iptr_f = (iy+nj1) + (iz+nk1) * ny + 1 * siz_slice_yz;
    f_hT21[iptr_f] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT22[iptr_f] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT23[iptr_f] = vecg1[2] + vecg2[2] + vecg3[2];

    //idx = ((iy+3) + (iz+3) * ny)*3*3;
    for (int ii = 0; ii < 3; ii++){
      for (int jj = 0; jj < 3; jj++){
        mat1[ii][jj] = FC.D31_2[idx + ii*3 + jj];
        mat2[ii][jj] = FC.D32_2[idx + ii*3 + jj];
        mat3[ii][jj] = FC.D33_2[idx + ii*3 + jj];
      }
    }

    fdlib_math_matmul3x1(mat1, vec1, vecg1);
    fdlib_math_matmul3x1(mat2, vec2, vecg2);
    fdlib_math_matmul3x1(mat3, vec3, vecg3);

    f_hT31[iptr_f] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT32[iptr_f] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT33[iptr_f] = vecg1[2] + vecg2[2] + vecg3[2];


    iptr = (i0+1) + (iy+3) *siz_line + (iz+3) * siz_slice;
    iptr_f = (iy+nj1) + (iz+nk1) * ny + 1 * siz_slice_yz;
    DxVx[5] = (Vx[iptr] - f_Vx[iptr_f]); 
    DxVy[5] = (Vy[iptr] - f_Vy[iptr_f]); 
    DxVz[5] = (Vz[iptr] - f_Vz[iptr_f]); 

    iptr_t = (iy+3) + (iz+3) * ny + 1 * siz_slice_yz;       vec_3[0] = f_Vx[iptr_t];
    iptr = (i0+1) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_3[1] = Vx[iptr];
    iptr = (i0+2) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_3[2] = Vx[iptr];
    VEC_24_B(DxVx[6], vec_3);

    iptr_t = (iy+3) + (iz+3) * ny + 1 * siz_slice_yz;       vec_3[0] = f_Vy[iptr_t];
    iptr = (i0+1) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_3[1] = Vy[iptr];
    iptr = (i0+2) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_3[2] = Vy[iptr];
    VEC_24_B(DxVy[6], vec_3);

    iptr_t = (iy+3) + (iz+3) * ny + 1 * siz_slice_yz;       vec_3[0] = f_Vz[iptr_t];
    iptr = (i0+1) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_3[1] = Vz[iptr];
    iptr = (i0+2) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_3[2] = Vz[iptr];
    VEC_24_B(DxVz[6], vec_3);

    iptr_f = (iy+3) + (iz+3) * ny + 1 * siz_slice_yz;       vec_5[0] = f_Vx[iptr_f];
    iptr = (i0+1) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[1] = Vx[iptr];
    iptr = (i0+2) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[2] = Vx[iptr];
    iptr = (i0+3) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[3] = Vx[iptr];
    iptr = (i0+4) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[4] = Vx[iptr];
    VEC_DRP_B(DxVx[7], vec_5);

    iptr_f = (iy+3) + (iz+3) * ny + 1 * siz_slice_yz;       vec_5[0] = f_Vy[iptr_f];
    iptr = (i0+1) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[1] = Vy[iptr];
    iptr = (i0+2) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[2] = Vy[iptr];
    iptr = (i0+3) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[3] = Vy[iptr];
    iptr = (i0+4) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[4] = Vy[iptr];
    VEC_DRP_B(DxVy[7], vec_5);

    iptr_f = (iy+3) + (iz+3) * ny + 1 * siz_slice_yz;       vec_5[0] = f_Vz[iptr_f];
    iptr = (i0+1) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[1] = Vz[iptr];
    iptr = (i0+2) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[2] = Vz[iptr];
    iptr = (i0+3) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[3] = Vz[iptr];
    iptr = (i0+4) + (iy+3) * siz_line + (iz+3) * siz_slice; vec_5[4] = Vz[iptr];
    VEC_DRP_B(DxVz[7], vec_5);

    for (int n = 0; n < 8 ; n++)
    {
      // n =  0  1  2  3  4  5  6  7
      // m = -3 -2 -1 -0 +0 +1 +2 +3
      int m;
      if(n < 4){
        m = n-3;
      }else{
        m = n-4;
      }
      if(m==0) continue; // do not update i0, n=3,4
      i = i0+m;

      iptr = i + (iy+3) * siz_line + (iz+3) * siz_slice;
      Vx_ptr = Vx + iptr;
      Vy_ptr = Vy + iptr;
      Vz_ptr = Vz + iptr;
      M_FD_SHIFT_PTR_MACDRP(DyVx[n],Vx_ptr,siz_line,jdir);
      M_FD_SHIFT_PTR_MACDRP(DyVy[n],Vy_ptr,siz_line,jdir);
      M_FD_SHIFT_PTR_MACDRP(DyVz[n],Vz_ptr,siz_line,jdir);

      M_FD_SHIFT_PTR_MACDRP(DzVx[n],Vx_ptr,siz_slice,kdir);
      M_FD_SHIFT_PTR_MACDRP(DzVy[n],Vy_ptr,siz_slice,kdir);
      M_FD_SHIFT_PTR_MACDRP(DzVz[n],Vz_ptr,siz_slice,kdir);

      if(is_free==1 && km==0){
        DzVx[n] = matVx2Vz[idx + 3*0 + 0] * DxVx[n]
                + matVx2Vz[idx + 3*0 + 1] * DxVy[n]
                + matVx2Vz[idx + 3*0 + 2] * DxVz[n]
                + matVy2Vz[idx + 3*0 + 0] * DyVx[n]
                + matVy2Vz[idx + 3*0 + 1] * DyVy[n]
                + matVy2Vz[idx + 3*0 + 2] * DyVz[n];

        DzVy[n] = matVx2Vz[idx + 3*1 + 0] * DxVx[n]
                + matVx2Vz[idx + 3*1 + 1] * DxVy[n]
                + matVx2Vz[idx + 3*1 + 2] * DxVz[n]
                + matVy2Vz[idx + 3*1 + 0] * DyVx[n]
                + matVy2Vz[idx + 3*1 + 1] * DyVy[n]
                + matVy2Vz[idx + 3*1 + 2] * DyVz[n];

        DzVz[n] = matVx2Vz[idx + 3*2 + 0] * DxVx[n]
                + matVx2Vz[idx + 3*2 + 1] * DxVy[n]
                + matVx2Vz[idx + 3*2 + 2] * DxVz[n]
                + matVy2Vz[idx + 3*2 + 0] * DyVx[n]
                + matVy2Vz[idx + 3*2 + 1] * DyVy[n]
                + matVy2Vz[idx + 3*2 + 2] * DyVz[n] ;
      } 
      if(is_free==1 && km==1){
        M_FD_SHIFT_PTR_MAC22(DzVx[n],Vx_ptr,siz_slice,kdir);
        M_FD_SHIFT_PTR_MAC22(DzVy[n],Vy_ptr,siz_slice,kdir);
        M_FD_SHIFT_PTR_MAC22(DzVz[n],Vz_ptr,siz_slice,kdir);
      }
      if(is_free==1 && km==2){
        M_FD_SHIFT_PTR_MAC24(DzVx[n],Vx_ptr,siz_slice,kdir);
        M_FD_SHIFT_PTR_MAC24(DzVy[n],Vy_ptr,siz_slice,kdir);
        M_FD_SHIFT_PTR_MAC24(DzVz[n],Vz_ptr,siz_slice,kdir);
      }

      // iptr = i + (iy+3) * siz_line + (iz+3) * siz_slice;
      lam = lam3d[iptr]; mu = mu3d[iptr];
      lam2mu  = lam + 2.0*mu;
      xix = xi_x[iptr]; xiy = xi_y[iptr]; xiz = xi_z[iptr];
      etx = et_x[iptr]; ety = et_y[iptr]; etz = et_z[iptr];
      ztx = zt_x[iptr]; zty = zt_y[iptr]; ztz = zt_z[iptr];

      hTxx[iptr] =   lam2mu * ( xix*DxVx[n] + etx*DyVx[n] + ztx*DzVx[n])
                   + lam    * ( xiy*DxVy[n] + ety*DyVy[n] + zty*DzVy[n]
                               +xiz*DxVz[n] + etz*DyVz[n] + ztz*DzVz[n]);

      hTyy[iptr] =    lam2mu * ( xiy*DxVy[n] + ety*DyVy[n] + zty*DzVy[n])
                    + lam    * ( xix*DxVx[n] + etx*DyVx[n] + ztx*DzVx[n]
                                +xiz*DxVz[n] + etz*DyVz[n] + ztz*DzVz[n]);

      hTzz[iptr] =    lam2mu * ( xiz*DxVz[n] + etz*DyVz[n] + ztz*DzVz[n])
                    + lam    * ( xix*DxVx[n] + etx*DyVx[n] + ztx*DzVx[n]
                                +xiy*DxVy[n] + ety*DyVy[n] + zty*DzVy[n]);

      hTxy[iptr] =  mu *(
                    xiy*DxVx[n] + xix*DxVy[n] +
                    ety*DyVx[n] + etx*DyVy[n] +
                    zty*DzVx[n] + ztx*DzVy[n] );

      hTxz[iptr] =  mu *(
                    xiz*DxVx[n] + xix*DxVz[n] +
                    etz*DyVx[n] + etx*DyVz[n] +
                    ztz*DzVx[n] + ztx*DzVz[n] ); 

      hTyz[iptr] =  mu *(
                    xiz*DxVy[n] + xiy*DxVz[n] +
                    etz*DyVy[n] + ety*DyVz[n] +
                    ztz*DzVy[n] + zty*DzVz[n] );
    } 
  } 
  return;
}
