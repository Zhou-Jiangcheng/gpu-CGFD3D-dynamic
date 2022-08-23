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
                         fault_t F,
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
  float *Tyz = w_cur_d + wav_d.Tyz_pos;
  float *Txz = w_cur_d + wav_d.Txz_pos;
  float *Txy = w_cur_d + wav_d.Txy_pos;

  float *hVx  = w_rhs_d + wav_d.Vx_pos;
  float *hVy  = w_rhs_d + wav_d.Vy_pos;
  float *hVz  = w_rhs_d + wav_d.Vz_pos;
  float *hTxx = w_rhs_d + wav_d.Txx_pos;
  float *hTyy = w_rhs_d + wav_d.Tyy_pos;
  float *hTzz = w_rhs_d + wav_d.Tzz_pos;
  float *hTyz = w_rhs_d + wav_d.Tyz_pos;
  float *hTxz = w_rhs_d + wav_d.Txz_pos;
  float *hTxy = w_rhs_d + wav_d.Txy_pos;

  float *f_Vx  = f_cur_d + FW.Vx_pos;
  float *f_Vy  = f_cur_d + FW.Vy_pos;
  float *f_Vz  = f_cur_d + FW.Vz_pos;
  float *f_T2x = f_cur_d + FW.T2x_pos;
  float *f_T2y = f_cur_d + FW.T2y_pos;
  float *f_T2z = f_cur_d + FW.T2z_pos;
  float *f_T3x = f_cur_d + FW.T3x_pos;
  float *f_T3y = f_cur_d + FW.T3y_pos;
  float *f_T3z = f_cur_d + FW.T3z_pos;

  float *f_hVx  = f_rhs_d + FW.Vx_pos;
  float *f_hVy  = f_rhs_d + FW.Vy_pos;
  float *f_hVz  = f_rhs_d + FW.Vz_pos;
  float *f_hT2x = f_rhs_d + FW.T2x_pos;
  float *f_hT2y = f_rhs_d + FW.T2y_pos;
  float *f_hT2z = f_rhs_d + FW.T2z_pos;
  float *f_hT3x = f_rhs_d + FW.T3x_pos;
  float *f_hT3y = f_rhs_d + FW.T3y_pos;
  float *f_hT3z = f_rhs_d + FW.T3z_pos;

  float *f_T1x = FW.T1x;
  float *f_T1y = FW.T1y;
  float *f_T1z = FW.T1z;

  float *f_hT1x = FW.hT1x;
  float *f_hT1y = FW.hT1y;
  float *f_hT1z = FW.hT1z;

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
                                                Txx, Tyy, Tzz,
                                                Tyz, Txz, Txy, 
                                                hVx, hVy, hVz, 
                                                f_T2x, f_T2y, f_T2z,
                                                f_T3x, f_T3y, f_T3z,
                                                f_hVx, f_hVy, f_hVz, 
                                                f_T1x, f_T1y, f_T1z,
                                                xi_x,  xi_y, xi_z,
                                                et_x, et_y, et_z, 
                                                zt_x, zt_y, zt_z,
                                                jac3d, slw3d, 
                                                F, isfree, i0,
                                                nj1, nj, nk1, nk, ny, 
                                                siz_line, siz_slice, siz_slice_yz,
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
                                                       Vx, Vy, Vz, 
                                                       hTxx, hTyy, hTzz, 
                                                       hTyz, hTxz, hTxy,
                                                       f_Vx, f_Vy, f_Vz,
                                                       f_hT2x, f_hT2y, f_hT2z,
                                                       f_hT3x, f_hT3y, f_hT3z, 
                                                       f_T1x, f_T1y, f_T1z,
                                                       f_hT1x, f_hT1y, f_hT1z,
                                                       xi_x, xi_y, xi_z, 
                                                       et_x, et_y, et_z, 
                                                       zt_x, zt_y, zt_z,
                                                       lam3d, mu3d, slw3d, 
                                                       F, matVx2Vz, matVy2Vz,
                                                       isfree, i0, nj1, nj, nk1, nk, ny, 
                                                       siz_line, siz_slice, siz_slice_yz,
                                                       idir, jdir, kdir)
    }
    if(idir == 0) 
    {
      sv_eq1st_curv_col_el_iso_rhs_fault_stress_B_gpu <<<grid, block>>>(
                                                       Vx, Vy, Vz, 
                                                       hTxx, hTyy, hTzz,
                                                       hTyz, hTxz, hTxy,
                                                       f_Vx, f_Vy, f_Vz,
                                                       f_hT2x, f_hT2y, f_hT2z,
                                                       f_hT3x, f_hT3y, f_hT3z,
                                                       f_T1x, f_T1y, f_T1z,
                                                       f_hT1x, f_hT1y, f_hT1z,
                                                       xi_x, xi_y, xi_z,
                                                       et_x, et_y, et_z,
                                                       zt_x, zt_y, zt_z,
                                                       lam3d, mu3d, slw3d,
                                                       F, matVx2Vz, matVy2Vz,
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
                       float * Tyz, float * Txz, float * Txy,
                       float * hVx, float * hVy, float * hVz,
                       float * f_T2x, float * f_T2y, float * f_T2z,
                       float * f_T3x, float * f_T3y, float * f_T3z,
                       float * f_hVx, float * f_hVy, float * f_hVz,
                       float * f_T1x, float * f_T1y, float * f_T1z,
                       float * xi_x,  float * xi_y, float * xi_z,
                       float * et_x,  float * et_y, float * et_z,
                       float * zt_x,  float * zt_y, float * zt_z,
                       float * jac3d, float * slw3d, fault_t F, int isfree, 
                       int i0, int nj1, int nj, int nk1, int nk, int ny, 
                       size_t siz_line, size_t siz_slice, size_t siz_slice_yz,
                       int idir, int jdir, int kdir)
{
  size_t iy = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iz = blockIdx.y * blockDim.y + threadIdx.y;

  float rrhojac;
  size_t iptr, iptr_f, iptr_t;
  float *T2x_ptr;
  float *T2y_ptr;
  float *T2z_ptr;
  float *T3x_ptr;
  float *T3y_ptr;
  float *T3z_ptr;

  float vecT1x[7], vecT1y[7], vecT1z[7];
  float vecT2x[7], vecT2y[7], vecT2z[7];
  float vecT3x[7], vecT3y[7], vecT3z[7];
  iptr_t = iy + iz * nj;
  if (iy < nj && iz < nk ) 
  { 
    if(F.united[iptr_t] == 1) return;
    int km = nk - (iz+1); 
    int n_free = km+3;

    for (int i = i0-3; i <= i0+3; i++)
    {
      int n = i0 - i; 
      if(n==0) continue; // skip Split nodes
      for (int l = -3; l <= 3; l++)
      {
        iptr = (i+l) + (iy+nj1) * siz_line + (iz+nk1) * siz_slice;
        vecT1x[l+3] = jac3d[iptr]*(xi_x[iptr]*Txx[iptr] + xi_y[iptr]*Txy[iptr] + xi_z[iptr]*Txz[iptr]);
        vecT1y[l+3] = jac3d[iptr]*(xi_x[iptr]*Txy[iptr] + xi_y[iptr]*Tyy[iptr] + xi_z[iptr]*Tyz[iptr]);
        vecT1z[l+3] = jac3d[iptr]*(xi_x[iptr]*Txz[iptr] + xi_y[iptr]*Tyz[iptr] + xi_z[iptr]*Tzz[iptr]);

        iptr = i + (iy+nj1+l) * siz_line + (iz+nk1) * siz_slice;
        vecT2x[l+3] = jac3d[iptr]*(et_x[iptr]*Txx[iptr] + et_y[iptr]*Txy[iptr] + et_z[iptr]*Txz[iptr]);
        vecT2y[l+3] = jac3d[iptr]*(et_x[iptr]*Txy[iptr] + et_y[iptr]*Tyy[iptr] + et_z[iptr]*Tyz[iptr]);
        vecT2z[l+3] = jac3d[iptr]*(et_x[iptr]*Txz[iptr] + et_y[iptr]*Tyz[iptr] + et_z[iptr]*Tzz[iptr]);

        iptr = i + (iy+nj1) *siz_line + (iz+nk1+l) * siz_slice;
        vecT3x[l+3] = jac3d[iptr]*(zt_x[iptr]*Txx[iptr] + zt_y[iptr]*Txy[iptr] + zt_z[iptr]*Txz[iptr]);
        vecT3y[l+3] = jac3d[iptr]*(zt_x[iptr]*Txy[iptr] + zt_y[iptr]*Tyy[iptr] + zt_z[iptr]*Tyz[iptr]);
        vecT3z[l+3] = jac3d[iptr]*(zt_x[iptr]*Txz[iptr] + zt_y[iptr]*Tyz[iptr] + zt_z[iptr]*Tzz[iptr]);
      }

      iptr_f = (iy+nj1) + (iz+nk1) * ny + 3 * siz_slice_yz; 
      vecT1x[n+3] = f_T1x[iptr_f]; // fault T1x
      vecT1y[n+3] = f_T1y[iptr_f]; // fault T1y
      vecT1z[n+3] = f_T1z[iptr_f]; // fault T1z

      //  TractionImg
      if (n==2) { // i0-2
        vecT1x[6] = 2.0*vecT1x[5] - vecT1x[4];
        vecT1y[6] = 2.0*vecT1y[5] - vecT1y[4];
        vecT1z[6] = 2.0*vecT1z[5] - vecT1z[4];
      }
      if (n==1) { // i0-1
        vecT1x[5] = 2.0*vecT1x[4] - vecT1x[3];
        vecT1y[5] = 2.0*vecT1y[4] - vecT1y[3];
        vecT1z[5] = 2.0*vecT1z[4] - vecT1z[3];
        vecT1x[6] = 2.0*vecT1x[4] - vecT1x[2];
        vecT1y[6] = 2.0*vecT1y[4] - vecT1y[2];
        vecT1z[6] = 2.0*vecT1z[4] - vecT1z[2];
      }
      if (n==-1) { // i0+1
        vecT1x[0] = 2.0*vecT1x[2] - vecT1x[4];
        vecT1y[0] = 2.0*vecT1y[2] - vecT1y[4];
        vecT1z[0] = 2.0*vecT1z[2] - vecT1z[4];
        vecT1x[1] = 2.0*vecT1x[2] - vecT1x[3];
        vecT1y[1] = 2.0*vecT1y[2] - vecT1y[3];
        vecT1z[1] = 2.0*vecT1z[2] - vecT1z[3];
      }
      if (n==-2) { // i0+2
        vecT1x[0] = 2.0*vecT1x[1] - vecT1x[2];
        vecT1y[0] = 2.0*vecT1y[1] - vecT1y[2];
        vecT1z[0] = 2.0*vecT1z[1] - vecT1z[2];
      }

      if(isfree == 1 && km<=3)
      {
        vecT3x[n_free] = 0.0;
        vecT3y[n_free] = 0.0;
        vecT3z[n_free] = 0.0;
        for (int l = n_free+1; l<7; l++){
          vecT3x[l] = -vecT3x[2*n_free-l];
          vecT3y[l] = -vecT3y[2*n_free-l];
          vecT3z[l] = -vecT3z[2*n_free-l];
        }
      }

      M_FD_VEC(DxT1x, vecT1x+3, idir);
      M_FD_VEC(DxT1y, vecT1y+3, idir);
      M_FD_VEC(DxT1z, vecT1z+3, idir);
      M_FD_VEC(DyT2x, vecT2x+3, jdir);
      M_FD_VEC(DyT2y, vecT2y+3, jdir);
      M_FD_VEC(DyT2z, vecT2z+3, jdir);
      M_FD_VEC(DzT3x, vecT3x+3, kdir);
      M_FD_VEC(DzT3y, vecT3y+3, kdir);
      M_FD_VEC(DzT3z, vecT3z+3, kdir);

      iptr = i + (iy+nj1) * siz_line + (iz+nk1) * siz_slice;
      rrhojac = slw3d[iptr] / jac3d[iptr];

      hVx[iptr] = (DxT1x+DyT2x+DzT3x)*rrhojac;
      hVy[iptr] = (DxT1y+DyT2y+DzT3y)*rrhojac;
      hVz[iptr] = (DxT1z+DyT2z+DzT3z)*rrhojac;
    } // end of loop i

    // update velocity at the fault plane
    // 0 for minus side on the fault
    // 1 for plus  side on the fault
    for (int m = 0; m < 2; m++)
    {
      iptr_f = (iy+nj1) + (iz+nk1) * ny
      if(m==0){ // "-" side
        DxT1x =     a_0*f_T1x[iptr_f+3*siz_slice_yz] 
                  - a_1*f_T1x[iptr_f+2*siz_slice_yz] 
                  - a_2*f_T1x[iptr_f+1*siz_slice_yz] 
                  - a_3*f_T1x[iptr_f+0*siz_slice_yz];

        DxT1y =     a_0*f_T1y[iptr_f+3*siz_slice_yz]
                  - a_1*f_T1y[iptr_f+2*siz_slice_yz]
                  - a_2*f_T1y[iptr_f+1*siz_slice_yz]
                  - a_3*f_T1y[iptr_f+0*siz_slice_yz];

        DxT1z =     a_0*f_T1z[iptr_f+3*siz_slice_yz]
                  - a_1*f_T1z[iptr_f+2*siz_slice_yz]
                  - a_2*f_T1z[iptr_f+1*siz_slice_yz]
                  - a_3*f_T1z[iptr_f+0*siz_slice_yz];
      }else{ // "+" side
        DxT1x =   - a_0*f_T1x[iptr_f+3*siz_slice_yz]
                  + a_1*f_T1x[iptr_f+4*siz_slice_yz]
                  + a_2*f_T1x[iptr_f+5*siz_slice_yz]
                  + a_3*f_T1x[iptr_f+6*siz_slice_yz];

        DxT1y =   - a_0*f_T1y[iptr_f+3*siz_slice_yz]
                  + a_1*f_T1y[iptr_f+4*siz_slice_yz]
                  + a_2*f_T1y[iptr_f+5*siz_slice_yz]
                  + a_3*f_T1y[iptr_f+6*siz_slice_yz];

        DxT1z =   - a_0*f_T1z[iptr_f+3*siz_slice_yz]
                  + a_1*f_T1z[iptr_f+4*siz_slice_yz]
                  + a_2*f_T1z[iptr_f+5*siz_slice_yz]
                  + a_3*f_T1z[iptr_f+6*siz_slice_yz];
      }
      T2x_ptr = f_T2x + m*siz_slice_yz + iptr_f;
      T2y_ptr = f_T2y + m*siz_slice_yz + iptr_f;
      T2z_ptr = f_T2z + m*siz_slice_yz + iptr_f;
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

      T3x_ptr = f_T3x + m*siz_slice_yz + iptr_f;
      T3y_ptr = f_T3y + m*siz_slice_yz + iptr_f;
      T3z_ptr = f_T3z + m*siz_slice_yz + iptr_f;
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
      // recalculate free surface
      if(isfree == 1 && km<=3)
      {
        for (int l=-3; l<=3; l++)
        {
          iptr_f = (iy+3) + (iz+3+l) * ny + m * siz_slice_yz;
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

      iptr = i0 + (iy+nj1) * siz_line + (iz+nk1) * siz_slice;
      rrhojac = slw3d[iptr] / jac3d[iptr];

      iptr_f = (iy+nj1) + (iz+nk1) * ny + m * siz_slice_yz; 
      f_hVx[iptr_f] = (DxT1x+DyT2x+DzT3x)*rrhojac;
      f_hVy[iptr_f] = (DxT1y+DyT2y+DzT3y)*rrhojac;
      f_hVz[iptr_f] = (DxT1z+DyT2z+DzT3z)*rrhojac;
    } 
  } 
  return;
}

__global__
void sv_eq1st_curv_col_el_iso_rhs_fault_stress_F_gpu(
                       float * Vx, float * Vy, float * Vz,
                       float * hTxx, float * hTyy, float * hTzz,
                       float * hTyz, float * hTxz, float * hTxy,
                       float * f_Vx, float * f_Vy, float * f_Vz,
                       float * f_hT2x, float * f_hT2y, float * f_hT2z,
                       float * f_hT3x, float * f_hT3y, float * f_hT3z,
                       float * f_T1x, float * f_T1y, float * f_T1z,
                       float * f_hT1x,float * f_hT1y,float * f_hT1z,
                       float * xi_x,  float * xi_y, float * xi_z,
                       float * et_x,  float * et_y, float * et_z,
                       float * zt_x,  float * zt_y, float * zt_z,
                       float * lam3d, float * mu3d, float * slw3d, 
                       fault_t F, float *matVx2Vz, float *matVy2Vz,
                       int isfree, int i0, 
                       int nj1, int nj, int nk1, int nk, int ny, 
                       size_t siz_line, size_t siz_slice, size_t siz_slice_yz,
                       int idir, int jdir, int kdir)
{
  int iy = blockIdx.x * blockDim.x + threadIdx.x;
  int iz = blockIdx.y * blockDim.y + threadIdx.y;

  size_t iptr, iptr_f, iptr_t;
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

  float matT1toVx_Min[3][3];
  float matVytoVx_Min[3][3];
  float matVztoVx_Min[3][3];
  float matT1toVx_Plus[3][3];
  float matVytoVx_Plus[3][3];
  float matVztoVx_Plus[3][3];

  float matT1toVxf_Min[3][3];
  float matVytoVxf_Min[3][3];
  float matT1toVxf_Plus[3][3];
  float matVytoVxf_Plus[3][3];

  float matPlus2Min1[3][3];
  float matPlus2Min2[3][3];
  float matPlus2Min3[3][3];
  float matPlus2Min4[3][3];
  float matPlus2Min5[3][3];

  float matPlus2Min1f[3][3], matPlus2Min2f[3][3], matPlus2Min3f[3][3];
  float matVx2Vz1[3][3], matVy2Vz1[3][3];
  float matVx2Vz2[3][3], matVy2Vz2[3][3];

  int idx = ((iy+3) + (iz+3) * ny)*3*3;
  iptr_t = iy + iz * nj;
  if (iy < nj && iz < nk ) { 
    if(F.united[iptr_t] == 1) return;
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
    if(F.rup_index_y[iptr_t] == 1)
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
    if(F.rup_index_y[iptr_t] == 0)
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
    if(F.rup_index_z[iptr_t] == 1){
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
    if(F.rup_index_z[iptr_t] == 0){
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

    for (int i = 0; i < 3; i++){
      for (int j = 0; j < 3; j++){
        matPlus2Min1[i][j] = FC.matPlus2Min1[idx + i*3 + j];
        matPlus2Min2[i][j] = FC.matPlus2Min2[idx + i*3 + j];
        matPlus2Min3[i][j] = FC.matPlus2Min3[idx + i*3 + j];
        matPlus2Min4[i][j] = FC.matPlus2Min4[idx + i*3 + j];
        matPlus2Min5[i][j] = FC.matPlus2Min5[idx + i*3 + j];

        matT1toVxm[i][j] = FC.matT1toVxm[idx + i*3 + j];
        matVytoVxm[i][j] = FC.matVytoVxm[idx + i*3 + j];
        matVztoVxm[i][j] = FC.matVztoVxm[idx + i*3 + j];
        matT1toVxp[i][j] = FC.matT1toVxp[idx + i*3 + j];
        matVytoVxp[i][j] = FC.matVytoVxp[idx + i*3 + j];
        matVztoVxp[i][j] = FC.matVztoVxp[idx + i*3 + j];
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
      if(F.faultgrid[iptr_t] == 1){
        iptr_f = (iy+3)+(iz+3)*ny;
        dtT1[0] = f_hT1x[iptr_f];   // inner use 1st order
        dtT1[1] = f_hT1y[iptr_f];
        dtT1[2] = f_hT1z[iptr_f];
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

      fdlib_math_matmul3x1(matT1toVx_Plus, dtT1, out1);
      fdlib_math_matmul3x1(matVytoVx_Plus, dyV2, out2);
      fdlib_math_matmul3x1(matVztoVx_Plus, dzV2, out3);

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

      fdlib_math_matmul3x1(matT1toVx_Min, dtT1, out1);
      fdlib_math_matmul3x1(matVytoVx_Min, dyV1, out2);
      fdlib_math_matmul3x1(matVztoVx_Min, dzV1, out3);

      DxVx[3] = out1[0] - out2[0] - out3[0];
      DxVy[3] = out1[1] - out2[1] - out3[1];
      DxVz[3] = out1[2] - out2[2] - out3[2];
    }
    // recalculate free surface point
    if(isfree == 1 && km==0) 
    {
      for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
          matVx2Vz1    [i][j] = FC.matVx2Vz1    [(iy+3)*9 + i*3 + j];
          matVx2Vz2    [i][j] = FC.matVx2Vz2    [(iy+3)*9 + i*3 + j];
          matVy2Vz1    [i][j] = FC.matVy2Vz1    [(iy+3)*9 + i*3 + j];
          matVy2Vz2    [i][j] = FC.matVy2Vz2    [(iy+3)*9 + i*3 + j];
          matPlus2Min1_f[i][j] = FC.matPlus2Min1f[(iy+3)*9 + i*3 + j];
          matPlus2Min2_f[i][j] = FC.matPlus2Min2f[(iy+3)*9 + i*3 + j];
          matPlus2Min3_f[i][j] = FC.matPlus2Min3f[(iy+3)*9 + i*3 + j];
          matT1toVxf_Min[i][j] = FC.matT1toVxf_Min[(iy+3)*9 + i*3 + j];
          matVytoVxf_Min[i][j] = FC.matVytoVxf_Min[(iy+3)*9 + i*3 + j];
          matT1toVxf_Plus[i][j] = FC.matT1toVxf_Plus[(iy+3)*9 + i*3 + j];
          matVytoVxf_Plus[i][j] = FC.matVytoVxf_Plus[(iy+3)*9 + i*3 + j];
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
        if(F.faultgrid[iptr_t] == 1){
          iptr_f = (iy+3)+(iz+3)*ny;
          dtT1[0] = f_hT1x[iptr_f];
          dtT1[1] = f_hT1y[iptr_f];
          dtT1[2] = f_hT1z[iptr_f];
        }else{
          dtT1[0] = 0.0;
          dtT1[1] = 0.0;
          dtT1[2] = 0.0;
        }

        // plus
        dyV2[0] = DyVx[4];
        dyV2[1] = DyVy[4];
        dyV2[2] = DyVz[4];

        fdlib_math_matmul3x1(matT1toVxf_Plus, dtT1, out1);
        fdlib_math_matmul3x1(matVytoVxf_Plus, dyV2, out2);

        DxVx[4] = out1[0] - out2[0];
        DxVy[4] = out1[1] - out2[1];
        DxVz[4] = out1[2] - out2[2];
        // minus
        dyV1[0] = DyVx[3];
        dyV1[1] = DyVy[3];
        dyV1[2] = DyVz[3];

        fdlib_math_matmul3x1(matT1toVxf_Min, dtT1, out1);
        fdlib_math_matmul3x1(matVytoVxf_Min, dyV1, out2);

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
      for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
          matPlus2Min1[i][j] = FC.matPlus2Min1[idx + i*3 + j];
          matPlus2Min2[i][j] = FC.matPlus2Min2[idx + i*3 + j];
          matPlus2Min3[i][j] = FC.matPlus2Min3[idx + i*3 + j];
          matPlus2Min4[i][j] = FC.matPlus2Min4[idx + i*3 + j];
          matPlus2Min5[i][j] = FC.matPlus2Min5[idx + i*3 + j];

          matT1toVx_Min[i][j] = FC.matT1toVx_Min[idx + i*3 + j];
          matVytoVx_Min[i][j] = FC.matVytoVx_Min[idx + i*3 + j];
          matVztoVx_Min[i][j] = FC.matVztoVx_Min[idx + i*3 + j];
          matT1toVx_Plus[i][j] = FC.matT1toVx_Plus[idx + i*3 + j];
          matVytoVx_Plus[i][j] = FC.matVytoVx_Plus[idx + i*3 + j];
          matVztoVx_Plus[i][j] = FC.matVztoVx_Plus[idx + i*3 + j];
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
        if(F.faultgrid[iptr_t] == 1){
          iptr_f = (iy+3)+(iz+3)*ny;
          dtT1[0] = f_hT1x[iptr_f];
          dtT1[1] = f_hT1y[iptr_f];
          dtT1[2] = f_hT1z[iptr_f];
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

        fdlib_math_matmul3x1(matT1toVx_Plus, dtT1, out1);
        fdlib_math_matmul3x1(matVytoVx_Plus, dyV2, out2);
        fdlib_math_matmul3x1(matVztoVx_Plus, dzV2, out3);

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

        fdlib_math_matmul3x1(matT1toVx_Min, dtT1, out1);
        fdlib_math_matmul3x1(matVytoVx_Min, dyV1, out2);
        fdlib_math_matmul3x1(matVztoVx_Min, dzV1, out3);

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
    for (int i = 0; i < 3; i++){
      for (int j = 0; j < 3; j++){
        mat1[i][j] = FC.D21_2[idx + i*3 + j];
        mat2[i][j] = FC.D22_2[idx + i*3 + j];
        mat3[i][j] = FC.D23_2[idx + i*3 + j];
      }
    }

    fdlib_math_matmul3x1(mat1, vec1, vecg1);
    fdlib_math_matmul3x1(mat2, vec2, vecg2);
    fdlib_math_matmul3x1(mat3, vec3, vecg3);

    iptr_f = (iy+nj1) + (iz+nk1) * ny + 1 * siz_slice_yz;
    f_hT2x[iptr_f] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT2y[iptr_f] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT2z[iptr_f] = vecg1[2] + vecg2[2] + vecg3[2];

    //idx = ((iy+3) + (iz+3) * ny)*3*3;
    for (int i = 0; i < 3; i++){
      for (int j = 0; j < 3; j++){
        mat1[i][j] = FC.D31_2[idx + i*3 + j];
        mat2[i][j] = FC.D32_2[idx + i*3 + j];
        mat3[i][j] = FC.D33_2[idx + i*3 + j];
      }
    }

    fdlib_math_matmul3x1(mat1, vec1, vecg1);
    fdlib_math_matmul3x1(mat2, vec2, vecg2);
    fdlib_math_matmul3x1(mat3, vec3, vecg3);

    f_hT3x[iptr_f] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT3y[iptr_f] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT3z[iptr_f] = vecg1[2] + vecg2[2] + vecg3[2];

    // calculate f_hT2, f_hT3 on the Minus side
    vec1[0] = DxVx[3]; vec1[1] = DxVy[3]; vec1[2] = DxVz[3];
    vec2[0] = DyVx[3]; vec2[1] = DyVy[3]; vec2[2] = DyVz[3];
    vec3[0] = DzVx[3]; vec3[1] = DzVy[3]; vec3[2] = DzVz[3];

    //idx = ((iy+3) + (iz+3) * ny)*3*3;
    for (int i = 0; i < 3; i++){
      for (int j = 0; j < 3; j++){
        mat1[i][j] = FC.D21_1[idx + i*3 + j];
        mat2[i][j] = FC.D22_1[idx + i*3 + j];
        mat3[i][j] = FC.D23_1[idx + i*3 + j];
      }
    }

    fdlib_math_matmul3x1(mat1, vec1, vecg1);
    fdlib_math_matmul3x1(mat2, vec2, vecg2);
    fdlib_math_matmul3x1(mat3, vec3, vecg3);

    iptr_f = (iy+nj1) + (iz+nk1) * ny + 0 * siz_slice_yz;
    f_hT2x[iptr_f] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT2y[iptr_f] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT2z[iptr_f] = vecg1[2] + vecg2[2] + vecg3[2];

    //idx = ((iy+3) + (iz+3) * ny)*3*3;
    for (int i = 0; i < 3; i++){
      for (int j = 0; j < 3; j++){
        mat1[i][j] = FC.D31_1[idx + i*3 + j];
        mat2[i][j] = FC.D32_1[idx + i*3 + j];
        mat3[i][j] = FC.D33_1[idx + i*3 + j];
      }
    }

    fdlib_math_matmul3x1(mat1, vec1, vecg1);
    fdlib_math_matmul3x1(mat2, vec2, vecg2);
    fdlib_math_matmul3x1(mat3, vec3, vecg3);

    f_hT3x[iptr_f] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT3y[iptr_f] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT3z[iptr_f] = vecg1[2] + vecg2[2] + vecg3[2];

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
                       float * hTyz, float * hTxz, float * hTxy,
                       float * f_Vx, float * f_Vy, float * f_Vz,
                       float * f_hT2x, float * f_hT2y, float * f_hT2z,
                       float * f_hT3x, float * f_hT3y, float * f_hT3z,
                       float * f_T1x, float * f_T1y, float * f_T1z,
                       float * f_hT1x,float * f_hT1y,float * f_hT1z,
                       float * xi_x,  float * xi_y, float * xi_z,
                       float * et_x,  float * et_y, float * et_z,
                       float * zt_x,  float * zt_y, float * zt_z,
                       float * lam3d, float * mu3d, float * slw3d, 
                       fault_t F, float *matVx2Vz, float *matVy2Vz,
                       int isfree, int i0, 
                       int nj1, int nj, int nk1, int nk, int ny, 
                       size_t siz_line, size_t siz_slice, size_t siz_slice_yz,
                       int idir, int jdir, int kdir)
{
  int iy = blockIdx.x * blockDim.x + threadIdx.x;
  int iz = blockIdx.y * blockDim.y + threadIdx.y;

  float iptr, iptr_f, iptr_t;
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

  float matT1toVx_Min[3][3];
  float matVytoVx_Min[3][3];
  float matVztoVx_Min[3][3];
  float matT1toVx_Plus[3][3];
  float matVytoVx_Plus[3][3];
  float matVztoVx_Plus[3][3];

  float matT1toVxf_Min[3][3];
  float matVytoVxf_Min[3][3];
  float matT1toVxf_Plus[3][3];
  float matVytoVxf_Plus[3][3];

  float matMin2Plus1[3][3];
  float matMin2Plus2[3][3];
  float matMin2Plus3[3][3];
  float matMin2Plus4[3][3];
  float matMin2Plus5[3][3];

  float matMin2Plus1f[3][3], matMin2Plus2f[3][3], matMin2Plus3f[3][3];
  float matVx2Vz1[3][3], matVy2Vz1[3][3];
  float matVx2Vz2[3][3], matVy2Vz2[3][3];

  int idx = ((iy+3) + (iz+3) * ny)*3*3;
  iptr_t = iy + iz * nj;
  if (iy < nj && iz < nk ) { 
    if(F.united[iptr_t] == 1) return;
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
    if(F.rup_index_y[iptr_t] == 1)
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
    if(F.rup_index_y[iptr_t] == 0)
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
    if(F.rup_index_z[iptr_t] == 1){
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
    if(F.rup_index_z[iptr_t] == 0){
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

    for (int i = 0; i < 3; i++){
      for (int j = 0; j < 3; j++){
        matMin2Plus1[i][j] = FC.matMin2Plus1[idx + i*3 + j];
        matMin2Plus2[i][j] = FC.matMin2Plus2[idx + i*3 + j];
        matMin2Plus3[i][j] = FC.matMin2Plus3[idx + i*3 + j];
        matMin2Plus4[i][j] = FC.matMin2Plus4[idx + i*3 + j];
        matMin2Plus5[i][j] = FC.matMin2Plus5[idx + i*3 + j];

        matT1toVx_Min[i][j] = FC.matT1toVx_Min[idx + i*3 + j];
        matVytoVx_Min[i][j] = FC.matVytoVx_Min[idx + i*3 + j];
        matVztoVx_Min[i][j] = FC.matVztoVx_Min[idx + i*3 + j];
        matT1toVx_Plus[i][j] = FC.matT1toVx_Plus[idx + i*3 + j];
        matVytoVx_Plus[i][j] = FC.matVytoVx_Plus[idx + i*3 + j];
        matVztoVx_Plus[i][j] = FC.matVztoVx_Plus[idx + i*3 + j];
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
      if(F.faultgrid[iptr_t] == 1){
        iptr_f = (iy+3)+(iz+3)*ny;
        dtT1[0] = f_hT1x[iptr_f];
        dtT1[1] = f_hT1y[iptr_f];
        dtT1[2] = f_hT1z[iptr_f];
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

      fdlib_math_matmul3x1(matT1toVx_Min, dtT1, out1);
      fdlib_math_matmul3x1(matVytoVx_Min, dyV1, out2);
      fdlib_math_matmul3x1(matVztoVx_Min, dzV1, out3);

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

      fdlib_math_matmul3x1(matT1toVx_Plus, dtT1, out1);
      fdlib_math_matmul3x1(matVytoVx_Plus, dyV2, out2);
      fdlib_math_matmul3x1(matVztoVx_Plus, dzV2, out3);

      DxVx[4] = out1[0] - out2[0] - out3[0];
      DxVy[4] = out1[1] - out2[1] - out3[1];
      DxVz[4] = out1[2] - out2[2] - out3[2];
    }
    // recalculate free surface point
    if(isfree == 1 && km==0) 
    {
      for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
          matVx2Vz1    [i][j] = FC.matVx2Vz1    [(iy+3)*9 + i*3 + j];
          matVx2Vz2    [i][j] = FC.matVx2Vz2    [(iy+3)*9 + i*3 + j];
          matVy2Vz1    [i][j] = FC.matVy2Vz1    [(iy+3)*9 + i*3 + j];
          matVy2Vz2    [i][j] = FC.matVy2Vz2    [(iy+3)*9 + i*3 + j];
          matMin2Plus1_f[i][j] = FC.matMin2Plus1f[(iy+3)*9 + i*3 + j];
          matMin2Plus2_f[i][j] = FC.matMin2Plus2f[(iy+3)*9 + i*3 + j];
          matMin2Plus3_f[i][j] = FC.matMin2Plus3f[(iy+3)*9 + i*3 + j];
          matT1toVxf_Min[i][j] = FC.matT1toVxf_Min[(iy+3)*9 + i*3 + j];
          matVytoVxf_Min[i][j] = FC.matVytoVxf_Min[(iy+3)*9 + i*3 + j];
          matT1toVxf_Plus[i][j] = FC.matT1toVxf_Plus[(iy+3)*9 + i*3 + j];
          matVytoVxf_Plus[i][j] = FC.matVytoVxf_Plus[(iy+3)*9 + i*3 + j];
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
        if(F.faultgrid[iptr_t] == 1){
          iptr_f = (iy+3)+(iz+3)*ny;
          dtT1[0] = f_hT1x1[iptr_f];
          dtT1[1] = f_hT1y2[iptr_f];
          dtT1[2] = f_hT1z3[iptr_f];
        }else{
          dtT1[0] = 0.0;
          dtT1[1] = 0.0;
          dtT1[2] = 0.0;
        }
        // minus
        dyV1[0] = DyVx[3];
        dyV1[1] = DyVy[3];
        dyV1[2] = DyVz[3];

        fdlib_math_matmul3x1(matT1toVxf_Min, dtT1, out1);
        fdlib_math_matmul3x1(matVytoVxf_Min, dyV1, out2);

        DxVx[3] = out1[0] - out2[0];
        DxVy[3] = out1[1] - out2[1];
        DxVz[3] = out1[2] - out2[2];
        // plus
        dyV2[0] = DyVx[4];
        dyV2[1] = DyVy[4];
        dyV2[2] = DyVz[4];

        fdlib_math_matmul3x1(matT1toVxf_Plus, dtT1, out1);
        fdlib_math_matmul3x1(matVytoVxf_Plus, dyV2, out2);

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
      for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
          matMin2Plus1[i][j] = FC.matMin2Plus1[idx + i*3 + j];
          matMin2Plus2[i][j] = FC.matMin2Plus2[idx + i*3 + j];
          matMin2Plus3[i][j] = FC.matMin2Plus3[idx + i*3 + j];
          matMin2Plus4[i][j] = FC.matMin2Plus4[idx + i*3 + j];
          matMin2Plus5[i][j] = FC.matMin2Plus5[idx + i*3 + j];

          matT1toVx_Min[i][j] = FC.matT1toVx_Min[idx + i*3 + j];
          matVytoVx_Min[i][j] = FC.matVytoVx_Min[idx + i*3 + j];
          matVztoVx_Min[i][j] = FC.matVztoVx_Min[idx + i*3 + j];
          matT1toVx_Plus[i][j] = FC.matT1toVx_Plus[idx + i*3 + j];
          matVytoVx_Plus[i][j] = FC.matVytoVx_Plus[idx + i*3 + j];
          matVztoVx_Plus[i][j] = FC.matVztoVx_Plus[idx + i*3 + j];
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
        if(F.faultgrid[iptr_t] == 1) {
          iptr_f = (iy+3)+(iz+3)*ny;
          dtT1[0] = f_hT1x1[iptr_f];
          dtT1[1] = f_hT1y2[iptr_f];
          dtT1[2] = f_hT1z3[iptr_f];
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

        fdlib_math_matmul3x1(matT1toVx_Min, dtT1, out1);
        fdlib_math_matmul3x1(matVytoVx_Min, dyV1, out2);
        fdlib_math_matmul3x1(matVztoVx_Min, dzV1, out3);

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

        fdlib_math_matmul3x1(matT1toVx_Plus, dtT1, out1);
        fdlib_math_matmul3x1(matVytoVx_Plus, dyV2, out2);
        fdlib_math_matmul3x1(matVztoVx_Plus, dzV2, out3);

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
    for (int i = 0; i < 3; i++){
      for (int j = 0; j < 3; j++){
        mat1[i][j] = FC.D21_1[idx + i*3 + j];
        mat2[i][j] = FC.D22_1[idx + i*3 + j];
        mat3[i][j] = FC.D23_1[idx + i*3 + j];
      }
    }

    fdlib_math_matmul3x1(mat1, vec1, vecg1);
    fdlib_math_matmul3x1(mat2, vec2, vecg2);
    fdlib_math_matmul3x1(mat3, vec3, vecg3);

    iptr_f = (iy+nj1) + (iz+nk1) * ny + 0 * siz_slice_yz;
    f_hT2x[iptr_f] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT2y[iptr_f] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT2z[iptr_f] = vecg1[2] + vecg2[2] + vecg3[2];

    //idx = ((iy+3) + (iz+3) * ny)*3*3;
    for (int i = 0; i < 3; i++){
      for (int j = 0; j < 3; j++){
        mat1[i][j] = FC.D31_1[idx + i*3 + j];
        mat2[i][j] = FC.D32_1[idx + i*3 + j];
        mat3[i][j] = FC.D33_1[idx + i*3 + j];
      }
    }

    fdlib_math_matmul3x1(mat1, vec1, vecg1);
    fdlib_math_matmul3x1(mat2, vec2, vecg2);
    fdlib_math_matmul3x1(mat3, vec3, vecg3);

    f_hT3x[iptr_f] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT3y[iptr_f] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT3z[iptr_f] = vecg1[2] + vecg2[2] + vecg3[2];

    // calculate f_hT2, f_hT3 on the Plus side 
    vec1[0] = DxVx[4]; vec1[1] = DxVy[4]; vec1[2] = DxVz[4];
    vec2[0] = DyVx[4]; vec2[1] = DyVy[4]; vec2[2] = DyVz[4];
    vec3[0] = DzVx[4]; vec3[1] = DzVy[4]; vec3[2] = DzVz[4];

    //idx = ((iy+3) + (iz+3) * ny)*3*3;
    for (int i = 0; i < 3; i++){
      for (int j = 0; j < 3; j++){
        mat1[i][j] = FC.D21_2[idx + i*3 + j];
        mat2[i][j] = FC.D22_2[idx + i*3 + j];
        mat3[i][j] = FC.D23_2[idx + i*3 + j];
      }
    }

    fdlib_math_matmul3x1(mat1, vec1, vecg1);
    fdlib_math_matmul3x1(mat2, vec2, vecg2);
    fdlib_math_matmul3x1(mat3, vec3, vecg3);

    iptr_f = (iy+nj1) + (iz+nk1) * ny + 1 * siz_slice_yz;
    f_hT2x[iptr_f] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT2y[iptr_f] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT2z[iptr_f] = vecg1[2] + vecg2[2] + vecg3[2];

    //idx = ((iy+3) + (iz+3) * ny)*3*3;
    for (int i = 0; i < 3; i++){
      for (int j = 0; j < 3; j++){
        mat1[i][j] = FC.D31_2[idx + i*3 + j];
        mat2[i][j] = FC.D32_2[idx + i*3 + j];
        mat3[i][j] = FC.D33_2[idx + i*3 + j];
      }
    }

    fdlib_math_matmul3x1(mat1, vec1, vecg1);
    fdlib_math_matmul3x1(mat2, vec2, vecg2);
    fdlib_math_matmul3x1(mat3, vec3, vecg3);

    f_hT3x[iptr_f] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT3y[iptr_f] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT3z[iptr_f] = vecg1[2] + vecg2[2] + vecg3[2];


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
