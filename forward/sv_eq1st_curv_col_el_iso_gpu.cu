/*******************************************************************************
 * solver of isotropic elastic 1st-order eqn using curv grid and collocated scheme
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "fdlib_mem.h"
#include "fdlib_math.h"
#include "sv_eq1st_curv_col_el_iso_gpu.h"
#include "cuda_common.h"

//#define SV_EQ1ST_CURV_COLGRD_ISO_DEBUG

/*******************************************************************************
 * perform one stage calculation of rhs
 ******************************************************************************/

void
sv_eq1st_curv_col_el_iso_onestage(
  float *w_cur_d,
  float *rhs_d, 
  wav_t  wav_d,
  gdinfo_t  gdinfo_d,
  gdcurv_metric_t metric_d,
  md_t md_d,
  bdryfree_t bdryfree_d,
  bdrypml_t  bdrypml_d,
  // include different order/stentil
  fd_op_t *fdx_op,
  fd_op_t *fdy_op,
  fd_op_t *fdz_op,
  const int myid, const int verbose)
{
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
  float *hVx   = rhs_d   + wav_d.Vx_pos ; 
  float *hVy   = rhs_d   + wav_d.Vy_pos ; 
  float *hVz   = rhs_d   + wav_d.Vz_pos ; 
  float *hTxx  = rhs_d   + wav_d.Txx_pos; 
  float *hTyy  = rhs_d   + wav_d.Tyy_pos; 
  float *hTzz  = rhs_d   + wav_d.Tzz_pos; 
  float *hTxz  = rhs_d   + wav_d.Txz_pos; 
  float *hTyz  = rhs_d   + wav_d.Tyz_pos; 
  float *hTxy  = rhs_d   + wav_d.Txy_pos; 

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

  // grid size
  int ni1 = gdinfo_d.ni1;
  int ni2 = gdinfo_d.ni2;
  int nj1 = gdinfo_d.nj1;
  int nj2 = gdinfo_d.nj2;
  int nk1 = gdinfo_d.nk1;
  int nk2 = gdinfo_d.nk2;

  int ni  = gdinfo_d.ni;
  int nj  = gdinfo_d.nj;
  int nk  = gdinfo_d.nk;
  int nx  = gdinfo_d.nx;
  int ny  = gdinfo_d.ny;
  int nz  = gdinfo_d.nz;
  size_t siz_line   = gdinfo_d.siz_line;
  size_t siz_slice  = gdinfo_d.siz_slice;
  size_t siz_volume = gdinfo_d.siz_volume;

  float *matVx2Vz = bdryfree_d.matVx2Vz2;
  float *matVy2Vz = bdryfree_d.matVy2Vz2;

  int idir = fdx_op->dir;
  int jdir = fdy_op->dir;
  int kdir = fdz_op->dir;
  

  {
    dim3 block(8,8,8);
    dim3 grid;
    grid.x = (ni+block.x-1)/block.x;
    grid.y = (nj+block.y-1)/block.y;
    grid.z = (nk+block.z-1)/block.z;
    sv_eq1st_curv_col_el_iso_rhs_inner_gpu <<<grid, block>>> (
                        Vx,Vy,Vz,Txx,Tyy,Tzz,Txz,Tyz,Txy,
                        hVx,hVy,hVz,hTxx,hTyy,hTzz,hTxz,hTyz,hTxy,
                        xi_x, xi_y, xi_z, et_x, et_y, et_z, zt_x, zt_y, zt_z,
                        lam3d, mu3d, slw3d,
                        ni1,ni,nj1,nj,nk1,nk,siz_line,siz_slice,
                        idir, jdir, kdir,
                        myid, verbose);
    CUDACHECK( cudaDeviceSynchronize() );
  }

  // free, abs, source in turn
  // free surface at z2
  if (bdryfree_d.is_at_sides[2][1] == 1)
  {
    // tractiong
    {
      dim3 block(8,8);
      dim3 grid;
      grid.x = (ni+block.x-1)/block.x;
      grid.y = (nj+block.y-1)/block.y;
      sv_eq1st_curv_col_el_iso_rhs_timg_z2_gpu  <<<grid, block>>> (
                          Txx,Tyy,Tzz,Txz,Tyz,Txy,hVx,hVy,hVz,
                          xi_x, xi_y, xi_z, et_x, et_y, et_z, zt_x, zt_y, zt_z,
                          jac3d, slw3d,
                          ni1,ni,nj1,nj,nk1,nk2,siz_line,siz_slice,
                          idir, jdir, kdir,
                          myid, verbose);
      cudaDeviceSynchronize();
    }
    // velocity: vlow
    {
      dim3 block(8,8);
      dim3 grid;
      grid.x = (ni+block.x-1)/block.x;
      grid.y = (nj+block.y-1)/block.y;
      sv_eq1st_curv_col_el_iso_rhs_vlow_z2_gpu  <<<grid, block>>> (
                        Vx,Vy,Vz,hTxx,hTyy,hTzz,hTxz,hTyz,hTxy,
                        xi_x, xi_y, xi_z, et_x, et_y, et_z, zt_x, zt_y, zt_z,
                        lam3d, mu3d, slw3d,
                        matVx2Vz,matVy2Vz,
                        ni1,ni,nj1,nj,nk1,nk2,siz_line,siz_slice,
                        idir, jdir, kdir,
                        myid, verbose);
      CUDACHECK( cudaDeviceSynchronize() );
    }
  }

  // cfs-pml, loop face inside
  if (bdrypml_d.is_enable == 1)
  {
    sv_eq1st_curv_col_el_iso_rhs_cfspml(Vx,Vy,Vz,Txx,Tyy,Tzz,Txz,Tyz,Txy,
                                        hVx,hVy,hVz,hTxx,hTyy,hTzz,hTxz,hTyz,hTxy,
                                        xi_x, xi_y, xi_z, et_x, et_y, et_z, zt_x, zt_y, zt_z,
                                        lam3d, mu3d, slw3d,
                                        nk2, siz_line,siz_slice,
                                        bdrypml_d, bdryfree_d,
                                        idir, jdir, kdir,
                                        myid, verbose);
  }
  
  // end func
  return;
}

/*******************************************************************************
 * calculate all points without boundaries treatment
 ******************************************************************************/

__global__ void
sv_eq1st_curv_col_el_iso_rhs_inner_gpu(
    float *  Vx , float *  Vy , float *  Vz ,
    float *  Txx, float *  Tyy, float *  Tzz,
    float *  Txz, float *  Tyz, float *  Txy,
    float * hVx , float * hVy , float * hVz ,
    float * hTxx, float * hTyy, float * hTzz,
    float * hTxz, float * hTyz, float * hTxy,
    float * xi_x, float * xi_y, float * xi_z,
    float * et_x, float * et_y, float * et_z,
    float * zt_x, float * zt_y, float * zt_z,
    float * lam3d, float * mu3d, float * slw3d,
    int ni1, int ni, int nj1, int nj, int nk1, int nk,
    size_t siz_line, size_t siz_slice,
    int idir, int jdir, int kdir,
    const int myid, const int verbose)
{
  
  // local var
  float DxTxx,DxTyy,DxTzz,DxTxy,DxTxz,DxTyz,DxVx,DxVy,DxVz;
  float DyTxx,DyTyy,DyTzz,DyTxy,DyTxz,DyTyz,DyVx,DyVy,DyVz;
  float DzTxx,DzTyy,DzTzz,DzTxy,DzTxz,DzTyz,DzVx,DzVy,DzVz;
  float lam,mu,lam2mu,slw;
  float xix,xiy,xiz,etx,ety,etz,ztx,zty,ztz;

  float * Vx_ptr;
  float * Vy_ptr;
  float * Vz_ptr;
  float * Txx_ptr;
  float * Txy_ptr;
  float * Txz_ptr;
  float * Tyy_ptr;
  float * Tzz_ptr;
  float * Tyz_ptr;

  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iz = blockIdx.z * blockDim.z + threadIdx.z;

  // caclu all points
  if(ix<ni && iy<nj && iz<nk)
  {
    size_t iptr = (ix+ni1) + (iy+nj1) * siz_line + (iz+nk1) * siz_slice;

    Vx_ptr = Vx + iptr;
    Vy_ptr = Vy + iptr;
    Vz_ptr = Vz + iptr;
    Txx_ptr = Txx + iptr;
    Tyy_ptr = Tyy + iptr;
    Tzz_ptr = Tzz + iptr;
    Txz_ptr = Txz + iptr;
    Tyz_ptr = Tyz + iptr;
    Txy_ptr = Txy + iptr;

    // Vx derivatives
    M_FD_SHIFT_PTR_MACDRP(DxVx, Vx_ptr, 1,         idir);
    M_FD_SHIFT_PTR_MACDRP(DyVx, Vx_ptr, siz_line,  jdir);
    M_FD_SHIFT_PTR_MACDRP(DzVx, Vx_ptr, siz_slice, kdir);

    // Vy derivatives
    M_FD_SHIFT_PTR_MACDRP(DxVy, Vy_ptr, 1,         idir);
    M_FD_SHIFT_PTR_MACDRP(DyVy, Vy_ptr, siz_line,  jdir);
    M_FD_SHIFT_PTR_MACDRP(DzVy, Vy_ptr, siz_slice, kdir);

    // Vz derivatives
    M_FD_SHIFT_PTR_MACDRP(DxVz, Vz_ptr, 1,         idir);
    M_FD_SHIFT_PTR_MACDRP(DyVz, Vz_ptr, siz_line,  jdir);
    M_FD_SHIFT_PTR_MACDRP(DzVz, Vz_ptr, siz_slice, kdir);

    // Txx derivatives
    M_FD_SHIFT_PTR_MACDRP(DxTxx, Txx_ptr, 1,         idir);
    M_FD_SHIFT_PTR_MACDRP(DyTxx, Txx_ptr, siz_line,  jdir);
    M_FD_SHIFT_PTR_MACDRP(DzTxx, Txx_ptr, siz_slice, kdir);

    // Tyy derivatives
    M_FD_SHIFT_PTR_MACDRP(DxTyy, Tyy_ptr, 1,         idir);
    M_FD_SHIFT_PTR_MACDRP(DyTyy, Tyy_ptr, siz_line,  jdir);
    M_FD_SHIFT_PTR_MACDRP(DzTyy, Tyy_ptr, siz_slice, kdir);

    // Tzz derivatives
    M_FD_SHIFT_PTR_MACDRP(DxTzz, Tzz_ptr, 1,         idir);
    M_FD_SHIFT_PTR_MACDRP(DyTzz, Tzz_ptr, siz_line,  jdir);
    M_FD_SHIFT_PTR_MACDRP(DzTzz, Tzz_ptr, siz_slice, kdir);

    // Txz derivatives
    M_FD_SHIFT_PTR_MACDRP(DxTxz, Txz_ptr, 1,         idir);
    M_FD_SHIFT_PTR_MACDRP(DyTxz, Txz_ptr, siz_line,  jdir);
    M_FD_SHIFT_PTR_MACDRP(DzTxz, Txz_ptr, siz_slice, kdir);

    // Tyz derivatives
    M_FD_SHIFT_PTR_MACDRP(DxTyz, Tyz_ptr, 1,         idir);
    M_FD_SHIFT_PTR_MACDRP(DyTyz, Tyz_ptr, siz_line,  jdir);
    M_FD_SHIFT_PTR_MACDRP(DzTyz, Tyz_ptr, siz_slice, kdir);

    // Txy derivatives
    M_FD_SHIFT_PTR_MACDRP(DxTxy, Txy_ptr, 1,         idir);
    M_FD_SHIFT_PTR_MACDRP(DyTxy, Txy_ptr, siz_line,  jdir);
    M_FD_SHIFT_PTR_MACDRP(DzTxy, Txy_ptr, siz_slice, kdir);

    
    // metric
    xix = xi_x[iptr];
    xiy = xi_y[iptr];
    xiz = xi_z[iptr];
    etx = et_x[iptr];
    ety = et_y[iptr];
    etz = et_z[iptr];
    ztx = zt_x[iptr];
    zty = zt_y[iptr];
    ztz = zt_z[iptr];

    // medium
    lam = lam3d[iptr];
    mu  =  mu3d[iptr];
    slw = slw3d[iptr];
    lam2mu = lam + 2.0 * mu;

    // moment equation
    hVx[iptr] = slw*( xix*DxTxx + xiy*DxTxy + xiz*DxTxz  
                     +etx*DyTxx + ety*DyTxy + etz*DyTxz 
                     +ztx*DzTxx + zty*DzTxy + ztz*DzTxz );
    hVy[iptr] = slw*( xix*DxTxy + xiy*DxTyy + xiz*DxTyz
                     +etx*DyTxy + ety*DyTyy + etz*DyTyz
                     +ztx*DzTxy + zty*DzTyy + ztz*DzTyz );
    hVz[iptr] = slw*( xix*DxTxz + xiy*DxTyz + xiz*DxTzz 
                     +etx*DyTxz + ety*DyTyz + etz*DyTzz
                     +ztx*DzTxz + zty*DzTyz + ztz*DzTzz );

    // Hooke's equatoin
    hTxx[iptr] =  lam2mu * ( xix*DxVx  +etx*DyVx + ztx*DzVx)
                + lam    * ( xiy*DxVy + ety*DyVy + zty*DzVy
                            +xiz*DxVz + etz*DyVz + ztz*DzVz);

    hTyy[iptr] = lam2mu * ( xiy*DxVy + ety*DyVy + zty*DzVy)
                +lam    * ( xix*DxVx + etx*DyVx + ztx*DzVx
                           +xiz*DxVz + etz*DyVz + ztz*DzVz);

    hTzz[iptr] = lam2mu * ( xiz*DxVz + etz*DyVz + ztz*DzVz)
                +lam    * ( xix*DxVx  +etx*DyVx  +ztx*DzVx
                           +xiy*DxVy + ety*DyVy + zty*DzVy);

    hTxy[iptr] = mu *(
                 xiy*DxVx + xix*DxVy
                +ety*DyVx + etx*DyVy
                +zty*DzVx + ztx*DzVy
                );
    hTxz[iptr] = mu *(
                 xiz*DxVx + xix*DxVz
                +etz*DyVx + etx*DyVz
                +ztz*DzVx + ztx*DzVz
                );
    hTyz[iptr] = mu *(
                 xiz*DxVy + xiy*DxVz
                +etz*DyVy + ety*DyVz
                +ztz*DzVy + zty*DzVz
                );
  }
}

/*******************************************************************************
 * free surface boundary
 ******************************************************************************/

/*
 * implement traction image boundary 
 */

__global__ void
sv_eq1st_curv_col_el_iso_rhs_timg_z2_gpu(
    float *  Txx, float *  Tyy, float *  Tzz,
    float *  Txz, float *  Tyz, float *  Txy,
    float * hVx , float * hVy , float * hVz ,
    float * xi_x, float * xi_y, float * xi_z,
    float * et_x, float * et_y, float * et_z,
    float * zt_x, float * zt_y, float * zt_z,
    float * jac3d, float * slw3d,
    int ni1, int ni, int nj1, int nj, int nk1, int nk2,
    size_t siz_line, size_t siz_slice, 
    int idir, int jdir, int kdir,
    const int myid, const int verbose)
{
  // local var
  float DxTx,DyTy,DzTz;
  float slwjac;
  float xix,xiy,xiz,etx,ety,etz,ztx,zty,ztz;

  // to save traction and other two dir force var
  float vecxi[5] = {0.0};
  float vecet[5] = {0.0};
  float veczt[5] = {0.0};
  int fdx_indx[5] = {-1,0,1,2,3};// Forward -1 ~ 3
  int fdy_indx[5] = {-1,0,1,2,3};
  int fdz_indx[5] = {-1,0,1,2,3};
  int n, iptr4vec;
  if(idir == 0)
  {
    for(int i=0; i<5; i++)
    {
      fdx_indx[i] = i-3; // Backward -1 ~ 3
    }
  } 

  if(jdir == 0)
  {
    for(int j=0; j<5; j++)
    {
      fdy_indx[j] = j-3; 
    }
  } else 

  if(kdir == 0)
  {
    for(int k=0; k<5; k++)
    {
      fdz_indx[k] = k-3; 
    }
  }  

  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;

  // last indx, free surface force Tx/Ty/Tz to 0 in cal
  int k_min = nk2 - fdz_indx[4];


  // point affected by timg
  for (int k=k_min; k <= nk2; k++)
  {
    // k corresponding to current point

    // index of free surface in veczt[];
    int n_free = nk2 - k - fdz_indx[0]; // first indx is negative
    // index distance between free surface and current point
    int index_dis = nk2-k;

    if(ix<ni && iy<nj)
    {
      size_t iptr = (ix+ni1) + (iy+nj1) * siz_line + k * siz_slice;
      // metric
      xix = xi_x[iptr];
      xiy = xi_y[iptr];
      xiz = xi_z[iptr];
      etx = et_x[iptr];
      ety = et_y[iptr];
      etz = et_z[iptr];
      ztx = zt_x[iptr];
      zty = zt_y[iptr];
      ztz = zt_z[iptr];

      // slowness and jac
      slwjac = slw3d[iptr] / jac3d[iptr];

      //
      // for hVx
      //

      // transform to conservative vars
      for (n=0; n<5; n++) {
        iptr4vec = iptr + fdx_indx[n];
        vecxi[n] = jac3d[iptr4vec] * (  xi_x[iptr4vec] * Txx[iptr4vec]
                                      + xi_y[iptr4vec] * Txy[iptr4vec]
                                      + xi_z[iptr4vec] * Txz[iptr4vec] );
      }
      for (n=0; n<5; n++) {
        iptr4vec = iptr + fdy_indx[n] * siz_line;
        vecet[n] = jac3d[iptr4vec] * (  et_x[iptr4vec] * Txx[iptr4vec]
                                      + et_y[iptr4vec] * Txy[iptr4vec]
                                      + et_z[iptr4vec] * Txz[iptr4vec] );
      }

      // blow surface -> cal
      for (n=0; n<n_free; n++) {
        iptr4vec = iptr + fdz_indx[n]  * siz_slice;
        veczt[n] = jac3d[iptr4vec] * (  zt_x[iptr4vec] * Txx[iptr4vec]
                                      + zt_y[iptr4vec] * Txy[iptr4vec]
                                      + zt_z[iptr4vec] * Txz[iptr4vec] );
      }

      // at surface -> set to 0
      veczt[n_free] = 0.0;

      // above surface -> mirror
      for (n=n_free+1; n<5; n++)
      {
        //int n_img = fdz_indx[n] - 2*(n-n_free);
        int n_img = index_dis - (n-n_free); // this method more easy to understand mirror point
        iptr4vec = iptr + n_img * siz_slice;
        veczt[n] = -jac3d[iptr4vec] * (  zt_x[iptr4vec] * Txx[iptr4vec]
                                       + zt_y[iptr4vec] * Txy[iptr4vec]
                                       + zt_z[iptr4vec] * Txz[iptr4vec] );
      }

      // deri
      M_FD_VEC_DRP(DxTx, vecxi, idir);
      M_FD_VEC_DRP(DyTy, vecet, jdir);
      M_FD_VEC_DRP(DzTz, veczt, kdir);

      hVx[iptr] = ( DxTx+DyTy+DzTz ) * slwjac;

      //
      // for hVy
      //

      // transform to conservative vars
      for (n=0; n<5; n++) {
        iptr4vec = iptr + fdx_indx[n];
        vecxi[n] = jac3d[iptr4vec] * (  xi_x[iptr4vec] * Txy[iptr4vec]
                                      + xi_y[iptr4vec] * Tyy[iptr4vec]
                                      + xi_z[iptr4vec] * Tyz[iptr4vec] );
      }
      for (n=0; n<5; n++) {
        iptr4vec = iptr + fdy_indx[n] * siz_line;
        vecet[n] = jac3d[iptr4vec] * (  et_x[iptr4vec] * Txy[iptr4vec]
                                      + et_y[iptr4vec] * Tyy[iptr4vec]
                                      + et_z[iptr4vec] * Tyz[iptr4vec] );
      }

      // blow surface -> cal
      for (n=0; n<n_free; n++) {
        iptr4vec = iptr + fdz_indx[n] * siz_slice;
        veczt[n] = jac3d[iptr4vec] * (  zt_x[iptr4vec] * Txy[iptr4vec]
                                      + zt_y[iptr4vec] * Tyy[iptr4vec]
                                      + zt_z[iptr4vec] * Tyz[iptr4vec] );
      }

      // at surface -> set to 0
      veczt[n_free] = 0.0;

      // above surface -> mirror
      for (n=n_free+1; n<5; n++) {
        //int n_img = fdz_indx[n] - 2*(n-n_free);
        int n_img = index_dis - (n-n_free);
        iptr4vec = iptr + n_img * siz_slice;
        veczt[n] = -jac3d[iptr4vec] * (  zt_x[iptr4vec] * Txy[iptr4vec]
                                       + zt_y[iptr4vec] * Tyy[iptr4vec]
                                       + zt_z[iptr4vec] * Tyz[iptr4vec] );
      }

      // deri
      M_FD_VEC_DRP(DxTx, vecxi, idir);
      M_FD_VEC_DRP(DyTy, vecet, jdir);
      M_FD_VEC_DRP(DzTz, veczt, kdir);

      hVy[iptr] = ( DxTx+DyTy+DzTz ) * slwjac;

      //
      // for hVz
      //

      // transform to conservative vars
      for (n=0; n<5; n++) {
        iptr4vec = iptr + fdx_indx[n];
        vecxi[n] = jac3d[iptr4vec] * (  xi_x[iptr4vec] * Txz[iptr4vec]
                                      + xi_y[iptr4vec] * Tyz[iptr4vec]
                                      + xi_z[iptr4vec] * Tzz[iptr4vec] );
      }
      for (n=0; n<5; n++) {
        iptr4vec = iptr + fdy_indx[n] * siz_line;
        vecet[n] = jac3d[iptr4vec] * (  et_x[iptr4vec] * Txz[iptr4vec]
                                      + et_y[iptr4vec] * Tyz[iptr4vec]
                                      + et_z[iptr4vec] * Tzz[iptr4vec] );
      }

      // blow surface -> cal
      for (n=0; n<n_free; n++) {
        iptr4vec = iptr + fdz_indx[n] * siz_slice;
        veczt[n] = jac3d[iptr4vec] * (  zt_x[iptr4vec] * Txz[iptr4vec]
                                      + zt_y[iptr4vec] * Tyz[iptr4vec]
                                      + zt_z[iptr4vec] * Tzz[iptr4vec] );
      }

      // at surface -> set to 0
      veczt[n_free] = 0.0;

      // above surface -> mirror
      for (n=n_free+1; n<5; n++) {
        //int n_img = fdz_indx[n] - 2*(n-n_free);
        int n_img = index_dis - (n-n_free);
        iptr4vec = iptr + n_img * siz_slice;
        veczt[n] = -jac3d[iptr4vec] * (  zt_x[iptr4vec] * Txz[iptr4vec]
                                       + zt_y[iptr4vec] * Tyz[iptr4vec]
                                       + zt_z[iptr4vec] * Tzz[iptr4vec] );
      }

      // for hVx 
      // deri
      M_FD_VEC_DRP(DxTx, vecxi, idir);
      M_FD_VEC_DRP(DyTy, vecet, jdir);
      M_FD_VEC_DRP(DzTz, veczt, kdir);

      hVz[iptr] = ( DxTx+DyTy+DzTz ) * slwjac;
    }
  }
}

/*
 * implement vlow boundary
 */

__global__ void
sv_eq1st_curv_col_el_iso_rhs_vlow_z2_gpu(
    float *  Vx , float *  Vy , float *  Vz ,
    float * hTxx, float * hTyy, float * hTzz,
    float * hTxz, float * hTyz, float * hTxy,
    float * xi_x, float * xi_y, float * xi_z,
    float * et_x, float * et_y, float * et_z,
    float * zt_x, float * zt_y, float * zt_z,
    float * lam3d, float * mu3d, float * slw3d,
    float * matVx2Vz, float * matVy2Vz,
    int ni1, int ni, int nj1, int nj, int nk1, int nk2,
    size_t siz_line, size_t siz_slice,
    int idir, int jdir, int kdir,
    const int myid, const int verbose)
{

  // local var
  int k;
  // local var
  float DxVx,DxVy,DxVz;
  float DyVx,DyVy,DyVz;
  float DzVx,DzVy,DzVz;
  float lam,mu,lam2mu,slw;
  float xix,xiy,xiz,etx,ety,etz,ztx,zty,ztz;

  float * Vx_ptr;
  float * Vy_ptr;
  float * Vz_ptr;

  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  // loop near surface layers
  for (int n=0; n < 3; n++)
  {
    // conver to k index, from surface to inner
    k = nk2 - n;

    if(ix<ni && iy<nj)
    {
      size_t iptr   = (ix+ni1) + (iy+nj1) * siz_line + k * siz_slice;

      // metric
      xix = xi_x[iptr];
      xiy = xi_y[iptr];
      xiz = xi_z[iptr];
      etx = et_x[iptr];
      ety = et_y[iptr];
      etz = et_z[iptr];
      ztx = zt_x[iptr];
      zty = zt_y[iptr];
      ztz = zt_z[iptr];

      // medium
      lam = lam3d[iptr];
      mu  =  mu3d[iptr];
      slw = slw3d[iptr];
      lam2mu = lam + 2.0 * mu;

      Vx_ptr = Vx + iptr;
      Vy_ptr = Vy + iptr;
      Vz_ptr = Vz + iptr;

      // Vx derivatives
      M_FD_SHIFT_PTR_MACDRP(DxVx, Vx_ptr, 1,        idir);
      M_FD_SHIFT_PTR_MACDRP(DyVx, Vx_ptr, siz_line, jdir);

      // Vy derivatives
      M_FD_SHIFT_PTR_MACDRP(DxVy, Vy_ptr, 1,        idir); 
      M_FD_SHIFT_PTR_MACDRP(DyVy, Vy_ptr, siz_line, jdir); 

      // Vz derivatives
      M_FD_SHIFT_PTR_MACDRP(DxVz, Vz_ptr, 1,        idir); 
      M_FD_SHIFT_PTR_MACDRP(DyVz, Vz_ptr, siz_line, jdir); 

      if (k==nk2) // at surface, convert
      {
        size_t ij = ((ix+ni1) + (iy+nj1) * siz_line)*9;
        DzVx = matVx2Vz[ij+3*0+0] * DxVx
             + matVx2Vz[ij+3*0+1] * DxVy
             + matVx2Vz[ij+3*0+2] * DxVz
             + matVy2Vz[ij+3*0+0] * DyVx
             + matVy2Vz[ij+3*0+1] * DyVy
             + matVy2Vz[ij+3*0+2] * DyVz;

        DzVy = matVx2Vz[ij+3*1+0] * DxVx
             + matVx2Vz[ij+3*1+1] * DxVy
             + matVx2Vz[ij+3*1+2] * DxVz
             + matVy2Vz[ij+3*1+0] * DyVx
             + matVy2Vz[ij+3*1+1] * DyVy
             + matVy2Vz[ij+3*1+2] * DyVz;

        DzVz = matVx2Vz[ij+3*2+0] * DxVx
             + matVx2Vz[ij+3*2+1] * DxVy
             + matVx2Vz[ij+3*2+2] * DxVz
             + matVy2Vz[ij+3*2+0] * DyVx
             + matVy2Vz[ij+3*2+1] * DyVy
             + matVy2Vz[ij+3*2+2] * DyVz;
      }
      if (k==nk2-1) // lower than surface, lower order
      {
        M_FD_SHIFT_PTR_MAC22(DzVx, Vx_ptr, siz_slice, kdir);
        M_FD_SHIFT_PTR_MAC22(DzVy, Vy_ptr, siz_slice, kdir);
        M_FD_SHIFT_PTR_MAC22(DzVz, Vz_ptr, siz_slice, kdir);
      }
      if (k==nk2-2)
      {
        M_FD_SHIFT_PTR_MAC24(DzVx, Vx_ptr, siz_slice, kdir);
        M_FD_SHIFT_PTR_MAC24(DzVy, Vy_ptr, siz_slice, kdir);
        M_FD_SHIFT_PTR_MAC24(DzVz, Vz_ptr, siz_slice, kdir);
      }

      // Hooke's equatoin
      hTxx[iptr] =  lam2mu * ( xix*DxVx  +etx*DyVx + ztx*DzVx)
                  + lam    * ( xiy*DxVy + ety*DyVy + zty*DzVy
                              +xiz*DxVz + etz*DyVz + ztz*DzVz);

      hTyy[iptr] = lam2mu * ( xiy*DxVy + ety*DyVy + zty*DzVy)
                  +lam    * ( xix*DxVx + etx*DyVx + ztx*DzVx
                             +xiz*DxVz + etz*DyVz + ztz*DzVz);

      hTzz[iptr] = lam2mu * ( xiz*DxVz + etz*DyVz + ztz*DzVz)
                  +lam    * ( xix*DxVx  +etx*DyVx  +ztx*DzVx
                             +xiy*DxVy + ety*DyVy + zty*DzVy);

      hTxy[iptr] = mu *(
                   xiy*DxVx + xix*DxVy
                  +ety*DyVx + etx*DyVy
                  +zty*DzVx + ztx*DzVy
                  );
      hTxz[iptr] = mu *(
                   xiz*DxVx + xix*DxVz
                  +etz*DyVx + etx*DyVz
                  +ztz*DzVx + ztx*DzVz
                  );
      hTyz[iptr] = mu *(
                   xiz*DxVy + xiy*DxVz
                  +etz*DyVy + ety*DyVz
                  +ztz*DzVy + zty*DzVz
                  );
    }
  }
}

/*******************************************************************************
 * CFS-PML boundary
 ******************************************************************************/

/*
 * cfspml, reference to each pml var inside function
 */

void
sv_eq1st_curv_col_el_iso_rhs_cfspml(
    float *  Vx , float *  Vy , float *  Vz ,
    float *  Txx, float *  Tyy, float *  Tzz,
    float *  Txz, float *  Tyz, float *  Txy,
    float * hVx , float * hVy , float * hVz ,
    float * hTxx, float * hTyy, float * hTzz,
    float * hTxz, float * hTyz, float * hTxy,
    float * xi_x, float * xi_y, float * xi_z,
    float * et_x, float * et_y, float * et_z,
    float * zt_x, float * zt_y, float * zt_z,
    float * lam3d, float *  mu3d, float * slw3d,
    int nk2, size_t siz_line, size_t siz_slice,
    bdrypml_t bdrypml, bdryfree_t bdryfree,
    int idir, int jdir, int kdir,
    const int myid, const int verbose)
{
  // check each side
  for (int idim=0; idim<CONST_NDIM; idim++)
  {
    for (int iside=0; iside<2; iside++)
    {
      // skip to next face if not cfspml
      if (bdrypml.is_at_sides[idim][iside] == 0) continue;

      // get index into local var
      int abs_ni1 = bdrypml.ni1[idim][iside];
      int abs_ni2 = bdrypml.ni2[idim][iside];
      int abs_nj1 = bdrypml.nj1[idim][iside];
      int abs_nj2 = bdrypml.nj2[idim][iside];
      int abs_nk1 = bdrypml.nk1[idim][iside];
      int abs_nk2 = bdrypml.nk2[idim][iside];

      
      int abs_ni = abs_ni2-abs_ni1+1; 
      int abs_nj = abs_nj2-abs_nj1+1; 
      int abs_nk = abs_nk2-abs_nk1+1; 
      {
        dim3 block(8,4,4);
        dim3 grid;
        grid.x = (abs_ni+block.x-1)/block.x;
        grid.y = (abs_nj+block.y-1)/block.y;
        grid.z = (abs_nk+block.z-1)/block.z;

        sv_eq1st_curv_col_el_iso_rhs_cfspml_gpu <<<grid, block>>> (
                                idim, iside,
                                Vx , Vy , Vz , Txx,  Tyy,  Tzz,
                                Txz,  Tyz,  Txy, hVx , hVy , hVz,
                                hTxx, hTyy, hTzz, hTxz, hTyz, hTxy,
                                xi_x, xi_y, xi_z, et_x, et_y, et_z,
                                zt_x, zt_y, zt_z, lam3d, mu3d, slw3d,
                                nk2, siz_line, siz_slice,
                                bdrypml, bdryfree, 
                                idir, jdir, kdir,
                                myid, verbose);
        cudaDeviceSynchronize();
      }
    } // iside
  } // idim
}

__global__ void
sv_eq1st_curv_col_el_iso_rhs_cfspml_gpu(int idim, int iside,
                                        float *  Vx , float *  Vy , float *  Vz ,
                                        float *  Txx, float *  Tyy, float *  Tzz,
                                        float *  Txz, float *  Tyz, float *  Txy,
                                        float * hVx , float * hVy , float * hVz ,
                                        float * hTxx, float * hTyy, float * hTzz,
                                        float * hTxz, float * hTyz, float * hTxy,
                                        float * xi_x, float * xi_y, float * xi_z,
                                        float * et_x, float * et_y, float * et_z,
                                        float * zt_x, float * zt_y, float * zt_z,
                                        float * lam3d, float *  mu3d, float * slw3d,
                                        int nk2, size_t siz_line, size_t siz_slice,
                                        bdrypml_t bdrypml, bdryfree_t bdryfree,
                                        int idir, int jdir, int kdir,
                                        const int myid, const int verbose)
{
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iz = blockIdx.z * blockDim.z + threadIdx.z;
  float *matVx2Vz = bdryfree.matVx2Vz2;
  float *matVy2Vz = bdryfree.matVy2Vz2;
  // local
  size_t iptr, iptr_a;
  float coef_A, coef_B, coef_D, coef_B_minus_1;

  float * Vx_ptr;
  float * Vy_ptr;
  float * Vz_ptr;
  float * Txx_ptr;
  float * Txy_ptr;
  float * Txz_ptr;
  float * Tyy_ptr;
  float * Tzz_ptr;
  float * Tyz_ptr;

  // get index into local var
  int abs_ni1 = bdrypml.ni1[idim][iside];
  int abs_ni2 = bdrypml.ni2[idim][iside];
  int abs_nj1 = bdrypml.nj1[idim][iside];
  int abs_nj2 = bdrypml.nj2[idim][iside];
  int abs_nk1 = bdrypml.nk1[idim][iside];
  int abs_nk2 = bdrypml.nk2[idim][iside];

  
  int abs_ni = abs_ni2-abs_ni1+1; 
  int abs_nj = abs_nj2-abs_nj1+1; 
  int abs_nk = abs_nk2-abs_nk1+1; 

  // val on point
  float DxTxx,DxTyy,DxTzz,DxTxy,DxTxz,DxTyz,DxVx,DxVy,DxVz;
  float DyTxx,DyTyy,DyTzz,DyTxy,DyTxz,DyTyz,DyVx,DyVy,DyVz;
  float DzTxx,DzTyy,DzTzz,DzTxy,DzTxz,DzTyz,DzVx,DzVy,DzVz;
  float lam,mu,lam2mu,slw;
  float xix,xiy,xiz,etx,ety,etz,ztx,zty,ztz;
  float hVx_rhs,hVy_rhs,hVz_rhs;
  float hTxx_rhs,hTyy_rhs,hTzz_rhs,hTxz_rhs,hTyz_rhs,hTxy_rhs;
  // for free surface
  float Dx_DzVx,Dy_DzVx,Dx_DzVy,Dy_DzVy,Dx_DzVz,Dy_DzVz;
  // get coef for this face
  float * ptr_coef_A = bdrypml.A[idim][iside];
  float * ptr_coef_B = bdrypml.B[idim][iside];
  float * ptr_coef_D = bdrypml.D[idim][iside];

  bdrypml_auxvar_t *auxvar = &(bdrypml.auxvar[idim][iside]);

  // get pml vars
  float * abs_vars_cur = auxvar->cur;
  float * abs_vars_rhs = auxvar->rhs;

  float * pml_Vx   = abs_vars_cur + auxvar->Vx_pos;
  float * pml_Vy   = abs_vars_cur + auxvar->Vy_pos;
  float * pml_Vz   = abs_vars_cur + auxvar->Vz_pos;
  float * pml_Txx  = abs_vars_cur + auxvar->Txx_pos;
  float * pml_Tyy  = abs_vars_cur + auxvar->Tyy_pos;
  float * pml_Tzz  = abs_vars_cur + auxvar->Tzz_pos;
  float * pml_Txz  = abs_vars_cur + auxvar->Txz_pos;
  float * pml_Tyz  = abs_vars_cur + auxvar->Tyz_pos;
  float * pml_Txy  = abs_vars_cur + auxvar->Txy_pos;

  float * pml_hVx  = abs_vars_rhs + auxvar->Vx_pos;
  float * pml_hVy  = abs_vars_rhs + auxvar->Vy_pos;
  float * pml_hVz  = abs_vars_rhs + auxvar->Vz_pos;
  float * pml_hTxx = abs_vars_rhs + auxvar->Txx_pos;
  float * pml_hTyy = abs_vars_rhs + auxvar->Tyy_pos;
  float * pml_hTzz = abs_vars_rhs + auxvar->Tzz_pos;
  float * pml_hTxz = abs_vars_rhs + auxvar->Txz_pos;
  float * pml_hTyz = abs_vars_rhs + auxvar->Tyz_pos;
  float * pml_hTxy = abs_vars_rhs + auxvar->Txy_pos;
  // for each dim
  if (idim == 0 ) // x direction
  {
    if(ix<abs_ni  && iy<abs_nj && iz<abs_nk)
    {
      iptr_a = iz*(abs_nj*abs_ni) + iy*abs_ni + ix;
      iptr   = (ix + abs_ni1) + (iy+abs_nj1) * siz_line + (iz+abs_nk1) * siz_slice;
      // pml coefs
      // int abs_i = ix;
      coef_D = ptr_coef_D[ix];
      coef_A = ptr_coef_A[ix];
      coef_B = ptr_coef_B[ix];
      coef_B_minus_1 = coef_B - 1.0;

      // metric
      xix = xi_x[iptr];
      xiy = xi_y[iptr];
      xiz = xi_z[iptr];

      // medium
      lam = lam3d[iptr];
      mu  =  mu3d[iptr];
      slw = slw3d[iptr];
      lam2mu = lam + 2.0 * mu;

      Vx_ptr = Vx + iptr;
      Vy_ptr = Vy + iptr;
      Vz_ptr = Vz + iptr;
      Txx_ptr = Txx + iptr;
      Tyy_ptr = Tyy + iptr;
      Tzz_ptr = Tzz + iptr;
      Txz_ptr = Txz + iptr;
      Tyz_ptr = Tyz + iptr;
      Txy_ptr = Txy + iptr;

      // xi derivatives
      M_FD_SHIFT_PTR_MACDRP(DxVx,  Vx_ptr,  1, idir);
      M_FD_SHIFT_PTR_MACDRP(DxVy,  Vy_ptr,  1, idir);
      M_FD_SHIFT_PTR_MACDRP(DxVz,  Vz_ptr,  1, idir);
      M_FD_SHIFT_PTR_MACDRP(DxTxx, Txx_ptr, 1, idir);
      M_FD_SHIFT_PTR_MACDRP(DxTyy, Tyy_ptr, 1, idir);
      M_FD_SHIFT_PTR_MACDRP(DxTzz, Tzz_ptr, 1, idir);
      M_FD_SHIFT_PTR_MACDRP(DxTxz, Txz_ptr, 1, idir);
      M_FD_SHIFT_PTR_MACDRP(DxTyz, Tyz_ptr, 1, idir);
      M_FD_SHIFT_PTR_MACDRP(DxTxy, Txy_ptr, 1, idir);

      // combine for corr and aux vars
       hVx_rhs = slw * ( xix*DxTxx + xiy*DxTxy + xiz*DxTxz );
       hVy_rhs = slw * ( xix*DxTxy + xiy*DxTyy + xiz*DxTyz );
       hVz_rhs = slw * ( xix*DxTxz + xiy*DxTyz + xiz*DxTzz );
      hTxx_rhs = lam2mu*xix*DxVx + lam*xiy*DxVy + lam*xiz*DxVz;
      hTyy_rhs = lam*xix*DxVx + lam2mu*xiy*DxVy + lam*xiz*DxVz;
      hTzz_rhs = lam*xix*DxVx + lam*xiy*DxVy + lam2mu*xiz*DxVz;
      hTxy_rhs = mu*( xiy*DxVx + xix*DxVy );
      hTxz_rhs = mu*( xiz*DxVx + xix*DxVz );
      hTyz_rhs = mu*( xiz*DxVy + xiy*DxVz );

      // 1: make corr to moment equation
      hVx[iptr] += coef_B_minus_1 * hVx_rhs - coef_B * pml_Vx[iptr_a];
      hVy[iptr] += coef_B_minus_1 * hVy_rhs - coef_B * pml_Vy[iptr_a];
      hVz[iptr] += coef_B_minus_1 * hVz_rhs - coef_B * pml_Vz[iptr_a];

      // make corr to Hooke's equatoin
      hTxx[iptr] += coef_B_minus_1 * hTxx_rhs - coef_B * pml_Txx[iptr_a];
      hTyy[iptr] += coef_B_minus_1 * hTyy_rhs - coef_B * pml_Tyy[iptr_a];
      hTzz[iptr] += coef_B_minus_1 * hTzz_rhs - coef_B * pml_Tzz[iptr_a];
      hTxz[iptr] += coef_B_minus_1 * hTxz_rhs - coef_B * pml_Txz[iptr_a];
      hTyz[iptr] += coef_B_minus_1 * hTyz_rhs - coef_B * pml_Tyz[iptr_a];
      hTxy[iptr] += coef_B_minus_1 * hTxy_rhs - coef_B * pml_Txy[iptr_a];
      
      // 2: aux var
      //   a1 = alpha + d / beta, dealt in abs_set_cfspml
      pml_hVx[iptr_a]  = coef_D * hVx_rhs  - coef_A * pml_Vx[iptr_a];
      pml_hVy[iptr_a]  = coef_D * hVy_rhs  - coef_A * pml_Vy[iptr_a];
      pml_hVz[iptr_a]  = coef_D * hVz_rhs  - coef_A * pml_Vz[iptr_a];
      pml_hTxx[iptr_a] = coef_D * hTxx_rhs - coef_A * pml_Txx[iptr_a];
      pml_hTyy[iptr_a] = coef_D * hTyy_rhs - coef_A * pml_Tyy[iptr_a];
      pml_hTzz[iptr_a] = coef_D * hTzz_rhs - coef_A * pml_Tzz[iptr_a];
      pml_hTxz[iptr_a] = coef_D * hTxz_rhs - coef_A * pml_Txz[iptr_a];
      pml_hTyz[iptr_a] = coef_D * hTyz_rhs - coef_A * pml_Tyz[iptr_a];
      pml_hTxy[iptr_a] = coef_D * hTxy_rhs - coef_A * pml_Txy[iptr_a];

      // add contributions from free surface condition
      //  not consider timg because conflict with main cfspml,
      //     need to revise in the future if required
      if (bdryfree.is_at_sides[CONST_NDIM-1][1]==1 && (iz+abs_nk1)==nk2)
      {
        // zeta derivatives
        size_t ij = ((ix+abs_ni1) + (iy+abs_nj1) * siz_line)*9;
        Dx_DzVx = matVx2Vz[ij+3*0+0] * DxVx
                + matVx2Vz[ij+3*0+1] * DxVy
                + matVx2Vz[ij+3*0+2] * DxVz;

        Dx_DzVy = matVx2Vz[ij+3*1+0] * DxVx
                + matVx2Vz[ij+3*1+1] * DxVy
                + matVx2Vz[ij+3*1+2] * DxVz;

        Dx_DzVz = matVx2Vz[ij+3*2+0] * DxVx
                + matVx2Vz[ij+3*2+1] * DxVy
                + matVx2Vz[ij+3*2+2] * DxVz;

        // metric
        ztx = zt_x[iptr];
        zty = zt_y[iptr];
        ztz = zt_z[iptr];

        // keep xi derivative terms, including free surface convered
        hTxx_rhs =    lam2mu * (            ztx*Dx_DzVx)
                    + lam    * (            zty*Dx_DzVy
                                +           ztz*Dx_DzVz);

        hTyy_rhs =   lam2mu * (            zty*Dx_DzVy)
                    +lam    * (            ztx*Dx_DzVx
                                          +ztz*Dx_DzVz);

        hTzz_rhs =   lam2mu * (            ztz*Dx_DzVz)
                    +lam    * (            ztx*Dx_DzVx
                                          +zty*Dx_DzVy);

        hTxy_rhs = mu *(
                     zty*Dx_DzVx + ztx*Dx_DzVy
                    );
        hTxz_rhs = mu *(
                     ztz*Dx_DzVx + ztx*Dx_DzVz
                    );
        hTyz_rhs = mu *(
                     ztz*Dx_DzVy + zty*Dx_DzVz
                    );

        // make corr to Hooke's equatoin
        hTxx[iptr] += (coef_B - 1.0) * hTxx_rhs;
        hTyy[iptr] += (coef_B - 1.0) * hTyy_rhs;
        hTzz[iptr] += (coef_B - 1.0) * hTzz_rhs;
        hTxz[iptr] += (coef_B - 1.0) * hTxz_rhs;
        hTyz[iptr] += (coef_B - 1.0) * hTyz_rhs;
        hTxy[iptr] += (coef_B - 1.0) * hTxy_rhs;

        // aux var
        //   a1 = alpha + d / beta, dealt in abs_set_cfspml
        pml_hTxx[iptr_a] += coef_D * hTxx_rhs;
        pml_hTyy[iptr_a] += coef_D * hTyy_rhs;
        pml_hTzz[iptr_a] += coef_D * hTzz_rhs;
        pml_hTxz[iptr_a] += coef_D * hTxz_rhs;
        pml_hTyz[iptr_a] += coef_D * hTyz_rhs;
        pml_hTxy[iptr_a] += coef_D * hTxy_rhs;
      }
    }
  }
  else if (idim == 1) // y direction
  {
    if(ix<abs_ni  && iy<abs_nj && iz<abs_nk)
    {
      iptr_a = iz*(abs_nj*abs_ni) + iy*abs_ni + ix;
      iptr   = (ix + abs_ni1) + (iy+abs_nj1)*siz_line + (iz+abs_nk1) * siz_slice;

      // pml coefs
      // int abs_j = iy;
      coef_D = ptr_coef_D[iy];
      coef_A = ptr_coef_A[iy];
      coef_B = ptr_coef_B[iy];
      coef_B_minus_1 = coef_B - 1.0;

      // metric
      etx = et_x[iptr];
      ety = et_y[iptr];
      etz = et_z[iptr];

      // medium
      lam = lam3d[iptr];
      mu  =  mu3d[iptr];
      slw = slw3d[iptr];
      lam2mu = lam + 2.0 * mu;

      Vx_ptr = Vx + iptr;
      Vy_ptr = Vy + iptr;
      Vz_ptr = Vz + iptr;
      Txx_ptr = Txx + iptr;
      Tyy_ptr = Tyy + iptr;
      Tzz_ptr = Tzz + iptr;
      Txz_ptr = Txz + iptr;
      Tyz_ptr = Tyz + iptr;
      Txy_ptr = Txy + iptr;

      // et derivatives
      M_FD_SHIFT_PTR_MACDRP(DyVx , Vx_ptr,  siz_line, jdir);
      M_FD_SHIFT_PTR_MACDRP(DyVy , Vy_ptr,  siz_line, jdir);
      M_FD_SHIFT_PTR_MACDRP(DyVz , Vz_ptr,  siz_line, jdir);
      M_FD_SHIFT_PTR_MACDRP(DyTxx, Txx_ptr, siz_line, jdir);
      M_FD_SHIFT_PTR_MACDRP(DyTyy, Tyy_ptr, siz_line, jdir);
      M_FD_SHIFT_PTR_MACDRP(DyTzz, Tzz_ptr, siz_line, jdir);
      M_FD_SHIFT_PTR_MACDRP(DyTxz, Txz_ptr, siz_line, jdir);
      M_FD_SHIFT_PTR_MACDRP(DyTyz, Tyz_ptr, siz_line, jdir);
      M_FD_SHIFT_PTR_MACDRP(DyTxy, Txy_ptr, siz_line, jdir);

      // combine for corr and aux vars
       hVx_rhs = slw * ( etx*DyTxx + ety*DyTxy + etz*DyTxz );
       hVy_rhs = slw * ( etx*DyTxy + ety*DyTyy + etz*DyTyz );
       hVz_rhs = slw * ( etx*DyTxz + ety*DyTyz + etz*DyTzz );
      hTxx_rhs = lam2mu*etx*DyVx + lam*ety*DyVy + lam*etz*DyVz;
      hTyy_rhs = lam*etx*DyVx + lam2mu*ety*DyVy + lam*etz*DyVz;
      hTzz_rhs = lam*etx*DyVx + lam*ety*DyVy + lam2mu*etz*DyVz;
      hTxy_rhs = mu*( ety*DyVx + etx*DyVy );
      hTxz_rhs = mu*( etz*DyVx + etx*DyVz );
      hTyz_rhs = mu*( etz*DyVy + ety*DyVz );

      // 1: make corr to moment equation
      hVx[iptr] += coef_B_minus_1 * hVx_rhs - coef_B * pml_Vx[iptr_a];
      hVy[iptr] += coef_B_minus_1 * hVy_rhs - coef_B * pml_Vy[iptr_a];
      hVz[iptr] += coef_B_minus_1 * hVz_rhs - coef_B * pml_Vz[iptr_a];

      // make corr to Hooke's equatoin
      hTxx[iptr] += coef_B_minus_1 * hTxx_rhs - coef_B * pml_Txx[iptr_a];
      hTyy[iptr] += coef_B_minus_1 * hTyy_rhs - coef_B * pml_Tyy[iptr_a];
      hTzz[iptr] += coef_B_minus_1 * hTzz_rhs - coef_B * pml_Tzz[iptr_a];
      hTxz[iptr] += coef_B_minus_1 * hTxz_rhs - coef_B * pml_Txz[iptr_a];
      hTyz[iptr] += coef_B_minus_1 * hTyz_rhs - coef_B * pml_Tyz[iptr_a];
      hTxy[iptr] += coef_B_minus_1 * hTxy_rhs - coef_B * pml_Txy[iptr_a];
      
      // 2: aux var
      //   a1 = alpha + d / beta, dealt in abs_set_cfspml
      pml_hVx[iptr_a]  = coef_D * hVx_rhs  - coef_A * pml_Vx[iptr_a];
      pml_hVy[iptr_a]  = coef_D * hVy_rhs  - coef_A * pml_Vy[iptr_a];
      pml_hVz[iptr_a]  = coef_D * hVz_rhs  - coef_A * pml_Vz[iptr_a];
      pml_hTxx[iptr_a] = coef_D * hTxx_rhs - coef_A * pml_Txx[iptr_a];
      pml_hTyy[iptr_a] = coef_D * hTyy_rhs - coef_A * pml_Tyy[iptr_a];
      pml_hTzz[iptr_a] = coef_D * hTzz_rhs - coef_A * pml_Tzz[iptr_a];
      pml_hTxz[iptr_a] = coef_D * hTxz_rhs - coef_A * pml_Txz[iptr_a];
      pml_hTyz[iptr_a] = coef_D * hTyz_rhs - coef_A * pml_Tyz[iptr_a];
      pml_hTxy[iptr_a] = coef_D * hTxy_rhs - coef_A * pml_Txy[iptr_a];

      // add contributions from free surface condition
      if (bdryfree.is_at_sides[CONST_NDIM-1][1]==1 && (iz+abs_nk1)==nk2)
      {
        // zeta derivatives
        size_t ij = ((ix+abs_ni1) + (iy+abs_nj1) * siz_line)*9;
        Dy_DzVx = matVy2Vz[ij+3*0+0] * DyVx
                + matVy2Vz[ij+3*0+1] * DyVy
                + matVy2Vz[ij+3*0+2] * DyVz;

        Dy_DzVy = matVy2Vz[ij+3*1+0] * DyVx
                + matVy2Vz[ij+3*1+1] * DyVy
                + matVy2Vz[ij+3*1+2] * DyVz;

        Dy_DzVz = matVy2Vz[ij+3*2+0] * DyVx
                + matVy2Vz[ij+3*2+1] * DyVy
                + matVy2Vz[ij+3*2+2] * DyVz;

        // metric
        ztx = zt_x[iptr];
        zty = zt_y[iptr];
        ztz = zt_z[iptr];

        hTxx_rhs =    lam2mu * (             ztx*Dy_DzVx)
                    + lam    * (             zty*Dy_DzVy
                                            +ztz*Dy_DzVz);

        hTyy_rhs =   lam2mu * (             zty*Dy_DzVy)
                    +lam    * (             ztx*Dy_DzVx
                                           +ztz*Dy_DzVz);

        hTzz_rhs =   lam2mu * (             ztz*Dy_DzVz)
                    +lam    * (             ztx*Dy_DzVx
                                           +zty*Dy_DzVy);

        hTxy_rhs = mu *(
                     zty*Dy_DzVx + ztx*Dy_DzVy
                    );
        hTxz_rhs = mu *(
                     ztz*Dy_DzVx + ztx*Dy_DzVz
                    );
        hTyz_rhs = mu *(
                     ztz*Dy_DzVy + zty*Dy_DzVz
                  );

        // make corr to Hooke's equatoin
        hTxx[iptr] += (coef_B - 1.0) * hTxx_rhs;
        hTyy[iptr] += (coef_B - 1.0) * hTyy_rhs;
        hTzz[iptr] += (coef_B - 1.0) * hTzz_rhs;
        hTxz[iptr] += (coef_B - 1.0) * hTxz_rhs;
        hTyz[iptr] += (coef_B - 1.0) * hTyz_rhs;
        hTxy[iptr] += (coef_B - 1.0) * hTxy_rhs;

        // aux var
        //   a1 = alpha + d / beta, dealt in abs_set_cfspml
        pml_hTxx[iptr_a] += coef_D * hTxx_rhs;
        pml_hTyy[iptr_a] += coef_D * hTyy_rhs;
        pml_hTzz[iptr_a] += coef_D * hTzz_rhs;
        pml_hTxz[iptr_a] += coef_D * hTxz_rhs;
        pml_hTyz[iptr_a] += coef_D * hTyz_rhs;
        pml_hTxy[iptr_a] += coef_D * hTxy_rhs;
      }
    }
  }
  else // z direction
  {
    if(ix<abs_ni  && iy<abs_nj && iz<abs_nk)
    {
      iptr_a = iz*(abs_nj*abs_ni) + iy*abs_ni + ix;
      iptr   = (ix + abs_ni1) + (iy+abs_nj1) * siz_line + (iz+abs_nk1) * siz_slice;
      // pml coefs
      // int abs_k = iz;
      coef_D = ptr_coef_D[iz];
      coef_A = ptr_coef_A[iz];
      coef_B = ptr_coef_B[iz];
      coef_B_minus_1 = coef_B - 1.0;

      // metric
      ztx = zt_x[iptr];
      zty = zt_y[iptr];
      ztz = zt_z[iptr];

      // medium
      lam = lam3d[iptr];
      mu  =  mu3d[iptr];
      slw = slw3d[iptr];
      lam2mu = lam + 2.0 * mu;

      Vx_ptr = Vx + iptr;
      Vy_ptr = Vy + iptr;
      Vz_ptr = Vz + iptr;
      Txx_ptr = Txx + iptr;
      Tyy_ptr = Tyy + iptr;
      Tzz_ptr = Tzz + iptr;
      Txz_ptr = Txz + iptr;
      Tyz_ptr = Tyz + iptr;
      Txy_ptr = Txy + iptr;
      
      // zt derivatives
      M_FD_SHIFT_PTR_MACDRP(DzVx , Vx_ptr,  siz_slice, kdir);
      M_FD_SHIFT_PTR_MACDRP(DzVy , Vy_ptr,  siz_slice, kdir);
      M_FD_SHIFT_PTR_MACDRP(DzVz , Vz_ptr,  siz_slice, kdir);
      M_FD_SHIFT_PTR_MACDRP(DzTxx, Txx_ptr, siz_slice, kdir);
      M_FD_SHIFT_PTR_MACDRP(DzTyy, Tyy_ptr, siz_slice, kdir);
      M_FD_SHIFT_PTR_MACDRP(DzTzz, Tzz_ptr, siz_slice, kdir);
      M_FD_SHIFT_PTR_MACDRP(DzTxz, Txz_ptr, siz_slice, kdir);
      M_FD_SHIFT_PTR_MACDRP(DzTyz, Tyz_ptr, siz_slice, kdir);
      M_FD_SHIFT_PTR_MACDRP(DzTxy, Txy_ptr, siz_slice, kdir);

      // combine for corr and aux vars
       hVx_rhs = slw * ( ztx*DzTxx + zty*DzTxy + ztz*DzTxz );
       hVy_rhs = slw * ( ztx*DzTxy + zty*DzTyy + ztz*DzTyz );
       hVz_rhs = slw * ( ztx*DzTxz + zty*DzTyz + ztz*DzTzz );
      hTxx_rhs = lam2mu*ztx*DzVx + lam*zty*DzVy + lam*ztz*DzVz;
      hTyy_rhs = lam*ztx*DzVx + lam2mu*zty*DzVy + lam*ztz*DzVz;
      hTzz_rhs = lam*ztx*DzVx + lam*zty*DzVy + lam2mu*ztz*DzVz;
      hTxy_rhs = mu*( zty*DzVx + ztx*DzVy );
      hTxz_rhs = mu*( ztz*DzVx + ztx*DzVz );
      hTyz_rhs = mu*( ztz*DzVy + zty*DzVz );

      // 1: make corr to moment equation
      hVx[iptr] += coef_B_minus_1 * hVx_rhs - coef_B * pml_Vx[iptr_a];
      hVy[iptr] += coef_B_minus_1 * hVy_rhs - coef_B * pml_Vy[iptr_a];
      hVz[iptr] += coef_B_minus_1 * hVz_rhs - coef_B * pml_Vz[iptr_a];

      // make corr to Hooke's equatoin
      hTxx[iptr] += coef_B_minus_1 * hTxx_rhs - coef_B * pml_Txx[iptr_a];
      hTyy[iptr] += coef_B_minus_1 * hTyy_rhs - coef_B * pml_Tyy[iptr_a];
      hTzz[iptr] += coef_B_minus_1 * hTzz_rhs - coef_B * pml_Tzz[iptr_a];
      hTxz[iptr] += coef_B_minus_1 * hTxz_rhs - coef_B * pml_Txz[iptr_a];
      hTyz[iptr] += coef_B_minus_1 * hTyz_rhs - coef_B * pml_Tyz[iptr_a];
      hTxy[iptr] += coef_B_minus_1 * hTxy_rhs - coef_B * pml_Txy[iptr_a];
      
      // 2: aux var
      //   a1 = alpha + d / beta, dealt in abs_set_cfspml
      pml_hVx[iptr_a]  = coef_D * hVx_rhs  - coef_A * pml_Vx[iptr_a];
      pml_hVy[iptr_a]  = coef_D * hVy_rhs  - coef_A * pml_Vy[iptr_a];
      pml_hVz[iptr_a]  = coef_D * hVz_rhs  - coef_A * pml_Vz[iptr_a];
      pml_hTxx[iptr_a] = coef_D * hTxx_rhs - coef_A * pml_Txx[iptr_a];
      pml_hTyy[iptr_a] = coef_D * hTyy_rhs - coef_A * pml_Tyy[iptr_a];
      pml_hTzz[iptr_a] = coef_D * hTzz_rhs - coef_A * pml_Tzz[iptr_a];
      pml_hTxz[iptr_a] = coef_D * hTxz_rhs - coef_A * pml_Txz[iptr_a];
      pml_hTyz[iptr_a] = coef_D * hTyz_rhs - coef_A * pml_Tyz[iptr_a];
      pml_hTxy[iptr_a] = coef_D * hTxy_rhs - coef_A * pml_Txy[iptr_a];
    }
  } // if which dim
}

/*******************************************************************************
 * free surface coef
 ******************************************************************************/

__global__ void
sv_eq1st_curv_col_el_iso_dvh2dvz_gpu(gdinfo_t        gdinfo_d,
                                     gdcurv_metric_t metric_d,
                                     md_t       md_d,
                                     bdryfree_t      bdryfree_d,
                                     const int verbose)
{
  int ni1 = gdinfo_d.ni1;
  int ni2 = gdinfo_d.ni2;
  int nj1 = gdinfo_d.nj1;
  int nj2 = gdinfo_d.nj2;
  int nk1 = gdinfo_d.nk1;
  int nk2 = gdinfo_d.nk2;
  int nx  = gdinfo_d.nx;
  int ny  = gdinfo_d.ny;
  int nz  = gdinfo_d.nz;
  size_t siz_line   = gdinfo_d.siz_line;
  size_t siz_slice  = gdinfo_d.siz_slice;
  size_t siz_volume = gdinfo_d.siz_volume;

  // point to each var
  float * xi_x = metric_d.xi_x;
  float * xi_y = metric_d.xi_y;
  float * xi_z = metric_d.xi_z;
  float * et_x = metric_d.eta_x;
  float * et_y = metric_d.eta_y;
  float * et_z = metric_d.eta_z;
  float * zt_x = metric_d.zeta_x;
  float * zt_y = metric_d.zeta_y;
  float * zt_z = metric_d.zeta_z;

  float * lam3d = md_d.lambda;
  float *  mu3d = md_d.mu;

  float *matVx2Vz = bdryfree_d.matVx2Vz2;
  float *matVy2Vz = bdryfree_d.matVy2Vz2;
  
  float A[3][3], B[3][3], C[3][3];
  float AB[3][3], AC[3][3];

  float e11, e12, e13, e21, e22, e23, e31, e32, e33;
  float lam2mu, lam, mu;
 
  int k = nk2;
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  if(ix<(ni2-ni1+1) && iy<(nj2-nj1+1))
  {
    size_t iptr = (ix+ni1) + (iy+nj1) * siz_line + k * siz_slice;
    e11 = xi_x[iptr];
    e12 = xi_y[iptr];
    e13 = xi_z[iptr];
    e21 = et_x[iptr];
    e22 = et_y[iptr];
    e23 = et_z[iptr];
    e31 = zt_x[iptr];
    e32 = zt_y[iptr];
    e33 = zt_z[iptr];

    lam    = lam3d[iptr];
    mu     =  mu3d[iptr];
    lam2mu = lam + 2.0f * mu;

    // first dim: irow; sec dim: jcol, as Fortran code
    A[0][0] = lam2mu*e31*e31 + mu*(e32*e32+e33*e33);
    A[0][1] = lam*e31*e32 + mu*e32*e31;
    A[0][2] = lam*e31*e33 + mu*e33*e31;
    A[1][0] = lam*e32*e31 + mu*e31*e32;
    A[1][1] = lam2mu*e32*e32 + mu*(e31*e31+e33*e33);
    A[1][2] = lam*e32*e33 + mu*e33*e32;
    A[2][0] = lam*e33*e31 + mu*e31*e33;
    A[2][1] = lam*e33*e32 + mu*e32*e33;
    A[2][2] = lam2mu*e33*e33 + mu*(e31*e31+e32*e32);
    fdlib_math_invert3x3(A);

    B[0][0] = -lam2mu*e31*e11 - mu*(e32*e12+e33*e13);
    B[0][1] = -lam*e31*e12 - mu*e32*e11;
    B[0][2] = -lam*e31*e13 - mu*e33*e11;
    B[1][0] = -lam*e32*e11 - mu*e31*e12;
    B[1][1] = -lam2mu*e32*e12 - mu*(e31*e11+e33*e13);
    B[1][2] = -lam*e32*e13 - mu*e33*e12;
    B[2][0] = -lam*e33*e11 - mu*e31*e13;
    B[2][1] = -lam*e33*e12 - mu*e32*e13;
    B[2][2] = -lam2mu*e33*e13 - mu*(e31*e11+e32*e12);

    C[0][0] = -lam2mu*e31*e21 - mu*(e32*e22+e33*e23);
    C[0][1] = -lam*e31*e22 - mu*e32*e21;
    C[0][2] = -lam*e31*e23 - mu*e33*e21;
    C[1][0] = -lam*e32*e21 - mu*e31*e22;
    C[1][1] = -lam2mu*e32*e22 - mu*(e31*e21+e33*e23);
    C[1][2] = -lam*e32*e23 - mu*e33*e22;
    C[2][0] = -lam*e33*e21 - mu*e31*e23;
    C[2][1] = -lam*e33*e22 - mu*e32*e23;
    C[2][2] = -lam2mu*e33*e23 - mu*(e31*e21+e32*e22);

    fdlib_math_matmul3x3(A, B, AB);
    fdlib_math_matmul3x3(A, C, AC);

    size_t ij = ((iy+nj1) * siz_line + (ix+ni1)) * 9;

    // save into mat
    for(int irow = 0; irow < 3; irow++){
      for(int jcol = 0; jcol < 3; jcol++){
        matVx2Vz[ij + irow*3 + jcol] = AB[irow][jcol];
        matVy2Vz[ij + irow*3 + jcol] = AC[irow][jcol];
      }
    }
  }

  return;
}
