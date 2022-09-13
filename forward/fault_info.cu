#include <stdio.h>
#include <stdlib.h>

#include "gd_info.h"
#include "fault_info.h"
#include "fdlib_math.h"

int
fault_coef_init(fault_coef_t *FC,
                gdinfo_t *gdinfo)
{
  int ny = gdinfo->ny;
  int nz = gdinfo->nz;

  FC->rho_f = (float *) malloc(sizeof(float)*ny*nz*2);
  FC->mu_f  = (float *) malloc(sizeof(float)*ny*nz*2);
  FC->lam_f = (float *) malloc(sizeof(float)*ny*nz*2);

  FC->D21_1 = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->D22_1 = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->D23_1 = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->D31_1 = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->D32_1 = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->D33_1 = (float *) malloc(sizeof(float)*ny*nz*3*3);

  FC->D21_2 = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->D22_2 = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->D23_2 = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->D31_2 = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->D32_2 = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->D33_2 = (float *) malloc(sizeof(float)*ny*nz*3*3);

  FC->matMin2Plus1 = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->matMin2Plus2 = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->matMin2Plus3 = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->matMin2Plus4 = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->matMin2Plus5 = (float *) malloc(sizeof(float)*ny*nz*3*3);

  FC->matPlus2Min1 = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->matPlus2Min2 = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->matPlus2Min3 = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->matPlus2Min4 = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->matPlus2Min5 = (float *) malloc(sizeof(float)*ny*nz*3*3);

  FC->matT1toVx_Min  = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->matVytoVx_Min  = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->matVztoVx_Min  = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->matT1toVx_Plus = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->matVytoVx_Plus = (float *) malloc(sizeof(float)*ny*nz*3*3);
  FC->matVztoVx_Plus = (float *) malloc(sizeof(float)*ny*nz*3*3);

  FC->vec_n  = (float *) malloc(sizeof(float)*ny*nz*3);
  FC->vec_s1 = (float *) malloc(sizeof(float)*ny*nz*3);
  FC->vec_s2 = (float *) malloc(sizeof(float)*ny*nz*3);
  FC->x_et   = (float *) malloc(sizeof(float)*ny*nz);
  FC->y_et   = (float *) malloc(sizeof(float)*ny*nz);
  FC->z_et   = (float *) malloc(sizeof(float)*ny*nz);

  // fault with free surface coef
  // NOTE: even this thread without free surface, still malloc 
  // this is because easy coding. this coef not used, if without free surface
   
  FC->matVx2Vz1     = (float *) malloc(sizeof(float)*ny*3*3);
  FC->matVy2Vz1     = (float *) malloc(sizeof(float)*ny*3*3);
  FC->matVx2Vz2     = (float *) malloc(sizeof(float)*ny*3*3);
  FC->matVy2Vz2     = (float *) malloc(sizeof(float)*ny*3*3);
  FC->matPlus2Min1f = (float *) malloc(sizeof(float)*ny*3*3);
  FC->matPlus2Min2f = (float *) malloc(sizeof(float)*ny*3*3);
  FC->matPlus2Min3f = (float *) malloc(sizeof(float)*ny*3*3);
  FC->matMin2Plus1f = (float *) malloc(sizeof(float)*ny*3*3);
  FC->matMin2Plus2f = (float *) malloc(sizeof(float)*ny*3*3);
  FC->matMin2Plus3f = (float *) malloc(sizeof(float)*ny*3*3);

  FC->matT1toVxf_Min  = (float *) malloc(sizeof(float)*ny*3*3);
  FC->matVytoVxf_Min  = (float *) malloc(sizeof(float)*ny*3*3);
  FC->matT1toVxf_Plus = (float *) malloc(sizeof(float)*ny*3*3);
  FC->matVytoVxf_Plus = (float *) malloc(sizeof(float)*ny*3*3);

  return 0;
}


int 
fault_coef_cal(gdinfo_t *gdinfo, 
               gdcurv_metric_t *metric, 
               md_t *md,
               int fault_i_global_index,
               fault_coef_t *FC)
{
  int ny = gdinfo->ny;
  int nz = gdinfo->nz;
  size_t siz_line = gdinfo->siz_line;
  size_t siz_slice = gdinfo->siz_slice;
  size_t siz_slice_yz = gdinfo->siz_slice_yz;
  // x direction only has 1 mpi. 
  int npoint_z = gdinfo->npoint_z;
  int gnk1 = gdinfo->gnk1;
  int i0 = fault_i_global_index + 3; //fault plane x index with ghost
  size_t iptr, iptr_f;
  float rho, mu, lam, lam2mu, jac;
  // point to each var
  float *jac3d = metric->jac;
  float *xi_x  = metric->xi_x;
  float *xi_y  = metric->xi_y;
  float *xi_z  = metric->xi_z;
  float *et_x  = metric->eta_x;
  float *et_y  = metric->eta_y;
  float *et_z  = metric->eta_z;
  float *zt_x  = metric->zeta_x;
  float *zt_y  = metric->zeta_y;
  float *zt_z  = metric->zeta_z;
  float *lam3d = md->lambda;
  float *mu3d  = md->mu;
  float *rho3d = md->rho;
  float e11, e12, e13, e21, e22, e23, e31, e32, e33;
  // Matrix form used by inversion and multiply. 
  // Convenient calculation
  float D11_1[3][3], D12_1[3][3], D13_1[3][3];
  float D11_2[3][3], D12_2[3][3], D13_2[3][3];
  // temp  matrix
  float mat1[3][3], mat2[3][3], mat3[3][3], mat4[3][3];
  float vec_n[3], vec_s1[3], vec_s2[3];

  // for free surface
  float A[3][3], B[3][3], C[3][3];
  float matVx2Vz1[3][3], matVy2Vz1[3][3];
  float matVx2Vz2[3][3], matVy2Vz2[3][3];
  float matVx1_free[3][3], matVy1_free[3][3];
  float matVx2_free[3][3], matVy2_free[3][3];
  float matPlus2Min1f[3][3], matPlus2Min2f[3][3], matPlus2Min3f[3][3];
  float matMin2Plus1f[3][3], matMin2Plus2f[3][3], matMin2Plus3f[3][3];
  // don't need strike constraint g1_2, g2_2, g3_2.
  // D11_12 D12_12, D13_12.  article has problem
  float g1_2, g2_2, g3_2; 
  float D11_12[3][3], D12_12[3][3], D13_12[3][3];
  float h1_3, h2_3, h3_3;
  float D11_13[3][3], D12_13[3][3], D13_13[3][3];
  float D11_1f[3][3], D12_1f[3][3], D13_1f[3][3];
  float D11_2f[3][3], D12_2f[3][3], D13_2f[3][3];
  
  float norm, Tovert[3][3];
  for(int k=0; k<nz; k++) 
  {
    for(int j=0; j<ny; j++) 
    {
      iptr = i0 + j * siz_line + k * siz_slice;  
      iptr_f = j + k * ny;
      rho = rho3d[iptr];
      mu  = mu3d[iptr];
      lam = lam3d[iptr];
      lam2mu = lam + 2.0*mu;
      jac = jac3d[iptr];
      e11 = xi_x[iptr];
      e12 = xi_y[iptr];
      e13 = xi_z[iptr];
      e21 = et_x[iptr];
      e22 = et_y[iptr];
      e23 = et_z[iptr];
      e31 = zt_x[iptr];
      e32 = zt_y[iptr];
      e33 = zt_z[iptr];
      //mimus media
      FC->rho_f[iptr_f+0*siz_slice_yz] = rho;
      FC->lam_f[iptr_f+0*siz_slice_yz] = lam;
      FC->mu_f [iptr_f+0*siz_slice_yz] = mu;
      //plus media
      FC->rho_f[iptr_f+1*siz_slice_yz] = rho;
      FC->lam_f[iptr_f+1*siz_slice_yz] = lam;
      FC->mu_f [iptr_f+1*siz_slice_yz] = mu;
      // minus -
      D11_1[0][0] = lam2mu*e11*e11+mu*(e12*e12+e13*e13);
      D11_1[0][1] = lam*e11*e12+mu*e12*e11;
      D11_1[0][2] = lam*e11*e13+mu*e13*e11;
      D11_1[1][0] = mu*e11*e12+lam*e12*e11;
      D11_1[1][1] = lam2mu*e12*e12+mu*(e11*e11+e13*e13);
      D11_1[1][2] = lam*e12*e13+mu*e13*e12;
      D11_1[2][0] = mu*e11*e13+lam*e13*e11;
      D11_1[2][1] = mu*e12*e13+lam*e13*e12;
      D11_1[2][2] = lam2mu*e13*e13+mu*(e11*e11+e12*e12);

      D12_1[0][0] = lam2mu*e11*e21+mu*(e12*e22+e13*e23);
      D12_1[0][1] = lam*e11*e22+mu*e12*e21;
      D12_1[0][2] = lam*e11*e23+mu*e13*e21;
      D12_1[1][0] = mu*e11*e22+lam*e12*e21;
      D12_1[1][1] = lam2mu*e12*e22+mu*(e11*e21+e13*e23);
      D12_1[1][2] = lam*e12*e23+mu*e13*e22;
      D12_1[2][0] = mu*e11*e23+lam*e13*e21;
      D12_1[2][1] = mu*e12*e23+lam*e13*e22;
      D12_1[2][2] = lam2mu*e13*e23+mu*(e11*e21+e12*e22);

      D13_1[0][0] = lam2mu*e11*e31+mu*(e12*e32+e13*e33);
      D13_1[0][1] = lam*e11*e32+mu*e12*e31;
      D13_1[0][2] = lam*e11*e33+mu*e13*e31;
      D13_1[1][0] = mu*e11*e32+lam*e12*e31;
      D13_1[1][1] = lam2mu*e12*e32+mu*(e11*e31+e13*e33);
      D13_1[1][2] = lam*e12*e33+mu*e13*e32;
      D13_1[2][0] = mu*e11*e33+lam*e13*e31;
      D13_1[2][1] = mu*e12*e33+lam*e13*e32;
      D13_1[2][2] = lam2mu*e13*e33+mu*(e11*e31+e12*e32);

      FC->D21_1[iptr_f*9+0] = lam2mu*e21*e11+mu*(e22*e12+e23*e13);
      FC->D21_1[iptr_f*9+1] = lam*e21*e12+mu*e22*e11;
      FC->D21_1[iptr_f*9+2] = lam*e21*e13+mu*e23*e11;
      FC->D21_1[iptr_f*9+3] = mu*e21*e12+lam*e22*e11;
      FC->D21_1[iptr_f*9+4] = lam2mu*e22*e12+mu*(e21*e11+e23*e13);
      FC->D21_1[iptr_f*9+5] = lam*e22*e13+mu*e23*e12;
      FC->D21_1[iptr_f*9+6] = mu*e21*e13+lam*e23*e11;
      FC->D21_1[iptr_f*9+7] = mu*e22*e13+lam*e23*e12;
      FC->D21_1[iptr_f*9+8] = lam2mu*e23*e13+mu*(e21*e11+e22*e12);

      FC->D22_1[iptr_f*9+0] = lam2mu*e21*e21+mu*(e22*e22+e23*e23);
      FC->D22_1[iptr_f*9+1] = lam*e21*e22+mu*e22*e21;
      FC->D22_1[iptr_f*9+2] = lam*e21*e23+mu*e23*e21;
      FC->D22_1[iptr_f*9+3] = mu*e21*e22+lam*e22*e21;
      FC->D22_1[iptr_f*9+4] = lam2mu*e22*e22+mu*(e21*e21+e23*e23);
      FC->D22_1[iptr_f*9+5] = lam*e22*e23+mu*e23*e22;
      FC->D22_1[iptr_f*9+6] = mu*e21*e23+lam*e23*e21;
      FC->D22_1[iptr_f*9+7] = mu*e22*e23+lam*e23*e22;
      FC->D22_1[iptr_f*9+8] = lam2mu*e23*e23+mu*(e21*e21+e22*e22);

      FC->D23_1[iptr_f*9+0] = lam2mu*e21*e31+mu*(e22*e32+e23*e33);
      FC->D23_1[iptr_f*9+1] = lam*e21*e32+mu*e22*e31;
      FC->D23_1[iptr_f*9+2] = lam*e21*e33+mu*e23*e31;
      FC->D23_1[iptr_f*9+3] = mu*e21*e32+lam*e22*e31;
      FC->D23_1[iptr_f*9+4] = lam2mu*e22*e32+mu*(e21*e31+e23*e33);
      FC->D23_1[iptr_f*9+5] = lam*e22*e33+mu*e23*e32;
      FC->D23_1[iptr_f*9+6] = mu*e21*e33+lam*e23*e31;
      FC->D23_1[iptr_f*9+7] = mu*e22*e33+lam*e23*e32;
      FC->D23_1[iptr_f*9+8] = lam2mu*e23*e33+mu*(e21*e31+e22*e32);

      FC->D31_1[iptr_f*9+0] = lam2mu*e31*e11+mu*(e32*e12+e33*e13);
      FC->D31_1[iptr_f*9+1] = lam*e31*e12+mu*e32*e11;
      FC->D31_1[iptr_f*9+2] = lam*e31*e13+mu*e33*e11;
      FC->D31_1[iptr_f*9+3] = mu*e31*e12+lam*e32*e11;
      FC->D31_1[iptr_f*9+4] = lam2mu*e32*e12+mu*(e31*e11+e33*e13);
      FC->D31_1[iptr_f*9+5] = lam*e32*e13+mu*e33*e12;
      FC->D31_1[iptr_f*9+6] = mu*e31*e13+lam*e33*e11;
      FC->D31_1[iptr_f*9+7] = mu*e32*e13+lam*e33*e12;
      FC->D31_1[iptr_f*9+8] = lam2mu*e33*e13+mu*(e31*e11+e32*e12);

      FC->D32_1[iptr_f*9+0] = lam2mu*e31*e21+mu*(e32*e22+e33*e23);
      FC->D32_1[iptr_f*9+1] = lam*e31*e22+mu*e32*e21;
      FC->D32_1[iptr_f*9+2] = lam*e31*e23+mu*e33*e21;
      FC->D32_1[iptr_f*9+3] = mu*e31*e22+lam*e32*e21;
      FC->D32_1[iptr_f*9+4] = lam2mu*e32*e22+mu*(e31*e21+e33*e23);
      FC->D32_1[iptr_f*9+5] = lam*e32*e23+mu*e33*e22;
      FC->D32_1[iptr_f*9+6] = mu*e31*e23+lam*e33*e21;
      FC->D32_1[iptr_f*9+7] = mu*e32*e23+lam*e33*e22;
      FC->D32_1[iptr_f*9+8] = lam2mu*e33*e23+mu*(e31*e21+e32*e22);

      FC->D33_1[iptr_f*9+0] = lam2mu*e31*e31+mu*(e32*e32+e33*e33);
      FC->D33_1[iptr_f*9+1] = lam*e31*e32+mu*e32*e31;
      FC->D33_1[iptr_f*9+2] = lam*e31*e33+mu*e33*e31;
      FC->D33_1[iptr_f*9+3] = mu*e31*e32+lam*e32*e31;
      FC->D33_1[iptr_f*9+4] = lam2mu*e32*e32+mu*(e31*e31+e33*e33);
      FC->D33_1[iptr_f*9+5] = lam*e32*e33+mu*e33*e32;
      FC->D33_1[iptr_f*9+6] = mu*e31*e33+lam*e33*e31;
      FC->D33_1[iptr_f*9+7] = mu*e32*e33+lam*e33*e32;
      FC->D33_1[iptr_f*9+8] = lam2mu*e33*e33+mu*(e31*e31+e32*e32);

      //->plus +
      D11_2[0][0] = lam2mu*e11*e11+mu*(e12*e12+e13*e13);
      D11_2[0][1] = lam*e11*e12+mu*e12*e11;
      D11_2[0][2] = lam*e11*e13+mu*e13*e11;
      D11_2[1][0] = mu*e11*e12+lam*e12*e11;
      D11_2[1][1] = lam2mu*e12*e12+mu*(e11*e11+e13*e13);
      D11_2[1][2] = lam*e12*e13+mu*e13*e12;
      D11_2[2][0] = mu*e11*e13+lam*e13*e11;
      D11_2[2][1] = mu*e12*e13+lam*e13*e12;
      D11_2[2][2] = lam2mu*e13*e13+mu*(e11*e11+e12*e12);

      D12_2[0][0] = lam2mu*e11*e21+mu*(e12*e22+e13*e23);
      D12_2[0][1] = lam*e11*e22+mu*e12*e21;
      D12_2[0][2] = lam*e11*e23+mu*e13*e21;
      D12_2[1][0] = mu*e11*e22+lam*e12*e21;
      D12_2[1][1] = lam2mu*e12*e22+mu*(e11*e21+e13*e23);
      D12_2[1][2] = lam*e12*e23+mu*e13*e22;
      D12_2[2][0] = mu*e11*e23+lam*e13*e21;
      D12_2[2][1] = mu*e12*e23+lam*e13*e22;
      D12_2[2][2] = lam2mu*e13*e23+mu*(e11*e21+e12*e22);

      D13_2[0][0] = lam2mu*e11*e31+mu*(e12*e32+e13*e33);
      D13_2[0][1] = lam*e11*e32+mu*e12*e31;
      D13_2[0][2] = lam*e11*e33+mu*e13*e31;
      D13_2[1][0] = mu*e11*e32+lam*e12*e31;
      D13_2[1][1] = lam2mu*e12*e32+mu*(e11*e31+e13*e33);
      D13_2[1][2] = lam*e12*e33+mu*e13*e32;
      D13_2[2][0] = mu*e11*e33+lam*e13*e31;
      D13_2[2][1] = mu*e12*e33+lam*e13*e32;
      D13_2[2][2] = lam2mu*e13*e33+mu*(e11*e31+e12*e32);

      FC->D21_2[iptr_f*9+0] = lam2mu*e21*e11+mu*(e22*e12+e23*e13);
      FC->D21_2[iptr_f*9+1] = lam*e21*e12+mu*e22*e11;
      FC->D21_2[iptr_f*9+2] = lam*e21*e13+mu*e23*e11;
      FC->D21_2[iptr_f*9+3] = mu*e21*e12+lam*e22*e11;
      FC->D21_2[iptr_f*9+4] = lam2mu*e22*e12+mu*(e21*e11+e23*e13);
      FC->D21_2[iptr_f*9+5] = lam*e22*e13+mu*e23*e12;
      FC->D21_2[iptr_f*9+6] = mu*e21*e13+lam*e23*e11;
      FC->D21_2[iptr_f*9+7] = mu*e22*e13+lam*e23*e12;
      FC->D21_2[iptr_f*9+8] = lam2mu*e23*e13+mu*(e21*e11+e22*e12);

      FC->D22_2[iptr_f*9+0] = lam2mu*e21*e21+mu*(e22*e22+e23*e23);
      FC->D22_2[iptr_f*9+1] = lam*e21*e22+mu*e22*e21;
      FC->D22_2[iptr_f*9+2] = lam*e21*e23+mu*e23*e21;
      FC->D22_2[iptr_f*9+3] = mu*e21*e22+lam*e22*e21;
      FC->D22_2[iptr_f*9+4] = lam2mu*e22*e22+mu*(e21*e21+e23*e23);
      FC->D22_2[iptr_f*9+5] = lam*e22*e23+mu*e23*e22;
      FC->D22_2[iptr_f*9+6] = mu*e21*e23+lam*e23*e21;
      FC->D22_2[iptr_f*9+7] = mu*e22*e23+lam*e23*e22;
      FC->D22_2[iptr_f*9+8] = lam2mu*e23*e23+mu*(e21*e21+e22*e22);

      FC->D23_2[iptr_f*9+0] = lam2mu*e21*e31+mu*(e22*e32+e23*e33);
      FC->D23_2[iptr_f*9+1] = lam*e21*e32+mu*e22*e31;
      FC->D23_2[iptr_f*9+2] = lam*e21*e33+mu*e23*e31;
      FC->D23_2[iptr_f*9+3] = mu*e21*e32+lam*e22*e31;
      FC->D23_2[iptr_f*9+4] = lam2mu*e22*e32+mu*(e21*e31+e23*e33);
      FC->D23_2[iptr_f*9+5] = lam*e22*e33+mu*e23*e32;
      FC->D23_2[iptr_f*9+6] = mu*e21*e33+lam*e23*e31;
      FC->D23_2[iptr_f*9+7] = mu*e22*e33+lam*e23*e32;
      FC->D23_2[iptr_f*9+8] = lam2mu*e23*e33+mu*(e21*e31+e22*e32);

      FC->D31_2[iptr_f*9+0] = lam2mu*e31*e11+mu*(e32*e12+e33*e13);
      FC->D31_2[iptr_f*9+1] = lam*e31*e12+mu*e32*e11;
      FC->D31_2[iptr_f*9+2] = lam*e31*e13+mu*e33*e11;
      FC->D31_2[iptr_f*9+3] = mu*e31*e12+lam*e32*e11;
      FC->D31_2[iptr_f*9+4] = lam2mu*e32*e12+mu*(e31*e11+e33*e13);
      FC->D31_2[iptr_f*9+5] = lam*e32*e13+mu*e33*e12;
      FC->D31_2[iptr_f*9+6] = mu*e31*e13+lam*e33*e11;
      FC->D31_2[iptr_f*9+7] = mu*e32*e13+lam*e33*e12;
      FC->D31_2[iptr_f*9+8] = lam2mu*e33*e13+mu*(e31*e11+e32*e12);

      FC->D32_2[iptr_f*9+0] = lam2mu*e31*e21+mu*(e32*e22+e33*e23);
      FC->D32_2[iptr_f*9+1] = lam*e31*e22+mu*e32*e21;
      FC->D32_2[iptr_f*9+2] = lam*e31*e23+mu*e33*e21;
      FC->D32_2[iptr_f*9+3] = mu*e31*e22+lam*e32*e21;
      FC->D32_2[iptr_f*9+4] = lam2mu*e32*e22+mu*(e31*e21+e33*e23);
      FC->D32_2[iptr_f*9+5] = lam*e32*e23+mu*e33*e22;
      FC->D32_2[iptr_f*9+6] = mu*e31*e23+lam*e33*e21;
      FC->D32_2[iptr_f*9+7] = mu*e32*e23+lam*e33*e22;
      FC->D32_2[iptr_f*9+8] = lam2mu*e33*e23+mu*(e31*e21+e32*e22);

      FC->D33_2[iptr_f*9+0] = lam2mu*e31*e31+mu*(e32*e32+e33*e33);
      FC->D33_2[iptr_f*9+1] = lam*e31*e32+mu*e32*e31;
      FC->D33_2[iptr_f*9+2] = lam*e31*e33+mu*e33*e31;
      FC->D33_2[iptr_f*9+3] = mu*e31*e32+lam*e32*e31;
      FC->D33_2[iptr_f*9+4] = lam2mu*e32*e32+mu*(e31*e31+e33*e33);
      FC->D33_2[iptr_f*9+5] = lam*e32*e33+mu*e33*e32;
      FC->D33_2[iptr_f*9+6] = mu*e31*e33+lam*e33*e31;
      FC->D33_2[iptr_f*9+7] = mu*e32*e33+lam*e33*e32;
      FC->D33_2[iptr_f*9+8] = lam2mu*e33*e33+mu*(e31*e31+e32*e32);


      for (int ii = 0; ii < 3; ii++) 
      {
        for (int jj = 0; jj < 3; jj++) 
        {
          int ij = 3*ii+jj;
          D11_1[ii][jj] *= jac;
          D12_1[ii][jj] *= jac;
          D13_1[ii][jj] *= jac;
          FC->D21_1[iptr_f*9+ij] *= jac;
          FC->D22_1[iptr_f*9+ij] *= jac;
          FC->D23_1[iptr_f*9+ij] *= jac;
          FC->D31_1[iptr_f*9+ij] *= jac;
          FC->D32_1[iptr_f*9+ij] *= jac;
          FC->D33_1[iptr_f*9+ij] *= jac;
          D11_2[ii][jj] *= jac;
          D12_2[ii][jj] *= jac;
          D13_2[ii][jj] *= jac;
          FC->D21_2[iptr_f*9+ij] *= jac;
          FC->D22_2[iptr_f*9+ij] *= jac;
          FC->D23_2[iptr_f*9+ij] *= jac;
          FC->D31_2[iptr_f*9+ij] *= jac;
          FC->D32_2[iptr_f*9+ij] *= jac;
          FC->D33_2[iptr_f*9+ij] *= jac;
        }                           
      }                             
    
      // NOTE two method calculate coef
      // method 1 by zhang zhenguo
      // method 2 by zheng wenqiang
      // K11*DxV+K12*DyV+K13*DzV=DtT1

      // method 1 coef 
      // minus to plus
      // invert(D11_2) * D 
      // D = D11_1, D12_1, D13_1, D12_2, D13_2
      for (int ii = 0; ii < 3; ii++) 
      {
        for (int jj = 0; jj < 3; jj++) 
        {
          mat1[ii][jj] = D11_2[ii][jj];
        }
      }
      fdlib_math_invert3x3(mat1);

      // invert(D11_2) * D11_1 
      fdlib_math_matmul3x3(mat1,D11_1,mat2);
      for (int ii = 0; ii < 3; ii++)
      {
        for (int jj = 0; jj < 3; jj++)
        {
          int ij = 3*ii+jj;
          FC->matMin2Plus1[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      // invert(D11_2) * D12_1 
      fdlib_math_matmul3x3(mat1,D12_1,mat2);
      for (int ii = 0; ii < 3; ii++)
      {
        for (int jj = 0; jj < 3; jj++)
        {
          int ij = 3*ii+jj;
          FC->matMin2Plus2[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      // invert(D11_2) * D13_1 
      fdlib_math_matmul3x3(mat1,D13_1,mat2);
      for (int ii = 0; ii < 3; ii++)
      {
        for (int jj = 0; jj < 3; jj++)
        {
          int ij = 3*ii+jj;
          FC->matMin2Plus3[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      // invert(D11_2) * D12_2
      fdlib_math_matmul3x3(mat1,D12_2,mat2);
      for (int ii = 0; ii < 3; ii++)
      {
        for (int jj = 0; jj < 3; jj++)
        {
          int ij = 3*ii+jj;
          FC->matMin2Plus4[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      // invert(D11_2) * D13_2
      fdlib_math_matmul3x3(mat1,D13_2,mat2);
      for (int ii = 0; ii < 3; ii++)
      {
        for (int jj = 0; jj < 3; jj++)
        {
          int ij = 3*ii+jj;
          FC->matMin2Plus5[iptr_f*9+ij] = mat2[ii][jj];
        }
      }

      // plus -> min
      // invert(D11_1) * D
      // D = D11_2, D12_2, D13_2, D12_1, D13_1
      for (int ii = 0; ii < 3; ii++) 
      {
        for (int jj = 0; jj < 3; jj++) 
        {
          mat1[ii][jj] = D11_1[ii][jj];
        }
      }
      fdlib_math_invert3x3(mat1);
      // invert(D11_1) * D11_2 
      fdlib_math_matmul3x3(mat1,D11_2,mat2);
      for (int ii = 0; ii < 3; ii++)
      {
        for (int jj = 0; jj < 3; jj++)
        {
          int ij = 3*ii+jj;
          FC->matPlus2Min1[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      // invert(D11_1) * D12_2
      fdlib_math_matmul3x3(mat1,D12_2,mat2);
      for (int ii = 0; ii < 3; ii++)
      {
        for (int jj = 0; jj < 3; jj++)
        {
          int ij = 3*ii+jj;
          FC->matPlus2Min2[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      // invert(D11_1) * D13_2 
      fdlib_math_matmul3x3(mat1,D13_2,mat2);
      for (int ii = 0; ii < 3; ii++)
      {
        for (int jj = 0; jj < 3; jj++)
        {
          int ij = 3*ii+jj;
          FC->matPlus2Min3[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      // invert(D11_1) * D12_1 
      fdlib_math_matmul3x3(mat1,D12_1,mat2);
      for (int ii = 0; ii < 3; ii++)
      {
        for (int jj = 0; jj < 3; jj++)
        {
          int ij = 3*ii+jj;
          FC->matPlus2Min4[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      // invert(D11_1) * D13_1
      fdlib_math_matmul3x3(mat1,D13_1,mat2);
      for (int ii = 0; ii < 3; ii++)
      {
        for (int jj = 0; jj < 3; jj++)
        {
          int ij = 3*ii+jj;
          FC->matPlus2Min5[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      // method 2 coef 
      // minus
      // T1 Vy Vz -> Vx
      // inversion D11_1 
      for (int ii = 0; ii < 3; ii++) 
      {
        for (int jj = 0; jj < 3; jj++) 
        {
          mat1[ii][jj] = D11_1[ii][jj];
        }
      }
      // invert(D11_1) 
      fdlib_math_invert3x3(mat1);
      for (int ii = 0; ii < 3; ii++)
      {
        for (int jj = 0; jj < 3; jj++)
        {
          int ij = 3*ii+jj;
          FC->matT1toVx_Min[iptr_f*9+ij] = mat1[ii][jj];
        }
      }
      // invert(D11_1) * D12_1 
      fdlib_math_matmul3x3(mat1,D12_1,mat2);
      for (int ii = 0; ii < 3; ii++)
      {
        for (int jj = 0; jj < 3; jj++)
        {
          int ij = 3*ii+jj;
          FC->matVytoVx_Min[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      // invert(D11_1) * D13_1 
      fdlib_math_matmul3x3(mat1,D13_1,mat2);
      for (int ii = 0; ii < 3; ii++)
      {
        for (int jj = 0; jj < 3; jj++)
        {
          int ij = 3*ii+jj;
          FC->matVztoVx_Min[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      // Plus side
      // T1 Vy Vz -> Vx
      for (int ii = 0; ii < 3; ii++) 
      {
        for (int jj = 0; jj < 3; jj++) 
        {
          mat1[ii][jj] = D11_2[ii][jj];
        }
      }
      // invert(D11_2) 
      fdlib_math_invert3x3(mat1);
      for (int ii = 0; ii < 3; ii++) 
      {
        for (int jj = 0; jj < 3; jj++) 
        {
          int ij = 3*ii+jj;
          FC->matT1toVx_Plus[iptr_f*9+ij] = mat1[ii][jj];
        }
      }
      // invert(D11_2) * D12_2 
      fdlib_math_matmul3x3(mat1,D12_2,mat2);
      for (int ii = 0; ii < 3; ii++) 
      {
        for (int jj = 0; jj < 3; jj++) 
        {
          int ij = 3*ii+jj;
          FC->matVytoVx_Plus[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      // invert(D11_2) * D13_2 
      fdlib_math_matmul3x3(mat1,D13_2,mat2);
      for (int ii = 0; ii < 3; ii++) 
      {
        for (int jj = 0; jj < 3; jj++) 
        {
          int ij = 3*ii+jj;
          FC->matVztoVx_Plus[iptr_f*9+ij] = mat2[ii][jj];
        }
      }

      mat3[0][0] = e11;
      mat3[0][1] = e12;
      mat3[0][2] = e13;
      mat3[1][0] = e21;
      mat3[1][1] = e22;
      mat3[1][2] = e23;
      mat3[2][0] = e31;
      mat3[2][1] = e32;
      mat3[2][2] = e33;
      fdlib_math_invert3x3(mat3);

      // iptr_f = j + k * ny;
      // strike
      FC->x_et[iptr_f] = mat3[0][1];
      FC->y_et[iptr_f] = mat3[1][1];
      FC->z_et[iptr_f] = mat3[2][1];

      vec_s1[0] = mat3[0][1];
      vec_s1[1] = mat3[1][1];
      vec_s1[2] = mat3[2][1];

      norm = fdlib_math_norm3(vec_s1);

      for (int i=0; i<3; i++)
      {
        vec_s1[i] /= norm;
      }
      // normal
      vec_n[0] = e11;
      vec_n[1] = e12;
      vec_n[2] = e13;
      norm = fdlib_math_norm3(vec_n);

      for (int i=0; i<3; i++)
      {
        vec_n[i] /= norm;
      }

      fdlib_math_cross_product(vec_n, vec_s1, vec_s2);

      for (int i=0; i<3; i++)
      {
          FC->vec_n [iptr_f*3+i] = vec_n [i];
          FC->vec_s1[iptr_f*3+i] = vec_s1[i];
          FC->vec_s2[iptr_f*3+i] = vec_s2[i];
      }

      if ((k-3+gnk1) == npoint_z-1) // free surface global index, index start 0
      {
        for (int ii = 0; ii < 3; ii++) {
          for (int jj = 0; jj < 3; jj++) {
            int ij = 3*ii+jj; 
            A[ii][jj] =  FC->D33_1[iptr_f*9+ij];
            B[ii][jj] = -FC->D31_1[iptr_f*9+ij];
            C[ii][jj] = -FC->D32_1[iptr_f*9+ij];
          }
        }
        fdlib_math_invert3x3(A);
        fdlib_math_matmul3x3(A, B, matVx2Vz1);
        fdlib_math_matmul3x3(A, C, matVy2Vz1);

        for (int ii = 0; ii < 3; ii++) {
          for (int jj = 0; jj < 3; jj++) {
            int ij = 3*ii+jj; 
            A[ii][jj] =  FC->D33_2[iptr_f*9+ij];
            B[ii][jj] = -FC->D31_2[iptr_f*9+ij];
            C[ii][jj] = -FC->D32_2[iptr_f*9+ij];
          }
        }
        fdlib_math_invert3x3(A);
        fdlib_math_matmul3x3(A, B, matVx2Vz2);
        fdlib_math_matmul3x3(A, C, matVy2Vz2);


        vec_n[0] = e11;
        vec_n[1] = e12;
        vec_n[2] = e13;

        vec_s1[0] = FC->x_et[iptr_f];
        vec_s1[1] = FC->y_et[iptr_f];
        vec_s1[2] = FC->z_et[iptr_f];

        fdlib_math_cross_product(vec_n, vec_s1, vec_s2);
        norm = fdlib_math_norm3(vec_s2);
        for (int i=0; i<3; i++)
        {
            vec_s2[i] /= norm;
        }

        g1_2 = 0.0;
        g2_2 = 0.0;
        g3_2 = 0.0; 

        h1_3 = vec_s2[0]*e11
             + vec_s2[1]*e12
             + vec_s2[2]*e13;
        h2_3 = vec_s2[0]*e21
             + vec_s2[1]*e22
             + vec_s2[2]*e23;
        h3_3 = vec_s2[0]*e31
             + vec_s2[1]*e32
             + vec_s2[2]*e33;

        norm = fdlib_math_norm3(vec_n);
        h1_3 = h1_3/norm;
        h2_3 = h2_3/norm;
        h3_3 = h3_3/norm;

        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            Tovert[ii][jj] = vec_s1[ii] * vec_n[jj];
            D11_12[ii][jj] = mu * g1_2 * Tovert[ii][jj];
            D12_12[ii][jj] = mu * g2_2 * Tovert[ii][jj];
            D13_12[ii][jj] = mu * g3_2 * Tovert[ii][jj];
          }
        }

        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            Tovert[ii][jj] = vec_s2[ii] * vec_n[jj];
            D11_13[ii][jj] = mu * h1_3 * Tovert[ii][jj];
            D12_13[ii][jj] = mu * h2_3 * Tovert[ii][jj];
            D13_13[ii][jj] = mu * h3_3 * Tovert[ii][jj];
            D11_1f[ii][jj] = D11_1[ii][jj] + D11_12[ii][jj] + D11_13[ii][jj];
            D12_1f[ii][jj] = D12_1[ii][jj] + D12_12[ii][jj] + D12_13[ii][jj];
            D13_1f[ii][jj] = D13_1[ii][jj] + D13_12[ii][jj] + D13_13[ii][jj];
          }
        }

        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            Tovert[ii][jj] = vec_s1[ii] * vec_n[jj];
            D11_12[ii][jj] = mu * g1_2 * Tovert[ii][jj];
            D12_12[ii][jj] = mu * g2_2 * Tovert[ii][jj];
            D13_12[ii][jj] = mu * g3_2 * Tovert[ii][jj];
          }
        }
        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            Tovert[ii][jj] = vec_s2[ii] * vec_n[jj];
            D11_13[ii][jj] = mu * h1_3 * Tovert[ii][jj];
            D12_13[ii][jj] = mu * h2_3 * Tovert[ii][jj];
            D13_13[ii][jj] = mu * h3_3 * Tovert[ii][jj];
            D11_2f[ii][jj] = D11_2[ii][jj] + D11_12[ii][jj] + D11_13[ii][jj];
            D12_2f[ii][jj] = D12_2[ii][jj] + D12_12[ii][jj] + D12_13[ii][jj];
            D13_2f[ii][jj] = D13_2[ii][jj] + D13_12[ii][jj] + D13_13[ii][jj];
          }
        }

        fdlib_math_matmul3x3(D13_1f, matVx2Vz1, mat1);
        fdlib_math_matmul3x3(D13_1f, matVy2Vz1, mat2);
        fdlib_math_matmul3x3(D13_2f, matVx2Vz2, mat3);
        fdlib_math_matmul3x3(D13_2f, matVy2Vz2, mat4);

        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            matVx1_free[ii][jj] = D11_1f[ii][jj] + mat1[ii][jj];
            matVy1_free[ii][jj] = D12_1f[ii][jj] + mat2[ii][jj];
            matVx2_free[ii][jj] = D11_2f[ii][jj] + mat3[ii][jj];
            matVy2_free[ii][jj] = D12_2f[ii][jj] + mat4[ii][jj];
          }
        }

        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            Tovert[ii][jj] = matVx1_free[ii][jj];
          }
        }
        fdlib_math_invert3x3(Tovert);
        fdlib_math_matmul3x3(Tovert, matVx2_free, matPlus2Min1f);
        fdlib_math_matmul3x3(Tovert, matVy2_free, matPlus2Min2f);
        fdlib_math_matmul3x3(Tovert, matVy1_free, matPlus2Min3f);
        // method 2 coef
        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            int ij = 3*ii+jj;
            FC->matT1toVxf_Min[j*9+ij] = Tovert[ii][jj];
            FC->matVytoVxf_Min[j*9+ij] = matPlus2Min3f[ii][jj];
          }
        }

        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            Tovert[ii][jj] = matVx2_free[ii][jj];
          }
        }
        fdlib_math_invert3x3(Tovert);
        fdlib_math_matmul3x3(Tovert, matVx1_free, matMin2Plus1f);
        fdlib_math_matmul3x3(Tovert, matVy1_free, matMin2Plus2f);
        fdlib_math_matmul3x3(Tovert, matVy2_free, matMin2Plus3f);

        // method 2 coef
        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            int ij = 3*ii+jj;
            FC->matT1toVxf_Plus[j*9+ij] = Tovert[ii][jj];
            FC->matVytoVxf_Plus[j*9+ij] = matMin2Plus3f[ii][jj];
          }
        }

        // save
        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            int ij = 3*ii+jj;
            FC->matVx2Vz1    [j*9+ij] = matVx2Vz1    [ii][jj];
            FC->matVy2Vz1    [j*9+ij] = matVy2Vz1    [ii][jj];
            FC->matVx2Vz2    [j*9+ij] = matVx2Vz2    [ii][jj];
            FC->matVy2Vz2    [j*9+ij] = matVy2Vz2    [ii][jj];
            FC->matPlus2Min1f[j*9+ij] = matPlus2Min1f[ii][jj];
            FC->matPlus2Min2f[j*9+ij] = matPlus2Min2f[ii][jj];
            FC->matPlus2Min3f[j*9+ij] = matPlus2Min3f[ii][jj];
            FC->matMin2Plus1f[j*9+ij] = matMin2Plus1f[ii][jj];
            FC->matMin2Plus2f[j*9+ij] = matMin2Plus2f[ii][jj];
            FC->matMin2Plus3f[j*9+ij] = matMin2Plus3f[ii][jj];
          }
        }
      }
    }                               
  }
  return 0;
}  

int
fault_init(fault_t *F,
           gdinfo_t *gdinfo)
{
  int nj = gdinfo->nj;
  int nk = gdinfo->nk;

  // for input
  F->T0x  = (float *) malloc(sizeof(float)*nj*nk);  // stress_init_x
  F->T0y  = (float *) malloc(sizeof(float)*nj*nk);  // stress_init_y
  F->T0z  = (float *) malloc(sizeof(float)*nj*nk);  // stress_init_z
  F->mu_s = (float *) malloc(sizeof(float)*nj*nk);
  F->mu_d = (float *) malloc(sizeof(float)*nj*nk);
  F->Dc   = (float *) malloc(sizeof(float)*nj*nk);
  F->C0   = (float *) malloc(sizeof(float)*nj*nk);
  // for output
  F->Tn      = (float *) malloc(sizeof(float)*nj*nk);
  F->Ts1     = (float *) malloc(sizeof(float)*nj*nk);
  F->Ts2     = (float *) malloc(sizeof(float)*nj*nk);
  F->slip    = (float *) malloc(sizeof(float)*nj*nk);
  F->slip1   = (float *) malloc(sizeof(float)*nj*nk); 
  F->slip2   = (float *) malloc(sizeof(float)*nj*nk);  
  F->Vs      = (float *) malloc(sizeof(float)*nj*nk);
  F->Vs1     = (float *) malloc(sizeof(float)*nj*nk);
  F->Vs2     = (float *) malloc(sizeof(float)*nj*nk);
  F->peak_Vs = (float *) malloc(sizeof(float)*nj*nk);
  F->init_t0 = (float *) malloc(sizeof(float)*nj*nk);
  // for inner
  F->tTn          = (float *) malloc(sizeof(float)*nj*nk);
  F->tTs1         = (float *) malloc(sizeof(float)*nj*nk);
  F->tTs2         = (float *) malloc(sizeof(float)*nj*nk);
  F->united       = (int *) malloc(sizeof(int)*nj*nk);
  F->faultgrid    = (int *) malloc(sizeof(int)*nj*nk);
  F->rup_index_y  = (int *) malloc(sizeof(int)*nj*nk);
  F->rup_index_z  = (int *) malloc(sizeof(int)*nj*nk);
  F->flag_rup     = (int *) malloc(sizeof(int)*nj*nk);
  F->init_t0_flag = (int *) malloc(sizeof(int)*nj*nk);

  memset(F->init_t0_flag, 0, sizeof(int)  *nj*nk);
  memset(F->slip,         0, sizeof(float)*nj*nk);
  memset(F->slip1,        0, sizeof(float)*nj*nk); 
  memset(F->slip2,        0, sizeof(float)*nj*nk); 
  memset(F->Vs1,          0, sizeof(float)*nj*nk);
  memset(F->Vs2,          0, sizeof(float)*nj*nk);
  memset(F->peak_Vs,      0, sizeof(float)*nj*nk);

  return 0;
}

int
fault_set(fault_t *F,
          fault_coef_t *FC,
          gdinfo_t *gdinfo,
          int bdry_has_free,
          int *fault_grid,
          char *init_stress_nc)
{
  int nj1 = gdinfo->nj1;
  int nk1 = gdinfo->nk1;
  int nj2 = gdinfo->nj2;
  int nk2 = gdinfo->nk2;
  int nj = gdinfo->nj;
  int nk = gdinfo->nk;
  int ny = gdinfo->ny;
  int gnj1 = gdinfo->gnj1;
  int gnk1 = gdinfo->gnk1;
  int npoint_y = gdinfo->npoint_y;
  int npoint_z = gdinfo->npoint_z;
  int gj, gk;
  size_t iptr_t, iptr_f; 
  float vec_n[3], vec_s1[3], vec_s2[3];

  nc_read_init_stress(F, gdinfo,init_stress_nc);

  for (int k=0; k<nk; k++)
  {
    for (int j=0; j<nj; j++)
    {
      iptr_t = j + k * nj;
      iptr_f = (j+nj1) + (k+nk1) * ny; //with ghost

      vec_n [0] = FC->vec_n [iptr_f*3 + 0];
      vec_n [1] = FC->vec_n [iptr_f*3 + 1];
      vec_n [2] = FC->vec_n [iptr_f*3 + 2];
      vec_s1[0] = FC->vec_s1[iptr_f*3 + 0];
      vec_s1[1] = FC->vec_s1[iptr_f*3 + 1];
      vec_s1[2] = FC->vec_s1[iptr_f*3 + 2];
      vec_s2[0] = FC->vec_s2[iptr_f*3 + 0];
      vec_s2[1] = FC->vec_s2[iptr_f*3 + 1];
      vec_s2[2] = FC->vec_s2[iptr_f*3 + 2];

      // transform init stress to local coordinate, not nessessary,
      // only for output 
      F->Tn [iptr_t] = F->T0x[iptr_t] * vec_n[0]
                     + F->T0y[iptr_t] * vec_n[1]
                     + F->T0z[iptr_t] * vec_n[2];
      F->Ts1[iptr_t] = F->T0x[iptr_t] * vec_s1[0]
                     + F->T0y[iptr_t] * vec_s1[1]
                     + F->T0z[iptr_t] * vec_s1[2];
      F->Ts2[iptr_t] = F->T0x[iptr_t] * vec_s2[0]
                     + F->T0y[iptr_t] * vec_s2[1]
                     + F->T0z[iptr_t] * vec_s2[2];

      gj = gnj1 + j;
      gk = gnk1 + k;
      // fault grid read from json, index start 1.
      // so gj need +1, C index from 0
      // rup_index = 0 means points out of fault.
      // NOTE: boundry 3 points out of fault, due to use different fd stencil
      if( gj+1 <= fault_grid[0]+3 || gj+1 >= fault_grid[1]-3 ){
        F->rup_index_y[iptr_t] = 0;
      }else{
        F->rup_index_y[iptr_t] = 1;
      }

      if( gk+1 <= fault_grid[2]+3 || gk+1 >= fault_grid[3]-3 ){
        F->rup_index_z[iptr_t] = 0;
      }else{
        F->rup_index_z[iptr_t] = 1;
      }
      
      // united is used for pml, pml with fault easy write code
      // united == 1, means contain pml and strong boundry 
      // usually unilateral pml < 30 and strong boundry >= 50
      if(bdry_has_free == 1) 
      {
        if( gj+1 > 30 && gj+1 <= npoint_y - 30 && gk+1 > 30) {
          F->united[iptr_t] = 0;
        }else{
          F->united[iptr_t] = 1;
        }
      } 
      else if(bdry_has_free == 0)
      {
        if( gj+1 > 30 && gj+1 <= npoint_y - 30 && gk+1 > 30 
            && gk+1 <= npoint_z - 30) {
          F->united[iptr_t] = 0;
        }else{
          F->united[iptr_t] = 1;
        }
      }
      if( gj+1 >= fault_grid[0]+3 && gj+1 <= fault_grid[1]-3 &&
          gk+1 >= fault_grid[2]+3 && gk+1 <= fault_grid[3]-3 ) {
        F->faultgrid[iptr_t] = 1;
      }else{
        F->faultgrid[iptr_t] = 0;
      }

      F->init_t0[iptr_t] = -9999.9;
      F->flag_rup[iptr_t] = 0;
    }
  }
  return 0;
}

int 
nc_read_init_stress(fault_t *F, 
                    gdinfo_t *gdinfo,
                    char *init_stress_nc)
{
  int nj = gdinfo->nj;
  int nk = gdinfo->nk;
  int gnj1 = gdinfo->gnj1;
  int gnk1 = gdinfo->gnk1;
  int ierr;
  int ncid;
  int varid;
  size_t start[] = {gnk1, gnj1};
  size_t count[] = {nk, nj}; // y vary first

  ierr = nc_open(init_stress_nc, NC_NOWRITE, &ncid); handle_nc_err(ierr);

  ierr = nc_inq_varid(ncid, "Tx", &varid); handle_nc_err(ierr);
  ierr = nc_get_vara_float(ncid, varid, start, count, F->T0x); handle_nc_err(ierr);

  ierr = nc_inq_varid(ncid, "Ty", &varid); handle_nc_err(ierr);
  ierr = nc_get_vara_float(ncid, varid, start, count, F->T0y); handle_nc_err(ierr);

  ierr = nc_inq_varid(ncid, "Tz", &varid); handle_nc_err(ierr);
  ierr = nc_get_vara_float(ncid, varid, start, count, F->T0z); handle_nc_err(ierr); 

  ierr = nc_inq_varid(ncid, "mu_s", &varid); handle_nc_err(ierr);
  ierr = nc_get_vara_float(ncid, varid, start, count, F->mu_s); handle_nc_err(ierr);

  ierr = nc_inq_varid(ncid, "mu_d", &varid); handle_nc_err(ierr);
  ierr = nc_get_vara_float(ncid, varid, start, count, F->mu_d); handle_nc_err(ierr);

  ierr = nc_inq_varid(ncid, "Dc", &varid); handle_nc_err(ierr);
  ierr = nc_get_vara_float(ncid, varid, start, count, F->Dc); handle_nc_err(ierr);

  ierr = nc_inq_varid(ncid, "C0", &varid); handle_nc_err(ierr);
  ierr = nc_get_vara_float(ncid, varid, start, count, F->C0); handle_nc_err(ierr);

  ierr = nc_close(ncid);handle_nc_err(ierr);
  
  return 0;
}
