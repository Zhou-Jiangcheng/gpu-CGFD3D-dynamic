#include <stdio.h>
#include <stdlib.h>




void 
init_fault_coef(gdinfo_t *gdinfo, 
                gdcurv_metric_t *metric, 
                md_t *md, 
                fault_coef_t *FC)
{
  int nx = gdinfo->nx;
  int ny = gdinfo->ny;
  int nz = gdinfo->nz;
  size_t siz_line = gdinfo->siz_line;
  size_t siz_slice = gdinfo->siz_slice;
  // x direction only has 1 mpi. npoint_x = nx
  int npoint_x = gdinfo->npoint_x;
  int npoint_y = gdinfo->npoint_y;
  int npoint_z = gdinfo->npoint_z;
  int i0 = npoint_x/2;
  size_t iptr, iptr_f;
  float rho, mu, lam, lam2mu;
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
  float  *mu3d = md->mu;
  float *rho3d = md->rho;
  float e11, e12, e13, e21, e22, e23, e31, e32, e33;
  float jac;
  // Matrix form used by inversion and multiply. 
  // Convenient calculation
  float D11_1[3][3], D12_1[3][3], D13_1[3][3];
  float D21_1[3][3], D22_1[3][3], D23_1[3][3];
  float D31_1[3][3], D32_1[3][3], D33_1[3][3];

  float D11_2[3][3], D12_2[3][3], D13_2[3][3];
  float D21_2[3][3], D22_2[3][3], D23_2[3][3];
  float D31_2[3][3], D32_2[3][3], D33_2[3][3];
  // temp  matrix
  float mat1[3][3], mat2[3][3], mat3[3][3], mat4[3][3];

  /* for free surface */
  float A[3][3], B[3][3], C[3][3];
  float matVx2Vz1[3][3], matVy2Vz1[3][3];
  float matVx2Vz2[3][3], matVy2Vz2[3][3];
  float matVx1_free[3][3], matVy1_free[3][3];
  float matVx2_free[3][3], matVy2_free[3][3];
  float matPlus2Min1f[3][3], matPlus2Min2f[3][3], matPlus2Min3f[3][3];
  float matMin2Plus1f[3][3], matMin2Plus2f[3][3], matMin2Plus3f[3][3];
  // don't need strike constraint g1_2, g2_2, g3_2. article has problem
  //float g1_2, g2_2, g3_2; 
  //float D11_12[3][3], D12_12[3][3], D13_12[3][3];
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
      iptr_f = (j + k * ny);
      mu = mu3d[iptr];
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
      // minus -
      FC->D11_1[iptr_f*9+0] = lam2mu*e11*e11+mu*(e12*e12+e13*e13);
      FC->D11_1[iptr_f*9+1] = lam*e11*e12+mu*e12*e11;
      FC->D11_1[iptr_f*9+2] = lam*e11*e13+mu*e13*e11;
      FC->D11_1[iptr_f*9+3] = mu*e11*e12+lam*e12*e11;
      FC->D11_1[iptr_f*9+4] = lam2mu*e12*e12+mu*(e11*e11+e13*e13);
      FC->D11_1[iptr_f*9+5] = lam*e12*e13+mu*e13*e12;
      FC->D11_1[iptr_f*9+6] = mu*e11*e13+lam*e13*e11;
      FC->D11_1[iptr_f*9+7] = mu*e12*e13+lam*e13*e12;
      FC->D11_1[iptr_f*9+8] = lam2mu*e13*e13+mu*(e11*e11+e12*e12);

      FC->D12_1[iptr_f*9+0] = lam2mu*e11*e21+mu*(e12*e22+e13*e23);
      FC->D12_1[iptr_f*9+1] = lam*e11*e22+mu*e12*e21;
      FC->D12_1[iptr_f*9+2] = lam*e11*e23+mu*e13*e21;
      FC->D12_1[iptr_f*9+3] = mu*e11*e22+lam*e12*e21;
      FC->D12_1[iptr_f*9+4] = lam2mu*e12*e22+mu*(e11*e21+e13*e23);
      FC->D12_1[iptr_f*9+5] = lam*e12*e23+mu*e13*e22;
      FC->D12_1[iptr_f*9+6] = mu*e11*e23+lam*e13*e21;
      FC->D12_1[iptr_f*9+7] = mu*e12*e23+lam*e13*e22;
      FC->D12_1[iptr_f*9+8] = lam2mu*e13*e23+mu*(e11*e21+e12*e22);

      FC->D13_1[iptr_f*9+0] = lam2mu*e11*e31+mu*(e12*e32+e13*e33);
      FC->D13_1[iptr_f*9+1] = lam*e11*e32+mu*e12*e31;
      FC->D13_1[iptr_f*9+2] = lam*e11*e33+mu*e13*e31;
      FC->D13_1[iptr_f*9+3] = mu*e11*e32+lam*e12*e31;
      FC->D13_1[iptr_f*9+4] = lam2mu*e12*e32+mu*(e11*e31+e13*e33);
      FC->D13_1[iptr_f*9+5] = lam*e12*e33+mu*e13*e32;
      FC->D13_1[iptr_f*9+6] = mu*e11*e33+lam*e13*e31;
      FC->D13_1[iptr_f*9+7] = mu*e12*e33+lam*e13*e32;
      FC->D13_1[iptr_f*9+8] = lam2mu*e13*e33+mu*(e11*e31+e12*e32);

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
      FC->D11_2[iptr_f*9+0] = lam2mu*e11*e11+mu*(e12*e12+e13*e13);
      FC->D11_2[iptr_f*9+1] = lam*e11*e12+mu*e12*e11;
      FC->D11_2[iptr_f*9+2] = lam*e11*e13+mu*e13*e11;
      FC->D11_2[iptr_f*9+3] = mu*e11*e12+lam*e12*e11;
      FC->D11_2[iptr_f*9+4] = lam2mu*e12*e12+mu*(e11*e11+e13*e13);
      FC->D11_2[iptr_f*9+5] = lam*e12*e13+mu*e13*e12;
      FC->D11_2[iptr_f*9+6] = mu*e11*e13+lam*e13*e11;
      FC->D11_2[iptr_f*9+7] = mu*e12*e13+lam*e13*e12;
      FC->D11_2[iptr_f*9+8] = lam2mu*e13*e13+mu*(e11*e11+e12*e12);

      FC->D12_2[iptr_f*9+0] = lam2mu*e11*e21+mu*(e12*e22+e13*e23);
      FC->D12_2[iptr_f*9+1] = lam*e11*e22+mu*e12*e21;
      FC->D12_2[iptr_f*9+2] = lam*e11*e23+mu*e13*e21;
      FC->D12_2[iptr_f*9+3] = mu*e11*e22+lam*e12*e21;
      FC->D12_2[iptr_f*9+4] = lam2mu*e12*e22+mu*(e11*e21+e13*e23);
      FC->D12_2[iptr_f*9+5] = lam*e12*e23+mu*e13*e22;
      FC->D12_2[iptr_f*9+6] = mu*e11*e23+lam*e13*e21;
      FC->D12_2[iptr_f*9+7] = mu*e12*e23+lam*e13*e22;
      FC->D12_2[iptr_f*9+8] = lam2mu*e13*e23+mu*(e11*e21+e12*e22);

      FC->D13_2[iptr_f*9+0] = lam2mu*e11*e31+mu*(e12*e32+e13*e33);
      FC->D13_2[iptr_f*9+1] = lam*e11*e32+mu*e12*e31;
      FC->D13_2[iptr_f*9+2] = lam*e11*e33+mu*e13*e31;
      FC->D13_2[iptr_f*9+3] = mu*e11*e32+lam*e12*e31;
      FC->D13_2[iptr_f*9+4] = lam2mu*e12*e32+mu*(e11*e31+e13*e33);
      FC->D13_2[iptr_f*9+5] = lam*e12*e33+mu*e13*e32;
      FC->D13_2[iptr_f*9+6] = mu*e11*e33+lam*e13*e31;
      FC->D13_2[iptr_f*9+7] = mu*e12*e33+lam*e13*e32;
      FC->D13_2[iptr_f*9+8] = lam2mu*e13*e33+mu*(e11*e31+e12*e32);

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


      for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
          int ij = 3*ii+jj;
          FC->D11_1[iptr_f*9+ij] *= jac;
          FC->D12_1[iptr_f*9+ij] *= jac;
          FC->D13_1[iptr_f*9+ij] *= jac;
          FC->D21_1[iptr_f*9+ij] *= jac;
          FC->D22_1[iptr_f*9+ij] *= jac;
          FC->D23_1[iptr_f*9+ij] *= jac;
          FC->D31_1[iptr_f*9+ij] *= jac;
          FC->D32_1[iptr_f*9+ij] *= jac;
          FC->D33_1[iptr_f*9+ij] *= jac;
          FC->D11_2[iptr_f*9+ij] *= jac;
          FC->D12_2[iptr_f*9+ij] *= jac;
          FC->D13_2[iptr_f*9+ij] *= jac;
          FC->D21_2[iptr_f*9+ij] *= jac;
          FC->D22_2[iptr_f*9+ij] *= jac;
          FC->D23_2[iptr_f*9+ij] *= jac;
          FC->D31_2[iptr_f*9+ij] *= jac;
          FC->D32_2[iptr_f*9+ij] *= jac;
          FC->D33_2[iptr_f*9+ij] *= jac;

          D11_1[ii][jj] = FC->D11_1[iptr_f*9+ij];
          D12_1[ii][jj] = FC->D12_1[iptr_f*9+ij];
          D13_1[ii][jj] = FC->D13_1[iptr_f*9+ij];
          D21_1[ii][jj] = FC->D21_1[iptr_f*9+ij];
          D22_1[ii][jj] = FC->D22_1[iptr_f*9+ij];
          D23_1[ii][jj] = FC->D23_1[iptr_f*9+ij];
          D31_1[ii][jj] = FC->D31_1[iptr_f*9+ij];
          D32_1[ii][jj] = FC->D32_1[iptr_f*9+ij];
          D33_1[ii][jj] = FC->D33_1[iptr_f*9+ij];

          D11_2[ii][jj] = FC->D11_2[iptr_f*9+ij];
          D12_2[ii][jj] = FC->D12_2[iptr_f*9+ij];
          D13_2[ii][jj] = FC->D13_2[iptr_f*9+ij];
          D21_2[ii][jj] = FC->D21_2[iptr_f*9+ij];
          D22_2[ii][jj] = FC->D22_2[iptr_f*9+ij];
          D23_2[ii][jj] = FC->D23_2[iptr_f*9+ij];
          D31_2[ii][jj] = FC->D31_2[iptr_f*9+ij];
          D32_2[ii][jj] = FC->D32_2[iptr_f*9+ij];
          D33_2[ii][jj] = FC->D33_2[iptr_f*9+ij];
        }                           
      }                             
    
      // NOTE two method calculate coef
      // method 1 by zhang zhenguo
      // method 2 by zheng wenqaing
      // K11*DxV+K12*DyV+K13*DzV=DtT1

      // method 2 coef 
      // minus
      // T1 Vy Vz -> Vx
      // inversion D11_1 
      for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
          mat1[ii][jj] = D11_1[ii][jj];
        }
      }
      fdlib_math_invert3x3(mat1);
      for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
          int ij = 3*ii+jj;
          FC->matT1toVx_Min[iptr_f*9+ij] = mat1[ii][jj];
        }
      }
      fdlib_math_matmul3x3(mat1,D12_1,mat2);
      for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
          int ij = 3*ii+jj;
          FC->matVytoVx_Min[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      fdlib_math_matmul3x3(mat1,D13_1,mat2);
      for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
          int ij = 3*ii+jj;
          FC->matVztoVx_Min[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      // Plus side
      // T1 Vy Vz -> Vx
      for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
          mat1[ii][jj] = D11_2[ii][jj];
        }
      }
      fdlib_math_invert3x3(mat1);
      for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
          int ij = 3*ii+jj;
          FC->matT1toVx_Plus[iptr_f*9+ij] = mat1[ii][jj];
        }
      }
      fdlib_math_matmul3x3(mat1,D12_2,mat2);
      for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
          int ij = 3*ii+jj;
          FC->matVytoVx_Plus[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      fdlib_math_matmul3x3(mat1,D13_2,mat2);
      for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
          int ij = 3*ii+jj;
          FC->matVztoVx_Plus[iptr_f*9+ij] = mat2[ii][jj];
        }
      }

      // method 1 coef 
      // minus to plus
      /* invert(D11_2) * D */
      /* D = D11_1, D12_1, D13_1, D12_2, D13_2 */
      for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
          mat1[ii][jj] = D11_2[ii][jj];
        }
      }
      fdlib_math_invert3x3(mat1);

      /* invert(D11_2) * D11_1 */
      fdlib_math_matmul3x3(mat1,D11_1,mat2);
      for (int ii = 0; ii < 3; ++ii){
        for (int jj = 0; jj < 3; ++jj){
          int ij = 3*ii+jj;
          FC->matMin2Plus1[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      /* invert(D11_2) * D12_1 */
      fdlib_math_matmul3x3(mat1,D12_1,mat2);
      for (int ii = 0; ii < 3; ++ii){
        for (int jj = 0; jj < 3; ++jj){
          int ij = 3*ii+jj;
          FC->matMin2Plus2[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      /* invert(D11_2) * D13_1 */
      fdlib_math_matmul3x3(mat1,D13_1,mat2);
      for (int ii = 0; ii < 3; ++ii){
        for (int jj = 0; jj < 3; ++jj){
          int ij = 3*ii+jj;
          FC->matMin2Plus3[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      /* invert(D11_2) * D12_2 */
      fdlib_math_matmul3x3(mat1,D12_2,mat2);
      for (int ii = 0; ii < 3; ++ii){
        for (int jj = 0; jj < 3; ++jj){
          int ij = 3*ii+jj;
          FC->matMin2Plus4[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      /* invert(D11_2) * D13_2 */
      fdlib_math_matmul3x3(mat1,D13_2,mat2);
      for (int ii = 0; ii < 3; ++ii){
        for (int jj = 0; jj < 3; ++jj){
          int ij = 3*ii+jj;
          FC->matMin2Plus5[iptr_f*9+ij] = mat2[ii][jj];
        }
      }

      /* plus -> min */
      /* invert(D11_1) * D */
      /* D = D11_2, D12_2, D13_2, D12_1, D13_1 */
      for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
          mat1[ii][jj] = D11_1[ii][jj];
        }
      }
      fdlib_math_invert3x3(mat1);
      /* invert(D11_1) * D11_2 */
      fdlib_math_matmul3x3(mat1,D11_2,mat2);
      for (int ii = 0; ii < 3; ++ii){
        for (int jj = 0; jj < 3; ++jj){
          int ij = 3*ii+jj;
          FC->matPlus2Min1[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      /* invert(D11_1) * D12_2 */
      fdlib_math_matmul3x3(mat1,D12_2,mat2);
      for (int ii = 0; ii < 3; ++ii){
        for (int jj = 0; jj < 3; ++jj){
          int ij = 3*ii+jj;
          FC->matPlus2Min2[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      /* invert(D11_1) * D13_2 */
      fdlib_math_matmul3x3(mat1,D13_2,mat2);
      for (int ii = 0; ii < 3; ++ii){
        for (int jj = 0; jj < 3; ++jj){
          int ij = 3*ii+jj;
          FC->matPlus2Min3[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      /* invert(D11_1) * D12_1 */
      fdlib_math_matmul3x3(mat1,D12_1,mat2);
      for (int ii = 0; ii < 3; ++ii){
        for (int jj = 0; jj < 3; ++jj){
          int ij = 3*ii+jj;
          FC->matPlus2Min4[iptr_f*9+ij] = mat2[ii][jj];
        }
      }
      /* invert(D11_1) * D13_1 */
      fdlib_math_matmul3x3(mat1,D13_1,mat2);
      for (int ii = 0; ii < 3; ++ii){
        for (int jj = 0; jj < 3; ++jj){
          int ij = 3*ii+jj;
          FC->matPlus2Min5[iptr_f*9+ij] = mat2[ii][jj];
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
          vec_n[ia] /= norm;
      }

      fdlib_math_cross_product(vec_n, vec_s1, vec_s2);

      for (int i=0; i<3; i++)
      {
          FC->vec_n [iptr_f*3+i] = vec_n [i];
          FC->vec_s1[iptr_f*3+i] = vec_s1[i];
          FC->vec_s2[iptr_f*3+i] = vec_s2[i];
      }

      if (k-3 == npoint_z-1) // free surface
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

        for (int ii = 0; ii < 3; ii++){
          for (int jj = 0; jj < 3; jj++){
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
        norm = fdlib_math_norm3(vec_n);

        vec_s1[0] = FC->x_et[iptr_f];
        vec_s1[1] = FC->y_et[iptr_f];
        vec_s1[2] = FC->z_et[iptr_f];

        fdlib_math_cross_product(vec_n, vec_s1, vec_s2);
        h1_3 = vec_s2[0]*e11
             + vec_s2[1]*e12
             + vec_s2[2]*e13
        h2_3 = vec_s2[0]*e21
             + vec_s2[1]*e22
             + vec_s2[2]*e23;
        h3_3 = vec_s2[0]*e31
             + vec_s2[1]*e32
             + vec_s2[2]*e33;

        h1_3 = h1_3/norm;
        h2_3 = h2_3/norm;
        h3_3 = h3_3/norm;

        for (int ii = 0; ii < 3; ii++){
          for (int jj = 0; jj < 3; jj++){
            Tovert[ii][jj] = vec_s2[ii] * vec_n[jj];
            D11_13[ii][jj] = mu * h1_3 * Tovert[ii][jj];
            D12_13[ii][jj] = mu * h2_3 * Tovert[ii][jj];
            D13_13[ii][jj] = mu * h3_3 * Tovert[ii][jj];
            D11_1f[ii][jj] = D11_1[ii][jj] + D11_13[ii][jj];
            D12_1f[ii][jj] = D12_1[ii][jj] + D12_13[ii][jj];
            D13_1f[ii][jj] = D13_1[ii][jj] + D13_13[ii][jj];
          }
        }

        for (int ii = 0; ii < 3; ii++){
          for (int jj = 0; jj < 3; jj++){
            Tovert[ii][jj] = vec_s2[ii] * vec_n[jj];
            D11_13[ii][jj] = mu * h1_3 * Tovert[ii][jj];
            D12_13[ii][jj] = mu * h2_3 * Tovert[ii][jj];
            D13_13[ii][jj] = mu * h3_3 * Tovert[ii][jj];
            D11_2f[ii][jj] = D11_2[ii][jj] + D11_13[ii][jj];
            D12_2f[ii][jj] = D12_2[ii][jj] + D12_13[ii][jj];
            D13_2f[ii][jj] = D13_2[ii][jj] + D13_13[ii][jj];
          }
        }

        fdlib_math_matmul3x3(D13_1f, matVx2Vz1, mat1);
        fdlib_math_matmul3x3(D13_1f, matVy2Vz1, mat2);
        fdlib_math_matmul3x3(D13_2f, matVx2Vz2, mat3);
        fdlib_math_matmul3x3(D13_2f, matVy2Vz2, mat4);

        for (int ii = 0; ii < 3; ii++){
          for (int jj = 0; jj < 3; jj++){
            matVx1_free[ii][jj] = D11_1f[ii][jj] + mat1[ii][jj];
            matVy1_free[ii][jj] = D12_1f[ii][jj] + mat2[ii][jj];
            matVx2_free[ii][jj] = D11_2f[ii][jj] + mat3[ii][jj];
            matVy2_free[ii][jj] = D12_2f[ii][jj] + mat4[ii][jj];
          }
        }

        for (int ii = 0; ii < 3; ii++){
          for (int jj = 0; jj < 3; jj++){
            Tovert[ii][jj] = matVx1_free[ii][jj];
          }
        }
        fdlib_math_invert3x3(Tovert);
        fdlib_math_matmul3x3(Tovert, matVx2_free, matPlus2Min1f);
        fdlib_math_matmul3x3(Tovert, matVy2_free, matPlus2Min2f);
        fdlib_math_matmul3x3(Tovert, matVy1_free, matPlus2Min3f);
        // method 2 coef
        for (int ii = 0; ii < 3; ii++){
          for (int jj = 0; jj < 3; jj++){
            int ij = 3*ii+jj;
            FC->matT1toVxf_Min[j*9+ij] = Tovert[ii][jj];
            FC->matVytoVxf_Min[j*9+ij] = matPlus2Min3f[ii][jj];
          }
        }

        for (int ii = 0; ii < 3; ii++){
          for (int jj = 0; jj < 3; jj++){
            Tovert[ii][jj] = matVx2_free[ii][jj];
          }
        }
        fdlib_math_invert3x3(Tovert);
        fdlib_math_matmul3x3(Tovert, matVx1_free, matMin2Plus1f);
        fdlib_math_matmul3x3(Tovert, matVy1_free, matMin2Plus2f);
        fdlib_math_matmul3x3(Tovert, matVy2_free, matMin2Plus3f);

        // method 2 coef
        for (int ii = 0; ii < 3; ii++){
          for (int jj = 0; jj < 3; jj++){
            int ij = 3*ii+jj;
            FC->matT1toVxf_Plus[j*9+ij] = Tovert[ii][jj];
            FC->matVytoVxf_Plus[j*9+ij] = matMin2Plus3f[ii][jj];
          }
        }

        // save
        for (int ii = 0; ii < 3; ii++){
          for (int jj = 0; jj < 3; jj++){
            int ij = 3*ii+jj;
            FC->matVx2Vz1    [j*9+ij] = matVx2Vz1    [ii][jj];
            FC->matVy2Vz1    [j*9+ij] = matVy2Vz1    [ii][jj];
            FC->matVx2Vz2    [j*9+ij] = matVx2Vz2    [ii][jj];
            FC->matVy2Vz2    [j*9+ij] = matVy2Vz2    [ii][jj];
            FC->matVx1_free  [j*9+ij] = matVx1_free  [ii][jj];
            FC->matVy1_free  [j*9+ij] = matVy1_free  [ii][jj];
            FC->matVx2_free  [j*9+ij] = matVx2_free  [ii][jj];
            FC->matVy2_free  [j*9+ij] = matVy2_free  [ii][jj];
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
  return;
}  


