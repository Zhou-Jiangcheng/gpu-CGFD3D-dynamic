#include <stdio.h>
#include <stdlib.h>

#include "gd_t.h"
#include "fault_info.h"
#include "fdlib_math.h"
#include "fdlib_mem.h"

int
fault_coef_init(fault_coef_t *FC,
                gd_t *gd,
                int number_fault,
                int *fault_x_index)
{
  int ny = gd->ny;
  int nz = gd->nz;

  FC->number_fault = number_fault;

  FC->fault_index = (int *) malloc(sizeof(int)*number_fault);

  for(int id=0; id<number_fault; id++)
  {
    FC->fault_index[id] = fault_x_index[id];
    
    fault_coef_one_t *thisone = FC->fault_coef_one + id; 

    thisone->rho_f = (float *) malloc(sizeof(float)*ny*nz*2);
    thisone->mu_f  = (float *) malloc(sizeof(float)*ny*nz*2);
    thisone->lam_f = (float *) malloc(sizeof(float)*ny*nz*2);

    thisone->D21_1 = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->D22_1 = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->D23_1 = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->D31_1 = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->D32_1 = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->D33_1 = (float *) malloc(sizeof(float)*ny*nz*3*3);

    thisone->D21_2 = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->D22_2 = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->D23_2 = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->D31_2 = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->D32_2 = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->D33_2 = (float *) malloc(sizeof(float)*ny*nz*3*3);

    thisone->matMin2Plus1 = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->matMin2Plus2 = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->matMin2Plus3 = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->matMin2Plus4 = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->matMin2Plus5 = (float *) malloc(sizeof(float)*ny*nz*3*3);

    thisone->matPlus2Min1 = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->matPlus2Min2 = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->matPlus2Min3 = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->matPlus2Min4 = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->matPlus2Min5 = (float *) malloc(sizeof(float)*ny*nz*3*3);

    thisone->matT1toVx_Min  = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->matVytoVx_Min  = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->matVztoVx_Min  = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->matT1toVx_Plus = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->matVytoVx_Plus = (float *) malloc(sizeof(float)*ny*nz*3*3);
    thisone->matVztoVx_Plus = (float *) malloc(sizeof(float)*ny*nz*3*3);

    thisone->vec_n  = (float *) malloc(sizeof(float)*ny*nz*3);
    thisone->vec_s1 = (float *) malloc(sizeof(float)*ny*nz*3);
    thisone->vec_s2 = (float *) malloc(sizeof(float)*ny*nz*3);
    thisone->x_et   = (float *) malloc(sizeof(float)*ny*nz);
    thisone->y_et   = (float *) malloc(sizeof(float)*ny*nz);
    thisone->z_et   = (float *) malloc(sizeof(float)*ny*nz);

    // fault with free surface coef
    // NOTE: even this thread without free surface, still malloc 
    // this is because easy coding. this coef not used, if without free surface
     
    thisone->matVx2Vz1     = (float *) malloc(sizeof(float)*ny*3*3);
    thisone->matVy2Vz1     = (float *) malloc(sizeof(float)*ny*3*3);
    thisone->matVx2Vz2     = (float *) malloc(sizeof(float)*ny*3*3);
    thisone->matVy2Vz2     = (float *) malloc(sizeof(float)*ny*3*3);
    thisone->matPlus2Min1f = (float *) malloc(sizeof(float)*ny*3*3);
    thisone->matPlus2Min2f = (float *) malloc(sizeof(float)*ny*3*3);
    thisone->matPlus2Min3f = (float *) malloc(sizeof(float)*ny*3*3);
    thisone->matMin2Plus1f = (float *) malloc(sizeof(float)*ny*3*3);
    thisone->matMin2Plus2f = (float *) malloc(sizeof(float)*ny*3*3);
    thisone->matMin2Plus3f = (float *) malloc(sizeof(float)*ny*3*3);

    thisone->matT1toVxf_Min  = (float *) malloc(sizeof(float)*ny*3*3);
    thisone->matVytoVxf_Min  = (float *) malloc(sizeof(float)*ny*3*3);
    thisone->matT1toVxf_Plus = (float *) malloc(sizeof(float)*ny*3*3);
    thisone->matVytoVxf_Plus = (float *) malloc(sizeof(float)*ny*3*3);
  }

  return 0;
}


int 
fault_coef_cal(gd_t *gd, 
               gd_metric_t *metric, 
               md_t *md,
               fault_coef_t *FC)
{
  int ny = gd->ny;
  int nz = gd->nz;
  size_t siz_iy = gd->siz_iy;
  size_t siz_iz = gd->siz_iz;
  size_t siz_slice_yz = gd->siz_slice_yz;
  // x direction only has 1 mpi. 
  int total_point_z = gd->total_point_z;
  int gnk1 = gd->gnk1;
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

  for(int id=0; id<FC->number_fault; id++)
  {
    int i0 = FC->fault_index[id] + 3; //fault plane x index with ghost

    fault_coef_one_t *thisone = FC->fault_coef_one + id;

    for(int k=0; k<nz; k++) 
    {
      for(int j=0; j<ny; j++) 
      {
        iptr = i0 + j * siz_iy + k * siz_iz;  
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
        thisone->rho_f[iptr_f+0*siz_slice_yz] = rho;
        thisone->lam_f[iptr_f+0*siz_slice_yz] = lam;
        thisone->mu_f [iptr_f+0*siz_slice_yz] = mu;
        //plus media
        thisone->rho_f[iptr_f+1*siz_slice_yz] = rho;
        thisone->lam_f[iptr_f+1*siz_slice_yz] = lam;
        thisone->mu_f [iptr_f+1*siz_slice_yz] = mu;
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

        thisone->D21_1[iptr_f*9+0] = lam2mu*e21*e11+mu*(e22*e12+e23*e13);
        thisone->D21_1[iptr_f*9+1] = lam*e21*e12+mu*e22*e11;
        thisone->D21_1[iptr_f*9+2] = lam*e21*e13+mu*e23*e11;
        thisone->D21_1[iptr_f*9+3] = mu*e21*e12+lam*e22*e11;
        thisone->D21_1[iptr_f*9+4] = lam2mu*e22*e12+mu*(e21*e11+e23*e13);
        thisone->D21_1[iptr_f*9+5] = lam*e22*e13+mu*e23*e12;
        thisone->D21_1[iptr_f*9+6] = mu*e21*e13+lam*e23*e11;
        thisone->D21_1[iptr_f*9+7] = mu*e22*e13+lam*e23*e12;
        thisone->D21_1[iptr_f*9+8] = lam2mu*e23*e13+mu*(e21*e11+e22*e12);

        thisone->D22_1[iptr_f*9+0] = lam2mu*e21*e21+mu*(e22*e22+e23*e23);
        thisone->D22_1[iptr_f*9+1] = lam*e21*e22+mu*e22*e21;
        thisone->D22_1[iptr_f*9+2] = lam*e21*e23+mu*e23*e21;
        thisone->D22_1[iptr_f*9+3] = mu*e21*e22+lam*e22*e21;
        thisone->D22_1[iptr_f*9+4] = lam2mu*e22*e22+mu*(e21*e21+e23*e23);
        thisone->D22_1[iptr_f*9+5] = lam*e22*e23+mu*e23*e22;
        thisone->D22_1[iptr_f*9+6] = mu*e21*e23+lam*e23*e21;
        thisone->D22_1[iptr_f*9+7] = mu*e22*e23+lam*e23*e22;
        thisone->D22_1[iptr_f*9+8] = lam2mu*e23*e23+mu*(e21*e21+e22*e22);

        thisone->D23_1[iptr_f*9+0] = lam2mu*e21*e31+mu*(e22*e32+e23*e33);
        thisone->D23_1[iptr_f*9+1] = lam*e21*e32+mu*e22*e31;
        thisone->D23_1[iptr_f*9+2] = lam*e21*e33+mu*e23*e31;
        thisone->D23_1[iptr_f*9+3] = mu*e21*e32+lam*e22*e31;
        thisone->D23_1[iptr_f*9+4] = lam2mu*e22*e32+mu*(e21*e31+e23*e33);
        thisone->D23_1[iptr_f*9+5] = lam*e22*e33+mu*e23*e32;
        thisone->D23_1[iptr_f*9+6] = mu*e21*e33+lam*e23*e31;
        thisone->D23_1[iptr_f*9+7] = mu*e22*e33+lam*e23*e32;
        thisone->D23_1[iptr_f*9+8] = lam2mu*e23*e33+mu*(e21*e31+e22*e32);

        thisone->D31_1[iptr_f*9+0] = lam2mu*e31*e11+mu*(e32*e12+e33*e13);
        thisone->D31_1[iptr_f*9+1] = lam*e31*e12+mu*e32*e11;
        thisone->D31_1[iptr_f*9+2] = lam*e31*e13+mu*e33*e11;
        thisone->D31_1[iptr_f*9+3] = mu*e31*e12+lam*e32*e11;
        thisone->D31_1[iptr_f*9+4] = lam2mu*e32*e12+mu*(e31*e11+e33*e13);
        thisone->D31_1[iptr_f*9+5] = lam*e32*e13+mu*e33*e12;
        thisone->D31_1[iptr_f*9+6] = mu*e31*e13+lam*e33*e11;
        thisone->D31_1[iptr_f*9+7] = mu*e32*e13+lam*e33*e12;
        thisone->D31_1[iptr_f*9+8] = lam2mu*e33*e13+mu*(e31*e11+e32*e12);

        thisone->D32_1[iptr_f*9+0] = lam2mu*e31*e21+mu*(e32*e22+e33*e23);
        thisone->D32_1[iptr_f*9+1] = lam*e31*e22+mu*e32*e21;
        thisone->D32_1[iptr_f*9+2] = lam*e31*e23+mu*e33*e21;
        thisone->D32_1[iptr_f*9+3] = mu*e31*e22+lam*e32*e21;
        thisone->D32_1[iptr_f*9+4] = lam2mu*e32*e22+mu*(e31*e21+e33*e23);
        thisone->D32_1[iptr_f*9+5] = lam*e32*e23+mu*e33*e22;
        thisone->D32_1[iptr_f*9+6] = mu*e31*e23+lam*e33*e21;
        thisone->D32_1[iptr_f*9+7] = mu*e32*e23+lam*e33*e22;
        thisone->D32_1[iptr_f*9+8] = lam2mu*e33*e23+mu*(e31*e21+e32*e22);

        thisone->D33_1[iptr_f*9+0] = lam2mu*e31*e31+mu*(e32*e32+e33*e33);
        thisone->D33_1[iptr_f*9+1] = lam*e31*e32+mu*e32*e31;
        thisone->D33_1[iptr_f*9+2] = lam*e31*e33+mu*e33*e31;
        thisone->D33_1[iptr_f*9+3] = mu*e31*e32+lam*e32*e31;
        thisone->D33_1[iptr_f*9+4] = lam2mu*e32*e32+mu*(e31*e31+e33*e33);
        thisone->D33_1[iptr_f*9+5] = lam*e32*e33+mu*e33*e32;
        thisone->D33_1[iptr_f*9+6] = mu*e31*e33+lam*e33*e31;
        thisone->D33_1[iptr_f*9+7] = mu*e32*e33+lam*e33*e32;
        thisone->D33_1[iptr_f*9+8] = lam2mu*e33*e33+mu*(e31*e31+e32*e32);

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

        thisone->D21_2[iptr_f*9+0] = lam2mu*e21*e11+mu*(e22*e12+e23*e13);
        thisone->D21_2[iptr_f*9+1] = lam*e21*e12+mu*e22*e11;
        thisone->D21_2[iptr_f*9+2] = lam*e21*e13+mu*e23*e11;
        thisone->D21_2[iptr_f*9+3] = mu*e21*e12+lam*e22*e11;
        thisone->D21_2[iptr_f*9+4] = lam2mu*e22*e12+mu*(e21*e11+e23*e13);
        thisone->D21_2[iptr_f*9+5] = lam*e22*e13+mu*e23*e12;
        thisone->D21_2[iptr_f*9+6] = mu*e21*e13+lam*e23*e11;
        thisone->D21_2[iptr_f*9+7] = mu*e22*e13+lam*e23*e12;
        thisone->D21_2[iptr_f*9+8] = lam2mu*e23*e13+mu*(e21*e11+e22*e12);

        thisone->D22_2[iptr_f*9+0] = lam2mu*e21*e21+mu*(e22*e22+e23*e23);
        thisone->D22_2[iptr_f*9+1] = lam*e21*e22+mu*e22*e21;
        thisone->D22_2[iptr_f*9+2] = lam*e21*e23+mu*e23*e21;
        thisone->D22_2[iptr_f*9+3] = mu*e21*e22+lam*e22*e21;
        thisone->D22_2[iptr_f*9+4] = lam2mu*e22*e22+mu*(e21*e21+e23*e23);
        thisone->D22_2[iptr_f*9+5] = lam*e22*e23+mu*e23*e22;
        thisone->D22_2[iptr_f*9+6] = mu*e21*e23+lam*e23*e21;
        thisone->D22_2[iptr_f*9+7] = mu*e22*e23+lam*e23*e22;
        thisone->D22_2[iptr_f*9+8] = lam2mu*e23*e23+mu*(e21*e21+e22*e22);

        thisone->D23_2[iptr_f*9+0] = lam2mu*e21*e31+mu*(e22*e32+e23*e33);
        thisone->D23_2[iptr_f*9+1] = lam*e21*e32+mu*e22*e31;
        thisone->D23_2[iptr_f*9+2] = lam*e21*e33+mu*e23*e31;
        thisone->D23_2[iptr_f*9+3] = mu*e21*e32+lam*e22*e31;
        thisone->D23_2[iptr_f*9+4] = lam2mu*e22*e32+mu*(e21*e31+e23*e33);
        thisone->D23_2[iptr_f*9+5] = lam*e22*e33+mu*e23*e32;
        thisone->D23_2[iptr_f*9+6] = mu*e21*e33+lam*e23*e31;
        thisone->D23_2[iptr_f*9+7] = mu*e22*e33+lam*e23*e32;
        thisone->D23_2[iptr_f*9+8] = lam2mu*e23*e33+mu*(e21*e31+e22*e32);

        thisone->D31_2[iptr_f*9+0] = lam2mu*e31*e11+mu*(e32*e12+e33*e13);
        thisone->D31_2[iptr_f*9+1] = lam*e31*e12+mu*e32*e11;
        thisone->D31_2[iptr_f*9+2] = lam*e31*e13+mu*e33*e11;
        thisone->D31_2[iptr_f*9+3] = mu*e31*e12+lam*e32*e11;
        thisone->D31_2[iptr_f*9+4] = lam2mu*e32*e12+mu*(e31*e11+e33*e13);
        thisone->D31_2[iptr_f*9+5] = lam*e32*e13+mu*e33*e12;
        thisone->D31_2[iptr_f*9+6] = mu*e31*e13+lam*e33*e11;
        thisone->D31_2[iptr_f*9+7] = mu*e32*e13+lam*e33*e12;
        thisone->D31_2[iptr_f*9+8] = lam2mu*e33*e13+mu*(e31*e11+e32*e12);

        thisone->D32_2[iptr_f*9+0] = lam2mu*e31*e21+mu*(e32*e22+e33*e23);
        thisone->D32_2[iptr_f*9+1] = lam*e31*e22+mu*e32*e21;
        thisone->D32_2[iptr_f*9+2] = lam*e31*e23+mu*e33*e21;
        thisone->D32_2[iptr_f*9+3] = mu*e31*e22+lam*e32*e21;
        thisone->D32_2[iptr_f*9+4] = lam2mu*e32*e22+mu*(e31*e21+e33*e23);
        thisone->D32_2[iptr_f*9+5] = lam*e32*e23+mu*e33*e22;
        thisone->D32_2[iptr_f*9+6] = mu*e31*e23+lam*e33*e21;
        thisone->D32_2[iptr_f*9+7] = mu*e32*e23+lam*e33*e22;
        thisone->D32_2[iptr_f*9+8] = lam2mu*e33*e23+mu*(e31*e21+e32*e22);

        thisone->D33_2[iptr_f*9+0] = lam2mu*e31*e31+mu*(e32*e32+e33*e33);
        thisone->D33_2[iptr_f*9+1] = lam*e31*e32+mu*e32*e31;
        thisone->D33_2[iptr_f*9+2] = lam*e31*e33+mu*e33*e31;
        thisone->D33_2[iptr_f*9+3] = mu*e31*e32+lam*e32*e31;
        thisone->D33_2[iptr_f*9+4] = lam2mu*e32*e32+mu*(e31*e31+e33*e33);
        thisone->D33_2[iptr_f*9+5] = lam*e32*e33+mu*e33*e32;
        thisone->D33_2[iptr_f*9+6] = mu*e31*e33+lam*e33*e31;
        thisone->D33_2[iptr_f*9+7] = mu*e32*e33+lam*e33*e32;
        thisone->D33_2[iptr_f*9+8] = lam2mu*e33*e33+mu*(e31*e31+e32*e32);

        for (int ii = 0; ii < 3; ii++) 
        {
          for (int jj = 0; jj < 3; jj++) 
          {
            int ij = 3*ii+jj;
            D11_1[ii][jj] *= jac;
            D12_1[ii][jj] *= jac;
            D13_1[ii][jj] *= jac;
            thisone->D21_1[iptr_f*9+ij] *= jac;
            thisone->D22_1[iptr_f*9+ij] *= jac;
            thisone->D23_1[iptr_f*9+ij] *= jac;
            thisone->D31_1[iptr_f*9+ij] *= jac;
            thisone->D32_1[iptr_f*9+ij] *= jac;
            thisone->D33_1[iptr_f*9+ij] *= jac;
            D11_2[ii][jj] *= jac;
            D12_2[ii][jj] *= jac;
            D13_2[ii][jj] *= jac;
            thisone->D21_2[iptr_f*9+ij] *= jac;
            thisone->D22_2[iptr_f*9+ij] *= jac;
            thisone->D23_2[iptr_f*9+ij] *= jac;
            thisone->D31_2[iptr_f*9+ij] *= jac;
            thisone->D32_2[iptr_f*9+ij] *= jac;
            thisone->D33_2[iptr_f*9+ij] *= jac;
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
            thisone->matMin2Plus1[iptr_f*9+ij] = mat2[ii][jj];
          }
        }
        // invert(D11_2) * D12_1 
        fdlib_math_matmul3x3(mat1,D12_1,mat2);
        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            int ij = 3*ii+jj;
            thisone->matMin2Plus2[iptr_f*9+ij] = mat2[ii][jj];
          }
        }
        // invert(D11_2) * D13_1 
        fdlib_math_matmul3x3(mat1,D13_1,mat2);
        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            int ij = 3*ii+jj;
            thisone->matMin2Plus3[iptr_f*9+ij] = mat2[ii][jj];
          }
        }
        // invert(D11_2) * D12_2
        fdlib_math_matmul3x3(mat1,D12_2,mat2);
        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            int ij = 3*ii+jj;
            thisone->matMin2Plus4[iptr_f*9+ij] = mat2[ii][jj];
          }
        }
        // invert(D11_2) * D13_2
        fdlib_math_matmul3x3(mat1,D13_2,mat2);
        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            int ij = 3*ii+jj;
            thisone->matMin2Plus5[iptr_f*9+ij] = mat2[ii][jj];
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
            thisone->matPlus2Min1[iptr_f*9+ij] = mat2[ii][jj];
          }
        }
        // invert(D11_1) * D12_2
        fdlib_math_matmul3x3(mat1,D12_2,mat2);
        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            int ij = 3*ii+jj;
            thisone->matPlus2Min2[iptr_f*9+ij] = mat2[ii][jj];
          }
        }
        // invert(D11_1) * D13_2 
        fdlib_math_matmul3x3(mat1,D13_2,mat2);
        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            int ij = 3*ii+jj;
            thisone->matPlus2Min3[iptr_f*9+ij] = mat2[ii][jj];
          }
        }
        // invert(D11_1) * D12_1 
        fdlib_math_matmul3x3(mat1,D12_1,mat2);
        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            int ij = 3*ii+jj;
            thisone->matPlus2Min4[iptr_f*9+ij] = mat2[ii][jj];
          }
        }
        // invert(D11_1) * D13_1
        fdlib_math_matmul3x3(mat1,D13_1,mat2);
        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            int ij = 3*ii+jj;
            thisone->matPlus2Min5[iptr_f*9+ij] = mat2[ii][jj];
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
            thisone->matT1toVx_Min[iptr_f*9+ij] = mat1[ii][jj];
          }
        }
        // invert(D11_1) * D12_1 
        fdlib_math_matmul3x3(mat1,D12_1,mat2);
        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            int ij = 3*ii+jj;
            thisone->matVytoVx_Min[iptr_f*9+ij] = mat2[ii][jj];
          }
        }
        // invert(D11_1) * D13_1 
        fdlib_math_matmul3x3(mat1,D13_1,mat2);
        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            int ij = 3*ii+jj;
            thisone->matVztoVx_Min[iptr_f*9+ij] = mat2[ii][jj];
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
            thisone->matT1toVx_Plus[iptr_f*9+ij] = mat1[ii][jj];
          }
        }
        // invert(D11_2) * D12_2 
        fdlib_math_matmul3x3(mat1,D12_2,mat2);
        for (int ii = 0; ii < 3; ii++) 
        {
          for (int jj = 0; jj < 3; jj++) 
          {
            int ij = 3*ii+jj;
            thisone->matVytoVx_Plus[iptr_f*9+ij] = mat2[ii][jj];
          }
        }
        // invert(D11_2) * D13_2 
        fdlib_math_matmul3x3(mat1,D13_2,mat2);
        for (int ii = 0; ii < 3; ii++) 
        {
          for (int jj = 0; jj < 3; jj++) 
          {
            int ij = 3*ii+jj;
            thisone->matVztoVx_Plus[iptr_f*9+ij] = mat2[ii][jj];
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
        thisone->x_et[iptr_f] = mat3[0][1];
        thisone->y_et[iptr_f] = mat3[1][1];
        thisone->z_et[iptr_f] = mat3[2][1];

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
          thisone->vec_n [iptr_f*3+i] = vec_n [i];
          thisone->vec_s1[iptr_f*3+i] = vec_s1[i];
          thisone->vec_s2[iptr_f*3+i] = vec_s2[i];
        }

        if ((k-3+gnk1) == total_point_z-1) // free surface global index, index start 0
        {
          for (int ii = 0; ii < 3; ii++) {
            for (int jj = 0; jj < 3; jj++) {
              int ij = 3*ii+jj; 
              A[ii][jj] =  thisone->D33_1[iptr_f*9+ij];
              B[ii][jj] = -thisone->D31_1[iptr_f*9+ij];
              C[ii][jj] = -thisone->D32_1[iptr_f*9+ij];
            }
          }
          fdlib_math_invert3x3(A);
          fdlib_math_matmul3x3(A, B, matVx2Vz1);
          fdlib_math_matmul3x3(A, C, matVy2Vz1);

          for (int ii = 0; ii < 3; ii++) {
            for (int jj = 0; jj < 3; jj++) {
              int ij = 3*ii+jj; 
              A[ii][jj] =  thisone->D33_2[iptr_f*9+ij];
              B[ii][jj] = -thisone->D31_2[iptr_f*9+ij];
              C[ii][jj] = -thisone->D32_2[iptr_f*9+ij];
            }
          }
          fdlib_math_invert3x3(A);
          fdlib_math_matmul3x3(A, B, matVx2Vz2);
          fdlib_math_matmul3x3(A, C, matVy2Vz2);


          vec_n[0] = e11;
          vec_n[1] = e12;
          vec_n[2] = e13;

          vec_s1[0] = thisone->x_et[iptr_f];
          vec_s1[1] = thisone->y_et[iptr_f];
          vec_s1[2] = thisone->z_et[iptr_f];

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
              thisone->matT1toVxf_Min[j*9+ij] = Tovert[ii][jj];
              thisone->matVytoVxf_Min[j*9+ij] = matPlus2Min3f[ii][jj];
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
              thisone->matT1toVxf_Plus[j*9+ij] = Tovert[ii][jj];
              thisone->matVytoVxf_Plus[j*9+ij] = matMin2Plus3f[ii][jj];
            }
          }

          // save
          for (int ii = 0; ii < 3; ii++)
          {
            for (int jj = 0; jj < 3; jj++)
            {
              int ij = 3*ii+jj;
              thisone->matVx2Vz1    [j*9+ij] = matVx2Vz1    [ii][jj];
              thisone->matVy2Vz1    [j*9+ij] = matVy2Vz1    [ii][jj];
              thisone->matVx2Vz2    [j*9+ij] = matVx2Vz2    [ii][jj];
              thisone->matVy2Vz2    [j*9+ij] = matVy2Vz2    [ii][jj];
              thisone->matPlus2Min1f[j*9+ij] = matPlus2Min1f[ii][jj];
              thisone->matPlus2Min2f[j*9+ij] = matPlus2Min2f[ii][jj];
              thisone->matPlus2Min3f[j*9+ij] = matPlus2Min3f[ii][jj];
              thisone->matMin2Plus1f[j*9+ij] = matMin2Plus1f[ii][jj];
              thisone->matMin2Plus2f[j*9+ij] = matMin2Plus2f[ii][jj];
              thisone->matMin2Plus3f[j*9+ij] = matMin2Plus3f[ii][jj];
            }
          }
        }
      }                               
    }
  }
  return 0;
}  

int
fault_init(fault_t *F,
           gd_t *gd,
           int number_fault,
           int *fault_x_index)
{
  int ny = gd->ny;
  int nz = gd->nz;

  F->number_fault = number_fault;

  F->fault_index = (int *) malloc(sizeof(int)*number_fault);

  F->ncmp = 11;
  // position of each var
  size_t *cmp_pos = (size_t *) malloc(sizeof(size_t)*F->ncmp);
  
  // name of each v4d
  char **cmp_name = (char **) fdlib_mem_malloc_2l_char(F->ncmp,
                                                       CONST_MAX_STRLEN,
                                                       "gd_curv_metric_init");
  
  // set value
  for (int icmp=0; icmp < F->ncmp; icmp++)
  {
    cmp_pos[icmp] = icmp * ny * nz;
  }
  int icmp = 0;
  sprintf(cmp_name[icmp],"%s","Tn");

  icmp += 1;
  sprintf(cmp_name[icmp],"%s","Ts1");
  
  icmp += 1;
  sprintf(cmp_name[icmp],"%s","Ts2");
  
  icmp += 1;
  sprintf(cmp_name[icmp],"%s","Vs");

  icmp += 1;
  sprintf(cmp_name[icmp],"%s","Vs1");
  
  icmp += 1;
  sprintf(cmp_name[icmp],"%s","Vs2");
  
  icmp += 1;
  sprintf(cmp_name[icmp],"%s","Slip");

  icmp += 1;
  sprintf(cmp_name[icmp],"%s","Slip1");

  icmp += 1;
  sprintf(cmp_name[icmp],"%s","Slip2");
  
  icmp += 1;
  sprintf(cmp_name[icmp],"%s","Peak_vs");

  icmp += 1;
  sprintf(cmp_name[icmp],"%s","Init_t0");

  // set pointer
  F->cmp_pos  = cmp_pos;
  F->cmp_name = cmp_name;

  for(int id=0; id<number_fault; id++)
  {
    F->fault_index[id] = fault_x_index[id];

    fault_one_t *thisone = F->fault_one + id;

    // for input
    thisone->T0x  = (float *) malloc(sizeof(float)*ny*nz);  // stress_init_x
    thisone->T0y  = (float *) malloc(sizeof(float)*ny*nz);  // stress_init_y
    thisone->T0z  = (float *) malloc(sizeof(float)*ny*nz);  // stress_init_z
    thisone->mu_s = (float *) malloc(sizeof(float)*ny*nz);
    thisone->mu_d = (float *) malloc(sizeof(float)*ny*nz);
    thisone->Dc   = (float *) malloc(sizeof(float)*ny*nz);
    thisone->C0   = (float *) malloc(sizeof(float)*ny*nz);
    // for output
    thisone->output  = (float *) malloc(sizeof(float)*ny*nz*F->ncmp);
    thisone->Tn  = thisone->output + F->cmp_pos[0];
    thisone->Ts1 = thisone->output + F->cmp_pos[1];
    thisone->Ts2 = thisone->output + F->cmp_pos[2];
    thisone->Vs  = thisone->output + F->cmp_pos[3];
    thisone->Vs1 = thisone->output + F->cmp_pos[4];
    thisone->Vs2 = thisone->output + F->cmp_pos[5];
    thisone->Slip  = thisone->output + F->cmp_pos[6];
    thisone->Slip1 = thisone->output + F->cmp_pos[7];
    thisone->Slip2 = thisone->output + F->cmp_pos[8];
    thisone->Peak_vs = thisone->output + F->cmp_pos[9];
    thisone->Init_t0 = thisone->output + F->cmp_pos[10];
    // for inner
    thisone->tTn          = (float *) malloc(sizeof(float)*ny*nz);
    thisone->tTs1         = (float *) malloc(sizeof(float)*ny*nz);
    thisone->tTs2         = (float *) malloc(sizeof(float)*ny*nz);
    thisone->united       = (int *) malloc(sizeof(int)*ny*nz);
    thisone->faultgrid    = (int *) malloc(sizeof(int)*ny*nz);
    thisone->rup_index_y  = (int *) malloc(sizeof(int)*ny*nz);
    thisone->rup_index_z  = (int *) malloc(sizeof(int)*ny*nz);
    thisone->flag_rup     = (int *) malloc(sizeof(int)*ny*nz);
    thisone->init_t0_flag = (int *) malloc(sizeof(int)*ny*nz);

    memset(thisone->Slip,         0, sizeof(float)*ny*nz);
    memset(thisone->Slip1,        0, sizeof(float)*ny*nz); 
    memset(thisone->Slip2,        0, sizeof(float)*ny*nz); 
    memset(thisone->Vs,           0, sizeof(float)*ny*nz);
    memset(thisone->Vs1,          0, sizeof(float)*ny*nz);
    memset(thisone->Vs2,          0, sizeof(float)*ny*nz);
    memset(thisone->init_t0_flag, 0, sizeof(int)  *ny*nz);
    memset(thisone->Peak_vs,      0, sizeof(float)*ny*nz);
  }

  return 0;
}

int
fault_set(fault_t *F,
          fault_coef_t *FC,
          gd_t *gd,
          int bdry_has_free,
          int *fault_grid,
          char *init_stress_dir)
{
  int nj1 = gd->nj1;
  int nk1 = gd->nk1;
  int nj2 = gd->nj2;
  int nk2 = gd->nk2;
  int nj = gd->nj;
  int nk = gd->nk;
  int ny = gd->ny;
  int gnj1 = gd->gnj1;
  int gnk1 = gd->gnk1;
  int total_point_y = gd->total_point_y;
  int total_point_z = gd->total_point_z;
  int gj, gk;
  size_t iptr_f; 
  float vec_n[3], vec_s1[3], vec_s2[3];

  for(int id=0; id<FC->number_fault; id++)
  {
    fault_one_t *F_thisone = F->fault_one + id;
    fault_coef_one_t *FC_thisone = FC->fault_coef_one + id;

    nc_read_init_stress(F_thisone, gd, id, init_stress_dir);

    for (int k=0; k<nk; k++)
    {
      for (int j=0; j<nj; j++)
      {
        iptr_f = (j+nj1) + (k+nk1) * ny; //with ghost

        vec_n [0] = FC_thisone->vec_n [iptr_f*3 + 0];
        vec_n [1] = FC_thisone->vec_n [iptr_f*3 + 1];
        vec_n [2] = FC_thisone->vec_n [iptr_f*3 + 2];
        vec_s1[0] = FC_thisone->vec_s1[iptr_f*3 + 0];
        vec_s1[1] = FC_thisone->vec_s1[iptr_f*3 + 1];
        vec_s1[2] = FC_thisone->vec_s1[iptr_f*3 + 2];
        vec_s2[0] = FC_thisone->vec_s2[iptr_f*3 + 0];
        vec_s2[1] = FC_thisone->vec_s2[iptr_f*3 + 1];
        vec_s2[2] = FC_thisone->vec_s2[iptr_f*3 + 2];

        // transform init stress to local coordinate, not nessessary,
        // only for output 
        F_thisone->Tn [iptr_f] = F_thisone->T0x[iptr_f] * vec_n[0]
                               + F_thisone->T0y[iptr_f] * vec_n[1]
                               + F_thisone->T0z[iptr_f] * vec_n[2];
        F_thisone->Ts1[iptr_f] = F_thisone->T0x[iptr_f] * vec_s1[0]
                               + F_thisone->T0y[iptr_f] * vec_s1[1]
                               + F_thisone->T0z[iptr_f] * vec_s1[2];
        F_thisone->Ts2[iptr_f] = F_thisone->T0x[iptr_f] * vec_s2[0]
                               + F_thisone->T0y[iptr_f] * vec_s2[1]
                               + F_thisone->T0z[iptr_f] * vec_s2[2];

        gj = gnj1 + j;
        gk = gnk1 + k;
        // rup_index = 0 means points out of fault.
        // NOTE: boundry 3 points out of fault, due to use different fd stencil
        if( gj <= fault_grid[0+4*id]+3 || gj >= fault_grid[1+4*id]-3 ){
          F_thisone->rup_index_y[iptr_f] = 0;
        }else{
          F_thisone->rup_index_y[iptr_f] = 1;
        }

        if( gk <= fault_grid[2+4*id]+3 || gk >= fault_grid[3+4*id]-3 ){
          F_thisone->rup_index_z[iptr_f] = 0;
        }else{
          F_thisone->rup_index_z[iptr_f] = 1;
        }
        
        // united is used for pml, pml with fault easy write code
        // united == 1, means contain pml and strong boundry 
        // usually unilateral pml < 40 and strong boundry >= 50
        if(bdry_has_free == 1) 
        {
          if( gj >= 40 && gj < total_point_y - 40 && gk >= 40) {
            F_thisone->united[iptr_f] = 0;
          }else{
            F_thisone->united[iptr_f] = 1;
          }
        } 
        else if(bdry_has_free == 0)
        {
          if( gj >= 40 && gj < total_point_y - 40 && gk >= 40 
              && gk < total_point_z - 40) {
            F_thisone->united[iptr_f] = 0;
          }else{
            F_thisone->united[iptr_f] = 1;
          }
        }
        if( gj >= fault_grid[0+4*id]+3 && gj <= fault_grid[1+4*id]-3 &&
            gk >= fault_grid[2+4*id]+3 && gk <= fault_grid[3+4*id]-3 ) {
          F_thisone->faultgrid[iptr_f] = 1;
        }else{
          F_thisone->faultgrid[iptr_f] = 0;
        }

        F_thisone->Init_t0[iptr_f] = -9999.9;
        F_thisone->flag_rup[iptr_f] = 0;
      }
    }
  }
  return 0;
}

int 
nc_read_init_stress(fault_one_t *F_thisone, 
                    gd_t *gd,
                    int id,
                    char *init_stress_dir)
{
  int nj = gd->nj;
  int nk = gd->nk;
  int nj1 = gd->nj1;
  int nk1 = gd->nk1;
  int nj2 = gd->nj2;
  int nk2 = gd->nk2;
  int ny = gd->ny;
  int gnj1 = gd->gnj1;
  int gnk1 = gd->gnk1;
  int ierr;
  int ncid;
  int varid;
  size_t start[] = {gnk1, gnj1};
  size_t count[] = {nk, nj}; // y vary first
  size_t iptr, iptr1;

  char in_file[CONST_MAX_STRLEN]; 
  sprintf(in_file,"%s/init_stress_%d.nc",init_stress_dir,id+1);  //id+1, due to C code start 0

  float *var_in = (float *) malloc(sizeof(float)*nj*nk);

  ierr = nc_open(in_file, NC_NOWRITE, &ncid); handle_nc_err(ierr);

  ierr = nc_inq_varid(ncid, "Tx", &varid); handle_nc_err(ierr);
  ierr = nc_get_vara_float(ncid, varid, start, count, var_in); handle_nc_err(ierr);
  for(int k=nk1; k<=nk2; k++) {
    for(int j=nj1; j<=nj2; j++) 
    {
      iptr = j + k*ny; 
      iptr1 = (j-3) + (k-3)*nj; 
      F_thisone->T0x[iptr] = var_in[iptr1];
    }
  }

  ierr = nc_inq_varid(ncid, "Ty", &varid); handle_nc_err(ierr);
  ierr = nc_get_vara_float(ncid, varid, start, count, var_in); handle_nc_err(ierr);
  for(int k=nk1; k<=nk2; k++) {
    for(int j=nj1; j<=nj2; j++) 
    {
      iptr = j + k*ny; 
      iptr1 = (j-3) + (k-3)*nj; 
      F_thisone->T0y[iptr] = var_in[iptr1];
    }
  }

  ierr = nc_inq_varid(ncid, "Tz", &varid); handle_nc_err(ierr);
  ierr = nc_get_vara_float(ncid, varid, start, count, var_in); handle_nc_err(ierr);
  for(int k=nk1; k<=nk2; k++) {
    for(int j=nj1; j<=nj2; j++) 
    {
      iptr = j + k*ny; 
      iptr1 = (j-3) + (k-3)*nj; 
      F_thisone->T0z[iptr] = var_in[iptr1];
    }
  }

  ierr = nc_inq_varid(ncid, "mu_s", &varid); handle_nc_err(ierr);
  ierr = nc_get_vara_float(ncid, varid, start, count, var_in); handle_nc_err(ierr);
  for(int k=nk1; k<=nk2; k++) {
    for(int j=nj1; j<=nj2; j++)
    {
      iptr = j + k*ny; 
      iptr1 = (j-3) + (k-3)*nj; 
      F_thisone->mu_s[iptr] = var_in[iptr1];
    }
  }

  ierr = nc_inq_varid(ncid, "mu_d", &varid); handle_nc_err(ierr);
  ierr = nc_get_vara_float(ncid, varid, start, count, var_in); handle_nc_err(ierr);
  for(int k=nk1; k<=nk2; k++) {
    for(int j=nj1; j<=nj2; j++)
    {
      iptr = j + k*ny; 
      iptr1 = (j-3) + (k-3)*nj; 
      F_thisone->mu_d[iptr] = var_in[iptr1];
    }
  }

  ierr = nc_inq_varid(ncid, "Dc", &varid); handle_nc_err(ierr);
  ierr = nc_get_vara_float(ncid, varid, start, count, var_in); handle_nc_err(ierr);
  for(int k=nk1; k<=nk2; k++) {
    for(int j=nj1; j<=nj2; j++)
    {
      iptr = j + k*ny; 
      iptr1 = (j-3) + (k-3)*nj; 
      F_thisone->Dc[iptr] = var_in[iptr1];
    }
  }

  ierr = nc_inq_varid(ncid, "C0", &varid); handle_nc_err(ierr);
  ierr = nc_get_vara_float(ncid, varid, start, count, var_in); handle_nc_err(ierr);
  for(int k=nk1; k<=nk2; k++) {
    for(int j=nj1; j<=nj2; j++)
    {
      iptr = j + k*ny; 
      iptr1 = (j-3) + (k-3)*nj; 
      F_thisone->C0[iptr] = var_in[iptr1];
    }
  }

  ierr = nc_close(ncid); handle_nc_err(ierr);

  free(var_in);
  
  return 0;
}
