#include <stdio.h>
#include <stdlib.h>

__global__
void fault_dvelo_cu(,
    int idir, int jdir, int kdir)
{
  int iy = blockIdx.x * blockDim.x + threadIdx.x;
  int iz = blockIdx.y * blockDim.y + threadIdx.y;
  int nj = gdinfo_d.nj;
  int nk = gdinfo_d.nk;
  int nx = gdinfo_d.nx;
  int ny = gdinfo_d.ny;
  int nz = gdinfo_d.nz;

  // OUTPUT
  size_t siz_line = gdinfo_d.siz_line;
  size_t siz_slice = gdinfo_d.siz_slice;
  size_t siz_volume = gdinfo_d.siz_volume;
  size_t siz_slice_yz = gdinfo_d.siz_slice_yz;
  size_t nyz2 = nyz * 2;

  float *hVx  = w_rhs_d + ;
  float *hVy  = w_rhs_d + ;
  float *hVz  = w_rhs_d + ;
  float *Txx = w_cur_d + wav_d.Txx_pos;
  float *Tyy = w_cur_d + wav_d.Tyy_pos;
  float *Tzz = w_cur_d + wav_d.Tzz_pos;
  float *Txy = w_cur_d + wav_d.Txz_pos;
  float *Txz = w_cur_d + wav_d.Tyz_pos;
  float *Tyz = w_cur_d + wav_d.Txy_pos;


  // INPUT
  real_t *XIX = M + 0 * stride;
  real_t *XIY = M + 1 * stride;
  real_t *XIZ = M + 2 * stride;
  real_t *ETX = M + 3 * stride;
  real_t *ETY = M + 4 * stride;
  real_t *ETZ = M + 5 * stride;
  real_t *ZTX = M + 6 * stride;
  real_t *ZTY = M + 7 * stride;
  real_t *ZTZ = M + 8 * stride;
  real_t *JAC = M + 9 * stride;
  real_t *RHO = M + 12 * stride;

  real_t *f_T21 = F.W + 3 * nyz2;
  real_t *f_T22 = F.W + 4 * nyz2;
  real_t *f_T23 = F.W + 5 * nyz2;
  real_t *f_T31 = F.W + 6 * nyz2;
  real_t *f_T32 = F.W + 7 * nyz2;
  real_t *f_T33 = F.W + 8 * nyz2;

  real_t *f_hVx  = F.hW + 0 * nyz2;
  real_t *f_hVy  = F.hW + 1 * nyz2;
  real_t *f_hVz  = F.hW + 2 * nyz2;


  real_t rrhojac;


  real_t vecT11[7], vecT12[7], vecT13[7];
  real_t vecT21[7], vecT22[7], vecT23[7];
  real_t vecT31[7], vecT32[7], vecT33[7];
  real_t DxTx[3],DyTy[3],DzTz[3];


  int mm, l, n;

  int pos, pos_m;
  int pos0, pos1, pos2;

  int i0 = nx/2;

  if (j < nj && k < nk ) { 
    if(F.united[j + k * nj]) return;
    int km = nk - k; // nk2-3, nk2-2, nk2-1 ==> (3, 2, 1)
    for (i = i0-3; i <= i0+3; i++) {
      //n = i - i0; // -3, -2, -1, +1, +2, +3, not at fault plane
      n = i0 - i; // +3, +2, +1, -1, -2, -3, not at fault plane
      if(n==0) continue; // skip Split nodes

      for (l = -3; l <= 3; l++){

        pos = j1 + k1 * ny + (i+l) * ny * nz;
        vecT11[l+3] = JAC[pos]*(XIX[pos]*w_Txx[pos] + XIY[pos]*w_Txy[pos] + XIZ[pos]*w_Txz[pos]);
        vecT12[l+3] = JAC[pos]*(XIX[pos]*w_Txy[pos] + XIY[pos]*w_Tyy[pos] + XIZ[pos]*w_Tyz[pos]);
        vecT13[l+3] = JAC[pos]*(XIX[pos]*w_Txz[pos] + XIY[pos]*w_Tyz[pos] + XIZ[pos]*w_Tzz[pos]);

        pos = (j1+l) + k1 * ny + i * ny * nz;
        vecT21[l+3] = JAC[pos]*(ETX[pos]*w_Txx[pos] + ETY[pos]*w_Txy[pos] + ETZ[pos]*w_Txz[pos]);
        vecT22[l+3] = JAC[pos]*(ETX[pos]*w_Txy[pos] + ETY[pos]*w_Tyy[pos] + ETZ[pos]*w_Tyz[pos]);
        vecT23[l+3] = JAC[pos]*(ETX[pos]*w_Txz[pos] + ETY[pos]*w_Tyz[pos] + ETZ[pos]*w_Tzz[pos]);

        pos = j1 + (k1+l) * ny + i * ny * nz;
        vecT31[l+3] = JAC[pos]*(ZTX[pos]*w_Txx[pos] + ZTY[pos]*w_Txy[pos] + ZTZ[pos]*w_Txz[pos]);
        vecT32[l+3] = JAC[pos]*(ZTX[pos]*w_Txy[pos] + ZTY[pos]*w_Tyy[pos] + ZTZ[pos]*w_Tyz[pos]);
        vecT33[l+3] = JAC[pos]*(ZTX[pos]*w_Txz[pos] + ZTY[pos]*w_Tyz[pos] + ZTZ[pos]*w_Tzz[pos]);
      }

      //pos = j1 + k1 * ny + 1 * ny * nz;
      pos = j1 + k1 * ny + 3 * ny * nz;
      vecT11[n+3] = F.T11[pos];// F->T11[1][j][k];
      vecT12[n+3] = F.T12[pos];// F->T12[1][j][k];
      vecT13[n+3] = F.T13[pos];// F->T13[1][j][k];

//#ifdef TractionLow
      // reduce order
      if(abs(n)==1){
        DxTx[0] = vec_L22(vecT11,3,FlagX)*rDH;
        DxTx[1] = vec_L22(vecT12,3,FlagX)*rDH;
        DxTx[2] = vec_L22(vecT13,3,FlagX)*rDH;
      }else if(abs(n)==2){
        DxTx[0] = vec_L24(vecT11,3,FlagX)*rDH;
        DxTx[1] = vec_L24(vecT12,3,FlagX)*rDH;
        DxTx[2] = vec_L24(vecT13,3,FlagX)*rDH;
      }else if(abs(n)==3){
        DxTx[0] = vec_L(vecT11,3,FlagX)*rDH;
        DxTx[1] = vec_L(vecT12,3,FlagX)*rDH;
        DxTx[2] = vec_L(vecT13,3,FlagX)*rDH;
      }
//#endif
#ifdef TractionImg
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

      DxTx[0] = vec_L(vecT11,3,FlagX)*rDH;
      DxTx[1] = vec_L(vecT12,3,FlagX)*rDH;
      DxTx[2] = vec_L(vecT13,3,FlagX)*rDH;
#endif
      if(par.freenode && km<=3){
        vecT31[km+2] = 0.0;
        vecT32[km+2] = 0.0;
        vecT33[km+2] = 0.0;
        for (l = km+3; l<7; l++){
          vecT31[l] = -vecT31[2*(km+2)-l];
          vecT32[l] = -vecT32[2*(km+2)-l];
          vecT33[l] = -vecT33[2*(km+2)-l];
        }
      }

      DyTy[0] = vec_L(vecT21,3,FlagY)*rDH;
      DyTy[1] = vec_L(vecT22,3,FlagY)*rDH;
      DyTy[2] = vec_L(vecT23,3,FlagY)*rDH;
      DzTz[0] = vec_L(vecT31,3,FlagZ)*rDH;
      DzTz[1] = vec_L(vecT32,3,FlagZ)*rDH;
      DzTz[2] = vec_L(vecT33,3,FlagZ)*rDH;

      pos = j1 + k1 * ny + i * ny * nz;

      rrhojac = 1.0 / (RHO[pos] * JAC[pos]);
      w_hVx[pos] = (DxTx[0]+DyTy[0]+DzTz[0])*rrhojac;
      w_hVy[pos] = (DxTx[1]+DyTy[1]+DzTz[1])*rrhojac;
      w_hVz[pos] = (DxTx[2]+DyTy[2]+DzTz[2])*rrhojac;

    } // end of loop i

    // update velocity at the fault plane
    // 0 for minus side on the fault
    // 1 for plus  side on the fault
    for (mm = 0; mm < 2; mm++){
      //km = NZ -(thisid[2]*nk+k-3);
      //rrhojac = F->rrhojac_f[mm][j][k];

//#ifdef TractionLow
      pos0 = j1 + k1 * ny + (3-1) * ny * nz;
      pos1 = j1 + k1 * ny + (3  ) * ny * nz;
      pos2 = j1 + k1 * ny + (3+1) * ny * nz;
      if(mm==0){
        DxTx[0] = (F.T11[pos1] - F.T11[pos0])*rDH;
        DxTx[1] = (F.T12[pos1] - F.T12[pos0])*rDH;
        DxTx[2] = (F.T13[pos1] - F.T13[pos0])*rDH;
      }else{
        DxTx[0] = (F.T11[pos2] - F.T11[pos1])*rDH;
        DxTx[1] = (F.T12[pos2] - F.T12[pos1])*rDH;
        DxTx[2] = (F.T13[pos2] - F.T13[pos1])*rDH;
      }
//#endif
#ifdef TractionImg
      real_t a0p,a0m;
      if(FlagX==FWD){
        a0p = a_0pF;
        a0m = a_0mF;
      }else{
        a0p = a_0pB;
        a0m = a_0mB;
      }
      if(mm==0){ // "-" side
        DxTx[0] = rDH*(
            a0m*F.T11[j1+k1*ny+3*ny*nz] -
            a_1*F.T11[j1+k1*ny+2*ny*nz] -
            a_2*F.T11[j1+k1*ny+1*ny*nz] -
            a_3*F.T11[j1+k1*ny+0*ny*nz] );
        DxTx[1] = rDH*(
            a0m*F.T12[j1+k1*ny+3*ny*nz] -
            a_1*F.T12[j1+k1*ny+2*ny*nz] -
            a_2*F.T12[j1+k1*ny+1*ny*nz] -
            a_3*F.T12[j1+k1*ny+0*ny*nz] );
        DxTx[2] = rDH*(
            a0m*F.T13[j1+k1*ny+3*ny*nz] -
            a_1*F.T13[j1+k1*ny+2*ny*nz] -
            a_2*F.T13[j1+k1*ny+1*ny*nz] -
            a_3*F.T13[j1+k1*ny+0*ny*nz] );
      }else{ // "+" side
        DxTx[0] = rDH*(
            a0p*F.T11[j1+k1*ny+3*ny*nz] +
            a_1*F.T11[j1+k1*ny+4*ny*nz] +
            a_2*F.T11[j1+k1*ny+5*ny*nz] +
            a_3*F.T11[j1+k1*ny+6*ny*nz] );
        DxTx[1] = rDH*(
            a0p*F.T12[j1+k1*ny+3*ny*nz] +
            a_1*F.T12[j1+k1*ny+4*ny*nz] +
            a_2*F.T12[j1+k1*ny+5*ny*nz] +
            a_3*F.T12[j1+k1*ny+6*ny*nz] );
        DxTx[2] = rDH*(
            a0p*F.T13[j1+k1*ny+3*ny*nz] +
            a_1*F.T13[j1+k1*ny+4*ny*nz] +
            a_2*F.T13[j1+k1*ny+5*ny*nz] +
            a_3*F.T13[j1+k1*ny+6*ny*nz] );
      }
#endif

      for (l = -3; l <= 3; l++){
        pos = (j1+l) + k1 * ny + mm * ny * nz;
        vecT21[l+3] = f_T21[pos];
        vecT22[l+3] = f_T22[pos];
        vecT23[l+3] = f_T23[pos];
        pos = j1 + (k1+l) * ny + mm * ny * nz;
        vecT31[l+3] = f_T31[pos];
        vecT32[l+3] = f_T32[pos];
        vecT33[l+3] = f_T33[pos];
      }

      if(par.freenode && km<=3){
        vecT31[km+2] = 0.0;
        vecT32[km+2] = 0.0;
        vecT33[km+2] = 0.0;
        for (l = km+3; l<7; l++){
          vecT31[l] = -vecT31[2*(km+2)-l];
          vecT32[l] = -vecT32[2*(km+2)-l];
          vecT33[l] = -vecT33[2*(km+2)-l];
        }

        DzTz[0] = vec_L(vecT31,3,FlagZ)*rDH;
        DzTz[1] = vec_L(vecT32,3,FlagZ)*rDH;
        DzTz[2] = vec_L(vecT33,3,FlagZ)*rDH;

      }else{
        if(F.rup_index_z[j + k * nj] % 7){
          DzTz[0] = vec_L22(vecT31,3,FlagZ)*rDH;
          DzTz[1] = vec_L22(vecT32,3,FlagZ)*rDH;
          DzTz[2] = vec_L22(vecT33,3,FlagZ)*rDH;
        }else{
          DzTz[0] = vec_L(vecT31,3,FlagZ)*rDH;
          DzTz[1] = vec_L(vecT32,3,FlagZ)*rDH;
          DzTz[2] = vec_L(vecT33,3,FlagZ)*rDH;
        }
      }

      if(F.rup_index_y[j + k * nj] % 7){
        DyTy[0] = vec_L22(vecT21,3,FlagY)*rDH;
        DyTy[1] = vec_L22(vecT22,3,FlagY)*rDH;
        DyTy[2] = vec_L22(vecT23,3,FlagY)*rDH;
      }else{
        DyTy[0] = vec_L(vecT21,3,FlagY)*rDH;
        DyTy[1] = vec_L(vecT22,3,FlagY)*rDH;
        DyTy[2] = vec_L(vecT23,3,FlagY)*rDH;
      }

      pos_m = j1 + k1 * ny + i0 * ny * nz;
      pos = j1 + k1 * ny + mm * ny * nz;
      rrhojac = 1.0 / (F.rho_f[pos] * JAC[pos_m]);

      pos = j1 + k1 * ny + mm * ny * nz; // mm = 0, 1
      f_hVx[pos] = (DxTx[0]+DyTy[0]+DzTz[0])*rrhojac;
      f_hVy[pos] = (DxTx[1]+DyTy[1]+DzTz[1])*rrhojac;
      f_hVz[pos] = (DxTx[2]+DyTy[2]+DzTz[2])*rrhojac;
    } // end of loop mm  update fault plane

  } // end j k
  return;
}

void fault_dvelo(Wave W, Fault F, realptr_t M,
    int FlagX, int FlagY, int FlagZ)
{
  dim3 block(16, 8, 1);
  dim3 grid(
      (hostParams.nj + block.x - 1) / block.x,
      (hostParams.nk + block.y - 1) / block.y,
      1);
  fault_dvelo_cu <<<grid, block>>> (W, F, M, FlagX, FlagY, FlagZ);
  return;
}

