#ifndef FT_INFO_H
#define FT_INFO_H

#include "gd_info.h"
#include "constants.h"

typedef struct{

  //Fault coefs
  float *D11_1;
  float *D12_1;
  float *D13_1;
  float *D21_1;
  float *D22_1;
  float *D23_1;
  float *D31_1;
  float *D32_1;
  float *D33_1;
  float *D11_2;
  float *D12_2;
  float *D13_2;
  float *D21_2;
  float *D22_2;
  float *D23_2;
  float *D31_2;
  float *D32_2;
  float *D33_2;

  float *matPlus2Min1; //zhangzhengguo method need
  float *matPlus2Min2;
  float *matPlus2Min3;
  float *matPlus2Min4;
  float *matPlus2Min5;
  float *matMin2Plus1;
  float *matMin2Plus2;
  float *matMin2Plus3;
  float *matMin2Plus4;
  float *matMin2Plus5;

  float *matT1toVx_Min; //zhangwenqiang method need
  float *matVytoVx_Min;
  float *matVztoVx_Min;
  float *matT1toVx_Plus;
  float *matVytoVx_Plus;
  float *matVztoVx_Plus;

  //with free surface
  float *matVx2Vz1;
  float *matVy2Vz1;
  float *matVx2Vz2;
  float *matVy2Vz2;
  float *matVx_free;
  float *matVy_free;

  float *matPlus2Min1f; //zhangzhengguo
  float *matPlus2Min2f;
  float *matPlus2Min3f;
  float *matMin2Plus1f;
  float *matMin2Plus2f;
  float *matMin2Plus3f;

  float *matT1toVxf_Min; //zhangwenqiang
  float *matVytoVxf_Min;
  float *matT1toVxf_Plus;
  float *matVytoVxf_Plus;
  
  // fault split node, + - media 
  float *lambda_f;
  float *mu_f;
  float *rho_f;

  int srci;
  int FaultAtThisNode;
  int *Faultgrid;
  int *united;
  Init_t0;
  MaxRate;
  flag_rup;
  rup_index_z;
  rup_index_y;
  PeakRate;
  Slip;
  hSlip;
  Slipy;
  Slipz;
  Ts1;
  Ts2;
  Tn;
  tTs1;
  tTs2;
  tTn;
  Ux;
  Uy;
  Uz;
  Dc_slip;
  str_peak;
  str_init_x;
  str_init_y;
  init_str_z;
  C0;
  Ratex;
  Ratey;
  Ratez;
  vec_n;
  vec_s1;
  vec_s2;
  x_et;
  y_et;
  z_et;
} fault_info_t


#endif
