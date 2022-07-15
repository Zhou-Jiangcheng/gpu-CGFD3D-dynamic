#ifndef FT_INFO_H
#define FT_INFO_H

#include "gd_info.h"
#include "constants.h"

/*************************************************
 * structure
 *************************************************/

typedef struct
{
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
  float *matVx1_free;
  float *matVy1_free;
  float *matVx2_free;
  float *matVy2_free;

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

  float *vec_n; //normal 
  float *vec_s1; //strike
  float *vec_s2; //dip
  float *x_et;
  float *y_et;
  float *z_et;

} fault_coef_t;

typedef struct
{
  float *W;
  float *Ws;
  float *mW;
  float *hW;
  float *tW;
  float *T11;
  float *T12;
  float *T13;
  float *mT11;
  float *mT12;
  float *mT13;
  float *hT11;
  float *hT12;
  float *hT13;
} fault_wav_t;

typedef struct
{
  float *str_init_x;
  float *str_init_y;
  float *str_init_z;
  float *friction;
  float *T0x;
  float *T0y;
  float *T0z;
  float *dT0x;
  float *dT0y;
  float *dT0z;
  float *Tn;
  float *Ts1;
  float *Ts2;
  float *tTn;
  float *tTs1;
  float *tTs2;
  float *mu_s;
  float *mu_d;
  float *Dc;
  float *C0;
  float *slip;
  float *slip1;
  float *slip2;
  float *Vs1;
  float *Vs2;
  float *init_t0;
  int *init_t0_flag;
  int *united;
  int *rup_index_y;
  int *rup_index_z;
  int *Faultgrid;
  int *flag_rup;
  int *first_rup;
  int *faultgrid;
} fault_t;

/*************************************************
 * function prototype
 *************************************************/

void 
fault_coef_cal(gdinfo_t *gdinfo, 
               gdcurv_metric_t *metric, 
               md_t *md, 
               fault_coef_t *FC);

void
fault_coef_init(fault_coef_t *FC
                gdinfo_t *gdinfo);

#endif
