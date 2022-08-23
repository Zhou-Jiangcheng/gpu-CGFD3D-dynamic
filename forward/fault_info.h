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
  float *D21_1;
  float *D22_1;
  float *D23_1;
  float *D31_1;
  float *D32_1;
  float *D33_1;

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
  float *lam_f;
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
  float *T0x;
  float *T0y;
  float *T0z;
  float *mu_s;
  float *mu_d;
  float *Dc;
  float *C0;

  float *Tn;
  float *Ts1;
  float *Ts2;
  float *slip;
  float *slip1;
  float *slip2;
  float *Vs;
  float *Vs1;
  float *Vs2;
  float *peak_Vs;
  float *init_t0;

  float *tTn;
  float *tTs1;
  float *tTs2;
  int *united;
  int *faultgrid;
  int *rup_index_y;
  int *rup_index_z;
  int *flag_rup;
  int *init_t0_flag;
} fault_t;

/*************************************************
 * function prototype
 *************************************************/

int
fault_coef_init(fault_coef_t *FC
                gdinfo_t *gdinfo);

int 
fault_coef_cal(gdinfo_t *gdinfo, 
               gdcurv_metric_t *metric, 
               md_t *md, 
               int fault_i_global_index,
               fault_coef_t *FC);

int
fault_init(fault_t *F
           gdinfo_t *gdinfo);

int
fault_set(fault_t *F
          fault_coef_t *FC,
          gdinfo_t *gdinfo,
          int bdry_has_free,
          float *fault_grid,
          char *init_stress_nc);

int 
nc_read_init_stress(fault_t *F, 
                    gdinfo_t *gdinfo, 
                    char *init_stress_nc);

#endif
