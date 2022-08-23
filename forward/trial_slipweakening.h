
int 
trial_slipweaking_onestage(
                  float *w_cur_d,
                  float *f_cur_d,
                  float *f_pre_d,
                  int i0,
                  int isfree,
                  int dt,
                  gdinfo_t gdinfo_d,
                  gdcurv_metric_t metric_d,
                  wav_t wav_d,
                  fault_wav_t FW,
                  fault_t F,
                  fault_coef_t FC,
                  fd_op_t *fdy_op,
                  fd_op_t *fdz_op,
                  const int myid, const int verbose);

__global__ void 
trial_slipweaking_gpu(
    float *Txx,   float *Tyy,   float *Tzz,
    float *Tyz,   float *Txz,   float *Txy,
    float *f_T2x, float *f_T2y, float *f_T2z,
    float *f_T3x, float *f_T3y, float *f_T3z,
    float *f_T1x, float *f_T1y, float *f_T1z,
    float *f_hVx, float *f_hVy, float *f_hVz,
    float *f_mVx, float *f_mVy, float *f_mVz,
    float *xi_x,  float *xi_y,  float *xi_z,
    float *jac3d, int i0, int isfree, int dt, 
    int nj, int nk, int ny,
    size_t siz_line, size_t siz_slice, 
    size_t siz_slice_yz, int jdir, int kdir,
    fault_t F, fault_coef_t FC);
