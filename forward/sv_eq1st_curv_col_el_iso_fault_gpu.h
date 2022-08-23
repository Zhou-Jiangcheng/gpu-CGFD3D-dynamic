
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
                         const int myid, const int verbose);

__global__ void 
sv_eq1st_curv_col_el_iso_rhs_fault_velo_gpu(
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
                       int idir, int jdir, int kdir);


__global__ void 
sv_eq1st_curv_col_el_iso_rhs_fault_stress_F_gpu(
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
                       int idir, int jdir, int kdir);

__global__ void 
sv_eq1st_curv_col_el_iso_rhs_fault_stress_B_gpu(
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
                       int idir, int jdir, int kdir);
