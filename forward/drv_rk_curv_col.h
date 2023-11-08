#ifndef DRV_RK_CURV_COL_H
#define DRV_RK_CURV_COL_H

#include "fd_t.h"
#include "gd_info.h"
#include "fault_info.h"
#include "mympi_t.h"
#include "gd_t.h"
#include "md_t.h"
#include "wav_t.h"
#include "fault_wav_t.h"
#include "bdry_free.h"
#include "bdry_pml.h"
#include "io_funcs.h"

/*************************************************
 * function prototype
 *************************************************/

void
drv_rk_curv_col_allstep(
  fd_t        *fd,
  gdinfo_t    *gdinfo,
  gd_metric_t *metric,
  md_t      *md,
  bdryfree_t *bdryfree,
  bdrypml_t  *bdrypml,
  wav_t  *wav,
  mympi_t    *mympi,
  fault_coef_t *fault_coef,
  fault_t *fault,
  fault_wav_t *fault_wav,
  iorecv_t   *iorecv,
  ioline_t   *ioline,
  iofault_t  *iofault,
  ioslice_t  *ioslice,
  iosnap_t   *iosnap,
  int imethod,
  // time
  float dt, int nt_total, float t0,
  char *output_fname_part,
  char *output_dir,
  int fault_i_global_indx,
  int io_time_skip,
  int qc_check_nan_num_of_step,
  const int output_all, // qc all var
  const int verbose);

#endif
