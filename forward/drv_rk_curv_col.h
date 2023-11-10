#ifndef DRV_RK_CURV_COL_H
#define DRV_RK_CURV_COL_H

#include "fd_t.h"
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

int
drv_rk_curv_col_allstep(
  fd_t        *fd,
  gdcurv_t    *gdcurv,
  gd_metric_t *metric,
  md_t        *md,
  par_t       *par,
  bdryfree_t  *bdryfree,
  bdrypml_t   *bdrypml,
  wav_t       *wav,
  mympi_t     *mympi,
  fault_coef_t *fault_coef,
  fault_t     *fault,
  fault_wav_t *fault_wav,
  iorecv_t    *iorecv,
  ioline_t    *ioline,
  iofault_t   *iofault,
  ioslice_t   *ioslice,
  iosnap_t    *iosnap,
  // time
  float dt, int nt_total, float t0,
  char *output_fname_part,
  char *output_dir,
  const int verbose);

#endif
