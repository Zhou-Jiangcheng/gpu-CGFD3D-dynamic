#ifndef ALLOC_H
#define ALLOC_H

#include "constants.h"
#include "fault_info.h"
#include "fault_wav_t.h"
#include "gd_t.h"
#include "fd_t.h"
#include "md_t.h"
#include "wav_t.h"
#include "bdry_t.h"

int init_gdinfo_device(gdcurv_t *gdcurv, gdcurv_t *gdcurv_d);

int 
init_gdcurv_device(gdcurv_t *gdcurv, gdcurv_t *gdcurv_d);

int
init_md_device(md_t *md, md_t *md_d);

int 
init_fd_device(fd_t *fd, fd_device_t *fd_device_d);

int
init_metric_device(gd_metric_t *metric, gd_metric_t *metric_d);

int 
init_fault_coef_device(gdcurv_t *gdcurv, fault_coef_t *FC, fault_coef_t *FC_d);

int 
init_fault_device(gdcurv_t *gdcurv, fault_t *F, fault_t *F_d);

int 
init_fault_wav_device(fault_wav_t *FW, fault_wav_t *FW_d);

int 
init_bdryfree_device(gdcurv_t *gdcurv, bdryfree_t *bdryfree, bdryfree_t *bdryfree_d);

int
init_bdrypml_device(gdcurv_t *gdcurv, bdrypml_t *bdrypml, bdrypml_t *bdrypml_d);

int 
init_bdryexp_device(gdcurv_t *gdcurv, bdryexp_t *bdryexp, bdryexp_t *bdryexp_d);

int 
init_wave_device(wav_t *wav, wav_t *wav_d);

float *
init_PGVAD_device(gdcurv_t *gdcurv);

float *
init_Dis_accu_device(gdcurv_t *gdcurv);

int *
init_neighid_device(int *neighid);

int 
dealloc_gdcurv_device(gdcurv_t gdcurv_d);

int 
dealloc_md_device(md_t md_d);

int
dealloc_metric_device(gd_metric_t metric_d);

int 
dealloc_fd_device(fd_device_t fd_device_d);

int 
dealloc_fault_coef_device(fault_coef_t FC_d);

int 
dealloc_fault_device(fault_t F_d);

int 
dealloc_fault_wav_device(fault_wav_t FW);

int 
dealloc_bdryfree_device(bdryfree_t bdryfree_d);

int 
dealloc_bdrypml_device(bdrypml_t bdrypml_d);

int
dealloc_bdryexp_device(bdryexp_t bdryexp_d);

int 
dealloc_wave_device(wav_t wav_d);

#endif
