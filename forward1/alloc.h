#ifndef ALLOC_H
#define ALLOC_H

#include "constants.h"
#include "gd_info.h"
#include "fault_info.h"
#include "fault_wav_t.h"
#include "gd_t.h"
#include "fd_t.h"
#include "md_t.h"
#include "wav_t.h"
#include "bdry_free.h"
#include "bdry_pml.h"

int
init_gdinfo_device(gdinfo_t *gdinfo, gdinfo_t *gdinfo_d);

int 
init_gdcart_device(gd_t *gdcart, gd_t *gdcart_d);

int 
init_gdcurv_device(gd_t *gdcurv, gd_t *gdcurv_d);

int
init_md_device(md_t *md, md_t *md_d);

int 
init_fd_device(fd_t *fd, fd_device_t *fd_device_d);

int
init_metric_device(gdcurv_metric_t *metric, gdcurv_metric_t *metric_d);

int 
init_fault_coef_device(gdinfo_t *gdinfo, fault_coef_t *FC, fault_coef_t *FC_d);

int 
init_fault_device(gdinfo_t *gdinfo, fault_t *F, fault_t *F_d);

int 
init_fault_wav_device(fault_wav_t *FW, fault_wav_t *FW_d);

int 
init_bdryfree_device(gdinfo_t *gdinfo, bdryfree_t *bdryfree, bdryfree_t *bdryfree_d);

int
init_bdrypml_device(gdinfo_t *gdinfo, bdrypml_t *bdrypml, bdrypml_t *bdrypml_d);

int 
init_wave_device(wav_t *wav, wav_t *wav_d);

float *
init_PGVAD_device(gdinfo_t *gdinfo);

float *
init_Dis_accu_device(gdinfo_t *gdinfo);

int *
init_neighid_device(int *neighid);

int 
dealloc_gdcurv_device(gd_t gdcurv_d);

int 
dealloc_md_device(md_t md_d);

int
dealloc_metric_device(gdcurv_metric_t metric_d);

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
dealloc_wave_device(wav_t wav_d);

#endif
