/*******************************************************************************
 * solver of isotropic elastic 1st-order eqn using curv grid and macdrp schem
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "fdlib_mem.h"
#include "fdlib_math.h"
#include "blk_t.h"
#include "drv_rk_curv_col.h"
#include "sv_curv_col_el_iso_gpu.h"
#include "sv_curv_col_el_iso_fault_gpu.h"
#include "trial_slipweakening.h"
#include "transform.h"
#include "fault_wav_t.h"
#include "alloc.h"
#include "cuda_common.h"

/*******************************************************************************
 * one simulation over all time steps, could be used in imaging or inversion
 *  simple MPI exchange without computing-communication overlapping
 ******************************************************************************/

int
drv_rk_curv_col_allstep(
  fd_t         *fd,
  gdcurv_t     *gdcurv,
  gd_metric_t  *metric,
  md_t         *md,
  par_t        *par,
  bdryfree_t   *bdryfree,
  bdrypml_t    *bdrypml,
  bdryexp_t    *bdryexp,
  wav_t        *wav,
  mympi_t      *mympi,
  fault_coef_t *fault_coef,
  fault_t      *fault,
  fault_wav_t  *fault_wav,
  iorecv_t     *iorecv,
  ioline_t     *ioline,
  iofault_t    *iofault,
  ioslice_t    *ioslice,
  iosnap_t     *iosnap,
  // time
  float dt, int nt_total, float t0,
  char *output_fname_part,
  char *output_dir,
  const int verbose)
{
  int imethod = par->imethod;
  int io_time_skip = par->io_time_skip;
  int qc_check_nan_number_of_step = par->qc_check_nan_number_of_step;

  int num_rk_stages = fd->num_rk_stages;
  int num_of_pairs =  fd->num_of_pairs;
  float *rk_a = fd->rk_a;
  float *rk_b = fd->rk_b;
  
  //gdcurv
  int ni = gdcurv->ni;
  int nj = gdcurv->nj;
  int nk = gdcurv->nk;
  // fault x index with ghost
  int i0 = par->fault_x_index[0] + gdcurv->fdx_nghosts;
  // mpi
  int myid = mympi->myid;
  int *topoid = mympi->topoid;
  MPI_Comm comm = mympi->comm;
  int *neighid_d = init_neighid_device(mympi->neighid);

  gdcurv_t     gdcurv_d;
  md_t         md_d;
  wav_t        wav_d;
  fd_device_t  fd_device_d;
  gd_metric_t  metric_d;
  fault_coef_t fault_coef_d;
  fault_t      fault_d;
  fault_wav_t  fault_wav_d;
  bdryfree_t   bdryfree_d;
  bdrypml_t    bdrypml_d;
  bdryexp_t    bdryexp_d;

  // init device struct, and copy data from host to device
  init_gdcurv_device(gdcurv, &gdcurv_d);
  init_md_device(md, &md_d);
  init_fd_device(fd, &fd_device_d);
  init_metric_device(metric, &metric_d);
  init_wave_device(wav, &wav_d);
  init_fault_coef_device(gdcurv, fault_coef, &fault_coef_d);
  init_fault_device(gdcurv, fault, &fault_d);
  init_fault_wav_device(fault_wav, &fault_wav_d);
  init_bdryfree_device(gdcurv, bdryfree, &bdryfree_d);
  init_bdrypml_device(gdcurv, bdrypml, &bdrypml_d);
  init_bdryexp_device(gdcurv, bdryexp, &bdryexp_d);

  // get device wavefield 
  float *w_buff = wav->v5d; // size number is V->siz_icmp * (V->ncmp+6)
  // GPU local pointer
  float *w_cur_d;
  float *w_pre_d;
  float *w_rhs_d;
  float *w_end_d;
  float *w_tmp_d;

  float *f_cur_d;
  float *f_pre_d;
  float *f_rhs_d;
  float *f_end_d;
  float *f_tmp_d;

  // get wavefield
  w_pre_d = wav_d.v5d + wav_d.siz_ilevel * 0; // previous level at n
  w_tmp_d = wav_d.v5d + wav_d.siz_ilevel * 1; // intermidate value
  w_rhs_d = wav_d.v5d + wav_d.siz_ilevel * 2; // for rhs
  w_end_d = wav_d.v5d + wav_d.siz_ilevel * 3; // end level at n+1

  // get fault wavefield
  f_pre_d = fault_wav_d.v5d + fault_wav_d.siz_ilevel * 0; // previous level at n
  f_tmp_d = fault_wav_d.v5d + fault_wav_d.siz_ilevel * 1; // intermidate value
  f_rhs_d = fault_wav_d.v5d + fault_wav_d.siz_ilevel * 2; // for rhs
  f_end_d = fault_wav_d.v5d + fault_wav_d.siz_ilevel * 3; // end level at n+1

  int   ipair, istage;
  float t_cur;
  float t_end; // time after this loop for nc output
  // for mpi message
  int   ipair_mpi, istage_mpi;
  // create fault slice nc output files
  if (myid==0 && verbose>0) fprintf(stdout,"prepare fault slice nc output ...\n"); 
  iofault_nc_t iofault_nc;
  io_fault_nc_create(iofault,
                     gdcurv->ni, gdcurv->nj, gdcurv->nk, topoid,
                     &iofault_nc);
  // create slice nc output files
  if (myid==0 && verbose>0) fprintf(stdout,"prepare slice nc output ...\n"); 
  ioslice_nc_t ioslice_nc;
  io_slice_nc_create(ioslice, wav->ncmp, wav->cmp_name,
                     gdcurv->ni, gdcurv->nj, gdcurv->nk, topoid,
                     &ioslice_nc);
  // create snapshot nc output files
  if (myid==0 && verbose>0) fprintf(stdout,"prepare snap nc output ...\n"); 
  iosnap_nc_t  iosnap_nc;
  io_snap_nc_create(iosnap, &iosnap_nc, topoid);

  // only y/z mpi
  int num_of_r_reqs = 8;
  int num_of_s_reqs = 8;

  // set pml for rk
  if(bdrypml_d.is_enable_pml == 1)
  {
    for (int idim=0; idim<CONST_NDIM; idim++) {
      for (int iside=0; iside<2; iside++) {
        if (bdrypml_d.is_sides_pml[idim][iside]==1) {
          bdrypml_auxvar_t *auxvar_d = &(bdrypml_d.auxvar[idim][iside]);
          auxvar_d->pre = auxvar_d->var + auxvar_d->siz_ilevel * 0;
          auxvar_d->tmp = auxvar_d->var + auxvar_d->siz_ilevel * 1;
          auxvar_d->rhs = auxvar_d->var + auxvar_d->siz_ilevel * 2;
          auxvar_d->end = auxvar_d->var + auxvar_d->siz_ilevel * 3;
        }
      }
    }
  }

  int isfree = bdryfree_d.is_sides_free[CONST_NDIM-1][1];
  // alloc free surface PGV, PGA and PGD
  float *PG_d = NULL;
  float *PG   = NULL;
  // Dis_accu is Displacemen accumulation, be uesd for PGD calculaton.
  float *Dis_accu_d   = NULL;
  if (isfree == 1)
  {
    PG_d = init_PGVAD_device(gdcurv);
    Dis_accu_d = init_Dis_accu_device(gdcurv);
    PG = (float *) fdlib_mem_calloc_1d_float(CONST_NDIM_5*gdcurv->ny*gdcurv->nx,0.0,"PGV,A,D malloc");
  }
  // calculate conversion matrix for free surface
  if (isfree == 1)
  {
    if (md_d.medium_type == CONST_MEDIUM_ELASTIC_ISO)
    {
      dim3 block(16,16);
      dim3 grid;
      grid.x = (ni+block.x-1)/block.x;
      grid.y = (nj+block.y-1)/block.y;
      sv_curv_col_el_iso_dvh2dvz_gpu <<<grid, block>>> (gdcurv_d,metric_d,md_d,bdryfree_d,verbose);
      CUDACHECK(cudaDeviceSynchronize());
    }
    else
    {
      fprintf(stderr,"ERROR: conversion matrix for medium_type=%d is not implemented\n",
                    md->medium_type);
      MPI_Abort(MPI_COMM_WORLD,1);
    }
  }

  //--------------------------------------------------------
  // time loop
  //--------------------------------------------------------

  if (myid==0 && verbose>0) fprintf(stdout,"start time loop ...\n"); 

  for (int it=0; it<nt_total; it++)
  {
    t_cur = it * dt + t0;
    t_end = t_cur +dt;

    if (myid==0 && verbose>10) fprintf(stdout,"-> it=%d, t=%f\n", it, t_cur);

    // mod to get ipair
    ipair = it % num_of_pairs;
    if (myid==0 && verbose>10) fprintf(stdout, " --> ipair=%d\n",ipair);

    // loop RK stages for one step
    for (istage=0; istage<num_rk_stages; istage++)
    {
      if (myid==0 && verbose>10) fprintf(stdout, " --> istage=%d\n",istage);

      // for mesg
      if (istage != num_rk_stages-1) {
        ipair_mpi = ipair;
        istage_mpi = istage + 1;
      } else {
        ipair_mpi = (it + 1) % num_of_pairs;
        istage_mpi = 0; 
      }

      // use pointer to avoid 1 copy for previous level value
      if (istage==0) {
        w_cur_d = w_pre_d;
        f_cur_d = f_pre_d;
        if(bdrypml_d.is_enable_pml == 1)
        {
          for (int idim=0; idim<CONST_NDIM; idim++) {
            for (int iside=0; iside<2; iside++) {
              bdrypml_d.auxvar[idim][iside].cur = bdrypml_d.auxvar[idim][iside].pre;
            }
          }
        }
      }
      else
      {
        w_cur_d = w_tmp_d;
        f_cur_d = f_tmp_d;
        if(bdrypml_d.is_enable_pml == 1)
        {
          for (int idim=0; idim<CONST_NDIM; idim++) {
            for (int iside=0; iside<2; iside++) {
              bdrypml_d.auxvar[idim][iside].cur = bdrypml_d.auxvar[idim][iside].tmp;
            }
          }
        }
      }

      // compute rhs
      switch (md_d.medium_type)
      {
        case CONST_MEDIUM_ELASTIC_ISO : {

          wave2fault_onestage(
                        w_cur_d, w_rhs_d, wav_d, 
                        f_cur_d, f_rhs_d, fault_wav_d,
                        i0, fault_d, metric_d, gdcurv_d);

          trial_slipweakening_onestage(
                        w_cur_d, f_cur_d, f_pre_d, 
                        i0, isfree, dt,
                        gdcurv_d, metric_d, wav_d, 
                        fault_wav_d, fault_d, fault_coef_d,
                        fd->pair_fdy_op[ipair][istage],
                        fd->pair_fdz_op[ipair][istage],
                        myid, verbose);

          fault2wave_onestage(
                        w_cur_d, wav_d, 
                        f_cur_d, fault_wav_d,
                        i0, fault_d, metric_d, gdcurv_d);

          sv_curv_col_el_iso_onestage(
                        w_cur_d, w_rhs_d, wav_d, gdcurv_d, fd_device_d, 
                        metric_d, md_d, bdryfree_d, bdrypml_d, 
                        fd->pair_fdx_op[ipair][istage],
                        fd->pair_fdy_op[ipair][istage],
                        fd->pair_fdz_op[ipair][istage],
                        myid, verbose);

          sv_curv_col_el_iso_fault_onestage(
                        w_cur_d, w_rhs_d, f_cur_d, f_rhs_d,
                        i0, isfree, imethod, wav_d, 
                        fault_wav_d, fault_d, fault_coef_d,
                        gdcurv_d, metric_d, md_d, bdryfree_d,  
                        fd->pair_fdx_op[ipair][istage],
                        fd->pair_fdy_op[ipair][istage],
                        fd->pair_fdz_op[ipair][istage],
                        myid, verbose);

          break;
        }
      //  synchronize onestage device func.
      CUDACHECK(cudaDeviceSynchronize());
      }

      // recv mesg
      MPI_Startall(num_of_r_reqs, mympi->pair_r_reqs[ipair_mpi][istage_mpi]);

      // rk start
      if (istage==0)
      {
        float coef_a = rk_a[istage] * dt;
        float coef_b = rk_b[istage] * dt;

        // temp wavefield
        {
          dim3 block(256);
          dim3 grid;
          grid.x = (wav_d.siz_ilevel + block.x - 1) / block.x;
          wav_update <<<grid, block>>> (wav_d.siz_ilevel, coef_a, w_tmp_d, w_pre_d, w_rhs_d);
        }
        {
          dim3 block(4,8,8);
          dim3 grid;
          grid.x = (2*fault_wav->ncmp + block.x - 1) / block.x;
          grid.y = (nj + block.y - 1) / block.y;
          grid.z = (nk + block.z - 1) / block.z;
          fault_wav_update <<<grid, block>>> (gdcurv_d, fault_wav->ncmp, coef_a, 
                                              fault_d, f_tmp_d, f_pre_d, f_rhs_d);

          fault2wave_onestage(
                        w_tmp_d, wav_d, 
                        f_tmp_d, fault_wav_d,
                        i0, fault_d, metric_d, gdcurv_d);
        }

        // pack and isend
        blk_macdrp_pack_mesg_gpu(w_tmp_d, fd, gdcurv, mympi, ipair_mpi, istage_mpi, wav->ncmp, myid);
        blk_macdrp_pack_fault_mesg_gpu(f_tmp_d, fd, gdcurv, mympi, ipair_mpi, istage_mpi, fault_wav->ncmp, myid);

        MPI_Startall(num_of_s_reqs, mympi->pair_s_reqs[ipair_mpi][istage_mpi]);
        
        // pml_tmp
        if(bdrypml_d.is_enable_pml == 1)
        {
          for (int idim=0; idim<CONST_NDIM; idim++) {
            for (int iside=0; iside<2; iside++) {
              if (bdrypml_d.is_sides_pml[idim][iside]==1) {
                bdrypml_auxvar_t *auxvar_d = &(bdrypml_d.auxvar[idim][iside]);
                dim3 block(256);
                dim3 grid;
                grid.x = (auxvar_d->siz_ilevel + block.x - 1) / block.x;
                wav_update <<<grid, block>>> (
                           auxvar_d->siz_ilevel, coef_a, auxvar_d->tmp, auxvar_d->pre, auxvar_d->rhs);
              }
            }
          }
        }
        // w_end
        {
          dim3 block(256);
          dim3 grid;
          grid.x = (wav_d.siz_ilevel + block.x - 1) / block.x;
          wav_update <<<grid, block>>> (wav_d.siz_ilevel, coef_b, w_end_d, w_pre_d, w_rhs_d);
        }
        {
          dim3 block(4,8,8);
          dim3 grid;
          grid.x = (2*fault_wav->ncmp + block.x - 1) / block.x;
          grid.y = (nj + block.y - 1) / block.y;
          grid.z = (nk + block.z - 1) / block.z;
          fault_wav_update <<<grid, block>>> (gdcurv_d, fault_wav->ncmp, coef_b, 
                                              fault_d, f_end_d, f_pre_d, f_rhs_d);
        }
        {
          dim3 block(8,8);
          dim3 grid;
          grid.x = (nj + block.x - 1) / block.x;
          grid.y = (nk + block.y - 1) / block.y;

          float coef = coef_b / dt;
          fault_stress_update_first <<<grid, block>>> (nj, nk, coef, fault_d);
        }
        // pml_end
        if(bdrypml_d.is_enable_pml == 1)
        {
          for (int idim=0; idim<CONST_NDIM; idim++) {
            for (int iside=0; iside<2; iside++) {
              if (bdrypml_d.is_sides_pml[idim][iside]==1) {
                bdrypml_auxvar_t *auxvar_d = &(bdrypml_d.auxvar[idim][iside]);
                dim3 block(256);
                dim3 grid;
                grid.x = (auxvar_d->siz_ilevel + block.x - 1) / block.x;
                wav_update <<<grid, block>>> (
                            auxvar_d->siz_ilevel, coef_b, auxvar_d->end, auxvar_d->pre, auxvar_d->rhs);
              }
            }
          }
        }
      }
      else if (istage<num_rk_stages-1)
      {
        float coef_a = rk_a[istage] * dt;
        float coef_b = rk_b[istage] * dt;
        //temp wavefield
        {
          dim3 block(256);
          dim3 grid;
          grid.x = (wav_d.siz_ilevel + block.x - 1) / block.x;
          wav_update <<<grid, block>>> (wav_d.siz_ilevel, coef_a, w_tmp_d, w_pre_d, w_rhs_d);
          //CUDACHECK(cudaDeviceSynchronize());
        }
        {
          dim3 block(4,8,8);
          dim3 grid;
          grid.x = (2*fault_wav->ncmp + block.x - 1) / block.x;
          grid.y = (nj + block.y - 1) / block.y;
          grid.z = (nk + block.z - 1) / block.z;
          fault_wav_update <<<grid, block>>> (gdcurv_d, fault_wav->ncmp, coef_a, 
                                              fault_d, f_tmp_d, f_pre_d, f_rhs_d);
          fault2wave_onestage(
                        w_tmp_d, wav_d, 
                        f_tmp_d, fault_wav_d,
                        i0, fault_d, metric_d, gdcurv_d);
        }

        // pack and isend
        blk_macdrp_pack_mesg_gpu(w_tmp_d, fd, gdcurv, mympi, ipair_mpi, istage_mpi, wav->ncmp, myid);
        blk_macdrp_pack_fault_mesg_gpu(f_tmp_d, fd, gdcurv, mympi, ipair_mpi, istage_mpi, fault_wav->ncmp, myid);
        MPI_Startall(num_of_s_reqs, mympi->pair_s_reqs[ipair_mpi][istage_mpi]);
        // pml_tmp
        if(bdrypml_d.is_enable_pml == 1)
        {
          for (int idim=0; idim<CONST_NDIM; idim++) {
            for (int iside=0; iside<2; iside++) {
              if (bdrypml_d.is_sides_pml[idim][iside]==1) {
                bdrypml_auxvar_t *auxvar_d = &(bdrypml_d.auxvar[idim][iside]);
                dim3 block(256);
                dim3 grid;
                grid.x = (auxvar_d->siz_ilevel + block.x - 1) / block.x;
                wav_update <<<grid, block>>> (
                           auxvar_d->siz_ilevel, coef_a, auxvar_d->tmp, auxvar_d->pre, auxvar_d->rhs);
              }
            }
          }
        }
        // w_end
        {
          dim3 block(256);
          dim3 grid;
          grid.x = (wav_d.siz_ilevel + block.x - 1) / block.x;
          wav_update_end <<<grid, block>>> (wav_d.siz_ilevel, coef_b, w_end_d, w_rhs_d);
        }
        {
          dim3 block(4,8,8);
          dim3 grid;
          grid.x = (2*fault_wav->ncmp + block.x - 1) / block.x;
          grid.y = (nj + block.y - 1) / block.y;
          grid.z = (nk + block.z - 1) / block.z;
          fault_wav_update_end <<<grid, block>>> (gdcurv_d, fault_wav->ncmp, coef_b, 
                                                  fault_d, f_end_d, f_rhs_d);
        }
        {
          dim3 block(8,8);
          dim3 grid;
          grid.x = (nj + block.x - 1) / block.x;
          grid.y = (nk + block.y - 1) / block.y;

          float coef = coef_b / dt;
          fault_stress_update <<<grid, block>>> (nj, nk, coef, fault_d);
        }
        // pml_end
        if(bdrypml_d.is_enable_pml == 1)
        {
          for (int idim=0; idim<CONST_NDIM; idim++) {
            for (int iside=0; iside<2; iside++) {
              if (bdrypml_d.is_sides_pml[idim][iside]==1) {
                bdrypml_auxvar_t *auxvar_d = &(bdrypml_d.auxvar[idim][iside]);
                dim3 block(256);
                dim3 grid;
                grid.x = (auxvar_d->siz_ilevel + block.x - 1) / block.x;
                wav_update_end <<<grid, block>>> (
                           auxvar_d->siz_ilevel, coef_b, auxvar_d->end, auxvar_d->rhs);
              }
            }
          }
        }
      }
      else // last stage
      {
        float coef_b = rk_b[istage] * dt;

        // w_end
        {
          dim3 block(256);
          dim3 grid;
          grid.x = (wav_d.siz_ilevel + block.x - 1) / block.x;
          wav_update_end <<<grid, block>>>(wav_d.siz_ilevel, coef_b, w_end_d, w_rhs_d);
        }
        {
          dim3 block(4,8,8);
          dim3 grid;
          grid.x = (2*fault_wav->ncmp + block.x - 1) / block.x;
          grid.y = (nj + block.y - 1) / block.y;
          grid.z = (nk + block.z - 1) / block.z;
          fault_wav_update_end <<<grid, block>>> (gdcurv_d, fault_wav->ncmp, coef_b, 
                                                  fault_d, f_end_d, f_rhs_d);
          fault2wave_onestage(
                        w_end_d, wav_d, 
                        f_end_d, fault_wav_d,
                        i0, fault_d, metric_d, gdcurv_d);
        }
        {
          dim3 block(8,8);
          dim3 grid;
          grid.x = (nj + block.x - 1) / block.x;
          grid.y = (nk + block.y - 1) / block.y;

          float coef = coef_b / dt;
          fault_stress_update <<<grid, block>>> (nj, nk, coef, fault_d);
        }
        
        // pack and isend
        blk_macdrp_pack_mesg_gpu(w_end_d, fd, gdcurv, mympi, ipair_mpi, istage_mpi, wav->ncmp, myid);
        blk_macdrp_pack_fault_mesg_gpu(f_end_d, fd, gdcurv, mympi, ipair_mpi, istage_mpi, fault_wav->ncmp, myid);
        MPI_Startall(num_of_s_reqs, mympi->pair_s_reqs[ipair_mpi][istage_mpi]);
        // pml_end
        if(bdrypml_d.is_enable_pml == 1)
        {
          for (int idim=0; idim<CONST_NDIM; idim++) {
            for (int iside=0; iside<2; iside++) {
              if (bdrypml->is_sides_pml[idim][iside]==1) {
                bdrypml_auxvar_t *auxvar_d = &(bdrypml_d.auxvar[idim][iside]);
                dim3 block(256);
                dim3 grid;
                grid.x = (auxvar_d->siz_ilevel + block.x - 1) / block.x;
                wav_update_end <<<grid, block>>> (
                           auxvar_d->siz_ilevel, coef_b, auxvar_d->end, auxvar_d->rhs);
              }
            }
          }
        }
      }

      MPI_Waitall(num_of_s_reqs, mympi->pair_s_reqs[ipair_mpi][istage_mpi], MPI_STATUS_IGNORE);
      MPI_Waitall(num_of_r_reqs, mympi->pair_r_reqs[ipair_mpi][istage_mpi], MPI_STATUS_IGNORE);
 
      if (istage != num_rk_stages-1) 
      {
        blk_macdrp_unpack_mesg_gpu(w_tmp_d, fd, gdcurv, mympi, ipair_mpi, istage_mpi, wav->ncmp, neighid_d);
        blk_macdrp_unpack_fault_mesg_gpu(f_tmp_d, fd, gdcurv, mympi, ipair_mpi, istage_mpi, fault_wav->ncmp, neighid_d);
      } else 
      {
        blk_macdrp_unpack_mesg_gpu(w_end_d, fd, gdcurv, mympi, ipair_mpi, istage_mpi, wav->ncmp,neighid_d);
        blk_macdrp_unpack_fault_mesg_gpu(f_end_d, fd, gdcurv, mympi, ipair_mpi, istage_mpi, fault_wav->ncmp, neighid_d);
      }
    } // RK stages

    //--------------------------------------------
    // QC
    //--------------------------------------------
    if (qc_check_nan_number_of_step >0  && (it % qc_check_nan_number_of_step) == 0) {
      if (myid==0 && verbose>10) fprintf(stdout,"-> check value nan\n");
        //wav_check_value(w_end);
    }
    //--------------------------------------------
    if (bdryexp_d.is_enable_ablexp == 1) {
      bdry_ablexp_apply(bdryexp_d, gdcurv, w_end_d, wav->ncmp);
    }

    //--------------------------------------------
    // save results
    //--------------------------------------------
    // calculate PGV, PGA and PGD for surface 
    if (isfree == 1)
    {
      dim3 block(8,8);
      dim3 grid;
      grid.x = (ni + block.x - 1) / block.x;
      grid.y = (nj + block.y - 1) / block.y;
      PG_calcu_gpu<<<grid, block>>> (w_end_d, w_pre_d, gdcurv_d, PG_d, Dis_accu_d, dt);
    }

    // calculate fault slip, Vs, ... at each dt  
    fault_var_update(f_end_d, it, dt, gdcurv_d, fault_d, fault_coef_d, fault_wav_d);

    //-- recv by interp
    io_recv_keep(iorecv, w_end_d, w_buff, it, wav->ncmp, wav->siz_icmp);

    //-- line values
    io_line_keep(ioline, w_end_d, w_buff, it, wav->ncmp, wav->siz_icmp);
    if((it+1)%io_time_skip == 0)
    {
      int it_skip = (int)(it/io_time_skip);
      // io fault var each dt, use w_buff as buff
      io_fault_nc_put(&iofault_nc, gdcurv, fault_d, w_buff, it_skip, t_end);
      // write slice, use w_buff as buff
      io_slice_nc_put(ioslice,&ioslice_nc,gdcurv,w_end_d,w_buff,it_skip,t_end,0,wav->ncmp-1);
    }
    // snapshot
    io_snap_nc_put(iosnap, &iosnap_nc, gdcurv, md, wav, 
                   w_end_d, w_buff, nt_total, it, t_end, 1,1,1);

    // swap w_pre and w_end pointer, avoid copying
    w_cur_d = w_pre_d; w_pre_d = w_end_d; w_end_d = w_cur_d;
    f_cur_d = f_pre_d; f_pre_d = f_end_d; f_end_d = f_cur_d;

    if(bdrypml_d.is_enable_pml == 1)
    {
      for (int idim=0; idim<CONST_NDIM; idim++) {
        for (int iside=0; iside<2; iside++) {
          bdrypml_auxvar_t *auxvar_d = &(bdrypml_d.auxvar[idim][iside]);
          auxvar_d->cur = auxvar_d->pre;
          auxvar_d->pre = auxvar_d->end;
          auxvar_d->end = auxvar_d->cur;
        }
      }
    }
  } // time loop

  cudaMemcpy(PG,PG_d,sizeof(float)*CONST_NDIM_5*gdcurv->ny*gdcurv->nx,cudaMemcpyDeviceToHost);
  if (isfree == 1)
  {
    PG_slice_output(PG,gdcurv,output_dir,output_fname_part,topoid);
  }
  // io fault init_t0, peak_Vs at final time, use w_buff as buff
  io_fault_end_t_nc_put(&iofault_nc, gdcurv, fault_d, w_buff);

  // finish all time loop calculate, cudafree device pointer
  CUDACHECK(cudaFree(PG_d));
  CUDACHECK(cudaFree(Dis_accu_d));
  CUDACHECK(cudaFree(neighid_d));
  dealloc_md_device(md_d);
  dealloc_metric_device(metric_d);
  dealloc_fd_device(fd_device_d);
  dealloc_fault_coef_device(fault_coef_d);
  dealloc_fault_device(fault_d);
  dealloc_fault_wav_device(fault_wav_d);
  dealloc_bdryfree_device(bdryfree_d);
  dealloc_bdrypml_device(bdrypml_d);
  dealloc_bdryexp_device(bdryexp_d);
  dealloc_wave_device(wav_d);

  // close nc
  io_fault_nc_close(&iofault_nc);
  io_slice_nc_close(&ioslice_nc);
  io_snap_nc_close(&iosnap_nc);

  return 0;
}

