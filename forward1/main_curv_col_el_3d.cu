/*******************************************************************************
 * Curvilinear Grid Finite Difference For Fault Dynamic Simulation 
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <time.h>
#include <mpi.h>

#include "constants.h"
#include "par_t.h"
#include "blk_t.h"

#include "media_discrete_model.h"
#include "sv_eq1st_curv_col.h"
#include "cuda_common.h"

int main(int argc, char** argv)
{
  int verbose = 1; // default fprint
  char *par_fname;
  char err_message[CONST_MAX_STRLEN];

//-------------------------------------------------------------------------------
// initial gpu device before start MPI
//-------------------------------------------------------------------------------
  setDeviceBeforeInit();
//-------------------------------------------------------------------------------
// start MPI and read par
//-------------------------------------------------------------------------------

  // init MPI

  int myid, mpi_size;
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &myid);
  MPI_Comm_size(comm, &mpi_size);

  // get commond-line argument

  if (myid==0) 
  {
    // argc checking
    if (argc < 2) {
      fprintf(stdout,"usage: cgfdm3d_elastic <par_file> <opt: verbose>\n");
      MPI_Finalize();
      exit(1);
    }

    par_fname = argv[1];

    if (argc >= 3) {
      verbose = atoi(argv[2]); // verbose number
      fprintf(stdout,"verbose=%d\n", verbose); fflush(stdout);
    }

    // bcast verbose to all nodes
    MPI_Bcast(&verbose, 1, MPI_INT, 0, comm);
  }
  else
  {
    // get verbose from id 0
    MPI_Bcast(&verbose, 1, MPI_INT, 0, comm);
  }

  if (myid==0 && verbose>0) fprintf(stdout,"comm=%d, size=%d\n", comm, mpi_size); 
  if (myid==0 && verbose>0) fprintf(stdout,"par file =  %s\n", par_fname); 

  // read par

  par_t *par = (par_t *) malloc(sizeof(par_t));

  par_mpi_get(par_fname, myid, comm, par, verbose);

  if (myid==0 && verbose>0) par_print(par);

//-------------------------------------------------------------------------------
// init blk_t
//-------------------------------------------------------------------------------

  if (myid==0 && verbose>0) fprintf(stdout,"create blk ...\n"); 

  // malloc blk
  blk_t *blk = (blk_t *) malloc(sizeof(blk_t));

  // malloc inner vars
  blk_init(blk, myid, verbose);

  fd_t            *fd            = blk->fd    ;
  mympi_t         *mympi         = blk->mympi ;
  gdinfo_t        *gdinfo        = blk->gdinfo;
  gd_t            *gdcurv        = blk->gd;
  gdcurv_metric_t *gdcurv_metric = blk->gdcurv_metric;
  md_t            *md            = blk->md;
  wav_t           *wav           = blk->wav;
  bdryfree_t      *bdryfree      = blk->bdryfree;
  bdrypml_t       *bdrypml       = blk->bdrypml;
  iorecv_t        *iorecv        = blk->iorecv;
  ioline_t        *ioline        = blk->ioline;
  iofault_t       *iofault       = blk->iofault;
  ioslice_t       *ioslice       = blk->ioslice;
  iosnap_t        *iosnap        = blk->iosnap;
  fault_t         *fault         = blk->fault;
  fault_coef_t    *fault_coef    = blk->fault_coef;
  fault_wav_t     *fault_wav     = blk->fault_wav;

  // set up fd_t
  //    not support selection scheme by par file yet
  if (myid==0 && verbose>0) fprintf(stdout,"set scheme ...\n"); 
  fd_set_macdrp(fd);

  // set mpi
  if (myid==0 && verbose>0) fprintf(stdout,"set mpi topo ...\n"); 
  mympi_set(mympi,
            par->number_of_mpiprocs_x,
            par->number_of_mpiprocs_y,
            par->number_of_mpiprocs_z,
            comm,
            myid, verbose);

  // set gdinfo
  gd_info_set(gdinfo, mympi,
              par->number_of_total_grid_points_x,
              par->number_of_total_grid_points_y,
              par->number_of_total_grid_points_z,
              par->abs_num_of_layers,
              fd->fdx_nghosts,
              fd->fdy_nghosts,
              fd->fdz_nghosts,
              verbose);

  // set str in blk
  blk_set_output(blk, mympi,
                 par->output_dir,
                 par->grid_export_dir,
                 par->media_export_dir,
                 verbose);

//-------------------------------------------------------------------------------
//-- grid generation or import
//-------------------------------------------------------------------------------

  if (myid==0 && verbose>0) fprintf(stdout,"allocate grid vars ...\n"); 

  // malloc var in gdcurv
  gd_curv_init(gdinfo, gdcurv);

  // malloc var in gdcurv_metric
  gd_curv_metric_init(gdinfo, gdcurv_metric);

  // generate grid coord
  switch (par->grid_generation_itype)
  {
    case PAR_GRID_FAULT_PLANE : {

        if (myid==0) fprintf(stdout,"gerate grid using fault plane...\n"); 
        gd_curv_gen_fault(gdcurv, gdinfo, par->number_of_total_grid_points_x, par->dh, par->fault_coord_nc);
        if (myid==0 && verbose>0) fprintf(stdout,"exchange coords ...\n"); 
        gd_curv_exchange(gdinfo,gdcurv->v4d,gdcurv->ncmp,mympi->neighid,mympi->topocomm);

        break;
    }
  }

  // cal min/max of this thread
  gd_curv_set_minmax(gdinfo,gdcurv);
  if (myid==0) {
    fprintf(stdout,"calculated min/max of grid/tile/cell\n"); 
    fflush(stdout);
  }

  // output
  if (par->is_export_grid==1)
  {
    if (myid==0) fprintf(stdout,"export coord to file ...\n"); 
    gd_curv_coord_export(gdinfo, gdcurv,
                         blk->output_fname_part,
                         blk->grid_export_dir);
  } else {
    if (myid==0) fprintf(stdout,"do not export coord\n"); 
  }
  fprintf(stdout, " --> done\n"); fflush(stdout);

  // cal metrics and output for QC
  switch (par->metric_method_itype)
  {
    case PAR_METRIC_CALCULATE : {

        if (myid==0 && verbose>0) fprintf(stdout,"calculate metrics ...\n"); 
        gd_curv_metric_cal(gdinfo,
                           gdcurv,
                           gdcurv_metric);

        if (myid==0 && verbose>0) fprintf(stdout,"exchange metrics ...\n"); 
        gd_curv_exchange(gdinfo,gdcurv_metric->v4d,gdcurv_metric->ncmp,mympi->neighid,mympi->topocomm);

        break;
    }
    case PAR_METRIC_IMPORT : {

        if (myid==0) fprintf(stdout,"import metric file ...\n"); 
        gd_curv_metric_import(gdcurv_metric, blk->output_fname_part, par->metric_import_dir);

        break;
    }
  }
  if (myid==0 && verbose>0) { fprintf(stdout, " --> done\n"); fflush(stdout); }

  // export metric
  if (par->is_export_metric==1)
  {
    if (myid==0) fprintf(stdout,"export metric to file ...\n"); 
    gd_curv_metric_export(gdinfo,gdcurv_metric,
                          blk->output_fname_part,
                          blk->grid_export_dir);
  } else {
    if (myid==0) fprintf(stdout,"do not export metric\n"); 
  }
  if (myid==0 && verbose>0) { fprintf(stdout, " --> done\n"); fflush(stdout); }
  // print basic info for QC
  fprintf(stdout,"gdcurv info at topoid=%d,%d,%d\n", mympi->topoid[0],mympi->topoid[1],mympi->topoid[2]); 
  //gd_print(gdcurv);

//-------------------------------------------------------------------------------
//-- media generation or import
//-------------------------------------------------------------------------------

  // allocate media vars
  if (myid==0 && verbose>0) {fprintf(stdout,"allocate media vars ...\n"); fflush(stdout);}
  md_init(gdinfo, md, par->media_itype, par->visco_itype);

  time_t t_start_md = time(NULL);
  // read or discrete velocity model
  switch (par->media_input_itype)
  {
    case PAR_MEDIA_UNIFORM : {

        if (myid==0) fprintf(stdout,"generate simple medium in code ...\n"); 

        if (md->medium_type == CONST_MEDIUM_ELASTIC_ISO) {
          md_gen_uniform_el_iso(md, par);
        }

        if (md->medium_type == CONST_MEDIUM_ELASTIC_VTI) {
          md_gen_uniform_el_vti(md, par);
        }

        if (md->medium_type == CONST_MEDIUM_ELASTIC_ANISO) {
          md_gen_uniform_el_aniso(md, par);
        }

        if (md->visco_type == CONST_VISCO_GRAVES_QS) {
          md_gen_uniform_Qs(md, par->visco_Qs_freq);
        }

        break;
    }

    case PAR_MEDIA_IMPORT : {

        if (myid==0) fprintf(stdout,"import discrete medium file ...\n"); 
        md_import(md, blk->output_fname_part, par->media_import_dir);

        break;
    }

    case PAR_MEDIA_3LAY : {

        if (myid==0) fprintf(stdout,"read and discretize 3D layer medium file ...\n"); 

        if (md->medium_type == CONST_MEDIUM_ELASTIC_ISO)
        {
            media_layer2model_el_iso(md->lambda, md->mu, md->rho,
                                     gdcurv->x3d, gdcurv->y3d, gdcurv->z3d,
                                     gdcurv->nx, gdcurv->ny, gdcurv->nz,
                                     MEDIA_USE_CURV,
                                     par->media_input_file,
                                     par->equivalent_medium_method);
        }
        else if (md->medium_type == CONST_MEDIUM_ELASTIC_VTI)
        {
            media_layer2model_el_vti(md->rho, md->c11, md->c33,
                                     md->c55,md->c66,md->c13,
                                     gdcurv->x3d, gdcurv->y3d, gdcurv->z3d,
                                     gdcurv->nx, gdcurv->ny, gdcurv->nz,
                                     MEDIA_USE_CURV,
                                     par->media_input_file,
                                     par->equivalent_medium_method);
        } else if (md->medium_type == CONST_MEDIUM_ELASTIC_ANISO)
        {
            media_layer2model_el_aniso(md->rho,
                                     md->c11,md->c12,md->c13,md->c14,md->c15,md->c16,
                                             md->c22,md->c23,md->c24,md->c25,md->c26,
                                                     md->c33,md->c34,md->c35,md->c36,
                                                             md->c44,md->c45,md->c46,
                                                                     md->c55,md->c56,
                                                                             md->c66,
                                     gdcurv->x3d, gdcurv->y3d, gdcurv->z3d,
                                     gdcurv->nx, gdcurv->ny, gdcurv->nz,
                                     MEDIA_USE_CURV,
                                     par->media_input_file,
                                     par->equivalent_medium_method);
        }

        break;
    }

    case PAR_MEDIA_3GRD : {

        if (myid==0) fprintf(stdout,"read and descretize 3D grid medium file ...\n"); 

        if (md->medium_type == CONST_MEDIUM_ELASTIC_ISO)
        {
            media_grid2model_el_iso(md->rho,md->lambda, md->mu, 
                                     gdcurv->x3d, gdcurv->y3d, gdcurv->z3d,
                                     gdcurv->nx, gdcurv->ny, gdcurv->nz,
                                     gdcurv->xmin,gdcurv->xmax,
                                     gdcurv->ymin,gdcurv->ymax,
                                     MEDIA_USE_CURV,
                                     par->media_input_file,
                                     par->equivalent_medium_method);
        }
        else if (md->medium_type == CONST_MEDIUM_ELASTIC_VTI)
        {
            media_grid2model_el_vti(md->rho, md->c11, md->c33,
                                     md->c55,md->c66,md->c13,
                                     gdcurv->x3d, gdcurv->y3d, gdcurv->z3d,
                                     gdcurv->nx, gdcurv->ny, gdcurv->nz,
                                     gdcurv->xmin,gdcurv->xmax,
                                     gdcurv->ymin,gdcurv->ymax,
                                     MEDIA_USE_CURV,
                                     par->media_input_file,
                                     par->equivalent_medium_method);
        } else if (md->medium_type == CONST_MEDIUM_ELASTIC_ANISO)
        {
            media_grid2model_el_aniso(md->rho,
                                     md->c11,md->c12,md->c13,md->c14,md->c15,md->c16,
                                             md->c22,md->c23,md->c24,md->c25,md->c26,
                                                     md->c33,md->c34,md->c35,md->c36,
                                                             md->c44,md->c45,md->c46,
                                                                     md->c55,md->c56,
                                                                             md->c66,
                                     gdcurv->x3d, gdcurv->y3d, gdcurv->z3d,
                                     gdcurv->nx, gdcurv->ny, gdcurv->nz,
                                     gdcurv->xmin,gdcurv->xmax,
                                     gdcurv->ymin,gdcurv->ymax,
                                     MEDIA_USE_CURV,
                                     par->media_input_file,
                                     par->equivalent_medium_method);
        }

        break;
    }

    case PAR_MEDIA_3BIN : {

        if (myid==0) fprintf(stdout,"read and descretize 3D bin medium file ...\n"); 

        if (md->medium_type == CONST_MEDIUM_ELASTIC_ISO)
        {
            media_bin2model_el_iso(md->rho,md->lambda, md->mu, 
                                   gdcurv->x3d, gdcurv->y3d, gdcurv->z3d,
                                   gdcurv->nx, gdcurv->ny, gdcurv->nz,
                                   gdcurv->xmin,gdcurv->xmax,
                                   gdcurv->ymin,gdcurv->ymax,
                                   MEDIA_USE_CURV,
                                   par->bin_order,
                                   par->bin_size,
                                   par->bin_spacing,
                                   par->bin_origin,
                                   par->bin_file_rho,
                                   par->bin_file_vp,
                                   par->bin_file_vs);
        }
        else if (md->medium_type == CONST_MEDIUM_ELASTIC_VTI)
        {
          fprintf(stdout,"error: not implement reading bin file for MEDIUM_ELASTIC_VTI\n");
          fflush(stdout);
          exit(1);
            /*
            media_bin2model_el_vti_thomsen(md->rho, md->c11, md->c33,
                                     md->c55,md->c66,md->c13,
                                     gdcurv->x3d, gdcurv->y3d, gdcurv->z3d,
                                     gdcurv->nx, gdcurv->ny, gdcurv->nz,
                                     gdcurv->xmin,gdcurv->xmax,
                                     gdcurv->ymin,gdcurv->ymax,
                                     MEDIA_USE_CURV,
                                     par->bin_order,
                                     par->bin_size,
                                     par->bin_spacing,
                                     par->bin_origin,
                                     par->bin_file_rho,
                                     par->bin_file_vp,
                                     par->bin_file_epsilon,
                                     par->bin_file_delta,
                                     par->bin_file_gamma);
          */
        }
        else if (md->medium_type == CONST_MEDIUM_ELASTIC_ANISO)
        {
          fprintf(stdout,"error: not implement reading bin file for MEDIUM_ELASTIC_ANISO\n");
          fflush(stdout);
          exit(1);
            /*
            media_bin2model_el_aniso(md->rho,
                                     md->c11,md->c12,md->c13,md->c14,md->c15,md->c16,
                                             md->c22,md->c23,md->c24,md->c25,md->c26,
                                                     md->c33,md->c34,md->c35,md->c36,
                                                             md->c44,md->c45,md->c46,
                                                                     md->c55,md->c56,
                                                                             md->c66,
                                     gdcurv->x3d, gdcurv->y3d, gdcurv->z3d,
                                     gdcurv->nx, gdcurv->ny, gdcurv->nz,
                                     gdcurv->xmin,gdcurv->xmax,
                                     gdcurv->ymin,gdcurv->ymax,
                                     MEDIA_USE_CURV,
                                     par->bin_order,
                                     par->bin_size,
                                     par->bin_spacing,
                                     par->bin_origin,
                                     par->bin_file_rho,
                                     par->bin_file_c11,
                                     par->bin_file_c12,
                                     par->bin_file_c13,
                                     par->bin_file_c14,
                                     par->bin_file_c15,
                                     par->bin_file_c16,
                                     par->bin_file_c22,
                                     par->bin_file_c23,
                                     par->bin_file_c24,
                                     par->bin_file_c25,
                                     par->bin_file_c26,
                                     par->bin_file_c33,
                                     par->bin_file_c34,
                                     par->bin_file_c35,
                                     par->bin_file_c36,
                                     par->bin_file_c44,
                                     par->bin_file_c45,
                                     par->bin_file_c46,
                                     par->bin_file_c55,
                                     par->bin_file_c56,
                                     par->bin_file_c66);
          */
        }

        break;
    } 
  }

  MPI_Barrier(comm);
  time_t t_end_md = time(NULL);
  
  if (myid==0 && verbose>0) {
    fprintf(stdout,"media Time of time :%f s \n", difftime(t_end_md,t_start_md));
  }
  // export grid media
  if (par->is_export_media==1)
  {
    if (myid==0) fprintf(stdout,"export discrete medium to file ...\n"); 

    md_export(gdinfo, md,
              blk->output_fname_part,
              blk->media_export_dir);
  } else {
    if (myid==0) fprintf(stdout,"do not export medium\n"); 
  }

//-------------------------------------------------------------------------------
//-- estimate/check/set time step
//-------------------------------------------------------------------------------

  float   t0 = par->time_start;
  float   dt = par->size_of_time_step;
  int     nt_total = par->number_of_time_steps;

  if (par->time_check_stability==1)
  {
    float dt_est[mpi_size];
    float dtmax, dtmaxVp, dtmaxL;
    int   dtmaxi, dtmaxj, dtmaxk;

    //-- estimate time step
    if (myid==0) fprintf(stdout,"   estimate time step ...\n"); 
    blk_dt_esti_curv(gdinfo, gdcurv,md,fd->CFL,
            &dtmax, &dtmaxVp, &dtmaxL, &dtmaxi, &dtmaxj, &dtmaxk);
    
    //-- print for QC
    fprintf(stdout, "-> topoid=[%d,%d,%d], dtmax=%f, Vp=%f, L=%f, i=%d, j=%d, k=%d\n",
            mympi->topoid[0],mympi->topoid[1], mympi->topoid[2], dtmax, dtmaxVp, dtmaxL, dtmaxi, dtmaxj, dtmaxk);
    
    // receive dtmax from each proc
    MPI_Allgather(&dtmax,1,MPI_REAL,dt_est,1,MPI_REAL,MPI_COMM_WORLD);
  
    if (myid==0)
    {
       int dtmax_mpi_id = 0;
       dtmax = 1e19;
       for (int n=0; n < mpi_size; n++)
       {
        fprintf(stdout,"max allowed dt at each proc: id=%d, dtmax=%g\n", n, dt_est[n]);
        if (dtmax > dt_est[n]) {
          dtmax = dt_est[n];
          dtmax_mpi_id = n;
        }
       }
       fprintf(stdout,"Global maximum allowed time step is %g at thread %d\n", dtmax, dtmax_mpi_id);

       // check valid
       if (dtmax <= 0.0) {
          fprintf(stderr,"ERROR: maximum dt <= 0, stop running\n");
          MPI_Abort(MPI_COMM_WORLD,-1);
       }

       //-- auto set stept
       if (dt < 0.0) {
          dt       = blk_keep_two_digi(dtmax);
          nt_total = (int) (par->time_window_length / dt + 0.5);

          fprintf(stdout, "-> Set dt       = %g according to maximum allowed value\n", dt);
          fprintf(stdout, "-> Set nt_total = %d\n", nt_total);
       }

       //-- if input dt, check value
       if (dtmax < dt) {
          fprintf(stdout, "Serious Error: dt=%f > dtmax=%f, stop!\n", dt, dtmax);
          MPI_Abort(MPI_COMM_WORLD, -1);
       }
    }
    
    //-- from root to all threads
    MPI_Bcast(&dt      , 1, MPI_REAL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nt_total, 1, MPI_INT , 0, MPI_COMM_WORLD);
  }

//-------------------------------------------------------------------------------
//-- fault init
//-------------------------------------------------------------------------------

  fault_coef_init(fault_coef, gdinfo); 
  fault_coef_cal(gdinfo, gdcurv_metric, md, par->fault_i_global_index, fault_coef);
  fault_init(fault, gdinfo);
  fault_set(fault, fault_coef, gdinfo, par->bdry_has_free, par->fault_grid, par->init_stress_nc);
  fault_wav_init(gdinfo, fault_wav, fd->num_rk_stages);

//-------------------------------------------------------------------------------
//-- allocate main var
//-------------------------------------------------------------------------------

  if (myid==0 && verbose>0) fprintf(stdout,"allocate solver vars ...\n"); 
  wav_init(gdinfo, wav, fd->num_rk_stages);

//-------------------------------------------------------------------------------
//-- setup output, may require coord info
//-------------------------------------------------------------------------------

  if (myid==0 && verbose>0) fprintf(stdout,"setup output info ...\n"); 

  // receiver: need to do
  io_recv_read_locate(gdinfo, gdcurv, iorecv,
                      nt_total, wav->ncmp, 
                      par->number_of_mpiprocs_z,
                      par->in_station_file,
                      comm, myid, verbose);

  // line
  io_line_locate(gdinfo, gdcurv, ioline,
                 wav->ncmp,
                 nt_total,
                 par->number_of_receiver_line,
                 par->receiver_line_index_start,
                 par->receiver_line_index_incre,
                 par->receiver_line_count,
                 par->receiver_line_name);
  
  // fault slice
  io_fault_locate(gdinfo,iofault,
                  par->fault_i_global_index, 
                  blk->output_fname_part,
                  blk->output_dir);
                  
  // slice
  io_slice_locate(gdinfo, ioslice,
                  par->number_of_slice_x,
                  par->number_of_slice_y,
                  par->number_of_slice_z,
                  par->slice_x_index,
                  par->slice_y_index,
                  par->slice_z_index,
                  blk->output_fname_part,
                  blk->output_dir);
  
  // snapshot
  io_snapshot_locate(gdinfo, iosnap,
                     par->number_of_snapshot,
                     par->snapshot_name,
                     par->snapshot_index_start,
                     par->snapshot_index_count,
                     par->snapshot_index_incre,
                     par->snapshot_time_start,
                     par->snapshot_time_incre,
                     par->snapshot_save_velocity,
                     par->snapshot_save_stress,
                     par->snapshot_save_strain,
                     blk->output_fname_part,
                     blk->output_dir);

//-------------------------------------------------------------------------------
//-- absorbing boundary etc auxiliary variables
//-------------------------------------------------------------------------------

  if (myid==0 && verbose>0) fprintf(stdout,"setup absorbingg boundary ...\n"); 
  
  if (par->bdry_has_cfspml == 1)
  {
    bdry_pml_set(gdinfo, gdcurv, wav, bdrypml,
                 mympi->neighid,
                 par->cfspml_is_sides,
                 par->abs_num_of_layers,
                 par->cfspml_alpha_max,
                 par->cfspml_beta_max,
                 par->cfspml_velocity,
                 verbose);
  }

//-------------------------------------------------------------------------------
//-- free surface preproc
//-------------------------------------------------------------------------------

  if (myid==0 && verbose>0) fprintf(stdout,"cal free surface matrix ...\n"); 

  if (par->bdry_has_free == 1)
  {
    bdry_free_set(gdinfo,bdryfree, mympi->neighid, par->free_is_sides, verbose);
  }

//-------------------------------------------------------------------------------
//-- setup mesg
//-------------------------------------------------------------------------------

  if (myid==0 && verbose>0) fprintf(stdout,"init mesg ...\n"); 
  blk_macdrp_mesg_init(mympi, fd, gdinfo->ni, gdinfo->nj, gdinfo->nk,
                  wav->ncmp, fault_wav->ncmp);

//-------------------------------------------------------------------------------
//-- qc
//-------------------------------------------------------------------------------

  mympi_print(mympi);

  gd_info_print(gdinfo);

  ioslice_print(ioslice);

  iosnap_print(iosnap);

//-------------------------------------------------------------------------------
//-- slover
//-------------------------------------------------------------------------------
  
  // convert rho to 1 / rho to reduce number of arithmetic cal
  md_rho_to_slow(md->rho, md->siz_volume);

  if (myid==0 && verbose>0) fprintf(stdout,"start solver ...\n"); 
  
  time_t t_start = time(NULL);

  sv_eq1st_curv_col_allstep(fd,gdinfo,gdcurv_metric,md,
                            bdryfree,bdrypml, wav, mympi,
                            fault_coef,fault,fault_wav,
                            iorecv,ioline,iofault,ioslice,iosnap,
                            par->imethod, dt,nt_total,t0,
                            blk->output_fname_part,
                            blk->output_dir,
                            par->fault_i_global_index,
                            par->io_time_skip,
                            par->check_nan_every_nummber_of_steps,
                            par->output_all,
                            verbose);

  time_t t_end = time(NULL);
  
  if (myid==0 && verbose>0) {
    fprintf(stdout,"\n\nRuning Time of time :%f s \n", difftime(t_end,t_start));
  }

//-------------------------------------------------------------------------------
//-- save station and line seismo to sac
//-------------------------------------------------------------------------------
  io_recv_output_sac(iorecv,dt,wav->ncmp,wav->cmp_name,
                      blk->output_dir,err_message);

  if(md->medium_type == CONST_MEDIUM_ELASTIC_ISO) {
    io_recv_output_sac_el_iso_strain(iorecv,md->lambda,md->mu,dt,
                      blk->output_dir,err_message);
  }

  io_line_output_sac(ioline,dt,wav->cmp_name,blk->output_dir);

//-------------------------------------------------------------------------------
//-- postprocess
//-------------------------------------------------------------------------------

  MPI_Finalize();

  return 0;
}