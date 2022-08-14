/*********************************************************************
 * setup fd operators
 **********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fdlib_mem.h"
#include "fdlib_math.h"
#include "blk_t.h"
#include "cuda_common.h"

int
blk_init(blk_t *blk,
         const int myid, const int verbose)
{
  int ierr = 0;

  // alloc struct vars
  blk->fd            = (fd_t *)malloc(sizeof(fd_t));
  blk->mympi         = (mympi_t *)malloc(sizeof(mympi_t));
  blk->gdinfo        = (gdinfo_t *)malloc(sizeof(gdinfo_t));
  blk->gd            = (gd_t        *)malloc(sizeof(gd_t     ));
  blk->gdcurv_metric = (gdcurv_metric_t *)malloc(sizeof(gdcurv_metric_t));
  blk->md            = (md_t      *)malloc(sizeof(md_t     ));
  blk->wav           = (wav_t      *)malloc(sizeof(wav_t     ));
  blk->bdryfree      = (bdryfree_t *)malloc(sizeof(bdryfree_t ));
  blk->bdrypml       = (bdrypml_t  *)malloc(sizeof(bdrypml_t ));
  blk->iorecv        = (iorecv_t   *)malloc(sizeof(iorecv_t ));
  blk->ioline        = (ioline_t   *)malloc(sizeof(ioline_t ));
  blk->iofault       = (iofault_t  *)malloc(sizeof(iofault_t ));
  blk->ioslice       = (ioslice_t  *)malloc(sizeof(ioslice_t ));
  blk->iosnap        = (iosnap_t   *)malloc(sizeof(iosnap_t ));
  blk->fault         = (fault_t    *)malloc(sizeof(fault_t ));
  blk->fault_coef    = (fault_coef_t  *)malloc(sizeof(fault_coef_t ));
  blk->fault_wav     = (fault_wav_t   *)malloc(sizeof(fault_wav_t ));

  sprintf(blk->name, "%s", "single");

  return ierr;
}

// set str
int
blk_set_output(blk_t *blk,
               mympi_t *mympi,
               char *output_dir,
               char *grid_export_dir,
               char *media_export_dir,
               const int verbose)
{
  // set name
  //sprintf(blk->name, "%s", name);

  // output name
  sprintf(blk->output_fname_part,"px%d_py%d_pz%d", mympi->topoid[0],mympi->topoid[1],mympi->topoid[2]);

  // output
  sprintf(blk->output_dir, "%s", output_dir);
  sprintf(blk->grid_export_dir, "%s", grid_export_dir);
  sprintf(blk->media_export_dir, "%s", media_export_dir);

  return 0;
}


/*********************************************************************
 * mpi message for macdrp scheme with rk
 *********************************************************************/

void
blk_macdrp_mesg_init(mympi_t *mympi,
                     fd_t *fd,
                     int ni,
                     int nj,
                     int nk,
                     int num_of_vars,
                     int num_of_vars_fault)
{
  // alloc
  mympi->pair_siz_sbuff_y1 = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_sbuff_y2 = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_sbuff_z1 = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_sbuff_z2 = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));

  mympi->pair_siz_rbuff_y1 = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_rbuff_y2 = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_rbuff_z1 = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_rbuff_z2 = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));

  mympi->pair_siz_sbuff_y1_fault = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_sbuff_y2_fault = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_sbuff_z1_fault = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_sbuff_z2_fault = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));

  mympi->pair_siz_rbuff_y1_fault = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_rbuff_y2_fault = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_rbuff_z1_fault = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_rbuff_z2_fault = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));

  mympi->pair_s_reqs       = (MPI_Request ***)malloc(fd->num_of_pairs * sizeof(MPI_Request **));
  mympi->pair_r_reqs       = (MPI_Request ***)malloc(fd->num_of_pairs * sizeof(MPI_Request **));
  for (int ipair = 0; ipair < fd->num_of_pairs; ipair++)
  {
    mympi->pair_siz_sbuff_y1[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_sbuff_y2[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_sbuff_z1[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_sbuff_z2[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));

    mympi->pair_siz_rbuff_y1[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_rbuff_y2[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_rbuff_z1[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_rbuff_z2[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));

    mympi->pair_siz_sbuff_y1_fault[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_sbuff_y2_fault[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_sbuff_z1_fault[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_sbuff_z2_fault[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));

    mympi->pair_siz_rbuff_y1_fault[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_rbuff_y2_fault[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_rbuff_z1_fault[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_rbuff_z2_fault[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));

    mympi->pair_s_reqs[ipair] = (MPI_Request **)malloc(fd->num_rk_stages * sizeof(MPI_Request *));
    mympi->pair_r_reqs[ipair] = (MPI_Request **)malloc(fd->num_rk_stages * sizeof(MPI_Request *));

    for (int istage = 0; istage < fd->num_rk_stages; istage++)
    {
      mympi->pair_s_reqs[ipair][istage] = (MPI_Request *)malloc(8 * sizeof(MPI_Request));
      mympi->pair_r_reqs[ipair][istage] = (MPI_Request *)malloc(8 * sizeof(MPI_Request));
    }
  }

  // mpi mesg
  mympi->siz_sbuff = 0;
  mympi->siz_rbuff = 0;
  for (int ipair = 0; ipair < fd->num_of_pairs; ipair++)
  {
    for (int istage = 0; istage < fd->num_rk_stages; istage++)
    {
      fd_op_t *fdy_op = fd->pair_fdy_op[ipair][istage];
      fd_op_t *fdz_op = fd->pair_fdz_op[ipair][istage];

      // wave exchange
      // y1 side, depends on right_len of y1 proc
      mympi->pair_siz_sbuff_y1[ipair][istage] = (ni * nk * fdy_op->right_len) * num_of_vars;
      // y2 side, depends on left_len of y2 proc
      mympi->pair_siz_sbuff_y2[ipair][istage] = (ni * nk * fdy_op->left_len ) * num_of_vars;

      mympi->pair_siz_sbuff_z1[ipair][istage] = (ni * nj * fdz_op->right_len) * num_of_vars;
      mympi->pair_siz_sbuff_z2[ipair][istage] = (ni * nj * fdz_op->left_len ) * num_of_vars;

      // y1 side, depends on left_len of cur proc
      mympi->pair_siz_rbuff_y1[ipair][istage] = (ni * nk * fdy_op->left_len ) * num_of_vars;
      // y2 side, depends on right_len of cur proc
      mympi->pair_siz_rbuff_y2[ipair][istage] = (ni * nk * fdy_op->right_len) * num_of_vars;

      mympi->pair_siz_rbuff_z1[ipair][istage] = (ni * nj * fdz_op->left_len ) * num_of_vars;
      mympi->pair_siz_rbuff_z2[ipair][istage] = (ni * nj * fdz_op->right_len) * num_of_vars;

      // fault wave exchange
      // minus and plus, multiply 2
      mympi->pair_siz_sbuff_y1_fault[ipair][istage] = (nk * fdy_op->right_len) * 2 * num_of_vars_fault;
      mympi->pair_siz_sbuff_y2_fault[ipair][istage] = (nk * fdy_op->left_len ) * 2 * num_of_vars_fault;

      mympi->pair_siz_sbuff_z1_fault[ipair][istage] = (nj * fdz_op->right_len) * 2 *  num_of_vars_fault;
      mympi->pair_siz_sbuff_z2_fault[ipair][istage] = (nj * fdz_op->left_len ) * 2 *  num_of_vars_fault;

      mympi->pair_siz_rbuff_y1_fault[ipair][istage] = (nk * fdy_op->left_len ) * 2 * num_of_vars_fault;
      mympi->pair_siz_rbuff_y2_fault[ipair][istage] = (nk * fdy_op->right_len) * 2 * num_of_vars_fault;

      mympi->pair_siz_rbuff_z1_fault[ipair][istage] = (nj * fdz_op->left_len ) * 2 * num_of_vars_fault;
      mympi->pair_siz_rbuff_z2_fault[ipair][istage] = (nj * fdz_op->right_len) * 2 * num_of_vars_fault;

      size_t siz_s =  mympi->pair_siz_sbuff_y1[ipair][istage]
                    + mympi->pair_siz_sbuff_y2[ipair][istage]
                    + mympi->pair_siz_sbuff_z1[ipair][istage]
                    + mympi->pair_siz_sbuff_z2[ipair][istage]
                    + mympi->pair_siz_sbuff_y1_fault[ipair][istage]
                    + mympi->pair_siz_sbuff_y2_fault[ipair][istage]
                    + mympi->pair_siz_sbuff_z1_fault[ipair][istage]
                    + mympi->pair_siz_sbuff_z2_fault[ipair][istage];

      size_t siz_r =  mympi->pair_siz_rbuff_y1[ipair][istage]
                    + mympi->pair_siz_rbuff_y2[ipair][istage]
                    + mympi->pair_siz_rbuff_z1[ipair][istage]
                    + mympi->pair_siz_rbuff_z2[ipair][istage]
                    + mympi->pair_siz_rbuff_y1_fault[ipair][istage]
                    + mympi->pair_siz_rbuff_y2_fault[ipair][istage]
                    + mympi->pair_siz_rbuff_z1_fault[ipair][istage]
                    + mympi->pair_siz_rbuff_z2_fault[ipair][istage];

      if (siz_s > mympi->siz_sbuff) mympi->siz_sbuff = siz_s;
      if (siz_r > mympi->siz_rbuff) mympi->siz_rbuff = siz_r;
    }
  }
  // alloc in gpu
  mympi->sbuff = (float *) cuda_malloc(mympi->siz_sbuff * sizeof(MPI_FLOAT));
  mympi->rbuff = (float *) cuda_malloc(mympi->siz_rbuff * sizeof(MPI_FLOAT));

  // set up pers communication
  for (int ipair = 0; ipair < fd->num_of_pairs; ipair++)
  {
    for (int istage = 0; istage < fd->num_rk_stages; istage++)
    {
      size_t siz_s_y1 = mympi->pair_siz_sbuff_y1[ipair][istage];
      size_t siz_s_y2 = mympi->pair_siz_sbuff_y2[ipair][istage];
      size_t siz_s_z1 = mympi->pair_siz_sbuff_z1[ipair][istage];
      size_t siz_s_z2 = mympi->pair_siz_sbuff_z2[ipair][istage];
      size_t siz_s_y1_fault = mympi->pair_siz_sbuff_y1_fault[ipair][istage];
      size_t siz_s_y2_fault = mympi->pair_siz_sbuff_y2_fault[ipair][istage];
      size_t siz_s_z1_fault = mympi->pair_siz_sbuff_z1_fault[ipair][istage];
      size_t siz_s_z2_fault = mympi->pair_siz_sbuff_z2_fault[ipair][istage];

      float *sbuff_y1 = mympi->sbuff;
      float *sbuff_y2 = sbuff_y1 + siz_s_y1;
      float *sbuff_z1 = sbuff_y2 + siz_s_y2;
      float *sbuff_z2 = sbuff_z1 + siz_s_z1;
      float *sbuff_y1_fault = sbuff_z2 + siz_s_z2;
      float *sbuff_y2_fault = sbuff_y1_fault + siz_s_y1_fault;
      float *sbuff_z1_fault = sbuff_y2_fault + siz_s_y2_fault;
      float *sbuff_z2_fault = sbuff_z1_fault + siz_s_z1_fault;
      
      float mympi->sbuff_fault = sbuff_y1_fault;
      // npair: xx, nstage: x, 
      int tag_pair_stage = ipair * 1000 + istage * 100;
      int tag[8] = { tag_pair_stage+21, tag_pair_stage+22, tag_pair_stage+31, tag_pair_stage+32, 
                     tag_pair_stage+210, tag_pair_stage+220, tag_pair_stage+310, tag_pair_stage+320};

      // send
      MPI_Send_init(sbuff_y1, siz_s_y1, MPI_FLOAT, mympi->neighid[2], tag[0], mympi->topocomm, &(mympi->pair_s_reqs[ipair][istage][0]));
      MPI_Send_init(sbuff_y2, siz_s_y2, MPI_FLOAT, mympi->neighid[3], tag[1], mympi->topocomm, &(mympi->pair_s_reqs[ipair][istage][1]));
      MPI_Send_init(sbuff_z1, siz_s_z1, MPI_FLOAT, mympi->neighid[4], tag[2], mympi->topocomm, &(mympi->pair_s_reqs[ipair][istage][2]));
      MPI_Send_init(sbuff_z2, siz_s_z2, MPI_FLOAT, mympi->neighid[5], tag[3], mympi->topocomm, &(mympi->pair_s_reqs[ipair][istage][3]));
      MPI_Send_init(sbuff_y1_fault, siz_s_y1_fault, MPI_FLOAT, mympi->neighid[2], tag[4], mympi->topocomm, &(mympi->pair_s_reqs[ipair][istage][4]));
      MPI_Send_init(sbuff_y2_fault, siz_s_y2_fault, MPI_FLOAT, mympi->neighid[3], tag[5], mympi->topocomm, &(mympi->pair_s_reqs[ipair][istage][5]));
      MPI_Send_init(sbuff_z1_fault, siz_s_z1_fault, MPI_FLOAT, mympi->neighid[4], tag[6], mympi->topocomm, &(mympi->pair_s_reqs[ipair][istage][6]));
      MPI_Send_init(sbuff_z2_fault, siz_s_z2_fault, MPI_FLOAT, mympi->neighid[5], tag[7], mympi->topocomm, &(mympi->pair_s_reqs[ipair][istage][7]));

      // recv
      size_t siz_r_y1 = mympi->pair_siz_rbuff_y1[ipair][istage];
      size_t siz_r_y2 = mympi->pair_siz_rbuff_y2[ipair][istage];
      size_t siz_r_z1 = mympi->pair_siz_rbuff_z1[ipair][istage];
      size_t siz_r_z2 = mympi->pair_siz_rbuff_z2[ipair][istage];
      size_t siz_r_y1_fault = mympi->pair_siz_rbuff_y1_fault[ipair][istage];
      size_t siz_r_y2_fault = mympi->pair_siz_rbuff_y2_fault[ipair][istage];
      size_t siz_r_z1_fault = mympi->pair_siz_rbuff_z1_fault[ipair][istage];
      size_t siz_r_z2_fault = mympi->pair_siz_rbuff_z2_fault[ipair][istage];

      float *rbuff_y1 = mympi->rbuff;
      float *rbuff_y2 = rbuff_y1 + siz_r_y1;
      float *rbuff_z1 = rbuff_y2 + siz_r_y2;
      float *rbuff_z2 = rbuff_z1 + siz_r_z1;
      float *rbuff_y1_fault = rbuff_z2 + siz_r_z2;
      float *rbuff_y2_fault = rbuff_y1_fault + siz_r_y1_fault;
      float *rbuff_z1_fault = rbuff_y2_fault + siz_r_y2_fault;
      float *rbuff_z2_fault = rbuff_z1_fault + siz_r_z1_fault;

      float mympi->rbuff_fault = rbuff_y1_fault;
      // recv
      MPI_Recv_init(rbuff_y1, siz_r_y1, MPI_FLOAT, mympi->neighid[2], tag[1], mympi->topocomm, &(mympi->pair_r_reqs[ipair][istage][0]));
      MPI_Recv_init(rbuff_y2, siz_r_y2, MPI_FLOAT, mympi->neighid[3], tag[0], mympi->topocomm, &(mympi->pair_r_reqs[ipair][istage][1]));
      MPI_Recv_init(rbuff_z1, siz_r_z1, MPI_FLOAT, mympi->neighid[4], tag[3], mympi->topocomm, &(mympi->pair_r_reqs[ipair][istage][2]));
      MPI_Recv_init(rbuff_z2, siz_r_z2, MPI_FLOAT, mympi->neighid[5], tag[2], mympi->topocomm, &(mympi->pair_r_reqs[ipair][istage][3]));
      MPI_Recv_init(rbuff_y1_fault, siz_r_y1_fault, MPI_FLOAT, mympi->neighid[2], tag[5], mympi->topocomm, &(mympi->pair_r_reqs[ipair][istage][4]));
      MPI_Recv_init(rbuff_y2_fault, siz_r_y2_fault, MPI_FLOAT, mympi->neighid[3], tag[4], mympi->topocomm, &(mympi->pair_r_reqs[ipair][istage][5]));
      MPI_Recv_init(rbuff_z1_fault, siz_r_z1_fault, MPI_FLOAT, mympi->neighid[4], tag[7], mympi->topocomm, &(mympi->pair_r_reqs[ipair][istage][6]));
      MPI_Recv_init(rbuff_z2_fault, siz_r_z2_fault, MPI_FLOAT, mympi->neighid[5], tag[6], mympi->topocomm, &(mympi->pair_r_reqs[ipair][istage][7]));
    }
  }

  return;
}

int 
blk_macdrp_pack_mesg_gpu(float * w_cur,
                         fd_t *fd,
                         gdinfo_t *gdinfo, 
                         mympi_t *mympi, 
                         int ipair_mpi,
                         int istage_mpi,
                         int num_of_vars,
                         int myid)
{
  int ni1 = gdinfo->ni1;
  int ni2 = gdinfo->ni2;
  int nj1 = gdinfo->nj1;
  int nj2 = gdinfo->nj2;
  int nk1 = gdinfo->nk1;
  int nk2 = gdinfo->nk2;
  size_t siz_line   = gdinfo->siz_line;
  size_t siz_slice  = gdinfo->siz_slice;
  size_t siz_volume = gdinfo->siz_volume;
  int ni = ni2-ni1+1;
  int nj = nj2-nj1+1;
  int nk = nk2-nk1+1;

  fd_op_t *fdy_op = fd->pair_fdy_op[ipair_mpi][istage_mpi];
  fd_op_t *fdz_op = fd->pair_fdz_op[ipair_mpi][istage_mpi];
  // ghost point
  int ny1_g = fdy_op->right_len;
  int ny2_g = fdy_op->left_len;
  int nz1_g = fdz_op->right_len;
  int nz2_g = fdz_op->left_len;
  size_t siz_sbuff_y1 = mympi->pair_siz_sbuff_y1[ipair_mpi][istage_mpi];
  size_t siz_sbuff_y2 = mympi->pair_siz_sbuff_y2[ipair_mpi][istage_mpi];
  size_t siz_sbuff_z1 = mympi->pair_siz_sbuff_z1[ipair_mpi][istage_mpi];
  
  float *sbuff_y1 = mympi->sbuff;
  float *sbuff_y2 = sbuff_y1 + siz_sbuff_y1;
  float *sbuff_z1 = sbuff_y2 + siz_sbuff_y2;
  float *sbuff_z2 = sbuff_z1 + siz_sbuff_z1;
  {
    dim3 block(8,ny1_g,8);
    dim3 grid;
    grid.x = (ni + block.x - 1) / block.x;
    grid.y = (ny1_g + block.y -1) / block.y;
    grid.z = (nk + block.z - 1) / block.z;
    blk_macdrp_pack_mesg_y1<<<grid, block >>>(
           w_cur, sbuff_y1, siz_line, siz_slice, siz_volume, num_of_vars,
           ni1, nj1, nk1, ni, ny1_g, nk);
    CUDACHECK(cudaDeviceSynchronize());
  }
  {
    dim3 block(8,ny2_g,8);
    dim3 grid;
    grid.x = (ni + block.x - 1) / block.x;
    grid.y = (ny2_g + block.y -1) / block.y;
    grid.z = (nk + block.z - 1) / block.z;
    blk_macdrp_pack_mesg_y2<<<grid, block >>>(
           w_cur, sbuff_y2, siz_line, siz_slice, siz_volume,
           num_of_vars, ni1, nj2, nk1, ni, ny2_g, nk);
    CUDACHECK(cudaDeviceSynchronize());
  }
  {
    dim3 block(8,8,nz1_g);
    dim3 grid;
    grid.x = (ni + block.x - 1) / block.x;
    grid.y = (nj + block.y - 1) / block.y;
    grid.z = (nz1_g + block.z - 1) / block.z;
    blk_macdrp_pack_mesg_z1<<<grid, block >>>(
           w_cur, sbuff_z1, siz_line, siz_slice, siz_volume,
           num_of_vars, ni1, nj1, nk1, ni, nj, nz1_g);
    CUDACHECK(cudaDeviceSynchronize());
  }
  {
    dim3 block(8,8,nz2_g);
    dim3 grid;
    grid.x = (ni + block.x - 1) / block.x;
    grid.y = (nj + block.y - 1) / block.y;
    grid.z = (nz2_g + block.z - 1) / block.z;
    blk_macdrp_pack_mesg_z2<<<grid, block >>>(
           w_cur, sbuff_z2, siz_line, siz_slice, siz_volume,
           num_of_vars, ni1, nj1, nk2, ni, nj, nz2_g);
    CUDACHECK(cudaDeviceSynchronize());
  }

  return 0;
}

__global__ void
blk_macdrp_pack_mesg_y1(
           float *w_cur, float *sbuff_y1, size_t siz_line, size_t siz_slice, size_t siz_volume,
           int num_of_vars, int ni1, int nj1, int nk1, int ni, int ny1_g, int nk)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  size_t iptr_b;
  size_t iptr;
  if(ix<ni && iy<ny1_g && iz<nk)
  {
    iptr     = (iz+nk1) * siz_slice + (iy+nj1) * siz_line + (ix+ni1);
    iptr_b   = iz*ni*ny1_g + iy*ni + ix;
    for(int i=0; i<num_of_vars; i++)
    {
      sbuff_y1[iptr_b + i*ny1_g*ni*nk] = w_cur[iptr + i*siz_volume];
    }
  }

  return;
}

__global__ void
blk_macdrp_pack_mesg_y2(
           float *w_cur, float *sbuff_y2, size_t siz_line, size_t siz_slice, size_t siz_volume,
           int num_of_vars, int ni1, int nj2, int nk1, int ni, int ny2_g, int nk)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  size_t iptr_b;
  size_t iptr;
  if(ix<ni && iy<ny2_g && iz<nk)
  {
    iptr     = (iz+nk1) * siz_slice + (iy+nj2-ny2_g+1) * siz_line + (ix+ni1);
    iptr_b   = iz*ni*ny2_g + iy*ni + ix;
    for(int i=0; i<num_of_vars; i++)
    {
      sbuff_y2[iptr_b + i*ny2_g*ni*nk] = w_cur[iptr + i*siz_volume];
    }
  }
  return;
}

__global__ void
blk_macdrp_pack_mesg_z1(
           float *w_cur, float *sbuff_z1, size_t siz_line, size_t siz_slice, size_t siz_volume,
           int num_of_vars, int ni1, int nj1, int nk1, int ni, int nj, int nz1_g)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  size_t iptr_b;
  size_t iptr;
  if(ix<ni && iy<nj && iz<nz1_g)
  {
    iptr     = (iz+nk1) * siz_slice+ (iy+nj1) * siz_line + (ix+ni1);
    iptr_b   = iz*ni*nj + iy*ni + ix;
    for(int i=0; i<num_of_vars; i++)
    {
      sbuff_z1[iptr_b + i*nz1_g*ni*nj] = w_cur[iptr + i*siz_volume];
    }
  }
  return;
}

__global__ void
blk_macdrp_pack_mesg_z2(
           float *w_cur, float *sbuff_z2, size_t siz_line, size_t siz_slice, size_t siz_volume,
           int num_of_vars, int ni1, int nj1, int nk2, int ni, int nj, int nz2_g)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  size_t iptr_b;
  size_t iptr;
  if(ix<ni && iy<nj && iz<nz2_g)
  {
    iptr     = (iz+nk2-nz2_g+1) * siz_slice + (iy+nj1) * siz_line + (ix+ni1);
    iptr_b   = iz*ni*nj + iy*ni + ix;
    for(int i=0; i<num_of_vars; i++)
    {
      sbuff_z2[iptr_b + i*nz2_g*ni*nj] = w_cur[iptr + i*siz_volume];
    }
  }
  return;
}

int 
blk_macdrp_unpack_mesg_gpu(float *w_cur, 
                           fd_t *fd,
                           gdinfo_t *gdinfo,
                           mympi_t *mympi, 
                           int ipair_mpi,
                           int istage_mpi,
                           int num_of_vars,
                           int *neighid)
{
  int ni1 = gdinfo->ni1;
  int ni2 = gdinfo->ni2;
  int nj1 = gdinfo->nj1;
  int nj2 = gdinfo->nj2;
  int nk1 = gdinfo->nk1;
  int nk2 = gdinfo->nk2;
  size_t siz_line   = gdinfo->siz_line;
  size_t siz_slice  = gdinfo->siz_slice;
  size_t siz_volume = gdinfo->siz_volume;

  int ni = ni2-ni1+1;
  int nj = nj2-nj1+1;
  int nk = nk2-nk1+1;
  
  fd_op_t *fdy_op = fd->pair_fdy_op[ipair_mpi][istage_mpi];
  fd_op_t *fdz_op = fd->pair_fdz_op[ipair_mpi][istage_mpi];
  // ghost point
  int ny1_g = fdy_op->right_len;
  int ny2_g = fdy_op->left_len;
  int nz1_g = fdz_op->right_len;
  int nz2_g = fdz_op->left_len;

  size_t siz_rbuff_y1 = mympi->pair_siz_rbuff_y1[ipair_mpi][istage_mpi];
  size_t siz_rbuff_y2 = mympi->pair_siz_rbuff_y2[ipair_mpi][istage_mpi];
  size_t siz_rbuff_z1 = mympi->pair_siz_rbuff_z1[ipair_mpi][istage_mpi];

  float *rbuff_y1 = mympi->rbuff;
  float *rbuff_y2 = rbuff_y1 + siz_rbuff_y1;
  float *rbuff_z1 = rbuff_y2 + siz_rbuff_y2;
  float *rbuff_z2 = rbuff_z1 + siz_rbuff_z1;
  {
    dim3 block(8,ny2_g,8);
    dim3 grid;
    grid.x = (ni + block.x - 1) / block.x;
    grid.y = (ny2_g + block.y -1) / block.y;
    grid.z = (nk + block.z - 1) / block.z;
    blk_macdrp_unpack_mesg_y1<<< grid, block >>>(
           w_cur, rbuff_y1, siz_line, siz_slice, siz_volume,
           num_of_vars, ni1, nj1, nk1, ni, ny2_g, nk, neighid);
    CUDACHECK(cudaDeviceSynchronize());
  }
  {
    dim3 block(8,ny1_g,8);
    dim3 grid;
    grid.x = (ni + block.x - 1) / block.x;
    grid.y = (ny1_g + block.y -1) / block.y;
    grid.z = (nk + block.z - 1) / block.z;
    blk_macdrp_unpack_mesg_y2<<< grid, block >>>(
           w_cur, rbuff_y2, siz_line, siz_slice, siz_volume,
           num_of_vars, ni1, nj2, nk1, ni, ny1_g, nk, neighid);
    CUDACHECK(cudaDeviceSynchronize());
  }
  {
    dim3 block(8,8,nz2_g);
    dim3 grid;
    grid.x = (ni + block.x - 1) / block.x;
    grid.y = (nj + block.y -1) / block.y;
    grid.z = (nz2_g + block.z - 1) / block.z;
    blk_macdrp_unpack_mesg_z1<<< grid, block >>>(
           w_cur, rbuff_z1, siz_line, siz_slice, siz_volume, 
           num_of_vars, ni1, ni1, nk1, ni, nj, nz2_g, neighid);
    CUDACHECK(cudaDeviceSynchronize());
  }
  {
    dim3 block(8,8,nz1_g);
    dim3 grid;
    grid.x = (ni + block.x - 1) / block.x;
    grid.y = (nj + block.y -1) / block.y;
    grid.z = (nz1_g + block.z - 1) / block.z;
    blk_macdrp_unpack_mesg_z2<<< grid, block >>>(
           w_cur, rbuff_z2, siz_line, siz_slice, siz_volume, 
           num_of_vars, ni1, nj1, nk2, ni, nj, nz1_g, neighid);
    CUDACHECK(cudaDeviceSynchronize());
  }

  return 0;
}

//from y2
__global__ void
blk_macdrp_unpack_mesg_y1(
           float *w_cur, float *rbuff_y1, size_t siz_line, size_t siz_slice, size_t siz_volume,
           int num_of_vars, int ni1, int nj1, int nk1, int ni, int ny2_g, int nk, int *neighid)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  size_t iptr_b;
  size_t iptr;
  if (neighid[2] != MPI_PROC_NULL) {
    if(ix<ni && iy<ny2_g && iz<nk){
      iptr   = (iz+nk1) * siz_slice + (iy+nj1-ny2_g) * siz_line + (ix+ni1);
      iptr_b = iz*ni*ny2_g + iy*ni + ix;
      for(int i=0; i<num_of_vars; i++)
      {
        w_cur[iptr + i*siz_volume] = rbuff_y1[iptr_b+ i*ny2_g*ni*nk];
      }
    }
  }
  return;
}

//from y1
__global__ void
blk_macdrp_unpack_mesg_y2(
           float *w_cur, float *rbuff_y2, size_t siz_line, size_t siz_slice, size_t siz_volume, 
           int num_of_vars, int ni1, int nj2, int nk1, int ni, int ny1_g, int nk, int *neighid)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  size_t iptr_b;
  size_t iptr;
  if (neighid[3] != MPI_PROC_NULL) {
    if(ix<ni && iy<ny1_g && iz<nk){
      iptr   = (iz+nk1) * siz_slice + (iy+nj2+1) * siz_line + (ix+ni1);
      iptr_b = iz*ni*ny1_g + iy*ni + ix;
      for(int i=0; i<num_of_vars; i++)
      {
        w_cur[iptr + i*siz_volume] = rbuff_y2[iptr_b+ i*ny1_g*ni*nk];
      }
    }
  }
  return;
}

//from z2
__global__ void
blk_macdrp_unpack_mesg_z1(
           float *w_cur, float *rbuff_z1, size_t siz_line, size_t siz_slice, size_t siz_volume, 
           int num_of_vars, int ni1, int nj1, int nk1, int ni, int nj, int nz2_g, int *neighid)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  size_t iptr_b;
  size_t iptr;
  if (neighid[4] != MPI_PROC_NULL)
  {
    if(ix<ni && iy<nj && iz<nz2_g)
    {
      iptr   = (ix+ni1) + (iy+nj1) * siz_line + (iz+nk1-nz2_g) * siz_slice;
      iptr_b = iz*ni*nj + iy*ni + ix;
      for(int i=0; i<num_of_vars; i++)
      {
        w_cur[iptr + i*siz_volume] = rbuff_z1[iptr_b + i*nz2_g*ni*nj];
      }
    }
  }
  return;
}

//from z1
__global__ void
blk_macdrp_unpack_mesg_z2(
           float *w_cur, float *rbuff_z2, size_t siz_line, size_t siz_slice, size_t siz_volume,
           int num_of_vars, int ni1, int nj1, int nk2, int ni, int nj, int nz1_g, int *neighid)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  size_t iptr_b;
  size_t iptr;
  if (neighid[5] != MPI_PROC_NULL)
  {
    if(ix<ni && iy<nj && iz<nz1_g)
    {
      iptr   = (ix+ni1) + (iy+nj1) * siz_line + (iz+nk2+1) * siz_slice;
      iptr_b = iz*ni*nj + iy*ni + ix;
      for(int i=0; i<num_of_vars; i++)
      {
        w_cur[iptr + i*siz_volume] = rbuff_z2[iptr_b+ i*nz1_g*ni*nj];
      }
    }
  }

  return;
}

int 
blk_macdrp_pack_fault_mesg_gpu(float * fw_cur,
                               fd_t *fd,
                               gdinfo_t *gdinfo, 
                               mympi_t *mympi, 
                               int ipair_mpi,
                               int istage_mpi,
                               int num_of_vars_fault,
                               int myid)
{
  int nj1 = gdinfo->nj1;
  int nj2 = gdinfo->nj2;
  int nk1 = gdinfo->nk1;
  int nk2 = gdinfo->nk2;
  int nj = gdinfo->nj;
  int nk = gdinfo->nk;
  int ny = gdinfo->ny;
  size_t siz_slice_yz = gdinfo->siz_slice_yz;

  fd_op_t *fdy_op = fd->pair_fdy_op[ipair_mpi][istage_mpi];
  fd_op_t *fdz_op = fd->pair_fdz_op[ipair_mpi][istage_mpi];
  // ghost point
  int ny1_g = fdy_op->right_len;
  int ny2_g = fdy_op->left_len;
  int nz1_g = fdz_op->right_len;
  int nz2_g = fdz_op->left_len;
  size_t siz_sbuff_y1_fault = mympi->pair_siz_sbuff_y1_fault[ipair_mpi][istage_mpi];
  size_t siz_sbuff_y2_fault = mympi->pair_siz_sbuff_y2_fault[ipair_mpi][istage_mpi];
  size_t siz_sbuff_z1_fault = mympi->pair_siz_sbuff_z1_fault[ipair_mpi][istage_mpi];
  
  float *sbuff_y1_fault = mympi->sbuff_fault;
  float *sbuff_y2_fault = sbuff_y1_fault + siz_sbuff_y1_fault;
  float *sbuff_z1_fault = sbuff_y2_fault + siz_sbuff_y2_fault;
  float *sbuff_z2_fault = sbuff_z1_fault + siz_sbuff_z1_fault;
  {
    dim3 block(ny1_g,8);
    dim3 grid;
    grid.x = (ny1_g + block.x -1) / block.x;
    grid.y = (nk + block.y - 1) / block.y;
    blk_macdrp_pack_fault_mesg_y1<<<grid, block >>>(
                                 fw_cur, sbuff_y1_fault, siz_slice_yz, 
                                 num_of_vars_fault, ny, nj1, nk1, ny1_g, nk);
    CUDACHECK(cudaDeviceSynchronize());
  }
  {
    dim3 block(ny2_g,8);
    dim3 grid;
    grid.x = (ny2_g + block.x -1) / block.x;
    grid.y = (nk + block.y - 1) / block.y;
    blk_macdrp_pack_fault_mesg_y2<<<grid, block >>>(
                                 fw_cur, sbuff_y2_fault, siz_slice_yz, 
                                 num_of_vars_fault, ny, nj2, nk1, ny2_g, nk);
    CUDACHECK(cudaDeviceSynchronize());
  }
  {
    dim3 block(8,nz1_g);
    dim3 grid;
    grid.x = (nj + block.x - 1) / block.x;
    grid.y = (nz1_g + block.y - 1) / block.y;
    blk_macdrp_pack_fault_mesg_z1<<<grid, block >>>(
                                 fw_cur, sbuff_z1_fault, siz_slice_yz, 
                                 num_of_vars_fault, ny, nj1, nk1, nj, nz1_g);
    CUDACHECK(cudaDeviceSynchronize());
  }
  {
    dim3 block(8,nz2_g);
    dim3 grid;
    grid.x = (nj + block.x - 1) / block.x;
    grid.y = (nz2_g + block.y - 1) / block.y;
    blk_macdrp_pack_fault_mesg_z2<<<grid, block >>>(
                                 fw_cur, sbuff_z2_fault, siz_slice_yz,
                                 num_of_vars_fault, ny, nj1, nk2, nj, nz2_g);
    CUDACHECK(cudaDeviceSynchronize());
  }

  return 0;
}

__global__ void
blk_macdrp_pack_fault_mesg_y1(
             float *fw_cur, float *sbuff_y1_fault, size_t siz_slice_yz, 
             int num_of_vars_fault, int ny, int nj1, int nk1, int ny1_g, int nk)
{
  int iy = blockIdx.x * blockDim.x + threadIdx.x;
  int iz = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iptr_b;
  size_t iptr;
  if(iy<ny1_g && iz<nk)
  {
    iptr     = (iz+nk1) * ny + (iy+nj1);
    iptr_b   = iz*ny1_g + iy;
    for(int i=0; i<2*num_of_vars; i++)
    {
      sbuff_y1_fault[iptr_b + i*ny1_g*nk] = fw_cur[iptr + i*siz_slice_yz];
    }
  }

  return;
}

__global__ void
blk_macdrp_pack_fault_mesg_y2(
             float *fw_cur, float *sbuff_y2_fault, size_t siz_slice_yz, 
             int num_of_vars_fault, int ny, int nj2, int nk1, int ny2_g, int nk)
{
  int iy = blockIdx.x * blockDim.x + threadIdx.x;
  int iz = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iptr_b;
  size_t iptr;
  if(iy<ny2_g && iz<nk)
  {
    iptr     = (iz+nk1) * ny + (iy+nj2-ny2_g+1);
    iptr_b   = iz*ny2_g + iy;
    for(int i=0; i<2*num_of_vars; i++)
    {
      sbuff_y2_fault[iptr_b + i*ny2_g*nk] = fw_cur[iptr + i*siz_slice_yz];
    }
  }

  return;
}

__global__ void
blk_macdrp_pack_fault_mesg_z1(
             float *fw_cur, float *sbuff_z1_fault, size_t siz_slice_yz, 
             int num_of_vars_fault, int ny, int nj1, int nk1, int nj, int nz1_g)
{
  int iy = blockIdx.x * blockDim.x + threadIdx.x;
  int iz = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iptr_b;
  size_t iptr;
  if(iy<nj && iz<nz1_g)
  {
    iptr     = (iz+nk1) * ny + (iy+nj1);
    iptr_b   = iz*nj + iy;
    for(int i=0; i<2*num_of_vars; i++)
    {
      sbuff_z1_fault[iptr_b + i*nz1_g*nj] = fw_cur[iptr + i*siz_slice_yz];
    }
  }

  return;
}

__global__ void
blk_macdrp_pack_fault_mesg_z1(
             float *fw_cur, float *sbuff_z2_fault, size_t siz_slice_yz, 
             int num_of_vars_fault, int ny, int nj1, int nk2, int nj, int nz2_g)
{
  int iy = blockIdx.x * blockDim.x + threadIdx.x;
  int iz = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iptr_b;
  size_t iptr;
  if(iy<nj && iz<nz2_g)
  {
    iptr     = (iz+nk2-nz2_g+1) * ny + (iy+nj1);
    iptr_b   = iz*nj + iy;
    for(int i=0; i<2*num_of_vars; i++)
    {
      sbuff_z2_fault[iptr_b + i*nz2_g*nj] = fw_cur[iptr + i*siz_slice_yz];
    }
  }

  return;
}

int 
blk_macdrp_unpack_fault_mesg_gpu(float *fw_cur, 
                                 fd_t *fd,
                                 gdinfo_t *gdinfo,
                                 mympi_t *mympi, 
                                 int ipair_mpi,
                                 int istage_mpi,
                                 int num_of_vars_fault,
                                 int *neighid)
{
  int nj1 = gdinfo->nj1;
  int nj2 = gdinfo->nj2;
  int nk1 = gdinfo->nk1;
  int nk2 = gdinfo->nk2;
  int nj = gdinfo->nj;
  int nk = gdinfo->nk;
  size_t siz_slice_yz = gdinfo->siz_slice_yz;
  
  fd_op_t *fdy_op = fd->pair_fdy_op[ipair_mpi][istage_mpi];
  fd_op_t *fdz_op = fd->pair_fdz_op[ipair_mpi][istage_mpi];
  // ghost point
  int ny1_g = fdy_op->right_len;
  int ny2_g = fdy_op->left_len;
  int nz1_g = fdz_op->right_len;
  int nz2_g = fdz_op->left_len;

  size_t siz_rbuff_y1_fault = mympi->pair_siz_rbuff_y_fault1[ipair_mpi][istage_mpi];
  size_t siz_rbuff_y2_fault = mympi->pair_siz_rbuff_y_fault2[ipair_mpi][istage_mpi];
  size_t siz_rbuff_z1_fault = mympi->pair_siz_rbuff_z_fault1[ipair_mpi][istage_mpi];

  float *rbuff_y1_fault = mympi->rbuff_fault;
  float *rbuff_y2_fault = rbuff_y1_fault + siz_rbuff_y1_fault;
  float *rbuff_z1_fault = rbuff_y2_fault + siz_rbuff_y2_fault;
  float *rbuff_z2_fault = rbuff_z1_fault + siz_rbuff_z1_fault;
  {
    dim3 block(ny2_g,8);
    dim3 grid;
    grid.x = (ny2_g + block.x -1) / block.x;
    grid.y = (nk + block.y - 1) / block.y;
    blk_macdrp_unpack_fault_mesg_y1<<< grid, block >>>(
           fw_cur, rbuff_y1_fault, siz_slice_yz, 
           num_of_vars_fault, ny, nj1, nk1, ny2_g, nk, neighid);
    CUDACHECK(cudaDeviceSynchronize());
  }
  {
    dim3 block(ny1_g,8);
    dim3 grid;
    grid.x = (ny1_g + block.x -1) / block.x;
    grid.y = (nk + block.y - 1) / block.y;
    blk_macdrp_unpack_fault_mesg_y2<<< grid, block >>>(
           fw_cur, rbuff_y2_fault, siz_slice_yz,
           num_of_vars_fault, ny, nj2, nk1, ny1_g, nk, neighid);
    CUDACHECK(cudaDeviceSynchronize());
  }
  {
    dim3 block(8,nz2_g);
    dim3 grid;
    grid.x = (nj + block.x -1) / block.x;
    grid.y = (nz2_g + block.y - 1) / block.y;
    blk_macdrp_unpack_fault_mesg_z1<<< grid, block >>>(
           fw_cur, rbuff_z1_fault, siz_slice_yz, 
           num_of_vars_fault, ny, nj1, nk1, nj, nz2_g, neighid);
    CUDACHECK(cudaDeviceSynchronize());
  }
  {
    dim3 block(8,nz1_g);
    dim3 grid;
    grid.x = (nj + block.x -1) / block.x;
    grid.y = (nz1_g + block.y - 1) / block.y;
    blk_macdrp_unpack_fault_mesg_z2<<< grid, block >>>(
           fw_cur, rbuff_z2_fault, siz_slice_yz, 
           num_of_vars_fault, ny, nj1, nk2, nj, nz1_g, neighid);
    CUDACHECK(cudaDeviceSynchronize());
  }

  return 0;
}

__global__ void
blk_macdrp_unpack_mesg_y1(
           float *fw_cur, float *rbuff_y1_fault, size_t siz_slice_yz, 
           int num_of_vars, int ny, int nj1, int nk1, int ny2_g, int nk, int *neighid)
{
  int iy = blockIdx.x * blockDim.x + threadIdx.x;
  int iz = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iptr_b;
  size_t iptr;
  if (neighid[2] != MPI_PROC_NULL) {
    if(iy<ny2_g && iz<nk){
      iptr   = (iz+nk1) * ny + (iy+nj1-ny2_g);
      iptr_b = iz*ny2_g + iy;
      for(int i=0; i<2*num_of_vars; i++)
      {
        fw_cur[iptr + i*siz_slice_yz] = rbuff_y1_fault[iptr_b+ i*ny2_g*nk];
      }
    }
  }
  return;
}

__global__ void
blk_macdrp_unpack_mesg_y2(
           float *fw_cur, float *rbuff_y2_fault, size_t siz_slice_yz, 
           int num_of_vars, int ny, int nj2, int nk1, int ny1_g, int nk, int *neighid)
{
  int iy = blockIdx.x * blockDim.x + threadIdx.x;
  int iz = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iptr_b;
  size_t iptr;
  if (neighid[3] != MPI_PROC_NULL) {
    if(iy<ny1_g && iz<nk){
      iptr   = (iz+nk1) * ny + (iy+nj2+1);
      iptr_b = iz*ny1_g + iy;
      for(int i=0; i<2*num_of_vars; i++)
      {
        fw_cur[iptr + i*siz_slice_yz] = rbuff_y2_fault[iptr_b+ i*ny1_g*nk];
      }
    }
  }
  return;
}

__global__ void
blk_macdrp_unpack_mesg_z1(
           float *fw_cur, float *rbuff_z1_fault, size_t siz_slice_yz, 
           int num_of_vars, int ny, int nj1, int nk1, int nj, int nz2_g, int *neighid)
{
  int iy = blockIdx.x * blockDim.x + threadIdx.x;
  int iz = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iptr_b;
  size_t iptr;
  if (neighid[4] != MPI_PROC_NULL) {
    if(iy<nj && iz<nz2_g){
      iptr   = (iz+nk1-nz2_g) * ny + (iy+nj1);
      iptr_b = iz*nj + iy;
      for(int i=0; i<2*num_of_vars; i++)
      {
        fw_cur[iptr + i*siz_slice_yz] = rbuff_z1_fault[iptr_b+ i*nz2_g*nj];
      }
    }
  }
  return;
}

__global__ void
blk_macdrp_unpack_mesg_z2(
           float *fw_cur, float *rbuff_z2_fault, size_t siz_slice_yz, 
           int num_of_vars, int ny, int nj1, int nk2, int nj, int nz1_g, int *neighid)
{
  int iy = blockIdx.x * blockDim.x + threadIdx.x;
  int iz = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iptr_b;
  size_t iptr;
  if (neighid[5] != MPI_PROC_NULL) {
    if(iy<nj && iz<nz1_g){
      iptr   = (iz+nk2+1) * ny + (iy+nj1);
      iptr_b = iz*nj + iy;
      for(int i=0; i<2*num_of_vars; i++)
      {
        fw_cur[iptr + i*siz_slice_yz] = rbuff_z2_fault[iptr_b+ i*nz1_g*nj];
      }
    }
  }
  return;
}
/*********************************************************************
 * estimate dt
 *********************************************************************/

int
blk_dt_esti_curv(gdinfo_t *gdinfo, gd_t *gdcurv, md_t *md,
    float CFL, float *dtmax, float *dtmaxVp, float *dtmaxL,
    int *dtmaxi, int *dtmaxj, int *dtmaxk)
{
  int ierr = 0;

  float dtmax_local = 1.0e10;
  float Vp;

  float *x3d = gdcurv->x3d;
  float *y3d = gdcurv->y3d;
  float *z3d = gdcurv->z3d;

  for (int k = gdinfo->nk1; k < gdinfo->nk2; k++)
  {
    for (int j = gdinfo->nj1; j < gdinfo->nj2; j++)
    {
      for (int i = gdinfo->ni1; i < gdinfo->ni2; i++)
      {
        size_t iptr = i + j * gdinfo->siz_line + k * gdinfo->siz_slice;

        if (md->medium_type == CONST_MEDIUM_ELASTIC_ISO) {
          Vp = sqrt( (md->lambda[iptr] + 2.0 * md->mu[iptr]) / md->rho[iptr] );
        } else if (md->medium_type == CONST_MEDIUM_ELASTIC_VTI) {
          float Vpv = sqrt( md->c33[iptr] / md->rho[iptr] );
          float Vph = sqrt( md->c11[iptr] / md->rho[iptr] );
          Vp = Vph > Vpv ? Vph : Vpv;
        } else if (md->medium_type == CONST_MEDIUM_ELASTIC_ANISO) {
          // need to implement accurate solution
          Vp = sqrt( md->c11[iptr] / md->rho[iptr] );
        } else if (md->medium_type == CONST_MEDIUM_ACOUSTIC_ISO) {
          Vp = sqrt( md->kappa[iptr] / md->rho[iptr] );
        }

        float dtLe = 1.0e20;
        float p0[] = { x3d[iptr], y3d[iptr], z3d[iptr] };

        // min L to 8 adjacent planes
        for (int kk = -1; kk <=1; kk++) {
          for (int jj = -1; jj <= 1; jj++) {
            for (int ii = -1; ii <= 1; ii++) {
              if (ii != 0 && jj !=0 && kk != 0)
              {
                float p1[] = { x3d[iptr-ii], y3d[iptr-ii], z3d[iptr-ii] };
                float p2[] = { x3d[iptr-jj*gdinfo->siz_line],
                               y3d[iptr-jj*gdinfo->siz_line],
                               z3d[iptr-jj*gdinfo->siz_line] };
                float p3[] = { x3d[iptr-kk*gdinfo->siz_slice],
                               y3d[iptr-kk*gdinfo->siz_slice],
                               z3d[iptr-kk*gdinfo->siz_slice] };

                float L = fdlib_math_dist_point2plane(p0, p1, p2, p3);

                if (dtLe > L) dtLe = L;
              }
            }
          }
        }

        // convert to dt
        float dt_point = CFL / Vp * dtLe;

        // if smaller
        if (dt_point < dtmax_local) {
          dtmax_local = dt_point;
          *dtmaxi = i;
          *dtmaxj = j;
          *dtmaxk = k;
          *dtmaxVp = Vp;
          *dtmaxL  = dtLe;
        }

      } // i
    } // i
  } //k

  *dtmax = dtmax_local;

  return ierr;
}

float
blk_keep_two_digi(float dt)
{
  char str[40];
  float dt_2;

  sprintf(str, "%6.4e", dt);

  str[3] = '0';
  str[4] = '0';
  str[5] = '0';

  sscanf(str, "%f", &dt_2);
  
  return dt_2;
}
