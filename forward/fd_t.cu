/*********************************************************************
 * setup fd operators
 **********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "constants.h"
#include "fdlib_mem.h"
#include "fd_t.h"

/*
 * set MacCormack DRP scheme with rk 4
 */

int 
fd_set_macdrp(fd_t *fd)
{
  int ierr = 0;

  //----------------------------------------------------------------------------
  // 4th rk scheme
  //----------------------------------------------------------------------------

  fd->num_rk_stages = 4;

  fd->rk_a = (float *) fdlib_mem_malloc_1d(
                          fd->num_rk_stages*sizeof(float),"fd_set_macdrp");
  fd->rk_b = (float *) fdlib_mem_malloc_1d(
                          fd->num_rk_stages*sizeof(float),"fd_set_macdrp");
  fd->rk_rhs_time = (float *) fdlib_mem_malloc_1d(
                          fd->num_rk_stages*sizeof(float), "fd_set_macdrp");

  fd->rk_a[0] = 0.5;
  fd->rk_a[1] = 0.5;
  fd->rk_a[2] = 1.0;

  fd->rk_b[0] = 1.0/6.0;
  fd->rk_b[1] = 1.0/3.0;
  fd->rk_b[2] = 1.0/3.0;
  fd->rk_b[3] = 1.0/6.0;

  // rhs side time in terms of percentage time step, useful for source stf
  fd->rk_rhs_time[0] = 0.0;
  fd->rk_rhs_time[1] = 0.5;
  fd->rk_rhs_time[2] = 0.5;
  fd->rk_rhs_time[3] = 1.0;

  //----------------------------------------------------------------------------
  // MacCormack-type scheme
  //----------------------------------------------------------------------------

  fd->CFL = 1.3;

  // set max
  fd->fdx_max_len = 5;
  fd->fdy_max_len = 5;
  fd->fdz_max_len = 5;
  fd->fdx_nghosts = 3;
  fd->fdy_nghosts = 3;
  fd->fdz_nghosts = 3;

  //----------------------------------------------------------------------------
  // 1d scheme for different points to surface, or different half length
  //----------------------------------------------------------------------------
#define  m_mac_max_len   5
#define  m_mac_num_lay   4
  // op index at all layers, zero not used point
  int  mac_all_indx[m_mac_num_lay][2][m_mac_max_len] =
  {
    { // back/forw at free, no use
      { -1,0,0,0,0 },
      {    0,1,0,0,0 } // 0, at free, no use
    },
    { // 1, 2nd
      { -1,0,0,0,0 },
      {    0,1,0,0,0 }
    }, 
    { // 2, 4th
      { -2,-1, 0,0,0 },
      {        0,1,2,0,0 }
    },
    { // 3, drp, normal op
      { -3,-2,-1,0,1 }, 
      {       -1,0,1,2,3 }
    }
  };

  // op coef at all layers
  float mac_all_coef[m_mac_num_lay][2][m_mac_max_len] =
  {
    { // at free
      {-1.0, 1.0, 0.0, 0.0, 0.0},
      {     -1.0, 1.0, 0.0, 0.0, 0.0 }
    },
    { // 1, 2nd
      {-1.0, 1.0, 0.0, 0.0, 0.0},
      {     -1.0, 1.0, 0.0, 0.0, 0.0 }
    }, 
    { // 2, 4th
      { 1.0/6.0, -8.0/6.0, 7.0/6.0, 0.0, 0.0 },
      {                   -7.0/6.0, 8.0/6.0, -1.0/6.0, 0.0, 0.0 } 
    },
    { // 3, drp, normal op
      { -0.04168 , 0.3334, -1.233 , 0.6326 , 0.30874 },
      {                   -0.30874,-0.6326 ,1.233 ,-0.3334,0.04168 }
    } 
  };

  int  mac_all_total_len[m_mac_num_lay][2] = 
  { 
    { 2, 2 },
    { 2, 2 },
    { 3, 3 },
    { 5, 5 }
  };

  // half len without cur point
  int  mac_all_half_len[m_mac_num_lay][2] = 
  { 
    { 1, 1 },
    { 1, 1 },
    { 2, 2 },
    { 3, 3 }
  };

  int  mac_all_left_len[m_mac_num_lay][2] = 
  { 
    { 1, 0 },
    { 1, 0 },
    { 2, 0 },
    { 3, 1 }
  };

  int  mac_all_right_len[m_mac_num_lay][2] = 
  { 
    { 0, 1 },
    { 0, 1 },
    { 0, 2 },
    { 1, 3 }
  };


  fd->num_of_pairs = 8;
  // BBB FFB FFF BBF
  // BFB FBB FBF BFF
  int FD_Flags[8][CONST_NDIM] = // 8 pairs, x/y/z 3 dim
  {
    { 0,  0,  0 },
    { 1,  1,  0 },
    { 1,  1,  1 },
    { 0,  0,  1 },
    { 0,  1,  0 },
    { 1,  0,  0 },
    { 1,  0,  1 },
    { 0,  1,  1 }
  }; //  0(B) 1(F)

  // alloc

  fd->pair_fdx_op = (fd_op_t ***)malloc(fd->num_of_pairs * sizeof(fd_op_t **));
  fd->pair_fdy_op = (fd_op_t ***)malloc(fd->num_of_pairs * sizeof(fd_op_t **));
  fd->pair_fdz_op = (fd_op_t ***)malloc(fd->num_of_pairs * sizeof(fd_op_t **));

  for (int ipair = 0; ipair < fd->num_of_pairs; ipair++)
  {
    fd->pair_fdx_op[ipair] = (fd_op_t **)malloc(fd->num_rk_stages * sizeof(fd_op_t *));
    fd->pair_fdy_op[ipair] = (fd_op_t **)malloc(fd->num_rk_stages * sizeof(fd_op_t *));
    fd->pair_fdz_op[ipair] = (fd_op_t **)malloc(fd->num_rk_stages * sizeof(fd_op_t *));

    for (int istage = 0; istage < fd->num_rk_stages; istage++)
    {
      fd->pair_fdx_op[ipair][istage] = (fd_op_t *)malloc(sizeof(fd_op_t));
      fd->pair_fdy_op[ipair][istage] = (fd_op_t *)malloc(sizeof(fd_op_t));
      fd->pair_fdz_op[ipair][istage] = (fd_op_t *)malloc(sizeof(fd_op_t)); 
    }
  }

  // set
  for (int ipair=0; ipair < fd->num_of_pairs; ipair++)
  {
    int idir0 = FD_Flags[ipair][0];
    int jdir0 = FD_Flags[ipair][1];
    int kdir0 = FD_Flags[ipair][2];

    for (int istage=0; istage < fd->num_rk_stages; istage++)
    {
      // switch forw/back based on mod value
      int idir = (idir0 + istage) % 2;
      int jdir = (jdir0 + istage) % 2;
      int kdir = (kdir0 + istage) % 2;
      fd_op_t *fdx_op = fd->pair_fdx_op[ipair][istage];
      fdx_op->total_len = mac_all_total_len[m_mac_num_lay-1][idir];
      fdx_op->left_len  = mac_all_left_len [m_mac_num_lay-1][idir];
      fdx_op->right_len = mac_all_right_len[m_mac_num_lay-1][idir];
      fdx_op->dir = idir;

      fdx_op->indx = (int   *)malloc(fdx_op->total_len * sizeof(int));
      fdx_op->coef = (float *)malloc(fdx_op->total_len * sizeof(float));
      for (int n=0; n < fdx_op->total_len; n++)
      {
        fdx_op->indx[n] = mac_all_indx[m_mac_num_lay-1][idir][n];
        fdx_op->coef[n] = mac_all_coef[m_mac_num_lay-1][idir][n];
      }

      fd_op_t *fdy_op = fd->pair_fdy_op[ipair][istage];
      fdy_op->total_len = mac_all_total_len[m_mac_num_lay-1][jdir];
      fdy_op->left_len  = mac_all_left_len [m_mac_num_lay-1][jdir];
      fdy_op->right_len = mac_all_right_len[m_mac_num_lay-1][jdir];
      fdy_op->dir = jdir;
      fdy_op->indx = (int   *)malloc(fdy_op->total_len * sizeof(int));
      fdy_op->coef = (float *)malloc(fdy_op->total_len * sizeof(float));
      for (int n=0; n < fdy_op->total_len; n++)
      {
        fdy_op->indx[n] = mac_all_indx[m_mac_num_lay-1][jdir][n];
        fdy_op->coef[n] = mac_all_coef[m_mac_num_lay-1][jdir][n];
      }

      fd_op_t *fdz_op = fd->pair_fdz_op[ipair][istage];
      fdz_op->total_len = mac_all_total_len[m_mac_num_lay-1][kdir];
      fdz_op->left_len  = mac_all_left_len [m_mac_num_lay-1][kdir];
      fdz_op->right_len = mac_all_right_len[m_mac_num_lay-1][kdir];
      fdz_op->dir = kdir;

      fdz_op->indx = (int   *)malloc(fdz_op->total_len * sizeof(int));
      fdz_op->coef = (float *)malloc(fdz_op->total_len * sizeof(float));
      for (int n=0; n < fdz_op->total_len; n++)
      {
        fdz_op->indx[n] = mac_all_indx[m_mac_num_lay-1][kdir][n];
        fdz_op->coef[n] = mac_all_coef[m_mac_num_lay-1][kdir][n];
      }

    } // istage
  } // ipair

  return ierr;
}

