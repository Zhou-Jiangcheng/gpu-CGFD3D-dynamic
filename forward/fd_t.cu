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
#define  m_mac_num_lay   3
  int  mac_all_left_len[m_mac_num_lay][2] = 
  { 
    { 1, 0 },
    { 2, 0 },
    { 3, 1 }
  };

  int  mac_all_right_len[m_mac_num_lay][2] = 
  { 
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
      fdx_op->left_len  = mac_all_left_len[m_mac_num_lay-1][idir];
      fdx_op->right_len = mac_all_right_len[m_mac_num_lay-1][idir];
      fdx_op->dir = idir;

      fd_op_t *fdy_op = fd->pair_fdy_op[ipair][istage];
      fdy_op->left_len  = mac_all_left_len[m_mac_num_lay-1][jdir];
      fdy_op->right_len = mac_all_right_len[m_mac_num_lay-1][jdir];
      fdy_op->dir = jdir;

      fd_op_t *fdz_op = fd->pair_fdz_op[ipair][istage];
      fdz_op->left_len  = mac_all_left_len[m_mac_num_lay-1][kdir];
      fdz_op->right_len = mac_all_right_len[m_mac_num_lay-1][kdir];
      fdz_op->dir = kdir;
    } // istage
  } // ipair

  return ierr;
}

