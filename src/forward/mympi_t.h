#ifndef MY_MPI_H
#define MY_MPI_H

#include <mpi.h>

#include "constants.h"

/*******************************************************************************
 * structure
 ******************************************************************************/

typedef struct {
  int       nprocx;
  int       nprocy;
  int       nprocz;

  int       myid;
  MPI_Comm  comm;

  int    topoid[3];
  int    neighid[CONST_NDIM_2];
  MPI_Comm    topocomm;

  size_t siz_sbuff;
  size_t siz_rbuff;
  float *sbuff;
  float *rbuff;

  size_t siz_sbuff_fault;
  size_t siz_rbuff_fault;
  float *sbuff_fault;
  float *rbuff_fault;

  // for macdrp
  size_t **pair_siz_sbuff_y1;
  size_t **pair_siz_sbuff_y2;
  size_t **pair_siz_sbuff_z1;
  size_t **pair_siz_sbuff_z2;

  size_t **pair_siz_rbuff_y1;
  size_t **pair_siz_rbuff_y2;
  size_t **pair_siz_rbuff_z1;
  size_t **pair_siz_rbuff_z2;

  size_t **pair_siz_sbuff_y1_fault;
  size_t **pair_siz_sbuff_y2_fault;
  size_t **pair_siz_sbuff_z1_fault;
  size_t **pair_siz_sbuff_z2_fault;

  size_t **pair_siz_rbuff_y1_fault;
  size_t **pair_siz_rbuff_y2_fault;
  size_t **pair_siz_rbuff_z1_fault;
  size_t **pair_siz_rbuff_z2_fault;

  MPI_Request ***pair_r_reqs;
  MPI_Request ***pair_s_reqs;

  MPI_Request ***pair_r_reqs_fault;
  MPI_Request ***pair_s_reqs_fault;

} mympi_t;

/*******************************************************************************
 * function prototype
 ******************************************************************************/

int
mympi_set(mympi_t *mympi,
          int number_of_mpiprocs_x,
          int number_of_mpiprocs_y,
          int number_of_mpiprocs_z,
          MPI_Comm comm, 
          const int myid, const int verbose);

int
mympi_print(mympi_t *mympi);

#endif
