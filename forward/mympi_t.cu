/*********************************************************************
 * mpi for this package
 **********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mympi_t.h"

//
// set grid size
//
int
mympi_set(mympi_t *mympi,
          int number_of_mpiprocs_x,
          int number_of_mpiprocs_y,
          int number_of_mpiprocs_z,
          MPI_Comm comm, 
          const int myid, const int verbose)
{
  int ierr = 0;

  mympi->nprocx = number_of_mpiprocs_x;
  mympi->nprocy = number_of_mpiprocs_y;
  mympi->nprocz = number_of_mpiprocs_z;

  mympi->myid = myid;
  mympi->comm = comm;

  // mpi topo
  int ndims[3]   = {number_of_mpiprocs_x, number_of_mpiprocs_y, number_of_mpiprocs_z};
  int periods[3] = {0,0,0};
  int reorder = 0; 

  // create Cartesian topology
  MPI_Cart_create(comm, 3, ndims, periods, reorder, &(mympi->topocomm));
  

  // get my local x,y coordinates
  MPI_Cart_coords(mympi->topocomm, mympi->myid, 3, mympi->topoid);

  // neighour
  MPI_Cart_shift(mympi->topocomm, 0, 1, &(mympi->neighid[0]), &(mympi->neighid[1]));
  MPI_Cart_shift(mympi->topocomm, 1, 1, &(mympi->neighid[2]), &(mympi->neighid[3]));
  MPI_Cart_shift(mympi->topocomm, 2, 1, &(mympi->neighid[4]), &(mympi->neighid[5]));

  return ierr;
}

int
mympi_print(mympi_t *mympi)
{    
  fprintf(stdout, "\n-------------------------------------------------------\n");
  fprintf(stdout, "print mympi info:\n");
  fprintf(stdout, "-------------------------------------------------------\n\n");

  fprintf(stdout, " myid = %d, topoid[%d,%d,%d]\n", mympi->myid,mympi->topoid[0],mympi->topoid[1],mympi->topoid[2]);
  fprintf(stdout, " neighid_x[%d,%d]\n", mympi->neighid[0], mympi->neighid[1]);
  fprintf(stdout, " neighid_y[%d,%d]\n", mympi->neighid[2], mympi->neighid[3]);
  fprintf(stdout, " neighid_z[%d,%d]\n", mympi->neighid[4], mympi->neighid[5]);
  
  return 0;
}
