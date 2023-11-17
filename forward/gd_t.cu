/*
********************************************************************************
* Curve grid metric calculation using MacCormack scheme                        *
********************************************************************************
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "netcdf.h"

#include "fdlib_mem.h"
#include "fdlib_math.h"
#include "fd_t.h"
#include "gd_t.h"
#include "constants.h"

int 
gd_curv_init(gdcurv_t *gdcurv)
{
  /*
   * 0-2: x3d, y3d, z3d
   */
  gdcurv->ncmp = CONST_NDIM;
  
  // vars
  gdcurv->v4d = (float *) fdlib_mem_calloc_1d_float(
                  gdcurv->siz_icmp * gdcurv->ncmp, 0.0, "gd_curv_init");
  if (gdcurv->v4d == NULL) {
      fprintf(stderr,"Error: failed to alloc coord vars\n");
      fflush(stderr);
  }
  
  // position of each v4d
  size_t *cmp_pos = (size_t *) fdlib_mem_calloc_1d_sizet(gdcurv->ncmp,
                                                         0,
                                                         "gd_curv_init");
  
  // name of each v4d
  char **cmp_name = (char **) fdlib_mem_malloc_2l_char(gdcurv->ncmp,
                                                       CONST_MAX_STRLEN,
                                                       "gd_curv_init");
  
  // set value
  int icmp = 0;
  cmp_pos[icmp] = icmp * gdcurv->siz_icmp;
  sprintf(cmp_name[icmp],"%s","x");
  gdcurv->x3d = gdcurv->v4d + cmp_pos[icmp];

  icmp += 1;
  cmp_pos[icmp] = icmp * gdcurv->siz_icmp;
  sprintf(cmp_name[icmp],"%s","y");
  gdcurv->y3d = gdcurv->v4d + cmp_pos[icmp];

  icmp += 1;
  cmp_pos[icmp] = icmp * gdcurv->siz_icmp;
  sprintf(cmp_name[icmp],"%s","z");
  gdcurv->z3d = gdcurv->v4d + cmp_pos[icmp];
  
  // set pointer
  gdcurv->cmp_pos  = cmp_pos;
  gdcurv->cmp_name = cmp_name;

  // alloc AABB vars
  gdcurv->cell_xmin = (float *) fdlib_mem_calloc_1d_float(
                  gdcurv->siz_icmp, 0.0, "gd_curv_init");
  gdcurv->cell_xmax = (float *) fdlib_mem_calloc_1d_float(
                  gdcurv->siz_icmp, 0.0, "gd_curv_init");
  gdcurv->cell_ymin = (float *) fdlib_mem_calloc_1d_float(
                  gdcurv->siz_icmp, 0.0, "gd_curv_init");
  gdcurv->cell_ymax = (float *) fdlib_mem_calloc_1d_float(
                  gdcurv->siz_icmp, 0.0, "gd_curv_init");
  gdcurv->cell_zmin = (float *) fdlib_mem_calloc_1d_float(
                  gdcurv->siz_icmp, 0.0, "gd_curv_init");
  gdcurv->cell_zmax = (float *) fdlib_mem_calloc_1d_float(
                  gdcurv->siz_icmp, 0.0, "gd_curv_init");
  if (gdcurv->cell_zmax == NULL) {
      fprintf(stderr,"Error: failed to alloc coord AABB vars\n");
      fflush(stderr);
  }

  gdcurv->tile_istart = (int *) fdlib_mem_calloc_1d_int(
                        GD_TILE_NX, 0.0, "gd_curv_init");
  gdcurv->tile_iend   = (int *) fdlib_mem_calloc_1d_int(
                        GD_TILE_NX, 0.0, "gd_curv_init");
  gdcurv->tile_jstart = (int *) fdlib_mem_calloc_1d_int(
                        GD_TILE_NY, 0.0, "gd_curv_init");
  gdcurv->tile_jend   = (int *) fdlib_mem_calloc_1d_int(
                        GD_TILE_NY, 0.0, "gd_curv_init");
  gdcurv->tile_kstart = (int *) fdlib_mem_calloc_1d_int(
                        GD_TILE_NZ, 0.0, "gd_curv_init");
  gdcurv->tile_kend   = (int *) fdlib_mem_calloc_1d_int(
                        GD_TILE_NZ, 0.0, "gd_curv_init");

  int size = GD_TILE_NX * GD_TILE_NY * GD_TILE_NZ;
  gdcurv->tile_xmin = (float *) fdlib_mem_calloc_1d_float(
                        size, 0.0, "gd_curv_init");
  gdcurv->tile_xmax = (float *) fdlib_mem_calloc_1d_float(
                        size, 0.0, "gd_curv_init");
  gdcurv->tile_ymin = (float *) fdlib_mem_calloc_1d_float(
                        size, 0.0, "gd_curv_init");
  gdcurv->tile_ymax = (float *) fdlib_mem_calloc_1d_float(
                        size, 0.0, "gd_curv_init");
  gdcurv->tile_zmin = (float *) fdlib_mem_calloc_1d_float(
                        size, 0.0, "gd_curv_init");
  gdcurv->tile_zmax = (float *) fdlib_mem_calloc_1d_float(
                        size, 0.0, "gd_curv_init");

  return 0;
}

int 
gd_curv_metric_init(gdcurv_t        *gdcurv,
                    gd_metric_t *metric)
{
  const int num_grid_vars = 10;
  /*
   * 0: jac
   * 1-3: xi_x, xi_y, xi_z
   * 4-6: eta_x, eta_y, eta_z
   * 7-9: zeta_x, zeta_y, zeta_z
   */

  metric->nx   = gdcurv->nx;
  metric->ny   = gdcurv->ny;
  metric->nz   = gdcurv->nz;
  metric->ncmp = num_grid_vars;

  metric->siz_iy  = metric->nx;
  metric->siz_iz  = metric->nx * metric->ny;
  metric->siz_icmp = metric->nx * metric->ny * metric->nz;
  
  // vars
  metric->v4d = (float *) fdlib_mem_calloc_1d_float(
                  metric->siz_icmp * metric->ncmp, 0.0, "gd_curv_init_g4d");
  if (metric->v4d == NULL) {
      fprintf(stderr,"Error: failed to alloc metric vars\n");
      fflush(stderr);
  }
  
  // position of each v4d
  size_t *cmp_pos = (size_t *) fdlib_mem_calloc_1d_sizet(metric->ncmp,
                                                         0, 
                                                         "gd_curv_metric_init");
  
  // name of each v4d
  char **cmp_name = (char **) fdlib_mem_malloc_2l_char(metric->ncmp,
                                                       CONST_MAX_STRLEN,
                                                       "gd_curv_metric_init");
  
  // set value
  for (int icmp=0; icmp < metric->ncmp; icmp++)
  {
    cmp_pos[icmp] = icmp * metric->siz_icmp;
  }

  int icmp = 0;
  sprintf(cmp_name[icmp],"%s","jac");
  metric->jac = metric->v4d + cmp_pos[icmp];

  icmp += 1;
  sprintf(cmp_name[icmp],"%s","xi_x");
  metric->xi_x = metric->v4d + cmp_pos[icmp];
  
  icmp += 1;
  sprintf(cmp_name[icmp],"%s","xi_y");
  metric->xi_y = metric->v4d + cmp_pos[icmp];
  
  icmp += 1;
  sprintf(cmp_name[icmp],"%s","xi_z");
  metric->xi_z = metric->v4d + cmp_pos[icmp];
  
  icmp += 1;
  sprintf(cmp_name[icmp],"%s","eta_x");
  metric->eta_x = metric->v4d + cmp_pos[icmp];
  
  icmp += 1;
  sprintf(cmp_name[icmp],"%s","eta_y");
  metric->eta_y = metric->v4d + cmp_pos[icmp];
  
  icmp += 1;
  sprintf(cmp_name[icmp],"%s","eta_z");
  metric->eta_z = metric->v4d + cmp_pos[icmp];
  
  icmp += 1;
  sprintf(cmp_name[icmp],"%s","zeta_x");
  metric->zeta_x = metric->v4d + cmp_pos[icmp];
  
  icmp += 1;
  sprintf(cmp_name[icmp],"%s","zeta_y");
  metric->zeta_y = metric->v4d + cmp_pos[icmp];
  
  icmp += 1;
  sprintf(cmp_name[icmp],"%s","zeta_z");
  metric->zeta_z = metric->v4d + cmp_pos[icmp];
  
  // set pointer
  metric->cmp_pos  = cmp_pos;
  metric->cmp_name = cmp_name;

  return 0;
}

//
// need to change to use fdlib_math.c
//
int
gd_curv_metric_cal(gdcurv_t    *gdcurv,
                   gd_metric_t *metric)
{
  int ni1 = gdcurv->ni1;
  int ni2 = gdcurv->ni2;
  int nj1 = gdcurv->nj1;
  int nj2 = gdcurv->nj2;
  int nk1 = gdcurv->nk1;
  int nk2 = gdcurv->nk2;
  int nx  = gdcurv->nx;
  int ny  = gdcurv->ny;
  int nz  = gdcurv->nz;
  size_t siz_iy   = gdcurv->siz_iy;
  size_t siz_iz   = gdcurv->siz_iz;
  size_t siz_icmp = gdcurv->siz_icmp;

  // point to each var
  float *x3d  = gdcurv->x3d;
  float *y3d  = gdcurv->y3d;
  float *z3d  = gdcurv->z3d;
  float *jac3d= metric->jac;
  float *xi_x = metric->xi_x;
  float *xi_y = metric->xi_y;
  float *xi_z = metric->xi_z;
  float *et_x = metric->eta_x;
  float *et_y = metric->eta_y;
  float *et_z = metric->eta_z;
  float *zt_x = metric->zeta_x;
  float *zt_y = metric->zeta_y;
  float *zt_z = metric->zeta_z;
  float *x3d_ptr;
  float *y3d_ptr;
  float *z3d_ptr; 
  float x_xi, x_et, x_zt;
  float y_xi, y_et, y_zt;
  float z_xi, z_et, z_zt;
  float jac;
  float vec1[3], vec2[3], vec3[3], vecg[3];

  for (size_t k = nk1; k <= nk2; k++){
    for (size_t j = nj1; j <= nj2; j++) {
      for (size_t i = ni1; i <= ni2; i++)
      {
        size_t iptr = i + j * siz_iy + k * siz_iz;

        x_xi = 0.0; x_et = 0.0; x_zt = 0.0;
        y_xi = 0.0; y_et = 0.0; y_zt = 0.0;
        z_xi = 0.0; z_et = 0.0; z_zt = 0.0;

        x3d_ptr = x3d + iptr;
        y3d_ptr = y3d + iptr;
        z3d_ptr = z3d + iptr;
        M_FD_SHIFT_PTR_CENTER(x_xi, x3d_ptr, 1);
        M_FD_SHIFT_PTR_CENTER(y_xi, y3d_ptr, 1);
        M_FD_SHIFT_PTR_CENTER(z_xi, z3d_ptr, 1);

        M_FD_SHIFT_PTR_CENTER(x_et, x3d_ptr, siz_iy);
        M_FD_SHIFT_PTR_CENTER(y_et, y3d_ptr, siz_iy);
        M_FD_SHIFT_PTR_CENTER(z_et, z3d_ptr, siz_iy);

        M_FD_SHIFT_PTR_CENTER(x_zt, x3d_ptr, siz_iz);
        M_FD_SHIFT_PTR_CENTER(y_zt, y3d_ptr, siz_iz);
        M_FD_SHIFT_PTR_CENTER(z_zt, z3d_ptr, siz_iz);

        vec1[0] = x_xi; vec1[1] = y_xi; vec1[2] = z_xi;
        vec2[0] = x_et; vec2[1] = y_et; vec2[2] = z_et;
        vec3[0] = x_zt; vec3[1] = y_zt; vec3[2] = z_zt;

        fdlib_math_cross_product(vec1, vec2, vecg);
        jac = fdlib_math_dot_product(vecg, vec3);
        jac3d[iptr]  = jac;

        fdlib_math_cross_product(vec2, vec3, vecg);
        xi_x[iptr] = vecg[0] / jac;
        xi_y[iptr] = vecg[1] / jac;
        xi_z[iptr] = vecg[2] / jac;

        fdlib_math_cross_product(vec3, vec1, vecg);
        et_x[iptr] = vecg[0] / jac;
        et_y[iptr] = vecg[1] / jac;
        et_z[iptr] = vecg[2] / jac;

        fdlib_math_cross_product(vec1, vec2, vecg);
        zt_x[iptr] = vecg[0] / jac;
        zt_y[iptr] = vecg[1] / jac;
        zt_z[iptr] = vecg[2] / jac;
      }
    }
  }
    
  //mirror_symmetry(gdcurv,metric->v4d,metric->ncmp);
  geometric_symmetry(gdcurv,metric->v4d,metric->ncmp);

  return 0;
}

//
// exchange metics/coords
//
int
gd_curv_exchange(gdcurv_t *gdcurv,
                 float *g3d,
                 int ncmp,
                 int *neighid,
                 MPI_Comm topocomm)
{
  int nx  = gdcurv->nx;
  int ny  = gdcurv->ny;
  int nz  = gdcurv->nz;
  int ni1 = gdcurv->ni1;
  int ni2 = gdcurv->ni2;
  int nj1 = gdcurv->nj1;
  int nj2 = gdcurv->nj2;
  int nk1 = gdcurv->nk1;
  int nk2 = gdcurv->nk2;

  size_t siz_iy   = gdcurv->siz_iy;
  size_t siz_iz   = gdcurv->siz_iz;
  size_t siz_icmp = gdcurv->siz_icmp;

  // extend to ghosts, using mpi exchange
  // NOTE in different myid, nx(or ny) may not equal
  // so send type DTypeXL not equal recv type DTypeXL
  size_t s_iptr;
  size_t r_iptr;

  MPI_Status status;
  MPI_Datatype DTypeXL, DTypeYL, DTypeZL;

  MPI_Type_vector(ny*nz,
                  3,
                  nx,
                  MPI_FLOAT,
                  &DTypeXL);
  MPI_Type_vector(nz,
                  3*nx,
                  nx*ny,
                  MPI_FLOAT,
                  &DTypeYL);
  MPI_Type_vector(3,
                  nx*ny,
                  nx*ny,
                  MPI_FLOAT,
                  &DTypeZL);
  MPI_Type_commit(&DTypeXL);
  MPI_Type_commit(&DTypeYL);
  MPI_Type_commit(&DTypeZL);

  for(int i=0; i<ncmp; i++)
  {
    //bdry x1 to bdry x2
    s_iptr = ni1 + i * siz_icmp;        //sendbuff point (ni1,ny1,nz1)
    r_iptr = (ni2+1) + i * siz_icmp;    //recvbuff point (ni2+1,ny1,nz1)
    MPI_Sendrecv(&g3d[s_iptr],1,DTypeXL,neighid[0],110,
                 &g3d[r_iptr],1,DTypeXL,neighid[1],110,
                 topocomm,&status);
    //bdry x2 to bdry x1
    s_iptr = (ni2-3+1) + i * siz_icmp;    //sendbuff point (ni2-3+1,ny1,nz1)
    r_iptr = (ni1-3) + i * siz_icmp;      //recvbuff point (ni1-3,ny1,nz1)
    MPI_Sendrecv(&g3d[s_iptr],1,DTypeXL,neighid[1],120,
                 &g3d[r_iptr],1,DTypeXL,neighid[0],120,
                 topocomm,&status);
    //bdry y1 to bdry y2
    s_iptr = nj1 * siz_iy + i * siz_icmp;        //sendbuff point (nx1,nj1,nz1)
    r_iptr = (nj2+1) * siz_iy + i * siz_icmp;    //recvbuff point (nx1,nj2+1,nz1)
    MPI_Sendrecv(&g3d[s_iptr],1,DTypeYL,neighid[2],210,
                 &g3d[r_iptr],1,DTypeYL,neighid[3],210,
                 topocomm,&status);
    //bdry y2 to bdry y1
    s_iptr = (nj2-3+1) * siz_iy + i * siz_icmp;   //sendbuff point (nx1,nj2-3+1,nz1)
    r_iptr = (nj1-3) * siz_iy + i * siz_icmp;     //recvbuff point (nx1,nj1-3,nz1)
    MPI_Sendrecv(&g3d[s_iptr],1,DTypeYL,neighid[3],220,
                 &g3d[r_iptr],1,DTypeYL,neighid[2],220,
                 topocomm,&status);
    //bdry z1 to bdry z2
    s_iptr = nk1 * siz_iz + i * siz_icmp;        //sendbuff point (nx1,ny1,nk1)
    r_iptr = (nk2+1) * siz_iz + i * siz_icmp;    //recvbuff point (nx1,ny1,nk2+1)
    MPI_Sendrecv(&g3d[s_iptr],1,DTypeZL,neighid[4],310,
                 &g3d[r_iptr],1,DTypeZL,neighid[5],310,
                 topocomm,&status);
    //bdry z2 to bdry z1
    s_iptr = (nk2-3+1) * siz_iz + i * siz_icmp;   //sendbuff point (nx1,ny1,nk2-3+1)
    r_iptr = (nk1-3) * siz_iz + i * siz_icmp;     //recvbuff point (nx1,ny1,nk1-3)
    MPI_Sendrecv(&g3d[s_iptr],1,DTypeZL,neighid[5],320,
                 &g3d[r_iptr],1,DTypeZL,neighid[4],320,
                 topocomm,&status);
  }

  return 0;
}

int
gd_curv_gen_fault(gdcurv_t *gdcurv,
                  int fault_i_global_indx,
                  float dh,
                  char *in_grid_fault_nc)
{
  int nx = gdcurv->nx;
  int ny = gdcurv->ny;
  int nz = gdcurv->nz;

  int ni = gdcurv->ni;
  int nj = gdcurv->nj;
  int nk = gdcurv->nk;

  int ni1 = gdcurv->ni1;
  int ni2 = gdcurv->ni2;
  int nj1 = gdcurv->nj1;
  int nj2 = gdcurv->nj2;
  int nk1 = gdcurv->nk1;
  int nk2 = gdcurv->nk2;
  int gni1 = gdcurv->gni1;
  int npoint_x = gdcurv->npoint_x;

  size_t siz_iy  = gdcurv->siz_iy;
  size_t siz_iz  = gdcurv->siz_iz;
  size_t iptr, iptr_b, iptr_c;

  float *x3d = gdcurv->x3d;
  float *y3d = gdcurv->y3d;
  float *z3d = gdcurv->z3d;

  float *fault_x = (float *) malloc(sizeof(float)*nj*nk);
  float *fault_y = (float *) malloc(sizeof(float)*nj*nk);
  float *fault_z = (float *) malloc(sizeof(float)*nj*nk);
  float *xline   = (float *) malloc(sizeof(float)*nx);

  nc_read_fault_geometry(fault_x, fault_y, fault_z, in_grid_fault_nc, gdcurv);

  int i0 = fault_i_global_indx + 3;  // now with ghost index 
  xline[i0] = 0.0;
  int width1 = 10;
  int width2 = 55;
  float compr;
  int dist;
  for(int i = i0-1; i>=ni1; i--)
  {
    dist = abs(i-i0); 
    if(dist < width1)
    {
      compr = 0;
    }
    if(dist>=width1 && dist < width2)
    {
      compr = 1.0 - cos(PI * (i - (i0 - width1))/(float)(width2-width1));
    }
    if(dist >= width2)
    {
      compr = 2.0;
    }
    compr = 0.5 + 0.25 * compr;
    xline[i] = xline[i+1] - dh *compr;
  }
  for(int i = i0+1; i<=ni2; i++)
  {
    dist = abs(i-i0); 
    if(dist < width1)
    {
      compr = 0;
    }
    if(dist>=width1 && dist < width2)
    {
      compr = 1.0 - cos(PI * (i - (i0 + width1))/(float)(width2-width1));
    }
    if(dist >= width2)
    {
      compr = 2.0;
    }
    compr = 0.5 + 0.25 * compr;
    xline[i] = xline[i-1] + dh *compr;
  }

  for (int k = nk1; k <= nk2; k++){
    for (int j = nj1; j <= nj2; j++){
      for (int i = ni1; i <= ni2; i++){

        //int gi = gni1 + i - 3; 
        //float x = fault_x[j-3 + (k-3) * nj] + gi * dh + x0;

        float x = fault_x[j-3 + (k-3) * nj] + xline[i];
        float y = fault_y[j-3 + (k-3) * nj];
        float z = fault_z[j-3 + (k-3) * nj];

        iptr = i + j * siz_iy + k * siz_iz;
        x3d[iptr] = x;
        y3d[iptr] = y;
        z3d[iptr] = z;
      }
    }
  }

  // extend to ghosts. 
  geometric_symmetry(gdcurv,gdcurv->v4d,gdcurv->ncmp);
   
  free(fault_x);
  free(fault_y);
  free(fault_z);
  free(xline);

  return 0;
}

int
nc_read_fault_geometry(float *fault_x, float *fault_y, float *fault_z,
                       char *in_grid_fault_nc, gdcurv_t *gdcurv)
{
  int nj = gdcurv->nj;
  int nk = gdcurv->nk;

  // thisid dimension 0, 1, 2, thisid[2] vary first
  int ierr;
  int ncid;
  int xid, yid, zid;
  size_t start[] = {gdcurv->gnk1, gdcurv->gnj1};
  size_t count[] = {nk, nj};
  
  ierr = nc_open(in_grid_fault_nc, NC_NOWRITE, &ncid); handle_nc_err(ierr);

  //read coords
  ierr = nc_inq_varid(ncid, "x", &xid); handle_nc_err(ierr);
  ierr = nc_inq_varid(ncid, "y", &yid); handle_nc_err(ierr);
  ierr = nc_inq_varid(ncid, "z", &zid); handle_nc_err(ierr);

  ierr = nc_get_vara_float(ncid, xid, start, count, fault_x); handle_nc_err(ierr);
  ierr = nc_get_vara_float(ncid, yid, start, count, fault_y); handle_nc_err(ierr);
  ierr = nc_get_vara_float(ncid, zid, start, count, fault_z); handle_nc_err(ierr);

  ierr = nc_close(ncid); handle_nc_err(ierr);

  return 0;
}

int
mirror_symmetry(gdcurv_t *gdcurv, float *v4d, int ncmp)
{
  int ni1 = gdcurv->ni1;
  int ni2 = gdcurv->ni2;
  int nj1 = gdcurv->nj1;
  int nj2 = gdcurv->nj2;
  int nk1 = gdcurv->nk1;
  int nk2 = gdcurv->nk2;
  int nx  = gdcurv->nx;
  int ny  = gdcurv->ny;
  int nz  = gdcurv->nz;
  size_t siz_iy  = gdcurv->siz_iy;
  size_t siz_iz  = gdcurv->siz_iz;
  size_t siz_icmp  = gdcurv->siz_icmp;

  size_t iptr, iptr1, iptr2; 
  for(int icmp=0; icmp<ncmp; icmp++){
    iptr = icmp * siz_icmp;
    // x1, mirror
    for (size_t k = 0; k < nz; k++){
      for (size_t j = 0; j < ny; j++){
        for (size_t i = 0; i < ni1; i++)
        {
          iptr1 = iptr + i + j * siz_iy + k * siz_iz;
          iptr2 = iptr + (2*ni1-i) + j * siz_iy +  k * siz_iz;
          v4d[iptr1] = v4d[iptr2];
        }
      }
    }
    // x2, mirror
    for (size_t k = 0; k < nz; k++){
      for (size_t j = 0; j < ny; j++){
        for (size_t i = ni2+1; i < nx; i++)
        {
          iptr1 = iptr + i + j * siz_iy + k * siz_iz;
          iptr2 = iptr + (2*ni2-i) + j * siz_iy + k * siz_iz;
          v4d[iptr1] = v4d[iptr2];
        }
      }
    }
    // y1, mirror
    for (size_t k = 0; k < nz; k++){
      for (size_t j = 0; j < nj1; j++){
        for (size_t i = 0; i < nx; i++)
        {
          iptr1 = iptr + i + j * siz_iy + k * siz_iz;
          iptr2 = iptr + i + (2*nj1-j) * siz_iy +  k * siz_iz;
          v4d[iptr1] = v4d[iptr2];
        }
      }
    }
    // y2, mirror
    for (size_t k = 0; k < nz; k++){
      for (size_t j = nj2+1; j < ny; j++){
        for (size_t i = 0; i < nx; i++)
        {
          iptr1 = iptr + i + j * siz_iy + k * siz_iz;
          iptr2 = iptr + i + (2*nj2-j) * siz_iy +  k * siz_iz;
          v4d[iptr1] = v4d[iptr2];
        }
      }
    }
    // z1, mirror
    for (size_t k = 0; k < nk1; k++) {
      for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
          iptr1 = iptr + i + j * siz_iy + k * siz_iz;
          iptr2 = iptr + i + j * siz_iy + (2*nk1-k) * siz_iz;
          v4d[iptr1] = v4d[iptr2];
        }
      }
    }
    // z2, mirror
    for (size_t k = nk2+1; k < nz; k++) {
      for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
          iptr1 = iptr + i + j * siz_iy + k * siz_iz;
          iptr2 = iptr + i + j * siz_iy + (2*nk2-k) * siz_iz;
          v4d[iptr1] = v4d[iptr2];
        }
      }
    }
  }

  return 0;
}

int 
geometric_symmetry(gdcurv_t *gdcurv,float *v4d, int ncmp)
{
  int ni1 = gdcurv->ni1;
  int ni2 = gdcurv->ni2;
  int nj1 = gdcurv->nj1;
  int nj2 = gdcurv->nj2;
  int nk1 = gdcurv->nk1;
  int nk2 = gdcurv->nk2;
  int nx  = gdcurv->nx;
  int ny  = gdcurv->ny;
  int nz  = gdcurv->nz;
  size_t siz_iy  = gdcurv->siz_iy;
  size_t siz_iz  = gdcurv->siz_iz;
  size_t siz_icmp  = gdcurv->siz_icmp;

  size_t iptr, iptr1, iptr2, iptr3; 
  for(int icmp=0; icmp<ncmp; icmp++){
    iptr = icmp * siz_icmp;
    // x1 
    for (size_t k = 0; k < nz; k++){
      for (size_t j = 0; j < ny; j++){
        for (size_t i = 0; i < ni1; i++)
        {
          iptr1 = iptr + i + j * siz_iy + k * siz_iz;
          iptr2 = iptr + ni1 + j * siz_iy + k * siz_iz;
          iptr3 = iptr + (2*ni1-i) + j * siz_iy + k * siz_iz;
          v4d[iptr1] = 2*v4d[iptr2] - v4d[iptr3];
        }
      }
    }
    // x2
    for (size_t k = 0; k < nz; k++){
      for (size_t j = 0; j < ny; j++){
        for (size_t i = ni2+1; i < nx; i++)
        {
          iptr1 = iptr + i + j * siz_iy + k * siz_iz;
          iptr2 = iptr + ni2 + j * siz_iy + k * siz_iz;
          iptr3 = iptr + (2*ni2-i) + j * siz_iy + k * siz_iz;
          v4d[iptr1] = 2*v4d[iptr2] - v4d[iptr3];
        }
      }
    }
    // y1 
    for (size_t k = 0; k < nz; k++){
      for (size_t j = 0; j < nj1; j++){
        for (size_t i = 0; i < nx; i++)
        {
          iptr1 = iptr + i + j * siz_iy + k * siz_iz;
          iptr2 = iptr + i + nj1 * siz_iy + k * siz_iz;
          iptr3 = iptr + i + (2*nj1-j) * siz_iy + k * siz_iz;
          v4d[iptr1] = 2*v4d[iptr2] - v4d[iptr3];
        }
      }
    }
    // y2 
    for (size_t k = 0; k < nz; k++){
      for (size_t j = nj2+1; j < ny; j++){
        for (size_t i = 0; i < nx; i++)
        {
          iptr1 = iptr + i + j * siz_iy + k * siz_iz;
          iptr2 = iptr + i + nj2 * siz_iy + k * siz_iz;
          iptr3 = iptr + i + (2*nj2-j) * siz_iy + k * siz_iz;
          v4d[iptr1] = 2*v4d[iptr2] - v4d[iptr3];
        }
      }
    }
    // z1
    for (size_t k = 0; k < nk1; k++){
      for (size_t j = 0; j < ny; j++){
        for (size_t i = 0; i < nx; i++)
        {
          iptr1 = iptr + i + j * siz_iy + k * siz_iz;
          iptr2 = iptr + i + j * siz_iy + nk1 * siz_iz;
          iptr3 = iptr + i + j * siz_iy + (2*nk1-k) * siz_iz;
          v4d[iptr1] = 2*v4d[iptr2] - v4d[iptr3];
        }
      }
    }
    // z2
    for (size_t k = nk2+1; k < nz; k++) {
      for (size_t j = 0; j < ny; j++){
        for (size_t i = 0; i < nx; i++)
        {
          iptr1 = iptr + i + j * siz_iy + k * siz_iz;
          iptr2 = iptr + i + j * siz_iy + nk2 * siz_iz;
          iptr3 = iptr + i + j * siz_iy + (2*nk2-k) * siz_iz;
          v4d[iptr1] = 2*v4d[iptr2] - v4d[iptr3];
        }
      }
    }
  }

  return 0;
}

//
// input/output
//
int
gd_curv_coord_export(gdcurv_t *gdcurv,
                     char *fname_coords,
                     char *output_dir)
{
  size_t *c3d_pos   = gdcurv->cmp_pos;
  char  **c3d_name  = gdcurv->cmp_name;
  int number_of_vars = gdcurv->ncmp;
  int nx = gdcurv->nx;
  int ny = gdcurv->ny;
  int nz = gdcurv->nz;
  int ni1 = gdcurv->ni1;
  int nj1 = gdcurv->nj1;
  int nk1 = gdcurv->nk1;
  int ni  = gdcurv->ni;
  int nj  = gdcurv->nj;
  int nk  = gdcurv->nk;
  int gni1 = gdcurv->ni1_to_glob_phys0;
  int gnj1 = gdcurv->nj1_to_glob_phys0;
  int gnk1 = gdcurv->nk1_to_glob_phys0;

  // construct file name
  char ou_file[CONST_MAX_STRLEN];
  sprintf(ou_file, "%s/coord_%s.nc", output_dir, fname_coords);
  
  // read in nc
  int ncid;
  int varid[gdcurv->ncmp];
  int dimid[CONST_NDIM];

  int ierr = nc_create(ou_file, NC_CLOBBER | NC_64BIT_OFFSET, &ncid); handle_nc_err(ierr);

  // define dimension
  ierr = nc_def_dim(ncid, "i", nx, &dimid[2]);
  ierr = nc_def_dim(ncid, "j", ny, &dimid[1]);
  ierr = nc_def_dim(ncid, "k", nz, &dimid[0]);

  // define vars
  for (int ivar=0; ivar<gdcurv->ncmp; ivar++) {
    ierr = nc_def_var(ncid, gdcurv->cmp_name[ivar], NC_FLOAT, CONST_NDIM, dimid, &varid[ivar]);
    handle_nc_err(ierr);
  }

  // attribute: index in output snapshot, index w ghost in thread
  int l_start[] = { ni1, nj1, nk1 };
  nc_put_att_int(ncid,NC_GLOBAL,"local_index_of_first_physical_points",
                   NC_INT,CONST_NDIM,l_start);

  int g_start[] = { gni1, gnj1, gnk1 };
  nc_put_att_int(ncid,NC_GLOBAL,"global_index_of_first_physical_points",
                   NC_INT,CONST_NDIM,g_start);

  int l_count[] = { ni, nj, nk };
  nc_put_att_int(ncid,NC_GLOBAL,"count_of_physical_points",
                   NC_INT,CONST_NDIM,l_count);

  // end def
  ierr = nc_enddef(ncid);  handle_nc_err(ierr);

  // add vars
  for (int ivar=0; ivar<gdcurv->ncmp; ivar++) {
    float *ptr = gdcurv->v4d + gdcurv->cmp_pos[ivar];
    ierr = nc_put_var_float(ncid, varid[ivar],ptr);
    handle_nc_err(ierr);
  }
  
  // close file
  ierr = nc_close(ncid); handle_nc_err(ierr);

  return 0;
}

int
gd_curv_coord_import(gdcurv_t *gdcurv, char *fname_coords, char *import_dir)
{
  // construct file name
  char in_file[CONST_MAX_STRLEN];
  sprintf(in_file, "%s/coord_%s.nc", import_dir, fname_coords);
  
  // read in nc
  int ncid;
  int varid;

  int ierr = nc_open(in_file, NC_NOWRITE, &ncid); handle_nc_err(ierr);

  // read vars
  for (int ivar=0; ivar<gdcurv->ncmp; ivar++)
  {
    float *ptr = gdcurv->v4d + gdcurv->cmp_pos[ivar];
    ierr = nc_inq_varid(ncid, gdcurv->cmp_name[ivar], &varid); 
    handle_nc_err(ierr);
    ierr = nc_get_var(ncid, varid, ptr); handle_nc_err(ierr);
  }
  
  // close file
  ierr = nc_close(ncid); handle_nc_err(ierr);

  return 0;
}


int
gd_curv_metric_export(gdcurv_t        *gdcurv,
                      gd_metric_t *metric,
                      char *fname_coords,
                      char *output_dir)
{
  size_t *g3d_pos   = metric->cmp_pos;
  char  **g3d_name  = metric->cmp_name;
  int  number_of_vars = metric->ncmp;
  int  nx = metric->nx;
  int  ny = metric->ny;
  int  nz = metric->nz;
  int  ni1 = gdcurv->ni1;
  int  nj1 = gdcurv->nj1;
  int  nk1 = gdcurv->nk1;
  int  ni  = gdcurv->ni;
  int  nj  = gdcurv->nj;
  int  nk  = gdcurv->nk;
  int  gni1 = gdcurv->ni1_to_glob_phys0;
  int  gnj1 = gdcurv->nj1_to_glob_phys0;
  int  gnk1 = gdcurv->nk1_to_glob_phys0;

  // construct file name
  char ou_file[CONST_MAX_STRLEN];
  sprintf(ou_file, "%s/metric_%s.nc", output_dir, fname_coords);
  
  // read in nc
  int ncid;
  int varid[number_of_vars];
  int dimid[CONST_NDIM];

  int ierr = nc_create(ou_file, NC_CLOBBER | NC_64BIT_OFFSET, &ncid); handle_nc_err(ierr);

  // define dimension
  ierr = nc_def_dim(ncid, "i", nx, &dimid[2]);
  ierr = nc_def_dim(ncid, "j", ny, &dimid[1]);
  ierr = nc_def_dim(ncid, "k", nz, &dimid[0]);

  // define vars
  for (int ivar=0; ivar<number_of_vars; ivar++) {
    ierr = nc_def_var(ncid, g3d_name[ivar], NC_FLOAT, CONST_NDIM, dimid, &varid[ivar]);
    handle_nc_err(ierr);
  }

  // attribute: index in output snapshot, index w ghost in thread
  int l_start[] = { ni1, nj1, nk1 };
  nc_put_att_int(ncid,NC_GLOBAL,"local_index_of_first_physical_points",
                   NC_INT,CONST_NDIM,l_start);

  int g_start[] = { gni1, gnj1, gnk1 };
  nc_put_att_int(ncid,NC_GLOBAL,"global_index_of_first_physical_points",
                   NC_INT,CONST_NDIM,g_start);

  int l_count[] = { ni, nj, nk };
  nc_put_att_int(ncid,NC_GLOBAL,"count_of_physical_points",
                   NC_INT,CONST_NDIM,l_count);

  // end def
  ierr = nc_enddef(ncid);  handle_nc_err(ierr);

  // add vars
  for (int ivar=0; ivar<number_of_vars; ivar++) {
    float *ptr = metric->v4d + g3d_pos[ivar];
    ierr = nc_put_var_float(ncid, varid[ivar],ptr);
    handle_nc_err(ierr);
  }
  
  // close file
  ierr = nc_close(ncid); handle_nc_err(ierr);

  return 0;
}

int
gd_curv_metric_import(gd_metric_t *metric, char *fname_coords, char *import_dir)
{
  // construct file name
  char in_file[CONST_MAX_STRLEN];
  sprintf(in_file, "%s/metric_%s.nc", import_dir, fname_coords);
  
  // read in nc
  int ncid;
  int varid;

  int ierr = nc_open(in_file, NC_NOWRITE, &ncid); handle_nc_err(ierr);

  // read vars
  for (int ivar=0; ivar<metric->ncmp; ivar++)
  {
    float *ptr = metric->v4d + metric->cmp_pos[ivar];

    ierr = nc_inq_varid(ncid, metric->cmp_name[ivar], &varid); handle_nc_err(ierr);

    ierr = nc_get_var(ncid, varid, ptr); handle_nc_err(ierr);
  }
  
  // close file
  ierr = nc_close(ncid); handle_nc_err(ierr);

  return 0;
}

/*
 * set min/max of grid for loc
 */
int
gd_curv_set_minmax(gdcurv_t *gdcurv)
{
  // all points including ghosts
  float xmin = gdcurv->x3d[0], xmax = gdcurv->x3d[0];
  float ymin = gdcurv->y3d[0], ymax = gdcurv->y3d[0];
  float zmin = gdcurv->z3d[0], zmax = gdcurv->z3d[0];
  for (size_t i = 0; i < gdcurv->siz_icmp; i++){
      xmin = xmin < gdcurv->x3d[i] ? xmin : gdcurv->x3d[i];
      xmax = xmax > gdcurv->x3d[i] ? xmax : gdcurv->x3d[i];
      ymin = ymin < gdcurv->y3d[i] ? ymin : gdcurv->y3d[i];
      ymax = ymax > gdcurv->y3d[i] ? ymax : gdcurv->y3d[i];
      zmin = zmin < gdcurv->z3d[i] ? zmin : gdcurv->z3d[i];
      zmax = zmax > gdcurv->z3d[i] ? zmax : gdcurv->z3d[i];
  }
  gdcurv->xmin = xmin;
  gdcurv->xmax = xmax;
  gdcurv->ymin = ymin;
  gdcurv->ymax = ymax;
  gdcurv->zmin = zmin;
  gdcurv->zmax = zmax;

  // all physics points without ghosts
  xmin = gdcurv->xmax;
  xmax = gdcurv->xmin;
  ymin = gdcurv->ymax;
  ymax = gdcurv->ymin;
  zmin = gdcurv->zmax;
  zmax = gdcurv->zmin;
  for (int k = gdcurv->nk1; k <= gdcurv->nk2; k++) {
    for (int j = gdcurv->nj1; j <= gdcurv->nj2; j++) {
      for (int i = gdcurv->ni1; i <= gdcurv->ni2; i++) {
         size_t iptr = i + j * gdcurv->siz_iy + k * gdcurv->siz_iz;
         xmin = xmin < gdcurv->x3d[iptr] ? xmin : gdcurv->x3d[iptr];
         xmax = xmax > gdcurv->x3d[iptr] ? xmax : gdcurv->x3d[iptr];
         ymin = ymin < gdcurv->y3d[iptr] ? ymin : gdcurv->y3d[iptr];
         ymax = ymax > gdcurv->y3d[iptr] ? ymax : gdcurv->y3d[iptr];
         zmin = zmin < gdcurv->z3d[iptr] ? zmin : gdcurv->z3d[iptr];
         zmax = zmax > gdcurv->z3d[iptr] ? zmax : gdcurv->z3d[iptr];
      }
    }
  }
  gdcurv->xmin_phy = xmin;
  gdcurv->xmax_phy = xmax;
  gdcurv->ymin_phy = ymin;
  gdcurv->ymax_phy = ymax;
  gdcurv->zmin_phy = zmin;
  gdcurv->zmax_phy = zmax;

  // set cell range, last cell along each dim unusage
  for (int k = 0; k < gdcurv->nz-1; k++) {
    for (int j = 0; j < gdcurv->ny-1; j++) {
      for (int i = 0; i < gdcurv->nx-1; i++) {
         size_t iptr = i + j * gdcurv->siz_iy + k * gdcurv->siz_iz;
         xmin = gdcurv->x3d[iptr];
         ymin = gdcurv->y3d[iptr];
         zmin = gdcurv->z3d[iptr];
         xmax = xmin;
         ymax = ymin;
         zmax = zmin;
         for (int n3=0; n3<2; n3++) {
         for (int n2=0; n2<2; n2++) {
         for (int n1=0; n1<2; n1++) {
           size_t iptr_pt = iptr + n3 * gdcurv->siz_iz + n2 * gdcurv->siz_iy + n1;
           xmin = xmin < gdcurv->x3d[iptr_pt] ? xmin : gdcurv->x3d[iptr_pt];
           xmax = xmax > gdcurv->x3d[iptr_pt] ? xmax : gdcurv->x3d[iptr_pt];
           ymin = ymin < gdcurv->y3d[iptr_pt] ? ymin : gdcurv->y3d[iptr_pt];
           ymax = ymax > gdcurv->y3d[iptr_pt] ? ymax : gdcurv->y3d[iptr_pt];
           zmin = zmin < gdcurv->z3d[iptr_pt] ? zmin : gdcurv->z3d[iptr_pt];
           zmax = zmax > gdcurv->z3d[iptr_pt] ? zmax : gdcurv->z3d[iptr_pt];
         }
         }
         }
         gdcurv->cell_xmin[iptr] = xmin;
         gdcurv->cell_xmax[iptr] = xmax;
         gdcurv->cell_ymin[iptr] = ymin;
         gdcurv->cell_ymax[iptr] = ymax;
         gdcurv->cell_zmin[iptr] = zmin;
         gdcurv->cell_zmax[iptr] = zmax;
      }
    }
  }

  // set tile range 

  // partition into average plus left at last
  int nx_avg  = gdcurv->ni / GD_TILE_NX; // only for physcial points
  int nx_left = gdcurv->ni % GD_TILE_NX;
  int ny_avg  = gdcurv->nj / GD_TILE_NY;
  int ny_left = gdcurv->nj % GD_TILE_NY;
  int nz_avg  = gdcurv->nk / GD_TILE_NZ;
  int nz_left = gdcurv->nk % GD_TILE_NZ;
  for (int k_tile = 0; k_tile < GD_TILE_NZ; k_tile++)
  {
    if (k_tile == 0) {
      gdcurv->tile_kstart[k_tile] = gdcurv->nk1;
    } else {
      gdcurv->tile_kstart[k_tile] = gdcurv->tile_kend[k_tile-1] + 1;
    }

    gdcurv->tile_kend  [k_tile] = gdcurv->tile_kstart[k_tile] + nz_avg -1;
    if (k_tile < nz_left) {
      gdcurv->tile_kend[k_tile] += 1;
    }

    for (int j_tile = 0; j_tile < GD_TILE_NY; j_tile++)
    {
      if (j_tile == 0) {
        gdcurv->tile_jstart[j_tile] = gdcurv->nj1;
      } else {
        gdcurv->tile_jstart[j_tile] = gdcurv->tile_jend[j_tile-1] + 1;
      }

      gdcurv->tile_jend  [j_tile] = gdcurv->tile_jstart[j_tile] + ny_avg -1;
      if (j_tile < ny_left) {
        gdcurv->tile_jend[j_tile] += 1;
      }

      for (int i_tile = 0; i_tile < GD_TILE_NX; i_tile++)
      {
        if (i_tile == 0) {
          gdcurv->tile_istart[i_tile] = gdcurv->ni1;
        } else {
          gdcurv->tile_istart[i_tile] = gdcurv->tile_iend[i_tile-1] + 1;
        }

        gdcurv->tile_iend  [i_tile] = gdcurv->tile_istart[i_tile] + nx_avg -1;
        if (i_tile < nx_left) {
          gdcurv->tile_iend[i_tile] += 1;
        }

        // use large value to init
        xmin = 1.0e26;
        ymin = 1.0e26;
        zmin = 1.0e26;
        xmax = -1.0e26;
        ymax = -1.0e26;
        zmax = -1.0e26;
        // for cells in each tile
        for (int k = gdcurv->tile_kstart[k_tile]; k <= gdcurv->tile_kend[k_tile]; k++)
        {
          size_t iptr_k = k * gdcurv->siz_iz;
          for (int j = gdcurv->tile_jstart[j_tile]; j <= gdcurv->tile_jend[j_tile]; j++)
          {
            size_t iptr_j = iptr_k + j * gdcurv->siz_iy;
            for (int i = gdcurv->tile_istart[i_tile]; i <= gdcurv->tile_iend[i_tile]; i++)
            {
              size_t iptr = i + iptr_j;
              xmin = xmin < gdcurv->cell_xmin[iptr] ? xmin : gdcurv->cell_xmin[iptr];
              xmax = xmax > gdcurv->cell_xmax[iptr] ? xmax : gdcurv->cell_xmax[iptr];
              ymin = ymin < gdcurv->cell_ymin[iptr] ? ymin : gdcurv->cell_ymin[iptr];
              ymax = ymax > gdcurv->cell_ymax[iptr] ? ymax : gdcurv->cell_ymax[iptr];
              zmin = zmin < gdcurv->cell_zmin[iptr] ? zmin : gdcurv->cell_zmin[iptr];
              zmax = zmax > gdcurv->cell_zmax[iptr] ? zmax : gdcurv->cell_zmax[iptr];
            }
          }
        }
        int iptr_tile = i_tile + j_tile * GD_TILE_NX + k_tile * GD_TILE_NX *GD_TILE_NY;
        gdcurv->tile_xmin[iptr_tile] = xmin;
        gdcurv->tile_xmax[iptr_tile] = xmax;
        gdcurv->tile_ymin[iptr_tile] = ymin;
        gdcurv->tile_ymax[iptr_tile] = ymax;
        gdcurv->tile_zmin[iptr_tile] = zmin;
        gdcurv->tile_zmax[iptr_tile] = zmax;

      }
    }
  } // k_tile

  return 0;
}

/*
 * convert curv coord to global index using MPI
 */

int
gd_curv_coord_to_glob_indx(gdcurv_t *gdcurv,
                           float sx,
                           float sy,
                           float sz,
                           MPI_Comm comm,
                           int myid,
                           int   *ou_si, int *ou_sj, int *ou_sk,
                           float *ou_sx_inc, float *ou_sy_inc, float *ou_sz_inc)
{
  int is_here = 0;
  
  //NOTE si_glob sj_glob sk_glob must less -3. due to ghost points length is 3.
  int si_glob = -1000;
  int sj_glob = -1000;
  int sk_glob = -1000;
  float sx_inc = 0.0;
  float sy_inc = 0.0;
  float sz_inc = 0.0;
  int si = 0;
  int sj = 0;
  int sk = 0;

  // if located in this thread
  is_here = gd_curv_coord_to_local_indx(gdcurv,sx,sy,sz,
                                    &si, &sj, &sk, &sx_inc, &sy_inc, &sz_inc);

  // if in this thread
  if ( is_here == 1)
  {
    // conver to global index
    si_glob = gd_info_ind_lcext2glphy_i(si, gdcurv);
    sj_glob = gd_info_ind_lcext2glphy_j(sj, gdcurv);
    sk_glob = gd_info_ind_lcext2glphy_k(sk, gdcurv);
  }

  // reduce global index and shift values
  int sendbufi = si_glob;
  MPI_Allreduce(&sendbufi, &si_glob, 1, MPI_INT, MPI_MAX, comm);

  sendbufi = sj_glob;
  MPI_Allreduce(&sendbufi, &sj_glob, 1, MPI_INT, MPI_MAX, comm);

  sendbufi = sk_glob;
  MPI_Allreduce(&sendbufi, &sk_glob, 1, MPI_INT, MPI_MAX, comm);

  float sendbuf = sx_inc;
  MPI_Allreduce(&sendbuf, &sx_inc, 1, MPI_FLOAT, MPI_SUM, comm);

  sendbuf = sy_inc;
  MPI_Allreduce(&sendbuf, &sy_inc, 1, MPI_FLOAT, MPI_SUM, comm);

  sendbuf = sz_inc;
  MPI_Allreduce(&sendbuf, &sz_inc, 1, MPI_FLOAT, MPI_SUM, comm);

  *ou_si = si_glob;
  *ou_sj = sj_glob;
  *ou_sk = sk_glob;
  *ou_sx_inc = sx_inc;
  *ou_sy_inc = sy_inc;
  *ou_sz_inc = sz_inc;

  return is_here; 
}

__device__ int
gd_curv_coord_to_glob_indx_gpu(gdcurv_t *gdcurv,
                               float sx,
                               float sy,
                               float sz,
                               MPI_Comm comm,
                               int myid,
                               int   *ou_si, int *ou_sj, int *ou_sk,
                               float *ou_sx_inc, float *ou_sy_inc, float *ou_sz_inc)
{
  int is_here = 0;

  //NOTE si_glob sj_glob sk_glob must less -3. due to ghost points length is 3.
  int si_glob = -1000;
  int sj_glob = -1000;
  int sk_glob = -1000;
  float sx_inc = 0.0;
  float sy_inc = 0.0;
  float sz_inc = 0.0;
  int si = 0;
  int sj = 0;
  int sk = 0;
  // if located in this thread
  is_here = gd_curv_coord_to_local_indx(gdcurv,sx,sy,sz,
                                    &si, &sj, &sk, &sx_inc, &sy_inc, &sz_inc);

  // if in this thread
  if ( is_here == 1)
  {
    // conver to global index
    si_glob = gd_info_ind_lcext2glphy_i(si, gdcurv);
    sj_glob = gd_info_ind_lcext2glphy_j(sj, gdcurv);
    sk_glob = gd_info_ind_lcext2glphy_k(sk, gdcurv);
  }

  *ou_si = si_glob;
  *ou_sj = sj_glob;
  *ou_sk = sk_glob;
  *ou_sx_inc = sx_inc;
  *ou_sy_inc = sy_inc;
  *ou_sz_inc = sz_inc;

  return is_here; 
}

/* 
 * if the nearest point in this thread then search its grid index
 *   return value:
 *      1 - in this thread
 *      0 - not in this thread
 */

__host__ __device__ int
gd_curv_coord_to_local_indx(gdcurv_t *gdcurv,
                            float sx, float sy, float sz,
                            int *si, int *sj, int *sk,
                            float *sx_inc, float *sy_inc, float *sz_inc)
{
  int is_here = 0; // default outside

  // not here if outside coord range
  if ( sx < gdcurv->xmin || sx > gdcurv->xmax ||
       sy < gdcurv->ymin || sy > gdcurv->ymax ||
       sz < gdcurv->zmin || sz > gdcurv->zmax)
  {
    return is_here;
  }

  int nx = gdcurv->nx;
  int ny = gdcurv->ny;
  int nz = gdcurv->nz;
  int ni1 = gdcurv->ni1;
  int ni2 = gdcurv->ni2;
  int nj1 = gdcurv->nj1;
  int nj2 = gdcurv->nj2;
  int nk1 = gdcurv->nk1;
  int nk2 = gdcurv->nk2;
  size_t siz_iy = gdcurv->siz_iy;
  size_t siz_iz = gdcurv->siz_iz;
  
  float *x3d = gdcurv->x3d;
  float *y3d = gdcurv->y3d;
  float *z3d = gdcurv->z3d;
  
  // init closest point
  float min_dist = sqrtf(  (sx - x3d[0]) * (sx - x3d[0])
      + (sy - y3d[0]) * (sy - y3d[0])
      + (sz - z3d[0]) * (sz - z3d[0]) );
  int min_dist_i = 0 ;
  int min_dist_j = 0 ;
  int min_dist_k = 0 ;

  // compute distance to each point
  for (int k=0; k<nz; k++) {
    for (int j=0; j<ny; j++) {
      for (int i=0; i<nx; i++)
      {
        size_t iptr = i + j * siz_iy + k * siz_iz;

        float x = x3d[iptr];
        float y = y3d[iptr];
        float z = z3d[iptr];

        float DistInt = sqrtf(  (sx - x) * (sx - x)
            + (sy - y) * (sy - y)
            + (sz - z) * (sz - z) );

        // replace closest point
        if (min_dist > DistInt)
        {
          min_dist = DistInt;
          min_dist_i = i;
          min_dist_j = j;
          min_dist_k = k;
        }
      }
    }
  }

  // if nearest index is outside phys region, not here
  if ( min_dist_i < ni1 || min_dist_i > ni2 ||
      min_dist_j < nj1 || min_dist_j > nj2 ||
      min_dist_k < nk1 || min_dist_k > nk2 )
  {
    is_here = 0;
    return is_here;
  }

  // in this thread
  is_here = 1;

  float points_x[8];
  float points_y[8];
  float points_z[8];
  float points_i[8];
  float points_j[8];
  float points_k[8];

  for (int kk=0; kk<2; kk++)
  {
    for (int jj=0; jj<2; jj++)
    {
      for (int ii=0; ii<2; ii++)
      {
        int cur_i = min_dist_i-1+ii;
        int cur_j = min_dist_j-1+jj;
        int cur_k = min_dist_k-1+kk;

        for (int n3=0; n3<2; n3++) {
          for (int n2=0; n2<2; n2++) {
            for (int n1=0; n1<2; n1++) {
              int iptr_cube = n1 + n2 * 2 + n3 * 4;
              int iptr = (cur_i+n1) + (cur_j+n2) * siz_iy +
                (cur_k+n3) * siz_iz;
              points_x[iptr_cube] = x3d[iptr];
              points_y[iptr_cube] = y3d[iptr];
              points_z[iptr_cube] = z3d[iptr];
              points_i[iptr_cube] = cur_i+n1;
              points_j[iptr_cube] = cur_j+n2;
              points_k[iptr_cube] = cur_k+n3;
            }
          }
        }
        if (isPointInHexahedron_c(sx,sy,sz,points_x,points_y,points_z) == 1)
        {
          float si_curv, sj_curv, sk_curv;

          gd_curv_coord2index_sample(sx,sy,sz,
              8,
              points_x,points_y,points_z,
              points_i,points_j,points_k,
              100,100,100,
              &si_curv, &sj_curv, &sk_curv);

          // convert to return values
          *si = min_dist_i;
          *sj = min_dist_j;
          *sk = min_dist_k;
          *sx_inc = si_curv - min_dist_i;
          *sy_inc = sj_curv - min_dist_j;
          *sz_inc = sk_curv - min_dist_k;

          return is_here;
        }
      }
    }
  }

  // if not in any cube due to bug, set default value
  //    if everything is right, should be return 10 line before
  *si = min_dist_i;
  *sj = min_dist_j;
  *sk = min_dist_k;
  *sx_inc = 0.0;
  *sy_inc = 0.0;
  *sz_inc = 0.0;

  return is_here;
}


/*
 * convert depth to axis
 */
__host__ __device__
int
gd_curv_depth_to_axis(gdcurv_t *gdcurv,
                      float sx,
                      float sy,
                      float *sz,
                      MPI_Comm comm,
                      int myid)
{
  int ierr = 0;

  // not here if outside coord range
  if ( sx < gdcurv->xmin || sx > gdcurv->xmax ||
       sy < gdcurv->ymin || sy > gdcurv->ymax )
  {
    return ierr;
  }

  float points_x[4];
  float points_y[4];
  float points_z[4];

  size_t iptr_k, iptr_j, iptr;

  // take upper-right cell, thus do not take last index
  int k_tile = GD_TILE_NZ - 1;
  {
    for (int j_tile = 0; j_tile < GD_TILE_NY; j_tile++)
    {
      for (int i_tile = 0; i_tile < GD_TILE_NX; i_tile++)
      {
        int iptr_tile = i_tile + j_tile * GD_TILE_NX + k_tile * GD_TILE_NX * GD_TILE_NY;
        if (  sx < gdcurv->tile_xmin[iptr_tile] ||
              sx > gdcurv->tile_xmax[iptr_tile] ||
              sy < gdcurv->tile_ymin[iptr_tile] ||
              sy > gdcurv->tile_ymax[iptr_tile])
        {
          // loop next tile
          continue;
        }

        // otherwise may in this tile
        int k = gdcurv->tile_kend[k_tile];
        {
          iptr_k = k * gdcurv->siz_iz;
          for (int j = gdcurv->tile_jstart[j_tile]; j <= gdcurv->tile_jend[j_tile]; j++)
          {
            iptr_j = iptr_k + j * gdcurv->siz_iy;
            for (int i = gdcurv->tile_istart[i_tile]; i <= gdcurv->tile_iend[i_tile]; i++)
            {
              iptr = i + iptr_j;

              // use AABB algorith
              if (  sx < gdcurv->cell_xmin[iptr] ||
                    sx > gdcurv->cell_xmax[iptr] ||
                    sy < gdcurv->cell_ymin[iptr] ||
                    sy > gdcurv->cell_ymax[iptr] )
              {
                // loop next cell
                continue;
              }

              // otherwise may in this cell, use inpolygon to check

              // set cell points
              for (int n2=0; n2<2; n2++) {
                for (int n1=0; n1<2; n1++) {
                  int iptr_cube = n1 + n2 * 2;
                  size_t iptr_pt = (i+n1) + (j+n2) * gdcurv->siz_iy + k * gdcurv->siz_iz;
                  points_x[iptr_cube] = gdcurv->x3d[iptr_pt];
                  points_y[iptr_cube] = gdcurv->y3d[iptr_pt];
                  points_z[iptr_cube] = gdcurv->z3d[iptr_pt];
                }
              }

              // interp z if in this cell
              if (fdlib_math_isPoint2InQuad(sx,sy,points_x,points_y) == 1)
              {
                float ztopo = fdlib_math_rdinterp_2d(sx,sy,4,points_x,points_y,points_z);
                 
                *sz = ztopo - (*sz);

                return ierr;
              }
              
            } // i
          } // j
        } // k

      } // i_tile
    } // j_tile
  } // k_tile

  return ierr;
}

/* 
 * find relative coord shift in this cell using sampling
 */

__host__ __device__ int
gd_curv_coord2shift_sample(float sx, float sy, float sz, 
    int num_points,
    float *points_x, // x coord of all points
    float *points_y,
    float *points_z,
    int    nx_sample,
    int    ny_sample,
    int    nz_sample,
    float *si_shift, // interped curv coord
    float *sj_shift,
    float *sk_shift)
{
  float Lx[2], Ly[2], Lz[2];

  // init closest point
  float min_dist = sqrtf(  (sx - points_x[0]) * (sx - points_x[0])
      + (sy - points_y[0]) * (sy - points_y[0])
      + (sz - points_z[0]) * (sz - points_z[0]) );
  int min_dist_i = 0 ;
  int min_dist_j = 0 ;
  int min_dist_k = 0 ;

  // linear interp for all sample
  for (int n3=0; n3<nz_sample+1; n3++)
  {
    Lz[1] = (float)(n3) / (float)(nz_sample);
    Lz[0] = 1.0 - Lz[1];
    for (int n2=0; n2<ny_sample+1; n2++)
    {
      Ly[1] = (float)(n2) / (float)(ny_sample);
      Ly[0] = 1.0 - Ly[1];
      for (int n1=0; n1<nx_sample+1; n1++)
      {
        Lx[1] = (float)(n1) / (float)(nx_sample);
        Lx[0] = 1.0 - Lx[1];

        // interp
        float x_pt=0;
        float y_pt=0;
        float z_pt=0;
        for (int kk=0; kk<2; kk++) {
          for (int jj=0; jj<2; jj++) {
            for (int ii=0; ii<2; ii++)
            {
              int iptr_cube = ii + jj * 2 + kk * 4;
              x_pt += Lx[ii]*Ly[jj]*Lz[kk] * points_x[iptr_cube];
              y_pt += Lx[ii]*Ly[jj]*Lz[kk] * points_y[iptr_cube];
              z_pt += Lx[ii]*Ly[jj]*Lz[kk] * points_z[iptr_cube];
            }
          }
        }

        // find min dist
        float DistInt = sqrtf(  (sx - x_pt) * (sx - x_pt)
            + (sy - y_pt) * (sy - y_pt)
            + (sz - z_pt) * (sz - z_pt) );

        // replace closest point
        if (min_dist > DistInt)
        {
          min_dist = DistInt;
          min_dist_i = n1;
          min_dist_j = n2;
          min_dist_k = n3;
        }
      } // n1
    } // n2
  } // n3

  *si_shift = (float)min_dist_i / (float)nx_sample;
  *sj_shift = (float)min_dist_j / (float)ny_sample;
  *sk_shift = (float)min_dist_k / (float)nz_sample;

  return 0;
}
/* 
 * find curv index using sampling
 */

__host__ __device__ int
gd_curv_coord2index_sample(float sx, float sy, float sz, 
    int num_points,
    float *points_x, // x coord of all points
    float *points_y,
    float *points_z,
    float *points_i, // curv coord of all points
    float *points_j,
    float *points_k,
    int    nx_sample,
    int    ny_sample,
    int    nz_sample,
    float *si_curv, // interped curv coord
    float *sj_curv,
    float *sk_curv)
{
  float Lx[2], Ly[2], Lz[2];
  // init closest point
  float min_dist = sqrtf(  (sx - points_x[0]) * (sx - points_x[0])
      + (sy - points_y[0]) * (sy - points_y[0])
      + (sz - points_z[0]) * (sz - points_z[0]) );
  int min_dist_i = 0 ;
  int min_dist_j = 0 ;
  int min_dist_k = 0 ;

  // linear interp for all sample
  for (int n3=0; n3<nz_sample+1; n3++)
  {
    Lz[1] = (float)(n3) / (float)(nz_sample);
    Lz[0] = 1.0 - Lz[1];
    for (int n2=0; n2<ny_sample+1; n2++)
    {
      Ly[1] = (float)(n2) / (float)(ny_sample);
      Ly[0] = 1.0 - Ly[1];
      for (int n1=0; n1<nx_sample+1; n1++)
      {
        Lx[1] = (float)(n1) / (float)(nx_sample);
        Lx[0] = 1.0 - Lx[1];

        // interp
        float x_pt=0;
        float y_pt=0;
        float z_pt=0;
        for (int kk=0; kk<2; kk++) {
          for (int jj=0; jj<2; jj++) {
            for (int ii=0; ii<2; ii++)
            {
              int iptr_cube = ii + jj * 2 + kk * 4;
              x_pt += Lx[ii]*Ly[jj]*Lz[kk] * points_x[iptr_cube];
              y_pt += Lx[ii]*Ly[jj]*Lz[kk] * points_y[iptr_cube];
              z_pt += Lx[ii]*Ly[jj]*Lz[kk] * points_z[iptr_cube];
            }
          }
        }

        // find min dist
        float DistInt = sqrtf(  (sx - x_pt) * (sx - x_pt)
            + (sy - y_pt) * (sy - y_pt)
            + (sz - z_pt) * (sz - z_pt) );

        // replace closest point
        if (min_dist > DistInt)
        {
          min_dist = DistInt;
          min_dist_i = n1;
          min_dist_j = n2;
          min_dist_k = n3;
        }
      } // n1
    } // n2
  } // n3

  *si_curv = points_i[0] + (float)min_dist_i / (float)nx_sample;
  *sj_curv = points_j[0] + (float)min_dist_j / (float)ny_sample;
  *sk_curv = points_k[0] + (float)min_dist_k / (float)nz_sample;

  return 0;
}

/* 
 * interp curv coord using inverse distance interp
 */

  int
gd_curv_coord2index_rdinterp(float sx, float sy, float sz, 
    int num_points,
    float *points_x, // x coord of all points
    float *points_y,
    float *points_z,
    float *points_i, // curv coord of all points
    float *points_j,
    float *points_k,
    float *si_curv, // interped curv coord
    float *sj_curv,
    float *sk_curv)
{
  float weight[num_points];
  float total_weight = 0.0 ;

  // cal weight
  int at_point_indx = -1;
  for (int i=0; i<num_points; i++)
  {
    float dist = sqrtf ((sx - points_x[i]) * (sx - points_x[i])
        + (sy - points_y[i]) * (sy - points_y[i])
        + (sz - points_z[i]) * (sz - points_z[i])
        );
    if (dist < 1e-9) {
      at_point_indx = i;
    } else {
      weight[i]   = 1.0 / dist;
      total_weight += weight[i];
    }
  }
  // if at a point
  if (at_point_indx > 0) {
    total_weight = 1.0;
    // other weight 0
    for (int i=0; i<num_points; i++) {
      weight[i] = 0.0;
    }
    // point weight 1
    weight[at_point_indx] = 1.0;
  }

  // interp

  *si_curv = 0.0;
  *sj_curv = 0.0;
  *sk_curv = 0.0;

  for (int i=0; i<num_points; i++)
  {
    weight[i] *= 1.0 / total_weight ;

    (*si_curv) += weight[i] * points_i[i];
    (*sj_curv) += weight[i] * points_j[i]; 
    (*sk_curv) += weight[i] * points_k[i];  

    fprintf(stdout,"---- i=%d,weight=%f,points_i=%f,points_j=%f,points_k=%f\n",
        i,weight[i],points_i[i],points_j[i],points_k[i]);
  }

  return 0;
}

float
gd_coord_get_x(gdcurv_t *gdcurv, int i, int j, int k)
{
  float var = 0.0;

  size_t iptr = i + j * gdcurv->siz_iy + k * gdcurv->siz_iz;
  var = gdcurv->x3d[iptr];

  return var;
}

float
gd_coord_get_y(gdcurv_t *gdcurv, int i, int j, int k)
{
  float var = 0.0;

  size_t iptr = i + j * gdcurv->siz_iy + k * gdcurv->siz_iz;
  var = gdcurv->y3d[iptr];

  return var;
}

float
gd_coord_get_z(gdcurv_t *gdcurv, int i, int j, int k)
{
  float var = 0.0;

  size_t iptr = i + j * gdcurv->siz_iy + k * gdcurv->siz_iz;
  var = gdcurv->z3d[iptr];

  return var;
}

/*
 * Input: vx, vy, vz are the EIGHT vertexes of the hexahedron 
 *
 *    ↑ +z       4----6
 *    |         /|   /|
 *             / 0--/-2
 *            5----7 /
 *            |/   |/
 *            1----3
 *
 *
 */
// c++ version is coding by jiangluqian
// c cersion is coding by lihualin
__host__ __device__
int isPointInHexahedron_c(float px,  float py,  float pz,
                          float *vx, float *vy, float *vz)
{
  float point[3] = {px, py, pz};
	/* 
	 * Just for cgfd3D, in which the grid mesh maybe not a hexahedron,
	 */
  // order is back front left right top bottom 
  float hexa[6][3][3] = {
  {{vx[0], vy[0], vz[0]},{vx[4], vy[4], vz[4]},{vx[6], vy[6], vz[6]}},
  {{vx[7], vy[7], vz[7]},{vx[5], vy[5], vz[5]},{vx[1], vy[1], vz[1]}},
  {{vx[5], vy[5], vz[5]},{vx[4], vy[4], vz[4]},{vx[0], vy[0], vz[0]}},
  {{vx[2], vy[2], vz[2]},{vx[6], vy[6], vz[6]},{vx[7], vy[7], vz[7]}},
  {{vx[4], vy[4], vz[4]},{vx[5], vy[5], vz[5]},{vx[7], vy[7], vz[7]}},
  {{vx[3], vy[3], vz[3]},{vx[1], vy[1], vz[1]},{vx[0], vy[0], vz[0]}},
  };

/* 
 * Check whether the point is in the polyhedron.
 * Note: The hexahedron must be convex!
 */
  float sign;
  float len_p2f;
  float p2f[3] = {0};
  float normal_unit[3] = {0};
  for(int i=0; i<6; i++)
  {
    point2face(hexa[i][0],point,p2f); 
    face_normal(hexa[i],normal_unit);
    sign = fdlib_math_dot_product(p2f,normal_unit);
    len_p2f=sqrt(fdlib_math_dot_product(p2f,p2f));
    sign /= len_p2f;
    if(sign < 0.0) return 0;
  }
  return 1;
}

__host__ __device__
int point2face(float *hexa1d,float *point, float *p2f)
{
  for(int i=0; i<3; i++)
  {
    p2f[i] = hexa1d[i] - point[i];
  }
  return 0;
}

__host__ __device__
int face_normal(float (*hexa2d)[3], float *normal_unit)
{
  float A[3];
  float B[3];
  float normal[3]; // normal vector
  float length;
  for(int i=0;i<3;i++)
  {
    A[i] = hexa2d[1][i] - hexa2d[0][i];
    B[i] = hexa2d[2][i] - hexa2d[0][i];
  }
  // calculate normal vector
  fdlib_math_cross_product(A, B, normal);
  // Normalized the normal vector
  length = sqrt(fdlib_math_dot_product(normal, normal));
  for(int i=0; i<3; i++)
  {
    normal_unit[i] = normal[i] / length;
  }

  return 0;
}

//
// set grid size
//

int
gd_info_set(gdcurv_t *const gdcurv,
            const mympi_t *const mympi,
            const int number_of_total_grid_points_x,
            const int number_of_total_grid_points_y,
            const int number_of_total_grid_points_z,
                  int abs_num_of_layers[][2],
            const int fdx_nghosts,
            int const fdy_nghosts,
            const int fdz_nghosts,
            const int verbose)
{
  int ierr = 0;

  // determine ni
  int nx_et = number_of_total_grid_points_x;

  // double cfspml load
  nx_et += abs_num_of_layers[0][0] + abs_num_of_layers[0][1];

  // partition into average plus left at last
  int nx_avg  = nx_et / mympi->nprocx;
  int nx_left = nx_et % mympi->nprocx;

  if (nx_avg < 2 * fdx_nghosts) {
    fprintf(stdout,"should not be less than 2 * fdx_nghosts");
    exit(1);
  }

  if (nx_avg<abs_num_of_layers[0][0] || nx_avg<abs_num_of_layers[0][1]) {
    fprintf(stdout,"should not be less than abs_num_of_layers");
    exit(1);
  }

  // default set to average value
  int ni = nx_avg;
  // subtract nlay for pml node
  if (mympi->neighid[0] == MPI_PROC_NULL) {
    ni -= abs_num_of_layers[0][0];
  }
  if (mympi->neighid[1] == MPI_PROC_NULL) {
    ni -= abs_num_of_layers[0][1];
  }
  // first nx_left node add one more point
  if (mympi->topoid[0] < nx_left) {
    ni++;
  }
  // global index
  if (mympi->topoid[0]==0) {
    gdcurv->gni1 = 0;
  } else {
    gdcurv->gni1 = mympi->topoid[0] * nx_avg - abs_num_of_layers[0][0];
  }
  if (nx_left != 0) {
    gdcurv->gni1 += (mympi->topoid[0] < nx_left) ? mympi->topoid[0] : nx_left;
  }

  // determine nj
  int ny_et = number_of_total_grid_points_y;
  // double cfspml load
  ny_et += abs_num_of_layers[1][0] + abs_num_of_layers[1][1];
  int ny_avg  = ny_et / mympi->nprocy;
  int ny_left = ny_et % mympi->nprocy;

  if (ny_avg < 2 * fdy_nghosts) {
    fprintf(stdout,"should not be less than 2 * fdy_nghosts");
    exit(1);
  }
  if (ny_avg<abs_num_of_layers[1][0] || ny_avg<abs_num_of_layers[1][1]) {
    fprintf(stdout,"should not be less than abs_num_of_layers");
    exit(1);
  }
  int nj = ny_avg;
  if (mympi->neighid[2] == MPI_PROC_NULL) {
    nj -= abs_num_of_layers[1][0];
  }
  if (mympi->neighid[3] == MPI_PROC_NULL) {
    nj -= abs_num_of_layers[1][1];
  }
  // not equal divided points given to first ny_left procs
  if (mympi->topoid[1] < ny_left) {
    nj++;
  }
  // global index
  if (mympi->topoid[1]==0) {
    gdcurv->gnj1 = 0;
  } else {
    gdcurv->gnj1 = mympi->topoid[1] * ny_avg - abs_num_of_layers[1][0];
  }
  if (ny_left != 0) {
    gdcurv->gnj1 += (mympi->topoid[1] < ny_left) ? mympi->topoid[1] : ny_left;
  }

  // determine nk
  int nz_et = number_of_total_grid_points_z;
  // double cfspml load
  // if has free_surface, abs_num_of_layers[2][1] = 0
  nz_et += abs_num_of_layers[2][0] + abs_num_of_layers[2][1];
  int nz_avg  = nz_et / mympi->nprocz;
  int nz_left = nz_et % mympi->nprocz;
  if (nz_avg < 2 * fdz_nghosts) {
    fprintf(stdout,"should not be less than 2 * fdz_nghosts");
    exit(1);
  }
  if (nz_avg<abs_num_of_layers[2][0] || nz_avg<abs_num_of_layers[2][1]) {
    fprintf(stdout,"should not be less than abs_num_of_layers");
    exit(1);
  }
  int nk = nz_avg;
  if (mympi->neighid[4] == MPI_PROC_NULL) {
    nk -= abs_num_of_layers[2][0];
  }
  if (mympi->neighid[5] == MPI_PROC_NULL) {
    nk -= abs_num_of_layers[2][1];
  }
  // not equal divided points given to first nz_left procs
  if (mympi->topoid[2] < nz_left) {
    nk++;
  }
  // global index
  if (mympi->topoid[2]==0) {
    gdcurv->gnk1 = 0;
  } else {
    gdcurv->gnk1 = mympi->topoid[2] * nz_avg - abs_num_of_layers[2][0];
  }
  if (nz_left != 0) {
    gdcurv->gnk1 += (mympi->topoid[2] < nz_left) ? mympi->topoid[2] : nz_left;
  }
  
  // add ghost points
  int nx = ni + 2 * fdx_nghosts;
  int ny = nj + 2 * fdy_nghosts;
  int nz = nk + 2 * fdz_nghosts;

  gdcurv->ni = ni;
  gdcurv->nj = nj;
  gdcurv->nk = nk;

  gdcurv->nx = nx;
  gdcurv->ny = ny;
  gdcurv->nz = nz;

  gdcurv->ni1 = fdx_nghosts;
  gdcurv->ni2 = gdcurv->ni1 + ni - 1;

  gdcurv->nj1 = fdy_nghosts;
  gdcurv->nj2 = gdcurv->nj1 + nj - 1;

  gdcurv->nk1 = fdz_nghosts;
  gdcurv->nk2 = gdcurv->nk1 + nk - 1;

  // global index end
  gdcurv->gni2 = gdcurv->gni1 + gdcurv->ni - 1;
  gdcurv->gnj2 = gdcurv->gnj1 + gdcurv->nj - 1;
  gdcurv->gnk2 = gdcurv->gnk1 + gdcurv->nk - 1;

  gdcurv->ni1_to_glob_phys0 = gdcurv->gni1;
  gdcurv->ni2_to_glob_phys0 = gdcurv->gni2;
  gdcurv->nj1_to_glob_phys0 = gdcurv->gnj1;
  gdcurv->nj2_to_glob_phys0 = gdcurv->gnj2;
  gdcurv->nk1_to_glob_phys0 = gdcurv->gnk1;
  gdcurv->nk2_to_glob_phys0 = gdcurv->gnk2;
  
  gdcurv->npoint_x = number_of_total_grid_points_x; 
  gdcurv->npoint_y = number_of_total_grid_points_y; 
  gdcurv->npoint_z = number_of_total_grid_points_z;

  // x dimention varies first
  gdcurv->siz_iy   = nx; 
  gdcurv->siz_iz   = nx * ny; 
  gdcurv->siz_icmp = gdcurv->siz_iz * nz;

  gdcurv->siz_slice_yz = ny * nz;
  gdcurv->siz_slice_yz2 = 2 * ny * nz;

  // set npoint_ghosts according to fdz_nghosts
  gdcurv->npoint_ghosts = fdz_nghosts;

  gdcurv->fdx_nghosts = fdx_nghosts;
  gdcurv->fdy_nghosts = fdy_nghosts;
  gdcurv->fdz_nghosts = fdz_nghosts;

  gdcurv->index_name = fdlib_mem_malloc_2l_char(
                        CONST_NDIM, CONST_MAX_STRLEN, "gdcurv name");

  // grid coord name
  sprintf(gdcurv->index_name[0],"%s","i");
  sprintf(gdcurv->index_name[1],"%s","j");
  sprintf(gdcurv->index_name[2],"%s","k");

  return ierr;
}

/*
 * give a local index ref, check if in this thread
 */

int
gd_info_lindx_is_inner(int i, int j, int k, gdcurv_t *gdcurv)
{
  int is_in = 0;

  if (   i >= gdcurv->ni1 && i <= gdcurv->ni2
      && j >= gdcurv->nj1 && j <= gdcurv->nj2
      && k >= gdcurv->nk1 && k <= gdcurv->nk2)
  {
    is_in = 1;
  }

  return is_in;
}  

/*
 * give a global index ref to phys0, check if in this thread
 */

int
gd_info_gindx_is_inner(int gi, int gj, int gk, gdcurv_t *gdcurv)
{
  int ishere = 0;

  if ( gi >= gdcurv->ni1_to_glob_phys0 && gi <= gdcurv->ni2_to_glob_phys0 &&
       gj >= gdcurv->nj1_to_glob_phys0 && gj <= gdcurv->nj2_to_glob_phys0 &&
       gk >= gdcurv->nk1_to_glob_phys0 && gk <= gdcurv->nk2_to_glob_phys0 )
  {
    ishere = 1;
  }

  return ishere;
}

/*
 * glphyinx, glextind, gp,ge
 * lcphyind, lcextind
 * gl: global
 * lc: local
 * inx: index
 * phy: physical points only, do not count ghost
 * ext: include extended points, with ghots points
 */

int
gd_info_gindx_is_inner_i(int gi, gdcurv_t *gdcurv)
{
  int ishere = 0;

  if ( gi >= gdcurv->ni1_to_glob_phys0 && gi <= gdcurv->ni2_to_glob_phys0)
  {
    ishere = 1;
  }

  return ishere;
}

int
gd_info_gindx_is_inner_j(int gj, gdcurv_t *gdcurv)
{
  int ishere = 0;

  if ( gj >= gdcurv->nj1_to_glob_phys0 && gj <= gdcurv->nj2_to_glob_phys0)
  {
    ishere = 1;
  }

  return ishere;
}

int
gd_info_gindx_is_inner_k(int gk, gdcurv_t *gdcurv)
{
  int ishere = 0;

  if ( gk >= gdcurv->nk1_to_glob_phys0 && gk <= gdcurv->nk2_to_glob_phys0)
  {
    ishere = 1;
  }

  return ishere;
}

/*
 * convert global index to local
 */

int
gd_info_ind_glphy2lcext_i(int gi, gdcurv_t *gdcurv)
{
  return gi - gdcurv->ni1_to_glob_phys0 + gdcurv->npoint_ghosts;
}

int
gd_info_ind_glphy2lcext_j(int gj, gdcurv_t *gdcurv)
{
  return gj - gdcurv->nj1_to_glob_phys0 + gdcurv->npoint_ghosts;
}

int
gd_info_ind_glphy2lcext_k(int gk, gdcurv_t *gdcurv)
{
  return gk - gdcurv->nk1_to_glob_phys0 + gdcurv->npoint_ghosts;
}

/*
 * convert local index to global
 */

__host__ __device__ int
gd_info_ind_lcext2glphy_i(int i, gdcurv_t *gdcurv)
{
  return i - gdcurv->npoint_ghosts + gdcurv->ni1_to_glob_phys0;
}

__host__ __device__ int
gd_info_ind_lcext2glphy_j(int j, gdcurv_t *gdcurv)
{
  return j - gdcurv->npoint_ghosts + gdcurv->nj1_to_glob_phys0;
}

__host__ __device__ int
gd_info_ind_lcext2glphy_k(int k, gdcurv_t *gdcurv)
{
  return k - gdcurv->npoint_ghosts + gdcurv->nk1_to_glob_phys0;
}

/*
 * print for QC
 */

int
gd_info_print(gdcurv_t *gdcurv)
{    
  fprintf(stdout, "-------------------------------------------------------\n");
  fprintf(stdout, "--> grid info:\n");
  fprintf(stdout, "-------------------------------------------------------\n");
  fprintf(stdout, " nx    = %-10d\n", gdcurv->nx);
  fprintf(stdout, " ny    = %-10d\n", gdcurv->ny);
  fprintf(stdout, " nz    = %-10d\n", gdcurv->nz);
  fprintf(stdout, " ni    = %-10d\n", gdcurv->ni);
  fprintf(stdout, " nj    = %-10d\n", gdcurv->nj);
  fprintf(stdout, " nk    = %-10d\n", gdcurv->nk);

  fprintf(stdout, " ni1   = %-10d\n", gdcurv->ni1);
  fprintf(stdout, " ni2   = %-10d\n", gdcurv->ni2);
  fprintf(stdout, " nj1   = %-10d\n", gdcurv->nj1);
  fprintf(stdout, " nj2   = %-10d\n", gdcurv->nj2);
  fprintf(stdout, " nk1   = %-10d\n", gdcurv->nk1);
  fprintf(stdout, " nk2   = %-10d\n", gdcurv->nk2);

  fprintf(stdout, " ni1_to_glob_phys0   = %-10d\n", gdcurv->gni1);
  fprintf(stdout, " ni2_to_glob_phys0   = %-10d\n", gdcurv->gni2);
  fprintf(stdout, " nj1_to_glob_phys0   = %-10d\n", gdcurv->gnj1);
  fprintf(stdout, " nj2_to_glob_phys0   = %-10d\n", gdcurv->gnj2);
  fprintf(stdout, " nk1_to_glob_phys0   = %-10d\n", gdcurv->gnk1);
  fprintf(stdout, " nk2_to_glob_phys0   = %-10d\n", gdcurv->gnk2);

  fprintf(stdout, "\n-------------------------------------------------------\n");
  fprintf(stdout, "print grid structure info:\n");
  fprintf(stdout, "-------------------------------------------------------\n\n");

  fprintf(stdout," xmin=%g, xmax=%g\n", gdcurv->xmin,gdcurv->xmax);
  fprintf(stdout," ymin=%g, ymax=%g\n", gdcurv->ymin,gdcurv->ymax);
  fprintf(stdout," zmin=%g, zmax=%g\n", gdcurv->zmin,gdcurv->zmax);
  /*
  for (int k_tile = 0; k_tile < GD_TILE_NZ; k_tile++)
  {
    fprintf(stdout," tile k=%d, pt k in (%d,%d)\n",
                k_tile, gdcurv->tile_kstart[k_tile],gdcurv->tile_kend[k_tile]);
  }
  for (int j_tile = 0; j_tile < GD_TILE_NY; j_tile++)
  {
    fprintf(stdout," tile j=%d, pt j in (%d,%d)\n",
                  j_tile, gdcurv->tile_jstart[j_tile],gdcurv->tile_jend[j_tile]);
  }
  for (int i_tile = 0; i_tile < GD_TILE_NX; i_tile++)
  {
    fprintf(stdout," tile i=%d, pt i in (%d,%d)\n",
                  i_tile, gdcurv->tile_istart[i_tile],gdcurv->tile_iend[i_tile]);
  }
  for (int k_tile = 0; k_tile < GD_TILE_NZ; k_tile++)
  {
    for (int j_tile = 0; j_tile < GD_TILE_NY; j_tile++)
    {
      for (int i_tile = 0; i_tile < GD_TILE_NX; i_tile++)
      {
        int iptr_tile = i_tile + j_tile * GD_TILE_NX + k_tile * GD_TILE_NX * GD_TILE_NY;
        fprintf(stdout," tile %d,%d,%d, range (%g,%g,%g,%g,%g,%g)\n",
                i_tile,j_tile,k_tile,
                gdcurv->tile_xmin[iptr_tile],
                gdcurv->tile_xmax[iptr_tile],
                gdcurv->tile_ymin[iptr_tile],
                gdcurv->tile_ymax[iptr_tile],
                gdcurv->tile_zmin[iptr_tile],
                gdcurv->tile_zmax[iptr_tile]);
      }
    }
  }
  */

  fflush(stdout);

  return(0);
}
