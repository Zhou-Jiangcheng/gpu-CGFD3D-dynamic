/*
 * 
 */

//#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "par_t.h"

/*
 * for MPI, master read, broadcast to all procs
 */
int
par_mpi_get(char *par_fname, int myid, MPI_Comm comm, par_t *par, int verbose)
{
  char *str;

  if (myid==0)
  {
    FILE *fp=fopen(par_fname,"r");
    if (!fp) {
      fprintf(stderr,"Error: can't open par file: %s\n", par_fname);
      MPI_Abort(MPI_COMM_WORLD,9);
    }
    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);

    // bcast len to all nodes
    MPI_Bcast(&len, 1, MPI_LONG, 0, comm);

    fseek(fp, 0, SEEK_SET);
    str = (char*)malloc(len+1);
    fread(str, 1, len, fp);
    fclose(fp);

    // bcast str
    MPI_Bcast(str, len+1, MPI_CHAR, 0, comm);
  }
  else
  {
    long len;
    // get len 
    MPI_Bcast(&len, 1, MPI_LONG, 0, comm);

    str = (char*)malloc(len+1);
    // get str
    MPI_Bcast(str, len+1, MPI_CHAR, 0, comm);
  }
    
  par_read_from_str(str, par);

  free(str);

  return 0;
}

/*
 * funcs to get par from alread read in str
 */
int 
par_read_from_str(const char *str, par_t *par)
{
  int ierr = 0;

  // allocate
  par->boundary_type_name = (char **)malloc(CONST_NDIM_2 * sizeof(char*));
  for (int i=0; i<CONST_NDIM_2; i++) {
    par->boundary_type_name[i] = (char *)malloc(30*sizeof(char));
  }

  // convert str to json
  cJSON *root = cJSON_Parse(str);
  if (NULL == root) {
    printf("Error at parsing json!\n");
    MPI_Abort(MPI_COMM_WORLD,9);
  }

  cJSON *item;
  cJSON *subitem, *thirditem, *snapitem, *lineitem;

  // no default
  if (item = cJSON_GetObjectItem(root, "number_of_total_grid_points_x")) {
    par->number_of_total_grid_points_x = item->valueint;
  }
  if (item = cJSON_GetObjectItem(root, "number_of_total_grid_points_y")) {
    par->number_of_total_grid_points_y = item->valueint;
  }
  if (item = cJSON_GetObjectItem(root, "number_of_total_grid_points_z")) {
    par->number_of_total_grid_points_z = item->valueint;
  }

  // default mpi threads
  par->number_of_mpiprocs_x = 1;
  par->number_of_mpiprocs_y = 1;
  par->number_of_mpiprocs_z = 1;

  if (item = cJSON_GetObjectItem(root, "number_of_mpiprocs_y")) {
    par->number_of_mpiprocs_y = item->valueint;
  }
  if (item = cJSON_GetObjectItem(root, "number_of_mpiprocs_z")) {
    par->number_of_mpiprocs_z = item->valueint;
  }

  // set default values to negative
  par->size_of_time_step = -1.0;
  par->number_of_time_steps = -1;
  par->time_window_length = -1.0;
  par->time_start = 0.0;
  par->time_check_stability = 1;
  par->io_time_skip = 1;

  if (item = cJSON_GetObjectItem(root, "size_of_time_step")) {
    par->size_of_time_step = item->valuedouble;
  }
  if (item = cJSON_GetObjectItem(root, "number_of_time_steps")) {
    par->number_of_time_steps = item->valueint;
  }
  if (item = cJSON_GetObjectItem(root, "time_window_length")) {
    par->time_window_length = item->valuedouble;
  }
  if (item = cJSON_GetObjectItem(root, "time_start")) {
    par->time_start = item->valuedouble;
  }
  if (item = cJSON_GetObjectItem(root, "time_check_stability")) {
    par->time_check_stability = item->valueint;
  }
  if (item = cJSON_GetObjectItem(root, "io_time_skip")) {
    par->io_time_skip = item->valueint;
  }

  if (par->size_of_time_step < 0.0 && par->time_window_length < 0)
  {
    fprintf(stderr," --> size_of_time_step   =%f\n", par->size_of_time_step);
    fprintf(stderr," --> time_window_length  =%f\n", par->time_window_length);
    fprintf(stderr,"Error: at lest one of above paras should > 0\n");
    MPI_Abort(MPI_COMM_WORLD,9);
  }

  if (par->size_of_time_step > 0.0)
  {
    if (par->number_of_time_steps < 0 && par->time_window_length < 0.0)
    {
      fprintf(stderr,"Error: both size_of_time_step=%f, time_window_length=%f less 0\n",
             par->size_of_time_step, par->time_window_length);
      MPI_Abort(MPI_COMM_WORLD,9);
    }

    if (par->number_of_time_steps < 0) 
    {
      par->number_of_time_steps = (int)(par->time_window_length / par->size_of_time_step + 0.5);
    }

    par->time_window_length = par->size_of_time_step * par->number_of_time_steps;
  }


  //
  // boundary default values
  //
  for (int idim=0; idim < CONST_NDIM; idim++) {
    for (int iside=0; iside < 2; iside++) {
      par->abs_num_of_layers[idim][iside] = 0;
      par->cfspml_is_sides[idim][iside] = 0;
      par->free_is_sides  [idim][iside] = 0;
      par->ablexp_is_sides[idim][iside] = 0;
    }
  }
  par->bdry_has_cfspml = 0;
  par->bdry_has_free   = 0;
  par->bdry_has_ablexp = 0;

  // x1 boundary, no default yet
  if (item = cJSON_GetObjectItem(root, "boundary_x_left"))
  {
    if (subitem = cJSON_GetObjectItem(item, "cfspml")) {
       sprintf(par->boundary_type_name[0], "%s", "cfspml");
       par_read_json_cfspml(subitem,
                            &(par->abs_num_of_layers[0][0]),
                            &(par->cfspml_alpha_max [0][0]),
                            &(par->cfspml_beta_max  [0][0]),
                            &(par->cfspml_velocity  [0][0]));
      par->cfspml_is_sides[0][0] = 1;
      par->bdry_has_cfspml = 1;
    }
	  if (subitem = cJSON_GetObjectItem(item, "ablexp")) {
      sprintf(par->boundary_type_name[0], "%s", "ablexp");
      par_read_json_ablexp(subitem,
                           &(par->abs_num_of_layers[0][0]),
                           &(par->ablexp_velocity  [0][0]));
      par->ablexp_is_sides[0][0] = 1;
      par->bdry_has_ablexp = 1;
    }
  }
  // x2 boundary, no default yet
  if (item = cJSON_GetObjectItem(root, "boundary_x_right"))
  {
    if (subitem = cJSON_GetObjectItem(item, "cfspml")) {
       sprintf(par->boundary_type_name[1], "%s", "cfspml");
       par_read_json_cfspml(subitem,
                            &(par->abs_num_of_layers[0][1]),
                            &(par->cfspml_alpha_max [0][1]),
                            &(par->cfspml_beta_max  [0][1]),
                            &(par->cfspml_velocity  [0][1]));
      par->cfspml_is_sides[0][1] = 1;
      par->bdry_has_cfspml = 1;
    }
	  if (subitem = cJSON_GetObjectItem(item, "ablexp")) {
      sprintf(par->boundary_type_name[1], "%s", "ablexp");
      par_read_json_ablexp(subitem,
                           &(par->abs_num_of_layers[0][1]),
                           &(par->ablexp_velocity  [0][1]));
      par->ablexp_is_sides[0][1] = 1;
      par->bdry_has_ablexp = 1;
    }
  }
  // y1 boundary, no default yet
  if (item = cJSON_GetObjectItem(root, "boundary_y_front"))
  {
    if (subitem = cJSON_GetObjectItem(item, "cfspml")) {
       sprintf(par->boundary_type_name[2], "%s", "cfspml");
       par_read_json_cfspml(subitem,
                            &(par->abs_num_of_layers[1][0]),
                            &(par->cfspml_alpha_max [1][0]),
                            &(par->cfspml_beta_max  [1][0]),
                            &(par->cfspml_velocity  [1][0]));
      par->cfspml_is_sides[1][0] = 1;
      par->bdry_has_cfspml = 1;
    }
	  if (subitem = cJSON_GetObjectItem(item, "ablexp")) {
      sprintf(par->boundary_type_name[2], "%s", "ablexp");
      par_read_json_ablexp(subitem,
                           &(par->abs_num_of_layers[1][0]),
                           &(par->ablexp_velocity  [1][0]));
      par->ablexp_is_sides[1][0] = 1;
      par->bdry_has_ablexp = 1;
    }
  }
  // y2 boundary, no default yet
  if (item = cJSON_GetObjectItem(root, "boundary_y_back"))
  {
    if (subitem = cJSON_GetObjectItem(item, "cfspml")) {
       sprintf(par->boundary_type_name[3], "%s", "cfspml");
       par_read_json_cfspml(subitem,
                            &(par->abs_num_of_layers[1][1]),
                            &(par->cfspml_alpha_max [1][1]),
                            &(par->cfspml_beta_max  [1][1]),
                            &(par->cfspml_velocity  [1][1]));
      par->cfspml_is_sides[1][1] = 1;
      par->bdry_has_cfspml = 1;
    }
	  if (subitem = cJSON_GetObjectItem(item, "ablexp")) {
      sprintf(par->boundary_type_name[3], "%s", "ablexp");
      par_read_json_ablexp(subitem,
                           &(par->abs_num_of_layers[1][1]),
                           &(par->ablexp_velocity  [1][1]));
      par->ablexp_is_sides[1][1] = 1;
      par->bdry_has_ablexp = 1;
    }
  }
  // z1 boundary, no default yet
  if (item = cJSON_GetObjectItem(root, "boundary_z_bottom"))
  {
    if (subitem = cJSON_GetObjectItem(item, "cfspml")) {
       sprintf(par->boundary_type_name[4], "%s", "cfspml");
       par_read_json_cfspml(subitem,
                            &(par->abs_num_of_layers[2][0]),
                            &(par->cfspml_alpha_max [2][0]),
                            &(par->cfspml_beta_max  [2][0]),
                            &(par->cfspml_velocity  [2][0]));
      par->cfspml_is_sides[2][0] = 1;
      par->bdry_has_cfspml = 1;
    }
	  if (subitem = cJSON_GetObjectItem(item, "ablexp")) {
      sprintf(par->boundary_type_name[4], "%s", "ablexp");
      par_read_json_ablexp(subitem,
                           &(par->abs_num_of_layers[2][0]),
                           &(par->ablexp_velocity  [2][0]));
      par->ablexp_is_sides[2][0] = 1;
      par->bdry_has_ablexp = 1;
    }
  }
  // z2 boundary, no default yet
  if (item = cJSON_GetObjectItem(root, "boundary_z_top"))
  {
    if (subitem = cJSON_GetObjectItem(item, "cfspml")) {
       sprintf(par->boundary_type_name[5], "%s", "cfspml");
       par_read_json_cfspml(subitem,
                            &(par->abs_num_of_layers[2][1]),
                            &(par->cfspml_alpha_max [2][1]),
                            &(par->cfspml_beta_max  [2][1]),
                            &(par->cfspml_velocity  [2][1]));
      par->cfspml_is_sides[2][1] = 1;
      par->bdry_has_cfspml = 1;
    }
	  if (subitem = cJSON_GetObjectItem(item, "ablexp")) {
      sprintf(par->boundary_type_name[5], "%s", "ablexp");
      par_read_json_ablexp(subitem,
                           &(par->abs_num_of_layers[2][1]),
                           &(par->ablexp_velocity  [2][1]));
      par->ablexp_is_sides[2][1] = 1;
      par->bdry_has_ablexp = 1;
    }
    if (subitem = cJSON_GetObjectItem(item, "free")) {
      sprintf(par->boundary_type_name[5], "%s", "free");
      par->free_is_sides[2][1] = 1;
      par->bdry_has_free = 1;
    }
  }
  //method
  if (item = cJSON_GetObjectItem(root, "dynamic_method")) {
    par->imethod = item->valueint;
  }
  if (item = cJSON_GetObjectItem(root, "fault_x_index")) 
  {
    par->number_fault = cJSON_GetArraySize(item);
    par->fault_x_index = (int *)malloc(par->number_fault * sizeof(int));
    // need minus 1. C index start from 1.
    for (int id=0; id<par->number_fault; id++)
    {
      par->fault_x_index[id] = cJSON_GetArrayItem(item, id)->valueint-1;
      if(par->fault_x_index[id] < 0 || par->fault_x_index[id] > par->number_of_total_grid_points_x)
      {
        fprintf(stderr,"fault index %d out calculate area!\n", par->fault_x_index[id]);
        fflush(stderr);
        exit(1);
      }
    }
  }
  if (item = cJSON_GetObjectItem(root, "fault_grid")) 
  {
    par->fault_grid = (int *)malloc(par->number_fault * 4 * sizeof(int));
    for (int id=0; id<par->number_fault; id++)
    {
      for(int j=0; j<4; j++) {
        // need minus 1. C index start from 1.
        par->fault_grid[j+4*id] = cJSON_GetArrayItem(item, j+4*id)->valueint-1;
      }
    }
  }

  //
  //-- grid
  //
  if (item = cJSON_GetObjectItem(root, "grid_generation_method")) {
    // fault import
    if (subitem = cJSON_GetObjectItem(item, "fault_plane")) {
      par->grid_generation_itype = FAULT_PLANE;
      if (thirditem = cJSON_GetObjectItem(subitem, "fault_geometry_dir")) {
         sprintf(par->fault_coord_dir, "%s", thirditem->valuestring);
      }
      if (thirditem = cJSON_GetObjectItem(subitem, "fault_init_stress_dir")) {
         sprintf(par->init_stress_dir, "%s", thirditem->valuestring);
      }
      if (thirditem = cJSON_GetObjectItem(subitem, "fault_inteval")) {
        par->dh = thirditem->valuedouble;
      }
    }
    // 3D grid import
    if (subitem = cJSON_GetObjectItem(item, "grid_import")) {
      par->grid_generation_itype = GRID_IMPORT;
      if (thirditem = cJSON_GetObjectItem(subitem, "import_dir")) {
         sprintf(par->grid_import_dir, "%s", thirditem->valuestring);
      }
      if (thirditem = cJSON_GetObjectItem(subitem, "fault_init_stress_dir")) {
         sprintf(par->init_stress_dir, "%s", thirditem->valuestring);
      }
    }
  }

  par->is_export_grid = 1;
  if (item = cJSON_GetObjectItem(root, "is_export_grid")) {
     par->is_export_grid = item->valueint;
  }
  if (item = cJSON_GetObjectItem(root, "grid_export_dir")) {
      sprintf(par->grid_export_dir,"%s",item->valuestring);
  }

  //
  //-- metric
  //

  par->metric_method_itype = PAR_METRIC_CALCULATE;
  if (item = cJSON_GetObjectItem(root, "metric_calculation_method")) {
    if (subitem = cJSON_GetObjectItem(item, "import")) {
        par->metric_method_itype = PAR_METRIC_IMPORT;
        sprintf(par->metric_import_dir, "%s", subitem->valuestring);
    }
    if (subitem = cJSON_GetObjectItem(item, "calculate")) {
        par->metric_method_itype = PAR_METRIC_CALCULATE;
    }
  }

  par->is_export_metric = 1;
  if (item = cJSON_GetObjectItem(root, "is_export_metric")) {
     par->is_export_metric = item->valueint;
  }

  //
  //-- medium
  //

  par->media_input_itype = PAR_MEDIA_IMPORT;
  if (item = cJSON_GetObjectItem(root, "medium"))
  {
    // medium is iso, vti or aniso
    if (subitem = cJSON_GetObjectItem(item, "type")) {
        sprintf(par->media_type, "%s", subitem->valuestring);
        if (strcmp(par->media_type, "elastic_iso")==0) {
          par->media_itype = CONST_MEDIUM_ELASTIC_ISO;
        } else if (strcmp(par->media_type, "elastic_vti")==0) {
          par->media_itype = CONST_MEDIUM_ELASTIC_VTI;
        } else if (strcmp(par->media_type, "elastic_aniso")==0) {
          par->media_itype = CONST_MEDIUM_ELASTIC_ANISO;
        } else if (strcmp(par->media_type, "acoustic_iso")==0) {
          par->media_itype = CONST_MEDIUM_ACOUSTIC_ISO;
        } else {
          fprintf(stderr,"ERROR: media_type=%s is unknown\n",par->media_type);
          MPI_Abort(MPI_COMM_WORLD,9);
        }
    }

    // input way
    if (subitem = cJSON_GetObjectItem(item, "input_way")) 
    {
        sprintf(par->media_input_way, "%s", subitem->valuestring);
    }

    // if input by code
    if (strcmp(par->media_input_way,"code") == 0)
    {
      par->media_input_itype = PAR_MEDIA_CODE;
      if (subitem = cJSON_GetObjectItem(item, "code")) 
      {
        // implement read func name later
      }
    }

    // if input by import
    if (strcmp(par->media_input_way,"import") == 0)
    {
      par->media_input_itype = PAR_MEDIA_IMPORT;
      if (subitem = cJSON_GetObjectItem(item, "import")) 
      { 
          sprintf(par->media_import_dir, "%s", subitem->valuestring);
      }
    }

    // if input by layer file
    if (strcmp(par->media_input_way,"infile_layer") == 0)
    {
      par->media_input_itype = PAR_MEDIA_3LAY;
      if (subitem = cJSON_GetObjectItem(item, "infile_layer")) 
      {
          sprintf(par->media_input_file, "%s", subitem->valuestring);
      }
    }

    // if input by grid file
    if (strcmp(par->media_input_way,"infile_grid") == 0)
    {
      par->media_input_itype = PAR_MEDIA_3GRD;
      if (subitem = cJSON_GetObjectItem(item, "infile_grid"))
      {
          sprintf(par->media_input_file, "%s", subitem->valuestring);
      }
    }

    // if input by bin file
    if (strcmp(par->media_input_way,"binfile") == 0)
    {
      par->media_input_itype = PAR_MEDIA_3BIN;
      if (subitem = cJSON_GetObjectItem(item, "binfile"))
      {
        // size
        if (thirditem = cJSON_GetObjectItem(subitem, "size")) {
          for (int i = 0; i < CONST_NDIM; i++) {
            par->bin_size[i] = cJSON_GetArrayItem(thirditem, i)->valueint;
          }
        }
        // spacing
        if (thirditem = cJSON_GetObjectItem(subitem, "spacing")) {
          for (int i = 0; i < CONST_NDIM; i++) {
            par->bin_spacing[i] = cJSON_GetArrayItem(thirditem, i)->valuedouble;
          }
        }
        // origin
        if (thirditem = cJSON_GetObjectItem(subitem, "origin")) {
          for (int i = 0; i < CONST_NDIM; i++) {
            par->bin_origin[i] = cJSON_GetArrayItem(thirditem, i)->valuedouble;
          }
        }
        // dim1
        if (thirditem = cJSON_GetObjectItem(subitem, "dim1"))
        {
          sprintf(par->bin_dim1_name, "%s", thirditem->valuestring);
          if (strcmp(par->bin_dim1_name,"x")==0) {
            par->bin_order[0] = 0;
          } else if (strcmp(par->bin_dim1_name,"y")==0) {
            par->bin_order[0] = 1;
          } else if (strcmp(par->bin_dim1_name,"z")==0) {
            par->bin_order[0] = 2;
          }
        }
        // dim2
        if (thirditem = cJSON_GetObjectItem(subitem, "dim2"))
        {
          sprintf(par->bin_dim2_name, "%s", thirditem->valuestring);
          if (strcmp(par->bin_dim2_name,"x")==0) {
            par->bin_order[1] = 0;
          } else if (strcmp(par->bin_dim2_name,"y")==0) {
            par->bin_order[1] = 1;
          } else if (strcmp(par->bin_dim2_name,"z")==0) {
            par->bin_order[1] = 2;
          }
        }
        // dim3
        if (thirditem = cJSON_GetObjectItem(subitem, "dim3"))
        {
          sprintf(par->bin_dim3_name, "%s", thirditem->valuestring);
          if (strcmp(par->bin_dim3_name,"x")==0) {
            par->bin_order[2] = 0;
          } else if (strcmp(par->bin_dim3_name,"y")==0) {
            par->bin_order[2] = 1;
          } else if (strcmp(par->bin_dim3_name,"z")==0) {
            par->bin_order[2] = 2;
          }
        }

        // rho file
        if (thirditem = cJSON_GetObjectItem(subitem, "rho"))
        {
          sprintf(par->bin_file_rho, "%s", thirditem->valuestring);
        }
        // Vp file
        if (thirditem = cJSON_GetObjectItem(subitem, "Vp"))
        {
          sprintf(par->bin_file_vp, "%s", thirditem->valuestring);
        }
        // Vs file
        if (thirditem = cJSON_GetObjectItem(subitem, "Vs"))
        {
          sprintf(par->bin_file_vs, "%s", thirditem->valuestring);
        }
        // epsilon file
        if (thirditem = cJSON_GetObjectItem(subitem, "epsilon"))
        {
          sprintf(par->bin_file_epsilon, "%s", thirditem->valuestring);
        }
        // delta file
        if (thirditem = cJSON_GetObjectItem(subitem, "delta"))
        {
          sprintf(par->bin_file_delta, "%s", thirditem->valuestring);
        }
        // gamma file
        if (thirditem = cJSON_GetObjectItem(subitem, "gamma"))
        {
          sprintf(par->bin_file_gamma, "%s", thirditem->valuestring);
        }
        // c11 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c11"))
        {
          sprintf(par->bin_file_c11, "%s", thirditem->valuestring);
        }
        // c12 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c12"))
        {
          sprintf(par->bin_file_c12, "%s", thirditem->valuestring);
        }
        // c13 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c13"))
        {
          sprintf(par->bin_file_c13, "%s", thirditem->valuestring);
        }
        // c14 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c14"))
        {
          sprintf(par->bin_file_c14, "%s", thirditem->valuestring);
        }
        // c15 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c15"))
        {
          sprintf(par->bin_file_c15, "%s", thirditem->valuestring);
        }
        // c16 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c16"))
        {
          sprintf(par->bin_file_c16, "%s", thirditem->valuestring);
        }
        // c22 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c22"))
        {
          sprintf(par->bin_file_c22, "%s", thirditem->valuestring);
        }
        // c23 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c23"))
        {
          sprintf(par->bin_file_c23, "%s", thirditem->valuestring);
        }
        // c24 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c24"))
        {
          sprintf(par->bin_file_c24, "%s", thirditem->valuestring);
        }
        // c25 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c25"))
        {
          sprintf(par->bin_file_c25, "%s", thirditem->valuestring);
        }
        // c26 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c26"))
        {
          sprintf(par->bin_file_c26, "%s", thirditem->valuestring);
        }
        // c33 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c33"))
        {
          sprintf(par->bin_file_c33, "%s", thirditem->valuestring);
        }
        // c34 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c34"))
        {
          sprintf(par->bin_file_c34, "%s", thirditem->valuestring);
        }
        // c35 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c35"))
        {
          sprintf(par->bin_file_c35, "%s", thirditem->valuestring);
        }
        // c36 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c36"))
        {
          sprintf(par->bin_file_c36, "%s", thirditem->valuestring);
        }
        // c44 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c44"))
        {
          sprintf(par->bin_file_c44, "%s", thirditem->valuestring);
        }
        // c45 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c45"))
        {
          sprintf(par->bin_file_c45, "%s", thirditem->valuestring);
        }
        // c46 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c46"))
        {
          sprintf(par->bin_file_c46, "%s", thirditem->valuestring);
        }
        // c55 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c55"))
        {
          sprintf(par->bin_file_c55, "%s", thirditem->valuestring);
        }
        // c56 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c56"))
        {
          sprintf(par->bin_file_c56, "%s", thirditem->valuestring);
        }
        // c66 file
        if (thirditem = cJSON_GetObjectItem(subitem, "c66"))
        {
          sprintf(par->bin_file_c66, "%s", thirditem->valuestring);
        }
        // need to add other model parameters

      } // find binfile
    } // if binfile

    if (subitem = cJSON_GetObjectItem(item, "equivalent_medium_method")) {
        sprintf(par->equivalent_medium_method, "%s", subitem->valuestring);
    }
  }

  par->is_export_media = 1;
  if (item = cJSON_GetObjectItem(root, "is_export_media")) {
     par->is_export_media = item->valueint;
  }
  if (item = cJSON_GetObjectItem(root, "media_export_dir")) {
      sprintf(par->media_export_dir,"%s",item->valuestring);
  }

  //
  //-- visco
  //

  par->visco_Qs_freq = 0.0;
  par->visco_itype = 0;
  if (item = cJSON_GetObjectItem(root, "visco_config")) {
    if (subitem = cJSON_GetObjectItem(item, "type")) {
        sprintf(par->visco_type, "%s", subitem->valuestring);
        if (strcmp(par->visco_type, "graves_Qs")==0) {
          par->visco_itype = CONST_VISCO_GRAVES_QS;
        } else {
          fprintf(stderr,"ERROR: visco_type is unknown\n");
          MPI_Abort(MPI_COMM_WORLD,9);
        }
    }
    if (subitem = cJSON_GetObjectItem(item, "Qs_freq")) {
        par->visco_Qs_freq = subitem->valuedouble;
    }
  }

  //-- output dir
  if (item = cJSON_GetObjectItem(root, "output_dir")) {
      sprintf(par->output_dir,"%s",item->valuestring);
  }

  //-- receiver
  if (item = cJSON_GetObjectItem(root, "in_station_file")) {
    sprintf(par->in_station_file, "%s", item->valuestring);
  }
  //--fault receiver
  if (item = cJSON_GetObjectItem(root, "fault_station_file")) {
    sprintf(par->fault_station_file, "%s", item->valuestring);
  }

  //-- receiver line
  if (item = cJSON_GetObjectItem(root, "receiver_line"))
  {
    par->number_of_receiver_line = cJSON_GetArraySize(item);
    par->receiver_line_index_start  = (int *)malloc(par->number_of_receiver_line*sizeof(int)*CONST_NDIM);
    par->receiver_line_index_incre  = (int *)malloc(par->number_of_receiver_line*sizeof(int)*CONST_NDIM);
    par->receiver_line_count  = (int *)malloc(par->number_of_receiver_line*sizeof(int));
    //par->receiver_line_time_interval  = (int *)malloc(par->number_of_receiver_line*sizeof(int));
    par->receiver_line_name = (char **)malloc(par->number_of_receiver_line*sizeof(char*));
    for (int n=0; n<par->number_of_receiver_line; n++) {
      par->receiver_line_name[n] = (char *)malloc(PAR_MAX_STRLEN*sizeof(char));
    }
    // each line
    for (int i=0; i < cJSON_GetArraySize(item) ; i++)
    {
      lineitem = cJSON_GetArrayItem(item, i);

      if (subitem = cJSON_GetObjectItem(lineitem, "name"))
      {
        sprintf(par->receiver_line_name[i],"%s",subitem->valuestring);
      }

      // need minus 1. C index start from 1.
      if (subitem = cJSON_GetObjectItem(lineitem, "grid_index_start"))
      {
        for (int j = 0; j < CONST_NDIM; j++) {
          par->receiver_line_index_start[i*CONST_NDIM+j] = cJSON_GetArrayItem(subitem, j)->valueint-1;
        }
      }

      if (subitem = cJSON_GetObjectItem(lineitem, "grid_index_incre"))
      {
        for (int j = 0; j < CONST_NDIM; j++) {
          par->receiver_line_index_incre[i*CONST_NDIM+j] = cJSON_GetArrayItem(subitem, j)->valueint;
        }
      }

      if (subitem = cJSON_GetObjectItem(lineitem, "grid_index_count"))
      {
         par->receiver_line_count[i] = subitem->valueint;
      }

      //if (subitem = cJSON_GetObjectItem(lineitem, "t_index_interval"))
      //{
      //   par->receiver_line_tinterval[i] = cJSON_GetArrayItem(subitem, j)->valueint;
      //}
    }
  }

  // slice
  if (item = cJSON_GetObjectItem(root, "slice"))
  {
    if (subitem = cJSON_GetObjectItem(item, "x_index"))
    {
      par->number_of_slice_x = cJSON_GetArraySize(subitem);
      par->slice_x_index  = (int *)malloc(par->number_of_slice_x*sizeof(int));
      // need minus 1. C index start from 1.
      for (int i=0; i < par->number_of_slice_x ; i++)
      {
        par->slice_x_index[i] = cJSON_GetArrayItem(subitem, i)->valueint-1;
      }
    }
    if (subitem = cJSON_GetObjectItem(item, "y_index"))
    {
      par->number_of_slice_y = cJSON_GetArraySize(subitem);
      par->slice_y_index  = (int *)malloc(par->number_of_slice_y*sizeof(int));
      // need minus 1. C index start from 1.
      for (int i=0; i < par->number_of_slice_y ; i++)
      {
        par->slice_y_index[i] = cJSON_GetArrayItem(subitem, i)->valueint-1;
      }
    }
    if (subitem = cJSON_GetObjectItem(item, "z_index"))
    {
      par->number_of_slice_z = cJSON_GetArraySize(subitem);
      // need minus 1. C index start from 1.
      par->slice_z_index  = (int *)malloc(par->number_of_slice_z*sizeof(int));
      for (int i=0; i < par->number_of_slice_z ; i++)
      {
        par->slice_z_index[i] = cJSON_GetArrayItem(subitem, i)->valueint-1;
      }
    }
  }

  // snapshot
  if (item = cJSON_GetObjectItem(root, "snapshot"))
  {
    par->number_of_snapshot = cJSON_GetArraySize(item);
    //fprintf(stdout,"size=%d, %d, %d\n", par->number_of_snapshot, sizeof(int), CONST_NDIM);
    //fflush(stdout);
    par->snapshot_index_start  = (int *)malloc(par->number_of_snapshot*sizeof(int)*CONST_NDIM);
    //if (par->snapshot_index_start == NULL) {
    //  fprintf(stdout,"eror\n");
    //  fflush(stdout);
    //}
    par->snapshot_index_count  = (int *)malloc(par->number_of_snapshot*sizeof(int)*CONST_NDIM);
    par->snapshot_index_incre = (int *)malloc(par->number_of_snapshot*sizeof(int)*CONST_NDIM);
    par->snapshot_time_start  = (int *)malloc(par->number_of_snapshot*sizeof(int));
    par->snapshot_time_incre = (int *)malloc(par->number_of_snapshot*sizeof(int));
    par->snapshot_save_velocity = (int *)malloc(par->number_of_snapshot*sizeof(int));
    par->snapshot_save_stress  = (int *)malloc(par->number_of_snapshot*sizeof(int));
    par->snapshot_save_strain = (int *)malloc(par->number_of_snapshot*sizeof(int));
    // name of snapshot
    par->snapshot_name = (char **)malloc(par->number_of_snapshot*sizeof(char*));
    for (int n=0; n<par->number_of_snapshot; n++) {
      par->snapshot_name[n] = (char *)malloc(PAR_MAX_STRLEN*sizeof(char));
    }

    // each snapshot
    for (int i=0; i < cJSON_GetArraySize(item) ; i++)
    {
      snapitem = cJSON_GetArrayItem(item, i);

      if (subitem = cJSON_GetObjectItem(snapitem, "name"))
      {
        sprintf(par->snapshot_name[i],"%s",subitem->valuestring);
      }

      // need minus 1. C index start from 1.
      if (subitem = cJSON_GetObjectItem(snapitem, "grid_index_start"))
      {
        for (int j = 0; j < CONST_NDIM; j++) {
          par->snapshot_index_start[i*CONST_NDIM+j] = cJSON_GetArrayItem(subitem, j)->valueint-1;
        }
      }

      if (subitem = cJSON_GetObjectItem(snapitem, "grid_index_count"))
      {
        for (int j = 0; j < CONST_NDIM; j++) {
          par->snapshot_index_count[i*CONST_NDIM+j] = cJSON_GetArrayItem(subitem, j)->valueint;
        }
      }

      if (subitem = cJSON_GetObjectItem(snapitem, "grid_index_incre"))
      {
        for (int j = 0; j < CONST_NDIM; j++) {
          par->snapshot_index_incre[i*CONST_NDIM+j] = cJSON_GetArrayItem(subitem, j)->valueint;
        }
      }

      if (subitem = cJSON_GetObjectItem(snapitem, "time_index_start")) {
        par->snapshot_time_start[i]  = subitem->valueint;
      }
      if (subitem = cJSON_GetObjectItem(snapitem, "time_index_incre")) {
        par->snapshot_time_incre[i]  = subitem->valueint;
      }
      if (subitem = cJSON_GetObjectItem(snapitem, "save_velocity")) {
        par->snapshot_save_velocity[i]  = subitem->valueint;
      }
      if (subitem = cJSON_GetObjectItem(snapitem, "save_stress")) {
        par->snapshot_save_stress[i]  = subitem->valueint;
      }
      if (subitem = cJSON_GetObjectItem(snapitem, "save_strain")) {
        par->snapshot_save_strain[i] = subitem->valueint;
      }
    }
  }

  //-- misc
  if (item = cJSON_GetObjectItem(root, "check_nan_every_number_of_steps")) {
      par->qc_check_nan_number_of_step = item->valueint;
  }
  if (item = cJSON_GetObjectItem(root, "output_all")) {
      par->output_all = item->valueint;
  }

  //if (item = cJSON_GetObjectItem(root, "grid_name")) {
  //    sprintf(par->grid_name,"%s",item->valuestring);
  //}

  cJSON_Delete(root);

  // set values to default ones if no input

  //-- check conditions

  // not implement dt estimation for general anisotropic media yet
  if (par->media_itype == CONST_MEDIUM_ELASTIC_ANISO &&
      par->size_of_time_step < 0.0)
  {
    fprintf(stderr, "ERROR: have not implemented dt estimation for anisotropic media\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  return ierr;
}

/*
 * funcs to read cfspml para from json str
 */
int 
par_read_json_cfspml(cJSON *item,
      int *nlay, float *amax, float *bmax, float *vel)
{
  cJSON *subitem;

  if (subitem = cJSON_GetObjectItem(item, "number_of_layers"))
  {
    *nlay = subitem->valueint;
  }
  if (subitem = cJSON_GetObjectItem(item, "alpha_max"))
  {
    *amax = subitem->valuedouble;
  }
  if (subitem = cJSON_GetObjectItem(item, "beta_max"))
  {
    *bmax = subitem->valuedouble;
  }
  if (subitem = cJSON_GetObjectItem(item, "ref_vel"))
  {
    *vel = subitem->valuedouble;
  }

  return 0;
}

/*
 * funcs to read ablexp para from json str
 */
int 
par_read_json_ablexp(cJSON *item, int *nlay, float *vel)
{
  cJSON *subitem;

  if (subitem = cJSON_GetObjectItem(item, "number_of_layers"))
  {
    *nlay = subitem->valueint;
  }
  if (subitem = cJSON_GetObjectItem(item, "ref_vel"))
  {
    *vel = subitem->valuedouble;
  }

  return 0;
}


int
par_print(par_t *par)
{    
  int ierr = 0;

  fprintf(stdout, "-------------------------------------------------------\n");
  //fprintf(stdout, "--> ESTIMATE MEMORY information.\n");
  //fprintf(stdout, "-------------------------------------------------------\n");
  //fprintf(stdout, "total memory size Byte: %20.5f  B\n", PSV->total_memory_size_Byte);
  //fprintf(stdout, "total memory size KB  : %20.5f KB\n", PSV->total_memory_size_KB  );
  //fprintf(stdout, "total memory size MB  : %20.5f MB\n", PSV->total_memory_size_MB  );
  //fprintf(stdout, "total memory size GB  : %20.5f GB\n", PSV->total_memory_size_GB  );
  //fprintf(stdout, "\n");
  //fprintf(stdout, "-------------------------------------------------------\n");
  //fprintf(stdout, "--> FOLDER AND FILE information.\n");
  //fprintf(stdout, "-------------------------------------------------------\n");
  //fprintf(stdout, "   OutFolderName: %s\n", OutFolderName);
  //fprintf(stdout, "       EventName: %s\n", OutPrefix);
  //fprintf(stdout, "     LogFilename: %s\n", LogFilename);
  //fprintf(stdout, " StationFilename: %s\n", StationFilename);
  //fprintf(stdout, "  SourceFilename: %s\n", SourceFilename);
  //fprintf(stdout, "   MediaFilename: %s\n", MediaFilename);
  //fprintf(stdout, "\n");

  fprintf(stdout, "-------------------------------------------------------\n");
  fprintf(stdout, "--> MPI information:\n");
  fprintf(stdout, "-------------------------------------------------------\n");
  fprintf(stdout, " number_of_mpiprocs_x = %-10d\n", par->number_of_mpiprocs_x);
  fprintf(stdout, " number_of_mpiprocs_y = %-10d\n", par->number_of_mpiprocs_y);
  fprintf(stdout, " number_of_mpiprocs_z = %-10d\n", par->number_of_mpiprocs_z);

  fprintf(stdout, "-------------------------------------------------------\n");
  fprintf(stdout, "--> Time Integration information:\n");
  fprintf(stdout, "-------------------------------------------------------\n");
  fprintf(stdout, " size_of_time_step = %10.4e\n", par->size_of_time_step);
  fprintf(stdout, " number_of_time_steps = %-10d\n", par->number_of_time_steps);

  fprintf(stdout, "-------------------------------------------------------\n");
  fprintf(stdout, "--> boundary layer information.\n");
  fprintf(stdout, "-------------------------------------------------------\n");
  fprintf(stdout, " boundary_condition  = %10s%10s%10s%10s%10s%10s\n", 
          par->boundary_type_name[0],
          par->boundary_type_name[1],
          par->boundary_type_name[2],
          par->boundary_type_name[3],
          par->boundary_type_name[4],
          par->boundary_type_name[5]);
  fprintf(stdout, " cfspml:\n");
  for (int idim=0; idim < CONST_NDIM; idim++)
  {
    for (int iside=0; iside < 2; iside++)
    {
      fprintf(stdout, "   idim=%d,iside=%d: nlay = %d,"
              " vel=%f, alpha_max=%f, beta_max=%f\n",
              idim, iside, 
              par->abs_num_of_layers[idim][iside],
              par->cfspml_velocity[idim][iside],
              par->cfspml_alpha_max[idim][iside],
              par->cfspml_beta_max[idim][iside]
              );
    }
  }
  fprintf(stdout, "\n");

  fprintf(stdout, "-------------------------------------------------------\n");
  fprintf(stdout, "--> GRID information:\n");
  fprintf(stdout, "-------------------------------------------------------\n");
  fprintf(stdout, " grid_export_dir = %s\n", par->grid_export_dir);
  fprintf(stdout, " number_of_total_grid_points_x = %-10d\n", par->number_of_total_grid_points_x);
  fprintf(stdout, " number_of_total_grid_points_y = %-10d\n", par->number_of_total_grid_points_y);
  fprintf(stdout, " number_of_total_grid_points_z = %-10d\n", par->number_of_total_grid_points_z);

  fprintf(stdout, " metric_method_itype = %d\n", par->metric_method_itype);

  fprintf(stdout, "-------------------------------------------------------\n");
  fprintf(stdout, "--> media info.\n");
  fprintf(stdout, "-------------------------------------------------------\n");
  fprintf(stdout, " media_type = %s\n", par->media_type);
  fprintf(stdout, " media_export_dir = %s\n", par->media_export_dir);

  if (par->media_input_itype == PAR_MEDIA_CODE) {
    fprintf(stdout, "\n --> uniform media by code\n");
  } else if (par->media_input_itype == PAR_MEDIA_IMPORT) {
    fprintf(stdout, "\n --> import from dir = %s\n", par->media_import_dir);
  } else if (par->media_input_itype == PAR_MEDIA_3LAY) {
    fprintf(stdout, "\n --> input layer file = %s\n", par->media_input_file);
  } else if (par->media_input_itype == PAR_MEDIA_3GRD) {
    fprintf(stdout, "\n --> input grid file = %s\n", par->media_input_file);
  }

  //fprintf(stdout, " media_input_type = %s\n", par->media_input_type);
  //fprintf(stdout, " media_input_itype = %d\n", par->media_input_itype);

  if (par->visco_itype == CONST_VISCO_GRAVES_QS) {
    fprintf(stdout, "-------------------------------------------------------\n");
    fprintf(stdout, "--> visco info.\n");
    fprintf(stdout, "-------------------------------------------------------\n");
    fprintf(stdout, " visco_type = %s\n", par->visco_type);
    fprintf(stdout, " visco_Qs_freq = %f\n", par->visco_Qs_freq);
  } else {
    fprintf(stdout, "--> no visco\n");
  }


  fprintf(stdout, "\n");
  fprintf(stdout, "-------------------------------------------------------\n");
  fprintf(stdout, "--> output information.\n");
  fprintf(stdout, "-------------------------------------------------------\n");
  fprintf(stdout, "--> output_dir = %s\n", par->output_dir);

  fprintf(stdout, "--> station list file:\n");
  fprintf(stdout, " in_station_file = %s\n", par->in_station_file);

  fprintf(stdout, "--> recivers lines:\n");
  fprintf(stdout, "number_of_receiver_line=%d\n", par->number_of_receiver_line);
  if (par->number_of_receiver_line > 0)
  {
    fprintf(stdout, "#  name  i0  j0  k0   di  dj   dk  count\n");
    for(int n=0; n<par->number_of_receiver_line; n++)
    {
       fprintf(stdout, "%6d %s %6d %6d %6d %6d %6d %6d %6d\n",
           n,
           par->receiver_line_name[n],
           par->receiver_line_index_start[n*3+0],
           par->receiver_line_index_start[n*3+1],
           par->receiver_line_index_start[n*3+2],
           par->receiver_line_index_incre[n*3+0],
           par->receiver_line_index_incre[n*3+1],
           par->receiver_line_index_incre[n*3+2],
           par->receiver_line_count[n]);
    }
  }

  fprintf(stdout, "--> slice:\n");
  fprintf(stdout, "number_of_slice_x=%d: ", par->number_of_slice_x);
  for(int n=0; n<par->number_of_slice_x; n++)
  {
     fprintf(stdout, "%6d", par->slice_x_index[n]);
  }
  fprintf(stdout, "\nnumber_of_slice_y=%d: ", par->number_of_slice_y);
  for(int n=0; n<par->number_of_slice_y; n++)
  {
     fprintf(stdout, "%6d", par->slice_y_index[n]);
  }
  fprintf(stdout, "\nnumber_of_slice_z=%d: ", par->number_of_slice_z);
  for(int n=0; n<par->number_of_slice_z; n++)
  {
     fprintf(stdout, "%6d", par->slice_z_index[n]);
  }
  fprintf(stdout, "\n");

  fprintf(stdout, "--> snapshot information.\n");
  fprintf(stdout, "number_of_snapshot=%d\n", par->number_of_snapshot);
  if (par->number_of_snapshot > 0)
  {
      fprintf(stdout, "#  name  i0  j0  k0  ni  nj  nk  di  dj   dk  it0  nt  dit\n");
      for(int n=0; n<par->number_of_snapshot; n++)
      {
         fprintf(stdout, "%6d %s %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d\n",
             n,
             par->snapshot_name[n],
             par->snapshot_index_start[n*3+0],
             par->snapshot_index_start[n*3+1],
             par->snapshot_index_start[n*3+2],
             par->snapshot_index_count[n*3+0],
             par->snapshot_index_count[n*3+1],
             par->snapshot_index_count[n*3+2],
             par->snapshot_index_incre[n*3+0],
             par->snapshot_index_incre[n*3+1],
             par->snapshot_index_incre[n*3+2],
             par->snapshot_time_start[n],
             par->snapshot_time_incre[n]);
      }
  }

  fprintf(stdout, "--> qc parameters:\n");
  fprintf(stdout, "check_nan_every_number_of_steps=%d\n", par->qc_check_nan_number_of_step);
  fprintf(stdout, "output_all=%d\n", par->output_all);

  return ierr;
}