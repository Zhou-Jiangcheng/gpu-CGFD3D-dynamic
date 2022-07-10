#!/bin/bash

#set -x
set -e

date

#-- system related dir
MPIDIR=/data3/lihl/software/openmpi-gnu-4.1.2

#-- program related dir
EXEC_WAVE=`pwd`/../main_curv_col_el_3d
echo "EXEC_WAVE=$EXEC_WAVE"

#-- input dir
INPUTDIR=`pwd`

#-- output and conf
PROJDIR=`pwd`/../project
PAR_FILE=${PROJDIR}/helanshan.json
GRID_DIR=${PROJDIR}/helanshan
MEDIA_DIR=${PROJDIR}/helanshan
SOURCE_DIR=${PROJDIR}/helanshan
OUTPUT_DIR=${PROJDIR}/helanshan

#-- create dir
mkdir -p $PROJDIR
mkdir -p $OUTPUT_DIR
mkdir -p $GRID_DIR
mkdir -p $MEDIA_DIR

#----------------------------------------------------------------------
#-- create main conf
#----------------------------------------------------------------------
cat << ieof > $PAR_FILE
{
  "number_of_total_grid_points_x" : 1121,
  "number_of_total_grid_points_y" : 1125,
  "number_of_total_grid_points_z" : 450,

  "number_of_mpiprocs_x" : 2,
  "number_of_mpiprocs_y" : 4,

  "#size_of_time_step" : 0.01,
  "#number_of_time_steps" : 7000,
  "time_window_length" : 50,
  "check_stability" : 1,

  "boundary_x_left" : {
      "cfspml" : {
          "number_of_layers" : 10,
          "alpha_max" : 3.14,
          "beta_max" : 2.0,
          "ref_vel"  : 7000.0
          }
      },
  "boundary_x_right" : {
      "cfspml" : {
          "number_of_layers" : 10,
          "alpha_max" : 3.14,
          "beta_max" : 2.0,
          "ref_vel"  : 7000.0
          }
      },
  "boundary_y_front" : {
      "cfspml" : {
          "number_of_layers" : 10,
          "alpha_max" : 3.14,
          "beta_max" : 2.0,
          "ref_vel"  : 7000.0
          }
      },
  "boundary_y_back" : {
      "cfspml" : {
          "number_of_layers" : 10,
          "alpha_max" : 3.14,
          "beta_max" : 2.0,
          "ref_vel"  : 7000.0
          }
      },
  "boundary_z_bottom" : {
      "cfspml" : {
          "number_of_layers" : 10,
          "alpha_max" : 3.14,
          "beta_max" : 2.0,
          "ref_vel"  : 7000.0
          }
      },
  "boundary_z_top" : {
      "free" : "timg"
      },

  "grid_generation_method" : {
      "#import" : "$GRID_DIR",
      "#cartesian" : {
        "origin"  : [0.0, 0.0, -5900.0 ],
        "inteval" : [ 100.0, 100.0, 100.0 ]
      },
      "layer_interp" : {
        "in_grid_layer_file" : "$INPUTDIR/prep_grid/helanshan_fault_area.gdlay",
        "refine_factor" : [ 1, 1, 1 ],
        "horizontal_start_index" : [ 3, 3 ],
        "vertical_last_to_top" : 0
      }
  },
  "is_export_grid" : 1,
  "grid_export_dir"   : "$GRID_DIR",

  "metric_calculation_method" : {
      "#import" : "$GRID_DIR",
      "calculate" : 1
  },
  "is_export_metric" : 0,

  "medium" : {
      "type" : "elastic_iso",
      "#type" : "elastic_vti",
      "#code" : "func_name_here",
      "#import" : "$MEDIA_DIR",
      "infile_layer" : "$INPUTDIR/prep_medium/helanshan.md3lay",
      "#infile_grid" : "$INPUTDIR/prep_medium/topolay_el_iso.md3grd",
      "equivalent_medium_method" : "loc"
  },
  "is_export_media" : 1,
  "media_export_dir"  : "$MEDIA_DIR",

  "#visco_config" : {
      "type" : "graves_Qs",
      "Qs_freq" : 1.0
  },

  "in_source_file" : "$INPUTDIR/prep_source/helanshan_source_left.valsrc",
  "is_export_source" : 1,
  "source_export_dir"  : "$SOURCE_DIR",

  "output_dir" : "$OUTPUT_DIR",

  "in_station_file" : "$INPUTDIR/station.list",

  "#receiver_line" : [
    {
      "name" : "line_x_1",
      "grid_index_start"    : [  50, 50, 59 ],
      "grid_index_incre"    : [  1,  0,  0 ],
      "grid_index_count"    : 2
    },
    {
      "name" : "line_y_1",
      "grid_index_start"    : [ 80, 49, 59 ],
      "grid_index_incre"    : [  0,  1,  0 ],
      "grid_index_count"    : 2
    } 
  ],

  "#slice" : {
      "x_index" : [ 190 ],
      "y_index" : [ 120 ],
      "z_index" : [ 59 ]
  },

  "snapshot" : [
    {
      "name" : "volume_vel",
      "grid_index_start" : [ 0, 0, 449 ],
      "grid_index_count" : [ 1121,1125, 1 ],
      "grid_index_incre" : [  1, 1, 1 ],
      "time_index_start" : 0,
      "time_index_incre" : 1,
      "save_velocity" : 1,
      "save_stress"   : 0,
      "save_strain"   : 0
    }
  ],

  "check_nan_every_nummber_of_steps" : 0,
  "output_all" : 0 
}
ieof

echo "+ created $PAR_FILE"

#-------------------------------------------------------------------------------
#-- Performce simulation
#-------------------------------------------------------------------------------
#

#-- get np
NUMPROCS_X=`grep number_of_mpiprocs_x ${PAR_FILE} | sed 's/:/ /g' | sed 's/,/ /g' | awk '{print $2}'`
NUMPROCS_Y=`grep number_of_mpiprocs_y ${PAR_FILE} | sed 's/:/ /g' | sed 's/,/ /g' | awk '{print $2}'`
NUMPROCS=$(( NUMPROCS_X*NUMPROCS_Y ))
echo $NUMPROCS_X $NUMPROCS_Y $NUMPROCS

#-- gen run script
cat << ieof > ${PROJDIR}/cgfd_sim.sh
#!/bin/bash

set -e
printf "\nUse $NUMPROCS CPUs on following nodes:\n"

printf "\nStart simualtion ...\n";
time $MPIDIR/bin/mpiexec -np $NUMPROCS $EXEC_WAVE $PAR_FILE 100 2>&1 |tee log
if [ $? -ne 0 ]; then
    printf "\nSimulation fail! stop!\n"
    exit 1
fi

ieof

#-------------------------------------------------------------------------------
#-- start run
#-------------------------------------------------------------------------------

chmod 755 ${PROJDIR}/cgfd_sim.sh
${PROJDIR}/cgfd_sim.sh
if [ $? -ne 0 ]; then
    printf "\nSimulation fail! stop!\n"
    exit 1
fi

date

# vim:ft=conf:ts=4:sw=4:nu:et:ai:
