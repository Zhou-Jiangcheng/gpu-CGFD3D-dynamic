#!/bin/bash

#set -x
set -e

date

#-- system related dir
MPIDIR=/data/apps/openmpi/4.1.5-cuda-aware

#-- program related dir
EXEC_WAVE=`pwd`/../../main_curv_col_el_3d
echo "EXEC_WAVE=$EXEC_WAVE"

#-- input dir
INPUTDIR=`pwd`

#-- output and conf
PROJDIR=`pwd`/../../project
PAR_FILE=${PROJDIR}/params.json
GRID_DIR=${PROJDIR}/output
MEDIA_DIR=${PROJDIR}/output
OUTPUT_DIR=${PROJDIR}/output

rm -rf ${PROJDIR}

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
  "number_of_total_grid_points_x" : 200,
  "number_of_total_grid_points_y" : 400,
  "number_of_total_grid_points_z" : 200,

  "number_of_mpiprocs_y" : 2,
  "number_of_mpiprocs_z" : 2,

  "dynamic_method" : 1,
  "fault_grid" : [51,350,51,200],

  "size_of_time_step" : 0.005,
  "number_of_time_steps" : 3000,
  "#time_window_length" : 4,
  "check_stability" : 1,
  "io_time_skip" : 2,

  "boundary_x_left" : {
      "ablexp" : {
          "number_of_layers" : 50,
          "ref_vel"  : 7000.0
          }
      },
  "boundary_x_right" : {
      "ablexp" : {
          "number_of_layers" : 50,
          "ref_vel"  : 7000.0
          }
      },
  "boundary_y_front" : {
      "ablexp" : {
          "number_of_layers" : 50,
          "ref_vel"  : 7000.0
          }
      },
  "boundary_y_back" : {
      "ablexp" : {
          "number_of_layers" : 50,
          "ref_vel"  : 7000.0
          }
      },
  "boundary_z_bottom" : {
      "ablexp" : {
          "number_of_layers" : 50,
          "ref_vel"  : 7000.0
          }
      },
  "boundary_z_top" : {
      "free" : "timg"
      },

  "grid_generation_method" : {
      "fault_plane" : {
        "fault_geometry_file" : "${INPUTDIR}/prep_fault/fault_coord.nc",
        "fault_init_stress_file" : "${INPUTDIR}/prep_fault/init_stress.nc",
        "fault_inteval" : 100.0
      },
      "#grid_with_fault" : {
        "grid_file" : "${INPUTDIR}/prep_fault/fault_coord.nc",
        "fault_init_stress_file" : "${INPUTDIR}/prep_fault/init_stress.nc",
        "fault_i_gobal_index" : 100.0
      }
  },
  "is_export_grid" : 1,
  "grid_export_dir"   : "$GRID_DIR",

  "metric_calculation_method" : {
      "#import" : "$GRID_DIR",
      "calculate" : 1
  },
  "is_export_metric" : 1,

  "medium" : {
      "type" : "elastic_iso",
      "#input_way" : "infile_layer",
      "#input_way" : "binfile",
      "input_way" : "code",
      "#binfile" : {
        "size"    : [1101, 1447, 1252],
        "spacing" : [-10, 10, 10],
        "origin"  : [0.0,0.0,0.0],
        "dim1" : "z",
        "dim2" : "x",
        "dim3" : "y",
        "Vp" : "$INPUTDIR/prep_medium/seam_Vp.bin",
        "Vs" : "$INPUTDIR/prep_medium/seam_Vs.bin",
        "rho" : "$INPUTDIR/prep_medium/seam_rho.bin"
      },
      "#import" : "$MEDIA_DIR",
      "code" : "",
      "#infile_layer" : "$INPUTDIR/prep_medium/basin_el_iso.md3lay",
      "#infile_grid" : "$INPUTDIR/prep_medium/topolay_el_iso.md3grd",
      "#equivalent_medium_method" : "loc",
      "#equivalent_medium_method" : "har"
  },

  "is_export_media" : 1,
  "media_export_dir"  : "$MEDIA_DIR",

  "#visco_config" : {
      "type" : "graves_Qs",
      "Qs_freq" : 1.0
  },

  "output_dir" : "$OUTPUT_DIR",

  "in_station_file" : "$INPUTDIR/prep_station/station.list",

  "#receiver_line" : [
    {
      "name" : "line_x_1",
      "grid_index_start"    : [  0, 49, 59 ],
      "grid_index_incre"    : [  1,  0,  0 ],
      "grid_index_count"    : 20
    },
    {
      "name" : "line_y_1",
      "grid_index_start"    : [ 19, 49, 59 ],
      "grid_index_incre"    : [  0,  1,  0 ],
      "grid_index_count"    : 20
    } 
  ],

  "#slice" : {
      "x_index" : [ 51 ],
      "y_index" : [ 120 ],
      "z_index" : [ 199 ]
  },

  "#snapshot" : [
    {
      "name" : "volume_vel",
      "grid_index_start" : [ 0, 0, 199 ],
      "grid_index_count" : [ 100,400, 1 ],
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

#-- get np
NUMPROCS_Y=`grep number_of_mpiprocs_y ${PAR_FILE} | sed 's/:/ /g' | sed 's/,/ /g' | awk '{print $2}'`
NUMPROCS_Z=`grep number_of_mpiprocs_z ${PAR_FILE} | sed 's/:/ /g' | sed 's/,/ /g' | awk '{print $2}'`
NUMPROCS=$(( NUMPROCS_Y*NUMPROCS_Z ))
echo $NUMPROCS_Y $NUMPROCS_Z $NUMPROCS

#-- gen run script
cat << ieof > ${PROJDIR}/cgfd_sim.sh
#!/bin/bash

set -e
printf "\nUse $NUMPROCS CPUs on following nodes:\n"

printf "\nStart simualtion ...\n";
time $MPIDIR/bin/mpiexec -np $NUMPROCS $EXEC_WAVE $PAR_FILE 100 0 2>&1 |tee log1
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
