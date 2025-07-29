#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=04:00:00
#$ -l h_vmem=4G

# Define a writable directory on the host
mkdir -p $TRACKER_STATES_HOST_DIR

# Avoid matplotlib cache issues
export MPLCONFIGDIR=/tmp/mpl_cache
mkdir -p $MPLCONFIGDIR

# Launch container with binding
apptainer run \
  --bind /nfs/chess/user/dbanco/SpotfetchMTT/tracker-deploy-chess/tracker_states:/tracker_states \
  --bind /nfs/chess/user/dbanco/SpotfetchMTT/tracker-deploy-chess/region_files:/region_files \
  --bind /nfs/chess/user/dbanco/SpotfetchMTT/tracker-deploy-chess/app/tracker_dex.py:/app/tracker_dex.py \
  sif/tracker_dex.sif