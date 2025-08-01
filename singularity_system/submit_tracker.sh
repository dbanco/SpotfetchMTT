#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=04:00:00
#$ -l h_vmem=4G

# Define a writable directory on the host

# Avoid matplotlib cache issues
export MPLCONFIGDIR=/tmp/mpl_cache
mkdir -p $MPLCONFIGDIR

export HOME=/tmp

# Launch container with binding
apptainer exec \
  --bind "${YAML_DIR}":/param_files \
  --bind "${DATA_DIR}":/data_dir \
  --bind "${SING_DIR}/tracker_states":/tracker_states \
  --bind "${SING_DIR}/app/tracker_dex.py":/app/tracker_dex.py \
  --bind "${MTT_DIR}/utilities.py":/app/utilities.py \
  --bind "${CONFIG_PATH}":/app/mtt_config.yaml \
  sif/tracker_dex.sif \
  python /app/tracker_dex.py --config /app/mtt_config.yaml
