#!/bin/bash
#$ -N tracker_job
#$ -cwd
#$ -V
#$ -l h_rt=01:00:00
#$ -l mem_free=4G
#$ -pe smp 1

# Load modules if needed
# module load apptainer

# Define region_id (or pass it from command line)
REGION_ID=$1

# Run tracker container
apptainer exec \
  --bind /nfs/chess/user/dbanco:/mnt \
  sif/tracker.sif \
  python /app/tracker.py --region_id "$REGION_ID"

Make sure tracker.py accepts a --region_id or similar argument.