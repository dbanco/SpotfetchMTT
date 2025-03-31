#!/bin/bash
#$ -N postgres_server
#$ -cwd
#$ -j y
#$ -o postgres_log.txt
#$ -l h_rt=168:00:00
#$ -l mem_free=2G
#$ -V

export POSTGRES_DIR=/nfs/chess/user/dbanco/SpotfetchMTT/tracker-deploy-chess/postgres_data
apptainer run   --bind $POSTGRES_DIR:/var/lib/postgresql/data   sif/postgres.sif
