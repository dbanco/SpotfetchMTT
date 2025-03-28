#!/bin/bash
#$ -N job_dispatcher
#$ -cwd
#$ -j y

apptainer exec tracker.sif python3 /app/job_dispatcher.py
