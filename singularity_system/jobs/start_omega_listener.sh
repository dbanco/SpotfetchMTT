#!/bin/bash
#$ -N omega_listener
#$ -cwd
#$ -j y

apptainer exec tracker.sif python3 /app/omega_listener.py
