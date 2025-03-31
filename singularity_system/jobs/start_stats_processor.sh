#!/bin/bash
#$ -N stats
#$ -cwd
#$ -j y

apptainer exec tracker.sif python3 /app/streamlit_app.py
