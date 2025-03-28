#!/bin/bash
#$ -N redis
#$ -cwd
#$ -j y

apptainer exec --env LC_ALL=C sif/redis.sif redis-server