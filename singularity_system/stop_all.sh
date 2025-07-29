#!/bin/bash

echo "Stopping all SpotfetchMTT jobs..."

# Define job name prefixes
JOB_PREFIXES=("redis" "postgres" "omega_listener" "job_dispatcher" "tracker")

# Find and delete all matching jobs
for prefix in "${JOB_PREFIXES[@]}"; do
    JOB_IDS=$(qstat | grep "$prefix" | awk '{print $1}')
    for id in $JOB_IDS; do
        echo "Deleting job $id ($prefix)"
        qdel $id
    done
done

echo "All jobs stopped."
