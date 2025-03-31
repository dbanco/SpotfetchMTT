#!/bin/bash

export REDIS_HOST="lnx7108.classe.cornell.edu"
export REDIS_PORT=6379
export POSTGRESS_HOST="lnx7108.classe.cornell.edu"

# RUN THIS SCRIPT IN APP_DIR
SIF_DIR="./sif"
SPOTFETCH_DIR="/nfs/chess/user/dbanco/SpotfetchMTT"
APP_DIR="/nfs/chess/user/dbanco/SpotfetchMTT/tracker-deploy-chess"
POSTGRES_DIR="${APP_DIR}/postgres_data"
REGION_DIR="${APP_DIR}/region_files"
TRACKER_STATE_DIR="${APP_DIR}/tracker_states"
DETECT_YAML_DIR="/nfs/chess/aux/cycles/2025-1/id3a/shanks-3731-d/reduced_data/parameter_files"

########## --- Launch Redis Locally ---
#echo "Starting Redis locally..."
#redis-server --bind 0.0.0.0 --port 6379 &
#REDIS_PID=$!
#sleep 2

########### --- Launch PostgreSQL Locally ---
#echo "Starting PostgreSQL locally..."
#apptainer run \
#  --bind "$POSTGRES_DIR:/var/lib/postgresql/data" \
#  "$SIF_DIR/postgres.sif" &
#POSTGRES_PID=$!
#sleep 5

# To check if they are running do
# ps aux | grep redis
# ps aux | grep postgres

# 3. Submit Region Listener
qsub -N region_listener -o logs/region_listener.out -e logs/region_listener.err -b y -cwd \
    -v REDIS_HOST=$REDIS_HOST \
    REDIS_PORT=$REDIS_PORT \
    apptainer exec   --bind region_files:/region_files \
                     --bind "${SPOTFETCH_DIR}/tracker-deploy-chess/app/region_listener.py":/app/region_listener.py \
                     --env REDIS_HOST=localhost \
                       sif/region_listener.sif   python /app/region_listener.py
sleep 10

# 4. Launch N tracker jobs
NUM_TRACKERS=5
for i in $(seq 1 $NUM_TRACKERS); do
    qsub -N tracker-$i -o logs/tracker-$i.out -e logs/tracker-$i.err \
    -v REDIS_HOST="$REDIS_HOST",POSTGRES_HOST="$POSTGRES_HOST" submit_tracker.sh
    sleep 1
done
echo "Submitted $NUM_TRACKERS tracker jobs."
sleep 5

# 5. Submit Dexela Listener
#SPOTFETCH_DIR="/nfs/chess/user/dbanco/SpotfetchMTT"
#DETECT_YAML_DIR="/nfs/chess/aux/cycles/2025-1/id3a/shanks-3731-d/reduced_data/parameter_files"
#DEX_DATA_DIR="/nfs/chess/raw/2025-1/id3a/shanks-3731-d/ti-2-test"
#qsub -N dexela_listener -o logs/dexela_listener.out -e logs/dexela_listener.err -b y -cwd \
#    -v REDIS_HOST=$REDIS_HOST \
#    REDIS_PORT=$REDIS_PORT \
#    apptainer run   --bind "${DETECT_YAML_DIR}":/param_files \
#                    --bind "${DEX_DATA_DIR}":/dex_data \
#                    --bind "${SPOTFETCH_DIR}/tracker-deploy-chess/region_files":/region_files \
#                    --bind "${SPOTFETCH_DIR}/tracker-deploy-chess/app/dexela_listener.py":/app/dexela_listener.py \
#                    --bind "${SPOTFETCH_DIR}/utilities.py":/app/utilities.py \
#                    sif/dexela_listener.sif python /app/dexela_listener.py
                    
                    
# To run locally:
SPOTFETCH_DIR="/nfs/chess/user/dbanco/SpotfetchMTT"
DETECT_YAML_DIR="/nfs/chess/aux/cycles/2025-1/id3a/shanks-3731-d/reduced_data/parameter_files"
DEX_DATA_DIR="/nfs/chess/raw/2025-1/id3a/shanks-3731-d/ti-2-test"
apptainer run   --bind "${DETECT_YAML_DIR}":/param_files \
                --bind "${DEX_DATA_DIR}":/dex_data \
                --bind "${SPOTFETCH_DIR}/tracker-deploy-chess/region_files":/region_files \
                --bind "${SPOTFETCH_DIR}/tracker-deploy-chess/app/dexela_listener.py":/app/dexela_listener.py \
                --bind "${SPOTFETCH_DIR}/utilities.py":/app/utilities.py \
                sif/dexela_listener.sif python /app/dexela_listener.py