#!/bin/bash
set -e
set -u

# 0. Read in system configuration from YAML file
CONFIG_PATH="/nfs/chess/user/dbanco/SpotfetchMTT/singularity_system/mtt_config.yaml"
eval $(python3 load_system_config.py "$CONFIG_PATH")

mkdir -p logs
mkdir -p "${POSTGRES_DIR}"

# 1.Launch Redis Locally
echo "Starting Redis locally..."
tmux has-session -t redis_session 2>/dev/null || \
tmux new-session -d -s redis_session 'redis-server --bind 0.0.0.0 --port 6379'

# 2. Launch PostgreSQL
./init_database.sh

# To check if they are running
ps aux | grep redis
ps aux | grep postgres

# 4. Launch N tracker jobs
for i in $(seq 1 $NUM_TRACKERS); do
    qsub -N tracker-$i -o logs/tracker-$i.out -e logs/tracker-$i.err \
      -v REDIS_HOST="$REDIS_HOST",POSTGRES_HOST="$POSTGRES_HOST",SING_DIR="$SING_DIR" \
      submit_tracker.sh
    sleep 1
done
echo "Submitted $NUM_TRACKERS tracker jobs."                
                    
# 5. Data listener
apptainer exec  --bind "${YAML_DIR}":/param_files \
                --bind "${DATA_DIR}":/data_dir \
                --bind "${APP_DIR}/dexela_listener.py":/app/dexela_listener.py \
                --bind "${MTT_DIR}/utilities.py":/app/utilities.py \
                --bind "${CONFIG_PATH}":/app/mtt_config.yaml \
                "${SIF_DIR}/dexela_listener.sif" \
                python /app/dexela_listener.py --config /app/mtt_config.yaml \
                > logs/listener.log 2>&1 &
echo $! > listener.pid

# 6. Stats processor
#apptainer run \
#  --bind "${APP_DIR}/stats_processor.py":/app/stats_processor.py \
#  "${SIF_DIR}"/stats_processor.sif \
#  python /app/stats_processor.py

