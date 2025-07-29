#!/bin/bash
set -e
set -u

export REDIS_HOST="lnx202.classe.cornell.edu"
export REDIS_PORT=6379
export POSTGRESS_HOST="lnx202.classe.cornell.edu"
USER="dbanco"
NUM_TRACKERS=5

SPOTFETCH_DIR="/nfs/chess/user/${USER}/SpotfetchMTT"
SYS_DIR="${SPOTFETCH_DIR}/singularity_system"
APP_DIR="${SYS_DIR}/app"
SIF_DIR="${SYS_DIR}/sif"
POSTGRES_DIR="${SYS_DIR}/postgres_data2"
REGION_DIR="${SYS_DIR}/region_files"
TRACKER_STATE_DIR="${SYS_DIR}/tracker_states"
DETECT_YAML_DIR="/nfs/chess/aux/cycles/2025-1/id3a/shanks-3731-d/reduced_data/parameter_files"
DEX_DATA_DIR="/nfs/chess/raw/2025-1/id3a/shanks-3731-d/ti-2-test"

mkdir -p logs
mkdir "${POSTGRES_DIR}"

# 1.Launch Redis Locally
echo "Starting Redis locally..."
tmux has-session -t redis_session 2>/dev/null || \
tmux new-session -d -s redis_session 'redis-server --bind 0.0.0.0 --port 6379'

# 2. Launch PostgreSQL
if [ ! -f "$POSTGRES_DIR/PG_VERSION" ]; then
  echo "Initializing PostgreSQL database cluster at $POSTGRES_DIR..."
  initdb -D "$POSTGRES_DIR"
fi
pg_ctl -D "$POSTGRES_DIR" -l "$POSTGRES_DIR/logfile" -o "-k /tmp" start
sleep 5

psql -U "$(whoami)" -tc "SELECT 1 FROM pg_roles WHERE rolname = 'postgres'" | grep -q 1 || \
  createuser -s postgres

psql -U postgres -tc "SELECT 1 FROM pg_roles WHERE rolname = '${USER}'" | grep -q 1 || \
  psql -U postgres -c "CREATE USER ${USER} WITH PASSWORD 'yourpassword';"

psql -U postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'measurements'" | grep -q 1 || \
  psql -U postgres -c "CREATE DATABASE measurements OWNER ${USER};"

psql -U "${USER}" -d measurements -f "${SYS_DIR}/schema_setup.sql"

# To check if they are running
ps aux | grep redis
ps aux | grep postgres

# 3. Submit Region Listener
qsub -N region_listener -o logs/region_listener.out -e logs/region_listener.err -b y -cwd -v REDIS_HOST="${REDIS_HOST}", REDIS_PORT="${REDIS_PORT}" \
    apptainer exec \
      --bind "${REGION_DIR}":/region_files \
      --bind "${APP_DIR}/region_listener.py":/app/region_listener.py \
      "$SIF_DIR/region_listener.sif" python /app/region_listener.py

# 4. Launch N tracker jobs
for i in $(seq 1 $NUM_TRACKERS); do
    qsub -N tracker-$i -o logs/tracker-$i.out -e logs/tracker-$i.err \
      -v REDIS_HOST="$REDIS_HOST",POSTGRES_HOST="$POSTGRES_HOST" \
      submit_tracker.sh
    sleep 1
done
echo "Submitted $NUM_TRACKERS tracker jobs."                
                    
# 5. Data listener
apptainer run   --bind "${DETECT_YAML_DIR}":/param_files \
                --bind "${DEX_DATA_DIR}":/dex_data \
                --bind "${REGION_DIR}":/region_files \
                --bind "${APP_DIR}/dexela_listener.py":/app/dexela_listener.py \
                --bind "${SPOTFETCH_DIR}/utilities.py":/app/utilities.py \
                "${SIF_DIR}/dexela_listener.sif" \
                python /app/dexela_listener.py

# 6. Stats processor
apptainer run \
  --bind "${APP_DIR}/stats_processor.py":/app/stats_processor.py \
  "${SIF_DIR}"/stats_processor.sif \
  python /app/stats_processor.py

