# LAUNCH COMMANDS:

# HOST COMPUTER
lnx7108.classe.cornell.edu

# POSTGRES 
export POSTGRES_DIR=/nfs/chess/user/dbanco/SpotfetchMTT/tracker-deploy-chess/postgres_data

mkdir -p $POSTGRES_DIR  # Ensure the data directory exists

# init database
apptainer exec \
  --bind $POSTGRES_DIR:/var/lib/postgresql/data \
  sif/postgres.sif \
  initdb -D /var/lib/postgresql/data
  
# start server
apptainer exec \
  --bind $POSTGRES_DIR:/var/lib/postgresql/data \
  sif/postgres.sif \
  postgres -D /var/lib/postgresql/data \
  -c listen_addresses='*' -p 5432
  
  apptainer exec \
  --bind $POSTGRES_DIR:/var/lib/postgresql/data \
  --bind /nfs/chess/user/dbanco/SpotfetchMTT/tracker-deploy-chess//pgsql_run:/var/run/postgresql \
  sif/postgres.sif \
  postgres -D /var/lib/postgresql/data \
  -k /var/run/postgresql \
  -c listen_addresses='*' \
  -p 5432

# REDIS
apptainer run --network --network-args "portmap=6379:6379/tcp" redis.sif

redis-server --bind 0.0.0.0 --port 6379

redis-cli -h localhost -p 6379 ping

nohup redis-server --bind 0.0.0.0 --port 6379 --dir /tmp/redis-empty > redis.log 2>&1 &

# OMEGA FRAME LISTENER
apptainer exec \
  --bind /nfs/chess/user/dbanco/SpotfetchMTT/tracker-deploy-chess/omega_frames:/omega_frames \
  --env REDIS_HOST=localhost \
  sif/omega_listener.sif \
  python /app/omega_frame_listener.py

# JOB DISPATCHER
apptainer run \
  --bind /nfs/chess/user/dbanco/c103_processing:/data \
  sif/job_dispatcher.sif



# TRACKER