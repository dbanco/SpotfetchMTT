#!/bin/bash
set -e

echo "Shutting down Spotfetch system..."

# 1. Kill tmux sessions for Redis and PostgreSQL
echo "Stopping Redis tmux session..."
pkill redis || true

echo "Stopping PostgreSQL..."
pg_ctl -D "/nfs/chess/user/dbanco/SpotfetchMTT/singularity_system/postgres_data" stop || echo "PostgreSQL not running or already stopped."

# Optional: kill any remaining PostgreSQL processes forcibly (use carefully)
pkill -f postgres || true

# 2. Kill Apptainer foreground jobs (dexela_listener, stats_processor)
echo "Killing apptainer jobs..."

if [ -f listener.pid ]; then
    pid=$(cat listener.pid)
    echo "Killing dexela_listener.py (PID $pid)..."
    kill $pid 2>/dev/null || echo "Process not running."
    rm listener.pid
else
    echo "listener.pid not found. Nothing to kill."
fi

pkill -f stats_processor.py || echo "stats_processor not running."

# 3. Delete all qsub jobs for current user (careful: this removes **all** your jobs)
echo "Deleting all submitted SGE jobs"
qstat -u "$USER" | awk 'NR>2 {print $1}' | xargs -r qdel

echo "Shutdown complete."

