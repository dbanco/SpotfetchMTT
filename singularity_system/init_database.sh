# 1. Initialize DB cluster if needed
if [ ! -f "$POSTGRES_DIR/PG_VERSION" ]; then
  echo "[INFO] Initializing PostgreSQL cluster at $POSTGRES_DIR"
  initdb -D "$POSTGRES_DIR"
fi

# 2. Start PostgreSQL
echo "[INFO] Starting PostgreSQL..."
pg_ctl -D "$POSTGRES_DIR" -l "$POSTGRES_DIR/logfile" -o "-k /tmp" start
sleep 5

# 3. Create postgres superuser (if missing)
if ! psql -U "$(whoami)" -d postgres -h /tmp -tc "SELECT 1 FROM pg_roles WHERE rolname = 'postgres'" | grep -q 1; then
  echo "[INFO] Creating 'postgres' role"
  createuser -s postgres -h /tmp || echo "[WARN] 'postgres' role may already exist"
fi

# 4. Create your user (if missing)
if ! psql -U postgres -d postgres -h /tmp -tc "SELECT 1 FROM pg_roles WHERE rolname = '${USER}'" | grep -q 1; then
  echo "[INFO] Creating user '$USER'"
  psql -U postgres -d postgres -h /tmp -c "CREATE USER ${USER} WITH PASSWORD 'yourpassword';"
fi

# 5. Create database if missing
if ! psql -U postgres -d postgres -h /tmp -tc "SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}'" | grep -q 1; then
  echo "[INFO] Creating database '$DB_NAME'"
  psql -U postgres -d postgres -h /tmp -c "CREATE DATABASE ${DB_NAME} OWNER ${USER};"
fi