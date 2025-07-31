import os
import sys
import psycopg2
from psycopg2 import sql
import yaml

# Load config
with open(sys.argv[1], 'r') as f:
    config = yaml.safe_load(f)

sys_cfg = config['system']
db_user = sys_cfg['db_user']
postgres_dir = os.path.join(sys_cfg['base_dir'], sys_cfg['postgres_dir'])
schema_file = os.path.join(os.environ.get("SYS_DIR", "."), "schema_setup.sql")

# Connection params
conn_params = {
    'dbname': sys_cfg["db_name"],
    'user': db_user
}

def run_sql(conn, query, params=None):
    with conn.cursor() as cur:
        cur.execute(query, params)

def role_exists(conn, role_name):
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role_name,))
        return cur.fetchone() is not None

def db_exists(conn, db_name):
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        return cur.fetchone() is not None

# Connect to postgres
try:
    conn = psycopg2.connect(**conn_params)
    conn.autocommit = True
except Exception as e:
    print(f"Could not connect to PostgreSQL: {e}")
    sys.exit(1)

# Ensure user exists
if not role_exists(conn, db_user):
    print(f"Creating user: {db_user}")
    run_sql(conn, sql.SQL("CREATE USER {} WITH PASSWORD %s").format(sql.Identifier(db_user)), ['yourpassword'])

# Ensure database exists
if not db_exists(conn, 'measurements'):
    print("Creating database: measurements")
    run_sql(conn, sql.SQL("CREATE DATABASE measurements OWNER {}").format(sql.Identifier(db_user)))

# Load schema
print("Loading schema...")
conn.close()
conn = psycopg2.connect(dbname='measurements', user=db_user)
with conn.cursor() as cur:
    with open(schema_file, 'r') as f:
        cur.execute(f.read())
conn.commit()
conn.close()

print("PostgreSQL setup complete.")
