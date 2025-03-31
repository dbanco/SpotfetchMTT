# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 23:06:20 2025

@author: dpqb1
"""

import psycopg2

# Database connection settings
DB_NAME = "tracker_data"
DB_USER = "dbanco"
DB_PASS = "yourpassword"
DB_HOST = "postgres_db"

def reset_database():
    """Drops all tables and recreates the schema from scratch."""
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
        cursor = conn.cursor()

        print("Dropping all tables...", flush=True)
        cursor.execute("DROP TABLE IF EXISTS measurements CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS tracks CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS regions CASCADE;")
        conn.commit()

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error resetting database: {e}", flush=True)
    
if __name__ == "__main__":
    reset_database()