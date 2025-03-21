# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 16:26:23 2025

@author: dpqb1
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import pandas as pd
import json

def fetch_variance_histogram():
    conn = psycopg2.connect(
        dbname="tracker_data",
        user="dbanco",
        password="yourpassword",
        host="postgres_db"
    )
    query = """
    SELECT scan_number, (features->'variance')::jsonb AS var
    FROM measurements
    WHERE detected = TRUE AND overlapping = FALSE
    ORDER BY scan_number;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

st.set_page_config(layout="wide")
st.title("📊 Variance Histogram Over Time")

data = fetch_variance_histogram()

# Extract max variance and bin by scan
# First parse JSON strings to lists
data["var"] = data["var"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
# Then compute max variance safely
scan_bins = sorted(data["scan_number"].unique())
hist_data = [data[data["scan_number"] == scan]["max_var"].values for scan in scan_bins]

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(np.array([np.histogram(h, bins=20)[0] for h in hist_data]), aspect='auto', origin='lower')
ax.set_xlabel("Histogram Bin")
ax.set_ylabel("Scan Number")
st.pyplot(fig)
