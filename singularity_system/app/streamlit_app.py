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
import time

def fetch_variance_histogram():
    conn = psycopg2.connect(
        dbname="postgres",
        user="dbanco",
        password="yourpassword",
        host="lnx202.classe.cornell.edu"
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


# def fetch_intensity():
#     conn = psycopg2.connect(
#         dbname="postgres",
#         user="dbanco",
#         password="yourpassword",
#         host="lnx7108.classe.cornell.edu"
#     )
#     query = """
#     SELECT scan_number, (features->'intensity')::jsonb AS var
#     FROM measurements
#     WHERE detected = TRUE AND overlapping = FALSE
#     ORDER BY scan_number;
#     """
#     df = pd.read_sql(query, conn)
#     conn.close()
#     return df

st.set_page_config(layout="wide")
st.title("📊 Major Axis Variance Histogram")

data = fetch_variance_histogram()

# intensity = fetch_intensity()


# Extract max variance and bin by scan
# First parse JSON strings to lists
data["var"] = data["var"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
# Then compute max variance safely
data["max_var"] = data["var"].apply(lambda v: max(v) if isinstance(v, list) and len(v) > 0 else None)
#st.write("Columns in DataFrame:", data.columns)
scan_bins = sorted(data["scan_number"].unique())
hist_data = [data[data["scan_number"] == scan]["max_var"].dropna().values for scan in scan_bins]

#st.write("scan_bins:", scan_bins)
#st.write("hist_data:", hist_data)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(np.array([np.histogram(h, bins=20)[0] for h in hist_data]), aspect='auto', origin='lower')
ax.set_xlabel("Histogram Bin")
ax.set_ylabel("Scan Number")
st.pyplot(fig)

### Line plot of average variance
avg_var = np.zeros((len(scan_bins)))
for scan in scan_bins:
    avg_var[scan] = np.mean(hist_data[scan])
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(avg_var,'o-')
ax2.set_xlabel("Scan Number")
ax2.set_ylabel("Average Major Axis Variance")
st.pyplot(fig2)

