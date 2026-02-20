import sqlite3
import pandas as pd
from langchain_core.tools import tool

# ==========================================
# 1. One-time Setup: Load CSV into SQLite
# ==========================================
def setup_database(csv_path="hospitals.csv"):
    """Reads the large CSV and creates a lightweight, searchable SQL database."""
    df = pd.read_csv(csv_path)
    df = df.fillna("Unknown") # Prevent null errors
    
    conn = sqlite3.connect("data/hospitals.db")
    # Load into a table called 'hospitals'
    df.to_sql("hospitals", conn, if_exists="replace", index=False)
    conn.close()
    print("Database successfully loaded into SQLite!")