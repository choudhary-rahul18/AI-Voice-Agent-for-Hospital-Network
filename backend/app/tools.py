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

# ==========================================
# 2. Define the Search Tool for Gemini
# ==========================================
@tool
def search_hospitals(hospital_name: str = None, city: str = None, limit: int = 5, offset: int = 0) -> str:
    """
    Search the hospital database by name or city.
    Call this tool when a user asks for hospitals in an area.
    If the user asks for "more" hospitals, use the 'offset' parameter to skip the ones you already provided (e.g., offset=5).
    """
    print(f"\n[SYSTEM LOG] 🛠️ TOOL TRIGGERED: Searching database for Name: '{hospital_name}', City: '{city}', Offset: {offset}")
    
    conn = sqlite3.connect("data/hospitals.db")
    cursor = conn.cursor()
    
    query = 'SELECT "HOSPITAL NAME", "Address", "CITY" FROM hospitals WHERE 1=1'
    params = []
    
    if hospital_name:
        query += ' AND "HOSPITAL NAME" LIKE ?'
        params.append(f"%{hospital_name}%")
        
    if city:
        query += ' AND "CITY" LIKE ?'
        params.append(f"%{city}%")
        
    # ADD THE OFFSET LOGIC HERE
    query += f" LIMIT {limit} OFFSET {offset}"
    
    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()
    
    if len(results) > 1 and hospital_name and not city:
        return f"SYSTEM_NOTE: Found {len(results)} hospitals named {hospital_name}. Tell the user: 'I have found several hospitals with this name. In which city are you looking for {hospital_name}?'"
    
    if not results:
        return "No hospitals found matching those criteria."
        
    output = "Database Results:\n"
    for row in results:
        output += f"- Name: {row[0]}, Address: {row[1]}, City: {row[2]}\n"
        
    return output

# List of tools to bind to Gemini later
loop_tools = [search_hospitals]