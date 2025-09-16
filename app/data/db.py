# from sqlalchemy import create_engine
# import urllib

# def get_db_engine():
    
#     params = urllib.parse.quote_plus(
#     "DRIVER={ODBC Driver 17 for SQL Server};"
#     "SERVER=10.201.1.86,50001;"
#     "DATABASE=Resume_Parser;"
#     "Trusted_Connection=Yes;"
# )
    
    
#     return create_engine(
#        f"mssql+pyodbc:///?odbc_connect={params}"
#     )



# from sqlalchemy import create_engine

# def get_db_engine():
#     return create_engine(
#         "mssql+pyodbc://@10.201.1.86,50001/Resume_Parser"
#         "?driver=ODBC+Driver+17+for+SQL+Server"
#         "&trusted_connection=yes"
#     )



import os
import pandas as pd
from sqlalchemy import create_engine, text

DB_FILE = "local.db"
EXCEL_FILE = "cleaned_output.xlsx"

def initialize_local_db():
    """Create local DB and load Excel if not already present."""
    db_exists = os.path.exists(DB_FILE)
    

    engine = create_engine(f"sqlite:///{DB_FILE}")

    if not db_exists:
        # Load Excel into DB
        df = pd.read_excel(EXCEL_FILE)
        
        df.to_sql("PMS_Defect_Backup_cleaned", engine, if_exists="replace", index=False)
        
        print(df)

        print("✅ Created DB and loaded Excel data.")

    return engine

def get_db_engine():
    """Return engine and ensure chat_log table exists."""
    engine = create_engine(f"sqlite:///{DB_FILE}")

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chat_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                question TEXT,
                answer TEXT,
                sql_query TEXT
            )
        """))
        conn.commit()

        # Show table schema
        result = conn.execute(text("PRAGMA table_info(chat_log)")).fetchall()
        columns = [row[1] for row in result]  # row[1] = column name
        print("chat_log columns:", columns)

    return engine


# Run once at startup
# initialize_local_db()
