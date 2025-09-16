import pandas as pd
from sqlalchemy import text
from app.data.db import get_db_engine

def fetch_defect_data():
    engine = get_db_engine()
    return pd.read_sql(text("SELECT * FROM PMS_Defect_Backup_cleaned"), engine)