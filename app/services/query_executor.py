import pandas as pd
from sqlalchemy import text
from app.data.db import get_db_engine

def run_sql_query(sql: str) -> pd.DataFrame:
    
    dangerous = ["drop", "delete", "update", "alter", "insert", "truncate", "create"]
    if any(word in sql.lower() for word in dangerous):
        raise ValueError("Destructive queries are not allowed. Only SELECT statements are permitted.")
    
    engine = get_db_engine()
    return pd.read_sql(text(sql), engine)