# from services.db import engine
from app.data.db import get_db_engine
import pandas as pd

def load_categorical_metadata():
    """
    Load distinct values from categorical columns.
    Add/remove columns as needed — dynamic & maintainable.
    """
    columns = [
    "VESSEL_NAME", "EQUIPMENT_CODE", "EQUIPMENT_NAME", "MAKER", "MODEL",
    "JOB_TITLE", "JOBORDER_CODE", "JOB_STATUS", "DEFECT_SECTION",
    "JOB_CATEGORY", "JOB_TYPE", "PRIORITY", "DESCRIPTION", "ISSUE_DATE",
    "RANK", "JOB_START_DATE", "JOB_END_DATE", "CLOSING_REPORT"
]
    metadata = {}
    engine = get_db_engine()
    with engine.connect() as conn:
        for col in columns:
            df = pd.read_sql(
                f"SELECT DISTINCT [{col}] FROM PMS_Defect_Backup_cleaned WHERE [{col}] IS NOT NULL",
                conn
            )
            metadata[col] = df[col].dropna().unique().tolist()
    return metadata
