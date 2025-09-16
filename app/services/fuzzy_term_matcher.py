from rapidfuzz import process, fuzz
from sqlalchemy import text
from app.data.db import get_db_engine

def load_known_terms():
    """
    Load distinct known terms grouped by column from PMS_Defect_Backup_cleaned.
    Returns {col_name: [values]} dict.
    """
    engine = get_db_engine()
    columns = [
        "VESSEL_NAME", "EQUIPMENT_NAME", "MAKER", "MODEL", "JOB_STATUS",
        "DEFECT_SECTION", "JOB_CATEGORY", "JOB_TYPE", "PRIORITY"
    ]

    known_terms_by_col = {}
    with engine.connect() as conn:
        for col in columns:
            query = text(f"SELECT DISTINCT {col} FROM [dbo].[PMS_Defect_Backup_cleaned]")
            results = conn.execute(query).fetchall()
            terms = [row[0].lower().strip() for row in results if row[0]]
            known_terms_by_col[col] = terms

    return known_terms_by_col


# 🔹 Cached in memory for performance
KNOWN_TERMS = load_known_terms()


def find_closest_terms(term: str, known_terms: list[str], threshold: int = 60):
    """
    Find closest matching terms from known_terms using fuzzy matching.
    Returns top N matches above threshold.
    """
    if not term or not known_terms:
        return None

    # 🔹 Search against ALL known terms, not truncated
    matches = process.extract(term, known_terms, scorer=fuzz.ratio, limit=len(known_terms))

    print("matches-----------------", matches[:50])  # show top 10 for debug

    # Filter by threshold
    filtered = [
        {"original": term, "term": candidate, "score": score}
        for candidate, score, _ in matches
        if score >= threshold   
    ]

    print("filtered-------------------", filtered)
    return filtered if filtered else None



