import datetime
from sqlalchemy import text
from app.data.db import get_db_engine

def save_chat_log(session_id, question, answer):
    engine = get_db_engine()
    now = datetime.datetime.now()
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO chat_log (session_id, question, answer, timestamp) VALUES (:s, :q, :a, :t)"),
            {"s": session_id, "q": question, "a": answer, "t": now}
        )
        conn.commit()

def get_recent_history(session_id, limit=3):
    engine = get_db_engine()
    with engine.connect() as conn:
        res = conn.execute(
            text("SELECT question, answer FROM chat_log WHERE session_id=:s ORDER BY timestamp DESC"),
            {"s": session_id}
        ).fetchmany(limit)
    return list(reversed(res))