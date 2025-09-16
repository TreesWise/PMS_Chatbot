# from services.llm_clients import get_llm
# from prompts.new_query_prompt import get_new_query_prompt
# from services.query_executor import run_sql_query
# from services.answer_summarizer import summarize_answer
# from agents.response_mode_agent import classify_response_mode
# from services.metadata_loader import load_categorical_metadata
# from services.list_answers import list_answer


# metadata = load_categorical_metadata()


# def generate_new_query(user_input):
#     llm = get_llm()
#     prompt = get_new_query_prompt()
#     chain = prompt | llm

#     # Generate SQL
#     response = chain.invoke({"input": user_input})
#     sql = response.content.strip()
#     print(f"[DEBUG] Generated SQL: {sql}")

#     # Run SQL → get dataframe
#     df = run_sql_query(sql)

#     # Summarize answer based on dataframe + original user question
    

#     mode = classify_response_mode(user_input)

#     if mode == "list":
#         # return nicely formatted table (markdown or streamlit dataframe)
#         # return df.to_markdown(index=False)
#         return list_answer(df, user_input)
#     else:
#         # return human text summary
#         return summarize_answer(df, user_input)





# from app.services.llm_clients import get_llm
# from app.services.query_executor import run_sql_query
# from app.services.answer_summarizer import summarize_answer
# from app.agents.response_mode_agent import classify_response_mode
# from app.services.metadata_loader import load_categorical_metadata
# from app.services.list_answers import list_answer
# from app.services.chat_manager import save_chat_log
# from app.data.db import get_db_engine

# from langchain_community.agent_toolkits.sql.base import create_sql_agent
# from langchain_community.utilities import SQLDatabase
# from langchain.agents import AgentType

# metadata = load_categorical_metadata()


# def generate_new_query(user_input, session_id=None):
#     """
#     Generate and run SQL using LangChain's SQL Agent.
#     Logs both the generated SQL and the final answer (if session_id provided).
#     """
#     llm = get_llm(temperature=0)
#     db = SQLDatabase(get_db_engine())



#     instructions = """
#     You are a SQL expert working with the '[dbo].[PMS_Defect_Backup_cleaned]' table. 
#     Your job is to generate SQL queries based on the user's request.
#     Ensure that:
#     1. The queries only involve the '[dbo].[PMS_Defect_Backup_cleaned]' table.
#     2. Use the metadata provided to ensure the correct handling of column values.
#     3. If user asks about defect text/type/keywords, search in: JOB_TITLE, DESCRIPTION, CLOSING_REPORT.
    
    
#     Here is the metadata (data dictionary):

#     • VESSEL_NAME → Name of the vessel where the defect was reported or the maintenance job was carried out.
#     • EQUIPMENT_CODE → Unique identifier or code assigned to a specific equipment on the vessel.
#     • EQUIPMENT_NAME → Descriptive name of the equipment related to the defect or maintenance activity.
#     • MAKER → Name of the manufacturer or company that produced the equipment.
#     • MODEL → Specific model or version of the equipment provided by the manufacturer.
#     • JOB_TITLE → Brief title or summary describing the defect or nature of the job carried out.
#     • JOBORDER_CODE → Unique job order number/reference used to track the maintenance or repair job.
#     • JOB_STATUS → Current state of the job (e.g., Open, In Progress, Completed, Closed).
#     • DEFECT_SECTION → Section or area of the vessel where the defect occurred.
#     • JOB_CATEGORY → Broad classification of the job such as Maintenance, Repair, or Inspection.
#     • JOB_TYPE → Specific type of job under the category.
#     • PRIORITY → Importance or urgency level of the job (e.g., High, Medium, Low).
#     • DESCRIPTION → Detailed explanation of the defect or job requirements.
#     • ISSUE_DATE → Date when the defect or job was reported (YYYY-MM-DD).
#     • RANK → Designation of the crew member.
#     • JOB_START_DATE → Date when the job started.
#     • JOB_END_DATE → Date when the job was completed.
#     • CLOSING_REPORT → Final remarks upon job closure.
#     """
    
#     # metadata_str = f"Here are the categorical metadata values: {metadata}"
#     # print("metadata------------------", metadata_str)
#     # agent_prompt = f"{instructions}\n{metadata_str}\nUser's question: {user_input}"
#     agent_prompt = f"{instructions}\nUser's question: {user_input}"

#     # Create SQL Agent
#     agent_executor = create_sql_agent(
#         llm=llm,
#         db=db,
#         agent_type=AgentType.OPENAI_FUNCTIONS,
#         verbose=True,
#         handle_parsing_errors=True
#     )

#     generated_sql = None
#     df = None
#     final_answer = None

#     try:
#         # Ask SQL Agent to handle the user query
#         raw_result = agent_executor.run(agent_prompt)
#         print(raw_result)

#         # Check if the output looks like SQL
#         if raw_result.strip().lower().startswith("select"):
#             generated_sql = raw_result
#             df = run_sql_query(generated_sql)
#             mode = classify_response_mode(user_input)
#             if mode == "list":
#                 final_answer = list_answer(df, user_input)
#             else:
#                 final_answer = summarize_answer(df, user_input)
#         else:
#             # If the agent returned a natural language answer
#             final_answer = raw_result
#             df = None

#     except Exception as e:
#         print(f"[ERROR] SQL Agent execution failed: {e}")
#         final_answer = "⚠️ Sorry, I couldn't process that query."
#         df = None

#     # If we have a DataFrame, decide list/summary mode
#     # if df is not None:
#     #     mode = classify_response_mode(user_input)
#     #     if mode == "list":
#     #         final_answer = list_answer(df, user_input)
#     #     else:
#     #         final_answer = summarize_answer(df, user_input)

#     # Log the SQL + Answer for debugging/analytics
#     if session_id:
#         log_text = f"[SQL]: {generated_sql if generated_sql else 'N/A'}\n[ANSWER]: {final_answer}"
#         save_chat_log(session_id, user_input, log_text)
    
#     return final_answer, df




from app.services.llm_clients import get_llm
from app.services.query_executor import run_sql_query
from app.services.answer_summarizer import summarize_answer
from app.agents.response_mode_agent import classify_response_mode
from app.services.metadata_loader import load_categorical_metadata
from app.services.list_answers import list_answer
from app.services.chat_manager import save_chat_log
from app.data.db import get_db_engine
from app.services.fuzzy_term_matcher import find_closest_terms, load_known_terms  # new helper you’ll create
from app.services.chat_manager import save_chat_log

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentType
import re
import pandas as pd
import numpy as np

metadata = load_categorical_metadata()


from langchain.callbacks.base import BaseCallbackHandler

class SQLLoggingCallback(BaseCallbackHandler):
    def __init__(self):
        self.generated_sql = None

    def on_tool_start(self, serialized, input_str, **kwargs):
        if serialized.get("name") == "sql_db_query":
            try:
                # Try JSON decode first
                import json
                inputs = json.loads(input_str)
                self.generated_sql = inputs.get("query")
            except Exception:
                # Fallback: eval if it's Python-style dict
                try:
                    inputs = eval(input_str)  # safe here since we control tool inputs
                    self.generated_sql = inputs.get("query")
                except Exception as e:
                    print(f"⚠️ Could not parse query input: {e}")
                    self.generated_sql = None

            print(f"\n🔍 SQL Query Executed:\n{self.generated_sql}\n")

def extract_sql_from_response(response):
    """
    Extract SQL query from the agent's response if it contains one.
    Returns the SQL query if found, None otherwise.
    """
    # Look for SQL patterns in the response
    sql_patterns = [
        r"```sql\s*(.*?)\s*```",  # SQL code blocks
        r"SELECT.*?;",             # SELECT statements
        r"select.*?;",             # lowercase select   
    ]
    
    for pattern in sql_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            sql_query = match.group(1) if pattern.startswith("```sql") else match.group(0)
            return sql_query.strip()
    
    return None
import re


def extract_columns_from_sql(sql: str, all_columns: list[str]) -> list[str]:
    """
    Extract column names from the WHERE clause only.
    Matches only valid schema columns.
    """
    if not sql:
        return []

    sql_lower = sql.lower()
    used_cols = []

    # Extract WHERE clause only
    where_match = re.search(r"\bwhere\b(.*?)\b(group by|order by|;|$)", sql_lower, re.DOTALL)
    if not where_match:
        return []

    where_clause = where_match.group(1)

    for col in all_columns:
        if col.lower() in where_clause:
            used_cols.append(col)

    return used_cols

def extract_like_terms(sql: str) -> list[str]:
    """
    Extract terms inside LIKE '%...%' clauses from the SQL.
    Example: WHERE MAKER LIKE '%WTSILX%' → ["WTSILX"]
    """
    return re.findall(r"LIKE\s*'%([^%]+)%'", sql, flags=re.IGNORECASE)

def clean_dataframe_for_json(df):
    """
    Clean DataFrame to make it JSON serializable by replacing NaN and infinite values.
    If DataFrame is empty or None, return an empty list (or dictionary).
    """
    if df is None or df.empty:
        return []  # Return an empty list for an empty or None DataFrame
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Replace NaN, NaT, and infinite values with None (which becomes null in JSON)
    df_clean = df_clean.replace([np.nan, pd.NaT], None)
    df_clean = df_clean.replace([np.inf, -np.inf], None)
    
    # Convert all columns to string to avoid any serialization issues
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].astype(str)
    
    return df_clean.to_dict(orient="records")  # Convert to list of dictionaries (JSON friendly)


# def generate_new_query(user_input, session_id=None):
#     """
#     Generate and run SQL using LangChain's SQL Agent.
#     Logs both the generated SQL and the final answer (if session_id provided).
#     """
#     llm = get_llm(temperature=0)
#     db = SQLDatabase(get_db_engine())


#     instructions = """
#     You are a SQL expert working with the '[dbo].[PMS_Defect_Backup_cleaned]' table.
#     Your job is to generate SQL queries based on the user's request.
#     When answering, ALWAYS return a JSON object with two keys:
#         - "sql" → the exact SQL query you used
#         - "answer" → the final natural language answer to the user

#         Example format:
#         {
#         "sql": "SELECT COUNT(*) FROM [dbo].[PMS_Defect_Backup_cleaned];",
#         "answer": "There are 152 defects recorded."
#         }
#     Ensure that:
#     1. The queries only involve the '[dbo].[PMS_Defect_Backup_cleaned]' table.
#     2. Use the metadata provided to ensure the correct handling of column values.
#     3. If user asks about defect text/type/keywords, search in: JOB_TITLE, DESCRIPTION, CLOSING_REPORT.
#     4. If the query involves **multiple defect types**, generate a single SQL query that uses 
#         **conditional aggregation** — one column per defect type.
#         Example format (replace <keyword> dynamically with the requested defect types):
#         ```sql
#         SELECT 
#             YEAR(ISSUE_DATE) AS Year,
#             SUM(CASE WHEN JOB_TITLE LIKE '%<keyword1>%' OR DESCRIPTION LIKE '%<keyword1>%' OR CLOSING_REPORT LIKE '%<keyword1>%' THEN 1 ELSE 0 END) AS <Keyword1>_Defects,
#             SUM(CASE WHEN JOB_TITLE LIKE '%<keyword2>%' OR DESCRIPTION LIKE '%<keyword2>%' OR CLOSING_REPORT LIKE '%<keyword2>%' THEN 1 ELSE 0 END) AS <Keyword2>_Defects,
#             ...
#         FROM [dbo].[PMS_Defect_Backup_cleaned]
#         GROUP BY YEAR(ISSUE_DATE)
#         ORDER BY Year;
#     5. If the user asks for "last N years", filter using
#         WHERE YEAR(ISSUE_DATE) >= YEAR(GETDATE()) - (N - 1)
#     6. Always return the SQL query in a code block (```sql) along with your explanation.
    
#     Here is the metadata (data dictionary):

#     • VESSEL_NAME →     Name of the vessel where the defect was reported or the maintenance job was carried out.
#     • EQUIPMENT_CODE → Unique identifier or code assigned to a specific equipment on the vessel.
#     • EQUIPMENT_NAME → Descriptive name of the equipment related to the defect or maintenance activity.
#     • MAKER → Name of the manufacturer or company that produced the equipment.
#     • MODEL → Specific model or version of the equipment provided by the manufacturer.
#     • JOB_TITLE → Brief title or summary describing the defect or nature of the job carried out.
#     • JOBORDER_CODE → Unique job order number/reference used to track the maintenance or repair job.
#     • JOB_STATUS → Current state of the job (These are the job status types: 2022, 97, Cancelled, COMPLETED, EQUIPMENT NOT OPERATIONAL, Pending, Postponed, Postponed to Dry Dock, Waiting for Assistance, Waiting for Spares, Work in Progress).
#     • DEFECT_SECTION → Section or area of the vessel where the defect occurred.
#     • JOB_CATEGORY → Broad classification of the job such as Maintenance, Repair, or Inspection.
#     • JOB_TYPE → Specific type of job under the category.
#     • PRIORITY → Importance or urgency level of the job (These are the Priorities: In Use, Normal, NOT OPERATIONAL, Repair).
#     • DESCRIPTION → Detailed explanation of the defect or job requirements.
#     • ISSUE_DATE → Date when the defect or job was reported (YYYY-MM-DD).
#     • RANK → Designation of the crew member.
#     • JOB_START_DATE → Date when the job started.
#     • JOB_END_DATE → Date when the job was completed.
#     • CLOSING_REPORT → Final remarks upon job closure.
#     """


def generate_new_query(user_input, session_id=None):
    """
    Generate and run SQL using LangChain's SQL Agent.
    Logs both the generated SQL and the final answer (if session_id provided).
    """
    llm = get_llm(temperature=0)
    db = SQLDatabase(get_db_engine())


    instructions = """
    You are a SQL expert working with the 'PMS_Defect_Backup_cleaned' table.
    Your job is to generate SQL queries based on the user's request.
    When answering, ALWAYS return a JSON object with two keys:
        - "sql" → the exact SQL query you used
        - "answer" → the final natural language answer to the user

        Example format:
        {
        "sql": "SELECT COUNT(*) FROM PMS_Defect_Backup_cleaned;",
        "answer": "There are 152 defects recorded."
        }
    Ensure that:
    1. The queries only involve the 'PMS_Defect_Backup_cleaned' table.
    2. Use the metadata provided to ensure the correct handling of column values.
    3. If user asks about defect text/type/keywords, search in: JOB_TITLE, DESCRIPTION, CLOSING_REPORT.
    4. If the query involves **multiple defect types**, generate a single SQL query that uses 
        **conditional aggregation** — one column per defect type.
        Example format (replace <keyword> dynamically with the requested defect types):
        ```sql
        SELECT 
            YEAR(ISSUE_DATE) AS Year,
            SUM(CASE WHEN JOB_TITLE LIKE '%<keyword1>%' OR DESCRIPTION LIKE '%<keyword1>%' OR CLOSING_REPORT LIKE '%<keyword1>%' THEN 1 ELSE 0 END) AS <Keyword1>_Defects,
            SUM(CASE WHEN JOB_TITLE LIKE '%<keyword2>%' OR DESCRIPTION LIKE '%<keyword2>%' OR CLOSING_REPORT LIKE '%<keyword2>%' THEN 1 ELSE 0 END) AS <Keyword2>_Defects,
            ...
        FROM PMS_Defect_Backup_cleaned
        GROUP BY YEAR(ISSUE_DATE)
        ORDER BY Year;
    5. If the user asks for "last N years", filter using
        WHERE YEAR(ISSUE_DATE) >= YEAR(GETDATE()) - (N - 1)
    6. Always return the SQL query in a code block (```sql) along with your explanation.
    
    Here is the metadata (data dictionary):

    • VESSEL_NAME →     Name of the vessel where the defect was reported or the maintenance job was carried out.
    • EQUIPMENT_CODE → Unique identifier or code assigned to a specific equipment on the vessel.
    • EQUIPMENT_NAME → Descriptive name of the equipment related to the defect or maintenance activity.
    • MAKER → Name of the manufacturer or company that produced the equipment.
    • MODEL → Specific model or version of the equipment provided by the manufacturer.
    • JOB_TITLE → Brief title or summary describing the defect or nature of the job carried out.
    • JOBORDER_CODE → Unique job order number/reference used to track the maintenance or repair job.
    • JOB_STATUS → Current state of the job (These are the job status types: 2022, 97, Cancelled, COMPLETED, EQUIPMENT NOT OPERATIONAL, Pending, Postponed, Postponed to Dry Dock, Waiting for Assistance, Waiting for Spares, Work in Progress).
    • DEFECT_SECTION → Section or area of the vessel where the defect occurred.
    • JOB_CATEGORY → Broad classification of the job such as Maintenance, Repair, or Inspection.
    • JOB_TYPE → Specific type of job under the category.
    • PRIORITY → Importance or urgency level of the job (These are the Priorities: In Use, Normal, NOT OPERATIONAL, Repair).
    • DESCRIPTION → Detailed explanation of the defect or job requirements.
    • ISSUE_DATE → Date when the defect or job was reported (YYYY-MM-DD).
    • RANK → Designation of the crew member.
    • JOB_START_DATE → Date when the job started.
    • JOB_END_DATE → Date when the job was completed.
    • CLOSING_REPORT → Final remarks upon job closure.
    """
    
    agent_prompt = f"{instructions}\nUser's question: {user_input}"

    # Create SQL Agent
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True
    )
    generated_sql = None
    df = None
    final_answer = None
    raw_result = None

    try:
        # callback = SQLLoggingCallback()
        # raw_result = agent_executor.invoke({"input": agent_prompt}, callbacks=[callback])
        # generated_sql = callback.generated_sql
        
        
        try:
            import json
            callback = SQLLoggingCallback()
            raw_result = agent_executor.invoke({"input": agent_prompt}, callbacks=[callback])
            if isinstance(raw_result, dict):
                # If the agent returns with "output", unwrap it
                response_text = raw_result.get("output") or raw_result
                if isinstance(response_text, str):
                    response_obj = json.loads(response_text)
                else:
                    response_obj = response_text
            else:
                response_obj = json.loads(raw_result)

            generated_sql = response_obj.get("sql")
            final_answer = response_obj.get("answer")
        except Exception as e:
            print(f"⚠️ Could not parse agent JSON: {e}")
            generated_sql = callback.generated_sql or extract_sql_from_response(raw_result)
            final_answer = raw_result

        print("✅ SQL from agent:", generated_sql)

        if generated_sql:
            df = run_sql_query(generated_sql)
            print(f"DataFrame after SQL query: {df}")
            # 🔹 If DataFrame is empty → try fuzzy matching
            if df is None or df.empty:
                print("⚠️ No results found. Checking for fuzzy matches...")

                known_terms_by_col = load_known_terms()
                
                
                used_cols = extract_columns_from_sql(generated_sql, list(known_terms_by_col.keys()))
                print(f"Columns detected in SQL: {used_cols}")
                
                like_terms = extract_like_terms(generated_sql)
                print(f"LIKE terms extracted: {like_terms}")

                suggestions = []
                for col in used_cols:
                    col_terms = known_terms_by_col.get(col, [])
                    # print("knwon terms ----------------------", col_terms)
                    for term in like_terms:
                        matches = find_closest_terms(term, col_terms)  # ✅ pass only term, not full query
                        if matches:
                            for m in matches:
                                m["column"] = col
                                suggestions.append(m)
                print("suggestions --------------------", suggestions)
                if suggestions:
                    if len(suggestions) == 1:
                        clarification_msg = (
                            f"I couldn’t find results in **{suggestions[0]['column']}** "
                            f"for **{suggestions[0]['original']}**. "
                            f"Did you mean **{suggestions[0]['term']}**?"
                        )
                    else:
                        suggestion_list = "\n".join(
                            [f"- {s['term']} (from {s['column']}, {s['score']}%)" for s in suggestions]
                        )
                        clarification_msg = (
                            f"I couldn’t find results. Did you mean one of these?\n{suggestion_list}"
                        )

                    save_chat_log(session_id, user_input, clarification_msg)
                    return clarification_msg, None, generated_sql

                
            # Only re-summarize if answer wasn't already provided by agent
            # if not final_answer or final_answer.strip() == raw_result.strip():
            else:
                mode = classify_response_mode(user_input)
                if mode == "list":
                    final_answer = list_answer(df, user_input)
                else:
                    final_answer = summarize_answer(df, user_input)

    except Exception as e:
        print(f"[ERROR] SQL Agent execution failed: {e}")
        final_answer = "⚠️ Sorry, I couldn't process that query."
        df = None

    # Clean the DataFrame for JSON serialization
    df_clean = clean_dataframe_for_json(df)

    # Log the SQL + Answer for debugging/analytics
    if session_id:
        log_text = f"[SQL]: {generated_sql if generated_sql else 'N/A'}\n[ANSWER]: {final_answer}"
        save_chat_log(session_id, user_input, log_text)
        
    return final_answer, df_clean, generated_sql