from app.services.llm_clients import get_llm
from app.services.mask import mask_dynamic
def summarize_answer(df, user_question):
    """
    Summarize the dataframe in natural language, tailored to the user's question.
    The LLM should not modify or normalize any values.
    """
    llm = get_llm()

    # Use only preview (e.g., first 5 rows) to avoid huge prompt
    data_preview = df.head(5).to_dict(orient='records')
    row_count = len(df)
    
    masked_data = mask_dynamic(df, user_question)   


    # Build prompt
    prompt = f"""
You are an expert data assistant.

User's question:
{user_question}

Here is a preview of the data:
{masked_data}

Total number of rows returned: {row_count}

Task:
- Provide a short, clear, human-like answer to the user's question.
- Do not modify or normalize the values. Return the exact values from the dataset as they appear, including any special characters, casing, or formatting.
- Summarize insights or findings, mention counts or highlights if relevant.
- Do NOT include raw tables or markdown.
- Answer naturally, as if explaining to a colleague, but **use the exact wording** and **formatting** from the dataset.
"""

    response = llm.invoke(prompt)
    return response.content.strip()






# from services.llm_clients import get_llm

# def summarize_answer(df, user_question):
#     """
#     Summarize the dataframe in bullet points, masking column names.
#     """
#     llm = get_llm()

#     # Use only preview (e.g., first 5 rows) to avoid huge prompt
#     data_preview = df.head(5).to_dict(orient='records')
#     row_count = len(df)

#     # Build prompt
#     prompt = f"""
# You are an expert data assistant.

# User's question:
# {user_question}

# Here is a preview of the data (first 5 rows):
# {data_preview}

# Total number of rows returned: {row_count}

# Task:
# - Summarize the data and answer the question using concise bullet points.
# - Replace actual column names with generic placeholders like "Column A", "Column B", etc.
# - Do NOT include raw tables, markdown, or actual column names.
# - Provide clear, human-friendly insights as if explaining to a colleague.
# """

#     response = llm.invoke(prompt)
#     return response.content.strip()







# from services.llm_clients import get_llm

# def summarize_answer(df, user_question):
#     """
#     Summarize the dataframe in bullet points, masking column names and avoiding tabular output.
#     """
#     llm = get_llm()

#     # Step 1: Mask columns
#     column_mapping = {col: f"Column {chr(65 + i)}" for i, col in enumerate(df.columns)}
#     masked_df = df.rename(columns=column_mapping)

#     # Step 2: Preview as descriptive entries, not JSON or table
#     preview_rows = masked_df.head(5).to_dict(orient="records")
#     preview_description = "\n".join([
#         f"- Row {i+1}: " + ", ".join(f"{k}: {v}" for k, v in row.items())
#         for i, row in enumerate(preview_rows)
#     ])
#     row_count = len(df)

#     # Step 3: Prompt
#     prompt = f"""
# You are an expert data assistant.

# User's question:
# {user_question}

# Here is a brief preview of the data with column names masked:
# {preview_description}

# Total number of rows: {row_count}

# Task:
# - Answer the user's question in clear, concise bullet points.
# - Do NOT show tables or JSON.
# - Use "Column A", "Column B", etc., instead of real column names.
# - Present insights like you're speaking to a non-technical teammate.
# """

#     response = llm.invoke(prompt)
#     return response.content.strip()
