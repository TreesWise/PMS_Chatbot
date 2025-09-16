from app.services.llm_clients import get_llm
import pandas as pd
from app.services.mask import mask_dynamic



def list_answer(df, user_question):
    """
    Handle list-style queries and provide a preview or the entire list based on LLM's interpretation.
    LLM decides how to present the data: top N, full data, or summary in point-wise answers.
    """
    # Get the LLM instance
    llm = get_llm()
    masked_data = mask_dynamic(df, user_question)
    # # Group by vessels if the user requested grouping
    # if "group" in user_question.lower() and "vessel" in user_question.lower():
    #     df_grouped = df.groupby("VESSEL_NAME").agg({"DEFECT_COUNT": "sum"}).reset_index()
    # else:
    #     df_grouped = df

    # Prepare a preview of the data (first 5 rows) to pass to the LLM
    # data_preview = df_grouped.head(5).to_dict(orient='records')

    # Build the prompt asking the LLM how to display the data
    prompt = f"""
    You are a helpful assistant summarizing answers from SQL query results.

    User's question:
    {user_question}

    The result of the SQL query in a dataframe/table format:
    {masked_data }

    Your job is to:
        - Modify the given dataframe or table into a point-wise explanation.
        - For each row, convert the column values into a bulleted or numbered list format.
        - If the user question ({user_question}) explicitly asks for all the data, return all rows.
        - If the user mentions a specific number of rows, return only that many rows.
        - If the user does not mention anything about row count, return only the first 10 rows.
        - Do not modify or normalize the values. Return the exact values from the dataset as they appear, including any special characters, casing, or formatting.
    """

    # Ask the LLM how to display the data
    response = llm.invoke(prompt)
    display_instruction = response.content.strip()
    
    print("display instruction------------------",display_instruction)

    # Directly return the LLM's response as point-wise answers
    return display_instruction
