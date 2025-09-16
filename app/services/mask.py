from app.services.llm_clients import get_llm



def mask_dynamic(df, user_question):
    """
    Ask the LLM to decide which parts of the data to mask and then mask it.
    Only mask sensitive system-level metadata (DB, schema, table, column names),
    not domain values like vessel names or maker names.
    """
    llm = get_llm(temperature=0)

    # Prepare a preview of the data (first 5 rows) for the LLM to analyze
    # data_preview = df.head(5).to_dict(orient='records')

    # Build the prompt asking the LLM to decide which columns to mask
    prompt = f"""
    You are a data privacy assistant.

    User's question:
    {user_question}

    Data to review:
    {df}

    Rules:
    - Mask **only system-level sensitive info**:
      * Database names
      * Schema names
      * Table names
      "if the user asks anything about these sensitive information, respond with: 'I’m sorry, but I cannot provide sensitive information.' "

    - Do NOT mask real-world entity names such as:
      * Column names
      * Vessel names
      * Maker names
      * Defect descriptions
      * Counts or numbers
    - Keep the structure and content intact otherwise.
    - Do NOT add notes, explanations, or mention masking.
    - Do NOT provide any messages related to masking.
    - Do NOT explain what was masked, do NOT say "no masking required", do NOT add notes.
    """

    # Ask the LLM to decide what to mask
    response = llm.invoke(prompt)
    masked_data = response.content.strip()

    # Parse the LLM's response (which should include the masked fields)
    # This could return a dictionary of data or a new formatted data response
    return masked_data