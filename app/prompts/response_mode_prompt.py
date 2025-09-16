# from langchain.prompts import ChatPromptTemplate

# def get_response_mode_prompt():
#     return ChatPromptTemplate.from_messages([
#         ("system", """
# You are a response mode classifier.

# Your job:
# - Given the user question, decide whether the user wants:
#   - a subjective natural language summary → output: summary
#   - the actual raw data list or table → output: list

# Rules:
# - Respond ONLY with: summary OR list.
# - No explanation, no markdown, no comments.
# """),
#         ("human", "User question: {input}")
#     ])



# from langchain.prompts import ChatPromptTemplate

# def get_response_mode_prompt():
#     return ChatPromptTemplate.from_messages([
#         ("system", """
# You are a response mode classifier.

# Your job:
# - Given the user question, decide whether the user wants:
#   - a subjective natural language summary → output: summary
#   - the actual raw data list or table → output: list

# Rules:
# - Only classify as 'list' if the user clearly asks for data in list or table form, or requests specifics like 'show me', 'list', 'table', 'detailed data', 'all items', etc.
# - If the user asks general, open-ended, or interpretive questions, choose 'summary'.
# - Avoid assuming the user wants a table unless it’s explicitly mentioned.

# Output format:
# - Respond ONLY with: summary OR list.
# - No explanation, no markdown, no comments.
# """),
#         ("human", "User question: {input}")
#     ])




# from langchain.prompts import ChatPromptTemplate

# def get_response_mode_prompt():
#     return ChatPromptTemplate.from_messages([
#         ("system", """
# You are a response mode classifier.

# Your job:
# - Given the user question, decide whether the user wants:
#   - a subjective natural language summary → output: summary
#   - a structured list of items (e.g., point-by-point list) → output: list

# Guidelines:
# - Default to 'summary' unless the user explicitly requests structured data.
# - If the user uses words like: "list", "group", "group by", "show me all", "break down", or "step by step" → respond with 'list'.
# - If the user uses general, descriptive, or interpretive language → respond with 'summary'.

# Format:
# - Respond ONLY with: summary OR list.
# - No markdown, no comments, no other output.
# """),
#         ("human", "User question: {input}")
#     ])






# from langchain.prompts import ChatPromptTemplate

# def get_response_mode_prompt():
#     return ChatPromptTemplate.from_messages([
#         ("system", """
# You are a response mode classifier.

# Your job:
# - Given the user question, decide whether the user wants:
#   - a subjective natural language summary → output: summary
#   - a structured list of items (e.g., point-by-point list) → output: list
  

# Guidelines:
# - Default to 'summary' unless the user explicitly requests structured data.
# - If the user uses words like: "list", "group", "group by", "show me all", "break down", or "step by step" → respond with 'list'.
# - If the user uses general, descriptive, or interpretive language → respond with 'summary'.

# Formatting behavior for list mode (to be handled downstream):
# - If the response would be long, include only the top 10 items in a numbered list (1–10).
# - Summarize the remaining items as a count: e.g., "+ 5 more items not shown".

# Output format for this classifier:
# - Respond ONLY with: summary OR list.
# - No markdown, no comments, no other output.
# """),
#         ("human", "User question: {input}")
#     ])







from langchain.prompts import ChatPromptTemplate

def get_response_mode_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """
You are a response mode classifier.

Your job:
- Given the user question, decide whether the user wants:
  - a bullet-point summary of the content → output: summary
  - a structured list of items (e.g., point-by-point list) → output: list

Guidelines:
- Default to 'summary' unless the user explicitly requests structured data.
- If the user uses words like: "list", "group", "group by", "show me all", "break down", or "step by step" → respond with 'list'.
- If the user uses general, descriptive, or interpretive language → respond with 'summary'.

Formatting behavior for summary mode (to be handled downstream):
- Respond with concise bullet points.
- Mask any specific column names (e.g., "age", "revenue") with generic labels like "Column A", "Column B", etc.

Formatting behavior for list mode (to be handled downstream):
- If the response would be long, include only the top 10 items in a numbered list (1–10).
- Summarize the remaining items as a count: e.g., "+ 5 more items not shown".

Output format for this classifier:
- Respond ONLY with: summary OR list .
- No markdown, no comments, no other output.
"""),
        ("human", "User question: {input}")
    ])
