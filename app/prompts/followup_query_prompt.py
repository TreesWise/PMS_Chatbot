from langchain.prompts import ChatPromptTemplate

def get_followup_query_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """
You are an expert query rewriter.

You will receive:
- Chat history (previous questions and answers)
- A new follow-up question

Your task:
- Produce a **single, full, standalone natural language question** that includes relevant context.
- Combine details if needed.
- Respond ONLY with the rewritten question. No explanation, no markdown.

⚠️ Important: Some names may contain special characters like <, >, &, etc.
- NEVER change or normalize these values.
- When generating SQL or rewriting queries, always use the exact original value from metadata or database, including special chars and casing.
"""),
        ("human", "Chat history:\n{history}\n\nFollow-up question: {input}")
    ])


# from langchain.prompts import ChatPromptTemplate

# def get_followup_query_prompt():
#     """
#     Prompt to rewrite user follow-up questions into full natural language questions.
#     """
#     system_text = """
# You are a smart query rewriter.

# Your job:
# - Combine the last 3 user queries & answers into a single, complete natural language question.
# - Keep context (e.g., if user asked 'make it 50' after asking for 20 least affected vessels, rewrite to:
#   "Provide the list of 50 least affected vessels by valve defects.")

# - Do NOT generate SQL here.
# - Output must be a clean, single natural language question.
# """
#     return ChatPromptTemplate.from_messages([
#         ("system", system_text),
#         ("human", "{input}\nContext:\n{history}")
#     ])
