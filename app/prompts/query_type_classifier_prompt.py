from langchain.prompts import ChatPromptTemplate

def get_query_type_classifier_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """
You are a query type classifier.

You will receive:
- The last few user questions
- The current user question

Your task:
- Decide whether the current question is:
  - a completely new, unrelated question → output: new
  - a follow-up related to previous questions → output: followup

Rules:
- Respond ONLY with the single word: "new" or "followup".
- No explanation, no markdown, no comments.
"""),
        ("human", "Previous questions:\n{history}\n\nCurrent question: {input}")
    ])
