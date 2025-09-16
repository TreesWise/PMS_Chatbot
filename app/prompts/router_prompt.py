from langchain.prompts import ChatPromptTemplate

def get_router_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """
You are a router deciding the user's intent.

Possible intents:
- query_generator: question to fetch/analyze data
- conversational: small talk / greeting
- task: reset chat, start over, end chat

If user asks a follow-up like "make it 50", "show me more", "filter by vessel" or related commands
and the previous context was dataset query → classify as query_generator.

Return only one of: query_generator, conversational, task.
"""),
        ("human", "{input}")
    ])