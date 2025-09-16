from langchain.prompts import ChatPromptTemplate

def get_conversation_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are a friendly conversational assistant."),
        ("human", "{input}")
    ])