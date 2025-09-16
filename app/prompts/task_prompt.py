from langchain.prompts import ChatPromptTemplate

def get_task_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "Recognize user wants to end/reset. Reply politely, then ask for new question if needed."),
        ("human", "{input}")
    ])