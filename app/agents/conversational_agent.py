from app.services.llm_clients import get_llm
from langchain.chains import LLMChain
from app.prompts.conversation_prompt import get_conversation_prompt

def run_conversational_agent(user_input):
    llm = get_llm()
    prompt = get_conversation_prompt()
    chain = prompt | llm
    
    response = chain.invoke({"input": user_input})

    # Extract text cleanly
    reply = response.content.strip() if hasattr(response, "content") else response["text"].strip()
    return reply