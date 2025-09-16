from app.services.llm_clients import get_llm
from app.prompts.response_mode_prompt import get_response_mode_prompt

def classify_response_mode(user_question):
    """
    Returns either: 'summary' or 'list'
    """
    llm = get_llm()
    prompt = get_response_mode_prompt()
    chain = prompt | llm

    response = chain.invoke({"input": user_question})
    mode = response.content.strip().lower()
    print(f"[DEBUG] Response mode classified: {mode}")
    return mode



