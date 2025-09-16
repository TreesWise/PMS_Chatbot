from app.services.llm_clients import get_llm
from app.prompts.router_prompt import get_router_prompt

def classify_intent(user_input):
    llm = get_llm()
    prompt = get_router_prompt()
    # response = llm(prompt.format_prompt(input=user_input).to_messages())
    response = llm.invoke(prompt.format_prompt(input=user_input).to_messages())
    text = response.content.strip().lower()
    print(f"[DEBUG] Router classified intent: {text}")  # add this for debugging
    return text
