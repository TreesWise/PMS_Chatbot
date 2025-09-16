from app.services.llm_clients import get_llm
from app.prompts.query_type_classifier_prompt import get_query_type_classifier_prompt
from app.services.chat_manager import get_recent_history

def classify_query_type(user_input, session_id):
    llm = get_llm()
    prompt = get_query_type_classifier_prompt()
    chain = prompt | llm

    # Get last few user questions only (skip answers)
    history = get_recent_history(session_id, limit=3)
    # previous_questions = "\n".join(q for q, a in history)

    response = chain.invoke({
        "input": user_input,
        "history": history
    })

    query_type = response.content.strip().lower()
    print(f"[DEBUG] Classified query type: {query_type}")
    return query_type
