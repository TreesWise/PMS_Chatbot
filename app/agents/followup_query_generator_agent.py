from app.services.llm_clients import get_llm
from app.prompts.followup_query_prompt import get_followup_query_prompt
from app.services.chat_manager import get_recent_history
from app.agents.new_query_generator_agent import generate_new_query

def rewrite_and_generate_query(user_input, session_id):
    llm = get_llm()
    prompt = get_followup_query_prompt()

    # Fetch last 3 Q&A pairs
    history = get_recent_history(session_id, limit=3)

    # Format as readable text
    formatted_history = "\n".join(
        f"User: {q}\nAssistant: {a}" for q, a in history
    )

    # Create the runnable chain
    chain = prompt | llm

    # Invoke chain with current follow-up question & recent history
    response = chain.invoke({
        "input": user_input,
        "history": formatted_history
    })

    # Extract generated text safely
    rewritten_text = response.content.strip()
    print(f"[DEBUG] Rewritten question: {rewritten_text}")

    # Generate fresh SQL from rewritten question
    return generate_new_query(rewritten_text, session_id)
