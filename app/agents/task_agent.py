from app.services.llm_clients import get_llm
from langchain.chains import LLMChain
from app.prompts.task_prompt import get_task_prompt
from app.agents.new_query_generator_agent import generate_new_query

def run_task_agent(user_input):
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=get_task_prompt())
    reply = chain.invoke({"input": user_input})
    fresh_data = generate_new_query("Show latest defects")
    return f"{reply}\n\n{fresh_data}"