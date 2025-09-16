from app.agents.new_query_generator_agent import generate_new_query
from app.agents.followup_query_generator_agent import rewrite_and_generate_query
from app.agents.query_type_classifier_agent import classify_query_type

def run_query_generator(user_input, session_id):
    mode = classify_query_type(user_input, session_id)
    if mode == "new":
        output, df, generated_sql = generate_new_query(user_input, session_id)
        return output, df, generated_sql
    else:
        output, df, generated_sql = rewrite_and_generate_query(user_input, session_id)
        return output, df, generated_sql


# from app.services.fuzzy_term_matcher import find_closest_term  # new helper you’ll create
# from app.services.chat_manager import save_chat_log

# def run_query_generator(user_input, session_id):
#     """
#     Run query generator, but intercept user input first:
#     - If spelling mistakes detected (fuzzy match <-> known terms), ask clarification.
#     - Else, proceed normally with new or follow-up query generation.
#     """
#     # Step 1: Check for fuzzy spelling issues
#     corrected_term = find_closest_term(user_input)

#     if corrected_term and corrected_term["score"] >= 85:
#         # If the match is strong but not exact → clarify with user
#         if corrected_term["term"].lower() not in user_input.lower():
#             clarification_msg = f"I couldn’t find results for **{corrected_term['original']}**. Did you mean **{corrected_term['term']}**?"
#             # Save clarification in chat log
#             save_chat_log(session_id, user_input, clarification_msg)
#             # Return clarification instead of SQL/data
#             return clarification_msg, None

#     # Step 2: Decide between new or follow-up query
#     mode = classify_query_type(user_input, session_id)
#     if mode == "new":
#         output, df = generate_new_query(user_input, session_id)
#         return output, df
#     else:
#         output, df = rewrite_and_generate_query(user_input, session_id)
#         return output, df
