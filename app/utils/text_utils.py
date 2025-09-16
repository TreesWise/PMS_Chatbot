def mask_columns(text: str) -> str:
    return text  # Add masking logic here if needed


# def extract_sql_from_response(text: str) -> str:
#     """
#     Remove markdown code fences (```sql ... ```) or plain ``` ... ```.
#     Keep only raw SQL string to execute safely.
#     """
#     text = text.strip()

#     # Check for triple backticks anywhere
#     if text.startswith("```") and text.endswith("```"):
#         # Split into lines and remove first and last
#         lines = text.splitlines()

#         # Handle case like: ```sql (first line)
#         if lines[0].strip().startswith("```"):
#             lines = lines[1:]
#         if lines and lines[-1].strip() == "```":
#             lines = lines[:-1]

#         text = "\n".join(lines)

#     return text.strip()
