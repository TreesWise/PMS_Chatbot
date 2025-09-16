# from app.services.llm_clients import get_llm
# from typing import Tuple
# import pandas as pd
# from langchain_core.messages import HumanMessage

# def check_if_graph_needed(query: str, answer: str, df: pd.DataFrame | None = None) -> Tuple[bool, str | None]:
#     """
#     Uses query, answer, and optional DataFrame to decide if a chart is needed.
#     Passes all the context to the LLM to decide intelligently which type of graph is needed.
#     """
#     # Build context for the LLM
#     df_preview = None
#     if isinstance(df, pd.DataFrame):
#         if not df.empty:
#             df_preview = df.head(20).to_dict(orient="records")

#     context = f"""
#     User Question: {query}
# Answer: {answer}
# Data Preview: {df_preview}

# Based on the above information, determine whether a chart is required to better understand the data. If yes, provide the most appropriate chart type. Consider the following:

# - Use a **line chart** when the data shows a trend or relationship over time.
# - Use a **bar chart** when comparing categorical data or discrete values.
# - Use a **histogram** to show the distribution of a continuous variable.
# - If the data involves multiple variables and you're comparing them, **stacked bar charts** or **scatter plots** may be useful.
# - If the query asks for relationships or correlations, **scatter plots** or **bubble charts** might be needed.
# - If the data shows time-based analysis, **line charts** or **area charts** might be appropriate.
# - If the data includes a frequency distribution or clusters, **histograms** or **pie charts** might apply.

# Please respond with "True" or "False" for whether a chart is needed and the type of chart ("line", "bar", "histogram", etc.). Example: "True, line" or "False". Do not explain, just give the answer.
#     """

#     # Call the LLM with context
#     llm = get_llm(temperature=0)

#     # Proper LLM call
#     response = llm.invoke([HumanMessage(content=context)])
#     response = response.content.strip().lower() 
    
#     print("response------------------------------------------",response)

#     if "true" in response:
#         # Extract chart type (line, bar, histogram, etc.)
#         if "line" in response:
#             return True, "line"
#         elif "bar" in response:
#             return True, "bar"
#         elif "histogram" in response:
#             return True, "histogram"
#         else:
#             # If the LLM provides an unrecognized chart type, default to "line"
#             return True, "line"
#     elif "false" in response:
#         return False, None
#     else:
#         # If the response is invalid, assume no graph is needed
#         return False, None




# from app.services.llm_clients import get_llm
# from typing import Tuple
# import pandas as pd
# from langchain.prompts import PromptTemplate
# from langchain_core.messages import HumanMessage


# # ----------------------------
# # 1. Define the PromptTemplate
# # ----------------------------
# chart_prompt_template = PromptTemplate(
#     input_variables=["query", "answer", "df_preview"],
#     template="""
# You are an assistant that decides whether a chart is needed to better visualize a dataset.

# Input:
# User Question: {query}
# Answer: {answer}
# Data Preview: {df_preview}

# Task:
# Based on the above, determine if a chart is required. If yes, suggest the most appropriate chart configuration.

# Guidelines:
# - Use a **line chart** when the data shows a trend or relationship over time (x-axis usually a date or year).
# - Use a **bar chart** when comparing categorical data or discrete values (x-axis = category, y-axis = count/value).
# - Use a **histogram** to show the distribution of a continuous variable (x-axis = variable, y-axis = frequency).
# - If the data involves multiple variables and you're comparing them, **stacked bar charts** or **scatter plots** may be useful.
# - If the query asks for relationships or correlations, **scatter plots** or **bubble charts** might be needed.
# - If the data shows time-based analysis, **line charts** or **area charts** might be appropriate.
# - If the data includes a frequency distribution or clusters, **histograms** or **pie charts** might apply.
# - If comparing values, one axis may contain multiple columns (provide them as a list).

# Output:
# Respond strictly in JSON format with the following keys:
# {{
#   "needs_chart": true/false,
#   "chart_type": "line" | "bar" | "histogram" | "scatter" | "pie" | "area" | "stacked_bar" | null,
#   "x_col": ["column_name1", "column_name2", ...],
#   "y_col": ["column_name1", "column_name2", ...]
# }}

# Examples:
# {{
#   "needs_chart": true,
#   "chart_type": "line",
#   "x_col": ["Year"],
#   "y_col": ["Defect_Count"]
# }}

# {{
#   "needs_chart": true,
#   "chart_type": "bar",
#   "x_col": ["Maker"],
#   "y_col": ["Job_Count", "Closed_Jobs"]
# }}

# {{
#   "needs_chart": false,
#   "chart_type": null,
#   "x_col": [],
#   "y_col": []
# }}
# """
# )


# # ----------------------------
# # 2. Graph agent function
# # ----------------------------
# def check_if_graph_needed(query: str, answer: str, df: pd.DataFrame | None = None) -> Tuple[bool, str | None]:
#     """
#     Uses query, answer, and optional DataFrame to decide if a chart is needed.
#     Passes all the context to the LLM to decide intelligently which type of graph is needed.
#     """
#     df_preview = None
#     if isinstance(df, pd.DataFrame):
#         if not df.empty:
#             df_preview = df.head(20).to_dict(orient="records")

#     # Format prompt using LangChain
#     formatted_prompt = chart_prompt_template.format(
#         query=query,
#         answer=answer,
#         df_preview=df_preview
#     )

#     # Call the LLM
#     llm = get_llm(temperature=0)
#     response = llm.invoke([HumanMessage(content=formatted_prompt)])
#     response = response.content.strip().lower()

#     print("response------------------------------------------", response)

#     # Naive parsing (you may want to use json.loads instead)
#     if "true" in response:
#         if "line" in response:
#             return True, "line"
#         elif "bar" in response:
#             return True, "bar"
#         elif "histogram" in response:
#             return True, "histogram"
#         elif "scatter" in response:
#             return True, "scatter"
#         elif "pie" in response:
#             return True, "pie"
#         elif "area" in response:
#             return True, "area"
#         elif "stacked_bar" in response:
#             return True, "stacked_bar"
#         else:
#             return True, "line"  # default fallback
#     elif "false" in response:
#         return False, None
#     else:
#         return False, None





from app.services.llm_clients import get_llm
from typing import Dict, Optional
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import json
import re


# ----------------------------
# 1. Define the PromptTemplate
# ----------------------------
chart_prompt_template = PromptTemplate(
    input_variables=["query", "answer", "df_preview"],
    template="""
You are an assistant that decides whether a chart is needed to better visualize a dataset.

Input:
User Question: {query}
Answer: {answer}
Data Preview: {df_preview}
Available Columns: {columns}


Task:
Based on the above, determine if a chart is required. If yes, suggest the most appropriate chart configuration.

Guidelines:
- IMPORTANT: Only use column names from **Available Columns** exactly as they appear (case-sensitive). Do not invent or rename columns (e.g., if "Valve_Defects" exists, do not output "Defect_Count"). 
- **Never use the same column for both x_col and y_col**, except in histograms where the same numeric column may be used.  
- Use a **line chart** when the data shows a trend or relationship over time (x-axis usually a date or year).
- Use a **bar chart** for categorical comparisons (x-axis = category, y-axis = value).
- Use a **histogram** to show the distribution of a continuous variable (x-axis = variable, y-axis = frequency).
- Use a **scatter plot** for relationships or correlations (both x and y are numerical).
- If the data involves multiple variables and you're comparing them, **stacked bar charts** or **scatter plots** may be useful.
- If the query asks for relationships or correlations, **scatter plots** or **bubble charts** might be needed.
- If the data shows time-based analysis, **line charts** or **area charts** might be appropriate.
- If the data includes a frequency distribution or clusters, **histograms** or **pie charts** might apply.
- If comparing values, one axis may contain multiple columns (provide them as a list).
- Suggest **appropriate axes** for the chosen chart type. For example, if using a **line chart**, use a date on the x-axis and a numerical variable on the y-axis.
- For histogram charts, use the **numeric column name** as y_col, do not invent 'Frequency'.
- Please respond with "True" or "False" for whether a chart is needed and the chart_type ("line", "bar", "histogram", etc.).

Output:
Respond strictly in JSON format with the following keys:
{{
  "needs_chart": true/false,
  "chart_type": "line" | "bar" | "histogram" | "scatter" | "pie" | "area" | "stacked_bar" | null,
  "x_col": ["column_name1", "column_name2", ...],
  "y_col": ["column_name1", "column_name2", ...]
}}

Examples:
{{
  "needs_chart": true,
  "chart_type": "line",
  "x_col": ["Year"],
  "y_col": ["Defect_Count"]
}}

{{
  "needs_chart": true,
  "chart_type": "bar",
  "x_col": ["Maker"],
  "y_col": ["Job_Count", "Closed_Jobs"]
}}

{{
  "needs_chart": false,
  "chart_type": null,
  "x_col": [],
  "y_col": []
}}
"""
)

# chart_prompt_template = PromptTemplate(
#     input_variables=["query", "answer", "df_preview", "columns"],
#     template="""
# You are an assistant that decides whether a chart is needed to better visualize a dataset.

# Input:
# User Question: {query}
# Answer: {answer}
# Data Preview: {df_preview}
# Available Columns: {columns}

# Task:
# Decide if a chart is needed. If yes, suggest the most appropriate chart configuration.  
# ⚠️ IMPORTANT: Only use column names from **Available Columns** exactly as they appear (case-sensitive).  
# ⚠️ Do not invent or rename columns (e.g., if "Valve_Defects" exists, do not output "Defect_Count").  
# ⚠️ Do not assign the same column for both `x_col` and `y_col` unless chart_type is "histogram".  

# Respond strictly in JSON with the following keys:  
# {{
#   "needs_chart": true/false,
#   "chart_type": "line" | "bar" | "histogram" | "scatter" | "pie" | "area" | "stacked_bar" | null,
#   "x_col": ["column_name1", "column_name2", ...],
#   "y_col": ["column_name1", "column_name2", ...]
# }}

# --------------------------
# Rules for deciding `needs_chart`:
# - Output **true** if the question/answer would be easier to interpret visually (e.g., trends, comparisons, distributions, correlations).  
# - Output **false** if the answer is simple, direct (single number or short text), or does not benefit from visualization.  
# - Only set true if **at least one numeric column** or **categorical distribution** is relevant.  
# - If no valid chart type can be applied, return false.  

# Rules for deciding `chart_type`:
# - Use **line chart** when analyzing trends over time (x-axis = date, year, or similar).  
# - Use **bar chart** for categorical comparisons (x-axis = category, y-axis = count/measure).  
# - Use **histogram** to show distribution of a single numeric variable.  
# - Use **scatter plot** for correlation or relationship between two numeric columns.  
# - Use **pie chart** only for proportion breakdowns of categories.  
# - Use **area chart** for cumulative or stacked trends over time.  
# - Use **stacked bar chart** for comparing multiple numeric columns across categories.  

# Rules for selecting `x_col`:
# - Always choose from the **Available Columns** only (no invented names).
# - For time-series, use a date/year column.  
# - For comparisons, use categorical columns (e.g., VESSEL_NAME, MAKER, JOB_STATUS).  
# - For histograms, x_col should be the numeric variable itself.  
# - For scatter plots, choose the independent variable(s) on the x-axis.  

# Rules for selecting `y_col`:
# - Always choose from the **Available Columns** only (no invented names).  
# - **Never use the same column for both x_col and y_col**, except in histograms where the same numeric column may be used.  

# --------------------------
# Examples:
# {{
#   "needs_chart": true,
#   "chart_type": "line",
#   "x_col": ["Year"],
#   "y_col": ["Defect_Count"]
# }}

# {{
#   "needs_chart": true,
#   "chart_type": "bar",
#   "x_col": ["Maker"],
#   "y_col": ["Job_Count", "Closed_Jobs"]
# }}

# {{
#   "needs_chart": false,
#   "chart_type": null,
#   "x_col": [],
#   "y_col": []
# }}
# """
# )

# ----------------------------
# 2. Graph agent function
# ----------------------------
def check_if_graph_needed(query: str, answer: str, df: pd.DataFrame | None = None) -> Optional[Dict]:
    """
    Uses query, answer, and optional DataFrame to decide if a chart is needed.
    Returns the full parsed JSON from the LLM, including x_col and y_col.
    """
    df_preview = None
    df = pd.DataFrame(df)
    print("type-------",type(df))
    print("df_columns----------------------------------------------------------------",df)
    columns = []
    if isinstance(df, pd.DataFrame) and not df.empty:
        df_preview = df.head(20).to_dict(orient="records")
        columns = df.columns.tolist()
    print("df_columns----------------------------------------------------------------",columns)
    # Format prompt using LangChain
    formatted_prompt = chart_prompt_template.format(
        query=query,
        answer=answer,
        df_preview=df_preview,
        columns=columns
    )

    # Call the LLM
    llm = get_llm(temperature=0)
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    response_text = response.content.strip()

    # Try to parse JSON safely
    try:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            return None

        chart_config = json.loads(match.group())

        # Ensure all expected keys exist
        for key in ["needs_chart", "chart_type", "x_col", "y_col"]:
            if key not in chart_config:
                chart_config[key] = None if key != "needs_chart" else False
        print("chart config----------------------------------------------------------------", chart_config)
        return chart_config

    except Exception as e:
        print("Error parsing chart JSON:", e)
        return None
