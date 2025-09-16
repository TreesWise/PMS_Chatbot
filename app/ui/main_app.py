import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import TypedDict, Optional, List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langgraph.graph import  END
import re
import numpy as np

# Import your custom modules
from app.agents.router_agent import classify_intent
from app.agents.query_generator_agent import run_query_generator
from app.agents.conversational_agent import run_conversational_agent
from app.agents.task_agent import run_task_agent
from app.agents.graph_agent import check_if_graph_needed
from app.services.chat_manager import save_chat_log
from app.services.mask import mask_dynamic

# --- LangGraph State Schema ---
class BotState(TypedDict):
    input: str
    session_id: str
    intent: str
    output: str
    generated_sql: str
    df: Optional[pd.DataFrame]
    needs_chart: bool
    graph_type: Optional[str]
    chart_url: Optional[str]

# --- LangGraph Node Definitions ---
def router_node(state: BotState) -> BotState:
    intent = classify_intent(state["input"])
    return {**state, "intent": intent}

def query_generator_node(state: BotState) -> dict:
    output, df , generated_sql= run_query_generator(state["input"], state["session_id"])
    return {**state, "output": output, "df": df, "generated_sql":generated_sql}

def conversational_node(state: BotState) -> BotState:
    output = run_conversational_agent(state["input"])
    return {**state, "output": output, "df": None}

def task_node(state: BotState) -> BotState:
    output = run_task_agent(state["input"])
    return {**state, "output": output, "df": None}

def mask_output_node(state: BotState) -> BotState:
    masked_output = mask_dynamic(state["output"], state["input"])
    return {**state, "output": masked_output}

def graph_agent_node(state: BotState) -> BotState:
    chart_config  = check_if_graph_needed(
        state["input"], state["output"], state.get("df")
    )

    chart_url = None
    df = state.get("df", None)

    # Print the type and value of df for debugging
    print(f"[DEBUG] Type of df: {type(df)}")
    print(f"[DEBUG] Value of df: {df}")

    # Convert to DataFrame if needed
    df = convert_to_dataframe(df)

    # Print the type and value after conversion
    print(f"[DEBUG] Type of df after conversion: {type(df)}")
    print(f"[DEBUG] Value of df after conversion: {df}")

    if df is not None and not df.empty:
        df = df.dropna()
        # Detect candidate numeric columns automatically
        for column in df.columns:
            # Try converting each column to numeric safely
            try:
                df[column] = pd.to_numeric(df[column])
            except Exception:
                pass

        # Find actual numeric columns after conversion
        numeric_cols = df.select_dtypes(include=["number"]).columns

        # Drop rows where ALL numeric columns are NaN
        if len(numeric_cols) > 0:
            df = df.dropna(subset=numeric_cols, how="all")

        print("df------------", df)


        # Proceed to generate the chart if needed
        if chart_config and chart_config.get("needs_chart"):
            top_n = extract_top_n_from_query(state["generated_sql"])
            chart_config_corrected = get_actual_columns(df, chart_config)
            chart_path = generate_chart(df, chart_config, state["session_id"], top_n=top_n)
            if chart_path:
                chart_url = f"/charts/{os.path.basename(chart_path)}"
    
    return {
        **state,
        "needs_chart": chart_config.get("needs_chart") if chart_config else False,
        "graph_type": chart_config.get("chart_type"),
        "chart_config": chart_config,
        "chart_url": chart_url
    }

import pandas as pd

def convert_to_dataframe(data: Any) -> pd.DataFrame:
    """
    Convert a list of dictionaries (or other structures) to a Pandas DataFrame.
    If the input is already a DataFrame, it will be returned as-is.
    """
    if isinstance(data, pd.DataFrame):
        return data  # Return DataFrame as-is if it's already a DataFrame
    elif isinstance(data, list):  # If the data is a list, convert to DataFrame
        try:
            # If data is a list of dictionaries, convert it to DataFrame
            return pd.DataFrame(data)
        except Exception as e:
            print(f"[ERROR] Failed to convert list to DataFrame: {e}")
            return None
    else:
        # If data is not list or DataFrame, return None or raise an error based on your use case
        print("[ERROR] Data is neither a list nor a DataFrame!")
        return None



def extract_top_n_from_query(query: str, default: int = 10) -> int:
    """Extracts a number like 'top 15' from query, otherwise returns default."""
    match = re.search(r"TOP\s+(\d+)", query, re.IGNORECASE)
    print("top n--------------------------------------------------------------------", match)
    if match:   
        return int(match.group(1))
    return default


def detect_plot_axes(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Auto-detect x (categorical or first column) and y (numeric) columns for plotting."""
    if df is None or df.empty:
        return None, None

    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Prefer categorical for x and numeric for y
    if categorical_cols and numeric_cols:
        return categorical_cols[0], numeric_cols[0]

    # If only numeric, use first as x, second as y (if exists)
    if len(numeric_cols) > 1:
        return numeric_cols[0], numeric_cols[1]
    elif numeric_cols:
        return None, numeric_cols[0]

    return None, None


def get_actual_columns(df: pd.DataFrame, chart_config: dict) -> dict:
    """
    Map LLM-provided column names to actual DataFrame columns (case-insensitive).
    """
    actual_config = chart_config.copy()
    df_cols_lower = {c.lower(): c for c in df.columns}

    # Map x_col
    actual_x = []
    for col in chart_config.get("x_col", []):
        col_lower = col.lower() if isinstance(col, str) else ""
        if col_lower in df_cols_lower:
            actual_x.append(df_cols_lower[col_lower])
        else:
            print(f"[WARN] x_col '{col}' not found in DataFrame, defaulting to first available column")
            actual_x.append(df.columns[0])  # Fallback to the first column (or another default)

    actual_config['x_col'] = actual_x

    # Map y_col
    actual_y = []
    for col in chart_config.get("y_col", []):
        col_lower = col.lower() if isinstance(col, str) else ""
        if col_lower in df_cols_lower:
            actual_y.append(df_cols_lower[col_lower])
        else:
            print(f"[WARN] y_col '{col}' not found in DataFrame, defaulting to first numeric column")
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                actual_y.append(numeric_cols[0])  # Fallback to the first numeric column

    actual_config['y_col'] = actual_y

    return actual_config



def generate_chart(df: pd.DataFrame, chart_config: Dict, session_id: str, top_n: int = 10) -> Optional[str]:
    """
    Generate chart automatically using LLM-provided chart_config.
    """
    try:
        if not chart_config or not chart_config.get("needs_chart"):
            return None

        graph_type = chart_config.get("chart_type")
        x_cols = chart_config.get("x_col", [])
        y_cols = chart_config.get("y_col", [])

        if not graph_type or not y_cols:
            print(f"[ERROR] Missing chart_type or y_col in chart_config")
            return None

        # Ensure that x_col and y_col are actual columns from the DataFrame
        # chart_config_corrected = get_actual_columns(df, chart_config)
        x_cols = chart_config.get("x_col", [])
        y_cols = chart_config.get("y_col", [])

        if not x_cols or not y_cols:
            print(f"[ERROR] Missing valid x_col or y_col after correction")
            return None

        # Extract the first x_col (support only one for now)
        x_col = x_cols[0] if isinstance(x_cols, list) else x_cols

        # y_cols can be multiple
        if isinstance(y_cols, str):
            y_cols = [y_cols]

        # If only one y_col, keep it as string for pandas plotting
        y_col = y_cols if len(y_cols) > 1 else y_cols[0]


        # Clean object columns
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].replace(
                re.compile(r"^\s*(none|null|nan|\s*)\s*$", flags=re.IGNORECASE),
                np.nan
            )
        df = df.dropna(subset=[c for c in [x_col] + y_cols if c is not None])
        
        
        
        # Handle Year column
        if x_col == 'Year':
            df[x_col] = pd.to_numeric(df[x_col], errors='coerce')

        if x_col.lower() == "year":
            df = df.sort_values(by=x_col, ascending=True).head(top_n)
        else:
            df = df.sort_values(by=y_cols[0], ascending=False).head(top_n)

        
        if df.empty:
            print("[ERROR] DataFrame is empty after cleaning")
            return None

        print("dataframe for generating chart ------------------------", df)
        print("x_col---------------------------------------",x_col)
        print("y_col---------------------------------------",y_col)
        # ---- Plotting ----
        plt.figure(figsize=(12, 6))
        if graph_type == "line":
            df.plot(x=x_col, y=y_col, kind="line", marker="o")
        elif graph_type == "bar":
            df.plot(x=x_col, y=y_col, kind="bar")
        elif graph_type == "histogram":
            df[y_cols[0]].plot(kind="hist", bins=20, alpha=0.7)
        elif graph_type == "scatter":
            if x_col and len(y_cols) == 1:
                df.plot(kind="scatter", x=x_col, y=y_cols[0])
            else:
                print("[ERROR] Scatter plot requires exactly one y_col")
                return None
        elif graph_type == "area":
            df.plot(x=x_col, y=y_col, kind="area")
        elif graph_type == "pie":
            if x_col and len(y_cols) == 1:
                df.set_index(x_col)[y_cols[0]].plot(kind="pie", autopct='%1.1f%%')
            else:
                print("[ERROR] Pie chart requires a single y_col")
                return None
        elif graph_type == "stacked_bar":
            df.plot(x=x_col, y=y_cols, kind="bar", stacked=True)
        else:
            df.plot(x=x_col, y=y_col, kind="line")  # fallback

        # ---- Labels ----
        plt.title(f"{graph_type.capitalize()} Chart (Top {top_n})")
        if x_col:
            plt.xlabel(x_col)
        plt.ylabel(", ".join(y_cols))
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # ---- Save ----
        charts_dir = "charts"
        os.makedirs(charts_dir, exist_ok=True)
        filename = f"{session_id}_{graph_type}_{uuid.uuid4().hex[:8]}.png"
        file_path = os.path.join(charts_dir, filename)

        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()
        return file_path

    except Exception as e:
        print(f"[ERROR] Failed to generate chart: {e}")
        return None


    
    

# def generate_chart(df: pd.DataFrame, chart_config: Dict, session_id: str, top_n: int = 10) -> Optional[str]:
#     """
#     Generate chart automatically using LLM-provided chart_config.
#     """
    
    
#     try:
#         if not chart_config or not chart_config.get("needs_chart"):
#             return None

#         graph_type = chart_config.get("chart_type")
#         x_cols = chart_config.get("x_col", [])
#         y_cols = chart_config.get("y_col", [])

#         if not graph_type or not y_cols:
#             print(f"[ERROR] Missing chart_type or y_col in chart_config")
#             return None

#         # Ensure that x_col and y_col are actual columns from the DataFrame
#         chart_config_corrected = get_actual_columns(df, chart_config)
#         x_cols = chart_config_corrected.get("x_col", [])
#         y_cols = chart_config_corrected.get("y_col", [])

#         if not x_cols or not y_cols:
#             print(f"[ERROR] Missing valid x_col or y_col after correction")
#             return None

#         # Extract the first column from x_cols and y_cols
#         x_col = x_cols[0] if x_cols else None
#         y_col = y_cols[0]


#         if x_col == 'Year':  # If 'Year' is the x-axis
#             df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
            
#         # If the x_col is 'Year', always sort by Year ascending
#         if x_col.lower() == "year":
#             df = df.sort_values(by=x_col, ascending=True).head(top_n)
#         else:
#             # For other cases, sort by y_col (top N)
#             df = df.sort_values(by=y_col, ascending=False).head(top_n)

            
#         # Clean object columns
#         for col in df.select_dtypes(include=["object"]).columns:
#             df[col] = df[col].replace(
#                 re.compile(r"^\s*(none|null|nan|\s*)\s*$", flags=re.IGNORECASE),
#                 np.nan
#             )
#         df = df.dropna(subset=[c for c in [x_col, y_col] if c is not None])
#         if df.empty:
#             print("[ERROR] DataFrame is empty after cleaning")
#             return None

#         # # If x_col exists, sort by y_col and take top_n
#         # if x_col:
#         #     df = df.sort_values(by=y_col, ascending=False).head(top_n)

#         # Generate chart as before
#         plt.figure(figsize=(12, 6))
#         if graph_type == "line":
#             df.plot(x=x_col, y=y_col, kind="line", marker="o")
#         elif graph_type == "bar":
#             df.plot(x=x_col, y=y_col, kind="bar")
#         elif graph_type == "histogram":
#             df[y_col].plot(kind="hist", bins=20, alpha=0.7)
#         elif graph_type == "scatter":
#             if x_col:
#                 df.plot(kind="scatter", x=x_col, y=y_col)
#             else:
#                 print("[ERROR] Scatter plot requires x_col")
#                 return None
#         elif graph_type == "area":
#             df.plot(x=x_col, y=y_cols, kind="area")
#         elif graph_type == "pie":
#             if x_col:
#                 df.set_index(x_col)[y_col].plot(kind="pie", autopct='%1.1f%%')
#             else:
#                 print("[ERROR] Pie chart requires x_col")
#                 return None
#         elif graph_type == "stacked_bar":
#             df.plot(x=x_col, y=y_cols, kind="bar", stacked=True)
#         else:
#             df.plot(x=x_col, y=y_col, kind="line")  # fallback

#         plt.title(f"{graph_type.capitalize()} Chart (Top {top_n})")
#         if x_col: plt.xlabel(x_col)
#         plt.ylabel(y_col)
#         plt.xticks(rotation=45, ha="right", fontsize=8)
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()

#         charts_dir = "charts"
#         os.makedirs(charts_dir, exist_ok=True)
#         filename = f"{session_id}_{graph_type}_{uuid.uuid4().hex[:8]}.png"
#         file_path = os.path.join(charts_dir, filename)

#         plt.savefig(file_path, dpi=300, bbox_inches="tight")
#         plt.close()
#         return file_path

#     except Exception as e:
#         print(f"[ERROR] Failed to generate chart: {e}")
#         return None



# def generate_chart(df: pd.DataFrame, chart_config: Dict, session_id: str, top_n: int = 10) -> Optional[str]:
#     """
#     Generate chart automatically using LLM-provided chart_config.
    
#     Args:
#         df: pandas DataFrame containing the data.
#         chart_config: dict containing 'needs_chart', 'chart_type', 'x_col', 'y_col'.
#         session_id: unique session ID for naming chart file.
#         top_n: number of top rows to display (if applicable).
    
#     Returns:
#         Path to saved chart image or None.
#     """
#     try:
#         if not chart_config or not chart_config.get("needs_chart"):
#             return None

#         graph_type = chart_config.get("chart_type")
        
#         print("graph type-----------------------------------------", graph_type)
#         x_cols = chart_config.get("x_col", [])
#         y_cols = chart_config.get("y_col", [])

        
        
#         # x_cols = get_actual_columns(df, x_cols)
#         # y_cols = get_actual_columns(df, y_cols)
        
        
#         if not graph_type or not y_cols:
#             print("[ERROR] Missing chart_type or y_col in chart_config")
#             return None

#         x_col = x_cols[0] if x_cols else None
#         y_col = y_cols[0]

#         # Clean object columns
#         for col in df.select_dtypes(include=["object"]).columns:
#             df[col] = df[col].replace(
#                 re.compile(r"^\s*(none|null|nan|\s*)\s*$", flags=re.IGNORECASE),
#                 np.nan
#             )
#         df = df.dropna(subset=[c for c in [x_col, y_col] if c is not None])
#         if df.empty:
#             print("[ERROR] DataFrame is empty after cleaning")
#             return None

#         # If x_col exists, sort by y_col and take top_n
#         if x_col:
#             df = df.sort_values(by=y_col, ascending=False).head(top_n)

#         plt.figure(figsize=(12, 6))

#         if graph_type == "line":
#             df.plot(x=x_col, y=y_col, kind="line", marker="o")
#         elif graph_type == "bar":
#             df.plot(x=x_col, y=y_col, kind="bar")
#         elif graph_type == "histogram":
#             df[y_col].plot(kind="hist", bins=20, alpha=0.7)
#         elif graph_type == "scatter":
#             if x_col:
#                 df.plot(kind="scatter", x=x_col, y=y_col)
#             else:
#                 print("[ERROR] Scatter plot requires x_col")
#                 return None
#         elif graph_type == "area":
#             df.plot(x=x_col, y=y_cols, kind="area")
#         elif graph_type == "pie":
#             if x_col:
#                 df.set_index(x_col)[y_col].plot(kind="pie", autopct='%1.1f%%')
#             else:
#                 print("[ERROR] Pie chart requires x_col")
#                 return None
#         elif graph_type == "stacked_bar":
#             df.plot(x=x_col, y=y_cols, kind="bar", stacked=True)
#         else:
#             df.plot(x=x_col, y=y_col, kind="line")  # fallback

#         plt.title(f"{graph_type.capitalize()} Chart (Top {top_n})")
#         if x_col: plt.xlabel(x_col)
#         plt.ylabel(y_col)
#         plt.xticks(rotation=45, ha="right", fontsize=8)
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()

#         charts_dir = "charts"
#         os.makedirs(charts_dir, exist_ok=True)
#         filename = f"{session_id}_{graph_type}_{uuid.uuid4().hex[:8]}.png"
#         file_path = os.path.join(charts_dir, filename)

#         plt.savefig(file_path, dpi=300, bbox_inches="tight")
#         plt.close()
#         return file_path

#     except Exception as e:
#         print(f"[ERROR] Failed to generate chart: {e}")
#         return None




# def generate_chart(df: pd.DataFrame, graph_type: str, session_id: str, top_n: int = 10) -> Optional[str]:
#     """Generate chart image and return file path."""
#     try:

#         for col in df.select_dtypes(include=["object"]).columns:
#             df[col] = df[col].replace(
#                 re.compile(r"^\s*(none|null|nan|\s*)\s*$", flags=re.IGNORECASE),
#                 np.nan
#             )
#         df = df.dropna(subset=df.select_dtypes(include=["object"]).columns)
#         if df.empty:
#             print("[ERROR] DataFrame is empty.")
#             return None

#         print(f"[DEBUG] DataFrame before plotting: {df.head(20)}")

#         # 🔹 Auto-detect axes
#         x_column, y_column = detect_plot_axes(df)
#         if not y_column:
#             print("[ERROR] Couldn't detect suitable y-axis column.")
#             return None
        
#         if x_column:
#             df = df.sort_values(by=y_column, ascending=False).head(top_n)


#         plt.figure(figsize=(12, 6))

#         if graph_type == "line":
#             df.plot(x=x_column, y=y_column, kind="line", marker="o")
#         elif graph_type == "bar":
#             df.plot(x=x_column, y=y_column, kind="bar")
#         elif graph_type == "histogram":
#             df[y_column].plot(kind="hist", bins=20, alpha=0.7)
#         else:
#             df.plot(x=x_column, y=y_column, kind="line")

#         plt.title(f"{graph_type.capitalize()} Chart (Top {top_n} + Others)")
#         if x_column: plt.xlabel(x_column)
#         plt.ylabel(y_column)
#         plt.xticks(rotation=45, ha="right", fontsize=8)
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()

#         charts_dir = "charts"
#         os.makedirs(charts_dir, exist_ok=True)
#         filename = f"{session_id}_{graph_type}_{uuid.uuid4().hex[:8]}.png"
#         file_path = os.path.join(charts_dir, filename)

#         plt.savefig(file_path, dpi=300, bbox_inches="tight")
#         plt.close()
#         return file_path

#     except Exception as e:
#         print(f"[ERROR] Failed to generate chart: {e}")
#         return None


# --- Define LangGraph Workflow ---
graph = StateGraph(BotState)
graph.add_node("router", RunnableLambda(router_node))
graph.add_node("query_generator", RunnableLambda(query_generator_node))
graph.add_node("conversational", RunnableLambda(conversational_node))
graph.add_node("task", RunnableLambda(task_node))
graph.add_node("mask", RunnableLambda(mask_output_node))
graph.add_node("graph_agent", RunnableLambda(graph_agent_node))

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda state: state["intent"],
    {
        "query_generator": "query_generator",
        "conversational": "conversational",
        "task": "task"
    }
)

graph.add_edge("query_generator", "mask")
graph.add_edge("conversational", "mask")
graph.add_edge("task", "mask")
graph.add_edge("mask", "graph_agent")
graph.add_edge("graph_agent", END)

graph_executor = graph.compile()

# --- FastAPI App ---
app = FastAPI(title="PMS Bot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files for charts
app.mount("/charts", StaticFiles(directory="charts"), name="charts")

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_input: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    needs_chart: bool
    graph_type: Optional[str]
    chart_url: Optional[str]
    data: Optional[List[Dict[str, Any]]]

@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    state = {
        "input": request.user_input,
        "session_id": session_id,
        "intent": "",
        "output": "",
        "df": None,
        "needs_chart": False,
        "graph_type": None,
        "chart_url": None
    }

    result = graph_executor.invoke(state)
    
    response_data = {
        "session_id": session_id,
        "response": result.get("output", ""),
        "needs_chart": result.get("needs_chart", False),
        "graph_type": result.get("graph_type"),
        "chart_url": result.get("chart_url")
        # "data": result.get("df").to_dict(orient="records") if result.get("df") is not None else None
    }

    # Save the chat log
    try:
        save_chat_log(session_id, request.user_input, response_data["response"])
    except Exception as e:
        print(f"[ERROR] Failed to log chat: {e}")

    return response_data

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "PMS Bot API is running"}

# Create charts directory on startup
@app.on_event("startup")
async def startup_event():
    os.makedirs("charts", exist_ok=True)
    print("Charts directory ready")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)