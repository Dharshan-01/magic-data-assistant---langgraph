# main.py

import os
import io
import re
import json
import pandas as pd
import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, List, Dict
from pydantic import BaseModel, Field
# This is crucial for Python < 3.12 to work with LangGraph
from typing_extensions import TypedDict

# --- Langchain & LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
# --- CORRECTED IMPORT: Using Google Generative AI ---
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from the .env file
load_dotenv()

# --- Initialize Google Gemini Client ---
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        print("Warning: GOOGLE_API_KEY not found. Chat will not work.")
except Exception as e:
    print(f"Error initializing API client: {e}")

# Create the FastAPI application instance
app = FastAPI(
    title="Magic Data Assistant V2 API",
    description="An API powered by a LangGraph Agent with Google Gemini to chat with multiple tables.",
    version="27.0.0" # Excel Upload Version
)

# --- CORS Middleware ---
origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    table_names: List[str]
    question: str

# --- Helper Functions ---
def pandas_to_sql_type(dtype):
    if "int" in dtype.name: return "BIGINT"
    elif "float" in dtype.name: return "FLOAT"
    elif "datetime" in dtype.name: return "TIMESTAMP"
    else: return "TEXT"

def process_dataframe_to_db(df: pd.DataFrame, table_name: str, db_url: str):
    """Helper function to create a table and insert data from a DataFrame."""
    with psycopg.connect(db_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            columns_with_types = [f'"{col_name}" {pandas_to_sql_type(dtype)}' for col_name, dtype in df.dtypes.items()]
            create_query = (
                f'DROP TABLE IF EXISTS "{table_name}";\n'
                f'CREATE TABLE "{table_name}" (id SERIAL PRIMARY KEY, {", ".join(columns_with_types)});'
            )
            cur.execute(create_query)
            
            # Use StringIO for efficient in-memory CSV handling for the COPY command
            output = io.StringIO()
            df.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N') # Handle NULLs correctly
            output.seek(0)
            column_names_for_copy = ", ".join([f'"{col}"' for col in df.columns])
            with cur.copy(f'COPY "{table_name}" ({column_names_for_copy}) FROM STDIN') as copy:
                while data := output.read(8192): # Read in chunks
                    copy.write(data)

# --- LangGraph Agent Implementation (Unchanged) ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    db_url: str
# ... (rest of the LangGraph implementation is the same) ...
@tool
def run_sql_query(sql_query: str = Field(description="The PostgreSQL query to be executed.")):
    """A tool to execute a SQL query against the database."""
    pass 
def agent_node(state: AgentState):
    print("---CALLING AGENT---")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}
def tool_node(state: AgentState):
    print("---EXECUTING TOOL---")
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    if not tool_calls:
        raise ValueError("The agent did not generate a tool call.")
    sql_query = tool_calls[0]["args"]["sql_query"]
    db_url = state["db_url"]
    try:
        with psycopg.connect(db_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query)
                query_upper = sql_query.strip().upper()
                if query_upper.startswith("SELECT"):
                    columns = [desc[0] for desc in cur.description]
                    results = [dict(zip(columns, row)) for row in cur.fetchall()]
                    tool_result = json.dumps(results, default=str)
                else:
                    rows_affected = cur.rowcount
                    tool_result = f"Query executed successfully. {rows_affected} rows were affected."
        return {"messages": [ToolMessage(content=tool_result, tool_call_id=tool_calls[0]["id"])]}
    except Exception as e:
        error_message = f"Error executing SQL: {e}"
        return {"messages": [ToolMessage(content=error_message, tool_call_id=tool_calls[0]["id"])]}
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "continue"
    else:
        return "end"
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("action", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
workflow.add_edge("action", "agent") 
graph = workflow.compile()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
llm_with_tools = llm.bind_tools([run_sql_query])


# --- API Endpoints ---

@app.get("/")
async def read_root():
    return FileResponse('index.html')

@app.post("/chat/")
async def chat_with_data(request: ChatRequest):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Google API key not configured.")
    try:
        db_url = os.getenv("NEON_CONNECTION_STRING")
        if not db_url:
            raise ValueError("NEON_CONNECTION_STRING not found in .env file")
        all_schemas = []
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                for table_name in request.table_names:
                    schema_query = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';"
                    cur.execute(schema_query)
                    schema_rows = cur.fetchall()
                    if schema_rows:
                        schema_str = f"Table '{table_name}': " + ", ".join([f'"{col[0]}" ({col[1]})' for col in schema_rows])
                        all_schemas.append(schema_str)
        if not all_schemas:
            raise HTTPException(status_code=404, detail="No valid tables found.")
        combined_schema = "\n".join(all_schemas)
        prompt = f"""
        You are a powerful data analyst agent. Your process is to first use the `run_sql_query` tool to get information, and then to summarize the result for the user.
        **Database Context:**
        - Tables available: {combined_schema}
        **User's Command:** "{request.question}"
        **Your Plan:**
        1. Formulate a SQL query to answer the user's question.
        2. Call the `run_sql_query` tool with that query.
        3. You will then receive a `ToolMessage` with the result. Analyze this result.
        4. Formulate a final, friendly, natural language answer for the user based on the tool's result.
        **CRITICAL FINAL INSTRUCTION:** Your final message must ONLY be the natural language answer. Do not include your internal monologue, the SQL query, or raw tool results. Just the final, clean answer.
        **CRITICAL RULE:** You MUST enclose all table and column names in double quotes (e.g., "Weekly_Sales").
        """
        initial_state = {"messages": [HumanMessage(content=prompt)],"db_url": db_url}
        final_state = None
        for s in graph.stream(initial_state):
            print(s)
            final_state = s
        final_answer = final_state['agent']["messages"][-1].content
        return {"answer": final_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# --- UPGRADED /upload-file/ endpoint ---
@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    """Handles both CSV and Excel file uploads."""
    db_url = os.getenv("NEON_CONNECTION_STRING")
    if not db_url:
        raise HTTPException(status_code=500, detail="Database connection string not configured.")

    try:
        contents = await file.read()
        created_tables = []
        
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension == '.csv':
            df = pd.read_csv(io.BytesIO(contents), parse_dates=True)
            table_name = re.sub(r'[^a-zA-Z0-9_]', '_', os.path.splitext(file.filename)[0]).lower()
            process_dataframe_to_db(df, table_name, db_url)
            created_tables.append(table_name)

        elif file_extension == '.xlsx':
            # Read all sheets from the Excel file
            all_sheets = pd.read_excel(io.BytesIO(contents), sheet_name=None)
            for sheet_name, df in all_sheets.items():
                table_name = re.sub(r'[^a-zA-Z0-9_]', '_', sheet_name).lower()
                process_dataframe_to_db(df, table_name, db_url)
                created_tables.append(table_name)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a CSV or XLSX file.")
        
        return {
            "message": f"Successfully processed file and created tables: {', '.join(created_tables)}",
            "created_tables": created_tables
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

