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
from langchain_openai import ChatOpenAI

# Load environment variables from the .env file
load_dotenv()

# --- Initialize OpenRouter Client ---
try:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        print("Warning: OPENROUTER_API_KEY not found. Chat will not work.")
except Exception as e:
    print(f"Error initializing API client: {e}")

# Create the FastAPI application instance
app = FastAPI(
    title="Magic Data Assistant API",
    description="An API powered by a LangGraph Agent to chat with your data.",
    version="9.8.0" # Final LangGraph Version with Autocommit
)

# --- CORS Middleware ---
origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    table_name: str
    question: str

# --- Helper Functions ---
def pandas_to_sql_type(dtype):
    if "int" in dtype.name: return "BIGINT"
    elif "float" in dtype.name: return "FLOAT"
    elif "datetime" in dtype.name: return "TIMESTAMP"
    else: return "TEXT"

# --- LangGraph Agent Implementation ---

# 1. Define the Agent's State
class AgentState(TypedDict):
    messages: List[BaseMessage]
    db_url: str
    table_name: str

# 2. Define the Tool
@tool
def run_sql_query(sql_query: str = Field(description="The PostgreSQL query to be executed.")):
    """A tool to execute a SQL query against the database and return a summary of the result."""
    pass

# 3. Define the Graph Nodes
def call_model(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# REFACTORED call_tool TO USE AUTOCOMMIT FOR GUARANTEED SAVES
def call_tool(state: AgentState):
    tool_calls = state["messages"][-1].tool_calls
    if not tool_calls:
        return {"messages": [HumanMessage(content="The model did not call a tool. Ending.")]}

    tool_call = tool_calls[0]
    sql_query = tool_call["args"]["sql_query"]
    db_url = state["db_url"]
    
    # Use a 'with' statement for the connection to ensure it's always closed.
    # Set autocommit=True to ensure all commands (DELETE, UPDATE, etc.) are saved immediately.
    try:
        with psycopg.connect(db_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query)
                query_upper = sql_query.strip().upper()
                
                if query_upper.startswith("SELECT"):
                    results = cur.fetchall()
                    if not results:
                        tool_result = "Query returned no results."
                    else:
                        tool_result = f"Query returned {len(results)} rows. Here are the first 3: {results[:3]}"
                else:
                    # For DELETE, UPDATE, INSERT, rowcount tells us what happened.
                    # No conn.commit() is needed due to autocommit=True.
                    rows_affected = cur.rowcount
                    tool_result = f"Query executed successfully. {rows_affected} rows were affected."

        return {"messages": [ToolMessage(content=tool_result, tool_call_id=tool_call["id"])]}
    except Exception as e:
        error_message = f"Error executing SQL query '{sql_query}'. Error: {e}"
        return {"messages": [ToolMessage(content=error_message, tool_call_id=tool_call["id"])]}

# 4. Define the Graph Edges
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return "end"
    else:
        return "continue"

# 5. Construct and Compile the Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
workflow.add_edge("action", "agent") 
graph = workflow.compile()

# Initialize the LLM with our tool, pointing to OpenRouter
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct:free", 
    openai_api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "Magic Data Assistant",
    },
)
llm_with_tools = llm.bind_tools([run_sql_query])


# --- API Endpoints ---

@app.get("/")
async def read_root():
    return FileResponse('index.html')

@app.post("/chat/")
async def chat_with_data(request: ChatRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured.")

    try:
        db_url = os.getenv("NEON_CONNECTION_STRING")
        if not db_url:
            raise ValueError("NEON_CONNECTION_STRING not found in .env file")
        
        conn = psycopg.connect(db_url)
        with conn.cursor() as cur:
            schema_query = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{request.table_name}';"
            cur.execute(schema_query)
            schema_rows = cur.fetchall()
            if not schema_rows:
                raise HTTPException(status_code=404, detail=f"Table '{request.table_name}' not found.")
            table_schema = ", ".join([f'"{col[0]}" ({col[1]})' for col in schema_rows])
        conn.close()

        # --- UPDATED PROMPT TO ALLOW ALL OPERATIONS ---
        prompt = f"""
        You are a powerful data analyst agent. Your job is to help a user interact with their database by translating their plain English commands into SQL queries.

        **Permissions:** You are fully authorized to perform any of the following actions:
        - **Read data** using `SELECT` statements.
        - **Modify data** using `UPDATE` statements.
        - **Add data** using `INSERT` statements.
        - **Remove data** using `DELETE` statements.

        **Context:**
        - You are interacting with a PostgreSQL database.
        - The table you need to query is named "{request.table_name}".
        - The schema of the table is: {table_schema}.
        
        **User's Command:** "{request.question}"

        **Your Plan:**
        1.  Carefully analyze the user's command to understand their intent (read, insert, update, or delete).
        2.  Generate the single, correct SQL query to perform the requested action.
        3.  Call the `run_sql_query` tool with that query.
        4.  Once you receive the result from the tool, formulate a final, friendly answer confirming what you have done.

        **CRITICAL FINAL INSTRUCTION:** Your final message must ONLY be the natural language answer for the user. Do not include your internal monologue, the SQL query, or the raw tool results in your final response. Just the answer.
        **CRITICAL RULE:** You MUST enclose all table and column names in double quotes (e.g., "Weekly_Sales").
        """
        
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "db_url": db_url,
            "table_name": request.table_name
        }
        
        final_state = None
        for s in graph.stream(initial_state):
            print(s) # This will print the agent's steps in your terminal for debugging
            final_state = s

        # The final answer is the last message from the agent in the graph's final state
        final_answer = final_state['agent']["messages"][-1].content
        
        return {"answer": final_answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# ... (The other endpoints: /upload-csv/, etc. remain unchanged) ...
@app.post("/upload-csv/")
async def create_upload_file(file: UploadFile = File(...)):
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), parse_dates=True)
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', os.path.splitext(file.filename)[0]).lower()
        conn = None
        try:
            db_url = os.getenv("NEON_CONNECTION_STRING")
            conn = psycopg.connect(db_url)
            with conn.cursor() as cur:
                columns_with_types = [f'"{col}" {pandas_to_sql_type(dtype)}' for col, dtype in df.dtypes.items()]
                create_query = (
                    f'DROP TABLE IF EXISTS "{table_name}";\n'
                    f"CREATE TABLE \"{table_name}\" (id SERIAL PRIMARY KEY, {', '.join(columns_with_types)});"
                )
                cur.execute(create_query)
                output = io.StringIO()
                df.to_csv(output, sep='\t', header=False, index=False)
                output.seek(0)
                column_names_for_copy = ", ".join([f'"{col}"' for col in df.columns])
                with cur.copy(f"COPY \"{table_name}\" ({column_names_for_copy}) FROM STDIN") as copy:
                    while data := output.read(1024):
                        copy.write(data)
                conn.commit()
            return {"message": f"Successfully created table '{table_name}' and inserted {len(df)} rows.", "database_table_created": table_name}
        except Exception as e:
            if conn: conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            if conn: conn.close()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error processing CSV file: {e}")

@app.post("/add-row/")
def add_row(table_name: str, row_data: Dict[Any, Any]):
    conn = None
    try:
        db_url = os.getenv("NEON_CONNECTION_STRING")
        conn = psycopg.connect(db_url)
        with conn.cursor() as cur:
            columns = row_data.keys()
            values_placeholder = ", ".join(["%s"] * len(columns))
            cols_formatted = ", ".join(f'"{col}"' for col in columns)
            insert_query = f'INSERT INTO "{table_name}" ({cols_formatted}) VALUES ({values_placeholder})'
            cur.execute(insert_query, list(row_data.values()))
            conn.commit()
        return {"message": f"Successfully added 1 row to '{table_name}'."}
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn: conn.close()