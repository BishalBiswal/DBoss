import os
import re
import time
import uuid
import asyncio
import uvicorn
import logging
import traceback
import threading
from pydantic import BaseModel
from dotenv import load_dotenv
from functools import lru_cache
from urllib.parse import quote_plus
from langchain_groq import ChatGroq
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from typing import Dict, Any, Optional, List
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.utilities import SQLDatabase
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser

logging.basicConfig(filename='logfile.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG
)
logger = logging.getLogger("sql_agent_app")

load_dotenv('key.env')
api_key = os.environ.get('GROQ_API_KEY')
user = os.environ.get('DB_USER')
password = os.environ.get('DB_PASSWORD')
host = os.environ.get('DB_HOST')
port = os.environ.get('DB_PORT')
database = os.environ.get('DB_NAME')

url = f'postgresql://{user}:{quote_plus(password)}@{host}:{port}/{database}'

def create_llm_client():
    try:
        return ChatGroq(
            api_key=api_key,
            model="gemma2-9b-it",
            temperature=0,  
            request_timeout=60,  
            max_retries=2        
        )
    except Exception as e:
        logger.error(f"Error creating LLM client: {str(e)}")
        raise

try:
    llm = create_llm_client()
except Exception as e:
    logger.critical(f"Failed to initialize LLM clients: {str(e)}")
    raise

try:
    engine = create_engine(
        'postgresql+psycopg2://',
        connect_args={
            'user': user,
            'password': password,
            'host': host,
            'port': port,
            'database': database
        },
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=300,  
        pool_pre_ping=True
    )
except Exception as e:
    logger.critical(f"Failed to create database engine: {str(e)}")
    raise

query_cache = {}
CACHE_EXPIRY = 300
cache_lock = threading.RLock()  

@contextmanager
def get_db_connection():
    connection = None
    try:
        connection = engine.connect()
        yield connection
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise
    finally:
        if connection:
            connection.close()

def validate_sql(query: str) -> bool:
    forbidden_keywords = ['DROP', 'DELETE', 'ALTER', 'INSERT', 'UPDATE', 'TRUNCATE', 'CREATE', 'GRANT', 'REVOKE']
    query_lower = query.lower()
    return not any(keyword.lower() in query_lower for keyword in forbidden_keywords)

def clean_sql_query(query: str) -> str:
    return query.replace("```sql", "").replace("```", "").strip()

class Safe_db(SQLDatabase):    
    def run(self, command: str, fetch: str = 'all', parameters: Optional[Dict[str, Any]] = None, **kwargs) -> dict[str, any]:
        cleaned_query = clean_sql_query(command)
        if not validate_sql(cleaned_query):
            raise ValueError(f"Query contains forbidden commands: {cleaned_query}")
        
        try:
            logger.info(f"Executing query: {cleaned_query}")
            if parameters:
                logger.info(f"With parameters: {parameters}")
            return super().run(cleaned_query, fetch, parameters=parameters, **kwargs)
        except Exception as e:
            logger.error(f"Database error on query {cleaned_query}: {str(e)}")
            if hasattr(self, '_engine'):
                self._engine.dispose()
            raise
    
    def run_no_throw(self, command: str, fetch: str = 'all'):
        try:
            return self.run(command, fetch)
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            logger.error(error_msg)
            return error_msg
        
    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        try:
            info = super().get_table_info(table_names)
            if len(info) > 10000:  
                logger.warning(f"Table info too large ({len(info)} chars), truncating")
                return info[:10000] + "\n... (truncated for size limits)"
            return info
        except Exception as e:
            logger.error(f"Error getting table info: {str(e)}")
            return f"Error retrieving table information: {str(e)}"

@lru_cache(maxsize=3)
def get_db():
    try:
        return Safe_db.from_uri(url)
    except Exception as e:
        logger.error(f"Error creating database connection: {str(e)}")
        raise

def reset_db_connection():
    try:
        engine.dispose()
        get_db.cache_clear()
        logger.info("Database connections reset")
        return True
    except Exception as e:
        logger.error(f"Error resetting connections: {str(e)}")
        return False

def clean_cache():
    with cache_lock:
        current_time = time.time()
        keys_to_delete = []
        
        for key, (timestamp, _, _) in query_cache.items():
            if current_time - timestamp > CACHE_EXPIRY:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del query_cache[key]
        
        logger.info(f"Cache cleaned, removed {len(keys_to_delete)} items")


class QueryRequest(BaseModel):
    question: str
    use_cache: bool = True
    force_refresh: bool = False

PSQL_AGENT_PREFIX = """
You are a SQL database agent that helps users with database queries. Your name is punk.

IMPORTANT INSTRUCTIONS:
1. ALWAYS follow this exact format for each step:
   - Question: the input question you must answer.
   - Thought: Consider what information you need and how to approach the query
   - Action: Choose a tool to use
   - Action Input: Provide the input for the tool (SQL query with NO backticks or formatting)
   - Observation: Review results
   - ... (repeat steps as needed)
   - Thought: I now know the final answer
   - Final Answer: Your clear, concise answer to the user's question
**IMPORTANT**:If the documents and context provided DO NOT contain information relevant to answering queries you MUST respond with: "I'm sorry I cannot help with that".
 

2. DO NOT add any greetings or explanations outside of this format
3. DO NOT combine steps or skip the format
4. DO NOT use backticks (```) when providing SQL in Action Input

Available tools:
"""

PSQL_AGENT_FORMAT_INSTRUCTIONS = """
Use EXACTLY this format:

Question: the input question you must answer.
Thought: your reasoning about the question and approach.
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action (SQL query with NO backticks).
Observation: the result of the action.
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer.
Final Answer: the answer to the original question.

Remember - NEVER start with "Hello" or any greeting - always start with "Thought:".
"""

app = FastAPI(title="SQL Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request {request_id} completed in {process_time:.3f}s with status {response.status_code}")
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}")
        raise

def is_db_relevant_question(question: str) -> bool:
    question_lower = question.lower().strip()
    
    if len(question_lower) < 5:
        return True
    
    db_keywords = [
        'database', 'table', 'query', 'sql', 'data', 'record', 'column', 'row','departments', 
        'select', 'from', 'where', 'join', 'group by', 'order by', 'count','db','everything',
        'show me', 'find', 'search', 'list', 'get', 'fetch', 'retrieve','name','tell','details',
        'how many', 'what is the', 'when was', 'who has', 'which','number','department','detail',
        'average', 'sum', 'total', 'compare', 'difference', 'report','student','tender'
    ]
    
    db_patterns = [
    r'in the (database|db|table|data)',
    r'from the (database|db|table|data)',
    r'(show|display|get|fetch|retrieve|find|detail|details) (all|the|some|any|top|latest)',
    r'(filter|sort|group|count) (by|the|all|of)',
    r'(highest|lowest|most|least|average|total)',
    r'(select|query|search|lookup|explore) (data|records|entries|rows|columns)',
    r'(how many|what is|list all|find all|show all)'
    ]
    
    if any(keyword in question_lower for keyword in db_keywords):
        return True
    
    if any(re.search(pattern, question_lower) for pattern in db_patterns):
        return True
    
    entity_patterns = [
        r'(how many|tell|which|name|what|when|who|where|all|details|list)\b.{1,30}\b(all|everything|customers|orders|products|employees|departments|transactions|accounts|users|sale)',
        r'(reports|summary|metrics|statistics|numbers|figures|data|information|everything) (on|for|about|of|in)',
        r'(top|recent|latest|oldest|first|last|all)\b.{1,20}\b(entries|records|transactions|orders|tenders|db)'
    ]
    
    if any(re.search(pattern, question_lower) for pattern in entity_patterns):
        return True
    return False

def handle_simple_query(question: str):
    question_lower = question.lower().strip()
    greetings = ["hello", "hi", "hey", "greetings","helo","halo","hai","hoi",]
    if any(question_lower == greeting for greeting in greetings):
        return {
            "answer": "Hello! I'm punk, your database assistant. How can I help you with your database queries today?",
            "cached": False
        }
    if question_lower in ["are you there", "are you working", "test"]:
        return {
            "answer": "Yes, I'm here and ready to help with your database queries!",
            "cached": False
        }
    
    if not is_db_relevant_question(question):
        return {
            "answer": "I cannot help you with that, please ask something related to the database. I'm designed to assist with database queries and data analysis.",
            "cached": False
        }
    return None

# Function to clean the agent's output
def clean_agent_output(output: str) -> str:
    """Clean the agent output to remove repeated 'Final Answer:' lines."""
    if isinstance(output, dict):
        # If it's a dictionary, extract the output field
        answer = output.get("output", "")
    else:
        answer = str(output)
    
    # If multiple "Final Answer:" appears, extract only the last part
    if answer.count("Final Answer:") > 1:
        parts = answer.split("Final Answer:")
        # Take only the last part after "Final Answer:"
        answer = parts[-1].strip()
    elif "Final Answer:" in answer:
        # Take everything after the first "Final Answer:"
        answer = answer.split("Final Answer:", 1)[1].strip()
    
    # Remove any remaining traces of the agent's thinking process
    patterns_to_remove = [
        r'Thought:.*?(?=Action:|Final Answer:|$)',
        r'Action:.*?(?=Action Input:|$)',
        r'Action Input:.*?(?=Observation:|$)',
        r'Observation:.*?(?=Thought:|$)'
    ]
    
    for pattern in patterns_to_remove:
        answer = re.sub(pattern, '', answer, flags=re.DOTALL)
    
    return answer.strip()

@app.post("/ask")
async def api_llm(request: QueryRequest, background_tasks: BackgroundTasks):
    query_id = str(uuid.uuid4())[:8]
    question = request.question
    logger.info(f"Query {query_id}: Processing '{question}'")
    background_tasks.add_task(clean_cache)
    simple_response = handle_simple_query(question)
    if simple_response:
        return simple_response
    
    if request.use_cache and not request.force_refresh:
        with cache_lock:
            if question in query_cache:
                timestamp, answer, is_error = query_cache[question]
                if time.time() - timestamp < CACHE_EXPIRY and not is_error:
                    logger.info(f"Query {query_id}: Cache hit")
                    return {"answer": answer, "cached": True}

    if request.force_refresh:
        reset_db_connection()
    
    try:
        db = get_db()
        
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)    
        sql_agent = create_sql_agent(
            prefix=PSQL_AGENT_PREFIX,
            format_instructions=PSQL_AGENT_FORMAT_INSTRUCTIONS,
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=8,
            output_parser=ReActSingleInputOutputParser()
        )
        
        loop = asyncio.get_event_loop()
        logger.info(f"Query {query_id}: Executing agent")
        
        try:
            res = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: sql_agent.invoke({'input': question})), 
                timeout=60.0    
            )
        except asyncio.TimeoutError:
            logger.error(f"Query {query_id}: Execution timed out")
            raise TimeoutError("Query execution timed out. Please try a simpler query.")
        
        answer = clean_agent_output(res)
        
        if "hello" in question.lower() or "hi " in question.lower():
            if not any(greeting in answer.lower() for greeting in ["hello", "hi", "hey", "greetings"]):
                answer = "Hello! I'm punk, your database assistant. " + answer
        
        with cache_lock:
            query_cache[question] = (time.time(), answer, False)
        
        logger.info(f"Query {query_id}: Completed successfully")
        return {"answer": answer, "cached": False}

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Query {query_id} failed: {str(e)}\n{error_trace}")
        error_message = str(e)
        status_code = 500  # Default to internal server error

        if "413" in error_message or "Payload Too Large" in error_message:
            error_message = "punk could not handle this much data. Please try a simpler or smaller question."
            status_code = 413

        elif "500" in error_message or "Internal Server Error" in error_message:
            reset_db_connection()
            error_message = "punk is overwhelmed, please try again with a different query."

        elif "parsing output" in error_message and "unexpected keyword" in error_message:
            reset_db_connection()
            error_message = "There was a temporary issue with the database connection. Please try again."
            status_code = 400  

        elif "TimeoutError" in error_message:
            error_message = "The query took too long to process. Please try a simpler question."
            status_code = 408
            
        # Add specific handling for the greeting error case
        elif "Hello!" in error_message and "database assistant" in error_message:
            # Extract the greeting from the error and use it as the answer
            try:
                greeting_match = re.search(r"Hello!(.*?)$", error_message, re.DOTALL)
                if greeting_match:
                    error_message = f"Hello!{greeting_match.group(1)}"
                    with cache_lock:
                        query_cache[question] = (time.time(), error_message, False)
                    return {"answer": error_message, "cached": False}
            except:
                pass
        
        with cache_lock:
            query_cache[question] = (time.time(), error_message, True)
        raise HTTPException(status_code=status_code, detail={"error": "Encountered an error try rephrasing or changing your query","details":error_message})

@app.post("/reset")
async def reset_system():
    try:
        with cache_lock:
            query_cache.clear()
        get_db.cache_clear()        
        engine.dispose()
        global llm
        llm = create_llm_client()
        
        db = get_db()
        tables = db.get_usable_table_names()
        
        return {
            "status": "success", 
            "message": "System reset complete",
            "tables_found": len(tables)
        }
    except Exception as e:
        logger.error(f"System reset failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        db = get_db()
        with get_db_connection() as conn:
            result = conn.execute(text("SELECT 1")).fetchone()
            db_ok = result[0] == 1
        
        llm_response = llm.invoke("Test")
        llm_ok = llm_response is not None
        
        with cache_lock:
            cache_size = len(query_cache)
        
        return {
            "status": "healthy" if (db_ok and llm_ok) else "degraded",
            "database": "connected" if db_ok else "error",
            "llm": "connected" if llm_ok else "error",
            "cache_size": cache_size
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    try:
        db = get_db()
        tables = db.get_usable_table_names()
        logger.info(f"Connected to database with {len(tables)} tables")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")
        raise