from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import Optional, List, Dict, Any
from src.QA_integration import QA_RAG
from langchain_neo4j import Neo4jGraph
import logging
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://aryabhatta.neu.edu:5177"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Models ----------------

class QARequest(BaseModel):
    question: str
    model: str = "gpt-4o"
    document_names: Optional[List[str]] = None
    session_id: str = "default_session"
    mode: str = "graph_vector_fulltext"
    write_access: bool = True
    neo4j_uri: Optional[str] = None
    neo4j_username: Optional[str] = None
    neo4j_password: Optional[str] = None
    neo4j_database: Optional[str] = None

class QAResponse(BaseModel):
    status: str
    data: Dict[str, Any]

# ---------------- Defaults ----------------

DEFAULT_NEO4J_URI = os.getenv("NEO4J_URI")
DEFAULT_NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
DEFAULT_NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

def create_graph_database_connection(uri, userName, password, database):
    try:
        return Neo4jGraph(
            url=uri,
            database=database,
            username=userName,
            password=password,
            refresh_schema=False,
            sanitize=True
        )
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect to Neo4j database.")

# ---------------- Error Handlers ----------------

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred."}
    )

# ---------------- Endpoints ----------------
@app.post("/chat_bot", response_model=QAResponse)
def qa_form_endpoint(
    question: str = Form(...),
    model: str = Form("gpt-4o"),
    session_id: str = Form("default_session"),
    mode: str = Form("graph_vector_fulltext"),
    write_access: bool = Form(True),
    neo4j_uri: Optional[str] = Form(None),
    neo4j_username: Optional[str] = Form(None),
    neo4j_password: Optional[str] = Form(None),
    neo4j_database: Optional[str] = Form(None)
):
    try:
        uri = neo4j_uri or DEFAULT_NEO4J_URI
        username = neo4j_username or DEFAULT_NEO4J_USERNAME
        password = neo4j_password or DEFAULT_NEO4J_PASSWORD
        database = neo4j_database or DEFAULT_NEO4J_DATABASE

        graph = create_graph_database_connection(uri, username, password, database)

        result = QA_RAG(
            graph=graph,
            model=model,
            question=question,
            document_names=None,
            session_id=session_id,
            mode=mode,
            write_access=write_access
        )

        response = { "status": "success", "data": result }

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /qa-form endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "GraphRAG FastAPI is running."}
