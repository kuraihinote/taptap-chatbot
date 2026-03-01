import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import settings
from .db import init_db, close_db
from .llm import init_llm, process_user_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Request / Response models                                           #
# ------------------------------------------------------------------ #

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000,
                       description="Natural language question about student analytics")
    state: Dict[str, Any] = Field(default_factory=dict,
                                  description="Optional conversation state from previous turn")


class ChatResponse(BaseModel):
    answer: str
    data: list = []
    state: Dict[str, Any] = {}
    success: bool = True


class HealthResponse(BaseModel):
    status: str
    database: str
    llm: str
    version: str


# ------------------------------------------------------------------ #
#  Lifespan — startup / shutdown                                       #
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting TapTap Analytics Chatbot...")
    # database setup is asynchronous now
    await init_db()
    logger.info("Database pool ready")
    await init_llm()
    logger.info("LLM agent ready")
    yield
    logger.info("Shutting down...")
    await close_db()


# ------------------------------------------------------------------ #
#  App                                                                 #
# ------------------------------------------------------------------ #

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="AI-powered analytics chatbot for TapTap student performance data",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
#  Routes                                                              #
# ------------------------------------------------------------------ #

@app.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "TapTap Analytics Chatbot API", "version": settings.VERSION, "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    from .db import pool
    from .llm import llm_processor
    return HealthResponse(
        status="healthy",
        database="connected" if pool else "disconnected",
        llm="initialised" if llm_processor else "not_initialised",
        version=settings.VERSION,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Process a natural language query about student analytics.

    Steps:
    1. Azure OpenAI LLM decides which SQL to run
    2. SQL is executed on Azure PostgreSQL via the tool call
    3. Azure OpenAI summarises the results in plain English
    """
    logger.info(f"Query: {request.query[:120]}")
    result = await process_user_query(request.query, state=request.state)

    if not result.get("success", False):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("answer", "Unknown error"),
        )

    return ChatResponse(
        answer=result["answer"],
        data=result.get("data", []),
        state=result.get("state", {}),
        success=True,
    )


@app.get("/info", response_model=Dict[str, Any])
async def get_info():
    """Return database schema info and example questions."""
    from .tools import get_database_summary
    db_summary = await get_database_summary()
    return {
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "llm_provider": "azure-openai-gpt-4o-mini",
        "database_info": db_summary,
        "example_questions": [
            "Who solved today's POD in IT domain?",
            "Who is the top student in MET test?",
            "Top 10 students by average coding score",
            "Students with employability score above 80",
            "Students at risk with score below 40",
            "Average verbal, coding, and reasoning score per student",
        ],
    }


# ------------------------------------------------------------------ #
#  Global exception handlers                                           #
# ------------------------------------------------------------------ #

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code, "success": False},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred", "status_code": 500, "success": False},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)