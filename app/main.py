import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import settings
from .db import close_db, get_cached_schema, init_db, load_schema
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
    query: str = Field(
        ..., min_length=1, max_length=2000,
        description="Natural language question about student analytics"
    )
    state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conversation state (messages history) from the previous turn"
    )


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
    logger.info("Starting TapTap Analytics Chatbot v%s …", settings.VERSION)

    # 1. Open DB pool
    await init_db()
    logger.info("Database pool ready")

    # 2. Introspect schema and cache it BEFORE the LLM is created so the
    #    system prompt already contains the live schema on the first request.
    await load_schema()
    logger.info("Schema description cached")

    # 3. Initialise the LangGraph agent (uses the cached schema)
    await init_llm()
    logger.info("LLM agent ready")

    yield

    logger.info("Shutting down …")
    await close_db()


# ------------------------------------------------------------------ #
#  App                                                                 #
# ------------------------------------------------------------------ #

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Schema-driven AI analytics chatbot for TapTap student performance data",
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
    return {
        "message": "TapTap Analytics Chatbot API",
        "version": settings.VERSION,
        "docs": "/docs",
    }


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

    The LangGraph agent dynamically writes SQL from the database schema,
    executes it via the `query_database` tool, and explains the results.
    """
    logger.info("Query: %s", request.query[:120])
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
    """Return the live schema description and example queries."""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "llm_provider": f"azure-openai-{settings.AZURE_OPENAI_DEPLOYMENT}",
        "schema_description": get_cached_schema(),
        "example_questions": [
            "Who solved today's POD in the IT domain?",
            "Who are the top 10 performers in MET test this week?",
            "Which students scored above 80 in coding but below 40 in reasoning?",
            "Show students with a POD streak greater than 5.",
            "Which college has the best average coding score?",
            "Who has the highest overall employability score?",
            "Show students at risk with scores below 40 in the last round.",
            "Which hackathon had the most participants?",
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
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred", "status_code": 500, "success": False},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)