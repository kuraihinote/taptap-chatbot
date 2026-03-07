"""
llm.py — Schema-Driven LangGraph Analytics Agent
=================================================

Architecture
------------
  Faculty Question
        │
        ▼
  [ LangGraph Agent ]
        │ ← system prompt contains live DB schema
        │
        ├── needs data → tool_node → query_database → PostgreSQL
        │                                 │
        └─────────── LLM summarises ◄─────┘
                     results in plain English

No hardcoded intents, no slot extractors, no SQL templates.
The LLM reasons dynamically from the schema.
"""

import json
import logging
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from .config import settings
from .db import get_cached_schema
from .tools import query_database

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are a PostgreSQL analytics assistant for TapTap, an education-technology platform.
Your job is to help college faculty understand student performance by querying a live
PostgreSQL database and explaining the results in clear, concise natural language.

━━━━━━━━━━━━━━━━━━━━━━ LIVE DATABASE SCHEMA ━━━━━━━━━━━━━━━━━━━━━━
{schema}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUERY RULES — follow these exactly:
  1. ALWAYS use the `query_database` tool to retrieve data. Never guess results.
  2. Use only SELECT statements — INSERT / UPDATE / DELETE / DROP / CREATE are forbidden.
  3. Always schema-qualify table names: `pod.pod_submission`, `public.user`, etc.
  4. Include a LIMIT clause (the system will auto-add one if you forget, but prefer explicit).
  5. JOIN `public.user` (u.id = <table>.user_id) whenever student names are needed.
  6. Use CURRENT_DATE for "today" comparisons with POD tables.
  7. Cast or convert data types as needed for comparisons (e.g. date columns).
  8. When returning data, present it as a readable summary — not raw JSON.

SCHEMA CLARIFICATIONS:
  • pod.pod_submission.status values are 'pass' or 'fail' (never 'solved' or 'completed')
  • domain filtering on POD: JOIN public.problem p ON p.id = ps.question_id, then WHERE 'IT' = ANY(p.domain)
  • public.user primary key is 'id' (varchar) — other tables reference it as user_id
  • When listing students who failed/passed POD, use DISTINCT on user to avoid duplicate rows per student — unless the user asks for "all attempts"
  • Always include the problem title in POD queries: JOIN public.problem p ON p.id = ps.question_id, SELECT p.title AS problem_title


SCOPE:
  • Only answer questions about students, tests, hackathons, employability, and POD.
  • If a question is off-topic (weather, general coding help, etc.), politely decline and
    remind the user what this chatbot is for.

RESPONSE FORMAT:
  • Lead with a one-sentence direct answer.
  • Follow with a concise table or list of key details if the result contains multiple rows.
  • End with a brief insight or observation if useful.
"""


def build_system_prompt() -> str:
    """Build the full system prompt with the cached schema injected."""
    schema = get_cached_schema()
    if not schema:
        logger.warning("Schema cache is empty — prompt will have no schema context")
    return _SYSTEM_TEMPLATE.format(schema=schema or "(schema not yet loaded)")


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

class LLMProcessor:
    """Schema-driven LangGraph agent processor."""

    def __init__(self) -> None:
        if not settings.AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY is not set in .env")

        self._llm = AzureChatOpenAI(
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            temperature=0.0,          # deterministic SQL generation
            max_retries=2,
        )
        self._tools = [query_database]
        self._llm_with_tools = self._llm.bind_tools(self._tools)
        self._graph = self._build_graph()

    # ── Graph construction ──────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        """Build the agent ↔ tool LangGraph."""

        def agent_node(state: AgentState) -> AgentState:
            """Call the LLM. If it emits tool calls they'll be handled next."""
            system = SystemMessage(content=build_system_prompt())
            messages = [system] + state["messages"]
            response = self._llm_with_tools.invoke(messages)
            return {"messages": [response]}

        async def async_agent_node(state: AgentState) -> AgentState:
            """Async variant of agent_node used in ainvoke."""
            system = SystemMessage(content=build_system_prompt())
            messages = [system] + state["messages"]
            response = await self._llm_with_tools.ainvoke(messages)
            return {"messages": [response]}

        async def tool_node(state: AgentState) -> AgentState:
            """Execute every tool call the LLM requested."""
            last = state["messages"][-1]
            if not isinstance(last, AIMessage) or not last.tool_calls:
                return {"messages": []}

            tool_messages: list[ToolMessage] = []
            for tc in last.tool_calls:
                if tc["name"] == "query_database":
                    result_str = await query_database.ainvoke(tc["args"])
                else:
                    result_str = json.dumps(
                        {"success": False, "error": f"Unknown tool: {tc['name']}"}
                    )
                tool_messages.append(
                    ToolMessage(content=result_str, tool_call_id=tc["id"])
                )
            return {"messages": tool_messages}

        def should_continue(state: AgentState) -> str:
            """Route: call tool if LLM requested one, else END."""
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tool"
            return END

        graph = StateGraph(AgentState)
        graph.add_node("agent", async_agent_node)
        graph.add_node("tool", tool_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_continue, {"tool": "tool", END: END})
        graph.add_edge("tool", "agent")
        return graph.compile()

    # ── Public interface ────────────────────────────────────────────────────

    async def process_user_query(
        self, user_query: str, state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a faculty query through the LangGraph agent.

        ``state`` may carry previous ``messages`` for multi-turn conversation.
        """
        state = state or {}
        history: list = state.get("messages", [])

        # Append the new human message to the conversation history
        history = history + [HumanMessage(content=user_query)]

        try:
            final_state = await self._graph.ainvoke({"messages": history})
        except Exception as e:
            logger.error("LangGraph invocation failed: %s", e, exc_info=True)
            return {
                "answer": f"Sorry, I encountered an error processing your request: {e}",
                "data": [],
                "state": state,
                "success": False,
            }

        final_messages = final_state.get("messages", [])

        # Extract the last AI text response
        answer = ""
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and msg.content:
                answer = msg.content
                break

        # Extract any data rows from embedded ToolMessages (for the API response)
        data: list[dict] = []
        for msg in final_messages:
            if isinstance(msg, ToolMessage):
                try:
                    parsed = json.loads(msg.content)
                    if parsed.get("success") and parsed.get("data"):
                        data.extend(parsed["data"])
                except (json.JSONDecodeError, AttributeError):
                    pass

        # Persist conversation history in state for next turn
        new_state = {**state, "messages": final_messages}

        return {
            "answer": answer or "I wasn't able to generate a response. Please try rephrasing.",
            "data": data,
            "state": new_state,
            "success": True,
        }


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

llm_processor: Optional[LLMProcessor] = None


async def init_llm() -> None:
    """Initialise the LLM processor singleton."""
    global llm_processor
    try:
        llm_processor = LLMProcessor()
        logger.info(
            "LangGraph analytics agent initialised (Azure OpenAI — %s)",
            settings.AZURE_OPENAI_DEPLOYMENT,
        )
    except Exception as e:
        logger.error("Failed to initialise LLM processor: %s", e)
        raise


async def process_user_query(
    user_query: str, state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Module-level entry point called by main.py."""
    global llm_processor
    if not llm_processor:
        raise RuntimeError("LLM processor not initialised — call init_llm() first")
    return await llm_processor.process_user_query(user_query, state)
