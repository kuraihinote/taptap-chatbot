import asyncio
import json
import logging
from typing import Dict, Any, List, Optional

from groq import Groq

from .config import settings
from .tools import execute_sql_query, get_tool_definition

logger = logging.getLogger(__name__)


class LLMProcessor:
    def __init__(self):
        if not settings.LLM_API_KEY:
            raise ValueError("LLM_API_KEY is not set in .env")

        self._client = Groq(api_key=settings.LLM_API_KEY)
        self._model = "llama-3.3-70b-versatile"
        self._tool = self._build_tool()

    def _build_tool(self) -> Dict[str, Any]:
        base = get_tool_definition()
        return {
            "type": "function",
            "function": {
                "name": base["name"],
                "description": base["description"],
                "parameters": base["parameters"],
            },
        }

    def _system_prompt(self) -> str:
        return (
            "You are an AI assistant for the TapTap Analytics Chatbot.\n"
            "You help faculty analyse student performance by querying a PostgreSQL database.\n\n"
            "DATABASE SCHEMA:\n"
            "  public.user                    — id (PK), first_name, last_name, email, roll_number, ...\n"
            "  public.test_submission         — id (PK), user_id (FK), problem_id, score, status, create_at\n"
            "  pod.pod_submission             — id (PK), user_id (FK), question_id, domain_id, status, create_at\n"
            "  pod.problem_of_the_day         — id (PK), date, question_id, difficulty, is_active\n\n"
            "CRITICAL COLUMN MAPPINGS:\n"
            "  • public.user: id (NOT user_id), first_name, last_name for full name\n"
            "  • pod.pod_submission: domain_id (integer, FK), status (e.g. 'solved', 'attempted')\n"
            "  • pod.problem_of_the_day: date (DATE field), id (problem ID)\n"
            "  • Timestamp columns: create_at (not created_at), update_at (not updated_at)\n\n"
            "RULES:\n"
            "  • Always call execute_sql_query to fetch data — never guess or make up numbers.\n"
            "  • Only SELECT queries are allowed.\n"
            "  • Always use schema-qualified table names (e.g. pod.pod_submission).\n"
            "  • JOIN public.user ON u.id = submission.user_id to get student names.\n"
            "  • Use CURRENT_DATE for 'today' queries and match with pod.problem_of_the_day.date.\n"
            "  • For domain filtering, use domain_id in pod.pod_submission (not pod.problem_of_the_day).\n"
            "  • Apply LIMIT 50 unless the user asks for more.\n"
            "  • After getting data, reply in clear natural language with the key facts.\n"
            "  • If the query returns no rows, say so clearly.\n"
        )

    def _call_groq_sync(self, messages: List[Dict]) -> Any:
        return self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=[self._tool],
            tool_choice="auto",
            max_tokens=4096,
            temperature=0.1,
        )

    async def process_user_query(self, user_query: str) -> Dict[str, Any]:
        try:
            messages: List[Dict] = [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": user_query},
            ]
            all_tool_results: List[Dict] = []

            for round_num in range(5):
                logger.info(f"[Round {round_num + 1}] Calling Groq...")
                response = await asyncio.to_thread(self._call_groq_sync, messages)
                choice = response.choices[0]
                msg = choice.message

                logger.info(f"[Round {round_num + 1}] finish_reason={choice.finish_reason}, tool_calls={bool(msg.tool_calls)}")

                # Model finished — no tool call
                if not msg.tool_calls:
                    logger.info(f"[Round {round_num + 1}] Final answer received")
                    return {
                        "answer": msg.content or "No response generated.",
                        "data": self._flatten(all_tool_results),
                        "success": True,
                    }

                # Append assistant message
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

                # Execute each tool call
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    query_text = args.get('query', '')
                    logger.info(f"[Round {round_num + 1}] Tool call: {tc.function.name} | SQL: {query_text}")
                    result = await self._run_tool(tc.function.name, args)
                    logger.info(f"[Round {round_num + 1}] Tool result: success={result.get('success')}, rows={result.get('row_count', 0)}, error={result.get('error', '')}")
                    all_tool_results.append(result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    })

            logger.warning("Tool loop exhausted after 5 rounds")
            return {
                "answer": "Unable to complete the analysis. Please rephrase your question.",
                "data": self._flatten(all_tool_results),
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "answer": f"Sorry, I encountered an error: {e}",
                "data": [],
                "success": False,
            }

    async def _run_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "execute_sql_query":
            return await execute_sql_query(args.get("query", ""))
        return {"success": False, "error": f"Unknown tool: {name}", "data": []}

    def _flatten(self, results: List[Dict]) -> List[Dict]:
        rows = []
        for r in results:
            if r.get("success") and r.get("data"):
                rows.extend(r["data"])
        return rows


llm_processor: Optional[LLMProcessor] = None


async def init_llm() -> None:
    global llm_processor
    try:
        llm_processor = LLMProcessor()
        logger.info("LLM processor initialised (Groq - llama-3.3-70b-versatile)")
    except Exception as e:
        logger.error(f"Failed to initialise LLM processor: {e}")
        raise


async def process_user_query(user_query: str) -> Dict[str, Any]:
    global llm_processor
    if not llm_processor:
        raise RuntimeError("LLM processor not initialised")
    return await llm_processor.process_user_query(user_query)