import logging
from typing import Dict, Any

from .config import settings
from .db import execute_query

logger = logging.getLogger(__name__)


async def execute_sql_query(query: str) -> Dict[str, Any]:
    """
    Execute a SQL SELECT query on behalf of the LLM tool call.
    Enforces the MAX_QUERY_RESULTS row cap.
    """
    try:
        # execute_query is now async
        result = await execute_query(query)

        if not result["success"]:
            return {
                "success": False,
                "error": result["error"],
                "data": [],
                "message": f"Query failed: {result['error']}",
            }

        data = result["data"]
        if len(data) > settings.MAX_QUERY_RESULTS:
            data = data[: settings.MAX_QUERY_RESULTS]
            logger.warning(f"Results capped at {settings.MAX_QUERY_RESULTS} rows")

        return {
            "success": True,
            "data": data,
            "row_count": len(data),
            "message": f"Returned {len(data)} row(s).",
        }

    except ValueError as e:
        logger.error(f"Query validation error: {e}")
        return {"success": False, "error": str(e), "data": [], "message": str(e)}

    except Exception as e:
        logger.error(f"Unexpected error in execute_sql_query: {e}")
        return {"success": False, "error": str(e), "data": [], "message": str(e)}


def get_tool_definition() -> Dict[str, Any]:
    """
    JSON-schema tool definition passed to the LLM as a FunctionDeclaration.
    """
    return {
        "name": "execute_sql_query",
        "description": (
            "Execute a SQL SELECT query on the PostgreSQL database to retrieve "
            "student performance data. Use this for any question about students, "
            "test scores, employability results, POD submissions, or hackathons. "
            "Only SELECT queries are permitted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A valid SQL SELECT statement. "
                        "Tables: public.user (id, first_name, last_name, email), "
                        "public.test_submission (user_id, problem_id, score, status, create_at), "
                        "pod.pod_submission (user_id, question_id, domain_id, status, create_at), "
                        "pod.problem_of_the_day (id, date, question_id, difficulty). "
                        "Use schema-qualified names (e.g. pod.pod_submission). "
                        "Always JOIN public.user ON u.id = table.user_id for student names. "
                        "Domain filtering uses domain_id from pod.pod_submission."
                    ),
                }
            },
            "required": ["query"],
        },
    }


async def get_database_summary() -> Dict[str, Any]:
    """Used by the /info endpoint to show available tables."""
    table_descriptions = {
        "public.user": "Student information (id, first_name, last_name, email, roll_number, ...)",
        "public.test_submission": "Test results (user_id, problem_id, score, status, create_at)",
        "pod.pod_submission": "POD submissions (user_id, question_id, domain_id, status, create_at)",
        "pod.problem_of_the_day": "Daily problems (id, date, question_id, difficulty, is_active)",
    }

    try:
        result = await execute_query(
            "SELECT table_schema, table_name "
            "FROM information_schema.tables "
            "WHERE table_schema IN ('public', 'pod') "
            "ORDER BY table_schema, table_name"
        )
        available = [f"{r['table_schema']}.{r['table_name']}" for r in result["data"]]
    except Exception as e:
        logger.error(f"Could not fetch table list: {e}")
        available = list(table_descriptions.keys())

    return {
        "available_tables": available,
        "table_descriptions": table_descriptions,
    }