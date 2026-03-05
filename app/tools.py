import json
import logging
from typing import Dict, Any

from langchain_core.tools import tool

from .db import execute_query
from .config import settings

logger = logging.getLogger(__name__)


@tool
async def query_database(query: str) -> str:
    """Execute a SQL SELECT query on the PostgreSQL analytics database.

    Use this tool whenever you need to retrieve data about students, tests,
    hackathons, employability scores, or Problem-of-the-Day (POD) submissions.

    Rules enforced automatically:
    - Only SELECT statements are allowed.
    - A LIMIT clause is added automatically if missing.
    - Schema-qualified table names are required (e.g. pod.pod_submission).

    Returns a JSON string with keys: success, data (list of row dicts),
    row_count, and optionally error.
    """
    try:
        result = await execute_query(query)

        # Enforce row cap
        if result.get("success") and len(result.get("data", [])) > settings.MAX_QUERY_RESULTS:
            result["data"] = result["data"][: settings.MAX_QUERY_RESULTS]
            result["row_count"] = settings.MAX_QUERY_RESULTS
            result["capped"] = True
            logger.warning("Results capped at %d rows", settings.MAX_QUERY_RESULTS)

        return json.dumps(result, default=str)

    except ValueError as e:
        logger.error("Query validation error: %s", e)
        return json.dumps({"success": False, "error": str(e), "data": []})

    except Exception as e:
        logger.error("Unexpected error in query_database: %s", e)
        return json.dumps({"success": False, "error": str(e), "data": []})


async def get_database_summary() -> Dict[str, Any]:
    """Return a brief description of available tables (used by /info endpoint)."""
    from .db import get_cached_schema
    return {"schema_description": get_cached_schema()}