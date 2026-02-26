import asyncpg
import logging
from typing import List, Dict, Any, Optional

from .config import settings

logger = logging.getLogger(__name__)

# Global connection pool
pool: Optional[asyncpg.Pool] = None


async def init_db() -> None:
    """Initialize the async PostgreSQL connection pool."""
    global pool
    try:
        pool = await asyncpg.create_pool(
            settings.DATABASE_URL,
            min_size=5,
            max_size=20,
            command_timeout=60,
        )
        logger.info("Database connection pool initialised")
    except Exception as e:
        logger.error(f"Failed to initialise database pool: {e}")
        raise


async def close_db() -> None:
    """Close the connection pool on shutdown."""
    global pool
    if pool:
        await pool.close()
        logger.info("Database connection pool closed")


def _validate_select_only(query: str) -> None:
    """
    Raise ValueError if the query is not a plain SELECT.
    Blocks any write or DDL operation.
    """
    stripped = query.strip().upper()

    if not stripped.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    dangerous = [
        "INSERT", "UPDATE", "DELETE", "DROP", "CREATE",
        "ALTER", "TRUNCATE", "EXECUTE", "MERGE", "GRANT", "REVOKE",
    ]
    for kw in dangerous:
        # word-boundary check to avoid false positives (e.g. column named "created")
        import re
        if re.search(rf"\b{kw}\b", stripped):
            raise ValueError(f"Forbidden keyword detected in query: {kw}")


async def execute_query(query: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Execute a SELECT query and return rows as a list of dicts.
    Raises ValueError for unsafe queries, RuntimeError if pool is missing.
    """
    global pool

    if not pool:
        raise RuntimeError("Database pool is not initialised.")

    _validate_select_only(query)

    try:
        async with pool.acquire() as conn:
            if params:
                rows = await conn.fetch(query, *params)
            else:
                rows = await conn.fetch(query)

        data = [dict(row) for row in rows]
        return {"success": True, "data": data, "row_count": len(data)}

    except asyncpg.PostgresError as e:
        logger.error(f"PostgreSQL error: {e}")
        return {"success": False, "error": str(e), "data": [], "row_count": 0}
    except Exception as e:
        logger.error(f"Unexpected DB error: {e}")
        return {"success": False, "error": str(e), "data": [], "row_count": 0}