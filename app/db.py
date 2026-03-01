import logging
from typing import Dict, Any, Optional

import asyncpg

from .config import settings

logger = logging.getLogger(__name__)

# asyncpg connection pool
pool: Optional[asyncpg.pool.Pool] = None


async def init_db() -> None:
    """Create an asyncpg pool for database connections."""
    global pool
    try:
        pool = await asyncpg.create_pool(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            database=settings.DB_NAME,
            ssl=settings.DB_SSL.lower() == "require",
            min_size=5,
            max_size=20,
        )
        logger.info("Asyncpg pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        raise


async def close_db() -> None:
    """Close the asyncpg pool."""
    global pool
    if pool:
        await pool.close()
        logger.info("Database pool closed")


async def execute_query(query: str) -> Dict[str, Any]:
    """Run a validated SELECT query using the pool.

    Returns a dict with keys 'success', 'data', and 'row_count'.
    """
    if not pool:
        raise RuntimeError("Database not initialized")

    stripped = query.strip().upper()
    if not stripped.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    dangerous = [
        "INSERT", "UPDATE", "DELETE", "DROP", "CREATE",
        "ALTER", "TRUNCATE", "EXECUTE", "MERGE", "GRANT", "REVOKE",
    ]
    import re
    for kw in dangerous:
        if re.search(rf"\b{kw}\b", stripped):
            raise ValueError(f"Forbidden keyword detected in query: {kw}")

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(query)
            data = [dict(row) for row in rows]
            return {"success": True, "data": data, "row_count": len(data)}
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        return {"success": False, "error": str(e), "data": [], "row_count": 0}
