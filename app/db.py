import logging
import re
from typing import Dict, Any, Optional

import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from .config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# asyncpg connection pool — used for all query execution
# ---------------------------------------------------------------------------
pool: Optional[asyncpg.pool.Pool] = None

# Cached schema description — populated once at startup by load_schema()
_schema_description: str = ""


async def init_db() -> None:
    """Create an asyncpg pool for database connections."""
    global pool
    try:
        ssl_val = settings.DB_SSL.lower() == "require"
        pool = await asyncpg.create_pool(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            database=settings.DB_NAME,
            ssl=ssl_val,
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


# ---------------------------------------------------------------------------
# Schema introspection — uses SQLAlchemy inspect over asyncpg
# ---------------------------------------------------------------------------

async def load_schema() -> str:
    """Introspect the database schema and cache the result.

    Queries ``information_schema`` directly so we avoid SQLAlchemy's sync
    inspect API and can stay fully async.  The result is stored in the
    module-level ``_schema_description`` variable and also returned.
    """
    global _schema_description

    if not pool:
        raise RuntimeError("Database not initialized before load_schema()")

    target_schemas = ("public", "pod")
    lines: list[str] = ["DATABASE SCHEMA\n" + "=" * 60]

    async with pool.acquire() as conn:
        # ── 1. Fetch all tables in the target schemas ──────────────────────
        tables_sql = """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema = ANY($1::text[])
              AND table_type = 'BASE TABLE'
            ORDER BY table_schema, table_name
        """
        table_rows = await conn.fetch(tables_sql, list(target_schemas))

        # ── 2. Fetch all column info at once ────────────────────────────────
        columns_sql = """
            SELECT table_schema, table_name, column_name, data_type,
                   is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = ANY($1::text[])
            ORDER BY table_schema, table_name, ordinal_position
        """
        col_rows = await conn.fetch(columns_sql, list(target_schemas))

        # ── 3. Fetch foreign key relationships ──────────────────────────────
        fk_sql = """
            SELECT
                tc.table_schema, tc.table_name, kcu.column_name,
                ccu.table_schema AS foreign_schema,
                ccu.table_name   AS foreign_table,
                ccu.column_name  AS foreign_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
               AND tc.table_schema    = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema = ANY($1::text[])
        """
        fk_rows = await conn.fetch(fk_sql, list(target_schemas))

    # ── Group columns by (schema, table) ───────────────────────────────────
    from collections import defaultdict
    cols_by_table: Dict[tuple, list] = defaultdict(list)
    for r in col_rows:
        cols_by_table[(r["table_schema"], r["table_name"])].append(r)

    fks_by_table: Dict[tuple, list] = defaultdict(list)
    for r in fk_rows:
        fks_by_table[(r["table_schema"], r["table_name"])].append(r)

    # ── Render ──────────────────────────────────────────────────────────────
    current_schema = None
    for t in table_rows:
        s, tbl = t["table_schema"], t["table_name"]
        if s != current_schema:
            lines.append(f"\nSchema: {s}")
            lines.append("-" * 40)
            current_schema = s

        lines.append(f"\n  TABLE: {s}.{tbl}")
        for col in cols_by_table[(s, tbl)]:
            nullable = "NULL" if col["is_nullable"] == "YES" else "NOT NULL"
            default  = f" DEFAULT {col['column_default']}" if col["column_default"] else ""
            lines.append(f"    • {col['column_name']}  {col['data_type']}  {nullable}{default}")

        for fk in fks_by_table[(s, tbl)]:
            lines.append(
                f"    → FK: {fk['column_name']} → "
                f"{fk['foreign_schema']}.{fk['foreign_table']}.{fk['foreign_column']}"
            )

    _schema_description = "\n".join(lines)
    logger.info("Schema description cached (%d chars)", len(_schema_description))
    return _schema_description


def get_cached_schema() -> str:
    """Return the schema string cached during startup."""
    return _schema_description


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def _inject_limit(query: str, default_limit: int) -> str:
    """Add a LIMIT clause if the query does not already have one."""
    normalised = query.upper()
    # Simple heuristic: if 'LIMIT' is already in the query, leave it alone.
    if re.search(r"\bLIMIT\b", normalised):
        return query
    # Strip trailing semicolon, add limit, re-add semicolon.
    stripped = query.rstrip().rstrip(";").rstrip()
    return f"{stripped} LIMIT {default_limit};"


async def execute_query(query: str) -> Dict[str, Any]:
    """Run a validated SELECT query using the asyncpg pool.

    Enforces:
    • Only SELECT statements are permitted.
    • Dangerous DML/DDL keywords are blocked.
    • Automatically injects a LIMIT clause when absent.
    """
    if not pool:
        raise RuntimeError("Database not initialized")

    stripped = query.strip()
    upper = stripped.upper()

    # ── Safety: must start with SELECT ─────────────────────────────────────
    if not re.match(r"^\s*SELECT\b", upper):
        raise ValueError(
            "Only SELECT queries are allowed. "
            "INSERT, UPDATE, DELETE, DROP and similar statements are blocked."
        )

    # ── Safety: block dangerous keywords ───────────────────────────────────
    dangerous = [
        "INSERT", "UPDATE", "DELETE", "DROP", "CREATE",
        "ALTER", "TRUNCATE", "EXECUTE", "MERGE", "GRANT", "REVOKE",
    ]
    for kw in dangerous:
        if re.search(rf"\b{kw}\b", upper):
            raise ValueError(f"Forbidden keyword detected in query: {kw}")

    # ── Auto-inject LIMIT ───────────────────────────────────────────────────
    query = _inject_limit(stripped, settings.DEFAULT_LIMIT)

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(query)
            data = [dict(row) for row in rows]
            return {"success": True, "data": data, "row_count": len(data)}
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        return {"success": False, "error": str(e), "data": [], "row_count": 0}
