import asyncio
import asyncpg
from app.config import settings

async def print_cols(schema, table):
    async with (await asyncpg.create_pool(settings.DATABASE_URL)) as pool:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT column_name,data_type FROM information_schema.columns
                WHERE table_schema=$1 AND table_name=$2
                ORDER BY ordinal_position
                """, schema, table)
            print(f"{schema}.{table}")
            for r in rows:
                print(r['column_name'], r['data_type'])

async def main():
    for tbl in [('public','domains'),('public','block')]:
        await print_cols(*tbl)

if __name__ == '__main__':
    asyncio.run(main())
