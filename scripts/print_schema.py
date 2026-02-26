import asyncio
import asyncpg
from app.config import settings

async def main():
    pool = await asyncpg.create_pool(settings.DATABASE_URL)
    async with pool.acquire() as conn:
        tables = [
            ('public','user'),
            ('public','test_submission'),
            ('pod','pod_submission'),
            ('pod','problem_of_the_day'),
        ]
        for schema,table in tables:
            print(f"\nSchema {schema}.{table}")
            rows = await conn.fetch("""
                SELECT column_name, data_type FROM information_schema.columns
                WHERE table_schema=$1 AND table_name=$2
            """, schema, table)
            for r in rows:
                print(r['column_name'], r['data_type'])
    await pool.close()

if __name__ == '__main__':
    asyncio.run(main())
