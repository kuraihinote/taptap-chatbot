import asyncio
import logging

from app.llm import init_llm, process_user_query
from app.db import init_db

logging.basicConfig(level=logging.INFO)

async def test():
    # simulate application startup
    await init_db()
    await init_llm()
    res = await process_user_query("Who solved today's POD in IT domain?")
    print(res)

if __name__ == '__main__':
    asyncio.run(test())
