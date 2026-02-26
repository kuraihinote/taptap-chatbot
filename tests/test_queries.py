import asyncio
import json
from app.db import init_db, close_db
from app.llm import init_llm, process_user_query

test_queries = [
    "Who solved today's POD in IT domain?",
    "Top 5 students by test score",
    "Which students have status 'solved' for POD?",
]

async def test():
    await init_db()
    await init_llm()
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        result = await process_user_query(query)
        print(f"Answer: {result['answer']}")
        print(f"Success: {result['success']}")
        if result.get('data'):
            print(f"Data rows: {len(result['data'])}")
    
    await close_db()

if __name__ == '__main__':
    asyncio.run(test())
