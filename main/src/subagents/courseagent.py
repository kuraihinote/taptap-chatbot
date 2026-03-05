from langchain.tools import tool
from sqlalchemy import text
from src.database import get_db
from src.models import gpt_4o_mini_llm
from src.logger import logger
from langchain.agents import create_agent
from src.redisClient import redis_client


#----------------------------------- helper tools to fetch data from database -----------------------------------#

@tool("get_domain_wise_all_technologies")
async def get_domain_wise_all_technologies(course_domain_id: int):
    """
    Get all courses and technologies available for a given course_domain_id.
    Returns a list of courses with their titles, descriptions, and levels.

    """
    try:
        db=next(get_db())
        query = """
            SELECT course_title, course_description, course_level
            FROM course.course
            WHERE course_domain_id = :course_domain_id
        """
        result = db.execute(text(query), {"course_domain_id": course_domain_id}).fetchall()

        if not result:
            return f"No courses found for domain_id={course_domain_id}"

        return [
            {
                "course_title": row.course_title,
                "course_description": row.course_description,
                "course_level": row.course_level
            }
            for row in result
        ]

    except Exception as e:
        return f"Error: failed to fetch technologies — {e}"


@tool("get_all_domains")
async def get_all_domains():
    """
    This function retrives all the domains in the platform from the database but only id and name fields
    and this id is used to fetch domain wise technologies using another tool get_domain_wise_all_technologies.
    Returns a list of domains with their IDs and names.
    """
    try:
        db=next(get_db())

        query = """
            SELECT id, name
            FROM course.course_domain
        """
        result = db.execute(text(query)).fetchall()

        if not result:
            return "No domains found."

        return [{"id": row.id, "name": row.name} for row in result]

    except Exception as e:
        return f"Error: failed to get domains — {e}"

@tool()    
async def get_courses_according_to_college(college_id):
    """
    This tool is used to get all the courses available.
    It will provide title , description, level, hours, is_paid and link of the course.
    Returns a list of courses with their details.
    """
    try:
        # 1. Try Redis cache
        redis_key = f"courses_for_college:{college_id}"
        cached_data = redis_client.get(redis_key)

        if cached_data:
            import json
            data=json.loads(cached_data)
            print("Data fetched from Redis Cache",data)  
            return data


        # 2. Cache miss → fetch from DB
        db = next(get_db())
        query = """
            SELECT
                c.id,
                c.course_title,
                c.course_description,
                c.course_level,
                c.course_hours,
                c.is_paid
            FROM
                course.course AS c
            JOIN
                course.course_allowed_colleges AS cac
                    ON c.id = cac.course_id
            WHERE
                c.course_domain_id in (1,2,3,4)
                AND c.is_published IS NOT NULL
                AND cac.college_id = :college_id
                AND c.display_in_my_college_tab = TRUE
            """
        result = db.execute(text(query), {"college_id": college_id}).fetchall()

        courses = [
            {
                "course_title": row.course_title,
                "course_description": row.course_description,
                "course_level": row.course_level,
                "course_hours": row.course_hours,
                "is_paid": row.is_paid,
                "link": f"https://taptap.blackbucks.me/course/{row.id}/?testType=course"
            } 
            for row in result
        ]

        # 3. Save to Redis for future (cache for 1 hour)
        import json
        redis_client.setex(redis_key, 3600, json.dumps(courses))
        print("data get from the db", courses)

        return courses
    except Exception as e:  
        return f"Error: failed to fetch courses according to college — {e}"

#------------------------------------------ course agent setup as a tool for supervisor agent ------------------------------------------#
   
TOOLS=[get_courses_according_to_college]

COURSE_AGENT_PROMPT =(
        "You are the **Courses Department Agent** for an EdTech platform. "
        "Your job is to answer user questions about available domains, courses, or technologies. "
        "You may use tools to fetch or analyze relevant data. "
        "If you don’t have the answer, politely say you don’t know."
        )

course_agent=create_agent(
    gpt_4o_mini_llm,
    tools=TOOLS,
    system_prompt=COURSE_AGENT_PROMPT
)





@tool("courses_department_agent")
async def courses_department_agent(request: str):
    """
    Courses Department Agent
    this agent handles queries about courses, domains, and technologies.
    Returns a response based on user messages.

    """
    try:
        print( "request received in course agent:",request)
        response = await course_agent.ainvoke({
        "messages": [{"role": "user", "content": request}]
    })
        logger.info(f"[Courses Department Agent] Response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error in courses_department_agent: {e}")
        return f"Error: failed to process course query — {e}"
    
if __name__ == "__main__":
    import asyncio
    result=asyncio.run(get_courses_according_to_college())
    print(result)