from langchain.tools import tool
from sqlalchemy import text
from src.database import get_db 
from src.models import gpt_4o_mini_llm
from src.logger import logger
from langchain.agents import create_agent
#----------------------------------- helper tools to fetch data from database -----------------------------------#
@tool()
async def company_wise_total_test_count():
    """
    This tool is used for to get what are the companies and how many test are their for each company.
    Returns the count of tests corresponded to company name.
    """
    try:
        db=next(get_db())
        query = """
            SELECT
            c.name AS company_name,
            COUNT(h.company_id) AS hack_count
            FROM
                public.hackathon h
            JOIN
                public.company c ON h.company_id = c.id
            WHERE
                h.status = 'published' -- Added filter for published status
            GROUP BY
                h.company_id,
                c.name
            ORDER BY
                hack_count DESC;
            """
        result = db.execute(text(query)).fetchall()
        return [
            {
                "company_name": row.company_name, 
                "hack_count": row.hack_count
            } for row in result]
    except Exception as e:
        logger.error(f"Error in company_wise_total_test_count tool: {e}")
        return "Error: failed to fetch total test count"
    
@tool()
async def get_tests_by_company(company_id: int):
    """
    This tool is used to get all the tests available for a given company_id.
    Returns a list of tests with their titles and descriptions.
    """
    try:
        db=next(get_db())
        query = """
            SELECT
            h.id,
            h.title AS hackathon_title
            FROM
            public.hackathon h
            WHERE
            h.company_id = :company_id and h.status = 'published';
            """
        result = db.execute(text(query), {"company_id": company_id}).fetchall()
        return [
            {
                "hackathon_title": row.hackathon_title, 
                "link":f"https://taptap.blackbucks.me/hackathon/{row.id}"
            } for row in result]
    except Exception as e:
        logger.error(f"Error in get_tests_by_company tool: {e}")
        return "Error: failed to fetch tests by company"
    
@tool()
async def get_company_id(company_name:str):
    """
    this tool is used for to get the company id based on the company name 
    """
    try:
        db=next(get_db())
        query="""
        SELECT id 
        FROM public.company 
        WHERE name ILIKE :company_name
        LIMIT 1;        """
        result = db.execute(text(query), {"company_name": f"%{company_name}%"}).fetchone() 
        if result:
            company_id = result[0]  # or result._mapping["id"]
            return {"status": "ok", "company_id": company_id}
        else:
            return f"No company found with name {company_name}" 
    except Exception as e:
        logger.error(f"Error in get_company_id tool: {e}")
        return "Error internal error unable to connect to the db"
    

system_prompt="""
You are CompanyTestsAgent, an AI agent designed to assist users in finding tests and hackathons offered by various companies on the TapTap platform.
Your primary role is to provide users with accurate and relevant information about the tests available for different companies.
When a user inquires about tests from a specific company, you should:
1. Use the `get_company_id` tool to retrieve the unique identifier for the specified company.
2. With the obtained company ID, utilize the `get_tests_by_company` tool to fetch

    the list of tests associated with that company. 
3. Present the information in a clear and concise manner, including test titles, descriptions, and links to participate.
Always ensure that the information you provide is up-to-date and relevant to the user's query.
"""
    

company_wise_test_agent=create_agent(
    gpt_4o_mini_llm,
    tools=[
        company_wise_total_test_count,
        get_tests_by_company,
        get_company_id
    ],
    system_prompt=system_prompt
)

@tool("companys_department_agent")
async def company_department_agent(request: str):
    """
    This is the company department agent which is responsible for handling all the queries related to company tests and hackathons.
    Returns a response based on user messages.

    """
    try:
        print( "request received in companys agent:",request)
        response = await company_wise_test_agent.ainvoke({
        "messages": [{"role": "user", "content": request}]
    })
        logger.info(f"[company Department Agent] Response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error in company_department_agent: {e}")
        return f"Error: failed to process company query — {e}"