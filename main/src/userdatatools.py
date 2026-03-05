from langchain.tools import tool
from sqlalchemy import text
from streamlit import exception
from src.redisClient import redis_client
from src.database import get_db
import json
from src.logger import logger
from bs4 import BeautifulSoup

def clean_html(text: str) -> str:
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ").strip()


async def get_user_details2(user_id:str):
    """
    This function is used for to get the user_data.
    """
    # redis_key=f"user_data:{user_id}"
    # cached_data=redis_client.get(redis_key)
    # if cached_data:
    #     user_data=json.loads(cached_data)
    #     print("user data fetched from cache",user_data)
    #     return user_data
    db=next(get_db())
    query="""
        select id,first_name,last_name,role,phone,email,dob,college_id,department_id,roll_number,is_bbtraining,is_bbverified,generated_resume_url,
        is_placed,strong_in,weak_in,average_in,top_three_skills 
        from
        public.user
        where id=:user_id
    """
    result=db.execute(text(query),{"user_id":user_id}).fetchone()
    if not result:
        return {"error": "User not found"}

    # Convert DB row to dict
    user_data = dict(result._mapping)

    # 3. Cache it in Redis
    # redis_client.setex(redis_key, 3600, json.dumps(user_data))

    print("User data fetched from DB and cached",user_data)
    return user_data



async def get_student_badges(user_id:str):
    db=next(get_db())
    query="""
    SELECT 
        b.badge_type,
        h.title
    FROM public.user_badge AS ub
    JOIN public.hackathon AS h  ON ub.hackathon_id = h.id
    JOIN public.badge AS b      ON ub.badge_id = b.id
    WHERE ub.user_id = :user_id;

    """
    result=db.execute(text(query),{"user_id":user_id}).fetchall()
    if not result:
        return "user not gained any badge"
    data=[dict(row._mapping) for row in result]
    return data




async def get_hackthon_participation(user_id:str):
    db=next(get_db())
    query="""
    SELECT
    h.title,
    uhp.current_score,
    uhp.report
    FROM public.user_hackathon_participation AS uhp
    JOIN public.hackathon AS h ON h.id = uhp.hackathon_id
    WHERE
        uhp.user_id = :user_id
        AND uhp.current_score > 0   
    ORDER BY uhp.update_at DESC
    LIMIT 3;

    """
    result=db.execute(text(query),{"user_id":user_id}).fetchall()
    if not result:
        return "user not gained any badge"
    result=[dict(row._mapping) for row in result]
    data = []

    for row_data in result:
        row = {}
        row["title"]=row_data["title"]
        
        for question_report in row_data["report"]["questionReports"]:
            skill = question_report["skill"]

            if skill in row:
                # Skill exists → update counts
                row[skill]["total_count"] += 1

                if question_report["report"]["status"] == "fail":
                    row[skill]["failed_count"] += 1
                else:
                    row[skill]["passed_count"] += 1

            else:
                # Skill does not exist → initialize
                row[skill] = {
                    "total_count": 1,
                    "failed_count": 1 if question_report["report"]["status"] == "fail" else 0,
                    "passed_count": 1 if question_report["report"]["status"] == "pass" else 0
                }

        data.append(row)

    return data


async def get_user_details(user_id:str):
    db=next(get_db())
    query="""
        select  linkedin_id,twitter_id,facebook_id,github_id,bb_training,introduction,technologies,languages,extra_curricular_activities,
    hobbies,profile_score,tools,frameworks,programming_languages,profile_summary,leet_code_id,hacker_rank_id,english_proficiency_tests,
    profile_title
    from resume.user_details where "user"=:user_id;
    """
    result=db.execute(text(query),{"user_id":user_id}).fetchone()
    if result is None:
        return {"error": "No user details found"}
    result=dict(result._mapping)
    return result

async def get_user_technical_experience(user_id:str):
    db=next(get_db())
    query="""
        select  company_name,job_title,skill,description
        from resume.technical_experience where user_id=:user_id;
    """
    result=db.execute(text(query),{"user_id":user_id}).fetchall()
    if not result:
        return "No prior technical experience"
    result=[dict(row._mapping) for row in result]
    return result


async def get_user_projects(user_id:str):
    db=next(get_db())
    query="""
    select name,url,technologies_used,description
    from resume.project where user_id=:user_id;
    """
    result=db.execute(text(query),{"user_id":user_id}).fetchall()
    if not result:
        return "No projects mentioned"
    
    result=[dict(row._mapping) for row in result]
    return result

async def get_user_goal(user_id:str):
    db=next(get_db())
    query="""
        select company_name,role,minimum_salary_lpa,maximum_salary_lpa,aptitude_target,coding_target,english_target,skill_want_to_learn,
        prioritize_goals
        from resume.my_goal where user_id=:user_id;
    """
    result=db.execute(text(query),{"user_id":user_id}).fetchall()
    if not result:
        return "No goal defined"
    
    result=[dict(row._mapping) for row in result]
    return result



async def get_user_certifications(user_id:str):
    db=next(get_db())
    query="""
    select title,issued_by from resume.certificate where user_id=:user_id;
    """
    result=db.execute(text(query),{"user_id":user_id}).fetchall()
    if not result:
        return "No certifications provided"
    
    result=[dict(row._mapping) for row in result]
    return result

async def get_user_achievements(user_id:str):
    db=next(get_db())
    query="""
    select title,institution,achievement from resume.achievement where user_id=:user_id;
    """
    result=db.execute(text(query),{"user_id":user_id}).fetchall()
    if not result:
        return "No achievements or not provided "
    
    result=[dict(row._mapping) for row in result]
    return result

async def get_user_data(user_id: str):
    try:
        redis_key=f"user_data:{user_id}"
        cached_data=redis_client.get(redis_key)
        if cached_data:
            print ("response from cached data")
            data=json.loads(cached_data)
            return data
        
        user_detail = await get_user_details(user_id)
        user_details2=await get_user_details2(user_id)
        technical_experience = await get_user_technical_experience(user_id)
        projects = await get_user_projects(user_id)
        goal = await get_user_goal(user_id)
        achievements = await get_user_achievements(user_id)
        certifications = await get_user_certifications(user_id)
        hackthon_participation=await get_hackthon_participation(user_id)
        user_badges=await get_student_badges(user_id)

        main_data=[{
            "user_details": user_detail | user_details2,
            "technical_experience": technical_experience,
            "projects": projects,
            "goal": goal,
            "achievements": achievements,
            "certifications": certifications,
            "hackthon_participation":hackthon_participation,
            "user_badges":user_badges

        }]

        print("data from the db")
        redis_client.setex(redis_key, 3600, json.dumps(main_data))


        return main_data
    except Exception as e:
        raise Exception(f"unable to get the user details:{str(e)}")
@tool
async def get_user_data_tool(user_id: str):
    """
    Retrieve all user-related information required for reasoning, conversation,
    and personalization. This tool should be used by the LLM whenever it needs
    detailed profile information about a specific user.

    INPUT:
    - user_id (str): Unique identifier of the user whose data needs to be fetched.

    OUTPUT:
    - A dictionary containing a structured "llm_profile" object with the user's
      introduction, skills, strengths, weaknesses, technologies, tools,
      frameworks, hobbies, languages, profile summary, profile score, and other
      high-level information useful for LLM-based reasoning.

    PURPOSE:
    - Provides all necessary profile information in a compact format.
    - Prevents exposing unnecessary or sensitive raw database fields.
    - Ensures consistent structured data for downstream LLM tasks such as:
        * personalized recommendations
        * exam assistance
        * career/skill guidance
        * progress evaluation
        * answering user-specific queries
    - Should be called whenever the LLM needs user context that is not currently
      available in the conversation memory.

    NOTE:
    - Do NOT use this tool for modifying user data.
    - This tool only retrieves and returns LLM-ready profile information.
    """
    try:
        main_data = await get_user_data(user_id)
        user_data = {}

        data = main_data[0]["user_details"]

        user_data["llm_profile"] = {
            "introduction": data["introduction"],
            "technologies": data["technologies"],
            "languages": data["languages"],
            "extra_curricular_activities": data["extra_curricular_activities"],
            "hobbies": data["hobbies"],
            "profile_score": data["profile_score"],
            "tools": data["tools"],
            "frameworks": data["frameworks"],
            "programming_languages": data["programming_languages"],
            "profile_summary": data["profile_summary"],
            "profile_title": data["profile_title"],
            "strong_in": data["strong_in"],
            "weak_in": data["weak_in"],
            "average_in": data["average_in"],
            "top_three_skills": data["top_three_skills"],
        }

        return user_data
    except Exception:
        raise Exception("Unable to fetch user details.")

def format_basic_user_data(data: dict, achievements, certifications, badges) -> str:
    intro = clean_html(data.get("introduction", ""))
    profile_summary = clean_html(data.get("profile_summary", ""))

    languages = "\n".join(
        [f"- {lang['name']} ({lang['proficiency']})" for lang in data.get("languages", [])]
    ) or "No languages listed."

    hobbies = ", ".join(data.get("hobbies", [])) or "No hobbies provided."
    extra = clean_html(data.get("extra_curricular_activities", "")) or "None"

    strong_in = ", ".join(data.get("strong_in", [])) or "None"
    weak_in = ", ".join(data.get("weak_in", [])) or "None"
    avg_in = ", ".join(data.get("average_in", [])) or "None"
    top_skills = ", ".join(data.get("top_three_skills", [])) or "None"

    achievements_text = "\n".join(
        [f"- {a['title']} ({a['institution']}): {clean_html(a['achievement'])}" for a in achievements]
    ) or "No achievements found."
    
    badges_text = ", ".join(badges) or "No badges found."

    return f"""
User Basic Details
------------------

Introduction:
{intro}

Languages Known:
{languages}

Hobbies:
{hobbies}

Extra Curricular Activities:
{extra}

Profile Title:
{data.get("profile_title", "Not provided")}

Profile Score:
{data.get("profile_score", "Not provided")}

Profile Summary:
{profile_summary}

Skill Strengths:
- Strong In: {strong_in}
- Weak In: {weak_in}
- Average In: {avg_in}

Top Three Skills:
{top_skills}

Achievements:
{achievements_text}

Badges:
{badges_text}
"""

@tool
async def get_basic_user_data_tool(user_id: str):
    """
     Retrieves the user's basic profile details such as hobbies, languages,
    achievements, certifications, and general strengths or weaknesses.  
    Used when high-level personal or non-technical information is required.
    
    """
    try:
        main_data = await get_user_data(user_id)
        user_data = {}

        data = main_data[0]["user_details"]

        user_data["user_basic_details"]= {
            "introduction": data["introduction"],
            "languages": data["languages"],
            "extra_curricular_activities": data["extra_curricular_activities"],
            "hobbies": data["hobbies"],
            "profile_score": data["profile_score"],
            "profile_summary": data["profile_summary"],
            "profile_title": data["profile_title"],
            "strong_in": data["strong_in"],
            "weak_in": data["weak_in"],
            "average_in": data["average_in"],
            "top_three_skills": data["top_three_skills"],
        }
        user_data["user_achievements"]=main_data[0]["achievements"]
        user_data["user_certifications"]=main_data[0]["certifications"]
        user_data["user_badges"]=main_data[0]["user_badges"]
        # result_text = format_basic_user_data(
        #     user_basic_details,
        #     user_achievements,
        #     user_certifications,
        #     user_badges
        # )

        # return result_text.strip()
        return user_data
    except Exception:
        raise Exception("Unable to fetch user details.")
    
def format_skill_user_data(detail, experience, projects, goal) -> str:
    def format_skill_block(title, items):
        if not items:
            return f"{title}:\nNo data available.\n"
        return f"{title}:\n" + "\n".join(
            [f"- {i['name']} ({i['percentage']}%)" for i in items]
        ) + "\n"

    intro = clean_html(detail.get("introduction", ""))
    summary = clean_html(detail.get("profile_summary", ""))
    languages = "\n".join(
        [f"- {lang['name']} ({lang['proficiency']})" for lang in detail.get("languages", [])]
    )

    exp_text = "\n".join(
        [f"- {e['company_name']} ({e['job_title']}): {clean_html(e['description'])}" for e in experience]
    ) or "No experience found."

    project_text = "\n".join(
        [f"- {p['name']} ({', '.join(p['technologies_used'])}): {clean_html(p['description'])}" for p in projects]
    ) or "No projects found."

    return f"""
User Technical Profile
-----------------------

Introduction:
{intro}

Languages:
{languages}

Profile Title:
{detail.get("profile_title", "Not provided")}

Profile Summary:
{summary}

{format_skill_block("Technologies", detail.get("technologies", []))}
{format_skill_block("Programming Languages", detail.get("programming_languages", []))}
{format_skill_block("Frameworks", detail.get("frameworks", []))}
{format_skill_block("Tools", detail.get("tools", []))}

Technical Experience:
{exp_text}

Projects:
{project_text}

Goal:
{goal}
"""
@tool
async def get_skills_user_data_tool(user_id: str):
    """
    Fetches the user's technical profile, including skills, technologies,
    work experience, projects, and career goals. Used when responses require
    skill-based or experience-based information about the user.   
    """
    try:
        main_data = await get_user_data(user_id)
        user_data = {}

        user_detail = main_data[0]["user_details"]

        user_data["user_basic_details"] = {
            "introduction": user_detail["introduction"],
            "languages": user_detail["languages"],
            "profile_score": user_detail["profile_score"],
            "profile_summary": user_detail["profile_summary"],
            "profile_title": user_detail["profile_title"],
        }
        user_data["user_technical_experience"]=main_data[0]["technical_experience"]
        user_data["user_projects"]=main_data[0]["projects"]
        user_data["user_goal"]=main_data[0]["goal"]
        # result_text = format_skill_user_data(
        #     user_basic_details,
        #     user_technical_experience,
        #     user_projects,
        #     user_goal
        # )

        # return result_text.strip()
        return user_data
        
    except Exception:
        raise Exception("Unable to fetch user details.")
    


@tool
async def get_user_test_results(user_id:str):
    """
        Fetches all past user assessments (tests, hackathons, coding challenges).

        Args:
            user_id (str): Unique user identifier.

        Returns:
            dict: A list of participation records, each containing:
                - title (str): Assessment name.
                - <skill> (dict): total_count, passed_count, failed_count for each skill.

    """
    try:
        main_data = await get_user_data(user_id)
        user_hackthon_participation_results=main_data[0]["hackthon_participation"]
        return user_hackthon_participation_results

        
    except Exception:
        raise Exception("unable to fetch user details right now.")



async def get_courses_according_to_college(user_id:str):
    
    try:
        user_data=await get_user_details2(user_id)
        college_id=user_data["college_id"]
        print("college_id :",college_id)

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
        logger.error(f"courses tool failed: {str(e)}")
        return f"Error: failed to fetch courses according to college"   


@tool
async def get_courses_according_to_college_tool(user_id:str):
    """
    This tool is used to get all the courses available in the platform  by taking input user_id.
    It will provide title , description, level, hours, is_paid and link of the course.
    Returns a list of courses with their details.
    """
    courses=await get_courses_according_to_college(user_id)
    return courses
    


