from langchain.tools import tool
from sqlalchemy import text
from src.database import get_db
from src.redisClient import redis_client
import json 
from youtube_search import YoutubeSearch
import re
import requests



def parse_views(view_str: str) -> int:
    """
    Convert '1,627,407 views' → 1627407
    """
    if not view_str:
        return 0

    num = re.sub(r"[^0-9]", "", view_str)
    return int(num) if num.isdigit() else 0

def is_embeddable(video_id: str) -> bool:
    """Return True only if the video can be embedded."""
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    res = requests.get(url)
    return res.status_code == 200

@tool
async def youtube_search(search_input: str):
    """
    Tool that searches YouTube vidoes and returns the most viewed video for a given search input.

    """
    try:

        raw = YoutubeSearch(search_input, 10).to_json()
        data = json.loads(raw)

        videos = data["videos"]
        results = []

        # Add normalized fields
        for v in videos:
            vid = v["id"]
            if not is_embeddable(vid):
                continue
            v["views_num"] = parse_views(v.get("views"))
            v["embed_url"] = f"https://www.youtube.com/embed/{v['id']}"
            results.append(v)
        
        # If nothing embeddable found — return empty string
        if len(results) == 0:
            print("No embeddable videos found")
            return "No video found"
            


        videos_sorted = sorted(
            results,
            key=lambda x: x["views_num"],
            reverse=True
        )
        print("videos:",videos_sorted)
        return {"embed_url":videos_sorted[0]["embed_url"]}
    except Exception as e:
        return None

@tool
async def get_company_hackathon_data_tool(company_list: list):
    """
    This tool is used to get all the hackathons or tests  available for each company, company tests , papers etc...
    8byte
    Accenture
    Airbus
    Amazon
    Analytics Quad4 (AQ4)
    Blackbucks Group
    Capgemini
    Cisco
    Cognizant GenC Elevate
    Congnizant GenC Next
    Deloitte
    Fino Labs
    Flipkart
    Freight Tiger
    HTC
    IBM
    Infosys
    L & T
    Magnaquest
    STATESTREET
    TCS Code Vita
    TCS NQT
    UST
    Virtusa
    Wipro
    Zenoti
    Returns a dictionary with company names as keys and lists of hackathons as values.

    """
    # redis_key="company_hackathon_data"
    # cached_data = redis_client.get(redis_key)
    # if cached_data:
    #     print("data get from  cache")
    #     if isinstance(cached_data, bytes):
    #         cached_json = cached_data.decode("utf-8")
    #     else:
    #         cached_json = cached_data
    #     full_data = json.loads(cached_json)
    #     return {
    #         company: full_data.get(company, [])
    #         for company in company_list
    #     }
        

    db = next(get_db())

    query = """
        SELECT 
            h.id AS hackathon_id,
            h.title AS hackathon_title,
            c.name AS company_name
        FROM 
            hackathon h
        JOIN 
            company c ON h.company_id = c.id
        WHERE 
            h.status = 'published'
            AND h.test_type_id = 40;
    """

    rows = db.execute(text(query)).fetchall()
    print("data get from db")

    company_data = {}

    for row in rows:
        company_name = row.company_name
        
        if company_name not in company_data:
            company_data[company_name] = []

        company_data[company_name].append({
            "link": f"https://taptap.blackbucks.me/hackathon/{row.hackathon_id}",
            "title": row.hackathon_title
        })
    # redis_client.setex(redis_key, 3600, json.dumps(company_data))  

    return {
        company: company_data.get(company, [])
        for company in company_list
    }


@tool
async def get_features_tool():
    """
    This tool provides what are the features offereing in the platform with navigation links .
    """
    data = """
Here are the features offered on the platform along with their descriptions and links:

1. Coding Track  
   Practice programming problems similar to platforms like LeetCode or HackerRank.  
   Link: https://taptap.blackbucks.me/practice/codingTrack/

2. Aptitude Track  
   Practice aptitude topics such as ratios, proportions, averages, percentages, etc.  
   Link: https://taptap.blackbucks.me/practice/aptitudeTrack/

3. SmartInterview  
   Prepare for interviews with communication practice and real-time interview simulations.  
   Includes:  
      - Resume-Only Interview Practice  
      - Job Description–Only Practice  
      - Resume + Job Description Mode  
      - College-Curated Interview Sets  
   Link: https://taptap.blackbucks.me/prep/smartInterview/

4. Profiling Tests  
   Personality and behavioral profiling tests to understand strengths and work styles.  
   Link: https://taptap.blackbucks.me/gest/profilingTests/

5. MET — Monthly Employability Tests  
   Monthly assessments designed to measure student employability performance.  
   Link: https://taptap.blackbucks.me/gest/employabilityTests/

6. Skill Tests  
   Technology-specific skill assessments such as HTML/CSS, JavaScript, ReactJS, and more.  
   Link: https://taptap.blackbucks.me/gest/skillTests/

7. Company-Wise Tests  
   Practice company-specific placement tests (e.g., TCS, Infosys, Wipro, etc.).  
   Link: https://taptap.blackbucks.me/prep/byCompany/

8. Technology-Wise Tests  
   Topic-focused practice such as Java Interview Questions, Python Interview Questions,  
   DSA Interview Questions, HR Mock Questions, etc.  
   Link: https://taptap.blackbucks.me/prep/byTechnology/

9. IELTS Preparation  
   Practice all four IELTS modules: Speaking, Listening, Writing, and Reading.  
   Link: https://taptap.blackbucks.me/studyAbroad/ielts/

10. GRE Preparation  
    Prepare for GRE sections: Verbal Reasoning, Analytical Writing, and Quantitative Reasoning.  
    Link: https://taptap.blackbucks.me/studyAbroad/gre/
"""
    return data




if __name__ == "__main__":
    import asyncio
    companies = [
        "Accenture",
        "Airbus",
        "Amazon",
        "Analytics Quad4 (AQ4)",
        "Blackbucks Group",
        "Capgemini",
        "Cisco",
        "Cognizant GenC Elevate",
        "Congnizant GenC Next",
        "Deloitte",
        "Fino Labs",
        "Flipkart",
        "Freight Tiger",
        "HTC",
        "IBM",
        "Infosys",
        "L & T",
        "Magnaquest",
        "STATESTREET",
        "TCS Code Vita",
        "TCS NQT",
        "UST",
        "Virtusa",
        "Wipro",
        "Zenoti"
    ]
    result = asyncio.run(get_company_hackathon_data_tool(companies))
    print(result)