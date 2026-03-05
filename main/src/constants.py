from dotenv import load_dotenv
import os 
load_dotenv()



DATABASE_URL=os.getenv("DATABASE_URL")
if not  DATABASE_URL: 
    raise Exception("database url not found")  

AZURE_gpt_4o_mini_CONFIG = {
    'api_key': os.getenv("AZURE_OPENAI_API_KEY4"),
    'api_version': os.getenv("AZURE_OPENAI_API_VERSION4"),
    'azure_endpoint': os.getenv("AZURE_OPENAI_ENDPOINT4"),
    'deployment_name': os.getenv("AZURE_OPENAI_DEPLOYMENT4")
}

if not AZURE_gpt_4o_mini_CONFIG:
    raise Exception("Azure config details not found")

