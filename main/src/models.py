from langchain_openai import AzureChatOpenAI
from src.constants import AZURE_gpt_4o_mini_CONFIG



gpt_4o_mini_llm=AzureChatOpenAI(
    openai_api_key=AZURE_gpt_4o_mini_CONFIG["api_key"],
    openai_api_version=AZURE_gpt_4o_mini_CONFIG["api_version"],
    azure_endpoint=AZURE_gpt_4o_mini_CONFIG["azure_endpoint"],
    deployment_name=AZURE_gpt_4o_mini_CONFIG["deployment_name"],
)

if not gpt_4o_mini_llm:
    raise Exception("LLM Initialization failed") 

#fine working    
# if __name__=="__main__":
#     print(gpt_4o_mini_llm.invoke("Hello, world!"))