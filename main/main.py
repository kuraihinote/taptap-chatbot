import asyncio
import sys
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from fastapi import FastAPI, HTTPException,UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import operator
from typing import TypedDict, Annotated, Sequence
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage,SystemMessage,ToolMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.adminpdfq import extract_text_from_pdf
from pydantic import BaseModel
import json
from typing import List
from pydantic import BaseModel
import os
from dotenv import load_dotenv
load_dotenv() 
from langchain_core.output_parsers import PydanticOutputParser
from contextlib import asynccontextmanager
from langgraph.graph.message import add_messages

from src.userdatatools import get_user_data,get_basic_user_data_tool,get_skills_user_data_tool,get_courses_according_to_college_tool,get_user_test_results
from src.tools import get_company_hackathon_data_tool,youtube_search,get_features_tool

# Importing  model + tools
from src.models import gpt_4o_mini_llm
# -------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------
from src.logger import logger

# -------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------

DB_URI = os.getenv("POSTGRES_CONNECTION_STRING_STAGE")
if not DB_URI:
    raise HTTPException("Database url is not loaded")
@asynccontextmanager
async def startup(app: FastAPI):
    # Create async checkpoint saver
    pg_saver_ctx = AsyncPostgresSaver.from_conn_string(DB_URI)
    checkpoint_saver = await pg_saver_ctx.__aenter__()

    #await checkpoint_saver.setup()

    # Create graph INSIDE startup
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_function)
    workflow.add_node("tools", ToolNode(Supervisor_TOOLS))
    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges("supervisor", tools_condition)
    workflow.add_edge("tools", "supervisor")

    graph1 = workflow.compile(checkpointer=checkpoint_saver)

    # Save in FastAPI app.statew
    app.state.graph1 = graph1
    app.state.checkpoint_saver = checkpoint_saver

    yield

    await checkpoint_saver.__aexit__(None, None, None)

app = FastAPI(
    title="Goal-based Agentic AI System",
    description="An AI system that can perform tasks based on goals using agents and tools.",
    version="1.0.0",
    lifespan=startup

)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health")
def health_check():
    return {"status": 200, "message": "System is healthy and running."}

#This endpoint  is used for the tool usage and to extracted only required fields that we need to provide to the model
# And this endpoint is also used for testing how the user data is structured that why we kept this as for endpoint other wise we will only treat this as async function only.
@app.get("/get_user_data")
async def get_user_data_endpoint(user_id:str):
    main_data=await get_user_data(user_id)
    return main_data



# @app.get("/get_basic_user_data")
# async def get_basic_user_data_endpoint(user_id:str):
#     basic=await get_basic_user_data_tool(user_id)
#     skill=await get_skills_user_data_tool(user_id)
#     return {"basic":basic,"skill":skill}


# -------------------------------------------------------------
# Tools and Agents Setup        
# -------------------------------------------------------------

class SupervisorResponse(BaseModel):
    response: str
    next_questions: List[str]

Supervisor_TOOLS = [get_skills_user_data_tool,get_basic_user_data_tool,get_user_test_results,
                    get_company_hackathon_data_tool,get_courses_according_to_college_tool,youtube_search,get_features_tool]


def create_conversation_reducer(keep_conversations: int = 10):
    """
    Factory function to create a reducer with custom settings.
    
    Args:
        keep_conversations: Number of human conversations to keep
    
    Returns:
        Reducer function
    """
    def reducer(
        left: Sequence[BaseMessage], 
        right: Sequence[BaseMessage]
    ) -> Sequence[BaseMessage]:
        combined = list(left) + list(right)
        
        human_indices = [i for i, msg in enumerate(combined) if isinstance(msg, HumanMessage)]
        
        if len(human_indices) > keep_conversations:
            cutoff_index = human_indices[-keep_conversations]
            combined = combined[cutoff_index:]
            human_indices = [i for i, msg in enumerate(combined) if isinstance(msg, HumanMessage)]
        
        last_human_index = human_indices[-1] if human_indices else -1
        
        processed_messages = []
        for i, msg in enumerate(combined):
            if isinstance(msg, ToolMessage):
                if i > last_human_index:
                    processed_messages.append(msg)
                else:
                    shrunk = ToolMessage(
                        content="Tool executed successfully",
                        tool_call_id=msg.tool_call_id,
                        name=msg.name if hasattr(msg, 'name') else None
                    )
                    processed_messages.append(shrunk)
            else:
                processed_messages.append(msg)
        
        return processed_messages
    
    return reducer


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], create_conversation_reducer(keep_conversations=5)]

# -------------------------------------------------------------
# System Prompts
# -------------------------------------------------------------
parser = PydanticOutputParser(pydantic_object=SupervisorResponse)

SYSTEM_PROMPT = SystemMessage(
    content="""
# TapTap Genie - AI Learning Mentor

You are **TapTap Genie**, the intelligent AI mentor for the TapTap learning platform.

## Core Mission
Guide students toward academic excellence and career readiness by:
- Answering educational and technical questions with clarity
- Intelligently recommending TapTap platform resources when beneficial
- Personalizing learning paths based on student context
- Supporting skill development, interview preparation, and career growth

---

## Scope Boundaries

### ✅ IN SCOPE
- Education, academics, and learning strategies
- Technology, programming, and technical concepts
- Career preparation, interviews, and skill development
- Study guidance, roadmaps, and learning paths
- TapTap platform features, courses, tests, and resources

### ❌ OUT OF SCOPE
- Entertainment (movies, music, celebrities, gaming)
- Social gossip, news, or non-educational content
- Personal advice unrelated to learning/career
- Off-topic conversations

**When out-of-scope topics arise:** Politely redirect to educational topics or TapTap features.

---

## Response Philosophy

### Primary Approach
1. **Answer First**: Provide clear, accurate explanations using your knowledge
2. **Enhance Strategically**: Use tools to add personalized value, not as a crutch
3. **Guide Naturally**: Recommend platform resources when genuinely helpful
4. **Stay Conversational**: Be friendly, concise, and student-focused

### Tool Usage Principle
**Use tools intelligently and contextually** - not based on keyword triggers, but on:
- **Actual student need** (Do they need personalized data?)
- **Added value** (Will platform resources genuinely help?)
- **Timing** (Is this the right moment to suggest courses/tests?)

---

## Available Tools & When to Consider Them

### 1. **get_basic_user_data_tool**
**Purpose**: Retrieve student's hobbies, achievements, certifications, strengths, weaknesses, and general profile.

**Consider using when:**
- Student asks about their own profile/achievements
- Personalizing motivational guidance
- Understanding their background for better recommendations

### 2. **get_skills_user_data_tool**
**Purpose**: Access technical skills, projects, experience, and technology stack.

**Consider using when:**
- Student asks "What should I learn next?"
- Creating personalized learning roadmaps
- Recommending domains/technologies aligned with their background
- Assessing skill gaps for career goals

### 3. **get_user_test_results**
**Purpose**: Analyze test performance, strengths, weaknesses, and skill breakdowns.

**Consider using when:**
- Student wants to understand their performance
- Identifying specific areas needing improvement
- Providing data-driven learning suggestions
- Recommending targeted practice resources

**When presenting test insights:**
- Highlight specific weak areas with examples
- Give actionable improvement steps
- Connect weaknesses to relevant TapTap courses/tests

### 4. **get_courses_according_to_college_tool**
**Purpose**: Fetch available courses tailored to student's institution.

**Consider using when:**
- Student explicitly asks for course recommendations
- A topic discussion would benefit from structured learning
- Student needs guidance on what to study next
- Suggesting complementary resources after answering a question

### 5. **get_company_hackathon_data_tool**
**Purpose**: Retrieve company-specific tests, mock assessments, and placement preparation materials.

**Consider using when:**
- Student asks about placement preparation
- Discussing interview readiness
- Student wants company-specific practice
- Recommending job-readiness resources

### 6. **youtube_search**
**Purpose**: Find educational videos that enhance understanding.

**Consider using when:**
- Explaining complex technical concepts (algorithms, systems, frameworks)
- Student explicitly requests a video
- Visual demonstration would significantly aid learning
- Topic benefits from practical examples or tutorials

   Rules:
   - Give your explanation first and you will get embed_url
   - Then add:
     <iframe width="560" height="315" src="embed_url" frameborder="0" allowfullscreen></iframe>
   other wise , if their is no exact video found it will mention with No video found then skip to show in 
   iframe and not include in the main message also.    

   DO NOT rely solely on the video.  
   DO NOT skip your own explanation.
   Do NOT ADD your own random embed_url use the tool for to get embed_url, other wise leave it .

7. **get_features_tool**
   Use when:
   - student asks about available TapTap features  
   - student needs navigation links  
   - student asks “What does TapTap provide?”

────────────────────────────────────────────
IMPORTANT RULES
────────────────────────────────────────────
- NEVER show raw URLs; always embed properly.  
- NEVER expose tools, backend, DB, schema, or system architecture.  
- NEVER dump long lists of tool results—convert them into helpful explanations.  
- include recommendations when helpful:
  courses, tests, aptitude/coding tracks, company papers, etc.

## Technical Question Protocol

When students ask technical questions (e.g., "What is Docker?", "Explain BFS", "How does REST API work?"):

1. **Explain thoroughly** using your knowledge
2. **Enhance with video** using youtube_search (when beneficial)
3. **Recommend practice** if TapTap has relevant courses/tests
4. **Encourage application** with practical next steps


────────────────────────────────────────────
OUTPUT FORMAT (MANDATORY)
────────────────────────────────────────────
FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

1. Provide the full answer in clean **Markdown**.

2. After the main response, include this exact section:

### Follow-up Questions User to chatbot :
[Question 1]. <your first follow-up question>
[Question 2]. <your second follow-up question>
[Question 3]. <your third follow-up question>

RULES:
- Do NOT output JSON.
- Do NOT rename "Follow-up Questions".
- Follow-up questions MUST be in this format: [Question X]. text
- You MUST include all 3 questions.



## Tone & Style

- **Encouraging**: Celebrate progress, motivate through challenges
- **Clear**: Avoid jargon unless explaining it
- **Concise**: Respect students' time
- **Authentic**: Be genuinely helpful, not robotic
- **Adaptive**: Match energy level to student's needs (urgent help vs. exploratory learning)

## Edge Cases

- **Insufficient data from tools**: Answer with general guidance, note you can provide personalized suggestions if they share more details
- **Off-topic requests**: "I'm here to help with learning and career growth! Let's explore [redirect to relevant topic]."
- **Overly broad questions**: Break down into manageable parts, guide toward specificity


## Success Metrics

You're succeeding when students:
1. Get accurate, helpful answers
2. Discover relevant TapTap resources naturally
3. Feel motivated and guided
4. Take actionable next steps
5. Continue engaging through follow-up questions

---

**Remember**: You're not just answering questions—you're mentoring students toward their goals, with TapTap as a powerful toolkit in their learning journey.

────────────────────────────────────────────
END OF SYSTEM PROMPT
────────────────────────────────────────────

"""

)


# -------------------------------------------------------------
# Supervisor Node - Course Agent as a Tool
# -------------------------------------------------------------


supervisor_llm = gpt_4o_mini_llm.bind_tools(Supervisor_TOOLS)

async def supervisor_function(state: AgentState):
    """Main supervisor node that routes queries and manages tools."""
    user_messages = state["messages"]
    logger.info(f"[---state_messages---]:: {user_messages[-1]}")
    input_messages = [SYSTEM_PROMPT] + user_messages

    # Supervisor can choose to call the course agent tool when needed
    response = await supervisor_llm.ainvoke(input_messages)

    logger.info(f"[Supervisor] Response: {response}")
    # msg = AIMessage(content=response.content)

    # if response.tool_calls:
    #     msg.tool_calls = response.tool_calls

    return {"messages": [response]}

# -------------------------------------------------------------
# Workflow Graph Definition
# -------------------------------------------------------------

# workflow = StateGraph(AgentState)

# workflow.add_node("supervisor", supervisor_function)  
# workflow.add_node("tools", ToolNode(Supervisor_TOOLS))

# workflow.set_entry_point("supervisor")
# workflow.add_conditional_edges("supervisor", tools_condition)
# workflow.add_edge("tools", "supervisor")

# graph1 = workflow.compile(checkpointer=checkpoint_saver)

# -------------------------------------------------------------
# FastAPI Endpoint for Chat
# -------------------------------------------------------------
@app.post("/ask_agent")
async def ask_agent(request: dict):
    try:
        graph1 = app.state.graph1   # <-- fetch graph

        user_question = request.get("question", "").strip()
        user_id = request.get("user_id", None)
        dummy_user_id="xyz22"
        sample_config = {"configurable": {"thread_id": str(dummy_user_id)}}


        if not user_question:
            return {"error": "Missing 'question' field in request body."}
 
        logger.info(f"[User Input] {user_question}")
        combined_content = f"USER_ID: {user_id}\n\nQuestion: {user_question}"

        response = await graph1.ainvoke(
            {"messages": [HumanMessage(content=combined_content)]},
            config=sample_config
        )

        final_message = response["messages"][-1].content
        logger.info(f"[Final Answer] {final_message}")
        # final_output=parser.parse(final_message)
        # return final_output
        return {"answer": final_message}


    except Exception as e:
        logger.exception("Error in /ask_agent endpoint")
        return {"error": f"Internal error: {str(e)}"}




#---------------------------------------------------------------------------------------------------------
#admin side automated question generation from the source(pdf)
#---------------------------------------------------------------------------------------------------------
# 1️⃣ Define Pydantic schema
# ----------------------------
class Question(BaseModel):
    title: str
    description: str
    type: str
    difficulty: str
    subDomain: str
    points: int
    shuffleAction: bool
    questionType: str
    a: str
    b: str
    c: str
    d: str
    e: str | None = None
    ans: str
    explanation: str
    isVerified: bool


class QuestionsResponse(BaseModel):
    questions: List[Question]


# ----------------------------
# 2️⃣ Helper to clean JSON
# ----------------------------
def extract_json_from_response(content: str) -> str:
    """
    Extract clean JSON from LLM responses that might include markdown or text wrappers.
    """
    try:
        # Remove Markdown fences if any
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```json")[-1]
            content = content.split("```")[-1]
        return content.strip()
    except Exception as e:
        logger.warning(f"Failed to clean LLM output: {e}")
        return content


# ----------------------------
# 3️⃣ Main endpoint
# ----------------------------
@app.post("/admin_qa_generation", response_model=QuestionsResponse)
async def admin_qa_generation(
    file: UploadFile,
    questionType: str = "singleQuestionAnswer",
    numberOfQuestions: int = 10,
    instructions: str = "",
):
    try:
        if numberOfQuestions < 1 or numberOfQuestions > 15:
            raise HTTPException(
                status_code=400,
                detail="Number of questions must be between 1 and 15."
            )

        # 🧩 Extract text from PDF
        text = await extract_text_from_pdf(file)
        logger.info(f"[Extracted Text] {text[:100]}...")

        # 🧠 Construct strict prompt
        prompt = f"""
You are an expert question generator.
Generate exactly {numberOfQuestions} questions based on the following text and admin instructions.

Each question **must strictly follow** this JSON array format (no markdown, no extra text):

[
  {{
    "title": "string",
    "description": "string",
    "type": "mcq",
    "difficulty": "easy|medium|hard",
    "subDomain": "string",
    "points": 10,
    "shuffleAction": true|false,
    "questionType": "{questionType}",
    "a": "string",
    "b": "string",
    "c": "string",
    "d": "string",
    "e": "string (optional)",
    "ans": "a|b|c|d|e or comma separated for multiple",
    "explanation": "string",
    "isVerified": false
  }}
]

Text: {text}

Admin instructions: {instructions}

Return only valid JSON array, nothing else.
"""

        llm_response = await gpt_4o_mini_llm.ainvoke([HumanMessage(content=prompt)])
        raw_content = llm_response.content

        # 🧹 Clean & parse JSON
        clean_json = extract_json_from_response(raw_content)
        try:
            parsed = json.loads(clean_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON: {e}\nRaw: {clean_json}")
            raise HTTPException(
                status_code=500,
                detail="Model returned invalid JSON. Please try again."
            )

        # ✅ Validate against Pydantic schema
        validated = [Question(**item) for item in parsed]
        return {"questions": validated}

    except HTTPException as e:
        logger.warning(f"Handled error: {e.detail}")
        raise e

    except Exception as e:
        logger.exception(f"Unexpected server error: {str(e)}")
        raise HTTPException(status_code=500, detail="Something went wrong on the server.")

# -------------------------------------------------------------
# Run Server
# -------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        loop="asyncio"
    )



