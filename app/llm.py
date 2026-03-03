import asyncio
import json
import logging
from typing import Dict, Any, List

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool

from .config import settings
from .tools import execute_sql_query

logger = logging.getLogger(__name__)


# Tool definition for LangChain
@tool
def query_database(query: str) -> dict:
    """Execute a SQL SELECT query on the PostgreSQL database."""
    return execute_sql_query(query)


class LLMProcessor:
    """LLM processor using Azure OpenAI with tool calling."""
    
    def __init__(self):
        if not settings.AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY not set in .env")
        
        self._client = AzureChatOpenAI(
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            temperature=0.1,
        )
        self._tools = [query_database]
        self._llm_with_tools = self._client.bind_tools(self._tools)
    
    def _system_prompt(self) -> str:
        """Generate the system prompt."""
        return (
            "You are an AI assistant for the TapTap Analytics Chatbot.\n"
            "You help faculty analyse student performance by querying a PostgreSQL database.\n\n"
            "DATABASE SCHEMA (high level):\n"
            "  public.user                          — id (PK), first_name, last_name, email, roll_number, ...\n"
            "  public.test_submission               — test submissions and scores\n"
            "  public.employability_track_submission — employability / placement assessments\n"
            "  pod.pod_submission                   — Problem of the Day submissions\n"
            "  pod.problem_of_the_day               — daily POD metadata\n\n"
            "INTENTS (what the faculty might ask):\n"
            "  • hackathon_overview              — high level metrics for a specific hackathon\n"
            "  • hackathon_participants          — which students joined a hackathon\n"
            "  • employability_overall_scores    — overall employability scores for a batch or group\n"
            "  • employability_section_scores    — aptitude / coding / reasoning breakdown\n"
            "  • test_top_performers             — top students in a specific test\n"
            "  • test_risk_students              — students below a risk threshold in a test\n"
            "  • pod_today_solvers               — who solved today's POD in a given domain\n"
            "  • pod_all_solvers                 — who solved POD historically in a given domain\n"
            "  • pod_streak_filter               — students with a POD streak above a threshold\n"
            "  • pod_top_scorers                 — top students by total POD score\n"
            "  • pod_failures                    — students who attempted POD but failed\n"
            "  • pod_problem_stats               — which POD problem had the most solvers or attempts (no domain needed)\n"
            "  • user_profile                    — details for a specific student\n\n"
            "RULES:\n"
            "  • Use only SELECT queries — never modify data.\n"
            "  • Always use schema-qualified table names (e.g. pod.pod_submission).\n"
            "  • JOIN public.user on the appropriate user_id column when you need names.\n"
            "  • Use CURRENT_DATE for 'today' when working with pod.problem_of_the_day.\n"
            "  • Apply LIMIT 50 unless the user clearly asks for more.\n"
            "  • After getting data, reply in clear natural language with the key facts.\n"
            "  • If the query returns no rows, say so clearly.\n"
        )
    
    # intent & slot configuration ------------------------------------------------
    # Each high‑level "intent" corresponds to a faculty use‑case.  We keep the
    # mapping to physical tables in _build_sql so the prompt stays business‑level.
    REQUIRED_SLOTS: Dict[str, List[str]] = {
        # Hackathons
        "hackathon_overview": ["hackathon_id"],
        "hackathon_participants": ["hackathon_id"],
        # Employability
        "employability_overall_scores": [],
        "employability_section_scores": ["section"],
        # Tests
        "test_top_performers": ["test_id"],
        "test_risk_students": ["test_id", "threshold"],
        # POD / daily problems
        "pod_today_solvers": ["domain_id"],
        "pod_all_solvers": ["domain_id"],
        "pod_streak_filter": ["threshold"],   
        "pod_top_scorers": [],                
        "pod_failures": [],                   
        "pod_problem_stats": [],
        # Student view
        "user_profile": ["user_id"],
    }

    async def _extract_intent_and_slots(self, user_input: str, state: Dict[str, Any]) -> (str, Dict[str, Any]):
        """Ask the LLM to detect the intent and pull out any parameters (slots).

        Returns a tuple ``(intent, slots)`` where ``intent`` is a simple string
        such as ``\"hackathon_overview\"`` and ``slots`` is a dict of keys
        and values. The model is instructed to emit **only** valid JSON.
        """
        prompt = (
            "You are a helper that maps faculty questions to analytics intents and "
            "extracts the required parameters (slots).\n\n"
            "Respond with a JSON object with two fields: \"intent\" and \"slots\".\n"
            "Allowed intents are:\n"
            "  - hackathon_overview\n"
            "  - hackathon_participants\n"
            "  - employability_overall_scores\n"
            "  - employability_section_scores\n"
            "  - test_top_performers\n"
            "  - test_risk_students\n"
            "  - pod_today_solvers\n"
            "  - pod_all_solvers\n"
            "  - pod_streak_filter\n"
            "  - pod_top_scorers\n"
            "  - pod_failures\n"
            "  - pod_problem_stats\n"
            "  - user_profile\n\n"
            "INTENT DISAMBIGUATION:\n"
            "  - pod_all_solvers     → asks WHO solved POD (returns student names)\n"
            "  - pod_problem_stats   → asks WHICH POD/problem had most solvers or attempts (returns problem titles and counts)\n"
            "  - pod_top_scorers     → asks which students have the highest total POD score\n\n"
            "\"intent\" must be exactly one of those strings (or null if you are unsure).\n"
            "\"slots\" must be an object containing any filters such as hackathon_id, "
            "test_id, user_id, batch, section (aptitude/coding/reasoning), "
            "threshold (risk score threshold), domain_id, or date.\n"
            "If you cannot decide, return null for intent and an empty object for slots.\n"
            "Do not output anything else.\n"
        )
        # Single conversation turn: system + user
        conversation = [SystemMessage(content=prompt), HumanMessage(content=user_input)]
        # Use the async chat interface and read the assistant's message content
        response = await self._client.ainvoke(conversation)
        text = response.content or ""
        try:
            obj = json.loads(text)
            intent = obj.get("intent")
            slots = obj.get("slots") or {}
            if not isinstance(slots, dict):
                slots = {}
            return intent, slots
        except json.JSONDecodeError:
            logger.warning("Intent/slot extraction returned non‑JSON: %s", text)
            return None, {}

    def _validate_slots(self, intent: str, slots: Dict[str, Any]) -> (bool, List[str]):
        """Check that we have all the required slots for the chosen intent."""
        missing: List[str] = []
        if intent in self.REQUIRED_SLOTS:
            for name in self.REQUIRED_SLOTS[intent]:
                if name not in slots or slots[name] in (None, ""):
                    missing.append(name)
        else:
            # Unknown intent: let the caller handle it as a clarification case.
            missing = []
        return len(missing) == 0, missing

    def _build_sql(self, intent: str, slots: Dict[str, Any]) -> str:
        """Construct a precise SELECT statement for each intent using real schema."""

        if intent == "pod_today_solvers":
            domain_id = slots.get("domain_id")
            domain_filter = f"AND ps.domain_id = {int(domain_id)}" if domain_id is not None else ""
            return (
                "SELECT DISTINCT "
                "u.first_name || ' ' || u.last_name AS student_name, "
                "u.email, "
                "p.title AS problem_title, "
                "p.difficulty, "
                "ps.status, "
                "ps.obtained_score AS score, "
                "ps.create_at AS submitted_at "
                "FROM pod.pod_submission ps "
                "JOIN public.user u ON u.id = ps.user_id "
                "JOIN public.problem p ON p.id = ps.question_id "
                "JOIN public.problem_of_the_day pod ON pod.id = ps.problem_of_the_day_id "
                f"WHERE pod.date = CURRENT_DATE "
                f"AND ps.status = 'pass' "
                f"{domain_filter} "
                "ORDER BY ps.create_at "
                "LIMIT 50;"
            )
        
        if intent == "pod_all_solvers":
            domain_id = slots.get("domain_id")
            domain_filter = f"AND ps.domain_id = {int(domain_id)}" if domain_id is not None else ""
            return (
                "SELECT DISTINCT "
                "u.first_name || ' ' || u.last_name AS student_name, "
                "u.email, "
                "p.title AS problem_title, "
                "p.difficulty, "
                "ps.status, "
                "ps.obtained_score AS score, "
                "ps.create_at AS submitted_at "
                "FROM pod.pod_submission ps "
                "JOIN public.user u ON u.id = ps.user_id "
                "JOIN public.problem p ON p.id = ps.question_id "
                f"WHERE ps.status = 'pass' "
                f"{domain_filter} "
                "ORDER BY ps.create_at DESC "
                "LIMIT 50;"
            )    

        if intent == "hackathon_overview":
            hackathon_id = slots.get("hackathon_id")
            return (
                "SELECT h.title, h.status, h.start_date, h.end_date, "
                "COUNT(DISTINCT r.user_id) AS participant_count, "
                "AVG(r.current_score) AS avg_score, "
                "MAX(r.current_score) AS top_score "
                "FROM public.hackathon h "
                "LEFT JOIN public.user_hackathon_participation r ON r.hackathon_id = h.id "
                f"WHERE h.id = {int(hackathon_id)} "
                "GROUP BY h.id, h.title, h.status, h.start_date, h.end_date "
                "LIMIT 1;"
            )

        if intent == "hackathon_participants":
            hackathon_id = slots.get("hackathon_id")
            return (
                "SELECT u.first_name || ' ' || u.last_name AS student_name, "
                "u.email, r.current_score AS score, r.start_time "
                "FROM public.user_hackathon_participation r "
                "JOIN public.user u ON u.id = r.user_id "
                f"WHERE r.hackathon_id = {int(hackathon_id)} "
                "ORDER BY r.current_score DESC "
                "LIMIT 50;"
            )

        if intent == "employability_overall_scores":
            return (
                "SELECT u.first_name || ' ' || u.last_name AS student_name, "
                "u.email, e.verbal_score, e.coding_score, e.reasoning_score, e.total_score "
                "FROM public.employability_track_submission e "
                "JOIN public.user u ON u.id = e.user_id "
                "ORDER BY e.total_score DESC "
                "LIMIT 50;"
            )

        if intent == "employability_section_scores":
            section = slots.get("section", "total")
            section_col_map = {
                "verbal": "e.verbal_score",
                "coding": "e.coding_score",
                "reasoning": "e.reasoning_score",
                "total": "e.total_score",
            }
            col = section_col_map.get(str(section).lower(), "e.total_score")
            return (
                "SELECT u.first_name || ' ' || u.last_name AS student_name, "
                f"u.email, {col} AS section_score "
                "FROM public.employability_track_submission e "
                "JOIN public.user u ON u.id = e.user_id "
                f"ORDER BY {col} DESC "
                "LIMIT 50;"
            )

        if intent == "test_top_performers":
            test_id = slots.get("test_id")
            return (
                "SELECT u.first_name || ' ' || u.last_name AS student_name, "
                "u.email, ts.score, ts.status, ts.create_at "
                "FROM public.test_submission ts "
                "JOIN public.user u ON u.id = ts.user_id "
                f"WHERE ts.problem_id = {int(test_id)} "
                "ORDER BY ts.score DESC "
                "LIMIT 50;"
            )

        if intent == "test_risk_students":
            test_id = slots.get("test_id")
            threshold = slots.get("threshold", 40)
            return (
                "SELECT u.first_name || ' ' || u.last_name AS student_name, "
                "u.email, ts.score, ts.status "
                "FROM public.test_submission ts "
                "JOIN public.user u ON u.id = ts.user_id "
                f"WHERE ts.problem_id = {int(test_id)} "
                f"AND ts.score < {int(threshold)} "
                "ORDER BY ts.score ASC "
                "LIMIT 50;"
            )

        if intent == "pod_streak_filter":
            threshold = slots.get("threshold", 5)
            return (
                "SELECT u.first_name || ' ' || u.last_name AS student_name, "
                "u.email, ps.streak_count, ps.is_active, ps.start_date, ps.end_date "
                "FROM public.pod_streak ps "
                "JOIN public.user u ON u.id = ps.user_id "
                f"WHERE ps.streak_count > {int(threshold)} "
                "ORDER BY ps.streak_count DESC "
                "LIMIT 50;"
            )

        if intent == "pod_top_scorers":
            return (
                "SELECT u.first_name || ' ' || u.last_name AS student_name, "
                "u.email, "
                "COUNT(ps.id) AS total_solved, "
                "SUM(ps.obtained_score) AS total_score "
                "FROM pod.pod_submission ps "
                "JOIN public.user u ON u.id = ps.user_id "
                "WHERE ps.status = 'pass' "
                "GROUP BY u.id, u.first_name, u.last_name, u.email "
                "ORDER BY total_score DESC "
                "LIMIT 50;"
            )

        if intent == "pod_failures":
            return (
                "SELECT u.first_name || ' ' || u.last_name AS student_name, "
                "u.email, "
                "p.title AS problem_title, "
                "ps.status, "
                "ps.create_at AS attempted_at "
                "FROM pod.pod_submission ps "
                "JOIN public.user u ON u.id = ps.user_id "
                "JOIN public.problem p ON p.id = ps.question_id "
                "WHERE ps.status = 'fail' "
                "ORDER BY ps.create_at DESC "
                "LIMIT 50;"
            )
            
        if intent == "pod_problem_stats":
            domain_id = slots.get("domain_id")
            domain_join_filter = f"AND ps.domain_id = {int(domain_id)}" if domain_id is not None else ""
            return (
                "SELECT p.title AS problem_title, "
                "p.difficulty, "
                "pod.date AS pod_date, "
                "COUNT(ps.id) AS total_attempts, "
                "COUNT(DISTINCT CASE WHEN ps.status = 'pass' THEN ps.user_id END) AS unique_solvers "
                "FROM public.problem_of_the_day pod "
                "JOIN public.problem p ON p.id = pod.question_id "
                f"LEFT JOIN pod.pod_submission ps ON ps.problem_of_the_day_id = pod.id {domain_join_filter} "
                "GROUP BY p.title, p.difficulty, pod.date "
                "ORDER BY unique_solvers DESC "
                "LIMIT 50;"
            )
          

        if intent == "user_profile":
            user_id = slots.get("user_id")
            safe_id = str(user_id).replace("'", "''")
            return (
                "SELECT u.id, u.first_name, u.last_name, u.email, u.phone, "
                "u.roll_number, u.role, u.is_placed, u.active_streak_count "
                "FROM public.user u "
                f"WHERE u.id = '{safe_id}' OR u.email ILIKE '%{safe_id}%' "
                "LIMIT 1;"
            )

        raise ValueError(f"Unknown intent '{intent}'")

    async def _generate_response(self, intent: str, slots: Dict[str, Any],
                                 query_result: Dict[str, Any]) -> str:
        """Summarise the query result in plain English using the LLM."""
        # query_result may contain datetime objects, which are not JSON serialisable
        # by default, so we convert unknown types to strings.
        safe_result = json.dumps(query_result, default=str)
        summary_prompt = (
            f"The user asked for data with intent '{intent}' and slots {slots}. "
            f"Here are the results as JSON: {safe_result}\n"
            "Write a clear, concise natural language summary of the key facts for a faculty member.\n"
        )
        messages = [
            SystemMessage(content=self._system_prompt()),
            HumanMessage(content=summary_prompt),
        ]
        response = await self._client.ainvoke(messages)
        return response.content or ""

    async def process_user_query(self, user_query: str, state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a user query following the high‑level workflow.

        ``state`` is a mutable dictionary that can be supplied by the caller to
        keep track of conversation context between turns.  A new empty dict is
        created if none is provided.
        """
        state = state or {}
        try:
            # 1. intent & slot extraction
            intent, slots = await self._extract_intent_and_slots(user_query, state)
            logger.info("detected intent=%s slots=%s", intent, slots)

            if not intent:
                answer = (
                    "This chatbot only answers questions about tests, POD, hackathons and employability. "
                    "Your question seems outside that. Try rephrasing about those topics."
                )
                state["intent"] = None
                return {"answer": answer, "data": [], "state": state, "success": True}

            # 1b. intent-specific slot normalisation / resolution
            if intent == "pod_today_solvers":
                # We expect a numeric domain_id FK in pod.pod_submission, but
                # the LLM may give us a name like "IT". We try to resolve that
                # via public.domains; if we cannot, we return options so the
                # frontend can show recommended choices.
                raw_domain_id = slots.get("domain_id")
                domain_name = None

                # If we received a non-numeric domain_id, treat it as a name.
                if isinstance(raw_domain_id, str) and not raw_domain_id.isdigit():
                    domain_name = raw_domain_id
                # Or, if the LLM used a separate slot name like domain_name
                if not domain_name and isinstance(slots.get("domain_name"), str):
                    domain_name = slots["domain_name"]

                if domain_name:
                    # Try to resolve by exact / case-insensitive match in public.domains
                    safe_name = domain_name.replace("'", "''")
                    lookup_sql = (
                        "SELECT id, domain FROM public.domains "
                        f"WHERE domain ILIKE '{safe_name}' "
                        "ORDER BY domain"
                    )
                    lookup_result = await execute_sql_query(lookup_sql)

                    if lookup_result.get("success") and lookup_result.get("data"):
                        rows = lookup_result["data"]
                        if len(rows) == 1:
                            # Unique match: bind the numeric id and continue.
                            slots["domain_id"] = rows[0]["id"]
                        else:
                            # Ambiguous: multiple similar domains, ask the user to pick.
                            options = [
                                {"id": r["id"], "label": r.get("domain", str(r["id"]))}
                                for r in rows
                            ]
                            state["intent"] = intent
                            state["missing_slots"] = ["domain_id"]
                            state["slot_options"] = {"domain_id": options}
                            state["slot_hints"] = {"domain_name": domain_name}
                            question = (
                                f"I found multiple domains matching '{domain_name}'. "
                                "Please choose the correct domain from the options shown."
                            )
                            return {
                                "answer": question,
                                "data": [],
                                "state": state,
                                "success": True,
                            }
                    else:
                        # No direct match; fall back to listing all available domains.
                        all_sql = (
                            "SELECT id, domain FROM public.domains "
                            "ORDER BY domain"
                        )
                        all_result = await execute_sql_query(all_sql)
                        options = [
                            {"id": r["id"], "label": r.get("domain", str(r["id"]))}
                            for r in all_result.get("data", [])
                        ]
                        state["intent"] = intent
                        state["missing_slots"] = ["domain_id"]
                        state["slot_options"] = {"domain_id": options}
                        state["slot_hints"] = {"domain_name": domain_name}
                        question = (
                            f"I couldn't find a domain matching '{domain_name}'. "
                            "Please choose one of the available domains, or ask again "
                            "using one of these domain names explicitly."
                        )
                        return {
                            "answer": question,
                            "data": [],
                            "state": state,
                            "success": True,
                        }

                # If we now have a numeric domain_id as a string, normalise to int
                if isinstance(slots.get("domain_id"), str) and slots["domain_id"].isdigit():
                    slots["domain_id"] = int(slots["domain_id"])

            if intent in ("hackathon_overview", "hackathon_participants"):
                # For hackathons we expect a numeric hackathon_id, but faculty
                # may refer to the title. Try to resolve by title; otherwise
                # return options so the frontend can show recommended choices.
                raw_hackathon_id = slots.get("hackathon_id")
                hackathon_name = None

                if isinstance(raw_hackathon_id, str) and not raw_hackathon_id.isdigit():
                    hackathon_name = raw_hackathon_id
                if not hackathon_name and isinstance(slots.get("hackathon_name"), str):
                    hackathon_name = slots["hackathon_name"]

                if hackathon_name:
                    safe_name = hackathon_name.replace("'", "''")
                    lookup_sql = (
                        "SELECT id, title FROM public.hackathon "
                        f"WHERE title ILIKE '%{safe_name}%' "
                        "ORDER BY start_date DESC"
                    )
                    lookup_result = await execute_sql_query(lookup_sql)

                    if lookup_result.get("success") and lookup_result.get("data"):
                        rows = lookup_result["data"]
                        if len(rows) == 1:
                            slots["hackathon_id"] = rows[0]["id"]
                        else:
                            options = [
                                {"id": r["id"], "label": r.get("title", f"Hackathon {r['id']}")}
                                for r in rows
                            ]
                            state["intent"] = intent
                            state["missing_slots"] = ["hackathon_id"]
                            state["slot_options"] = {"hackathon_id": options}
                            state["slot_hints"] = {"hackathon_name": hackathon_name}
                            question = (
                                f"I found multiple hackathons matching '{hackathon_name}'. "
                                "Please choose the correct hackathon from the options shown."
                            )
                            return {
                                "answer": question,
                                "data": [],
                                "state": state,
                                "success": True,
                            }
                    else:
                        all_sql = (
                            "SELECT id, title FROM public.hackathon "
                            "WHERE status = 'published' "
                            "ORDER BY start_date DESC LIMIT 20"
                        )
                        all_result = await execute_sql_query(all_sql)
                        options = [
                            {"id": r["id"], "label": r.get("title", f"Hackathon {r['id']}")}
                            for r in all_result.get("data", [])
                        ]
                        state["intent"] = intent
                        state["missing_slots"] = ["hackathon_id"]
                        state["slot_options"] = {"hackathon_id": options}
                        state["slot_hints"] = {"hackathon_name": hackathon_name}
                        question = (
                            f"I couldn't find a hackathon matching '{hackathon_name}'. "
                            "Please choose one of the available hackathons, or ask again "
                            "using one of their titles explicitly."
                        )
                        return {
                            "answer": question,
                            "data": [],
                            "state": state,
                            "success": True,
                        }

                if isinstance(slots.get("hackathon_id"), str) and slots["hackathon_id"].isdigit():
                    slots["hackathon_id"] = int(slots["hackathon_id"])

            if intent in ("test_top_performers", "test_risk_students"):
                # For tests we expect a numeric test_id (or problem_id depending on context).
                # Faculty might refer to a test or round by its name. Try to resolve across
                # round, test_type, and test_category tables.
                raw_test_id = slots.get("test_id")
                test_name = None

                if isinstance(raw_test_id, str) and not raw_test_id.isdigit():
                    test_name = raw_test_id
                if not test_name and isinstance(slots.get("test_name"), str):
                    test_name = slots["test_name"]
                if not test_name and isinstance(slots.get("round_name"), str):
                    test_name = slots["round_name"]

                if test_name:
                    safe_name = test_name.replace("'", "''")
                    # Try to resolve in round table first
                    lookup_sql = (
                        "SELECT id, name AS title FROM public.round "
                        f"WHERE name ILIKE '%{safe_name}%' "
                        "UNION "
                        "SELECT id, type_name AS title FROM public.test_type "
                        f"WHERE type_name ILIKE '%{safe_name}%' "
                        "UNION "
                        "SELECT id, category_name AS title FROM public.test_category "
                        f"WHERE category_name ILIKE '%{safe_name}%' "
                        "LIMIT 20"
                    )
                    
                    lookup_result = await execute_sql_query(lookup_sql)

                    if lookup_result.get("success") and lookup_result.get("data"):
                        rows = lookup_result["data"]
                        if len(rows) == 1:
                            slots["test_id"] = rows[0]["id"]
                        else:
                            options = [
                                {"id": r["id"], "label": r.get("title", f"Test {r['id']}")}
                                for r in rows
                            ]
                            state["intent"] = intent
                            state["missing_slots"] = ["test_id"]
                            state["slot_options"] = {"test_id": options}
                            state["slot_hints"] = {"test_name": test_name}
                            question = (
                                f"I found multiple tests/rounds matching '{test_name}'. "
                                "Please choose the correct one from the options shown."
                            )
                            return {
                                "answer": question,
                                "data": [],
                                "state": state,
                                "success": True,
                            }
                    else:
                        all_sql = (
                            "SELECT id, name AS title FROM public.round ORDER BY id DESC LIMIT 10 "
                        )
                        all_result = await execute_sql_query(all_sql)
                        options = [
                            {"id": r["id"], "label": r.get("title", f"Round {r['id']}")}
                            for r in all_result.get("data", [])
                        ]
                        state["intent"] = intent
                        state["missing_slots"] = ["test_id"]
                        state["slot_options"] = {"test_id": options}
                        state["slot_hints"] = {"test_name": test_name}
                        question = (
                            f"I couldn't find a test matching '{test_name}'. "
                            "Please choose one of the recent rounds, or ask again "
                            "using an exact test name."
                        )
                        return {
                            "answer": question,
                            "data": [],
                            "state": state,
                            "success": True,
                        }

                if isinstance(slots.get("test_id"), str) and slots["test_id"].isdigit():
                    slots["test_id"] = int(slots["test_id"])

            # 2. validate required slots for this intent
            valid, missing = self._validate_slots(intent, slots)
            if not valid:
                # Intent-specific recommended patterns to help the user phrase
                # a follow-up query that will succeed.
                recommendations = {
                    "test_top_performers": (
                        "For example: \"Top 10 students in coding for test_id = 4\"."
                    ),
                    "test_risk_students": (
                        "For example: \"Students with score below 40 in test_id = 4\"."
                    ),
                    "hackathon_overview": (
                        "For example: \"Show overview for hackathon_id = 10\" "
                        "or mention the hackathon title."
                    ),
                    "hackathon_participants": (
                        "For example: \"List participants in hackathon_id = 10\"."
                    ),
                    "pod_today_solvers": (
                        "For example: \"Who solved today's POD in domain 'IT'?\" "
                        "or \"... in domain_id = 1\"."
                    ),
                }
                base = (
                    "I need a bit more information to answer that. "
                    f"Please provide the following: {', '.join(missing)}."
                )
                extra = recommendations.get(intent, "")
                question = f"{base} {extra}".strip()
                # update state so intent and missing slots are remembered for the frontend
                state["intent"] = intent
                state["missing_slots"] = missing
                return {
                    "answer": question,
                    "data": [],
                    "state": state,
                    "success": True,
                }

            # 3. build SQL and run tool
            sql = self._build_sql(intent, slots)
            logger.info("built sql: %s", sql)
            result = await execute_sql_query(sql)

            # If the DB call failed (e.g. missing table), return a friendly message
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown database error")
                answer = (
                    "I tried to look up that information in the database, "
                    "but the underlying table or query is not available right now. "
                    f"Technical details: {error_msg}"
                )
                state.update({"intent": intent, **slots})
                return {
                    "answer": answer,
                    "data": [],
                    "state": state,
                    "success": True,
                }

            # 4. final response
            answer = await self._generate_response(intent, slots, result)

            # 5. save state
            state.update({"intent": intent, **slots})
            return {
                "answer": answer,
                "data": result.get("data", []),
                "state": state,
                "success": result.get("success", True),
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "answer": f"Sorry, I encountered an error: {e}",
                "data": [],
                "state": state or {},
                "success": False,
            }
    
    def _flatten(self, results: List[Dict]) -> List[Dict]:
        """Flatten tool results into a single list of rows."""
        rows = []
        for r in results:
            if r.get("success") and r.get("data"):
                rows.extend(r["data"])
        return rows


llm_processor: Any = None


async def init_llm() -> None:
    """Initialize the LLM processor."""
    global llm_processor
    try:
        llm_processor = LLMProcessor()
        logger.info("LLM processor initialised (Azure OpenAI - gpt-4o-mini)")
    except Exception as e:
        logger.error(f"Failed to initialise LLM processor: {e}")
        raise


async def process_user_query(user_query: str, state: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process a user query; optionally supply a mutable ``state`` dict.

    The returned dictionary will include an updated ``state`` key so callers
    can continue the conversation in subsequent requests.
    """
    global llm_processor
    if not llm_processor:
        raise RuntimeError("LLM processor not initialised")
    return await llm_processor.process_user_query(user_query, state)
