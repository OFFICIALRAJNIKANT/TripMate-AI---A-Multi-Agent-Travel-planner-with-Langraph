import os
import operator
import uuid
from typing import Annotated, TypedDict

import certifi
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq

from tools.flight_tool import search_flights
from tools.tavily_tool import tavily_search


# =========================
# Environment
# =========================

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# =========================
# Configuration
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Please add it to your .env file."
    )


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


# =========================
# LLM
# =========================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
)


# =========================
# Utility
# =========================

def truncate_text(text: str, max_chars: int) -> str:
    """Limit text size before sending it to the LLM."""

    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n...[truncated]"


# =========================
# State
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


# =========================
# Flight Agent
# =========================

def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight results fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Hotel Agent
# =========================

def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Itinerary Agent
# =========================

def itinerary_agent(state: TravelState):
    flight_results = truncate_text(
        state["flight_results"],
        3500,
    )

    hotel_results = truncate_text(
        state["hotel_results"],
        3500,
    )

    prompt = f"""
Create a complete travel itinerary.

User Query:
{state["user_query"]}

Flight Results:
{flight_results}

Hotel Results:
{hotel_results}

Requirements:
- Make the itinerary practical and easy to follow.
- Consider the user's stated budget.
- Organize the trip day by day.
- Include sightseeing suggestions.
- Use the available flight and hotel information.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are an expert travel planner."
            ),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Final Response Agent
# =========================

def final_agent(state: TravelState):
    flight_results = truncate_text(
        state["flight_results"],
        2500,
    )

    hotel_results = truncate_text(
        state["hotel_results"],
        2500,
    )

    itinerary = truncate_text(
        state["itinerary"],
        5000,
    )

    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state["user_query"]}

Flights:
{flight_results}

Hotels:
{hotel_results}

Itinerary:
{itinerary}

Format the final answer using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

Important:
- Be clear and practical.
- Preserve important flight and hotel information.
- Do not invent flight prices.
- Mention that the live flight API may not provide ticket prices.
- Keep the response concise and useful for real travel planning.
- Avoid unnecessary repetition.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a professional AI travel booking assistant."
                )
            ),
            HumanMessage(content=final_prompt),
        ]
    )

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Build Graph
# =========================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# =========================
# PostgreSQL Checkpointer
# =========================

DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(
    checkpointer=checkpointer
)


# =========================
# Function for FastAPI
# =========================

def run_travel_agent(
    user_input: str,
    thread_id: str | None = None,
):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0,
        },
        config=config,
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get(
            "flight_results",
            "",
        ),
        "hotel_results": result.get(
            "hotel_results",
            "",
        ),
        "itinerary": result.get(
            "itinerary",
            "",
        ),
        "llm_calls": result.get(
            "llm_calls",
            0,
        ),
    }
