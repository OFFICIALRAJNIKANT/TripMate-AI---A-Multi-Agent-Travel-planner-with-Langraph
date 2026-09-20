from pathlib import Path
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend import run_travel_agent


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="TripMate AI",
    description="LangGraph Multi-Agent Travel Planner with FastAPI Frontend",
    version="1.0.0",
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


# ============================================================
# TEMPLATES
# ============================================================

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)


# ============================================================
# REQUEST MODEL
# ============================================================

class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


# ============================================================
# TRAVEL PLANNER API
# ============================================================

@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):

    try:
        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty.",
                },
            )

        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id,
        )

        return JSONResponse(
            content={
                "success": True,
                "thread_id": result["thread_id"],
                "answer": result["answer"],
                "flight_results": result["flight_results"],
                "hotel_results": result["hotel_results"],
                "itinerary": result["itinerary"],
                "llm_calls": result["llm_calls"],
            }
        )

    except Exception as e:

        print("\n" + "=" * 60)
        print("TRIPMATE API ERROR")
        print("=" * 60)

        traceback.print_exc()

        print("=" * 60 + "\n")

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
            },
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "AI Travel Planner API is running",
    }


# ============================================================
# FAVICON
# ============================================================

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )