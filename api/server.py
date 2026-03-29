import os
import time
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ValidationError

from env.environment import environment
from env.models import (
    Action, Observation, EpisodeState,
    DifficultyLevel, ActionType,
    StepResponse, ResetResponse, TaskListResponse,
    BaselineResponse, BaselineResult,
    GraderRequest, GraderResponse,
    HealthResponse, TaskInfo
)
from env.tasks import task_manager, ACTION_SCHEMA
from env.graders import grade


# ─────────────────────────────────────────────
#  STARTUP / SHUTDOWN
# ─────────────────────────────────────────────

_startup_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    environment.reset(difficulty="easy")
    yield

# ─────────────────────────────────────────────
#  APP DEFINITION
# ─────────────────────────────────────────────

app = FastAPI(
    title       = "SQL Query Debugger — OpenEnv Environment",
    description = (
        "An OpenEnv-compliant reinforcement learning environment where AI agents "
        "learn to debug SQL queries across syntax errors, logic bugs, and performance issues. "
        "Built for the META x PyTorch x SST OpenEnv Hackathon."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────
#  GLOBAL EXCEPTION HANDLER
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code = 500,
        content     = {"error": str(exc), "type": type(exc).__name__}
    )


# ─────────────────────────────────────────────
#  FAVICON — fix 404
# ─────────────────────────────────────────────

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Returns 204 No Content instead of 404 for favicon requests."""
    return Response(status_code=204)


# ─────────────────────────────────────────────
#  1. /health — GET
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check. Always returns 200. Used by HF Space health monitoring."""
    return HealthResponse(
        status  = "ok",
        version = "1.0.0",
        uptime  = round(time.time() - _startup_time, 2)
    )


# ─────────────────────────────────────────────
#  2. /reset — POST
# ─────────────────────────────────────────────

class ResetBody(BaseModel):
    difficulty: Optional[str] = None
    task_id:    Optional[str] = None

@app.post("/reset", response_model=Observation, tags=["Environment"])
async def reset(body: ResetBody = ResetBody()):
    """
    Starts a fresh episode. Returns the initial Observation the agent sees.
    Edge case: always returns valid Observation even if dataset issues occur.
    """
    try:
        obs = environment.reset(
            difficulty = body.difficulty,
            task_id    = body.task_id
        )
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


# ─────────────────────────────────────────────
#  3. /step — POST
# ─────────────────────────────────────────────

@app.post("/step", response_model=StepResponse, tags=["Environment"])
async def step(action: Action):
    """
    Submits an action to the environment.
    Returns (observation, reward, done, info).
    Edge cases: null action, malformed payload, episode already done.
    """
    try:
        response = environment.step(action)
        return response
    except ValidationError as e:
        from env.models import Reward
        return StepResponse(
            observation = environment._build_observation(),
            reward      = Reward(
                score     = -0.1,
                breakdown = {"validation_error": -0.1},
                feedback  = f"Malformed action: {str(e)}"
            ),
            done = False,
            info = {"error": "validation_error", "detail": str(e)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


# ─────────────────────────────────────────────
#  4. /state — GET
# ─────────────────────────────────────────────

@app.get("/state", response_model=EpisodeState, tags=["Environment"])
async def state():
    """
    Returns full current environment state.
    Works before reset() is called — returns default empty state.
    Always JSON-serializable. Never crashes.
    """
    return environment.state()


# ─────────────────────────────────────────────
#  5. /tasks — GET
# ─────────────────────────────────────────────

@app.get("/tasks", response_model=TaskListResponse, tags=["Tasks"])
async def tasks():
    """
    Lists all 15 tasks with full action schema definitions.
    Validator checks for action field definitions, not just task names.
    """
    all_tasks = task_manager.list_all_tasks()
    return TaskListResponse(
        tasks        = all_tasks,
        total        = len(all_tasks),
        action_types = [a.value for a in ActionType]
    )


# ─────────────────────────────────────────────
#  6. /grader — POST
# ─────────────────────────────────────────────

@app.post("/grader", response_model=GraderResponse, tags=["Grading"])
async def grader(request: GraderRequest):
    """
    Grades a completed episode action.
    Returns float score 0.0-1.0. Never crashes.
    Edge cases: null action → 0.0, unknown task → 0.0.
    """
    try:
        if request.action is None:
            return GraderResponse(
                score     = 0.0,
                feedback  = "No action provided for grading.",
                breakdown = {"error": "null_action"}
            )
        score, breakdown, feedback = grade(request.action, request.task_id)
        return GraderResponse(
            score     = score,
            feedback  = feedback,
            breakdown = breakdown
        )
    except Exception as e:
        return GraderResponse(
            score     = 0.0,
            feedback  = f"Grader error: {str(e)}",
            breakdown = {"error": str(e)}
        )


# ─────────────────────────────────────────────
#  7. /baseline — POST
# ─────────────────────────────────────────────

@app.post("/baseline", response_model=BaselineResponse, tags=["Baseline"])
async def baseline():
    """
    Runs the baseline agent against all 3 difficulty levels.
    Returns scores JSON. Must complete within 60 seconds.
    Edge case: OPENAI_API_KEY not set → continues with rule-based agent.
    """
    try:
        import baseline as baseline_module
        results = await asyncio.wait_for(
            asyncio.to_thread(baseline_module.run_baseline),
            timeout=55.0
        )
        return results
    except asyncio.TimeoutError:
        return BaselineResponse(
            results=[
                BaselineResult(
                    task_id    = "timeout",
                    difficulty = DifficultyLevel.EASY,
                    score      = 0.0,
                    steps      = 0,
                    feedback   = "Baseline timed out after 55 seconds."
                )
            ],
            average_score=0.0
        )
    except Exception as e:
        return BaselineResponse(
            results=[
                BaselineResult(
                    task_id    = "error",
                    difficulty = DifficultyLevel.EASY,
                    score      = 0.0,
                    steps      = 0,
                    feedback   = f"Baseline error: {str(e)}"
                )
            ],
            average_score=0.0
        )


# ─────────────────────────────────────────────
#  ROOT — project info
# ─────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    return {
        "name":        "SQL Query Debugger — OpenEnv Environment",
        "version":     "1.0.0",
        "docs":        "/docs",
        "health":      "/health",
        "endpoints":   ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline", "/health"],
        "hackathon":   "META x PyTorch x SST OpenEnv Hackathon",
        "domain":      "SQL Query Debugging",
        "tasks_count": 15,
    }