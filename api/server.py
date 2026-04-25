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
    HealthResponse, TaskInfo, ProgressResponse
)
from env.tasks import task_manager, ACTION_SCHEMA
from env.graders import grade, grade_db_action, _is_scenario_task, _get_scenario


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
    title       = "SQL Database Engineer Agent — OpenEnv Environment",
    description = (
        "An OpenEnv-compliant reinforcement learning environment where AI agents "
        "learn to act like senior database engineers. "
        "The agent manages a simulated production database over 50+ steps: "
        "inspecting slow queries, creating indexes, rewriting queries, partitioning tables. "
        "Built for the META x PyTorch x SST OpenEnv Hackathon Finals — April 25-26, Bangalore."
    ),
    version     = "2.0.0",
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
#  FAVICON
# ─────────────────────────────────────────────

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


# ─────────────────────────────────────────────
#  1. /health — GET
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check. Always returns 200."""
    return HealthResponse(
        status  = "ok",
        version = "2.0.0",
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
    Starts a fresh episode. Initializes DatabaseSimulator.
    Returns the initial Observation with DB state and slow queries.
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
    Round 2 actions: inspect_query, create_index, rewrite_query,
    partition_table, analyze_statistics, analyze_indexes, submit_report.
    Returns (observation, reward, done, info) with DB performance delta.
    """
    try:
        response = environment.step(action)
        return response
    except ValidationError as e:
        from env.models import Reward
        return StepResponse(
            observation = environment._build_observation(),
            reward      = Reward(
                score     = 0.001,
                breakdown = {"validation_error": 0.001},
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
    """Returns full current environment state including performance history."""
    return environment.state()


# ─────────────────────────────────────────────
#  5. /tasks — GET
# ─────────────────────────────────────────────

@app.get("/tasks", response_model=TaskListResponse, tags=["Tasks"])
async def tasks():
    """
    Lists all 30 tasks (15 Round 2 scenarios + 15 Round 1 cases).
    Includes complete action schema for all 15 action types.
    """
    all_tasks = task_manager.list_all_tasks()
    return TaskListResponse(
        tasks        = all_tasks,
        total        = len(all_tasks),
        action_types = [a.value for a in ActionType]
    )


# ─────────────────────────────────────────────
#  6. /grader — POST  (FIXED)
# ─────────────────────────────────────────────

@app.post("/grader", response_model=GraderResponse, tags=["Grading"])
async def grader(request: GraderRequest):
    """
    Grades an action for a given task_id. STATELESS — does not change episode state.

    Routing:
      Round 2 scenario IDs (easy_s001, medium_s002, hard_s003):
        - submit_report   → computes score from current DB performance delta
        - all other types → grade_db_action() scores action quality vs scenario

      Round 1 task IDs (easy_001, medium_001, hard_001):
        → grade() → grade_easy/medium/hard() (original Round 1 graders)

    Score is ALWAYS strictly between 0.001 and 0.999.
    NEVER crashes — all exceptions caught and returned as 0.001.

    FIXES applied vs original:
      - Round 2 non-terminal actions now route to grade_db_action() instead of
        grade_easy() which was looking for "fixed_query" in Round 2 payloads
        and returning 0.001 for every create_index / analyze_indexes / inspect_query
      - submit_report score now uses db_simulator state from environment directly
        instead of brittle action_counts dict lookup which could be empty or stale
    """
    try:
        if request.action is None:
            return GraderResponse(
                score     = 0.001,
                feedback  = "No action provided for grading.",
                breakdown = {"error": "null_action"}
            )

        task_id     = request.task_id or ""
        action_type = (
            request.action.action_type.value
            if hasattr(request.action.action_type, "value")
            else str(request.action.action_type)
        )

        # ── ROUND 2: DB ENGINEERING SCENARIO ─────────────────────
        if _is_scenario_task(task_id):

            # submit_report: use live DB state from environment simulator
            if action_type == "submit_report":
                return _grade_submit_report(request, task_id)

            # All other Round 2 actions: stateless scenario-aware grading
            score, breakdown, feedback = grade_db_action(request.action, task_id)
            score = max(0.001, min(0.999, score))
            return GraderResponse(score=score, feedback=feedback, breakdown=breakdown)

        # ── ROUND 1: SQL DEBUGGING TASK ───────────────────────────
        score, breakdown, feedback = grade(request.action, task_id)
        score = max(0.001, min(0.999, score))
        return GraderResponse(score=score, feedback=feedback, breakdown=breakdown)

    except Exception as e:
        return GraderResponse(
            score     = 0.001,
            feedback  = f"Grader error: {str(e)}",
            breakdown = {"error": str(e)}
        )


def _grade_submit_report(request: GraderRequest, task_id: str) -> GraderResponse:
    """
    Grade a submit_report action for a Round 2 scenario.

    Score components:
      60% — performance improvement (baseline → current)
      20% — step efficiency (fewer steps = higher bonus)
      10% — base credit for submitting
      10% — report summary quality

    Falls back gracefully if DB simulator state is unavailable.
    """
    try:
        ep_state = environment.state()

        # Get performance data from environment state
        # Use action_counts as the store (set by environment.py during steps)
        ac          = ep_state.action_counts or {}
        perf_history = ac.get("_perf_history", [])
        baseline     = float(ac.get("_baseline_score", 0.0))
        current      = float(perf_history[-1]) if perf_history else baseline
        steps_used   = ep_state.step_count
        max_steps    = 50  # Round 2 default

        # If no perf history (called before reset, or env in wrong state):
        # fall back to scenario-based quality score
        if not perf_history or baseline == 0.0:
            scenario = _get_scenario(task_id)
            if scenario:
                baseline = float(scenario.get("performance_score_baseline", 0.0))
                target   = float(scenario.get("target_score", 85.0))
                # Score based on report quality only
                summary    = str((request.action.payload or {}).get("summary", ""))
                base_score = 0.15 + min(len(summary) / 400, 0.25)
                return GraderResponse(
                    score    = round(max(0.001, min(0.999, base_score)), 4),
                    feedback = (
                        f"Report graded on quality only (episode state unavailable). "
                        f"Run a full episode via /reset then /step to get performance-based score."
                    ),
                    breakdown = {"report_quality": round(base_score, 4), "note": "no_episode_state"}
                )

        max_possible     = max(1.0, 100.0 - baseline)
        perf_improvement = max(0.0, (current - baseline) / max_possible)
        step_efficiency  = max(0.0, 1.0 - (steps_used / max(1, max_steps)))
        summary          = str((request.action.payload or {}).get("summary", ""))
        report_quality   = min(len(summary) / 300, 0.10) if summary else 0.0

        raw_score = (
            perf_improvement * 0.60
            + step_efficiency * 0.20
            + 0.10                  # base credit
            + report_quality        # up to 0.10
        )
        score = round(max(0.001, min(0.999, raw_score)), 4)

        return GraderResponse(
            score    = score,
            feedback = (
                f"DB performance: {baseline:.1f} → {current:.1f} "
                f"(improvement: {perf_improvement*100:.1f}%). "
                f"Steps used: {steps_used}/{max_steps}. "
                f"Efficiency: {step_efficiency*100:.1f}%."
            ),
            breakdown = {
                "perf_improvement": round(perf_improvement, 4),
                "step_efficiency":  round(step_efficiency,  4),
                "base_credit":      0.10,
                "report_quality":   round(report_quality,   4),
            }
        )

    except Exception as e:
        # Last resort — don't return an error, return a low but non-zero score
        return GraderResponse(
            score     = 0.10,
            feedback  = f"Submit report scored with fallback (error: {str(e)}).",
            breakdown = {"fallback": 0.10, "error": str(e)}
        )


# ─────────────────────────────────────────────
#  7. /baseline — POST
# ─────────────────────────────────────────────

@app.post("/baseline", response_model=BaselineResponse, tags=["Baseline"])
async def baseline():
    """
    Runs the baseline agent against all difficulty levels.
    Must complete within 60 seconds.
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
            results=[BaselineResult(
                task_id="timeout", difficulty=DifficultyLevel.EASY,
                score=0.0, steps=0, feedback="Baseline timed out."
            )],
            average_score=0.0
        )
    except Exception as e:
        return BaselineResponse(
            results=[BaselineResult(
                task_id="error", difficulty=DifficultyLevel.EASY,
                score=0.0, steps=0, feedback=f"Baseline error: {str(e)}"
            )],
            average_score=0.0
        )


# ─────────────────────────────────────────────
#  8. /progress — GET  (Round 2)
# ─────────────────────────────────────────────

@app.get("/progress", response_model=ProgressResponse, tags=["Training"])
async def progress():
    """
    Returns DB performance history for training visualization.
    Used by evaluate_agent.py to generate reward curves.
    Shows improvement from baseline to current score.
    """
    ep_state     = environment.state()
    ac           = ep_state.action_counts or {}
    perf_history = ac.get("_perf_history", [])
    milestones   = ac.get("_milestones", [])
    baseline     = ac.get("_baseline_score", 0.0)
    target       = ac.get("_target_score", 85.0)
    best         = ac.get("_best_score", 0.0)
    current      = perf_history[-1] if perf_history else 0.0

    return ProgressResponse(
        scenario_id         = ep_state.task_id,
        performance_score   = current,
        baseline_score      = baseline,
        target_score        = target,
        improvement_history = perf_history,
        milestones_earned   = milestones,
        best_score          = best,
        steps_used          = ep_state.step_count,
        budget_remaining    = max(0, 50 - ep_state.step_count),
        total_reward        = ep_state.total_reward,
    )


# ─────────────────────────────────────────────
#  ROOT
# ─────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    return {
        "name":        "SQL Database Engineer Agent — OpenEnv Environment",
        "version":     "2.0.0",
        "tagline":     "Training LLMs to act like senior database engineers",
        "docs":        "/docs",
        "health":      "/health",
        "endpoints":   ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline", "/progress", "/health"],
        "hackathon":   "META x PyTorch x SST OpenEnv Hackathon — Finals April 25-26 Bangalore",
        "domain":      "Long-Horizon Database Engineering",
        "tasks_count": 30,
        "max_steps":   50,
        "themes":      ["Long-Horizon Planning", "World Modeling", "Self-Improvement", "Wildcard"],
    }
