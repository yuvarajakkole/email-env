"""
Environment Server — FastAPI REST API
--------------------------------------
Exposes the Email Triage Environment over HTTP for remote agents.
Compatible with Hugging Face Spaces (port 8000).

Endpoints:
  POST /reset              → Start new episode
  POST /step               → Submit action, get reward
  GET  /state              → Get current state
  GET  /tasks              → List available tasks
  GET  /health             → Health check
  GET  /docs               → Swagger UI (FastAPI auto-generated)
"""

from __future__ import annotations

import json
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors  import CORSMiddleware
from pydantic import BaseModel

from config import settings   # loads .env, never exposes raw key
from env.core import EmailTriageEnv
from env.models import AgentAction, EmailObservation, StepReward, EpisodeResult, EpisodeState
from tasks.manager import TaskManager


# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────

app = FastAPI(
    title       = "Email Triage RL Environment",
    description = "OpenEnv-compliant environment for email classification and triage.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Session registry (task_id → env instance)
_sessions: Dict[str, EmailTriageEnv] = {}


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "medium"
    seed:    int = 42

class ResetResponse(BaseModel):
    session_id:  str
    observation: Optional[Dict[str, Any]]
    task_config: Dict[str, Any]

class StepRequest(BaseModel):
    session_id: str
    action:     Dict[str, Any]

class StepResponse(BaseModel):
    observation: Optional[Dict[str, Any]]
    reward:      Dict[str, Any]
    done:        bool
    info:        Dict[str, Any]

class EpisodeResultResponse(BaseModel):
    session_id: str
    result:     Dict[str, Any]


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0", "config": settings.safe_summary()}


@app.get("/tasks")
def list_tasks():
    from tasks.manager import TASK_CONFIGS
    return {
        task_id: {
            "name":        cfg.name,
            "description": cfg.description,
            "difficulty":  cfg.difficulty,
            "n_emails":    cfg.n_emails,
            "max_steps":   cfg.max_steps,
        }
        for task_id, cfg in TASK_CONFIGS.items()
    }


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    from tasks.manager import TASK_CONFIGS
    if req.task_id not in TASK_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown task: {req.task_id}")

    env = EmailTriageEnv(task_id=req.task_id, seed=req.seed)
    obs = env.reset()

    session_id = f"{req.task_id}_{req.seed}"
    _sessions[session_id] = env
    cfg = TASK_CONFIGS[req.task_id]

    return ResetResponse(
        session_id  = session_id,
        observation = obs.dict() if obs else None,
        task_config = {
            "task_id":         cfg.task_id,
            "name":            cfg.name,
            "difficulty":      cfg.difficulty,
            "n_emails":        cfg.n_emails,
            "label_space":     cfg.label_space,
            "require_action":  cfg.require_action,
            "require_response":cfg.require_response,
        },
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session {req.session_id} not found. Call /reset first.")

    try:
        action = AgentAction(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(
        observation = obs.dict() if obs else None,
        reward      = reward.dict(),
        done        = done,
        info        = info,
    )


@app.get("/state/{session_id}")
def get_state(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return env.state()


@app.get("/result/{session_id}", response_model=EpisodeResultResponse)
def get_result(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    result = env.episode_result()
    return EpisodeResultResponse(session_id=session_id, result=result.dict())


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

# if __name__ == "__main__":
#     uvicorn.run("server:app", host="0.0.0.0", port=settings.port, reload=False)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)