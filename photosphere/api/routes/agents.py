# ============================================================
# photosphere/api/routes/agent.py
# Autonomous Agent REST endpoints
#
# POST /v1/agent/start   — launch a 72-hour agent
# GET  /v1/agent/{id}    — get status + latest findings
# POST /v1/agent/{id}/stop — gracefully stop the agent
# GET  /v1/agent/list    — list all active agents
# ============================================================

import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from convective.agents.autonomous_agent import (
    AutonomousAgent, register_agent, get_agent, list_agents
)
from shared.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────

class AgentStartRequest(BaseModel):
    goal: str = Field(
        ...,
        description="The strategic objective for the 72-hour monitoring run.",
        example="Monitor regulatory updates in clinical NLP and identify compliance risks",
    )
    topics: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of topic queries to monitor via the KU API.",
        example=["regulatory NLP clinical", "FDA clinical trials AI", "HIPAA data compliance"],
    )
    cycle_interval_minutes: Optional[int] = Field(
        30,
        ge=5,
        le=120,
        description="Minutes between monitoring cycles (default: 30).",
    )


class AgentStartResponse(BaseModel):
    agent_id:   str
    status:     str
    goal:       str
    topics:     list[str]
    message:    str
    dashboards: dict


class AgentStopResponse(BaseModel):
    agent_id: str
    status:   str
    message:  str


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@router.post("/agent/start", response_model=AgentStartResponse)
async def start_agent(
    body: AgentStartRequest,
    background_tasks: BackgroundTasks,
):
    """
    Launch a 72-hour autonomous monitoring agent.

    The agent:
    - Calls the KU API every 30 min to discover new documents
    - Ingests and processes them through the SI fusion pipeline
    - Synthesises findings using chain-of-thought reasoning
    - Alerts when knowledge decay thresholds are breached
    - Generates comprehensive reports every 12 hours
    - Produces a final intelligence report at 72 hours

    Returns immediately with an agent_id. Poll /v1/agent/{agent_id} for status.
    """
    agent = AutonomousAgent()
    register_agent(agent)

    # Override cycle interval if requested
    if body.cycle_interval_minutes != 30:
        agent.CYCLE_INTERVAL_SECONDS = body.cycle_interval_minutes * 60

    # Run the agent in the background — it will persist state to Redis
    background_tasks.add_task(agent.run, goal=body.goal, topics=body.topics)

    logger.info("agent_started_via_api", extra={
        "agent_id": agent._agent_id,
        "goal":     body.goal,
        "topics":   body.topics,
    })

    return AgentStartResponse(
        agent_id=agent._agent_id,
        status="running",
        goal=body.goal,
        topics=body.topics,
        message=(
            f"Agent launched. Running every {body.cycle_interval_minutes} min "
            f"for 72 hours. {len(body.topics)} topic(s) being monitored."
        ),
        dashboards={
            "agent_status":  f"/v1/agent/{agent._agent_id}",
            "redis_insight": "http://localhost:8001",
            "grafana":       "http://localhost:3000",
            "jaeger":        "http://localhost:16686",
        },
    )


@router.get("/agent/{agent_id}")
async def get_agent_status(agent_id: str):
    """
    Get the current status of a running agent.

    Returns:
    - status: running | paused | completed | failed
    - cycle_count: how many cycles have run
    - total_docs: documents ingested so far
    - recent_finding: latest synthesis insight
    - decay_alerts: any freshness threshold breaches
    - latest_report: most recent comprehensive report (if available)
    """
    # Try in-memory first, then Redis
    agent = get_agent(agent_id)

    if agent:
        status = agent.get_status()
    else:
        # Agent may be from a previous process — reload from Redis
        agent = AutonomousAgent.load_from_redis(agent_id)
        if agent:
            status = agent.get_status()
        else:
            status = None

    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"Agent {agent_id} not found. "
                   f"It may have expired (agents are kept for 80 hours after start).",
        )

    return status


@router.post("/agent/{agent_id}/stop", response_model=AgentStopResponse)
async def stop_agent(agent_id: str):
    """
    Gracefully stop a running agent.
    The current cycle will complete, then the agent shuts down and saves its final state.
    """
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(
            status_code=404,
            detail=f"Agent {agent_id} not found in current process.",
        )

    agent.stop()

    logger.info("agent_stopped_via_api", extra={"agent_id": agent_id})

    return AgentStopResponse(
        agent_id=agent_id,
        status="stopping",
        message="Stop signal sent. Agent will finish its current cycle then shut down.",
    )


@router.get("/agent/list/all")
async def list_all_agents():
    """List all active agents and their current status."""
    return {
        "agents": list_agents(),
        "count":  len(list_agents()),
    }


@router.get("/agent/{agent_id}/report")
async def get_agent_report(agent_id: str):
    """
    Get all generated reports for an agent.
    Reports are generated every 12 hours and at completion.
    """
    agent = get_agent(agent_id) or AutonomousAgent.load_from_redis(agent_id)

    if not agent or not agent._state:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found.")

    return {
        "agent_id": agent_id,
        "goal":     agent._state.goal,
        "reports":  agent._state.reports,
        "count":    len(agent._state.reports),
    }