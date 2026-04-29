"""
reasoner.py — Stepwise Reasoning Layer

Before any agent runs, the StepwiseReasoner analyses the user message and
produces a structured ReasoningPlan:

  1. Intent      — what the user wants in plain English
  2. Route       — which agent(s) to call: "scheduler", "gmail", "both", "general"
  3. Steps       — ordered list of reasoning steps the agent should follow
  4. Constraints — any rules that apply (no weekends, 10am-4pm, etc.)
  5. Missing     — parameters the user forgot to provide (if any)

The plan is used by MultiAgent for routing and is streamed to the frontend
so the user can see the reasoning before the answer arrives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from utils import llm

# ── TYPES ─────────────────────────────────────────────────────────────────────
Route = Literal["scheduler", "gmail", "both", "general"]


@dataclass
class ReasoningPlan:
    intent:      str               # one-sentence summary of what the user wants
    route:       Route             # which agent(s) to invoke
    steps:       list[str]         # ordered reasoning steps
    constraints: list[str]         # applicable rules / guardrails
    missing:     list[str]         # required info not provided by the user
    raw:         str = field(repr=False, default="")  # raw LLM output


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
_REASONER_SYSTEM = """You are a reasoning layer for a multi-agent assistant.
Your job is to analyse the user's message and produce a structured JSON plan
that tells the downstream agents exactly what to do and in what order.

Available agents:
  - scheduler  : Calendly — create links, list/cancel events, manage invitees
  - gmail      : Gmail    — send, read, draft, reply, forward emails
  - both       : run scheduler first, then gmail (cross-app task)
  - general    : answer directly without any tool

Constraints that always apply:
  - No scheduling on weekends
  - No scheduling outside 10 am – 4 pm
  - Always call CALENDLY_GET_CURRENT_USER before any Calendly operation
  - Never ask the user for URIs or UUIDs — fetch them with tools

Respond ONLY with valid JSON in this exact shape (no markdown, no extra text):
{
  "intent":      "<one sentence: what the user wants>",
  "route":       "<scheduler | gmail | both | general>",
  "steps":       ["<step 1>", "<step 2>", ...],
  "constraints": ["<constraint that applies>", ...],
  "missing":     ["<info the user forgot to provide>", ...]
}

Rules for filling each field:
- intent      : concise, plain English, ≤ 20 words
- route       : pick exactly one of the four values
- steps       : 2–6 concrete steps the agent should follow in order
- constraints : only list constraints that actually apply to this request
- missing     : only list truly required info that is absent (leave [] if nothing is missing)
"""


# ── REASONER ──────────────────────────────────────────────────────────────────
class StepwiseReasoner:
    """
    Calls the LLM once to produce a ReasoningPlan for the user's message.
    Fast and cheap — uses the same LLM as the agents but with a short prompt.
    """

    async def plan(self, user_message: str) -> ReasoningPlan:
        """
        Analyse user_message and return a ReasoningPlan.
        Falls back to a safe default plan if the LLM returns invalid JSON.
        """
        try:
            response = await llm.ainvoke([
                SystemMessage(content=_REASONER_SYSTEM),
                HumanMessage(content=user_message),
            ])
            raw = response.content.strip()

            # Strip markdown code fences if the model added them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            data = json.loads(raw)

            return ReasoningPlan(
                intent      = data.get("intent", ""),
                route       = data.get("route", "general"),
                steps       = data.get("steps", []),
                constraints = data.get("constraints", []),
                missing     = data.get("missing", []),
                raw         = raw,
            )

        except Exception as e:
            print(f"[Reasoner] WARNING: could not parse plan — {e}. Using fallback.")
            return _fallback_plan(user_message)


def _fallback_plan(user_message: str) -> ReasoningPlan:
    """
    Keyword-based fallback used when the LLM reasoner fails.
    Mirrors the old keyword routing so the system still works.
    """
    text = user_message.lower()

    scheduling_kw = {
        "schedule", "meeting", "event", "calendly", "booking", "book",
        "cancel", "invite", "invitee", "availability", "slot", "link",
        "reschedule", "appointment", "scheduling",
    }
    gmail_kw = {
        "email", "gmail", "mail", "send", "inbox", "draft", "reply",
        "forward", "subject", "unread", "read", "thread", "compose",
        "cc", "bcc", "recipient",
    }

    is_sched = any(kw in text for kw in scheduling_kw)
    is_gmail = any(kw in text for kw in gmail_kw)

    if is_sched and is_gmail:
        route = "both"
    elif is_sched:
        route = "scheduler"
    elif is_gmail:
        route = "gmail"
    else:
        route = "general"

    return ReasoningPlan(
        intent      = "Process user request (fallback plan)",
        route       = route,
        steps       = ["Analyse request", "Call appropriate agent", "Return result"],
        constraints = [],
        missing     = [],
        raw         = "",
    )


# ── SINGLETON ─────────────────────────────────────────────────────────────────
reasoner = StepwiseReasoner()
