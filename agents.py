from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

from utils import (
    llm,
    get_calendly_tools,
    get_gmail_tools,
    SCHEDULER_SYSTEM_PROMPT,
    GMAIL_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
)
from reasoner import reasoner, ReasoningPlan

# ── LOAD TOOLS VIA FACTORY FUNCTIONS ─────────────────────────────────────────
try:
    _calendly_tools = get_calendly_tools()
except RuntimeError as e:
    print(f"[agents] WARN: {e} — scheduler_agent will have no tools")
    _calendly_tools = []

try:
    _gmail_tools = get_gmail_tools()
except RuntimeError as e:
    print(f"[agents] WARN: {e} — gmail_agent will have no tools")
    _gmail_tools = []

# ── AGENTS ────────────────────────────────────────────────────────────────────
scheduler_agent = create_agent(
    model=llm,
    tools=_calendly_tools,
    system_prompt=SCHEDULER_SYSTEM_PROMPT,
    name="scheduler_agent",
)

gmail_agent = create_agent(
    model=llm,
    tools=_gmail_tools,
    system_prompt=GMAIL_SYSTEM_PROMPT,
    name="gmail_agent",
)

supervisor_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=SUPERVISOR_SYSTEM_PROMPT,
    name="supervisor_agent",
)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _plan_to_context(plan: ReasoningPlan) -> str:
    """
    Converts a ReasoningPlan into a context block prepended to the user
    message so the agent knows the pre-computed reasoning.
    """
    lines = [
        "=== REASONING PLAN ===",
        f"Intent     : {plan.intent}",
        f"Route      : {plan.route}",
        "Steps      :",
    ]
    for i, step in enumerate(plan.steps, 1):
        lines.append(f"  {i}. {step}")
    if plan.constraints:
        lines.append("Constraints:")
        for c in plan.constraints:
            lines.append(f"  • {c}")
    if plan.missing:
        lines.append("Missing info (ask user):")
        for m in plan.missing:
            lines.append(f"  • {m}")
    lines.append("======================")
    return "\n".join(lines)


# ── MULTI-AGENT ORCHESTRATOR ──────────────────────────────────────────────────
class MultiAgent:
    """
    Flow for every request:
      1. StepwiseReasoner analyses the message → ReasoningPlan
      2. Plan determines which agent(s) to call
      3. Plan context is prepended to the agent's input
      4. Agent runs with tools and returns answer
    """

    def __init__(self):
        self.history: list[dict] = []

    def _build_messages(self, user_input: str) -> dict:
        """Build messages payload from history + current input."""
        messages = []
        for msg in self.history[-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=user_input))
        return {"messages": messages}

    async def run(self, user_input: str) -> str:
        """Blocking call — reasons first, then runs the agent."""
        self.history.append({"role": "user", "content": user_input})

        try:
            # ── Step 1: Reason ────────────────────────────────────────────
            plan = await reasoner.plan(user_input)
            print(f"[Reasoner] intent={plan.intent!r}  route={plan.route}")

            # ── Step 2: Build enriched input ──────────────────────────────
            enriched = f"{_plan_to_context(plan)}\n\nUser request: {user_input}"

            # If missing required info, return early and ask the user
            if plan.missing:
                missing_str = "\n".join(f"- {m}" for m in plan.missing)
                answer = (
                    "Before I can help, I need a bit more information:\n\n"
                    + missing_str
                )
                self.history.append({"role": "assistant", "content": answer})
                return answer

            payload = self._build_messages(enriched)

            # ── Step 3: Route to agent ────────────────────────────────────
            if plan.route == "both":
                print("[MultiAgent] → scheduler_agent + gmail_agent (cross-app)")
                sched_result = await scheduler_agent.ainvoke(payload)
                sched_answer = sched_result["messages"][-1].content

                combined = (
                    f"{enriched}\n\n"
                    f"[Scheduling result]:\n{sched_answer}"
                )
                gmail_result = await gmail_agent.ainvoke(
                    self._build_messages(combined)
                )
                answer = gmail_result["messages"][-1].content

            elif plan.route == "scheduler":
                print("[MultiAgent] → scheduler_agent")
                result = await scheduler_agent.ainvoke(payload)
                answer = result["messages"][-1].content

            elif plan.route == "gmail":
                print("[MultiAgent] → gmail_agent")
                result = await gmail_agent.ainvoke(payload)
                answer = result["messages"][-1].content

            else:
                print("[MultiAgent] → supervisor_agent")
                result = await supervisor_agent.ainvoke(payload)
                answer = result["messages"][-1].content

        except Exception as e:
            print(f"[MultiAgent] ERROR during run: {e}")
            self.history.pop()
            raise RuntimeError(f"Agent execution failed: {e}") from e

        self.history.append({"role": "assistant", "content": answer})
        return answer

    async def stream(self, user_input: str):
        """
        Async generator — yields SSE chunks.

        SSE event types:
          data: [REASONING_START]          → reasoning phase begins
          data: [INTENT: <text>]           → what the user wants
          data: [ROUTE: <route>]           → which agent will handle it
          data: [STEP: <n>|<text>]         → reasoning step n
          data: [CONSTRAINT: <text>]       → applicable constraint
          data: [MISSING: <text>]          → missing required info
          data: [REASONING_END]            → reasoning phase complete
          data: [TOOL_CALL: <name>]        → agent calling a tool
          data: [TOOL_DONE: <name>]        → tool finished
          data: <token>                    → LLM response token
          data: [ERROR: <msg>]             → error
          data: [DONE]                     → stream complete
        """
        self.history.append({"role": "user", "content": user_input})

        try:
            # ── Step 1: Reason and stream the plan ────────────────────────
            yield "data: [REASONING_START]\n\n"

            plan = await reasoner.plan(user_input)
            print(f"[Reasoner] intent={plan.intent!r}  route={plan.route}")

            yield f"data: [INTENT: {plan.intent}]\n\n"
            yield f"data: [ROUTE: {plan.route}]\n\n"

            for i, step in enumerate(plan.steps, 1):
                yield f"data: [STEP: {i}|{step}]\n\n"

            for constraint in plan.constraints:
                yield f"data: [CONSTRAINT: {constraint}]\n\n"

            for missing in plan.missing:
                yield f"data: [MISSING: {missing}]\n\n"

            yield "data: [REASONING_END]\n\n"

            # ── Early exit if required info is missing ────────────────────
            if plan.missing:
                missing_str = "\n".join(f"- {m}" for m in plan.missing)
                answer = (
                    "Before I can help, I need a bit more information:\n\n"
                    + missing_str
                )
                yield f"data: {answer}\n\n"
                yield "data: [DONE]\n\n"
                self.history.append({"role": "assistant", "content": answer})
                return

            # ── Step 2: Build enriched input ──────────────────────────────
            enriched = f"{_plan_to_context(plan)}\n\nUser request: {user_input}"

            # ── Step 3: Pick agent ────────────────────────────────────────
            if plan.route == "both":
                print("[MultiAgent/stream] → scheduler_agent + gmail_agent")
                sched_result = await scheduler_agent.ainvoke(
                    self._build_messages(enriched)
                )
                sched_answer = sched_result["messages"][-1].content
                combined = (
                    f"{enriched}\n\n"
                    f"[Scheduling result]:\n{sched_answer}"
                )
                agent   = gmail_agent
                payload = self._build_messages(combined)

            elif plan.route == "scheduler":
                print("[MultiAgent/stream] → scheduler_agent")
                agent   = scheduler_agent
                payload = self._build_messages(enriched)

            elif plan.route == "gmail":
                print("[MultiAgent/stream] → gmail_agent")
                agent   = gmail_agent
                payload = self._build_messages(enriched)

            else:
                print("[MultiAgent/stream] → supervisor_agent")
                agent   = supervisor_agent
                payload = self._build_messages(enriched)

            # ── Step 4: Stream agent events ───────────────────────────────
            full_answer = ""
            async for event in agent.astream_events(payload, version="v2"):
                kind = event.get("event", "")

                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        full_answer += chunk.content
                        yield f"data: {chunk.content}\n\n"

                elif kind == "on_tool_start":
                    yield f"data: [TOOL_CALL: {event.get('name', 'tool')}]\n\n"

                elif kind == "on_tool_end":
                    yield f"data: [TOOL_DONE: {event.get('name', 'tool')}]\n\n"

        except Exception as e:
            print(f"[MultiAgent] ERROR during stream: {e}")
            self.history.pop()
            yield f"data: [ERROR: {e}]\n\n"
            yield "data: [DONE]\n\n"
            return

        yield "data: [DONE]\n\n"
        self.history.append({"role": "assistant", "content": full_answer})


# ── SINGLETON ─────────────────────────────────────────────────────────────────
multi_agent = MultiAgent()
