from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agents import multi_agent

app = FastAPI(
    description="Multi-agent system: Calendly scheduler + Gmail agent, powered by Composio + OpenRouter",
    version="1.0.0"
)


# ── SCHEMAS ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    message: str
    answer: str
    history_length: int


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Calendly + Gmail Multi-Agent API is running!"}


@app.get("/agents")
async def agents_info():
    """List available agents and their capabilities."""
    return {
        "agents": [
            {
                "name": "scheduler_agent",
                "toolkit": "CALENDLY",
                "handles": [
                    "Create scheduling / single-use links",
                    "List and fetch events",
                    "Cancel events",
                    "List available time slots",
                ]
            },
            {
                "name": "gmail_agent",
                "toolkit": "GMAIL",
                "handles": [
                    "Send emails",
                    "Read and search inbox",
                    "Reply and forward emails",
                    "Create and manage drafts",
                    "Delete emails"
                ]
            },
            {
                "name": "supervisor_agent",
                "toolkit": "none",
                "handles": [
                    "General questions",
                    "Cross-app routing (Calendly + Gmail together)"
                ]
            }
        ]
    }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream a response using Server-Sent Events (SSE).

    SSE event types emitted:
    - data: <token>              → LLM token chunk (append to UI)
    - data: [TOOL_CALL: <name>]  → agent called a tool (show spinner)
    - data: [TOOL_DONE: <name>]  → tool finished
    - data: [ERROR: <msg>]       → something went wrong
    - data: [DONE]               → stream complete
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    return StreamingResponse(
        multi_agent.stream(request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the multi-agent system (blocking — waits for full response).

    Routing:
    - Calendly / scheduling keywords  → scheduler_agent
    - Gmail / email keywords          → gmail_agent
    - Both keywords in one message    → scheduler_agent then gmail_agent (cross-app)
    - General questions               → supervisor_agent

    Example requests:
    - "Create a scheduling link for a 30-minute meeting"
    - "List my upcoming Calendly events"
    - "Send an email to john@example.com about our meeting"
    - "Schedule a meeting and email the invite link to the team"
    - "Search my inbox for emails from alice@example.com"
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        answer = await multi_agent.run(request.message)
        return ChatResponse(
            message=request.message,
            answer=answer,
            history_length=len(multi_agent.history)
        )
    except Exception as e:
        print(f"[Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/history")
async def get_history():
    """Get the current conversation history."""
    return {
        "history": multi_agent.history,
        "total_turns": len(multi_agent.history)
    }


@app.delete("/history")
async def clear_history():
    """Clear the conversation history."""
    multi_agent.history.clear()
    return {"message": "Conversation history cleared."}
