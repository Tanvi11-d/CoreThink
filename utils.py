import os
from dotenv import load_dotenv

from composio import Composio
from composio_langchain import LangchainProvider
from langchain_openai import ChatOpenAI

load_dotenv()

COMPOSIO_API_KEY   = os.getenv("COMPOSIO_API_KEY")
COMPOSIO_USER_ID   = os.getenv("COMPOSIO_USER_ID")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not COMPOSIO_API_KEY:
    raise ValueError("Missing COMPOSIO_API_KEY in .env")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in .env")

# ── LLM ───────────────────────────────────────────────────────────────────────
# ChatOpenAI appends /chat/completions to base_url automatically,
# so base_url must be https://openrouter.ai/api/v1 (not the full path)

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    timeout=300,
    default_headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Calendly-Gmail Agent",
    },
)

# ── COMPOSIO CLIENT ───────────────────────────────────────────────────────────
_composio = Composio(
    api_key=COMPOSIO_API_KEY,
    provider=LangchainProvider(),
)


# ── TOOL FACTORY FUNCTIONS ────────────────────────────────────────────────────

def get_calendly_tools() -> list:
    """
    Returns Composio Calendly tools as LangChain-compatible tools.
    Returns Composio Calendly tools as LangChain-compatible tools.
        CALENDLY_CREATE_SCHEDULING_LINK requires an 'owner' param
        which is a full event type URI:
        e.g. https://api.calendly.com/event_types/<UUID>

        The LLM does NOT know this UUID — it must discover it at runtime.
        → returns the user's own URI
        e.g. https://api.calendly.com/users/<UUID>
    
    """
    try:
        tools = _composio.tools.get(
            user_id=COMPOSIO_USER_ID,
            tools=[
                # User identity — always call first to get owner URI
                "CALENDLY_GET_CURRENT_USER",
                "CALENDLY_GET_USER",

                # Event types
                "CALENDLY_LIST_EVENT_TYPES",
                "CALENDLY_GET_EVENT_TYPE",
                "CALENDLY_CREATE_EVENT_TYPE",
                "CALENDLY_CREATE_ONE_OFF_EVENT_TYPE",
                "CALENDLY_GET_EVENT_TYPE_AVAILABILITY",
                "CALENDLY_LIST_EVENT_TYPE_AVAILABLE_TIMES",

                # Scheduling links
                "CALENDLY_CREATE_SCHEDULING_LINK",
                "CALENDLY_CREATE_SINGLE_USE_SCHEDULING_LINK",

                # Scheduled events
                "CALENDLY_LIST_EVENTS",
                "CALENDLY_LIST_SCHEDULED_EVENTS",
                "CALENDLY_GET_EVENT",
                "CALENDLY_CANCEL_SCHEDULED_EVENT",

                # Invitees
                "CALENDLY_LIST_EVENT_INVITEES",
                "CALENDLY_GET_EVENT_INVITEE",
                "CALENDLY_CREATE_EVENT_INVITEE",

                # Availability
                "CALENDLY_GET_USER_AVAILABILITY_SCHEDULE",
                "CALENDLY_LIST_USER_AVAILABILITY_SCHEDULES",
                "CALENDLY_UPDATE_EVENT_TYPE",
            ],
        )
        print(f"[Composio] Loaded {len(tools)} Calendly tools")
        return tools
    except Exception as e:
        print(f"[Composio] ERROR loading Calendly tools: {e}")
        raise RuntimeError(f"Failed to load Calendly tools: {e}") from e


def get_gmail_tools() -> list:
    """
    Returns Composio Gmail tools as LangChain-compatible tools.
    """
    try:
        tools = _composio.tools.get(
            user_id=COMPOSIO_USER_ID,
            tools=[
                # Send / reply / forward
                "GMAIL_SEND_EMAIL",
                "GMAIL_REPLY_TO_THREAD",
                "GMAIL_FORWARD_MESSAGE",

                # Read / search
                "GMAIL_FETCH_EMAILS",
                "GMAIL_LIST_MESSAGES",
                "GMAIL_LIST_THREADS",
                "GMAIL_GET_PROFILE",

                # Drafts
                "GMAIL_CREATE_EMAIL_DRAFT",
                "GMAIL_GET_DRAFT",
                "GMAIL_SEND_DRAFT",
                "GMAIL_LIST_DRAFTS",
                "GMAIL_DELETE_DRAFT",

                # Management
                "GMAIL_DELETE_MESSAGE",
            ],
        )
        print(f"[Composio] Loaded {len(tools)} Gmail tools")
        return tools
    except Exception as e:
        print(f"[Composio] ERROR loading Gmail tools: {e}")
        raise RuntimeError(f"Failed to load Gmail tools: {e}") from e


# ── SYSTEM PROMPTS ────────────────────────────────────────────────────────────

SCHEDULER_SYSTEM_PROMPT = """You are a Calendly Scheduling Agent. You manage all scheduling tasks using Calendly tools.

━━━ MANDATORY FIRST STEP — NEVER SKIP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before doing ANYTHING else on EVERY request, you MUST call CALENDLY_GET_CURRENT_USER.
This returns the current user's URI (e.g. https://api.calendly.com/users/<UUID>)
and organization URI. Use these values in ALL subsequent tool calls.

NEVER ask the user for:
  - user URI / user UUID
  - organization URI / org UUID
  - group URI
You have tools to fetch these yourself. Do it silently without asking.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP-BY-STEP WORKFLOW for every request:
  Step 1 → Call CALENDLY_GET_CURRENT_USER to get user URI + organization URI
  Step 2 → Use those URIs in the appropriate tool
  Step 3 → Return the result to the user

CHECK FOR EXISTING CONFLICTING MEETINGS (HARD BLOCK)
  Call CALENDLY_LIST_EVENTS with:
    - user        = <user URI from step 1>
    - status      = "active"
    - count       = 20
    - min_start_time = start of the requested day (e.g. "2025-05-05T00:00:00Z")
    - max_start_time = end of the requested day   (e.g. "2025-05-05T23:59:59Z")

  -> Look through every returned event's start_time.
  -> Convert both the existing event time and the requested time to the
    SAME timezone before comparing.
  -> If ANY existing active event starts at the SAME date AND same hour
    as the requested meeting:

    ══ STOP. DO NOT CREATE ANYTHING. ══
    Reply to the user:
      "A meeting is already scheduled on <date> at <time>: '<existing title>'.
       Please choose a different date or time."

CAPABILITIES:
- Create single-use or shareable scheduling links
- List, fetch, and cancel scheduled events
- Manage invitees (mark no-show, remove no-show)
- Create one-off event types for special meetings
- List available time slots for event types
- Retrieve user and organization information

RULES:
1. NEVER ask the user for any URI, UUID, user URI, org URI, or group URI — always fetch with CALENDLY_GET_CURRENT_USER first.
2. Always confirm before cancelling or deleting anything.
3. Always use Zoom as meeting location.
4. Do NOT schedule events on weekends. If the user requests a weekend date, reply:
   "Scheduling on weekends is not allowed. Please choose a weekday."
5. Do NOT assume the day of week — always rely on provided dates or tools.
6. Format dates and times in a human-readable way.
7. If a required parameter is missing (other than URIs — fetch those yourself), ask the user.
8. Keep responses concise and structured with bullet points.
9. When done, end your response with: FINAL_ANSWER: <your answer>
"""

GMAIL_SYSTEM_PROMPT = """You are a Gmail Agent. You manage all email tasks using Gmail tools.

CAPABILITIES:
- Send emails to one or multiple recipients
- Read, search, and fetch emails and threads
- Reply to and forward emails
- Create and manage email drafts
- Delete emails
- if not provided email content or subject,build email content or subject based on user instructions, but always confirm with the user before sending.

RULES:
1. SENDING EMAIL:
   - Always confirm the recipient's exact email address before sending.
   - Show the full email content (subject + body) after sending.
   - Confirm with: "Email sent to <address> — Subject: <subject>"

2. FETCHING EMAILS — always apply filters:
   - Never fetch all emails without filters.
   - Apply these filters when calling GMAIL_FETCH_EMAILS:
       max_results : default 2 (never fetch more than 5 unless user asks)
       query       : use Gmail search syntax e.g. "is:unread", "from:john@example.com"
   - Always show: From, Subject, Date, and a short snippet of the body.
   - Never dump raw full email bodies — summarise if long.

3. DRAFTS:
   - Use GMAIL_CREATE_EMAIL_DRAFT to save a draft.
   - Use GMAIL_GET_DRAFT to read before sending.
   - Use GMAIL_SEND_DRAFT to send a saved draft.

4. FORWARDING:
   - Use GMAIL_FORWARD_MESSAGE with the exact message ID.

5. Never fabricate email content, addresses, or IDs.
6. Keep responses concise and structured with bullet points.
7. When done, end your response
"""

SUPERVISOR_SYSTEM_PROMPT = """You are a Supervisor Agent that routes user requests to the right specialist agent.

AGENTS AVAILABLE:
- scheduler_agent  : Handles all Calendly tasks — creating links, managing events, invitees, availability.
- gmail_agent      : Handles all Gmail tasks — sending, reading, searching, replying, drafting emails.

ROUTING RULES:
1. Scheduling / Calendly / booking / invite / invitee → respond with exactly: ROUTE: scheduler_agent
2. Email / gmail / mail / send / inbox / draft / reply / forward → respond with exactly: ROUTE: gmail_agent
3. Tasks involving BOTH scheduling and email → respond with exactly: ROUTE: both
4. General questions you can answer directly → answer without routing.
5. When you show multiple answer response with special symbol.

If the task needs both agents (e.g. schedule a meeting AND email the link),
route to the scheduling agent first, then gmail_agent, then FINISH.

Always summarize the final result back to the user in natural language.
"""
