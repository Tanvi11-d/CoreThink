"""
Streamlit frontend for the Calendly + Gmail Multi-Agent API.
Connects to FastAPI /chat/stream (SSE) and renders:
  - Reasoning plan (intent, route, steps, constraints) in a collapsible expander
  - Tool calls as inline status badges
  - LLM tokens progressively with proper markdown formatting

Run:
    streamlit run app.py
"""

import requests
import streamlit as st

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_BASE    = "http://localhost:8000"
STREAM_URL  = f"{API_BASE}/chat/stream"
HISTORY_URL = f"{API_BASE}/history"

# ── PAGE SETUP ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroFlow AI Agent",
    layout="centered",
)

st.title("🤖 NeuroFlow AI Agent")

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── CHAT HISTORY ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Stored messages may have a reasoning block — render as-is (markdown)
        st.markdown(msg["content"])

# ── CHAT INPUT ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything about scheduling or email…"):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Stream assistant response ─────────────────────────────────────────────
    with st.chat_message("assistant"):

        # Reasoning plan collected here, rendered in expander after REASONING_END
        reasoning: dict = {
            "intent":      "",
            "route":       "",
            "steps":       [],
            "constraints": [],
            "missing":     [],
            "active":      False,   # True while between REASONING_START / REASONING_END
        }

        reasoning_expander = None   # created once REASONING_START arrives
        tool_status        = st.empty()
        response_area      = st.empty()

        full_response  = ""
        error_occurred = False

        try:
            with requests.post(
                STREAM_URL,
                json={"message": prompt},
                stream=True,
                timeout=240,
            ) as resp:
                resp.raise_for_status()

                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line or not raw_line.startswith("data:"):
                        continue

                    data = raw_line[len("data:"):].strip()

                    # ── Stream complete ───────────────────────────────────
                    if data == "[DONE]":
                        tool_status.empty()
                        break

                    # ── Reasoning phase start ─────────────────────────────
                    elif data == "[REASONING_START]":
                        reasoning["active"] = True
                        # Placeholder expander — will be filled as events arrive
                        reasoning_expander = st.expander(
                            "🧠 Reasoning plan", expanded=True
                        )

                    # ── Reasoning phase end ───────────────────────────────
                    elif data == "[REASONING_END]":
                        reasoning["active"] = False
                        # Re-render the expander with all collected data
                        if reasoning_expander is not None:
                            with reasoning_expander:
                                if reasoning["intent"]:
                                    st.markdown(
                                        f"**Intent:** {reasoning['intent']}"
                                    )
                                if reasoning["route"]:
                                    route_emoji = {
                                        "scheduler": "📅",
                                        "gmail":     "📧",
                                        "both":      "📅 → 📧",
                                        "general":   "🧠",
                                    }.get(reasoning["route"], "🔀")
                                    st.markdown(
                                        f"**Route:** {route_emoji} `{reasoning['route']}`"
                                    )
                                if reasoning["steps"]:
                                    st.markdown("**Steps:**")
                                    for i, step in enumerate(reasoning["steps"], 1):
                                        st.markdown(f"&nbsp;&nbsp;{i}. {step}")
                                if reasoning["constraints"]:
                                    st.markdown("**Constraints:**")
                                    for c in reasoning["constraints"]:
                                        st.markdown(f"&nbsp;&nbsp;• {c}")
                                if reasoning["missing"]:
                                    st.warning(
                                        "**Missing info:**\n"
                                        + "\n".join(
                                            f"- {m}" for m in reasoning["missing"]
                                        )
                                    )

                    # ── Reasoning fields ──────────────────────────────────
                    elif data.startswith("[INTENT: "):
                        reasoning["intent"] = data[len("[INTENT: "):-1].strip()

                    elif data.startswith("[ROUTE: "):
                        reasoning["route"] = data[len("[ROUTE: "):-1].strip()

                    elif data.startswith("[STEP: "):
                        # Format: [STEP: <n>|<text>]
                        inner = data[len("[STEP: "):-1]
                        parts = inner.split("|", 1)
                        step_text = parts[1].strip() if len(parts) == 2 else inner
                        reasoning["steps"].append(step_text)

                    elif data.startswith("[CONSTRAINT: "):
                        reasoning["constraints"].append(
                            data[len("[CONSTRAINT: "):-1].strip()
                        )

                    elif data.startswith("[MISSING: "):
                        reasoning["missing"].append(
                            data[len("[MISSING: "):-1].strip()
                        )

                    # ── Tool events ───────────────────────────────────────
                    elif data.startswith("[TOOL_CALL: "):
                        tool_name = data[len("[TOOL_CALL: "):-1].strip()
                        tool_status.info(f"🔧 Calling `{tool_name}`…")

                    elif data.startswith("[TOOL_DONE: "):
                        tool_name = data[len("[TOOL_DONE: "):-1].strip()
                        tool_status.success(f"✅ `{tool_name}` done")

                    # ── Error ─────────────────────────────────────────────
                    elif data.startswith("[ERROR: "):
                        err_msg = data[len("[ERROR: "):-1].strip()
                        tool_status.error(f"⚠️ {err_msg}")
                        error_occurred = True
                        break

                    # ── Token chunk ───────────────────────────────────────
                    else:
                        full_response += data
                        response_area.markdown(full_response + " ▌")

        except requests.exceptions.ConnectionError:
            full_response = (
                "⚠️ **Cannot connect to the API.**  \n"
                "Make sure FastAPI is running:  \n"
                "```\nuvicorn main:app --reload\n```"
            )
            error_occurred = True
        except requests.exceptions.Timeout:
            full_response = "⚠️ **Request timed out.** The agent is taking too long."
            error_occurred = True
        except Exception as e:
            full_response = f"⚠️ **Unexpected error:** {e}"
            error_occurred = True

        # Final clean render — no cursor
        if full_response:
            response_area.markdown(full_response)
        elif not error_occurred:
            response_area.markdown("*(no response)*")

    # Save to session history
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
