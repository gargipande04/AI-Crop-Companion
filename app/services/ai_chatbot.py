"""
ai_chatbot.py — CropGPT Expert Chat Engine

Streams responses from Claude Opus 4.6 via Server-Sent Events (SSE).
Requires the ANTHROPIC_API_KEY environment variable to be set.

Exports:
  - ChatMsg / ChatRequest : Pydantic models for the /chat request body
  - chat()                : async FastAPI endpoint that returns a StreamingResponse
"""

import json
import os

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

try:
    import anthropic as _anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("anthropic package not installed — /chat endpoint disabled. Install with: pip install anthropic")

_AGRI_SYSTEM = """You are an expert agricultural advisor for Crop Companion, a farm intelligence platform.
You help farmers with practical, actionable guidance on:
- Plant diseases (Black Knot, Chlorosis, Dog Vomit Slime Mold, Elderberry Rust, Golden Canker,
  Gymnosporangium Rusts, Peach Leaf Curl, Powdery Mildew, Sooty Mold, Tar Spot)
- Farm pest management (Africanized Honey Bees, Aphids, Armyworms, Brown Marmorated Stink Bugs,
  Cabbage Loopers, Citrus Canker, Colorado Potato Beetles, Corn Borers, Corn Earworms,
  Fall Armyworms, Fruit Flies, Spider Mites, Thrips, Tomato Hornworms, Western Corn Rootworms)
- Crop yield optimisation and sustainable farming practices
- Irrigation, soil health, and pesticide use

Always give clear, farmer-friendly advice. Be concise and practical. Where relevant, mention
both organic/low-input options and conventional chemical options. If unsure, say so honestly.
Use markdown formatting (## headers, - bullet points, **bold**). Do not use tables."""


class ChatMsg(BaseModel):
    """A single message in the conversation history (role: 'user' or 'assistant')."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    message: str
    history: list[ChatMsg] = []  # Previous turns for multi-turn context


async def chat(req: ChatRequest):
    """Stream an agricultural expert response via Server-Sent Events.

    Builds the full conversation history, opens a streaming connection to the
    Claude API, and yields JSON-encoded SSE frames as text arrives. The final
    frame is 'data: [DONE]'. Any API error is yielded as an error frame so
    the frontend can display it gracefully.
    """
    if not HAS_ANTHROPIC:
        raise HTTPException(status_code=503, detail="anthropic package not installed. Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY environment variable not set.")

    client = _anthropic.AsyncAnthropic(api_key=api_key)
    messages = [{"role": m.role, "content": m.content} for m in req.history]
    messages.append({"role": "user", "content": req.message})

    async def generate():
        try:
            async with client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=1024,
                system=_AGRI_SYSTEM,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
