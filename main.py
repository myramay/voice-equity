"""
VoiceEquity - Multimodal Civic Assistant Backend
FastAPI server that accepts images + voice questions and returns plain-language
answers via Gemma 4 (Google GenAI SDK), auto-detecting and responding in the
user's language.
"""

import os
import base64
import logging
from typing import Optional

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voiceequity")

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = FastAPI(title="VoiceEquity", version="1.0.0")

# Allow all origins for local dev / hackathon demo purposes.
# In production, restrict this to your actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Google GenAI SDK configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY environment variable not set. "
        "The /analyze endpoint will fail until it is provided."
    )

# Configure the SDK once at startup so every request reuses the same session.
genai.configure(api_key=GEMINI_API_KEY)

# Gemma 4 multimodal model — supports text + image input.
# gemma-4-31b-it is the flagship dense model; swap for gemma-4-e4b-it if
# you need faster / lower-cost responses at a small accuracy trade-off.
MODEL_NAME = "gemma-4-31b-it"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are VoiceEquity, a friendly civic assistant designed to help
people who may have low literacy or limited English proficiency understand important
documents, forms, signs, and official notices.

When given an image and a question, you MUST:
1. Identify what the document or image shows in ONE short sentence.
2. Answer the user's specific question using the simplest possible language —
   short sentences, no jargon, no legalese.
3. ALWAYS respond in the EXACT same language the user's question is written in.
   If the question is in Spanish, answer in Spanish. If in French, answer in French.
   If in Arabic, answer in Arabic. Match the language precisely.
4. End your response with a section labeled "Next Step:" (translated to the user's
   language) containing ONE clear, concrete action the user should take right now.

Keep the total response under 150 words. Be warm, encouraging, and never
condescending. Assume the user is intelligent but may be unfamiliar with
bureaucratic language."""

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """
    Payload sent from the browser to /analyze.

    Fields:
      image_b64   – Base64-encoded image (JPEG or PNG). Empty string if no image.
      question    – The user's question, either typed or speech-to-text transcribed.
      language_hint – Optional ISO language code (e.g. 'es', 'fr') supplied by the
                      language selector.  Gemma auto-detects from the question text;
                      this hint is appended to the prompt for extra accuracy.
      mime_type   – MIME type of the uploaded image, e.g. 'image/jpeg'.
    """
    image_b64: str = ""
    question: str
    language_hint: Optional[str] = None
    mime_type: str = "image/jpeg"


class AnalyzeResponse(BaseModel):
    """
    Response returned to the browser.

    Fields:
      answer        – Plain-language answer from Gemma 4.
      detected_lang – Language Gemma responded in (extracted from response header).
    """
    answer: str
    detected_lang: str = "en"


# ---------------------------------------------------------------------------
# Helper – build the Gemma prompt parts list
# ---------------------------------------------------------------------------

def build_prompt_parts(request: AnalyzeRequest) -> list:
    """
    Construct the list of content 'parts' that the GenAI SDK expects.

    If an image was provided, it is included as an inline blob so that Gemma
    can perform visual understanding alongside the text question.
    The language hint (when given) is appended to the question text so Gemma
    has an unambiguous signal about the desired output language.
    """
    parts = []

    # Include the image when provided.
    if request.image_b64:
        image_bytes = base64.b64decode(request.image_b64)
        parts.append({"inline_data": {"mime_type": request.mime_type, "data": base64.b64encode(image_bytes).decode()}})

    # Build the text portion of the prompt.
    text_prompt = request.question
    if request.language_hint and request.language_hint != "auto":
        text_prompt += f"\n\n[Please respond in language code: {request.language_hint}]"

    parts.append({"text": text_prompt})
    return parts


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """
    Simple liveness check used by deployment platforms and the frontend
    to confirm the server is running before the user submits a request.
    """
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Core endpoint: receives an image + question and returns a plain-language
    answer from Gemma 4.

    Flow:
      1. Validate that we have at least a question.
      2. Build the multimodal prompt parts (image blob + text).
      3. Call the Gemma model through the Google GenAI SDK.
      4. Parse the text response and return it to the browser.

    Raises HTTP 400 if no question is provided.
    Raises HTTP 500 on any model or SDK error (with a user-friendly message).
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Please provide a question.")

    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server is missing GEMINI_API_KEY. Contact the administrator.",
        )

    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_PROMPT,
        )

        parts = build_prompt_parts(request)
        logger.info("Sending request to %s (image=%s)", MODEL_NAME, bool(request.image_b64))

        response = model.generate_content(parts)
        answer_text = response.text.strip()

        # Naively detect the response language from the language_hint so the
        # browser TTS voice selector can be pre-loaded with the right locale.
        # Gemma itself handles the actual translation; this is just metadata.
        detected = request.language_hint if request.language_hint and request.language_hint != "auto" else "en"

        return AnalyzeResponse(answer=answer_text, detected_lang=detected)

    except Exception as exc:
        logger.error("Gemma API error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Model error: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Serve the single-page frontend
# ---------------------------------------------------------------------------
# Mount static assets if they exist (e.g. icons, images added later).
# The index.html is served at the root so the app works as a single URL.

@app.get("/")
async def serve_index():
    """Serve the single-page frontend HTML file."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(index_path)


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
