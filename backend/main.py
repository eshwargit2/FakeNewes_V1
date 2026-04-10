import os
import re
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Gemini SDK (Google Generative AI)
import google.generativeai as genai

SYSTEM_INSTRUCTION = (
    "You are a professional fact-checker. Analyze the given content and classify it as REAL or FAKE. "
    "Consider emotional language, logical consistency, credibility, and misinformation patterns. Respond in this format:\n"
    "Label: REAL or FAKE\n"
    "Reason:\n"
    "Confidence: Low/Medium/High"
)

PREFERRED_TEXT_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

PREFERRED_MULTIMODAL_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


def _parse_factcheck_response(text: str) -> dict:
    """
    Parse the model's text response into a structured JSON dict.
    Falls back gracefully if the response isn't perfectly formatted.
    """
    label = None
    reason = None
    confidence = None

    # Match "Label: REAL" / "Label: FAKE"
    m = re.search(r"(?im)^\s*Label\s*:\s*(REAL|FAKE)\s*$", text)
    if m:
        label = m.group(1).upper()

    # Match "Confidence: Low/Medium/High"
    m = re.search(r"(?im)^\s*Confidence\s*:\s*(Low|Medium|High)\s*$", text)
    if m:
        confidence = m.group(1).title()

    # Try to capture everything after "Reason:" up to "Confidence:" (or end)
    m = re.search(r"(?is)^\s*Reason\s*:\s*(.*?)(?:\n\s*Confidence\s*:|\Z)", text)
    if m:
        reason = m.group(1).strip()

    # Fallbacks
    if not reason:
        reason = text.strip()

    if label not in {"REAL", "FAKE"}:
        label = "UNKNOWN"

    if confidence not in {"Low", "Medium", "High"}:
        confidence = "Unknown"

    return {
        "label": label,
        "reason": reason,
        "confidence": confidence,
        "raw": text.strip(),
    }


def _require_api_key() -> str:
    """
    Loads GEMINI_API_KEY from environment (supports .env).
    """
    # Try both common locations for this project:
    # 1) project root .env
    # 2) backend/.env
    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set GEMINI_API_KEY in your environment (or in a .env file)."
        )
    return api_key


def _configure_gemini():
    api_key = _require_api_key()
    genai.configure(api_key=api_key)


def _list_generate_models() -> list[str]:
    """List models that support generateContent."""
    available = set()
    for mdl in genai.list_models():
        methods = set(getattr(mdl, "supported_generation_methods", []) or [])
        if "generateContent" in methods:
            name = getattr(mdl, "name", "")  # e.g. "models/gemini-1.5-flash"
            if name.startswith("models/"):
                name = name.split("/", 1)[1]
            if name:
                available.add(name)
    return sorted(available)


def _pick_models(require_image: bool) -> list[str]:
    """
    Return an ordered list of candidate models:
    preferred models first, then all other available ones.
    """
    preferred = PREFERRED_MULTIMODAL_MODELS if require_image else PREFERRED_TEXT_MODELS
    available = _list_generate_models()
    if not available:
        raise RuntimeError("No Gemini model with generateContent is available for this API key.")

    ordered: list[str] = []
    for name in preferred:
        if name in available:
            ordered.append(name)
    for name in available:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _is_quota_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "429" in msg or "quota" in msg or "rate limit" in msg


app = FastAPI(title="Fake News Detection API (Gemini)")

# Allow local frontend dev (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze")
async def analyze(
    text: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
):
    """
    Analyze fake/real news from text and/or image.
    """
    if (text is None or not text.strip()) and image is None:
        raise HTTPException(status_code=400, detail="Provide text and/or an image.")

    try:
        _configure_gemini()

        user_text = (text or "").strip()

        # If image exists, use vision model and pass both image + text
        if image is not None:
            contents = []
            if user_text:
                contents.append(
                    f"{SYSTEM_INSTRUCTION}\n\nNews text (if any):\n{user_text}\n\n"
                    "If the image contains text or context, use it too."
                )
            else:
                contents.append(
                    f"{SYSTEM_INSTRUCTION}\n\n"
                    "Analyze the uploaded image content for misinformation or fake news."
                )

            image_bytes = await image.read()
            if not image_bytes:
                raise HTTPException(status_code=400, detail="Uploaded image is empty.")

            # Gemini Vision expects an "inline_data" dict with base64-like bytes payload.
            contents.append(
                {
                    "inline_data": {
                        "mime_type": image.content_type or "image/jpeg",
                        "data": image_bytes,
                    }
                }
            )

            last_err = None
            for model_name in _pick_models(require_image=True):
                try:
                    model = genai.GenerativeModel(model_name)
                    resp = model.generate_content(contents)
                    parsed = _parse_factcheck_response(getattr(resp, "text", "") or "")
                    return {
                        "ok": True,
                        "input": {"text": bool(user_text), "image": True},
                        "model": model_name,
                        "result": parsed,
                    }
                except Exception as e:
                    last_err = e

            if last_err and _is_quota_error(last_err):
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "Gemini quota exceeded. Please check billing/quota in Google AI Studio, "
                        "wait for reset, or switch to another API key/project."
                    ),
                )
            raise last_err or RuntimeError("Generation failed.")

        # Otherwise text-only
        prompt = f"{SYSTEM_INSTRUCTION}\n\nNews text:\n{user_text}"
        last_err = None
        for model_name in _pick_models(require_image=False):
            try:
                model = genai.GenerativeModel(model_name)
                resp = model.generate_content(prompt)
                parsed = _parse_factcheck_response(getattr(resp, "text", "") or "")
                return {
                    "ok": True,
                    "input": {"text": True, "image": False},
                    "model": model_name,
                    "result": parsed,
                }
            except Exception as e:
                last_err = e

        if last_err and _is_quota_error(last_err):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini quota exceeded. Please check billing/quota in Google AI Studio, "
                    "wait for reset, or switch to another API key/project."
                ),
            )
        raise last_err or RuntimeError("Generation failed.")

    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Keep this beginner-friendly: return a safe error message
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

