import os
import re

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

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
    label = None
    reason = None
    confidence = None

    m = re.search(r"(?im)^\s*Label\s*:\s*(REAL|FAKE)\s*$", text)
    if m:
        label = m.group(1).upper()

    m = re.search(r"(?im)^\s*Confidence\s*:\s*(Low|Medium|High)\s*$", text)
    if m:
        confidence = m.group(1).title()

    m = re.search(r"(?is)^\s*Reason\s*:\s*(.*?)(?:\n\s*Confidence\s*:|\Z)", text)
    if m:
        reason = m.group(1).strip()

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


def _configure_gemini():
    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY).")
    genai.configure(api_key=api_key)


def _list_generate_models() -> list[str]:
    available = set()
    for mdl in genai.list_models():
        methods = set(getattr(mdl, "supported_generation_methods", []) or [])
        if "generateContent" in methods:
            name = getattr(mdl, "name", "")
            if name.startswith("models/"):
                name = name.split("/", 1)[1]
            if name:
                available.add(name)
    return sorted(available)


def _pick_models(require_image: bool) -> list[str]:
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


app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/analyze")
def analyze():
    """
    Accepts multipart/form-data:
      - text (optional)
      - image (optional file)
    """
    try:
        _configure_gemini()

        text = (request.form.get("text") or "").strip()
        image = request.files.get("image")

        if not text and not image:
            return jsonify({"ok": False, "error": "Provide text and/or an image."}), 400

        if image:
            image_bytes = image.read()
            if not image_bytes:
                return jsonify({"ok": False, "error": "Uploaded image is empty."}), 400

            contents = []
            if text:
                contents.append(
                    f"{SYSTEM_INSTRUCTION}\n\nNews text (if any):\n{text}\n\n"
                    "If the image contains text or context, use it too."
                )
            else:
                contents.append(
                    f"{SYSTEM_INSTRUCTION}\n\nAnalyze the uploaded image content for misinformation or fake news."
                )

            contents.append(
                {
                    "inline_data": {
                        "mime_type": image.mimetype or "image/jpeg",
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
                    return jsonify(
                        {
                            "ok": True,
                            "input": {"text": bool(text), "image": True},
                            "model": model_name,
                            "result": parsed,
                        }
                    )
                except Exception as e:
                    last_err = e

            if last_err and _is_quota_error(last_err):
                return (
                    jsonify(
                        {
                            "ok": False,
                            "error": (
                                "Gemini quota exceeded. Please check billing/quota in Google AI Studio, "
                                "wait for reset, or switch to another API key/project."
                            ),
                        }
                    ),
                    429,
                )
            raise last_err or RuntimeError("Generation failed.")

        prompt = f"{SYSTEM_INSTRUCTION}\n\nNews text:\n{text}"
        last_err = None
        for model_name in _pick_models(require_image=False):
            try:
                model = genai.GenerativeModel(model_name)
                resp = model.generate_content(prompt)
                parsed = _parse_factcheck_response(getattr(resp, "text", "") or "")
                return jsonify(
                    {
                        "ok": True,
                        "input": {"text": True, "image": False},
                        "model": model_name,
                        "result": parsed,
                    }
                )
            except Exception as e:
                last_err = e

        if last_err and _is_quota_error(last_err):
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": (
                            "Gemini quota exceeded. Please check billing/quota in Google AI Studio, "
                            "wait for reset, or switch to another API key/project."
                        ),
                    }
                ),
                429,
            )
        raise last_err or RuntimeError("Generation failed.")

    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": f"Server error: {e}"}), 500


if __name__ == "__main__":
    # Run: python app.py
    app.run(host="0.0.0.0", port=8000, debug=True)

