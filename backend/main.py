"""
Document Classification API
Determines if an uploaded English PDF is an academic diploma/degree certificate.
Uses Gemini 1.5 Flash for zero-cost multimodal classification.
"""

import io
import json
import os
import re
import logging

import fitz  # PyMuPDF
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MAX_PAGES = 3
MAX_FILE_SIZE_MB = 10
RENDER_DPI = 200  # good balance between quality and size

CLASSIFICATION_PROMPT = (
    "You are a forensic document classifier. Analyze this image. "
    "Is it an English academic diploma or degree certificate? "
    "Return ONLY valid JSON with the following structure: "
    '{"is_diploma": boolean, "confidence": integer 0-100, "reason": "short string"}.'
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Diploma Classification API",
    version="1.0.0",
    description="Lightweight API to classify whether a PDF is an academic diploma.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Gemini client (lazy init)
# ---------------------------------------------------------------------------
_gemini_client: genai.Client | None = None


def get_gemini_client() -> genai.Client:
    """Return a cached Gemini client, creating one on first call."""
    global _gemini_client
    if _gemini_client is None:
        api_key = GEMINI_API_KEY
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY environment variable is not set.",
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_first_page_jpeg(pdf_bytes: bytes) -> bytes:
    """Open a PDF from bytes and render the first page as JPEG."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=RENDER_DPI)
    jpeg_bytes = pix.tobytes(output="jpeg", jpg_quality=85)
    doc.close()
    return jpeg_bytes


def parse_gemini_json(text: str) -> dict:
    """
    Extract and parse JSON from the Gemini response text.
    Handles cases where the model wraps JSON in markdown code fences.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = cleaned.rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: find first { ... } block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from Gemini response: {text!r}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def health():
    return {"status": "ok", "service": "diploma-classifier"}


@app.post("/api/classify-diploma")
async def classify_diploma(file: UploadFile = File(...)):
    """
    Accept a PDF upload, pre-filter by page count, render page 1 as JPEG,
    and classify via Gemini 1.5 Flash.
    """

    # --- Validate content type -------------------------------------------------
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Only PDF files are accepted.",
        )

    # --- Read & size-check -----------------------------------------------------
    pdf_bytes = await file.read()
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB.",
        )

    # --- Heuristic pre-filtering: page count -----------------------------------
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = doc.page_count
        doc.close()
    except Exception as exc:
        logger.error("Failed to open PDF: %s", exc)
        raise HTTPException(status_code=400, detail="Could not open the PDF file.")

    if page_count > MAX_PAGES:
        logger.info("Rejected: %d pages (limit %d)", page_count, MAX_PAGES)
        return {
            "is_diploma": False,
            "confidence": 95,
            "reason": "Page limit exceeded",
        }

    # --- Render first page to JPEG ---------------------------------------------
    try:
        jpeg_bytes = render_first_page_jpeg(pdf_bytes)
    except Exception as exc:
        logger.error("Failed to render PDF page: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to render the PDF page.")

    logger.info(
        "PDF accepted — pages=%d, pdf_size=%.1f KB, jpeg_size=%.1f KB",
        page_count,
        len(pdf_bytes) / 1024,
        len(jpeg_bytes) / 1024,
    )

    # --- Classify via Gemini 1.5 Flash -----------------------------------------
    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                        types.Part.from_text(text=CLASSIFICATION_PROMPT),
                    ]
                )
            ],
        )
        raw_text = response.text
        logger.info("Gemini raw response: %s", raw_text)
    except Exception as exc:
        logger.error("Gemini API call failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"AI classification service error: {exc}",
        )

    # --- Parse and return ------------------------------------------------------
    try:
        result = parse_gemini_json(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("Failed to parse Gemini JSON: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="AI returned an unparseable response.",
        )

    # Normalise types to be safe
    return {
        "is_diploma": bool(result.get("is_diploma", False)),
        "confidence": int(result.get("confidence", 0)),
        "reason": str(result.get("reason", "Unknown")),
    }


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
