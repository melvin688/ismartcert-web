/**
 * Diploma Classification API — Cloudflare Worker
 *
 * POST /api/classify-diploma  (multipart/form-data, field: "file")
 *
 * Workflow:
 *   1. Accept PDF upload
 *   2. Heuristic pre-filter: reject if > 3 pages (via pdf-lib)
 *   3. Extract text from PDF (unpdf)
 *   4. Send extracted text to Gemini 2.5 Flash-Lite for classification
 *   5. Return { is_diploma, confidence, reason }
 */

import { PDFDocument } from "pdf-lib";
import { extractText } from "unpdf";

// ── Constants ────────────────────────────────────────────────────────────────
const MAX_PAGES = 3;
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
const RATE_LIMIT_MS = 15000; // Minimum 15 seconds between Gemini API calls

const CLASSIFICATION_PROMPT = `You are a forensic document classifier specializing in academic credentials.

Analyze the following text extracted from a PDF document. Determine whether this document is an academic diploma, degree certificate, or academic transcript.

Look for these key indicators:
- University or institution name
- Degree title (Bachelor, Master, PhD, Diploma, etc.)
- Student/graduate name
- Conferral/graduation date
- Official language like "conferred upon", "awarded to", "has completed", "hereby certifies"
- Academic fields of study
- Signatures of officials (Dean, Registrar, Chancellor, etc.)

If you find MULTIPLE strong indicators above, classify as a diploma (is_diploma: true).
If the text is mostly empty or has very few words, it may be a scanned/image-based diploma — still classify as true if the few words present suggest academic credentials.

Return ONLY valid JSON: {"is_diploma": boolean, "confidence": integer 0-100, "reason": "short explanation"}`;

// ── CORS headers (allow all origins) ─────────────────────────────────────────
const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...CORS, "Content-Type": "application/json" },
  });
}

// ── Rate limiter (per-isolate, prevents rapid-fire Gemini calls) ─────────────
let lastGeminiCall = 0;

// ── Entry point ──────────────────────────────────────────────────────────────
export default {
  async fetch(request, env) {
    // Preflight
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS });
    }

    const url = new URL(request.url);

    // Health
    if (url.pathname === "/" && request.method === "GET") {
      return jsonResponse({ status: "ok", service: "diploma-classifier" });
    }

    // Classification
    if (url.pathname === "/api/classify-diploma" && request.method === "POST") {
      return handleClassify(request, env);
    }

    return jsonResponse({ error: "Not found" }, 404);
  },
};

// ── Classification handler ───────────────────────────────────────────────────
async function handleClassify(request, env) {
  try {
    // 1. Parse upload --------------------------------------------------------
    const formData = await request.formData();
    const file = formData.get("file");

    if (!file || !(file instanceof File)) {
      return jsonResponse({ error: "No file uploaded. Send a 'file' field." }, 400);
    }

    const pdfBuffer = await file.arrayBuffer();

    if (pdfBuffer.byteLength > MAX_FILE_SIZE) {
      return jsonResponse(
        { error: `File too large (${(pdfBuffer.byteLength / 1048576).toFixed(1)} MB). Max ${MAX_FILE_SIZE / 1048576} MB.` },
        400,
      );
    }

    // 2. Heuristic: page count -----------------------------------------------
    let pageCount;
    try {
      const pdfDoc = await PDFDocument.load(pdfBuffer, {
        ignoreEncryption: true,
        updateMetadata: false,
      });
      pageCount = pdfDoc.getPageCount();
    } catch (e) {
      console.error("PDF load error:", e.message);
      return jsonResponse({ error: "Could not open the PDF file." }, 400);
    }

    if (pageCount > MAX_PAGES) {
      return jsonResponse({
        is_diploma: false,
        confidence: 95,
        reason: "Page limit exceeded — diplomas are typically 1 page.",
      });
    }

    // 3. Extract text from PDF -----------------------------------------------
    let extractedText = "";
    try {
      const result = await extractText(new Uint8Array(pdfBuffer));
      extractedText = result.text.trim();
      console.log(`Extracted ${extractedText.length} chars from ${result.totalPages} pages`);
    } catch (e) {
      console.error("Text extraction error:", e.message);
      // If extraction fails, try raw extraction as fallback
      extractedText = extractRawText(pdfBuffer);
    }

    // If very little text found, it might be a scanned/image-based PDF
    if (extractedText.length < 10) {
      extractedText =
        "(Very little or no text could be extracted. It may be a scanned/image-based document.) Raw content hints: " +
        extractRawText(pdfBuffer);
    }

    // Limit text to avoid excessive token usage
    if (extractedText.length > 5000) {
      extractedText = extractedText.substring(0, 5000) + "\n[...truncated]";
    }

    // 4. Rate limit — prevent rapid-fire Gemini calls -------------------------
    const now = Date.now();
    const elapsed = now - lastGeminiCall;
    if (elapsed < RATE_LIMIT_MS) {
      const waitSec = Math.ceil((RATE_LIMIT_MS - elapsed) / 1000);
      return jsonResponse(
        { error: `Rate limited. Please wait ${waitSec} seconds before trying again.` },
        429,
      );
    }
    lastGeminiCall = now;

    // 5. Send extracted text to Gemini 2.5 Flash-Lite -------------------------
    const geminiUrl =
      "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent" +
      `?key=${env.GEMINI_API_KEY}`;

    const geminiRes = await fetch(geminiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              {
                text:
                  CLASSIFICATION_PROMPT +
                  "\n\n--- EXTRACTED PDF TEXT ---\n" +
                  extractedText,
              },
            ],
          },
        ],
        generationConfig: {
          temperature: 0.1,
          maxOutputTokens: 256,
        },
      }),
    });

    if (!geminiRes.ok) {
      const errText = await geminiRes.text();
      console.error("Gemini error:", geminiRes.status, errText);
      if (geminiRes.status === 429) {
        return jsonResponse(
          { error: "AI quota exceeded. Please try again later." },
          429,
        );
      }
      return jsonResponse({ error: "AI service error: " + geminiRes.status }, 502);
    }

    const geminiData = await geminiRes.json();
    const rawText =
      geminiData.candidates?.[0]?.content?.parts?.[0]?.text ?? "";

    console.log("Gemini raw response:", rawText);

    // 6. Parse AI response ---------------------------------------------------
    const result = parseGeminiJson(rawText);

    return jsonResponse({
      is_diploma: Boolean(result.is_diploma),
      confidence: Number(result.confidence) || 0,
      reason: String(result.reason || "Unknown"),
    });
  } catch (err) {
    console.error("Unhandled error:", err.message, err.stack);
    return jsonResponse({ error: "Internal server error: " + err.message }, 500);
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Fallback: extract readable ASCII/UTF text strings from raw PDF bytes.
 * Useful when structured text extraction fails (encrypted/scanned PDFs).
 */
function extractRawText(buffer) {
  const bytes = new Uint8Array(buffer);
  const chunks = [];
  let current = "";

  for (let i = 0; i < bytes.length && i < 200000; i++) {
    const c = bytes[i];
    // Printable ASCII range
    if (c >= 32 && c <= 126) {
      current += String.fromCharCode(c);
    } else {
      if (current.length > 4) {
        chunks.push(current);
      }
      current = "";
    }
  }
  if (current.length > 4) chunks.push(current);

  // Filter out PDF structural commands, keep likely text content
  const filtered = chunks.filter((s) => {
    if (s.startsWith("/") || s.startsWith("<<") || s.startsWith(">>"))
      return false;
    if (/^[\d\s.]+$/.test(s)) return false;
    if (
      /^(obj|endobj|stream|endstream|xref|trailer|startxref)$/i.test(s.trim())
    )
      return false;
    return true;
  });

  return filtered.join(" ").substring(0, 3000);
}

function parseGeminiJson(text) {
  // Strip markdown code fences and thinking tags
  let cleaned = text
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .replace(/```(?:json)?\s*/g, "")
    .replace(/```/g, "")
    .trim();
  try {
    return JSON.parse(cleaned);
  } catch {
    const match = cleaned.match(/\{[\s\S]*\}/);
    if (match) return JSON.parse(match[0]);
    throw new Error("Could not parse AI response: " + text);
  }
}
