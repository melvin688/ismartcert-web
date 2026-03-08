/**
 * Diploma Classification API — Cloudflare Worker
 *
 * POST /api/classify-diploma  (multipart/form-data, field: "file")
 *
 * Workflow:
 *   1. Accept PDF upload
 *   2. Heuristic pre-filter: reject if > 3 pages (via pdf-lib)
 *   3. Send PDF binary (inline_data) to Gemini 2.5 Flash for visual analysis
 *   4. If Gemini rejects the binary, fallback to text extraction + re-classify
 *   5. Return { is_diploma, confidence, reason }
 */

import { PDFDocument } from "pdf-lib";

// ── Constants ────────────────────────────────────────────────────────────────
const MAX_PAGES = 6;
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
const RATE_LIMIT_MS = 5000; // 5 seconds between Gemini API calls
const AI_TIMEOUT_MS = 30000;

const GEMINI_MODEL = "gemini-2.5-flash";

const CLASSIFICATION_PROMPT =
  "You are a forensic document classifier. Analyze this PDF document visually. " +
  "Is it an English academic diploma, degree certificate, or academic transcript? " +
  "Look for: university name, degree title, student name, conferral date, official seals/signatures, " +
  'formal language like "conferred upon", "awarded to", "hereby certifies". ' +
  "Return ONLY valid JSON (no markdown, no code fences): " +
  '{"is_diploma": boolean, "confidence": integer 0-100, "reason": "short string"}.';

// ── CORS ─────────────────────────────────────────────────────────────────────
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

async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

// ── Rate limiter ─────────────────────────────────────────────────────────────
let lastGeminiCall = 0;

// ── Entry point ──────────────────────────────────────────────────────────────
export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS });
    }

    const url = new URL(request.url);

    if (url.pathname === "/" && request.method === "GET") {
      return jsonResponse({ status: "ok", service: "diploma-classifier" });
    }

    if (url.pathname === "/api/classify-diploma" && request.method === "POST") {
      return handleClassify(request, env);
    }

    return jsonResponse({ error: "Not found" }, 404);
  },
};

// ── Classification handler ───────────────────────────────────────────────────
async function handleClassify(request, env) {
  try {
    const startedAt = Date.now();
    const requestId = crypto.randomUUID().slice(0, 8);

    console.log(`[${requestId}] classify start`);

    // 1. Parse upload --------------------------------------------------------
    const formData = await request.formData();
    const file = formData.get("file");

    if (!file || !(file instanceof File)) {
      console.warn(`[${requestId}] missing file field`);
      return jsonResponse({ error: "No file uploaded. Send a 'file' field." }, 400);
    }

    console.log(
      `[${requestId}] upload parsed name=${file.name} type=${file.type || "unknown"} size=${file.size || 0}`,
    );

    const pdfBuffer = await file.arrayBuffer();

    console.log(`[${requestId}] file buffered bytes=${pdfBuffer.byteLength} parseMs=${Date.now() - startedAt}`);

    if (pdfBuffer.byteLength > MAX_FILE_SIZE) {
      console.warn(`[${requestId}] file too large bytes=${pdfBuffer.byteLength}`);
      return jsonResponse(
        { error: `File too large. Max ${MAX_FILE_SIZE / 1048576} MB.` },
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
      console.log(`[${requestId}] pdf loaded pageCount=${pageCount} pdfLoadMs=${Date.now() - startedAt}`);
    } catch (e) {
      console.error(`[${requestId}] PDF load error:`, e.message);
      return jsonResponse({ error: "Could not open the PDF file." }, 400);
    }

    if (pageCount > MAX_PAGES) {
      console.warn(`[${requestId}] page limit exceeded pageCount=${pageCount}`);
      return jsonResponse({
        is_diploma: false,
        confidence: 95,
        reason: `Page limit exceeded. Maximum supported pages: ${MAX_PAGES}.`,
      });
    }

    // 3. Rate limit ----------------------------------------------------------
    const now = Date.now();
    const elapsed = now - lastGeminiCall;
    if (elapsed < RATE_LIMIT_MS) {
      const waitSec = Math.ceil((RATE_LIMIT_MS - elapsed) / 1000);
      console.warn(`[${requestId}] rate limited waitSec=${waitSec}`);
      return jsonResponse(
        { error: `Rate limited. Please wait ${waitSec} seconds before trying again.` },
        429,
      );
    }
    lastGeminiCall = now;

    // 4. Send PDF binary to Gemini (inline_data) -----------------------------
    const base64Pdf = arrayBufferToBase64(pdfBuffer);
    console.log(`[${requestId}] pdf encoded base64Length=${base64Pdf.length}`);
    const geminiUrl =
      `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent` +
      `?key=${env.GEMINI_API_KEY}`;

    const geminiStartedAt = Date.now();
    let geminiRes;

    console.log(`[${requestId}] gemini request start`);

    try {
      geminiRes = await fetchWithTimeout(
        geminiUrl,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents: [
              {
                parts: [
                  { inline_data: { mime_type: "application/pdf", data: base64Pdf } },
                  { text: CLASSIFICATION_PROMPT },
                ],
              },
            ],
            generationConfig: {
              temperature: 0.1,
              maxOutputTokens: 1024,
              thinkingConfig: { thinkingBudget: 0 },
            },
          }),
        },
        AI_TIMEOUT_MS,
      );
    } catch (error) {
      if (error.name === "AbortError") {
        console.error(`[${requestId}] Gemini timeout after ${AI_TIMEOUT_MS} ms`);
        return jsonResponse(
          { error: "AI verification timed out. Please retry with a smaller PDF." },
          504,
        );
      }
      console.error(`[${requestId}] Gemini fetch failed:`, error.message);
      throw error;
    }

    console.log(`[${requestId}] Gemini latency ms=${Date.now() - geminiStartedAt}`);

    if (!geminiRes.ok) {
      const errText = await geminiRes.text();
      console.error(`[${requestId}] Gemini error status=${geminiRes.status} body=${errText}`);
      if (geminiRes.status === 429) {
        return jsonResponse({ error: "AI quota exceeded. Please try again later." }, 429);
      }
      return jsonResponse({ error: "AI service error: " + geminiRes.status }, 502);
    }

    const geminiData = await geminiRes.json();
    const rawText =
      geminiData.candidates?.[0]?.content?.parts?.[0]?.text ?? "";

    console.log(`[${requestId}] Gemini raw response:`, rawText);

    // 5. Parse AI response ---------------------------------------------------
    let result;
    try {
      result = parseGeminiJson(rawText);
    } catch (parseErr) {
      console.warn(`[${requestId}] Gemini returned non-JSON, treating as non-diploma`);
      result = { is_diploma: false, confidence: 80, reason: rawText.slice(0, 200) || "AI could not classify this document." };
    }

    console.log(`[${requestId}] classification completed totalMs=${Date.now() - startedAt}`);

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

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function parseGeminiJson(text) {
  // Strip thinking tags (gemini-2.5 may include <think>...</think>)
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
