from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import os

from chatbot import init_session_metrics, process_user_query, all_conditions


app = FastAPI(title="Medical Chatbot Model API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    query: str
    patient_name: str = "Patient"


QUERY_CACHE = {}
SESSION_METRICS = init_session_metrics()
if os.getenv("VERCEL"):
    REPORT_DIR = Path("/tmp/generated_reports")
else:
    REPORT_DIR = Path("generated_reports")
REPORT_DIR.mkdir(exist_ok=True)


def _write_basic_pdf(text, output_path: Path):
    lines = [line[:110] for line in text.splitlines()]
    lines = lines[:90] if lines else ["No prescription generated yet."]

    def esc(s):
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content_lines = ["BT", "/F1 10 Tf", "50 790 Td", "12 TL"]
    for idx, line in enumerate(lines):
        if idx == 0:
            content_lines.append(f"({esc(line)}) Tj")
        else:
            content_lines.append("T*")
            content_lines.append(f"({esc(line)}) Tj")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", errors="replace")

    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
    )
    objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Courier >> endobj\n")
    objects.append(
        f"5 0 obj << /Length {len(stream)} >> stream\n".encode("latin-1") + stream + b"\nendstream endobj\n"
    )

    pdf = bytearray(b"%PDF-1.4\n")
    xref_positions = [0]
    for obj in objects:
        xref_positions.append(len(pdf))
        pdf.extend(obj)

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(xref_positions)}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for pos in xref_positions[1:]:
        pdf.extend(f"{pos:010d} 00000 n \n".encode("latin-1"))
    pdf.extend((f"trailer << /Size {len(xref_positions)} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF").encode("latin-1"))

    output_path.write_bytes(pdf)


def parse_prescription_table(message: str):
    """Extract key/value fields from the ASCII prescription table output."""
    if not message:
        return {}

    parsed = {}
    current_field = None

    for raw_line in message.splitlines():
        line = raw_line.rstrip()
        if "|" not in line:
            continue
        if line.strip().startswith("+"):
            continue

        # Split table cells and remove edge bars.
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) < 2:
            continue

        field, value = parts[0], parts[1]
        if field == "Field" and value == "Details":
            continue

        if field:
            current_field = field
            parsed[current_field] = value
        elif current_field:
            parsed[current_field] = (parsed.get(current_field, "") + " " + value).strip()

    top_related = parsed.get("Top Related Matches", "")
    possible_conditions = []
    if top_related:
        for chunk in top_related.split(";"):
            item = chunk.strip()
            if not item:
                continue
            name = item.split("(")[0].strip()
            if name:
                possible_conditions.append(name)

    description_value = parsed.get("Medical Definition", "") or parsed.get("Description", "")
    cause_value = parsed.get("Etiology (Causes)", "") or parsed.get("Cause", "")
    symptoms_value = parsed.get("Clinical Symptoms", "") or parsed.get("Symptoms", "")
    care_plan_value = parsed.get("Management / Solution", "") or parsed.get("Solution / Care Plan", "")

    return {
        "condition": parsed.get("Probable Condition", ""),
        "confidence": parsed.get("Confidence", ""),
        "description": description_value,
        "cause": cause_value,
        "symptoms": symptoms_value,
        "care_plan": care_plan_value,
        "urgent": parsed.get("Seek Urgent Care If", ""),
        "sources": parsed.get("Verified Sources", ""),
        "possible_conditions": possible_conditions,
        "raw_fields": parsed,
    }


@app.get("/health")
def health():
    return {"ok": True, "service": "medical-chatbot-model-api"}


@app.get("/conditions")
def conditions():
    return {"conditions": [c.title() for c in all_conditions()]}


@app.post("/predict")
def predict(payload: PredictRequest):
    result = process_user_query(
        payload.query,
        payload.patient_name,
        QUERY_CACHE,
        SESSION_METRICS,
    )
    message = result.get("message", "")

    return {
        "status": result.get("status", "unknown"),
        "patient_name": result.get("patient_name", payload.patient_name),
        "message": message,
        "structured": parse_prescription_table(message),
    }


@app.post("/predict/pdf")
def predict_pdf(payload: PredictRequest):
    result = process_user_query(
        payload.query,
        payload.patient_name,
        QUERY_CACHE,
        SESSION_METRICS,
    )
    message = result.get("message", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch for ch in payload.patient_name if ch.isalnum() or ch in ("_", "-")) or "Patient"
    file_path = REPORT_DIR / f"prescription_{safe_name}_{timestamp}.pdf"
    _write_basic_pdf(message, file_path)

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/pdf",
    )
