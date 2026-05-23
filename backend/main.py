import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import storage
import llm
import qa_agent
import pdf_parser
import analyzer
import law_loader


SEED_FILE = Path(__file__).parent.parent / "data" / "ny_law_seed.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.create_tables()
    if storage.get_law_count() == 0:
        count = storage.seed_from_file(str(SEED_FILE))
        print(f"Seeded {count} NY tenant laws from {SEED_FILE.name}")
    else:
        print(f"ClickHouse ready — {storage.get_law_count()} laws loaded")
    yield


app = FastAPI(title="HouseToHome API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "law_count": storage.get_law_count(),
        "provider": llm.provider(),
        "provider_label": llm.provider_label(),
    }


class UploadTextRequest(BaseModel):
    text: str
    session_id: str | None = None


@app.post("/upload")
async def upload(
    file: UploadFile | None = File(default=None),
    session_id: str | None = Form(default=None),
    text: str | None = Form(default=None),
):
    sid = session_id or str(uuid.uuid4())

    if file is not None:
        raw = await file.read()
        lease_text = pdf_parser.extract_text(raw)
        filename = file.filename or ""
    elif text:
        lease_text = text
        filename = ""
    else:
        return {"ok": False, "error": "Provide either a PDF file or text body"}

    storage.store_lease(sid, lease_text, filename)
    return {
        "session_id": sid,
        "char_count": len(lease_text),
        "ok": True,
        "lease_text": lease_text,
    }


@app.post("/upload/text")
async def upload_text(req: UploadTextRequest):
    sid = req.session_id or str(uuid.uuid4())
    storage.store_lease(sid, req.text)
    return {
        "session_id": sid,
        "char_count": len(req.text),
        "ok": True,
        "lease_text": req.text,
    }


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None


@app.post("/ask")
async def ask(req: AskRequest):
    return await qa_agent.answer(req.question, req.session_id)


@app.get("/analyze/stream")
async def analyze_stream(
    session_id: str = Query(...),
    preferences: str = Query(default="{}"),
):
    prefs = json.loads(preferences)
    lease_text = storage.get_lease(session_id)
    if not lease_text:
        async def error_stream():
            yield f"event: error\ndata: {json.dumps({'message': 'Session not found or no lease uploaded'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def event_stream():
        t0 = time.time()
        counts = {"total": 0, "RED": 0, "YELLOW": 0, "INFO": 0}
        try:
            yield f"event: stage\ndata: {json.dumps({'stage': 'law_lookup', 'message': 'Loading NY tenant laws from ClickHouse...'})}\n\n"
            yield f"event: stage\ndata: {json.dumps({'stage': 'analysis', 'message': f'{llm.provider_label()} analyzing your lease against preferences and law...'})}\n\n"

            async for finding in analyzer.analyze(lease_text, prefs, session_id):
                counts["total"] += 1
                counts[finding.severity] = counts.get(finding.severity, 0) + 1
                yield f"event: finding\ndata: {json.dumps({'finding': finding.model_dump()})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            duration_ms = int((time.time() - t0) * 1000)
            done_payload = {
                "total_findings": counts["total"],
                "critical": counts.get("RED", 0),
                "warnings": counts.get("YELLOW", 0),
                "duration_ms": duration_ms,
            }
            yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/seed-laws")
async def seed_laws():
    result = await law_loader.seed_from_nimble()
    return result
