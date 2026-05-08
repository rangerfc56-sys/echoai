#!/usr/bin/env python3
"""
FastAPI layer for echoai: audio -> Hinglish + English; English (+ domain template) -> clinical JSON.

Deploy on a GPU pod (single worker recommended — ASR model stays resident):

  uvicorn app:app --host 0.0.0.0 --port 8080 --workers 1
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Load .env before importing workflow (OpenAI/ASR paths)
try:
    from dotenv import load_dotenv

    _env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(_env):
        load_dotenv(_env, override=False)
except Exception:
    pass

import clinic_workflow as cw

API_TMP = os.path.join(cw.BASE_DIR, ".tmp", "api")
os.makedirs(API_TMP, exist_ok=True)


def _cors_origins() -> List[str]:
    raw = (os.getenv("CORS_ORIGINS") or "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [x.strip() for x in raw.split(",") if x.strip()]


app = FastAPI(
    title="echoai",
    description="Hinglish ASR + English conversation; domain-templated clinical summary JSON.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _english_transcript_to_str(raw: Union[str, Dict[str, Any], List[Any]]) -> str:
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            raise ValueError("english_transcript is empty")
        return s
    return json.dumps(raw, ensure_ascii=False, indent=2)


class TranscribeResponse(BaseModel):
    hinglish_transcript: str
    english_transcript: str = Field(description="English Doctor/Patient text — use as english_transcript for POST /v1/clinical-summary")
    english_conversation: str = Field(description="Same text as english_transcript (alias for older clients)")
    openai_model: str = Field(description="Model used for Hinglish→English step")
    hinglish_to_english_backend: str


class ClinicalSummaryRequest(BaseModel):
    """English transcript (string or JSON) + specialty template."""

    model_config = ConfigDict(extra="ignore")

    english_transcript: Optional[Union[str, Dict[str, Any], List[Any]]] = Field(
        default=None,
        description="Required unless english_conversation is set: plain text (Doctor:/Patient: lines) or structured JSON.",
    )
    medical_domain: str = Field(
        default="general_medicine",
        description="Template key: data/<medical_domain>.json (e.g. general_medicine, radiology).",
    )
    summary_backend: str = Field(default=cw.DEFAULT_SUMMARY_BACKEND)
    openai_model: Optional[str] = Field(default=None, description="Override OPENAI_MODEL for this call")
    english_conversation: Optional[str] = Field(
        default=None,
        description="Deprecated alias for english_transcript when it is plain text; ignored if english_transcript is set.",
    )

    @model_validator(mode="after")
    def _require_transcript(self) -> ClinicalSummaryRequest:
        has_et = self.english_transcript is not None
        has_ec = bool((self.english_conversation or "").strip())
        if not has_et and not has_ec:
            raise ValueError("Provide english_transcript (JSON or string) and medical_domain.")
        return self


class ClinicalSummaryResponse(BaseModel):
    clinical_summary: Dict[str, Any] = Field(description="Structured discharge-style summary fields")
    medical_domain: str
    summary_backend: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/domains")
def list_domains() -> Dict[str, Any]:
    manifest_path = os.path.join(cw.DOMAIN_DATA_DIR, "domains.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    keys = cw.list_medical_domains()
    return {"default": "general_medicine", "domains": keys}


@app.post("/v1/transcribe", response_model=TranscribeResponse)
def transcribe(audio: UploadFile = File(...)) -> TranscribeResponse:
    if not audio.filename:
        raise HTTPException(status_code=400, detail="Missing audio filename")
    suffix = os.path.splitext(audio.filename)[1] or ".wav"
    fd, path = tempfile.mkstemp(prefix="echoai_", suffix=suffix, dir=API_TMP)
    os.close(fd)
    try:
        with open(path, "wb") as out:
            shutil.copyfileobj(audio.file, out)
        hinglish = cw.transcribe_hinglish(path, progress=None)
        if not hinglish.strip():
            raise HTTPException(status_code=422, detail="No speech detected")
        model = cw.DEFAULT_OPENAI_MODEL
        english = cw.to_english_conversation(hinglish, model)
        return TranscribeResponse(
            hinglish_transcript=hinglish,
            english_transcript=english,
            english_conversation=english,
            openai_model=model,
            hinglish_to_english_backend=cw.DEFAULT_HINGLISH_TO_ENGLISH_BACKEND,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}") from e
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _resolve_summary_transcript(body: ClinicalSummaryRequest) -> str:
    """Prefer english_transcript; else deprecated english_conversation string."""
    if body.english_transcript is not None:
        return _english_transcript_to_str(body.english_transcript)
    return (body.english_conversation or "").strip()


@app.post("/v1/clinical-summary", response_model=ClinicalSummaryResponse)
def clinical_summary(body: ClinicalSummaryRequest) -> ClinicalSummaryResponse:
    try:
        text = _resolve_summary_transcript(body)
        if not text.strip():
            raise ValueError("Resolved transcript is empty.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    model = (body.openai_model or "").strip() or cw.DEFAULT_OPENAI_MODEL
    domain = (body.medical_domain or cw.DEFAULT_MEDICAL_DOMAIN).strip() or cw.DEFAULT_MEDICAL_DOMAIN
    backend = (body.summary_backend or cw.DEFAULT_SUMMARY_BACKEND).strip().lower()
    if backend not in cw.VALID_SUMMARY_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"summary_backend must be one of: {', '.join(cw.VALID_SUMMARY_BACKENDS)}",
        )
    try:
        out = cw.build_print_summary(
            text,
            model,
            summary_backend=backend,
            medical_domain=domain,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {e}") from e
    ps = out.get("print_summary")
    if not isinstance(ps, dict):
        raise HTTPException(status_code=500, detail="Invalid summary shape")
    return ClinicalSummaryResponse(
        clinical_summary=ps,
        medical_domain=cw.load_medical_domain_config(domain).get("key", domain),
        summary_backend=backend,
    )
