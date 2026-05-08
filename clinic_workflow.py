#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOMAIN_DATA_DIR = os.path.join(BASE_DIR, "data")
LOCAL_TMP_DIR = os.path.join(BASE_DIR, ".tmp", "gradio")
os.makedirs(LOCAL_TMP_DIR, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", LOCAL_TMP_DIR)

try:
    from dotenv import load_dotenv

    _ENV_FILE = os.path.join(BASE_DIR, ".env")
    if os.path.exists(_ENV_FILE):
        load_dotenv(_ENV_FILE, override=False)
except Exception:
    pass

DEFAULT_MEDICAL_DOMAIN = (os.getenv("MEDICAL_DOMAIN") or "general_medicine").strip().lower() or "general_medicine"

import gradio as gr
import librosa
import numpy as np
import torch
from openai import APIStatusError, OpenAI
from transformers import pipeline

ASR_MODEL_DIR = os.getenv(
    "ASR_MODEL_DIR", "/home/surya/anuj/script2video/whisper-hindi2hinglish-prime"
)
ASR_FALLBACK_MODEL_DIRS = [
    p.strip()
    for p in (os.getenv("ASR_FALLBACK_MODEL_DIRS") or "").split(",")
    if p.strip()
]
if not ASR_FALLBACK_MODEL_DIRS:
    ASR_FALLBACK_MODEL_DIRS = [
        "/home/surya/anuj/script2video/whisper-hindi-large-v2",
        "/home/surya/anuj/script2video/whisper-hindi-small",
    ]
TARGET_SR = 16000
CHUNK_SECONDS = int(os.getenv("ASR_CHUNK_SECONDS", "15"))
CHUNK_OVERLAP_SECONDS = float(os.getenv("ASR_CHUNK_OVERLAP_SECONDS", "2.0"))
MIN_CHUNK_SECONDS = 0.2
# Default: OpenAI flagship (see https://platform.openai.com/docs/models); override with OPENAI_MODEL (e.g. o3, gpt-5.4-mini).
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
DEFAULT_PORT = int(os.getenv("CLINIC_WORKFLOW_PORT", "7865"))
VALID_HINGLISH_TO_ENGLISH_BACKENDS = ("openai", "anthropic")
_h2e_backend_env = (os.getenv("HINGLISH_TO_ENGLISH_BACKEND", "openai") or "openai").strip().lower()
DEFAULT_HINGLISH_TO_ENGLISH_BACKEND = (
    _h2e_backend_env
    if _h2e_backend_env in VALID_HINGLISH_TO_ENGLISH_BACKENDS
    else "openai"
)
DEFAULT_ANTHROPIC_TRANSLATION_MODEL = os.getenv(
    "ANTHROPIC_TRANSLATION_MODEL", "claude-opus-4-6"
)
VALID_SUMMARY_BACKENDS = ("openai", "biobert")
_summary_backend_env = (os.getenv("SUMMARY_BACKEND", "openai") or "openai").strip().lower()
DEFAULT_SUMMARY_BACKEND = (
    _summary_backend_env if _summary_backend_env in VALID_SUMMARY_BACKENDS else "openai"
)

try:
    import textbook_rag
except ImportError:
    textbook_rag = None  # type: ignore

_ASR_PIPELINE = None


def _has_model_weights(model_dir: str) -> bool:
    if not model_dir or not os.path.isdir(model_dir):
        return False
    try:
        names = os.listdir(model_dir)
    except Exception:
        return False
    for name in names:
        if name == "pytorch_model.bin":
            return True
        if name.endswith(".safetensors") and ("model" in name.lower() or name.lower() == "model.safetensors"):
            return True
    return False


def _candidate_asr_model_dirs() -> List[str]:
    ordered = [ASR_MODEL_DIR, *ASR_FALLBACK_MODEL_DIRS]
    unique: List[str] = []
    seen = set()
    for path in ordered:
        p = (path or "").strip()
        if not p or p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


def _load_audio_with_ffmpeg(audio_path: str) -> Tuple[np.ndarray, int]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        audio_path,
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="ignore").strip()
        raise ValueError(f"ffmpeg decode failed: {stderr or f'rc={proc.returncode}'}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("ffmpeg decode produced empty audio.")
    return audio, TARGET_SR


def load_audio_mono(audio_path: str) -> Tuple[np.ndarray, int]:
    try:
        return _load_audio_with_ffmpeg(audio_path)
    except Exception:
        audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        if audio is None or len(audio) == 0 or not np.isfinite(audio).all():
            raise ValueError("Could not decode valid audio data.")
        return audio, sr


def get_asr_pipeline():
    global _ASR_PIPELINE
    if _ASR_PIPELINE is not None:
        return _ASR_PIPELINE
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    errors: List[str] = []
    for model_dir in _candidate_asr_model_dirs():
        if not _has_model_weights(model_dir):
            errors.append(f"{model_dir}: missing model weights")
            continue
        try:
            _ASR_PIPELINE = pipeline(
                "automatic-speech-recognition",
                model=model_dir,
                tokenizer=model_dir,
                feature_extractor=model_dir,
                dtype=dtype,
                device=device,
            )
            if model_dir != ASR_MODEL_DIR:
                print(f"[ASR] Falling back to model: {model_dir}")
            return _ASR_PIPELINE
        except Exception as e:
            errors.append(f"{model_dir}: {e}")
    raise RuntimeError("Unable to initialize ASR pipeline. " + " | ".join(errors))


def chunk_audio(
    audio: np.ndarray,
    sr: int,
    chunk_seconds: int = CHUNK_SECONDS,
    overlap_seconds: float = CHUNK_OVERLAP_SECONDS,
) -> List[np.ndarray]:
    chunk_len = int(sr * chunk_seconds)
    overlap_len = int(sr * max(0.0, overlap_seconds))
    min_len = int(sr * MIN_CHUNK_SECONDS)
    if chunk_len <= 0:
        return []
    if overlap_len >= chunk_len:
        overlap_len = max(0, chunk_len // 3)
    step = max(1, chunk_len - overlap_len)
    chunks: List[np.ndarray] = []
    for start in range(0, len(audio), step):
        chunk = audio[start : start + chunk_len]
        if len(chunk) >= min_len:
            chunks.append(chunk)
        if start + chunk_len >= len(audio):
            break
    return chunks


def _norm_merge_token(token: str) -> str:
    return token.strip(".,!?;:\"'()[]{}").lower()


def merge_chunk_texts(chunk_texts: List[str], min_overlap_tokens: int = 2, max_overlap_tokens: int = 24) -> str:
    merged_tokens: List[str] = []
    for raw in chunk_texts:
        text = (raw or "").strip()
        if not text:
            continue
        cur_tokens = text.split()
        if not cur_tokens:
            continue
        if not merged_tokens:
            merged_tokens.extend(cur_tokens)
            continue

        upper = min(max_overlap_tokens, len(merged_tokens), len(cur_tokens))
        overlap = 0
        for k in range(upper, min_overlap_tokens - 1, -1):
            tail = [_norm_merge_token(t) for t in merged_tokens[-k:]]
            head = [_norm_merge_token(t) for t in cur_tokens[:k]]
            if tail == head:
                overlap = k
                break
        merged_tokens.extend(cur_tokens[overlap:])
    return " ".join(merged_tokens).strip()


def transcribe_hinglish(audio_path: str, progress: Optional[gr.Progress] = None) -> str:
    audio, sr = load_audio_mono(audio_path)
    chunks = chunk_audio(audio, sr, CHUNK_SECONDS, CHUNK_OVERLAP_SECONDS)
    if not chunks:
        return ""
    asr = get_asr_pipeline()
    all_text: List[str] = []
    total = len(chunks)
    for i, chunk in enumerate(chunks, start=1):
        if progress:
            progress(i / total, desc=f"ASR chunk {i}/{total}")
        out = asr({"array": chunk, "sampling_rate": sr}, return_timestamps=False)
        txt = out.get("text", "").strip()
        if txt and txt.lower() != "nan":
            all_text.append(txt)
    return merge_chunk_texts(all_text)


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Model did not return JSON.")
    return json.loads(match.group(0))


def extract_demographics_from_text(text: str) -> Dict[str, Optional[str]]:
    s = text or ""
    name = None
    age = None
    gender = None

    m = re.search(
        r"my name is\s+([A-Za-z][A-Za-z .'-]{0,80}?)\s+and i am\s+(\d{1,3})\s*years?\s*old",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        name = m.group(1).strip(" .,:;")
        age = m.group(2).strip()
    else:
        m_name = re.search(r"my name is\s+([A-Za-z][A-Za-z .'-]{0,80})", s, flags=re.IGNORECASE)
        if m_name:
            name = m_name.group(1).strip(" .,:;")
        m_age = re.search(r"\bi am\s+(\d{1,3})\s*years?\s*old\b", s, flags=re.IGNORECASE)
        if m_age:
            age = m_age.group(1).strip()

    m_gender = re.search(r"\b(male|female|man|woman)\b", s, flags=re.IGNORECASE)
    if m_gender:
        g = m_gender.group(1).lower()
        if g in {"male", "man"}:
            gender = "Male"
        elif g in {"female", "woman"}:
            gender = "Female"

    return {"patient_name": name, "patient_age": age, "patient_gender": gender}


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    base_url = os.getenv("OPENAI_BASE_URL")
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _openai_temperature_kw(openai_model: str, temperature: float) -> Dict[str, Any]:
    """Many newer OpenAI models only allow the default temperature (omit the param)."""
    m = (openai_model or "").strip().lower().split("/")[-1]
    if re.match(r"^o\d", m):
        return {}
    if m.startswith("gpt-5"):
        return {}
    return {"temperature": temperature}


def _openai_chat_completions_create(client: OpenAI, **kwargs: Any) -> Any:
    """Call chat.completions; retry without temperature if the model only allows the default."""
    try:
        return client.chat.completions.create(**kwargs)
    except APIStatusError as e:
        if e.status_code != 400 or "temperature" not in kwargs:
            raise
        detail = ((getattr(e, "message", None) or "") + json.dumps(getattr(e, "body", "") or {})).lower()
        if "temperature" not in detail or "unsupported" not in detail:
            raise
        retry_kw = {k: v for k, v in kwargs.items() if k != "temperature"}
        return client.chat.completions.create(**retry_kw)


def _hinglish_to_english_prompts(transcript_text: str) -> Tuple[str, str]:
    system = (
        "You are a medical transcription assistant. Convert the input Hindi/Hinglish doctor-patient "
        "transcript into clear English conversation format with speaker labels. "
        "Resolve noisy or ambiguous phrases using context (e.g. sound-alikes for medical terms); "
        "do not add diagnoses, symptoms, or facts not supported by the transcript."
    )
    user = (
        "Rewrite in this format only:\n"
        "Doctor: ...\nPatient: ...\n\n"
        "Preserve medical facts. Do not add new facts.\n\n"
        f"Input transcript:\n{transcript_text}"
    )
    return system, user


def _to_english_anthropic(system: str, user: str, model: str) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "HINGLISH_TO_ENGLISH_BACKEND=anthropic requires the anthropic package. "
            "Install with: pip install anthropic"
        ) from e
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set (HINGLISH_TO_ENGLISH_BACKEND=anthropic).")
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=8192,
        temperature=0.1,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts: List[str] = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts).strip()


def to_english_conversation(transcript_text: str, openai_model: str) -> str:
    system, user = _hinglish_to_english_prompts(transcript_text)
    if DEFAULT_HINGLISH_TO_ENGLISH_BACKEND == "anthropic":
        return _to_english_anthropic(system, user, DEFAULT_ANTHROPIC_TRANSLATION_MODEL)
    client = get_openai_client()
    resp = _openai_chat_completions_create(
        client,
        model=openai_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **_openai_temperature_kw(openai_model, 0.1),
    )
    return (resp.choices[0].message.content or "").strip()


def _norm_list(v: Any) -> List[str]:
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    return [str(x).strip() for x in v if str(x).strip()]


def _textbook_rag_enabled() -> bool:
    return (os.getenv("TEXTBOOK_RAG_ENABLED", "1") or "").strip().lower() not in ("0", "false", "no")


def _textbook_paths() -> Tuple[str, str]:
    default_pdf = os.path.join(
        BASE_DIR,
        "An Insiders Guide to Clinical Medicine (Archith Boloor) .pdf",
    )
    pdf = os.getenv("TEXTBOOK_RAG_PDF", default_pdf)
    idx = os.getenv(
        "TEXTBOOK_RAG_INDEX_DIR",
        os.path.join(os.path.dirname(BASE_DIR), "agent", "textbook_rag"),
    )
    return pdf, idx


def _retrieve_textbook_context_block(client: OpenAI, transcript_text: str) -> str:
    """Retrieved textbook excerpts for a single discharge-summary call (no extra LLM pass)."""
    if not _textbook_rag_enabled() or textbook_rag is None:
        return ""
    pdf, idx_dir = _textbook_paths()
    if not os.path.isfile(pdf):
        return ""
    try:
        os.makedirs(idx_dir, exist_ok=True)
        if not textbook_rag.ensure_index(client, pdf, idx_dir):
            return ""
        k = int((os.getenv("TEXTBOOK_RAG_TOP_K") or "10").strip() or "10")
        max_ctx = int((os.getenv("TEXTBOOK_RAG_CONTEXT_CHARS") or "28000").strip() or "28000")
        chunks = textbook_rag.retrieve(client, transcript_text, idx_dir, k=k, extra_query="")
        if not chunks:
            return ""
        return textbook_rag.format_chunks_for_prompt(chunks, max_chars=max_ctx)
    except Exception:
        return ""


def _normalize_print_summary(ps: Dict[str, Any], inferred: Dict[str, Optional[str]]) -> Dict[str, Any]:
    return {
        "department": ps.get("department"),
        "patient_name": ps.get("patient_name") or inferred.get("patient_name"),
        "patient_age": str(ps.get("patient_age")).strip() if ps.get("patient_age") is not None else inferred.get("patient_age"),
        "patient_gender": ps.get("patient_gender") or inferred.get("patient_gender"),
        "cr_no": ps.get("cr_no"),
        "admission_no": ps.get("admission_no"),
        "department_unit": ps.get("department_unit"),
        "consultant_faculty": ps.get("consultant_faculty"),
        "ward": ps.get("ward"),
        "room_bed": ps.get("room_bed"),
        "patient_category": ps.get("patient_category"),
        "discharge_type": ps.get("discharge_type"),
        "father_spouse_name": ps.get("father_spouse_name"),
        "occupation": ps.get("occupation"),
        "contact_no": ps.get("contact_no"),
        "address": ps.get("address"),
        "state_country": ps.get("state_country"),
        "doa": ps.get("doa"),
        "date_of_discharge": ps.get("date_of_discharge"),
        "print_report_datetime": ps.get("print_report_datetime"),
        "final_diagnosis": _norm_list(ps.get("final_diagnosis")),
        "differential_diagnosis": _norm_list(ps.get("differential_diagnosis")),
        "chief_complaints": _norm_list(ps.get("chief_complaints")),
        "history_of_present_illness": ps.get("history_of_present_illness"),
        "past_history": ps.get("past_history"),
        "personal_history": ps.get("personal_history"),
        "drug_history": ps.get("drug_history"),
        "family_history": ps.get("family_history"),
        "general_examination": ps.get("general_examination"),
        "systemic_examination": ps.get("systemic_examination"),
        "hospital_course": ps.get("hospital_course"),
        "advice_on_discharge": _norm_list(ps.get("advice_on_discharge")),
    }


def _apply_diagnosis_grounding_rules(final_diagnosis: List[str], transcript_text: str) -> List[str]:
    if not final_diagnosis:
        return []

    t = (transcript_text or "").lower()
    explicit_ich = bool(re.search(r"\bintracerebral hemorrhage\b|\bich\b", t, flags=re.IGNORECASE))
    has_sdh_term = bool(re.search(r"\bsubdural\b|\bsdh\b", t, flags=re.IGNORECASE))
    has_old_new_blood = bool(
        re.search(
            r"(old\s+and\s+new\s+blood\s+(?:accumulation|collection))|(old blood.*new blood)|(new blood.*old blood)",
            t,
            flags=re.IGNORECASE,
        )
    )
    has_burr_hole = bool(re.search(r"\bburr\s*hole\b", t, flags=re.IGNORECASE))
    has_left_ul_ll_weakness = bool(
        re.search(
            r"left\s+(?:hand|arm|upper limb).{0,80}left\s+(?:leg|lower limb|foot).{0,120}(?:weak|weakness)",
            t,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"(?:weak|weakness).{0,120}left\s+(?:hand|arm|upper limb).{0,80}left\s+(?:leg|lower limb|foot)",
            t,
            flags=re.IGNORECASE,
        )
    )
    has_right_side_collection = bool(
        has_old_new_blood and re.search(r"\bright\s+side\b", t, flags=re.IGNORECASE)
    )

    normalized: List[str] = []
    for item in final_diagnosis:
        d = (item or "").strip()
        if not d:
            continue
        low = d.lower()
        old_new_collection_phrase = bool(
            re.search(
                r"(old\s+and\s+new\s+blood\s+(?:accumulation|collection))|(old blood.*new blood)|(new blood.*old blood)",
                low,
                flags=re.IGNORECASE,
            )
        )
        if old_new_collection_phrase and (has_sdh_term or has_burr_hole):
            if has_right_side_collection:
                d = "Acute on chronic right subdural hematoma"
            else:
                d = "Acute on chronic subdural hematoma"
            if has_left_ul_ll_weakness:
                d = f"{d} with left upper and lower limb weakness"

        if "intracerebral hemorrhage" in low and not explicit_ich:
            if has_sdh_term or (has_old_new_blood and has_burr_hole):
                if has_right_side_collection:
                    d = "Acute on chronic right subdural hematoma"
                else:
                    d = "Acute on chronic subdural hematoma"
            elif has_old_new_blood:
                d = "Acute on chronic right-sided blood collection"
            if has_left_ul_ll_weakness:
                d = f"{d} with left upper and lower limb weakness"
        normalized.append(d)
    return _dedupe(normalized)


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _split_candidate_items(text: str) -> List[str]:
    cleaned = re.sub(r"\b(and|plus)\b", ",", (text or ""), flags=re.IGNORECASE)
    parts = re.split(r"[;,/]|(?:\.\s+)", cleaned)
    items: List[str] = []
    for part in parts:
        chunk = re.sub(r"\s+", " ", part).strip(" .,:;-")
        if not chunk:
            continue
        if len(chunk) > 120:
            continue
        items.append(chunk)
    return items


def _speaker_lines(text: str, speaker_prefix: str) -> List[str]:
    items: List[str] = []
    prefix = f"{speaker_prefix}:"
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith(prefix.lower()):
            items.append(line[len(prefix) :].strip())
    return items


def _extract_with_patterns(lines: List[str], patterns: List[str]) -> List[str]:
    values: List[str] = []
    for line in lines:
        for pat in patterns:
            m = re.search(pat, line, flags=re.IGNORECASE)
            if not m:
                continue
            values.extend(_split_candidate_items(m.group(1)))
    return _dedupe(values)


def _extract_candidate_phrases(lines: List[str], patterns: List[str]) -> List[str]:
    values: List[str] = []
    for line in lines:
        for pat in patterns:
            m = re.search(pat, line, flags=re.IGNORECASE)
            if not m:
                continue
            phrase = re.sub(r"\s+", " ", m.group(1)).strip(" .,:;-")
            if phrase:
                values.append(phrase)
    return _dedupe(values)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    out: List[str] = []
    for part in parts:
        sentence = re.sub(r"\s+", " ", part).strip(" ")
        if sentence:
            out.append(sentence)
    return out


def _is_question_like(text: str) -> bool:
    s = (text or "").strip().lower()
    if not s:
        return False
    if "?" in s:
        return True
    return bool(
        re.match(
            r"^(do|did|does|is|are|was|were|can|could|should|would|will|what|when|where|why|how|have|has|had)\b",
            s,
            flags=re.IGNORECASE,
        )
    )


def _get_affirmative_doctor_sentences(doctor_lines: List[str]) -> List[str]:
    out: List[str] = []
    for line in doctor_lines:
        for sentence in _split_sentences(line):
            if not _is_question_like(sentence):
                out.append(sentence.strip())
    return out


def _clean_diagnosis_phrases(items: List[str]) -> List[str]:
    blocked = re.compile(
        r"^(headache|vomiting|fever|seizures?|unconsciousness|difficulty speaking|itching|black stools?)\b",
        flags=re.IGNORECASE,
    )
    cleaned: List[str] = []
    for item in items:
        s = re.sub(r"\s+", " ", (item or "")).strip(" .,:;-")
        if not s:
            continue
        if _is_question_like(s):
            continue
        if blocked.search(s):
            continue
        if re.search(r"\b(been taking|khaini|chewing tobacco)\b", s, flags=re.IGNORECASE):
            continue
        cleaned.append(s)
    return _dedupe(cleaned)


def _extract_biobert_diagnosis(transcript_text: str, doctor_lines: List[str]) -> List[str]:
    affirmative = _get_affirmative_doctor_sentences(doctor_lines)
    candidates = _extract_candidate_phrases(
        affirmative,
        [
            r"^(?:final diagnosis|diagnosis|impression)\s*(?:is|:)?\s*(.+)$",
            r"^(?:the\s+)?(?:ct|mri|scan)\s*(?:scan\s*)?(?:shows|showed|reveals|revealed|suggests|suggested)\s+(.+)$",
            r"^(?:findings?\s*(?:are|show|showed|suggest|suggested)\s*)(.+)$",
            r"^(?:you have|this is|consistent with)\s+(.+)$",
        ],
    )
    if not candidates:
        m = re.search(
            r"(old\s+and\s+new\s+blood\s+(?:accumulation|collection).{0,120})",
            transcript_text or "",
            flags=re.IGNORECASE,
        )
        if m:
            candidates.append(m.group(1).strip(" .,:;-"))

    cleaned = _clean_diagnosis_phrases(candidates)
    return _apply_diagnosis_grounding_rules(cleaned, transcript_text)


def _extract_biobert_chief_complaints(patient_lines: List[str]) -> List[str]:
    complaints: List[str] = []
    for line in patient_lines:
        s = (line or "").strip()
        if not s or _is_question_like(s):
            continue

        low = s.lower()
        if re.search(r"\b(eyes?.{0,24}yellow|urine.{0,24}yellow|yellow.{0,24}(eyes?|urine)|jaundice)\b", low):
            complaints.append("yellow discoloration of eyes/urine")
        if re.search(
            r"(left\s+(?:hand|arm|upper limb).{0,80}left\s+(?:leg|lower limb|foot).{0,120}(?:weak|weakness))|(left\s+side.{0,40}(?:weak|weakness))",
            low,
        ):
            complaints.append("left upper and lower limb weakness")
        if re.search(r"\bheadache\b", low) and not re.search(r"\b(no|without|denies?)\s+headache\b", low):
            complaints.append("headache")
        if re.search(r"\bvomit(?:ing)?\b", low) and not re.search(
            r"\b(no|without|denies?)\s+vomit(?:ing)?\b", low
        ):
            complaints.append("vomiting")
        if re.search(r"\bfever\b", low) and not re.search(r"\b(no|without|denies?)\s+fever\b", low):
            complaints.append("fever")
        if re.search(r"\bseizures?\b", low) and not re.search(
            r"\b(no|without|denies?)\s+seizures?\b", low
        ):
            complaints.append("seizures")
        if re.search(r"\bunconscious(?:ness)?\b", low) and not re.search(
            r"\b(no|without|denies?)\s+unconscious(?:ness)?\b",
            low,
        ):
            complaints.append("loss of consciousness")
    return _dedupe(complaints)


def _extract_biobert_advice(doctor_lines: List[str]) -> List[str]:
    advice: List[str] = []
    for line in doctor_lines:
        for sentence in _split_sentences(line):
            s = sentence.strip(" ")
            if not s or _is_question_like(s):
                continue
            if not re.search(
                r"\b(avoid|continue|start|take|follow up|do not stop|if needed|if you experience|danger signs?|emergency|as needed)\b",
                s,
                flags=re.IGNORECASE,
            ):
                continue
            s = re.sub(r"^from now on,\s*", "", s, flags=re.IGNORECASE)
            advice.append(s.strip(" ."))
    return _dedupe(advice)


def _extract_biobert_hospital_course(doctor_lines: List[str]) -> Optional[str]:
    course: List[str] = []
    for sentence in _get_affirmative_doctor_sentences(doctor_lines):
        if re.search(
            r"\b(avoid|continue|start|take|follow up|if you experience|danger signs?|emergency|as needed)\b",
            sentence,
            flags=re.IGNORECASE,
        ):
            continue
        if re.search(
            r"\b(ct|mri|scan|blood accumulation|burr hole|surgery|icu|observation|admit|admitted|treated|stable|discharge)\b",
            sentence,
            flags=re.IGNORECASE,
        ):
            course.append(sentence.strip())
    return " ".join(course[:4]).strip() or None


def _slug_medical_domain(key: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", (key or "").strip().lower()).strip("_")
    return s or "general_medicine"


def list_medical_domains() -> List[str]:
    if not os.path.isdir(DOMAIN_DATA_DIR):
        return ["general_medicine"]
    keys = [
        n[:-5]
        for n in sorted(os.listdir(DOMAIN_DATA_DIR))
        if n.endswith(".json") and n != "domains.json"
    ]
    return keys or ["general_medicine"]


def load_medical_domain_config(domain_key: str) -> Dict[str, Any]:
    slug = _slug_medical_domain(domain_key)
    path = os.path.join(DOMAIN_DATA_DIR, f"{slug}.json")
    if not os.path.isfile(path):
        slug = "general_medicine"
        path = os.path.join(DOMAIN_DATA_DIR, f"{slug}.json")
    default_style = (
        "hospital General Medicine printouts (full narrative HPI, numbered diagnoses/complaints/advice, "
        "examination subsections)"
    )
    if not os.path.isfile(path):
        return {
            "key": slug,
            "display_name": "General Medicine",
            "style_descriptor": default_style,
            "department_default": "Department of General Medicine",
            "system_addon": "",
        }
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("key", slug)
    data.setdefault("style_descriptor", default_style)
    data.setdefault("department_default", "Department of General Medicine")
    data.setdefault("system_addon", "")
    data.setdefault("display_name", slug.replace("_", " ").title())
    return data


def build_print_summary_openai(
    transcript_text: str, openai_model: str, medical_domain: str = DEFAULT_MEDICAL_DOMAIN
) -> Dict[str, Any]:
    dc = load_medical_domain_config(medical_domain)
    style = (dc.get("style_descriptor") or "").strip() or (
        "hospital General Medicine printouts (full narrative HPI, numbered diagnoses/complaints/advice, "
        "examination subsections)"
    )
    addon = (dc.get("system_addon") or "").strip()
    dept_hint = (dc.get("department_default") or "").strip()
    client = get_openai_client()
    inferred = extract_demographics_from_text(transcript_text)
    doctor_lines = _speaker_lines(transcript_text, "Doctor")
    diagnosis_candidates = _extract_candidate_phrases(
        doctor_lines or [line.strip() for line in (transcript_text or "").splitlines() if line.strip()],
        [
            r"(?:final diagnosis|diagnosis|impression)\s*(?:is|:)?\s*(.+)$",
            r"(?:ct|mri|scan)\s*(?:shows|showed|suggests|suggested)\s+(.+)$",
            r"(?:you have|this is|consistent with)\s+(.+)$",
        ],
    )
    schema_hint = {
        "print_summary": {
            "department": "string|null",
            "patient_name": "string|null",
            "patient_age": "string|null",
            "patient_gender": "string|null",
            "cr_no": "string|null",
            "admission_no": "string|null",
            "department_unit": "string|null",
            "consultant_faculty": "string|null",
            "ward": "string|null",
            "room_bed": "string|null",
            "patient_category": "string|null",
            "discharge_type": "string|null",
            "father_spouse_name": "string|null",
            "occupation": "string|null",
            "contact_no": "string|null",
            "address": "string|null",
            "state_country": "string|null",
            "doa": "string|null",
            "date_of_discharge": "string|null",
            "print_report_datetime": "string|null",
            "final_diagnosis": ["string"],
            "differential_diagnosis": ["string"],
            "chief_complaints": ["string"],
            "history_of_present_illness": "string|null",
            "past_history": "string|null",
            "personal_history": "string|null",
            "drug_history": "string|null",
            "family_history": "string|null",
            "general_examination": "string|null",
            "systemic_examination": "string|null",
            "hospital_course": "string|null",
            "advice_on_discharge": ["string"],
        }
    }

    system = (
        "You are a clinical documentation assistant producing one structured DISCHARGE SUMMARY in the style of "
        f"{style}. "
        "AUTHORITATIVE FACTS (patient-specific): the transcript only — names, dates, symptoms, exam, investigations, "
        "treatments, advice, identifiers. Never invent CR numbers, admission IDs, ward/bed, labs, imaging results, vitals, or "
        "medications not grounded in the transcript. Use null for missing administrative fields. "
        "When the user message includes a TEXTBOOK EXCERPTS block, weave that material together with sound internal-medicine "
        "knowledge into the SAME standard sections (not a separate appendix): use it to tighten terminology, pathophysiology "
        "framing, and typical discharge phrasing; expand HPI, hospital course, and examination prose to read like sample "
        "discharge summaries, including concise pertinent negatives that are standard for the documented presentation when "
        "the transcript does not affirm the opposite (do not contradict anything stated in the visit). "
        "Do not add a new working diagnosis solely from the textbook; final_diagnosis and differential_diagnosis must still "
        "reflect what the encounter supports. "
        "OUTPUT SHAPE: header fields, FINAL DIAGNOSIS (list items), CHIEF COMPLAINTS (duration in the string when known), "
        "HISTORY OF PRESENT ILLNESS as rich formal narrative; optional past_history, personal_history, drug_history, "
        "family_history; general_examination and systemic_examination (null if nothing to report); "
        "differential_diagnosis 3–4 items only when alternatives/rule-outs appear in the transcript; "
        "hospital_course as cohesive narrative; advice_on_discharge as discrete lines. "
        "DIAGNOSIS STYLE RULES: "
        "1) Preserve specificity as spoken (acuity, laterality, deficits). "
        "2) Prefer '<acuity> <laterality> <condition> with <deficit>' for neuro when supported. "
        "3) Do NOT output 'intracerebral hemorrhage' unless transcript explicitly says ICH/intracerebral hemorrhage. "
        "4) Old+new blood with burr hole context may be phrased 'acute on chronic subdural hematoma' only when supported. "
        "5) Never copy question prompts as diagnosis. "
        "6) No ICD codes unless spoken."
    )
    if addon:
        system = system + "\n\nDOMAIN-SPECIFIC GUIDANCE:\n" + addon
    ref_block = _retrieve_textbook_context_block(client, transcript_text)
    user_parts = [
        "Return STRICT JSON with this shape (same keys only):\n" + json.dumps(schema_hint, ensure_ascii=True),
        f"\nDemographics inferred by regex (use if present): {json.dumps(inferred, ensure_ascii=True)}",
        f"\nDoctor-stated diagnosis candidate phrases (prefer these exact meanings): {json.dumps(diagnosis_candidates, ensure_ascii=True)}",
        "\nPreferred diagnosis phrasing examples (apply only when supported by transcript):\n"
        "- acute on chronic right subdural hematoma with left upper and lower limb weakness\n"
        "- right middle cerebral artery infarct with left hemiparesis\n"
        "- viral febrile illness",
    ]
    if dept_hint:
        user_parts.append(
            f"\nPreferred department header (use when consistent with the transcript; else null): {dept_hint}"
        )
    user_parts.append("\nTranscript/Conversation:\n" + transcript_text)
    if ref_block.strip():
        user_parts.append(
            "\nTEXTBOOK EXCERPTS (retrieved clinical reference — integrate into sections above; do not add a separate "
            '"reference" heading in JSON):\n'
            + ref_block
        )
    user = "".join(user_parts)

    _temp = _openai_temperature_kw(openai_model, 0.1)
    try:
        resp = _openai_chat_completions_create(
            client,
            model=openai_model,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **_temp,
        )
    except Exception:
        resp = _openai_chat_completions_create(
            client,
            model=openai_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **_temp,
        )
    content = (resp.choices[0].message.content or "").strip()
    data = _extract_json(content)
    ps = data.get("print_summary") or {}
    normalized = _normalize_print_summary(ps, inferred)
    normalized["final_diagnosis"] = _apply_diagnosis_grounding_rules(
        normalized.get("final_diagnosis") or [],
        transcript_text,
    )
    return {"print_summary": normalized}


def build_print_summary_biobert(transcript_text: str) -> Dict[str, Any]:
    inferred = extract_demographics_from_text(transcript_text)
    doctor_lines = _speaker_lines(transcript_text, "Doctor")
    patient_lines = _speaker_lines(transcript_text, "Patient")
    final_diagnosis = _extract_biobert_diagnosis(transcript_text, doctor_lines)
    chief_complaints = _extract_biobert_chief_complaints(patient_lines)
    advice_on_discharge = _extract_biobert_advice(doctor_lines)
    history_of_present_illness = " ".join(patient_lines[:3]).strip() or None
    hospital_course = _extract_biobert_hospital_course(doctor_lines)

    ps = {
        "patient_name": inferred.get("patient_name"),
        "patient_age": inferred.get("patient_age"),
        "patient_gender": inferred.get("patient_gender"),
        "contact_no": None,
        "address": None,
        "state_country": None,
        "doa": None,
        "date_of_discharge": None,
        "print_report_datetime": None,
        "final_diagnosis": final_diagnosis,
        "chief_complaints": chief_complaints,
        "history_of_present_illness": history_of_present_illness,
        "hospital_course": hospital_course,
        "advice_on_discharge": advice_on_discharge,
    }
    return {"print_summary": _normalize_print_summary(ps, inferred)}


def build_print_summary(
    transcript_text: str,
    openai_model: str,
    summary_backend: str = DEFAULT_SUMMARY_BACKEND,
    medical_domain: str = DEFAULT_MEDICAL_DOMAIN,
) -> Dict[str, Any]:
    backend = (summary_backend or DEFAULT_SUMMARY_BACKEND).strip().lower()
    if backend not in VALID_SUMMARY_BACKENDS:
        raise ValueError(
            f"Unsupported summary backend '{summary_backend}'. Use one of: {', '.join(VALID_SUMMARY_BACKENDS)}"
        )
    if backend == "openai":
        return build_print_summary_openai(transcript_text, openai_model, medical_domain=medical_domain)
    return build_print_summary_biobert(transcript_text)


def _fmt_list(items: Any) -> str:
    if not items:
        return "- Not specified"
    return "\n".join([f"- {str(i)}" for i in items if str(i).strip()]) or "- Not specified"


def _fmt_numbered(items: Any) -> str:
    xs = _norm_list(items)
    if not xs:
        return "Not specified"
    return "\n".join(f"{i + 1}. {x}" for i, x in enumerate(xs))


def _hdr_line(label: str, value: Any) -> str:
    v = value
    if v is None or (isinstance(v, str) and not v.strip()):
        v = "Not specified"
    return f"{label}: {v}"


def _age_gender_line(summary: Dict[str, Any]) -> str:
    age = summary.get("patient_age")
    age_s = str(age).strip() if age is not None else ""
    gender = (summary.get("patient_gender") or "").strip()
    letter = "—"
    if gender:
        low = gender.lower()
        if low.startswith("m") or low in ("male", "man"):
            letter = "M"
        elif low.startswith("f") or low in ("female", "woman"):
            letter = "F"
        else:
            letter = gender[0].upper()
    if age_s:
        return f"{age_s} Yr/{letter}"
    return "Not specified"


def print_summary_to_markdown(summary: Dict[str, Any]) -> str:
    dept = (summary.get("department") or "Department of General Medicine").strip()
    lines: List[str] = [
        dept,
        "Discharge Summary",
        "",
        _hdr_line("CR No.", summary.get("cr_no")),
        _hdr_line("Name", summary.get("patient_name")),
        _hdr_line("Age/ Gender", _age_gender_line(summary)),
        _hdr_line("Admission No.", summary.get("admission_no")),
        _hdr_line("Department Unit", summary.get("department_unit")),
        _hdr_line("Consultant/ Faculty", summary.get("consultant_faculty")),
        _hdr_line("Ward", summary.get("ward")),
        _hdr_line("Room/ Bed", summary.get("room_bed")),
        _hdr_line("Patient Category", summary.get("patient_category")),
        _hdr_line("Discharge Type", summary.get("discharge_type")),
        _hdr_line("Father/ Spouse Name", summary.get("father_spouse_name")),
        _hdr_line("Occupation", summary.get("occupation")),
        _hdr_line("Contact No.", summary.get("contact_no")),
        _hdr_line("Address", summary.get("address")),
        _hdr_line("State/ Country", summary.get("state_country")),
        _hdr_line("D.O.A.", summary.get("doa")),
        _hdr_line("Date of Discharge", summary.get("date_of_discharge")),
        _hdr_line("D.O. Print Report", summary.get("print_report_datetime")),
        "",
        "FINAL DIAGNOSIS:",
        _fmt_numbered(summary.get("final_diagnosis")),
    ]
    ddx = _norm_list(summary.get("differential_diagnosis"))
    if ddx:
        lines.extend(["", "DIFFERENTIAL DIAGNOSIS:", _fmt_numbered(ddx)])
    lines.extend(
        [
            "",
            "CHIEF COMPLAINTS:",
            _fmt_numbered(summary.get("chief_complaints")),
            "",
            "HISTORY OF PRESENT ILLNESS:",
            (summary.get("history_of_present_illness") or "Not specified").strip(),
        ]
    )
    for title, key in (
        ("Past History", "past_history"),
        ("Personal History", "personal_history"),
        ("Drug History", "drug_history"),
        ("Family History", "family_history"),
        ("General Examination", "general_examination"),
        ("Systemic Examination", "systemic_examination"),
    ):
        block = (summary.get(key) or "").strip()
        if block:
            lines.extend(["", title, block])
    lines.extend(
        [
            "",
            "HOSPITAL COURSE:",
            (summary.get("hospital_course") or "Not specified").strip(),
            "",
            "ADVICE ON DISCHARGE:",
            _fmt_numbered(summary.get("advice_on_discharge")),
        ]
    )
    return "\n".join(lines) + "\n"


def run_step1_ui(audio_path: str, progress=None) -> Tuple[str, str, str]:
    if not audio_path:
        return "", "", "Upload audio first."

    try:
        hinglish = transcribe_hinglish(audio_path, progress=progress)
    except Exception as e:
        return "", "", f"ASR error: {e}"

    if not hinglish:
        return "", "", "No speech detected."

    try:
        english = to_english_conversation(hinglish, DEFAULT_OPENAI_MODEL)
    except Exception as e:
        return hinglish, "", f"English conversion ({DEFAULT_HINGLISH_TO_ENGLISH_BACKEND}) failed: {e}"

    return hinglish, english, "Step 1 complete. Review transcript, then click Generate Final Report."


def run_step2_ui(english_transcript: str, summary_backend: str) -> Tuple[str, str, str]:
    text = (english_transcript or "").strip()
    if not text:
        return "English Transcript is empty. Run step 1 first.", "", ""
    backend = (summary_backend or DEFAULT_SUMMARY_BACKEND).strip().lower()
    try:
        structured = build_print_summary(
            text,
            DEFAULT_OPENAI_MODEL,
            summary_backend=backend,
        )
    except Exception as e:
        return f"{backend} (report generation) failed: {e}", "", ""

    report = print_summary_to_markdown(structured.get("print_summary", {}))
    raw_json = json.dumps(structured, ensure_ascii=True, indent=2)
    return f"Step 2 complete. Final report generated with '{backend}'.", report, raw_json


def build_gradio_app() -> gr.Blocks:
    with gr.Blocks(title="Doctor-Patient Audio Workflow") as demo:
        gr.Markdown(
            "# Doctor-Patient Workflow\n"
            "Step 1: Upload audio and run Whisper -> Hinglish + English conversation.\n"
            "Step 2: Click Generate Final Report."
        )
        audio_in = gr.Audio(type="filepath", label="Audio Input (Doctor + Patient)")
        run_step1_btn = gr.Button("1) Run ASR + English Conversation", variant="primary")

        hinglish_out = gr.Textbox(label="Hinglish Transcript", lines=10)
        english_out = gr.Textbox(label="English Transcript (Conversation)", lines=16)
        summary_backend_in = gr.Dropdown(
            choices=list(VALID_SUMMARY_BACKENDS),
            value=DEFAULT_SUMMARY_BACKEND,
            label="Summary Backend",
        )
        run_step2_btn = gr.Button("2) Generate Final Report", variant="secondary")

        status_out = gr.Textbox(label="Status", lines=2)
        clinic_out = gr.Textbox(label="Final Report (Editable)", lines=28, interactive=True)
        raw_json_out = gr.Code(label="Raw JSON Output", language="json")

        run_step1_btn.click(
            fn=run_step1_ui,
            inputs=[audio_in],
            outputs=[hinglish_out, english_out, status_out],
        )

        run_step2_btn.click(
            fn=run_step2_ui,
            inputs=[english_out, summary_backend_in],
            outputs=[status_out, clinic_out, raw_json_out],
        )
    return demo


def run_cli(args: argparse.Namespace) -> None:
    if not args.audio:
        raise ValueError("--audio is required in cli mode")
    hinglish = transcribe_hinglish(args.audio, progress=None)
    english = to_english_conversation(hinglish, args.openai_model or DEFAULT_OPENAI_MODEL)
    structured = build_print_summary(
        english,
        args.openai_model or DEFAULT_OPENAI_MODEL,
        summary_backend=args.summary_backend or DEFAULT_SUMMARY_BACKEND,
        medical_domain=args.medical_domain or DEFAULT_MEDICAL_DOMAIN,
    )
    clinic_md = print_summary_to_markdown(structured.get("print_summary", {}))
    raw_json = json.dumps(structured, ensure_ascii=True, indent=2)
    print("\n=== HINGLISH ===\n", hinglish)
    print("\n=== ENGLISH ===\n", english)
    print("\n=== CLINIC NOTE ===\n", clinic_md)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(raw_json)
        print(f"\nSaved JSON: {args.output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audio -> Hinglish ASR -> English transcript -> OpenAI/BioBERT print summary workflow"
    )
    parser.add_argument("--mode", choices=["gradio", "cli"], default="gradio")
    parser.add_argument("--audio", default="", help="Path to audio file for CLI mode")
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument(
        "--summary-backend",
        choices=list(VALID_SUMMARY_BACKENDS),
        default=DEFAULT_SUMMARY_BACKEND,
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--medical-domain",
        default=DEFAULT_MEDICAL_DOMAIN,
        help="Template under data/<name>.json (default: general_medicine).",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "cli":
        run_cli(args)
        return
    app = build_gradio_app()
    app.launch(server_name="0.0.0.0", server_port=args.port, share=True, show_error=True)


if __name__ == "__main__":
    main()
