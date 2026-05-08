# echoai

## Doctor-Patient Audio Workflow

Workflow implemented in:
- `clinic_workflow.py`

Pipeline:
1. Input doctor+patient audio
2. `whisper-hindi2hinglish-prime` ASR in 30s chunks
3. Hinglish transcript
4. OpenAI model converts Hinglish transcript to English conversation
5. Structured clinic summary via selectable backend: `openai` or `biobert`

## FastAPI (deployment)

GPU pod (single worker keeps one ASR model loaded):

```bash
cd /home/surya/anuj/script2video/whisper
pip install -r requirements-clinic.txt   # includes fastapi + uvicorn + pymupdf optional extras
export OPENAI_API_KEY="YOUR_KEY"
uvicorn app:app --host 0.0.0.0 --port 8080 --workers 1
```

Interactive OpenAPI: **`http://<host>:8080/docs`** (try requests and schemas in the browser).

### HTTP API — curl examples and sample responses

Use `BASE=http://127.0.0.1:8080` locally; in production replace with your pod URL or ingress.

**`GET /health`**

```bash
curl -s "${BASE:-http://127.0.0.1:8080}/health"
```

Sample response:

```json
{"status": "ok"}
```

**`GET /v1/domains`**

```bash
curl -s "${BASE:-http://127.0.0.1:8080}/v1/domains"
```

Sample response:

```json
{
  "default": "general_medicine",
  "domains": ["general_medicine", "radiology"]
}
```

**`POST /v1/transcribe`** — `multipart/form-data`, form field **`audio`** (file)

```bash
curl -s -F "audio=@/path/to/clinic_audio.wav" \
  "${BASE:-http://127.0.0.1:8080}/v1/transcribe"
```

Sample response (strings shortened with `…`; real responses are full length):

```json
{
  "hinglish_transcript": "Haan, aaie baithie. …",
  "english_transcript": "Doctor: Yes, come in and sit down. …\n\nPatient: Namaste, doctor. …",
  "english_conversation": "Doctor: Yes, come in and sit down. …\n\nPatient: Namaste, doctor. …",
  "openai_model": "gpt-5.5",
  "hinglish_to_english_backend": "openai"
}
```

`english_transcript` and `english_conversation` carry the same English text.

**`POST /v1/clinical-summary`** — `Content-Type: application/json`

Body fields:

| Field | Notes |
|-------|--------|
| `english_transcript` | String **or** JSON object/array. Required unless `english_conversation` is sent. |
| `english_conversation` | Plain-string alias for the transcript; ignored if `english_transcript` is set. |
| `medical_domain` | Template key under `data/` (default `general_medicine`). |
| `summary_backend` | `openai` or `biobert` (optional). |
| `openai_model` | Override model for this call (optional). |

Minimal inline example:

```bash
curl -s "${BASE:-http://127.0.0.1:8080}/v1/clinical-summary" \
  -H "Content-Type: application/json" \
  -d '{"english_transcript":"Doctor: Hello.\n\nPatient: Cough for 3 weeks.","medical_domain":"general_medicine"}'
```

From a JSON file on disk:

```bash
curl -s "${BASE:-http://127.0.0.1:8080}/v1/clinical-summary" \
  -H "Content-Type: application/json" \
  -d @clinical_payload_direct.json
```

Override `medical_domain` without editing the file:

```bash
jq '. + {medical_domain: "radiology"}' clinical_payload_direct.json \
  | curl -s "${BASE:-http://127.0.0.1:8080}/v1/clinical-summary" \
    -H "Content-Type: application/json" -d @-
```

Sample response (fields depend on transcript and model; example shape only):

```json
{
  "clinical_summary": {
    "department": "Department of General Medicine",
    "patient_name": "Anish",
    "patient_age": "25",
    "patient_gender": "Male",
    "final_diagnosis": ["…"],
    "chief_complaints": ["…"],
    "history_of_present_illness": "…",
    "hospital_course": "…",
    "advice_on_discharge": ["…"]
  },
  "medical_domain": "general_medicine",
  "summary_backend": "openai"
}
```

Integrators should build the JSON body in application code (HTTP client `POST` with JSON); shell `jq` is optional.

Pipe transcribe → summary without renaming keys:

```bash
curl -s -F "audio=@/path/to/audio.wav" http://127.0.0.1:8080/v1/transcribe \
  | curl -s http://127.0.0.1:8080/v1/clinical-summary -H "Content-Type: application/json" -d @- | jq .
```

Optional overrides on the pipe:

```bash
curl -s -F "audio=@/path/to/audio.wav" http://127.0.0.1:8080/v1/transcribe \
  | jq '. + {medical_domain: "radiology", summary_backend: "openai"}' \
  | curl -s http://127.0.0.1:8080/v1/clinical-summary -H "Content-Type: application/json" -d @- | jq .
```

CORS: set `CORS_ORIGINS` to `*` or comma-separated origins (see `.env.example`).

**Concurrency:** `/v1/transcribe` and `/v1/clinical-summary` use plain **`def`** handlers so FastAPI runs them in a **thread pool** — long Whisper/OpenAI work does **not** block the asyncio event loop. Another client can call `/v1/clinical-summary` while audio is processing (subject to CPU, thread pool size, and OpenAI rate limits). Multiple **simultaneous audio** uploads still **share one GPU** for ASR, so they may wait on each other there.

Add a specialty by copying `data/general_medicine.json` to `data/<name>.json` and listing it in `data/domains.json`.

## RunPod (Docker, ~6GB+ VRAM)

Goal: one **GPU Pod** you can **Stop** and **Start** anytime; models and Hugging Face cache survive on a **Network Volume** so restarts are fast and stable.

### Why a Network Volume

RunPod container disk is **ephemeral** unless you attach storage. Put Whisper weights and caches on a **Network Volume** (e.g. mounted at `/workspace`) so after **Stop → Start** you do not re-download multi‑GB models.

Suggested layout on the volume:

- `/workspace/models/<your-whisper-folder>/` — full fine-tuned Whisper directory (weights + tokenizer + config), same as local `ASR_MODEL_DIR`
- `/workspace/.cache/huggingface/` — transformers cache (set by `HF_HOME` / `TRANSFORMERS_CACHE` in the image)

### Build and push the image

From this repo directory (where `Dockerfile` lives):

```bash
docker build -t YOUR_DOCKERHUB_USER/echoai-whisper:latest .
docker push YOUR_DOCKERHUB_USER/echoai-whisper:latest
```

The base image is `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`. If RunPod’s GPU driver needs a different CUDA, change the `FROM` line in `Dockerfile` to a matching [PyTorch Docker tag](https://hub.docker.com/r/pytorch/pytorch/tags).

### Create the Pod on RunPod

1. **Template / Deploy**: Custom container → your pushed image `YOUR_DOCKERHUB_USER/echoai-whisper:latest`.
2. **GPU**: Pick a template with **at least ~6GB VRAM** (more headroom if the model is large or you use larger batches).
3. **Container disk**: Small is fine if models live on the volume.
4. **Network Volume**: Create or attach a volume; mount it at **`/workspace`** (matches default `ASR_MODEL_DIR` and caches in the Dockerfile).
5. **Expose HTTP**: Map container port **`8080`** (or set env `PORT` and map that port in RunPod; the baked-in `HEALTHCHECK` assumes **8080** unless you change the Dockerfile).
6. **Environment variables** (minimum):

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required for English + clinical summary (OpenAI backend). |
| `ASR_MODEL_DIR` | Path **inside** the pod to Whisper weights (e.g. `/workspace/models/whisper-hindi2hinglish-prime`). |
| `OPENAI_MODEL` | Optional; default in code is `gpt-5.5`. |
| `CORS_ORIGINS` | Optional; e.g. `*` or your front-end origin. |

Optional: `ASR_FALLBACK_MODEL_DIRS`, `SUMMARY_BACKEND`, `TEXTBOOK_RAG_ENABLED=0` on RunPod if you do not ship a textbook PDF.

7. **Start** the pod. First request to `/v1/transcribe` loads the ASR pipeline (can take **1–3+ minutes** on cold start); `Dockerfile` `HEALTHCHECK` allows **180s** start period before marking unhealthy.

### Pause and restart

- **Stop** the pod in the RunPod UI: sends **SIGTERM**; `uvicorn` exits. No code change required for graceful shutdown of in-flight requests (clients may see disconnects).
- **Start** again: same image + same volume → **`ASR_MODEL_DIR`** and HF cache still on disk → **much faster** second boot than a fresh container.
- **Do not** rely on the container layer alone for secrets; keep **`OPENAI_API_KEY`** in RunPod env or their secrets UI.

### Stability checklist

- Keep **`uvicorn --workers 1`** (default in `Dockerfile`) so only **one** process loads the GPU model.
- Use **`GET /health`** for RunPod / load balancer health checks.
- Prefer **one concurrent heavy transcribe** per GPU; scale **pods** or **GPUs** for more parallel ASR.

### Local smoke test of the image

```bash
docker run --rm -it --gpus all \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e ASR_MODEL_DIR=/workspace/models/your-model \
  -v /path/on/host/models:/workspace/models:ro \
  -p 8080:8080 \
  YOUR_DOCKERHUB_USER/echoai-whisper:latest
```

## Setup

Use the existing model env:

```bash
cd /home/surya/anuj/script2video/whisper-hindi2hinglish-prime
source .venv/bin/activate
pip install openai gradio librosa python-dotenv
```

Optional BioBERT code checkout:

```bash
cd /home/surya/anuj/script2video/whisper
git clone https://github.com/dmis-lab/biobert.git
```

Set API key:

```bash
export OPENAI_API_KEY="YOUR_KEY"
# optional:
# export OPENAI_MODEL="gpt-4.1-mini"
# export OPENAI_BASE_URL="https://..."
```

Or copy and edit:

```bash
cd /home/surya/anuj/script2video/whisper
cp .env.example .env
```

All workflow config/temp files are kept inside this folder:
- `/home/surya/anuj/script2video/whisper/.env`
- `/home/surya/anuj/script2video/whisper/.tmp/`

## Run (Gradio)

```bash
cd /home/surya/anuj/script2video/whisper
python clinic_workflow.py --mode gradio --port 7865
```

## Run (CLI)

```bash
cd /home/surya/anuj/script2video/whisper
python clinic_workflow.py \
  --mode cli \
  --audio /path/to/input.wav \
  --openai-model "gpt-4.1-mini" \
  --summary-backend "openai" \
  --output-json /tmp/clinic_note_openai.json
```

Or use the BioBERT-style local extractor for `build_print_summary`:

```bash
cd /home/surya/anuj/script2video/whisper
python clinic_workflow.py \
  --mode cli \
  --audio /path/to/input.wav \
  --summary-backend "biobert" \
  --output-json /tmp/clinic_note.json
```

## Notes

- This creates a **draft** clinic summary; clinician review is required.
- If OpenAI step fails, Hinglish ASR output is still returned.
- `--summary-backend biobert` applies to `build_print_summary`; English conversation conversion still uses OpenAI.
# echoai
# echoai
