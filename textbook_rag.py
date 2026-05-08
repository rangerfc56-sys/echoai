"""
Lightweight RAG over a clinical textbook PDF: chunk, embed (OpenAI), cosine retrieval.
Index cache lives under whisper/agent/textbook_rag/ (see AGENTS.md).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(BASE_DIR)


def _load_local_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        env_path = os.path.join(BASE_DIR, ".env")
        if os.path.isfile(env_path):
            load_dotenv(env_path, override=False)
    except Exception:
        pass
DEFAULT_INDEX_DIR = os.path.join(_REPO_ROOT, "agent", "textbook_rag")
DEFAULT_TEXTBOOK_PDF = os.path.join(
    BASE_DIR,
    "An Insiders Guide to Clinical Medicine (Archith Boloor) .pdf",
)
CHUNK_CHARS = int(os.getenv("TEXTBOOK_RAG_CHUNK_CHARS", "1400"))
CHUNK_OVERLAP = int(os.getenv("TEXTBOOK_RAG_CHUNK_OVERLAP", "200"))
EMBED_MODEL = os.getenv("TEXTBOOK_EMBEDDING_MODEL", "text-embedding-3-small")
EMBED_BATCH = int(os.getenv("TEXTBOOK_EMBED_BATCH", "64"))


def _index_paths(index_dir: str) -> Tuple[str, str]:
    meta_path = os.path.join(index_dir, "meta.json")
    data_path = os.path.join(index_dir, "index.npz")
    return meta_path, data_path


def _pdf_text_by_page(pdf_path: str) -> List[str]:
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    pages: List[str] = []
    for i in range(len(doc)):
        t = (doc[i].get_text() or "").strip()
        pages.append(t)
    doc.close()
    return pages


def _chunk_pages(pages: List[str]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for page_idx, page_text in enumerate(pages):
        if not page_text:
            continue
        start = 0
        n = len(page_text)
        while start < n:
            end = min(start + CHUNK_CHARS, n)
            piece = page_text[start:end].strip()
            if len(piece) > 80:
                chunks.append({"text": piece, "page": page_idx + 1})
            if end >= n:
                break
            start = max(0, end - CHUNK_OVERLAP)
    return chunks


def _normalize_vec(v: List[float]) -> List[float]:
    s = math.sqrt(sum(x * x for x in v))
    if s <= 0:
        return v
    return [x / s for x in v]


def build_index(pdf_path: str, index_dir: str, client: Any) -> None:
    os.makedirs(index_dir, exist_ok=True)
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"Textbook PDF not found: {pdf_path}")
    pages = _pdf_text_by_page(pdf_path)
    raw_chunks = _chunk_pages(pages)
    if not raw_chunks:
        raise RuntimeError("No text extracted from PDF; check file or pymupdf.")

    embeddings: List[List[float]] = []
    for i in range(0, len(raw_chunks), EMBED_BATCH):
        batch = raw_chunks[i : i + EMBED_BATCH]
        texts = [c["text"][:8000] for c in batch]
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        for item in sorted(resp.data, key=lambda d: d.index):
            embeddings.append(_normalize_vec(list(item.embedding)))

    import numpy as np

    matrix = np.array(embeddings, dtype=np.float32)
    texts_out = [c["text"] for c in raw_chunks]
    pages_out = [c["page"] for c in raw_chunks]
    meta = {
        "pdf_path": pdf_path,
        "pdf_sha256": _file_sha256(pdf_path),
        "embed_model": EMBED_MODEL,
        "n_chunks": len(raw_chunks),
        "chunk_chars": CHUNK_CHARS,
        "overlap": CHUNK_OVERLAP,
    }
    _, data_path = _index_paths(index_dir)
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    np.savez_compressed(data_path, E=matrix, P=np.array(pages_out, dtype=np.int32))
    with open(os.path.join(index_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for t, p in zip(texts_out, pages_out):
            f.write(json.dumps({"text": t, "page": int(p)}, ensure_ascii=True) + "\n")


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_index(index_dir: str) -> Optional[Tuple[Any, List[str], List[int]]]:
    meta_path, data_path = _index_paths(index_dir)
    chunks_path = os.path.join(index_dir, "chunks.jsonl")
    if not (os.path.isfile(meta_path) and os.path.isfile(data_path) and os.path.isfile(chunks_path)):
        return None
    import numpy as np

    z = np.load(data_path)
    texts: List[str] = []
    pages: List[int] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            pages.append(int(row["page"]))
    return z["E"], texts, pages


def _query_text(transcript: str, extra: str = "") -> str:
    t = (transcript or "").strip()
    if len(t) > 12000:
        t = t[:12000] + "\n[...truncated...]"
    x = (extra or "").strip()
    if x:
        t = f"{t}\n\nContext for retrieval:\n{x}"
    return t


def embed_query(client: Any, text: str) -> List[float]:
    text = text.strip()[:8000] or "."
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return _normalize_vec(list(resp.data[0].embedding))


def retrieve(
    client: Any,
    transcript: str,
    index_dir: str,
    k: int = 8,
    extra_query: str = "",
) -> List[Dict[str, Any]]:
    import numpy as np

    loaded = _load_index(index_dir)
    if loaded is None:
        return []
    matrix, texts, pages = loaded
    q = np.array(embed_query(client, _query_text(transcript, extra_query)), dtype=np.float32)

    sims = matrix @ q
    idx = np.argsort(-sims)[:k]
    out: List[Dict[str, Any]] = []
    for i in idx:
        ii = int(i)
        out.append(
            {
                "text": texts[ii],
                "page": pages[ii],
                "score": float(sims[ii]),
            }
        )
    return out


def ensure_index(client: Any, pdf_path: str, index_dir: str) -> bool:
    """Build index on disk if missing or PDF changed. Returns True if index is ready."""
    if not os.path.isfile(pdf_path):
        return False
    meta_path, data_path = _index_paths(index_dir)
    cur_hash = _file_sha256(pdf_path)
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("pdf_sha256") == cur_hash and os.path.isfile(data_path):
                loaded = _load_index(index_dir)
                return loaded is not None
        except Exception:
            pass
    build_index(pdf_path, index_dir, client)
    return _load_index(index_dir) is not None


def format_chunks_for_prompt(chunks: List[Dict[str, Any]], max_chars: int = 24000) -> str:
    parts: List[str] = []
    used = 0
    for i, c in enumerate(chunks):
        block = f"--- REFERENCE {i+1} (textbook p. {c['page']}) ---\n{c['text']}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


if __name__ == "__main__":
    import argparse

    _load_local_dotenv()

    from openai import OpenAI

    p = argparse.ArgumentParser(description="Build textbook embedding index")
    p.add_argument(
        "--build",
        action="store_true",
        help="Build/update the index (default: build when you run this script)",
    )
    p.add_argument("--pdf", default=os.getenv("TEXTBOOK_RAG_PDF", DEFAULT_TEXTBOOK_PDF))
    p.add_argument("--index-dir", default=os.getenv("TEXTBOOK_RAG_INDEX_DIR", DEFAULT_INDEX_DIR))
    args = p.parse_args()
    client = OpenAI()
    ensure_index(client, args.pdf, args.index_dir)
    print("Index ready:", args.index_dir)
