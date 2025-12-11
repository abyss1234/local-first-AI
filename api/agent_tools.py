import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, ValidationError

from rag import retrieve

# ---------- Tool schemas (strict) ----------

ToolName = Literal["search_docs", "write_note", "make_todo_from_answer"]

class SearchDocsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=10)

class WriteNoteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(..., min_length=1, max_length=120)
    content: str = Field(..., min_length=1)

class MakeTodoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., min_length=1)

# ---------- Guards / utils ----------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def sanitize_title_to_filename(title: str) -> str:
    # remove path separators and weird chars
    t = title.strip()
    t = t.replace("\\", " ").replace("/", " ")
    t = re.sub(r"\.+", ".", t)
    t = re.sub(r"[^a-zA-Z0-9 _.-]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        t = "note"
    # safe filename
    t = t[:80].strip().replace(" ", "_")
    return t

def safe_join(base_dir: str, filename: str) -> str:
    base_abs = os.path.abspath(base_dir)
    target = os.path.abspath(os.path.join(base_abs, filename))
    if not target.startswith(base_abs + os.sep) and target != base_abs:
        raise ValueError("Blocked path traversal")
    return target

def summarize_for_log(obj: Any, max_chars: int = 800) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    if len(s) > max_chars:
        return s[:max_chars] + "…"
    return s

def append_jsonl(log_path: str, record: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(log_path))
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def build_sources_block(hits_payload: List[Dict[str, Any]]) -> str:
    # Boundary markers reduce indirect prompt injection influence (treat sources as DATA)
    blocks = []
    for h in hits_payload:
        file = h["file"]
        chunk_id = h["chunk_id"]
        text = h["text"]
        blocks.append(f"<<SOURCE file={file} chunk_id={chunk_id}>>\n{text}\n<<END>>")
    return "\n\n".join(blocks)

# ---------- Tools ----------

async def tool_search_docs(query: str, top_k: int = 5) -> Dict[str, Any]:
    hits = await retrieve(query, top_k=top_k)
    out = []
    citations = []
    for h in hits:
        p = h.payload or {}
        file = p.get("file", "unknown")
        chunk_id = int(p.get("chunk_id", -1))
        text = (p.get("text") or "").strip()
        snippet = (p.get("snippet") or text[:240]).replace("\n", " ").strip()

        out.append({"file": file, "chunk_id": chunk_id, "text": text, "snippet": snippet})
        citations.append({"file": file, "chunk_id": chunk_id, "snippet": snippet})

    return {"hits": out, "citations": citations}

def tool_write_note(title: str, content: str, notes_dir: str = "/app/notes") -> Dict[str, Any]:
    ensure_dir(notes_dir)
    fname = sanitize_title_to_filename(title) + ".md"
    path = safe_join(notes_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title.strip()}\n\n")
        f.write(content.strip() + "\n")
    return {"note_path": path}

def tool_make_todo_from_answer(text: str) -> Dict[str, Any]:
    # Deterministic (more controllable than asking the model again)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    items: List[str] = []

    for ln in lines:
        m = re.match(r"^(\-|\*|\d+[\).\]])\s+(.*)$", ln)
        if m:
            items.append(m.group(2).strip())

    if not items:
        # fallback: split by sentences
        parts = re.split(r"[。.!?]\s+|\n+", text.strip())
        items = [p.strip() for p in parts if p.strip()][:8]

    todos = [{"id": i + 1, "task": t} for i, t in enumerate(items)]
    return {"todos": todos}

# ---------- Tool router (allowlist + schema validation) ----------

async def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "search_docs":
        parsed = SearchDocsArgs(**args)
        return await tool_search_docs(parsed.query, parsed.top_k)

    if name == "write_note":
        parsed = WriteNoteArgs(**args)
        return tool_write_note(parsed.title, parsed.content)

    if name == "make_todo_from_answer":
        parsed = MakeTodoArgs(**args)
        return tool_make_todo_from_answer(parsed.text)

    raise ValueError("Tool not allowed")
