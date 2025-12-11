import json
import os
import time
from typing import Any, Dict
from datetime import datetime, timezone
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

def utc_iso():
    return datetime.now(timezone.utc).isoformat()

def append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        err = None
        status = 200
        try:
            resp = await call_next(request)
            status = resp.status_code
            return resp
        except Exception as e:
            err = str(e)
            status = 500
            raise
        finally:
            ms = (time.perf_counter() - start) * 1000.0
            append_jsonl("/app/logs/requests.jsonl", {
                "ts": utc_iso(),
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "latency_ms": round(ms, 2),
                "error": err,
            })
