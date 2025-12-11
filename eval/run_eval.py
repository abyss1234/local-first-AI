import json
import argparse
import os
import math
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")

def embed(text: str):
    r = requests.post(f"{OLLAMA_URL}/api/embed", json={"model": EMBED_MODEL, "input": [text]}, timeout=120)
    r.raise_for_status()
    return r.json()["embeddings"][0]

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na*nb + 1e-12)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="/app/eval/questions.jsonl")
    ap.add_argument("--out", default="/app/eval/report.md")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    rows = []
    with open(args.questions, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    precision_sum = 0.0
    recall_sum = 0.0
    relevancy_sum = 0.0
    relevancy_n = 0
    failures = []

    for r in rows:
        q = r["question"]
        expected = set(r.get("expected_files", []))

        url = f"{API_URL}/chat_rag?dry_run={'true' if args.dry_run else 'false'}"
        resp = requests.post(url, json={"message": q, "top_k": args.top_k}, timeout=180)
        if resp.status_code != 200:
            failures.append((r["id"], q, f"HTTP {resp.status_code}: {resp.text[:200]}"))
            continue

        data = resp.json()
        cits = data.get("citations", [])
        got_files = [c.get("file") for c in cits if c.get("file")]

        # context_precision@k：top-k 里命中期望来源的比例
        hit_count = sum(1 for f in got_files if f in expected)
        precision = hit_count / max(1, args.top_k)
        precision_sum += precision

        # context_recall：是否至少命中一个期望来源（也可改成“命中比例”）
        recall = 1.0 if any(f in expected for f in got_files) else 0.0
        recall_sum += recall

        # answer_relevancy：只在非 dry_run 且有 answer 时计算
        ans = data.get("answer", "") or ""
        if (not args.dry_run) and ans.strip():
            qv = embed(q)
            av = embed(ans)
            rel = cosine(qv, av)
            relevancy_sum += rel
            relevancy_n += 1

        # 记录失败样本：recall=0
        if recall == 0.0:
            failures.append((r["id"], q, f"no expected hit. got={got_files} expected={list(expected)}"))

    n = max(1, len(rows))
    report = []
    report.append("# Eval Report\n")
    report.append(f"- samples: {len(rows)}\n")
    report.append(f"- top_k: {args.top_k}\n")
    report.append(f"- dry_run: {args.dry_run}\n\n")
    report.append("## Metrics\n")
    report.append(f"- context_precision@k: {precision_sum/n:.3f}\n")
    report.append(f"- context_recall: {recall_sum/n:.3f}\n")
    if relevancy_n > 0:
        report.append(f"- answer_relevancy (embedding cosine): {relevancy_sum/relevancy_n:.3f}\n")
    else:
        report.append(f"- answer_relevancy: (skipped)\n")

    report.append("\n## Failures (first 10)\n")
    for item in failures[:10]:
        report.append(f"- id={item[0]} q={item[1]} reason={item[2]}\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("".join(report))

    print(f"wrote: {args.out}")

if __name__ == "__main__":
    main()
