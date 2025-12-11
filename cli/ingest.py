import argparse
import asyncio
import os
import sys

# Ensure /app is on PYTHONPATH so we can import rag.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from rag import ingest_directory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="/app/docs", help="Docs directory (default: /app/docs)")
    args = parser.parse_args()

    if not os.path.isdir(args.docs):
        raise SystemExit(f"Docs dir not found: {args.docs}")

    res = asyncio.run(ingest_directory(args.docs))
    print(f"files_processed={res.files_processed} points_upserted={res.points_upserted}")

if __name__ == "__main__":
    main()
