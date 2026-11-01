"""Shared paths and provenance utilities. Never load target-derived predictors."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIDTERM = ROOT / 'reference' / 'midterm'
RAW = ROOT / 'data' / 'raw'
CLEAN = ROOT / 'data' / 'clean'
OUT = ROOT / 'outputs'
DOCS = ROOT / 'docs'
SEED = 42

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def read_jsonl(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
