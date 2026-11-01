"""Utilities shared by the data collection and training scripts."""
from __future__ import annotations
import hashlib
import json
import re
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data' / 'raw'
CLEAN = ROOT / 'data' / 'clean'
OUT = ROOT / 'outputs'
SEED = 42


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def hash_file(path: Path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for piece in iter(lambda: f.read(1024 * 1024), b''):
            h.update(piece)
    return h.hexdigest()


def normalize_text(text: object) -> str:
    if not text:
        return ''
    s = re.sub(r'<(script|style)\b[^>]*>.*?</\1\s*>', ' ', str(text), flags=re.I | re.S)
    s = re.sub(r'<[^>]*>', ' ', s)
    s = unescape(s)
    return re.sub(r'\s+', ' ', s).strip()


def flatten_names(values):
    """Handle strings, lists of strings, and IGDB-style nested objects."""
    if values is None:
        return []
    if isinstance(values, str):
        return [values] if values.strip() else []
    if isinstance(values, dict):
        values = [values]
    if not isinstance(values, (list, tuple)):
        return []
    names = []
    for obj in values:
        if isinstance(obj, str):
            name = obj
        elif isinstance(obj, dict):
            name = obj.get('name', '')
        else:
            name = ''
        if isinstance(name, str) and name.strip():
            names.append(name.strip().replace('|', '/'))
    return sorted(set(names), key=str.lower)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')
