"""Optional credential-free download of openly published, combined game metadata.

This is a third-party *precompiled* Steam/HLTB/IGDB-derived dataset. It is
NOT an original IGDB query or direct HowLongToBeat crawl. It is provided as a
transparent fallback when Twitch credentials are unavailable. The file may be
large (~50+ MB compressed), and its upstream content/licensing can change.
"""
from __future__ import annotations
import argparse
import gzip
import json
import random
import sys
from pathlib import Path
import requests

from common import RAW, SEED, hash_file, normalize_text, utc_now, write_json

URLS = [
    'https://media.githubusercontent.com/media/leinstay/steamdb/main/steamdb.min.json.gz',
    'https://github.com/leinstay/steamdb/raw/refs/heads/main/steamdb.min.json.gz',
]
ATTRIBUTION = 'https://github.com/leinstay/steamdb'


def download_source(path: Path, timeout: int = 90):
    """Download full compressed upstream snapshot; no fabricated fallback."""
    failures = []
    for url in URLS:
        tmp = path.with_name(path.name + '.part')
        try:
            print('Downloading snapshot from', url, flush=True)
            r = requests.get(url, stream=True, timeout=(20, timeout),
                             headers={'User-Agent': 'AcademicCourseProject/1.0 (respectful-public-download)'})
            r.raise_for_status()
            with tmp.open('wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            with gzip.open(tmp, 'rb') as f:
                first = f.read(1)
                if first not in (b'[', b'{', b'\n', b' ', b'\t'):
                    raise ValueError('Unexpected JSON/gzip content')
            tmp.replace(path)
            return url
        except (requests.RequestException, OSError, ValueError) as exc:
            failures.append(f'{url}: {type(exc).__name__}: {exc}')
            if tmp.exists():
                tmp.unlink()
    raise RuntimeError('Download failed. Obtain the .gz from the source repository and use --source-file.\n' + '\n'.join(failures))


def iter_snapshot(source: Path):
    try:
        import ijson
    except ImportError as exc:
        raise SystemExit('Please run: pip install -r requirements.txt (missing ijson)') from exc
    opener = gzip.open if source.suffix == '.gz' else open
    with opener(source, 'rb') as fh:
        leading = fh.read(1024).lstrip()
        fh.seek(0)
        prefix = 'item' if leading.startswith(b'[') else 'games.item'
        for entry in ijson.items(fh, prefix):
            if isinstance(entry, dict):
                yield entry


def eligible(game):
    if game.get('kind') not in (None, 'steam'):
        return False
    if game.get('steam_appid') is None or game.get('hltb_main_hours') is None:
        return False
    try:
        y = float(game['hltb_main_hours'])
    except (TypeError, ValueError):
        return False
    if not (0 < y <= 1000):
        return False
    return len(normalize_text(game.get('description', ''))) >= 45 and bool(game.get('name'))


def reservoir_sample(rows, desired: int):
    rng = random.Random(SEED)
    sample, eligible_count, total = [], 0, 0
    for row in rows:
        total += 1
        if not eligible(row):
            continue
        eligible_count += 1
        if len(sample) < desired:
            sample.append(row)
        else:
            j = rng.randrange(eligible_count)
            if j < desired:
                sample[j] = row
    return sorted(sample, key=lambda d: (int(d.get('steam_appid') or 0), str(d.get('name')))), eligible_count, total


def collect(source_file: Path | None, desired: int = 6500, min_rows: int = 3000):
    RAW.mkdir(parents=True, exist_ok=True)
    if desired < min_rows:
        raise ValueError('--desired must be at least --min-rows')
    stamp = utc_now().replace(':', '-').replace('+', '_')
    source = source_file or RAW / f'upstream_steamdb_{stamp}.json.gz'
    fetched_url = '(local source file)'
    if not source_file:
        fetched_url = download_source(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    sample, eligible_count, total = reservoir_sample(iter_snapshot(source), desired)
    if len(sample) < min_rows:
        raise RuntimeError(f'Only {len(sample)} usable games, below required {min_rows}. No dataset was fabricated.')
    out = RAW / 'games_raw.jsonl'
    with out.open('w', encoding='utf-8') as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
    manifest = {
        'collection_method': 'Programmatic HTTP download and deterministic reservoir selection from a public third-party compilation',
        'primary_record_provenance': 'Game Gauntlets merged Steam + HowLongToBeat + IGDB and others',
        'NOT_direct_IGDB_or_HLTB_scrape': True,
        'source_page': ATTRIBUTION, 'download_url': fetched_url,
        'upstream_license_stated': 'GPL-3.0 for compiled dataset; original descriptions have third-party rights',
        'collected_at_utc': utc_now(), 'snapshot_file': str(source.relative_to(RAW.parent.parent)) if source.is_relative_to(RAW.parent.parent) else str(source),
        'snapshot_sha256': hash_file(source), 'raw_subset_sha256': hash_file(out),
        'total_records_in_snapshot': total, 'eligible_before_sampling': eligible_count,
        'raw_subset_rows': len(sample), 'seed': SEED,
        'sampling': 'reservoir sampling of eligible records (not sorted head), fixed seed 42',
        'target_field': 'hltb_main_hours (hours)', 'original_raw_unmodified_fields': True,
    }
    write_json(RAW / 'collection_manifest.json', manifest)
    print(f'Upstream: {total:,} | eligible: {eligible_count:,} | collected raw: {len(sample):,}')
    print(f'Wrote {out}\nWrote {RAW / "collection_manifest.json"}')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-file', type=Path, help='Previously downloaded steamdb.min.json.gz or .json')
    ap.add_argument('--desired', type=int, default=6500)
    ap.add_argument('--min-rows', type=int, default=3000)
    args = ap.parse_args()
    collect(args.source_file, args.desired, args.min_rows)
