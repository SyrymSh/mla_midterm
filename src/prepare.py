"""Clean actual raw records. No synthetic row generation or target inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import RAW, CLEAN, SEED, hash_file, flatten_names, normalize_text, utc_now, write_json

OUT_FIELDS = [
    'game_id','steam_appid','title','main_story_hours','release_year','price_usd',
    'primary_genre','genres_text','tags_text','primary_developer','description',
    'description_word_count','price_missing','source_url',
]


def valid_year(value):
    try:
        if value is None or value == '':
            return np.nan
        x = pd.to_datetime(value, utc=True, errors='coerce')
        if pd.isna(x):
            return np.nan
        year = int(x.year)
        return float(year) if 1970 <= year <= 2026 else np.nan
    except (ValueError, OverflowError, TypeError):
        return np.nan


def parse_record(g):
    name = normalize_text(g.get('name') or g.get('title'))
    description = normalize_text(g.get('description') or g.get('summary'))
    genres = flatten_names(g.get('genres'))
    tags = flatten_names(g.get('tags'))
    devs = flatten_names(g.get('developers'))
    if not devs:
        devs = flatten_names(g.get('developer'))
    year = valid_year(g.get('release_date'))
    if np.isnan(year) and g.get('first_release_timestamp') is not None:
        try:
            year = float(pd.Timestamp(int(g['first_release_timestamp']), unit='s', tz='UTC').year)
        except (ValueError, TypeError, OverflowError):
            pass
    target = g.get('hltb_main_hours')
    if target is None:
        target = g.get('main_story_hours')
    try:
        target = float(target)
    except (ValueError, TypeError):
        target = np.nan
    raw_price = g.get('price_usd')
    try:
        price = float(raw_price) / 100 if raw_price is not None else np.nan
        if price < 0 or price > 1000:
            price = np.nan
    except (ValueError, TypeError):
        price = np.nan
    appid = g.get('steam_appid')
    try:
        appid = int(appid) if appid not in ('', None) else None
    except (TypeError, ValueError):
        appid = None
    gid = g.get('igdb_game_id') or g.get('id') or appid
    source = g.get('hltb_url') or g.get('igdb_url') or (f'https://store.steampowered.com/app/{appid}' if appid else '')
    return dict(
        game_id=str(gid), steam_appid=appid, title=name,
        main_story_hours=target, release_year=year, price_usd=price,
        primary_genre=genres[0] if genres else 'Unknown',
        genres_text='|'.join(genres), tags_text='|'.join(tags),
        primary_developer=devs[0] if devs else 'Unknown',
        description=description, description_word_count=len(description.split()),
        price_missing=int(np.isnan(price)), source_url=source,
    )


def clean(raw_path: Path = RAW / 'games_raw.jsonl', min_rows: int = 3000):
    if not raw_path.exists():
        raise FileNotFoundError(f'Missing real raw snapshot: {raw_path}. Run src/collect_igdb.py or src/collect_mirror.py first.')
    entries = []
    with raw_path.open(encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                try:
                    entries.append(parse_record(json.loads(line)))
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f'Invalid raw record on line {lineno}') from exc
    df = pd.DataFrame(entries, columns=OUT_FIELDS)
    raw_size = len(df)
    bad_target = df.main_story_hours.isna() | ~np.isfinite(df.main_story_hours) | ~df.main_story_hours.between(.1, 1000)
    bad_text = df.description_word_count < 8
    bad_title = df.title.str.len().fillna(0) < 2
    excluded = bad_target | bad_text | bad_title
    df = df.loc[~excluded].copy()
    # Preserve one row per title+year; port/edition entries with different IDs may be duplicates.
    before_dedup = len(df)
    df['_dedup_title'] = df.title.str.lower().str.replace(r'\W+', '', regex=True)
    df = df.drop_duplicates(['_dedup_title', 'release_year'], keep='first')
    df = df.drop(columns=['_dedup_title'])
    df = df.drop_duplicates(['game_id'], keep='first')
    df = df.sort_values('game_id').reset_index(drop=True)
    if len(df) < min_rows:
        raise ValueError(f'Only {len(df)} clean real records, under requirement of {min_rows}; collect more games. Cleaning did not fill targets.')
    CLEAN.mkdir(parents=True, exist_ok=True)
    out = CLEAN / 'games_clean.csv'
    df.to_csv(out, index=False, encoding='utf-8')
    diagnostics = {
        'prepared_at_utc': utc_now(), 'raw_path': str(raw_path),
        'raw_rows': raw_size, 'excluded_target': int(bad_target.sum()),
        'excluded_text': int(bad_text.sum()), 'excluded_title': int(bad_title.sum()),
        'excluded_any': int(excluded.sum()),
        'duplicate_or_near_duplicate_removed': int(before_dedup - len(df)),
        'cleaned_rows': len(df), 'columns': list(df.columns),
        'missing_price_percent': round(100 * df.price_usd.isna().mean(), 2),
        'missing_release_year_percent': round(100 * df.release_year.isna().mean(), 2),
        'empty_tags_percent': round(100 * df.tags_text.fillna('').eq('').mean(), 2),
        'median_target_hours': round(float(df.main_story_hours.median()), 3),
        'target_p95_hours': round(float(df.main_story_hours.quantile(.95)), 3),
        'csv_sha256': hash_file(out),
        'notes': 'No missing target is imputed. Missing numeric predictors are imputed inside the train-only fitted pipeline.'
    }
    write_json(CLEAN / 'cleaning_report.json', diagnostics)
    print(json.dumps({k: v for k, v in diagnostics.items() if k not in ('columns', 'csv_sha256')}, indent=2))
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=RAW / 'games_raw.jsonl')
    ap.add_argument('--min-rows', type=int, default=3000)
    args = ap.parse_args()
    clean(args.input, args.min_rows)
