"""Collect IGDB main-story-completion data via the official authenticated API.

Official time-to-beat field `hastily` = seconds to the credits without
notable side content. `normally` includes extras and is NOT our target.
The user must supply their own Twitch developer Client ID and Secret via
process environment. Never put secrets in a notebook, command, or repository.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import time
from pathlib import Path
import requests
from common import RAW, SEED, hash_file, utc_now, write_json

AUTH_URL = 'https://id.twitch.tv/oauth2/token'
API_URL = 'https://api.igdb.com/v4/'
STEAM_ID = re.compile(r'(?:store\.steampowered\.com/app/)(\d+)', re.I)


class IGDB:
    def __init__(self, delay: float = .28):
        self.client_id = os.getenv('IGDB_CLIENT_ID', '').strip()
        secret = os.getenv('IGDB_CLIENT_SECRET', '').strip()
        if not self.client_id or not secret:
            raise RuntimeError('Missing IGDB_CLIENT_ID or IGDB_CLIENT_SECRET environment variables. See .env.example')
        auth = requests.post(AUTH_URL, data={
            'client_id': self.client_id, 'client_secret': secret,
            'grant_type': 'client_credentials'}, timeout=30)
        auth.raise_for_status()
        token = auth.json()['access_token']
        self.headers = {'Client-ID': self.client_id, 'Authorization': 'Bearer ' + token,
                        'Accept': 'application/json', 'Content-Type': 'text/plain'}
        self.delay = delay

    def query(self, endpoint: str, body: str):
        for attempt in range(6):
            try:
                response = requests.post(API_URL + endpoint, headers=self.headers,
                                         data=body.encode('utf-8'), timeout=(15, 70))
                if response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(20, 2 ** attempt))
                    continue
                response.raise_for_status()
                result = response.json()
                time.sleep(self.delay)
                if not isinstance(result, list):
                    raise RuntimeError(f'Unexpected response for {endpoint}: {str(result)[:300]}')
                return result
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(min(15, 2 ** attempt))
        raise RuntimeError(f'{endpoint} request repeatedly failed')


def parse_detail(detail, ttb):
    genres = [v.get('name', '') for v in detail.get('genres', []) if isinstance(v, dict)]
    developers = []
    for item in detail.get('involved_companies', []):
        if isinstance(item, dict) and item.get('developer'):
            company = item.get('company', {})
            if isinstance(company, dict) and company.get('name'):
                developers.append(company['name'])
    websites = detail.get('websites', [])
    steam_id = None
    for entry in websites:
        if isinstance(entry, dict):
            m = STEAM_ID.search(str(entry.get('url', '')))
            if m:
                steam_id = int(m.group(1))
                break
    return {
        'id': detail['id'], 'igdb_game_id': detail['id'],
        'name': detail.get('name', ''), 'description': detail.get('summary', ''),
        'first_release_timestamp': detail.get('first_release_date'),
        'genres': genres, 'tags': [], 'developers': developers,
        'steam_appid': steam_id, 'price_usd': None,
        'release_date': None,
        'hltb_main_hours': round(float(ttb['hastily']) / 3600, 5),
        'time_submissions': ttb.get('count'),
        'time_source': 'IGDB game_time_to_beats.hastily / 3600 (NOT HLTB)',
        'igdb_url': (f'https://www.igdb.com/games/{detail["slug"]}' if detail.get('slug') else ''),
    }


def steam_enrich(game, session: requests.Session):
    """Optional Steam Store metadata, no private credentials; observe rate limits."""
    appid = game.get('steam_appid')
    if not appid:
        return
    url = 'https://store.steampowered.com/api/appdetails'
    try:
        r = session.get(url, params={'appids': appid, 'cc': 'us', 'l': 'english'}, timeout=25)
        if r.status_code == 429:
            print('Steam returned 429; enrichment paused (missing prices will be imputed).')
            return False
        r.raise_for_status()
        result = r.json().get(str(appid), {})
        if result.get('success'):
            data = result.get('data', {})
            game['description'] = data.get('short_description') or game['description']
            price = data.get('price_overview')
            if price and isinstance(price.get('initial'), (int, float)):
                game['price_usd'] = price['initial']
            elif data.get('is_free'):
                game['price_usd'] = 0
            tags = data.get('genres', [])
            if tags and not game.get('genres'):
                game['genres'] = [g.get('description', '') for g in tags]
    except (requests.RequestException, ValueError):
        pass
    return True


def collect(desired: int = 6500, min_rows: int = 3000, max_pages: int = 50,
            steam_enrich_limit: int = 0):
    api = IGDB()
    times = {}
    for page in range(max_pages):
        rows = api.query('game_time_to_beats',
                         f'fields game_id,hastily,count; where hastily > 0; sort game_id asc; limit 500; offset {page*500};')
        if not rows:
            break
        for entry in rows:
            if entry.get('game_id') and entry.get('hastily', 0) > 0:
                times[int(entry['game_id'])] = entry
        print('Time-to-beat records:', len(times), flush=True)
        if len(times) >= desired * 1.45:
            break
    ids = sorted(times)
    collected = []
    for start in range(0, len(ids), 450):
        chunk = ids[start:start + 450]
        query = (f'fields id,name,slug,summary,first_release_date,genres.name,'
                 f'involved_companies.company.name,involved_companies.developer,websites.url; '
                 f'where id = ({",".join(map(str,chunk))}); limit 500;')
        details = api.query('games', query)
        for game in details:
            game_id = game.get('id')
            if game_id not in times:
                continue
            item = parse_detail(game, times[game_id])
            if len(item['description'].strip()) < 45:
                continue
            if not (0 < item['hltb_main_hours'] <= 1000):
                continue
            collected.append(item)
        if len(collected) >= desired:
            break
    collected = sorted(collected, key=lambda x: x['igdb_game_id'])[:desired]
    if len(collected) < min_rows:
        raise RuntimeError(f'Collected {len(collected)} eligible rows, below {min_rows}. Increase --max-pages or check IGDB coverage.')
    if steam_enrich_limit:
        with requests.Session() as session:
            for idx, game in enumerate(collected[:steam_enrich_limit]):
                if steam_enrich(game, session) is False:
                    break
                time.sleep(.35)
                if (idx + 1) % 100 == 0:
                    print('Steam enrichment attempted:', idx + 1)
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / 'games_raw.jsonl'
    with out.open('w', encoding='utf-8') as f:
        for g in collected:
            f.write(json.dumps(g, ensure_ascii=False) + '\n')
    write_json(RAW / 'collection_manifest.json', {
        'collection_method': 'IGDB official API + optional Steam store appdetails',
        'main_source': 'https://api.igdb.com/v4/game_time_to_beats',
        'additional_source': 'https://store.steampowered.com/api/appdetails',
        'collected_at_utc': utc_now(), 'raw_subset_rows': len(collected),
        'raw_subset_sha256': hash_file(out), 'source_snapshot': 'games_raw.jsonl',
        'target_field': 'hastily (seconds) / 3600; excludes notable extras',
        'source_credentials': 'Twitch client credentials; not saved in dataset',
        'steam_enrichment_limit': steam_enrich_limit,
        'collection_note': 'IGDB may have limited Steam app links; missing prices/tags are real missing fields, not invented values',
    })
    print('Collected', len(collected), 'games into', out)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--desired', type=int, default=6500)
    ap.add_argument('--min-rows', type=int, default=3000)
    ap.add_argument('--max-pages', type=int, default=50)
    ap.add_argument('--steam-enrich-limit', type=int, default=0,
                    help='At most this many Steam appdetails requests (0 disables)')
    args = ap.parse_args()
    collect(args.desired, args.min_rows, args.max_pages, args.steam_enrich_limit)
