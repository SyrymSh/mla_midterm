"""Add real IGDB and Steam metadata to the exact frozen midterm cohort.

Credentials come from environment variables or one unlogged stdin JSON line.
OAuth secrets/tokens are never written. Requests are cached with timestamps.
Metadata requests never use time-to-beat fields or individual Steam reviews.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from common import RAW, MIDTERM, append_jsonl, read_jsonl, sha256, utc_now, write_json

UA = 'UniversityCourseProject/1.0 (public video game metadata; academic research)'
STEAM_APP = re.compile(r'store\.steampowered\.com/app/(\d+)', re.I)

def request(url, data=None, headers=None, attempts=5):
    """Bounded exponential backoff, Retry-After, no credential-bearing errors."""
    for attempt in range(attempts):
        try:
            req = Request(url, data=data, headers={'User-Agent':UA, **(headers or {})})
            with urlopen(req, timeout=70) as response:
                return response.read().decode('utf-8')
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == attempts-1:
                raise RuntimeError(f'HTTP {error.code} from upstream endpoint') from None
            pause = max(2 ** (attempt+1), float(error.headers.get('Retry-After','0')))
            time.sleep(min(pause, 120))
        except (URLError, TimeoutError, OSError):
            if attempt == attempts-1:
                raise RuntimeError('Upstream connection failed after retries') from None
            time.sleep(2 ** (attempt+1))
    raise RuntimeError('Request exhausted retries')

def get_json(url, data=None, headers=None):
    return json.loads(request(url, data, headers))

def credentials(stdin=False):
    if stdin:
        print('Waiting for credentials on stdin (not saved or echoed).', flush=True)
        if sys.stdin.isatty():
            import getpass
            line = getpass.getpass('Credentials JSON (hidden): ')
        else:
            line = sys.stdin.readline()
        obj = json.loads(line)
        return obj['client_id'], obj['client_secret']
    client = os.getenv('IGDB_CLIENT_ID','').strip()
    secret = os.getenv('IGDB_CLIENT_SECRET','').strip()
    if not client or not secret:
        raise RuntimeError('Set IGDB_CLIENT_ID and IGDB_CLIENT_SECRET in this process.')
    return client, secret

def igdb_collect(ids, stdin=False):
    path=RAW/'igdb_metadata.jsonl'
    seen={int(row['data']['id']) for row in read_jsonl(path)}
    pending=[gid for gid in ids if gid not in seen]
    if not pending:
        return
    client, secret=credentials(stdin)
    token=get_json('https://id.twitch.tv/oauth2/token', urlencode({
        'client_id':client,'client_secret':secret,'grant_type':'client_credentials'
    }).encode(), {'Content-Type':'application/x-www-form-urlencoded'})['access_token']
    del secret
    headers={'Client-ID':client, 'Authorization':'Bearer '+token, 'Content-Type':'text/plain'}
    fields=('id,name,slug,first_release_date,updated_at,game_type.type,'
            'genres.name,themes.name,keywords.name,game_modes.name,player_perspectives.name,'
            'platforms.name,game_engines.name,collections.name,franchises.name,'
            'parent_game,version_parent,dlcs,expansions,standalone_expansions,'
            'involved_companies.company.name,involved_companies.developer,involved_companies.publisher,'
            'websites.url,external_games.uid,external_games.url,external_games.external_game_source.name')
    for start in range(0,len(pending),150):
        chunk=pending[start:start+150]
        query=f'fields {fields}; where id = ({",".join(map(str,chunk))}); limit 500;'
        payload=get_json('https://api.igdb.com/v4/games',query.encode(),headers)
        if not isinstance(payload,list):
            raise RuntimeError('Expected IGDB array')
        stamp=utc_now()
        for item in payload:
            append_jsonl(path,{'collected_at_utc':stamp,'endpoint':'https://api.igdb.com/v4/games','data':item})
        print(f'IGDB metadata cached: {min(start+150,len(pending))}/{len(pending)} requested; {len(payload)} returned',flush=True)
        time.sleep(.3)

def make_steam_map():
    metadata={int(x['data']['id']):x['data'] for x in read_jsonl(RAW/'igdb_metadata.jsonl')}
    with (MIDTERM/'data/clean/games_clean.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    mapping=[]
    for row in rows:
        gid=int(row['game_id'])
        old=int(float(row['steam_appid'])) if row['steam_appid'] else None
        item=metadata.get(gid,{})
        website_ids={int(m.group(1)) for site in item.get('websites',[]) if (m:=STEAM_APP.search(site.get('url','')))}
        external_ids=set()
        for external in item.get('external_games',[]):
            source=external.get('external_game_source',{})
            if isinstance(source,dict) and source.get('name','').lower()=='steam' and str(external.get('uid','')).isdigit():
                external_ids.add(int(external['uid']))
            if m:=STEAM_APP.search(external.get('url','')):
                external_ids.add(int(m.group(1)))
        candidates=website_ids|external_ids
        # Explicit source IDs only. Never fuzzy-match titles or choose among conflicts.
        if old and (not candidates or old in candidates):
            appid,method=old,'midterm_explicit_steam_url'
        elif len(candidates)==1:
            appid,method=next(iter(candidates)),'igdb_explicit_external_id_or_url'
        else:
            appid,method=None,'unmatched_or_ambiguous'
        mapping.append({'game_id':gid,'steam_appid':appid,'method':method,'old_steam_appid':old,
                        'candidate_appids':sorted(candidates),'old_id_conflict':bool(old and candidates and old not in candidates)})
    write_json(RAW/'steam_id_mapping.json',mapping)
    return sorted({x['steam_appid'] for x in mapping if x['steam_appid']})

def steam_collect(appids):
    # Retrieve tag dictionary once. Numeric tags remain valid if a label is absent.
    dictionary=RAW/'steam_tag_dictionary.json'
    if not dictionary.exists():
        url='https://api.steampowered.com/IStoreService/GetTagList/v1/?language=english'
        try:
            write_json(dictionary,{'collected_at_utc':utc_now(),'url':url,'response':get_json(url)})
        except RuntimeError as e:
            print('Tag labels unavailable; retaining original Steam tag IDs:',e,flush=True)
    for kind,filename,size in [('browse','steam_browse.jsonl',40),('prices','steam_prices.jsonl',50)]:
        path=RAW/filename
        seen={int(x['appid']) for x in read_jsonl(path)}
        pending=[appid for appid in appids if appid not in seen]
        for start in range(0,len(pending),size):
            chunk=pending[start:start+size]
            if kind=='browse':
                query={'ids':[{'appid':n} for n in chunk],
                       'context':{'language':'english','country_code':'US'},
                       'data_request':{'include_basic_info':True,'include_tag_count':20,
                                       'include_platforms':True,'include_release':True,
                                       'include_supported_languages':True}}
                url='https://api.steampowered.com/IStoreBrowseService/GetItems/v1/?'+urlencode({'input_json':json.dumps(query,separators=(',',':'))})
                payload=get_json(url)
                items=payload.get('response',{}).get('store_items',[])
                by_id={int(x.get('appid',x.get('id',0))):x for x in items}
            else:
                url='https://store.steampowered.com/api/appdetails?'+urlencode({
                    'appids':','.join(map(str,chunk)),'cc':'us','l':'english','filters':'price_overview'})
                payload=get_json(url)
                by_id={int(k):v for k,v in payload.items()}
            stamp=utc_now()
            for appid in chunk:
                append_jsonl(path,{'appid':appid,'collected_at_utc':stamp,'url':url,
                                   'data':by_id.get(appid),'returned':appid in by_id})
            print(f'Steam {kind} cached: {min(start+size,len(pending))}/{len(pending)} pending apps',flush=True)
            time.sleep(1.6)

def source_checks():
    checks={
        'steam_robots':'https://store.steampowered.com/robots.txt',
        'steam_api_terms':'https://steamcommunity.com/dev/apiterms',
        'steam_subscriber_agreement':'https://store.steampowered.com/subscriber_agreement/',
        'igdb_docs':'https://api-docs.igdb.com/',
        'igdb_robots':'https://www.igdb.com/robots.txt',
    }
    for name,url in checks.items():
        path=RAW/'source_checks'/f'{name}.txt'
        if path.exists():
            continue
        try:
            text=request(url)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(f'URL: {url}\nRetrieved UTC: {utc_now()}\n\n'+text,encoding='utf-8')
        except RuntimeError as e:
            print('Source check unavailable:',name,str(e),flush=True)
            write_json(RAW/'source_checks'/f'{name}_status.json',{'url':url,'checked_at_utc':utc_now(),'status':str(e)})

def run(only_steam=False,stdin=False):
    RAW.mkdir(parents=True,exist_ok=True)
    with (MIDTERM/'data/clean/games_clean.csv').open(encoding='utf-8-sig',newline='') as stream:
        ids=sorted(int(x['game_id']) for x in csv.DictReader(stream))
    source_checks()
    if not only_steam:
        igdb_collect(ids,stdin)
    appids=make_steam_map()
    print('Explicitly matched unique Steam apps:',len(appids),flush=True)
    steam_collect(appids)
    files={str(p.relative_to(RAW)).replace('\\','/'):{'sha256':sha256(p),'bytes':p.stat().st_size}
           for p in RAW.rglob('*') if p.is_file() and p.name!='collection_manifest.json'}
    write_json(RAW/'collection_manifest.json',{
        'completed_at_utc':utc_now(),'original_cohort_rows':len(ids),
        'igdb_metadata_rows':len(read_jsonl(RAW/'igdb_metadata.jsonl')),
        'unique_steam_apps_requested':len(appids),'region':'US','price_currency':'USD',
        'sources':['IGDB official v4 API','Steam public IStoreBrowseService/GetItems/v1',
                   'Steam public appdetails price_overview','Steam IStoreService/GetTagList/v1'],
        'target_policy':'Unmodified midterm IGDB hastily/3600 target; no new time fields collected.',
        'credentials':'Environment or stdin only. Never included in snapshots.',
        'files':files})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only-steam',action='store_true')
    parser.add_argument('--credentials-stdin',action='store_true')
    args=parser.parse_args()
    run(args.only_steam,args.credentials_stdin)
