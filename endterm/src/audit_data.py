"""Trace cleaning decisions and observed prices/tags back to the saved sources."""
import html
import re
import numpy as np
import pandas as pd
from common import CLEAN, MIDTERM, RAW, read_jsonl, utc_now, write_json
import json

def text(value):
    value=re.sub(r'<(script|style)\b[^>]*>.*?</\1\s*>',' ',str(value or ''),flags=re.I|re.S)
    return re.sub(r'\s+',' ',html.unescape(re.sub(r'<[^>]*>',' ',value))).strip()

def run():
    records=[]
    for line,g in enumerate(read_jsonl(MIDTERM/'data/raw/games_raw.jsonl'),1):
        title=text(g.get('name') or g.get('title'))
        description=text(g.get('description') or g.get('summary'))
        target=pd.to_numeric(g.get('hltb_main_hours',g.get('main_story_hours')),errors='coerce')
        date=pd.to_datetime(g.get('release_date'),utc=True,errors='coerce')
        year=date.year if pd.notna(date) and 1970<=date.year<=2026 else np.nan
        if pd.isna(year) and g.get('first_release_timestamp') is not None:
            stamp=pd.to_datetime(g['first_release_timestamp'],unit='s',utc=True,errors='coerce')
            year=stamp.year if pd.notna(stamp) else np.nan
        records.append({'raw_line':line,'game_id':str(g.get('igdb_game_id') or g.get('id') or g.get('steam_appid')),
                        'title':title,'main_story_hours':target,'release_year':year,
                        'description_word_count':len(description.split()),
                        'normalized_title':re.sub(r'\W+','',title.lower())})
    audit=pd.DataFrame(records)
    audit['invalid_target']=~np.isfinite(audit.main_story_hours) | ~audit.main_story_hours.between(.1,1000)
    audit['short_description']=audit.description_word_count.lt(8)
    audit['short_title']=audit.title.str.len().lt(2)
    invalid=audit[['invalid_target','short_description','short_title']].any(axis=1)
    eligible=audit.loc[~invalid]
    duplicate_title=eligible.duplicated(['normalized_title','release_year'],keep='first')
    audit['duplicate_title_year']=False
    audit.loc[eligible.index,'duplicate_title_year']=duplicate_title
    remaining=eligible.loc[~duplicate_title]
    audit['duplicate_id']=False
    audit.loc[remaining.index,'duplicate_id']=remaining.duplicated('game_id',keep='first')
    reasons=['invalid_target','short_description','short_title','duplicate_title_year','duplicate_id']
    audit['kept']=~audit[reasons].any(axis=1)
    audit['reason']=audit[reasons].apply(lambda r:', '.join(k for k,v in r.items() if v) or 'kept',axis=1)
    original=pd.read_csv(MIDTERM/'data/clean/games_clean.csv',dtype={'game_id':str}).set_index('game_id')
    reproduced=audit.loc[audit.kept].set_index('game_id').sort_index()
    assert reproduced.index.tolist()==original.index.tolist(), 'Cleaning does not reproduce the frozen cohort'
    np.testing.assert_allclose(reproduced.main_story_hours,original.main_story_hours,rtol=0,atol=1e-10)
    audit.drop(columns='normalized_title').to_csv(CLEAN/'cleaning_audit.csv',index=False)
    stages=[{'step':'Raw snapshot','rows_before':len(audit),'removed':0,'rows_after':len(audit)},
            {'step':'Invalid target / insufficient text / unusable title','rows_before':len(audit),
             'removed':int(invalid.sum()),'rows_after':int((~invalid).sum())},
            {'step':'Duplicate normalized title and release year','rows_before':len(eligible),
             'removed':int(duplicate_title.sum()),'rows_after':len(remaining)},
            {'step':'Duplicate game ID','rows_before':len(remaining),'removed':int(audit.duplicate_id.sum()),
             'rows_after':int(audit.kept.sum())}]
    pd.DataFrame(stages).to_csv(CLEAN/'cleaning_steps.csv',index=False)
    enriched=pd.read_csv(CLEAN/'games_enriched.csv',dtype={'game_id':str})
    prices={x['appid']:x.get('data') or {} for x in read_jsonl(RAW/'steam_prices.jsonl')}
    browse={x['appid']:x.get('data') or {} for x in read_jsonl(RAW/'steam_browse.jsonl')}
    dictionary=json.loads((RAW/'steam_tag_dictionary.json').read_text(encoding='utf-8'))['response']['response']['tags']
    tag_names={int(t['tagid']):t['name'] for t in dictionary}
    verified_prices=verified_tags=0
    for row in enriched.itertuples(index=False):
        appid=int(row.matched_steam_appid) if pd.notna(row.matched_steam_appid) else None
        result=prices.get(appid,{})
        quote=(result.get('data') or {}).get('price_overview') or {}
        expected=quote.get('initial')/100 if quote.get('currency')=='USD' and quote.get('initial') is not None else np.nan
        if pd.isna(expected):
            assert pd.isna(row.steam_regular_price_usd)
        else:
            assert np.isclose(row.steam_regular_price_usd,expected,atol=1e-10)
            verified_prices+=1
        for field,key,divisor in [('steam_current_price_usd','final',100),('steam_discount_percent','discount_percent',1)]:
            source=quote.get(key)/divisor if quote.get('currency')=='USD' and quote.get(key) is not None else np.nan
            actual=getattr(row,field)
            assert (pd.isna(source) and pd.isna(actual)) or (pd.notna(source) and np.isclose(actual,source,atol=1e-10))
        item=browse.get(appid,{})
        tags=item.get('tagids') if item.get('success')==1 else None
        expected_tags='|'.join(tag_names.get(int(t),f'tag_id_{t}').replace('|','/') for t in tags) if tags else None
        if expected_tags is None:
            assert pd.isna(row.steam_tags)
        else:
            assert row.steam_tags==expected_tags
            verified_tags+=1
    write_json(CLEAN/'source_verification.json',{
        'verified_at_utc':utc_now(),'raw_rows':len(audit),'reproduced_clean_rows':int(audit.kept.sum()),
        'invalid_target_rows':int(audit.invalid_target.sum()),'short_description_rows':int(audit.short_description.sum()),
        'short_title_rows':int(audit.short_title.sum()),'duplicate_rows':int(audit.duplicate_title_year.sum()+audit.duplicate_id.sum()),
        'price_rows_verified_against_raw_cents':verified_prices,'tag_rows_verified_against_raw_ids':verified_tags,
        'cohort_and_labels_reproduced':True})
    print('Cleaning lineage, observed prices and tags verified.',flush=True)
    return audit

if __name__=='__main__':
    run()
