"""Left-join real metadata onto the frozen cohort. Never alter or impute labels."""
from __future__ import annotations
import json
import shutil
import numpy as np
import pandas as pd
from common import CLEAN, MIDTERM, RAW, read_jsonl, sha256, utc_now, write_json

def names(item, field):
    values=item.get(field)
    if not isinstance(values,list) or not values:
        return None
    tokens=sorted({str(x['name']).strip().replace('|','/') for x in values if isinstance(x,dict) and x.get('name')},key=str.lower)
    return '|'.join(tokens) if tokens else None

def count_list(item, field):
    value=item.get(field)
    return len(value) if isinstance(value,list) else np.nan

def prepare():
    base=pd.read_csv(MIDTERM/'data/clean/games_clean.csv',dtype={'game_id':str})
    original_columns=list(base.columns)
    metadata={str(x['data']['id']):x for x in read_jsonl(RAW/'igdb_metadata.jsonl')}
    mapping={str(x['game_id']):x for x in json.loads((RAW/'steam_id_mapping.json').read_text(encoding='utf-8'))}
    browse={x['appid']:x for x in read_jsonl(RAW/'steam_browse.jsonl')}
    prices={x['appid']:x for x in read_jsonl(RAW/'steam_prices.jsonl')}
    tags_path=RAW/'steam_tag_dictionary.json'
    tags_response=json.loads(tags_path.read_text(encoding='utf-8'))['response'] if tags_path.exists() else {}
    tag_names={int(t['tagid']):t['name'] for t in tags_response.get('response',{}).get('tags',[]) if 'tagid' in t and 'name' in t}
    extras=[]
    for _,row in base.iterrows():
        gid=row.game_id
        envelope=metadata.get(gid,{})
        g=envelope.get('data',{})
        match=mapping[gid]
        appid=match['steam_appid']
        steam_envelope=browse.get(appid,{})
        s=steam_envelope.get('data') or {}
        success=s.get('success')==1
        s=s if success else {}
        price_envelope=prices.get(appid,{})
        price_result=price_envelope.get('data') or {}
        price=(price_result.get('data') or {}).get('price_overview') or {}
        # Only an explicit USD quote is a price. Unpriced/free/unavailable is NOT zero.
        usd=price.get('currency')=='USD'
        initial=price.get('initial') if usd else None
        final=price.get('final') if usd else None
        r={'game_id':gid,'matched_steam_appid':appid,'steam_match_method':match['method'],
           'steam_old_id_conflict':int(match['old_id_conflict']),
           'igdb_metadata_available':int(bool(g)), 'steam_metadata_available':int(success),
           'igdb_collected_at_utc':envelope.get('collected_at_utc'),
           'steam_collected_at_utc':steam_envelope.get('collected_at_utc'),
           'steam_price_collected_at_utc':price_envelope.get('collected_at_utc'),
           'igdb_game_type':(g.get('game_type') or {}).get('type'),
           'igdb_parent_game_id':g.get('parent_game'),'igdb_version_parent_id':g.get('version_parent'),
           'steam_name':s.get('name'), 'steam_item_type':str(s.get('type',0)) if success else None,
           'steam_regular_price_usd':initial/100 if initial is not None else np.nan,
           'steam_current_price_usd':final/100 if final is not None else np.nan,
           'steam_discount_percent':price.get('discount_percent') if usd else np.nan,
           'steam_price_currency':price.get('currency'),
           'steam_price_region':'US' if price_envelope else None,
           'steam_source_url':f'https://store.steampowered.com/app/{appid}/' if appid else None}
        for field in ['themes','keywords','game_modes','player_perspectives','platforms','game_engines']:
            r['igdb_'+field]=names(g,field)
            r['igdb_'+field+'_count']=count_list(g,field)
        for field in ['dlcs','expansions','standalone_expansions']:
            r['igdb_'+field+'_count']=count_list(g,field)
        companies=g.get('involved_companies')
        for role in ['developer','publisher']:
            values=sorted({c['company']['name'] for c in companies or [] if c.get(role) and isinstance(c.get('company'),dict) and c['company'].get('name')})
            r['igdb_'+role+'s']='|'.join(values) if values else None
            r['igdb_'+role+'_count']=len(values) if companies is not None else np.nan
        modes=g.get('game_modes')
        for mode,label in [('Single player','single_player'),('Multiplayer','multiplayer'),('Co-operative','coop')]:
            r['igdb_'+label]=int(any(x.get('name')==mode for x in modes)) if modes else np.nan
        tag_values=s.get('tagids')
        r['steam_tags']='|'.join(tag_names.get(int(t),f'tag_id_{t}').replace('|','/') for t in tag_values) if tag_values else None
        r['steam_tag_count']=len(tag_values) if tag_values is not None else np.nan
        categories=s.get('categories')
        category_ids=set()
        for value in (categories or {}).values():
            if isinstance(value,list): category_ids.update(value)
        r['steam_categories']='|'.join(str(v) for v in sorted(category_ids)) if categories else None
        for code,name in [(2,'single_player'),(1,'multiplayer'),(9,'coop'),(38,'online_coop'),(22,'achievements_listed'),(30,'workshop'),(23,'cloud'),(28,'full_controller_support')]:
            r['steam_'+name]=int(code in category_ids) if categories is not None else np.nan
        platforms=s.get('platforms')
        for platform in ['windows','mac','steamos_linux']:
            r['steam_'+platform]=int(bool(platforms.get(platform,False))) if platforms else np.nan
        languages=s.get('supported_languages')
        r['steam_language_count']=sum(bool(v.get('supported')) for v in languages) if isinstance(languages,list) else np.nan
        extras.append(r)
    joined=base.merge(pd.DataFrame(extras),on='game_id',validate='one_to_one',how='left',sort=False)
    assert len(joined)==len(base) and joined.game_id.tolist()==base.game_id.tolist()
    assert np.array_equal(joined.main_story_hours.to_numpy(),base.main_story_hours.to_numpy())
    CLEAN.mkdir(parents=True,exist_ok=True)
    joined.to_csv(CLEAN/'games_enriched.csv',index=False)
    shutil.copyfile(MIDTERM/'data/clean/fixed_split.csv',CLEAN/'fixed_split.csv')
    assert sha256(CLEAN/'fixed_split.csv')==sha256(MIDTERM/'data/clean/fixed_split.csv')
    coverage=[]
    for column in joined:
        available=joined[column].notna() & joined[column].astype(str).str.strip().ne('')
        before=float(base[column].notna().mean()*100) if column in base else 0.0
        coverage.append({'column':column,'nonmissing_rows':int(available.sum()),
                         'coverage_percent':float(100*available.mean()),'midterm_coverage_percent':before,
                         'unique_nonmissing_values':int(joined.loc[available,column].nunique()),
                         'is_new_column':column not in original_columns})
    pd.DataFrame(coverage).to_csv(CLEAN/'feature_coverage.csv',index=False)
    split=pd.read_csv(CLEAN/'fixed_split.csv',dtype={'game_id':str})
    audit=joined.merge(split,on='game_id',validate='one_to_one')
    train_apps=set(audit.loc[audit.partition.eq('train'),'matched_steam_appid'].dropna())
    overlaps=audit.partition.eq('test') & audit.matched_steam_appid.isin(train_apps)
    audit.loc[overlaps,['game_id','title','matched_steam_appid','partition']].to_csv(CLEAN/'cross_partition_steam_overlap.csv',index=False)
    audit.loc[audit.main_story_hours.ge(100),['game_id','title','main_story_hours','igdb_game_modes','igdb_game_type','source_url','partition']].to_csv(CLEAN/'long_duration_label_audit.csv',index=False)
    report={'prepared_at_utc':utc_now(),'original_rows':len(base),'enriched_rows':len(joined),
            'original_columns':len(original_columns),'enriched_columns':len(joined.columns),
            'igdb_returned_rows':len(metadata),'steam_matched_rows':int(joined.matched_steam_appid.notna().sum()),
            'steam_metadata_rows':int(joined.steam_metadata_available.sum()),
            'usd_price_rows':int(joined.steam_regular_price_usd.notna().sum()),
            'steam_tag_rows':int(joined.steam_tags.notna().sum()),'steam_tag_dictionary_size':len(tag_names),
            'test_rows_with_steam_app_in_training':int(overlaps.sum()),
            'target_unchanged':True,'split_unchanged':True,
            'split_sha256':sha256(CLEAN/'fixed_split.csv'),'dataset_sha256':sha256(CLEAN/'games_enriched.csv'),
            'policy':'No target imputation, no invented prices/tags, no cohort expansion, no removal based on test errors. Raw missing stays missing; any predictor imputation occurs inside training folds.'}
    write_json(CLEAN/'enrichment_report.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return joined

if __name__=='__main__':
    prepare()
