"""Verify source integrity, independent score recomputation and executed artifacts."""
import json
import zipfile
import re
from urllib.parse import unquote
import joblib
import numpy as np
import pandas as pd
from pypdf import PdfReader
from common import ROOT, RAW, CLEAN, OUT, DOCS, MIDTERM, sha256, utc_now, write_json
from train import load_data, metrics
from features import input_columns, MultiHot

def run(artifacts=True):
    raw_manifest=json.loads((RAW/'collection_manifest.json').read_text())
    for name,entry in raw_manifest['files'].items():
        assert sha256(RAW/name)==entry['sha256'],f'Raw snapshot changed: {name}'
    original=ROOT.parent/'Midterm_Syrym_Shadiyarbek_SE2424'
    originals_checked=0
    if original.exists():
        for path in original.rglob('*'):
            if path.is_file():
                assert sha256(path)==sha256(MIDTERM/path.relative_to(original)),'Original submission changed'
                originals_checked+=1
    df,train,test=load_data()
    assert len(df)==5229 and len(train)==4183 and len(test)==1046
    assert sha256(CLEAN/'fixed_split.csv')==sha256(MIDTERM/'data/clean/fixed_split.csv')
    old=pd.read_csv(MIDTERM/'data/clean/games_clean.csv',dtype={'game_id':str})
    assert old.game_id.tolist()==df.game_id.tolist()
    np.testing.assert_array_equal(old.main_story_hours,df.main_story_hours)
    comparison=pd.read_csv(OUT/'model_comparison.csv')
    required={'Median baseline','Linear regression original features','Ridge original features',
              'Decision tree original features','KNN original features','Random forest',
              'Tuned gradient boosting','Tuned neural network'}
    assert required.issubset(set(comparison.model)), 'A course-required model is missing'
    raw_frame=pd.read_csv(CLEAN/'games_enriched.csv')
    dictionary=pd.read_csv(CLEAN/'data_dictionary.csv')
    assert set(dictionary.field)==set(raw_frame.columns), 'Incomplete field dictionary'
    assert set(dictionary.loc[dictionary.model_input,'field'])==set(input_columns('full'))
    assert raw_frame.game_id.nunique()>=1000 and raw_frame.description.fillna('').str.strip().ne('').all()
    source_check=json.loads((CLEAN/'source_verification.json').read_text())
    assert source_check['cohort_and_labels_reproduced']
    assert source_check['price_rows_verified_against_raw_cents']==int(raw_frame.steam_regular_price_usd.notna().sum())
    assert source_check['tag_rows_verified_against_raw_ids']==int(raw_frame.steam_tags.notna().sum())
    assert not {'description','main_story_hours','title','game_id','hastily','normally','completely'} & set(input_columns('full'))
    assert not set(MultiHot.tokens('Short|Long|Short game|Long playtime'))
    predictions=pd.read_csv(OUT/'test_predictions.csv',dtype={'game_id':str})
    assert predictions.game_id.tolist()==test.game_id.tolist()
    np.testing.assert_array_equal(predictions.actual_hours,test.main_story_hours)
    folds=pd.read_csv(OUT/'cv_fold_scores.csv')
    for row in comparison.itertuples(index=False):
        recalculated=metrics(predictions.actual_hours,predictions[row.model])
        for key,value in recalculated.items():
            assert np.isclose(getattr(row,key),value,rtol=1e-10,atol=1e-10),f'Metric mismatch: {row.model} {key}'
        scores=folds.loc[folds.model.eq(row.model),'validation_MAE_hours']
        assert len(scores)==3
        assert np.isclose(scores.mean(),row.train_cv_MAE_hours)
        assert np.isclose(scores.std(ddof=1),row.train_cv_MAE_std_hours)
    selection=json.loads((OUT/'model_selection.json').read_text())
    expected=comparison[comparison.feature_set.eq('full')].sort_values('train_cv_MAE_hours',kind='stable').iloc[0].model
    assert selection['winner']==expected
    manifest=json.loads((OUT/'experiment_manifest.json').read_text())
    assert manifest['dataset_sha256']==sha256(CLEAN/'games_enriched.csv')
    assert manifest['split_sha256']==sha256(CLEAN/'fixed_split.csv')
    assert manifest['plan_sha256']==sha256(OUT/'experiment_plan.json')
    if 'classical_baseline_plan_sha256' in manifest:
        assert manifest['classical_baseline_plan_sha256']==sha256(OUT/'classical_baseline_plan.json')
    model=joblib.load(OUT/'best_model.joblib')
    regenerated=np.maximum(0,model.predict(test[input_columns('full')]))
    np.testing.assert_allclose(regenerated,predictions.predicted_hours,rtol=1e-10,atol=1e-10)
    report={'verified_at_utc':utc_now(),'raw_manifest_files_verified':len(raw_manifest['files']),
            'original_files_preserved':originals_checked,'cohort_rows':len(df),'test_predictions_verified':len(test),
            'model_metric_rows_recomputed':len(comparison),'winner_matches_CV_selection':True,
            'saved_model_predictions_match':True,'target_and_split_preserved':True}
    report['complete_field_dictionary']=True
    report['source_price_and_tag_checks']=True
    report['course_models_present']=True
    if artifacts:
        notebook=json.loads((ROOT/'notebooks/Video_Game_Endterm.ipynb').read_text(encoding='utf-8'))
        code=[c for c in notebook['cells'] if c['cell_type']=='code']
        assert code and all(c['execution_count'] is not None for c in code)
        assert not [o for c in code for o in c['outputs'] if o['output_type']=='error']
        for filename in ['ENDTERM_REPORT.md','DATA_CARD.md','CONTRIBUTIONS.md','DEFENSE_GUIDE.md','CRITERIA_CHECKLIST.md']:
            assert (DOCS/filename).is_file(), f'Missing documentation: {filename}'
        assert 'OpenAI Codex' in (DOCS/'CONTRIBUTIONS.md').read_text(encoding='utf-8')
        missing=[]
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',(ROOT/'README.md').read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'): continue
            path=unquote(target.strip('<>').split('#',1)[0])
            if not (ROOT/path).exists(): missing.append(path)
        assert not missing, f'Broken README links: {missing}'
        report['documentation_links_valid']=True
        report['AI_use_disclosure_present']=True
        report['notebook_executed_code_cells']=len(code)
        report['report_pages']=len(PdfReader(DOCS/'Endterm_Report.pdf').pages)
        report['slide_pdf_pages']=len(PdfReader(DOCS/'Endterm_Defense.pdf').pages)
        assert report['slide_pdf_pages']==9
        with zipfile.ZipFile(DOCS/'Endterm_Defense.pptx') as archive:
            names=archive.namelist()
            slides=[n for n in names if n.startswith('ppt/slides/slide') and n.endswith('.xml')]
            charts=[n for n in names if '/charts/' in n and n.rsplit('/',1)[-1].startswith('chart') and n.endswith('.xml')]
            assert len(slides)==9 and len(charts)>=5
            workbooks=[n for n in names if n.startswith('ppt/embeddings/') and n.endswith('.xlsx')]
            assert len(workbooks)>=5, 'Editable chart workbooks are missing'
            report['editable_slide_count']=len(slides)
            report['native_chart_count']=len(charts)
            report['embedded_chart_workbooks']=len(workbooks)
        report['deliverable_sha256']={p.relative_to(ROOT).as_posix():sha256(p) for p in [ROOT/'notebooks/Video_Game_Endterm.ipynb',DOCS/'Endterm_Report.pdf',DOCS/'Endterm_Defense.pptx',DOCS/'Endterm_Defense.pdf']}
    write_json(OUT/'validation_summary.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__':
    run()
