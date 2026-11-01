"""Frozen-cohort, train-only tuning and matched feature ablation.

Descriptions and every completion-time field other than the target are excluded.
The already-published midterm test set is reused for stage comparison, not claimed
as a new independent benchmark. Select configurations before any new test scoring.
"""
from __future__ import annotations
import json
import platform
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, make_scorer
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits
from common import CLEAN, OUT, MIDTERM, SEED, sha256, utc_now, write_json
from features import GROUPS, input_columns, preprocessing

def clipped_mae(y,pred):
    return mean_absolute_error(y,np.maximum(0,np.asarray(pred,dtype=float)))

SCORER=make_scorer(clipped_mae,greater_is_better=False)

def load_data():
    df=pd.read_csv(CLEAN/'games_enriched.csv',dtype={'game_id':str})
    for field in ['primary_genre','igdb_game_type','steam_item_type']:
        df[field]=df[field].map(lambda x:str(x) if pd.notna(x) else np.nan).astype(object)
    split=pd.read_csv(CLEAN/'fixed_split.csv',dtype={'game_id':str})
    if len(split)!=len(df) or split.game_id.duplicated().any() or set(split.game_id)!=set(df.game_id):
        raise ValueError('Frozen cohort/split mismatch')
    df=df.merge(split,on='game_id',validate='one_to_one',how='left',sort=False)
    if not set(df.partition)=={'train','test'}:
        raise ValueError('Invalid partition labels')
    return df,df[df.partition.eq('train')].copy(),df[df.partition.eq('test')].copy()

def estimator(reg,feature_set='full',target='log'):
    pipe=Pipeline([('features',preprocessing(feature_set)),('model',reg)])
    if target=='raw': return pipe
    if target=='scaled_log':
        transformer=Pipeline([('log',FunctionTransformer(np.log1p,inverse_func=np.expm1,validate=True)),('scale',StandardScaler())])
        return TransformedTargetRegressor(regressor=pipe,transformer=transformer)
    return TransformedTargetRegressor(regressor=pipe,func=np.log1p,inverse_func=np.expm1)

def metrics(y,pred):
    y=np.asarray(y,dtype=float)
    p=np.maximum(0,np.asarray(pred,dtype=float))
    return {'test_MAE_hours':float(mean_absolute_error(y,p)),
            'test_RMSE_hours':float(np.sqrt(mean_squared_error(y,p))),
            'test_R2':float(r2_score(y,p)),
            'test_MedAE_hours':float(median_absolute_error(y,p)),
            'test_sMAPE_percent':float(np.mean(2*np.abs(y-p)/(y+p))*100)}

def bootstrap_improvement(y,reference,new,n=5000):
    """Paired game bootstrap, conditional on these fitted models and this cohort."""
    deltas=np.abs(np.asarray(y)-np.asarray(reference))-np.abs(np.asarray(y)-np.asarray(new))
    rng=np.random.default_rng(SEED)
    distribution=np.array([rng.choice(deltas,size=len(deltas),replace=True).mean() for _ in range(n)])
    return {'MAE_reduction_hours':float(deltas.mean()),'lower_95_hours':float(np.quantile(distribution,.025)),
            'upper_95_hours':float(np.quantile(distribution,.975)),'bootstrap_resamples':n,
            'interpretation':'Positive values favor new model. Conditional on this frozen cohort; not a causal estimate or fresh-test guarantee.'}

def classical_candidates():
    """Week 3-4 references use the original features and the same fixed folds."""
    return {
        'Linear regression original features':(estimator(LinearRegression(),'midterm'),None,'midterm'),
        'Decision tree original features':(estimator(DecisionTreeRegressor(max_depth=10,min_samples_leaf=10,random_state=SEED),'midterm'),None,'midterm'),
        'KNN original features':(estimator(KNeighborsRegressor(n_neighbors=25,weights='distance',n_jobs=2),'midterm'),None,'midterm'),
    }

def write_model_specifications(comparison):
    reasons={
        'Median baseline':('Training-label median','Minimum-complexity reference for absolute error.'),
        'Linear regression original features':('OLS; log1p target','Tests whether unregularized linear relations are sufficient.'),
        'Decision tree original features':('max_depth=10; min_samples_leaf=10; seed=42','Captures nonlinear thresholds while limiting tree complexity.'),
        'KNN original features':('25 neighbours; distance weights; numeric scaling','Tests local similarity. Scaling prevents release year from dominating distance.'),
        'Random forest':('250 trees; min_samples_leaf=3; max_features=0.7; bootstrap=True; seed=42','Averages bootstrap-trained trees to reduce single-tree variance.'),
        'Tuned gradient boosting':('250 iterations; learning_rate=0.07; absolute_error; leaf/min-leaf/L2 grid','Fits the primary hour-scale absolute-error objective with tuned complexity.'),
        'Tuned neural network':('Hidden layers (32,) or (64,32); alpha=0.01 or 0.1; max_iter=350; early stopping','Tests nonlinear combinations of structured inputs with regularization.'),
    }
    records=[]
    for row in comparison.to_dict('records'):
        name=row['model']
        if name.startswith('Ridge'):
            parameters,reason='alpha=20; solver=lsqr; log1p target','Regularized linear reference; identical settings isolate feature-set effects.'
        elif name.startswith('ExtraTrees'):
            parameters,reason='200 trees; min_samples_leaf=3; max_features=0.8; seed=42','Randomized tree ensemble; identical settings isolate feature-set effects.'
        else:
            parameters,reason=reasons[name]
        records.append({'model':name,'feature_set':row['feature_set'],'parameters':parameters,
                        'selected_search_parameters':row.get('best_parameters','{}'),'reason':reason})
    pd.DataFrame(records).to_csv(OUT/'model_specifications.csv',index=False)

def append_classical_baselines():
    """Add missing references without rerunning or reselecting the ensembles."""
    _,train,test=load_data()
    manifest=json.loads((OUT/'experiment_manifest.json').read_text())
    assert manifest['dataset_sha256']==sha256(CLEAN/'games_enriched.csv')
    assert manifest['split_sha256']==sha256(CLEAN/'fixed_split.csv')
    comparison=pd.read_csv(OUT/'model_comparison.csv')
    predictions=pd.read_csv(OUT/'test_predictions.csv',dtype={'game_id':str})
    assert predictions.game_id.tolist()==test.game_id.tolist()
    pending={k:v for k,v in classical_candidates().items() if k not in set(comparison.model)}
    if not pending:
        write_model_specifications(comparison)
        return comparison
    write_json(OUT/'classical_baseline_plan.json',{
        'recorded_at_utc':utc_now(),'models':list(pending),'feature_set':'midterm',
        'parameters':{'linear':'OLS','tree':{'max_depth':10,'min_samples_leaf':10},
                      'knn':{'n_neighbors':25,'weights':'distance'}},
        'cv':'Same seeded training-only 3-fold KFold as the existing comparison.',
        'purpose':'Add Week 3-4 reference models required by the supplied rubric.',
        'selection_policy':'Original-feature references do not replace the locked expanded-model selection.',
        'test_exposure':'Test outcomes were already observed. This is a supplementary comparison.'})
    x=train[input_columns('full')]
    y=train.main_story_hours.to_numpy()
    folds=list(KFold(3,shuffle=True,random_state=SEED).split(x))
    records=[]
    fold_records=[]
    fitted={}
    for name,(model,_,feature_set) in pending.items():
        start=time.monotonic()
        print('Training reference model:',name,flush=True)
        with threadpool_limits(limits=2):
            scores=-cross_validate(model,x,y,cv=folds,scoring=SCORER,error_score='raise')['test_score']
            model.fit(x,y)
        fitted[name]=model
        records.append({'model':name,'feature_set':feature_set,'train_cv_MAE_hours':float(scores.mean()),
                        'train_cv_MAE_std_hours':float(scores.std(ddof=1)),
                        'training_seconds':round(time.monotonic()-start,2),'tuned':False,'best_parameters':'{}'})
        fold_records.extend({'model':name,'fold':i+1,'validation_MAE_hours':float(v)} for i,v in enumerate(scores))
        print('CV MAE:',round(scores.mean(),3),'hours',flush=True)
    pd.DataFrame(records).to_csv(OUT/'classical_baseline_cv.csv',index=False)
    for record in records:
        pred=np.maximum(0,fitted[record['model']].predict(test[input_columns('full')]))
        assert np.isfinite(pred).all()
        record.update(metrics(test.main_story_hours,pred))
        predictions[record['model']]=pred
    comparison=pd.concat([comparison,pd.DataFrame(records)],ignore_index=True)
    comparison.to_csv(OUT/'model_comparison.csv',index=False)
    predictions.to_csv(OUT/'test_predictions.csv',index=False)
    fold_scores=pd.read_csv(OUT/'cv_fold_scores.csv')
    pd.concat([fold_scores,pd.DataFrame(fold_records)],ignore_index=True).to_csv(OUT/'cv_fold_scores.csv',index=False)
    manifest['comparison_updated_at_utc']=utc_now()
    manifest['comparison_models']=comparison.model.tolist()
    manifest['classical_baseline_plan_sha256']=sha256(OUT/'classical_baseline_plan.json')
    write_json(OUT/'experiment_manifest.json',manifest)
    write_model_specifications(comparison)
    return comparison

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'search').mkdir(exist_ok=True)
    df,train,test=load_data()
    x=train[input_columns('full')]
    y=train.main_story_hours.to_numpy()
    xtest=test[input_columns('full')]
    ytest=test.main_story_hours.to_numpy()
    folds=list(KFold(3,shuffle=True,random_state=SEED).split(x))
    # A predeclared finite search. All ablations keep the estimator constant.
    candidates={
        'Median baseline':(estimator(DummyRegressor(strategy='median'),'midterm','raw'),None,'midterm'),
        **classical_candidates(),
        'Ridge original features':(estimator(Ridge(alpha=20,solver='lsqr'),'midterm'),None,'midterm'),
        'Ridge expanded IGDB':(estimator(Ridge(alpha=20,solver='lsqr'),'igdb'),None,'igdb'),
        'Ridge IGDB + Steam':(estimator(Ridge(alpha=20,solver='lsqr')),None,'full'),
        'ExtraTrees original features':(estimator(ExtraTreesRegressor(n_estimators=200,min_samples_leaf=3,max_features=.8,random_state=SEED,n_jobs=2),'midterm'),None,'midterm'),
        'ExtraTrees expanded IGDB':(estimator(ExtraTreesRegressor(n_estimators=200,min_samples_leaf=3,max_features=.8,random_state=SEED,n_jobs=2),'igdb'),None,'igdb'),
        'ExtraTrees IGDB + Steam':(estimator(ExtraTreesRegressor(n_estimators=200,min_samples_leaf=3,max_features=.8,random_state=SEED,n_jobs=2)),None,'full'),
        'Random forest':(estimator(RandomForestRegressor(n_estimators=250,min_samples_leaf=3,max_features=.7,random_state=SEED,n_jobs=2)),None,'full'),
        'Tuned gradient boosting':(estimator(HistGradientBoostingRegressor(loss='absolute_error',max_iter=250,learning_rate=.07,random_state=SEED),'full','raw'),{
            'model__max_leaf_nodes':[15,31], 'model__min_samples_leaf':[20,50], 'model__l2_regularization':[1,10]},'full'),
        'Tuned neural network':(estimator(MLPRegressor(max_iter=350,early_stopping=True,validation_fraction=.15,n_iter_no_change=20,random_state=SEED),'full','scaled_log'),{
            'regressor__model__hidden_layer_sizes':[(32,),(64,32)],
            'regressor__model__alpha':[.01,.1]},'full'),
    }
    write_json(OUT/'experiment_plan.json',{
        'created_before_test_evaluation_utc':utc_now(),'seed':SEED,'cv':'3-fold shuffled KFold on training IDs only',
        'metric':'MAE in original hours','candidate_order':list(candidates),
        'grids':{name:grid for name,(_,grid,_) in candidates.items() if grid},
        'feature_groups':{k:input_columns(k) for k in GROUPS},
        'ablation':'Ridge and ExtraTrees unchanged across original / IGDB / IGDB + Steam.',
        'target':'Frozen midterm main_story_hours, no clipping, no label refresh or imputation.',
        'test_note':'Midterm test outcomes have already been published. This is a stage comparison, not a new independent test.'})
    results=[]
    fitted={}
    cv_details=[]
    for name,(model,grid,feature_set) in candidates.items():
        start=time.monotonic()
        print('TRAIN CV:',name,flush=True)
        with threadpool_limits(limits=2):
            if grid:
                search=GridSearchCV(model,grid,scoring=SCORER,cv=folds,n_jobs=1,error_score='raise',refit=True,return_train_score=True)
                search.fit(x,y)
                model=search.best_estimator_
                params=search.best_params_
                scores=np.array([-search.cv_results_[f'split{i}_test_score'][search.best_index_] for i in range(3)])
                pd.DataFrame(search.cv_results_).to_csv(OUT/'search'/('gradient_boosting.csv' if 'boosting' in name else 'neural_network.csv'),index=False)
            else:
                cv=cross_validate(model,x,y,cv=folds,scoring=SCORER,error_score='raise',n_jobs=1)
                scores=-cv['test_score']
                model.fit(x,y)
                params={}
        fitted[name]=model
        results.append({'model':name,'feature_set':feature_set,'train_cv_MAE_hours':float(scores.mean()),
                        'train_cv_MAE_std_hours':float(scores.std(ddof=1)),
                        'training_seconds':round(time.monotonic()-start,2),'tuned':bool(grid),
                        'best_parameters':json.dumps(params)})
        cv_details.extend({'model':name,'fold':i+1,'validation_MAE_hours':float(score)} for i,score in enumerate(scores))
        pd.DataFrame(results).to_csv(OUT/'training_cv_progress.csv',index=False)
        print('CV MAE:',round(scores.mean(),3),'+/-',round(scores.std(ddof=1),3),'hours',flush=True)
    comparison=pd.DataFrame(results)
    eligible=comparison[comparison.feature_set.eq('full')]
    winner=eligible.sort_values('train_cv_MAE_hours',kind='stable').iloc[0].model
    # Persist selection before predicting any new test results.
    write_json(OUT/'model_selection.json',{'selected_at_utc':utc_now(),'winner':winner,
        'rule':'Lowest mean training CV MAE among expanded full-feature candidates',
        'cv_MAE_hours':float(eligible.set_index('model').loc[winner,'train_cv_MAE_hours']),
        'plan_sha256':sha256(OUT/'experiment_plan.json')})
    print('MODEL SELECTED BEFORE TEST EVALUATION:',winner,flush=True)
    predictions=test[['game_id','title','primary_genre','main_story_hours','igdb_single_player','steam_metadata_available','matched_steam_appid']].copy().rename(columns={'main_story_hours':'actual_hours'})
    all_predictions={}
    for name,model in fitted.items():
        pred=np.maximum(0,model.predict(xtest))
        if not np.isfinite(pred).all(): raise ValueError('Nonfinite predictions')
        all_predictions[name]=pred
        row_index=comparison.index[comparison.model.eq(name)][0]
        for key,value in metrics(ytest,pred).items(): comparison.loc[row_index,key]=value
        predictions[name]=pred
    selected=fitted[winner]
    predictions['predicted_hours']=all_predictions[winner]
    predictions['absolute_error']=(predictions.predicted_hours-predictions.actual_hours).abs()
    predictions['residual_hours']=predictions.actual_hours-predictions.predicted_hours
    predictions.to_csv(OUT/'test_predictions.csv',index=False)
    comparison.to_csv(OUT/'model_comparison.csv',index=False)
    pd.DataFrame(cv_details).to_csv(OUT/'cv_fold_scores.csv',index=False)
    write_model_specifications(comparison)
    joblib.dump(selected,OUT/'best_model.joblib',compress=3)
    neural=fitted['Tuned neural network'].regressor_.named_steps['model']
    write_json(OUT/'neural_network_diagnostics.json',{
        'hidden_layer_sizes':list(neural.hidden_layer_sizes), 'alpha':float(neural.alpha),
        'iterations':int(neural.n_iter_), 'max_iter':int(neural.max_iter),
        'early_stopping':bool(neural.early_stopping),'validation_fraction':float(neural.validation_fraction),
        'loss_curve':[float(v) for v in neural.loss_curve_],
        'internal_validation_R2_curve':[float(v) for v in neural.validation_scores_],
        'target_transform':'log1p then train-fitted StandardScaler; inverse before hour metrics',
        'note':'Loss and internal validation R2 are in standardized log-target space, not hours.'})
    paired={
        'selected_vs_median':bootstrap_improvement(ytest,all_predictions['Median baseline'],all_predictions[winner]),
        'selected_vs_retrained_original_ridge':bootstrap_improvement(ytest,all_predictions['Ridge original features'],all_predictions[winner]),
        'ridge_igdb_gain':bootstrap_improvement(ytest,all_predictions['Ridge original features'],all_predictions['Ridge expanded IGDB']),
        'ridge_steam_gain':bootstrap_improvement(ytest,all_predictions['Ridge expanded IGDB'],all_predictions['Ridge IGDB + Steam']),
        'extratrees_igdb_gain':bootstrap_improvement(ytest,all_predictions['ExtraTrees original features'],all_predictions['ExtraTrees expanded IGDB']),
        'extratrees_steam_gain':bootstrap_improvement(ytest,all_predictions['ExtraTrees expanded IGDB'],all_predictions['ExtraTrees IGDB + Steam']),
    }
    old=pd.read_csv(MIDTERM/'outputs/test_predictions.csv',dtype={'game_id':str})
    old_pred=old.set_index('game_id').reindex(test.game_id)
    if old_pred.predicted_hours.notna().all():
        paired['selected_vs_submitted_midterm_ridge']=bootstrap_improvement(ytest,old_pred.predicted_hours.to_numpy(),all_predictions[winner])
    write_json(OUT/'paired_bootstrap.json',paired)
    train_apps=set(train.matched_steam_appid.dropna())
    unseen=~test.matched_steam_appid.isin(train_apps)
    subsets=[]
    for name,mask in [('All original test games',np.ones(len(test),dtype=bool)),
                      ('No Steam app overlap with training',unseen.to_numpy()),
                      ('IGDB lists single-player',test.igdb_single_player.eq(1).to_numpy()),
                      ('Steam metadata available',test.steam_metadata_available.eq(1).to_numpy()),
                      ('Steam metadata unavailable',test.steam_metadata_available.eq(0).to_numpy())]:
        if mask.sum(): subsets.append({'subset':name,'rows':int(mask.sum()),**metrics(ytest[mask],all_predictions[winner][mask])})
    pd.DataFrame(subsets).to_csv(OUT/'sensitivity_subsets.csv',index=False)
    predictions['duration_band']=pd.cut(predictions.actual_hours,[0,5,20,50,100,np.inf],labels=['0-5 h','>5-20 h','>20-50 h','>50-100 h','>100 h'])
    predictions.groupby('duration_band',observed=True).absolute_error.agg(['count','mean','median']).to_csv(OUT/'error_by_duration.csv')
    predictions.groupby('primary_genre',observed=True).absolute_error.agg(['count','mean','median']).query('count >= 10').sort_values('mean',ascending=False).to_csv(OUT/'error_by_genre.csv')
    predictions.nlargest(15,'absolute_error')[['game_id','title','actual_hours','predicted_hours','absolute_error']].to_csv(OUT/'largest_errors.csv',index=False)
    # Importance uses a validation fold INSIDE the training partition, never test outcomes.
    fit_idx,val_idx=folds[0]
    with threadpool_limits(limits=2):
        importance_model=clone(selected).fit(x.iloc[fit_idx],y[fit_idx])
        importance=permutation_importance(importance_model,x.iloc[val_idx],y[val_idx],scoring=SCORER,n_repeats=3,random_state=SEED,n_jobs=1)
        grouped_scores=None
        groups=train.matched_steam_appid.map(lambda v:f'steam_{int(v)}' if pd.notna(v) else None)
        groups=groups.fillna(train.game_id.map(lambda v:'igdb_'+v))
        grouped_cv=GroupKFold(3)
        grouped_scores=-cross_validate(clone(selected),x,y,groups=groups,cv=grouped_cv,scoring=SCORER,error_score='raise')['test_score']
    pd.DataFrame({'feature':x.columns,'validation_MAE_increase_hours':importance.importances_mean,
                  'std_hours':importance.importances_std}).sort_values('validation_MAE_increase_hours',ascending=False).to_csv(OUT/'permutation_importance.csv',index=False)
    manifest={'completed_at_utc':utc_now(),'seed':SEED,'rows':len(df),'training_rows':len(train),'test_rows':len(test),
              'cv_folds':3,'best_cv_model':winner,'selection':'Training CV MAE only; locked before test scoring.',
              'dataset_sha256':sha256(CLEAN/'games_enriched.csv'),'split_sha256':sha256(CLEAN/'fixed_split.csv'),
              'plan_sha256':sha256(OUT/'experiment_plan.json'),'target_unchanged':True,'descriptions_used':False,
              'input_columns':input_columns('full'),'encoded_features':int(selected.regressor_.named_steps['features'].transform(x.iloc[:1]).shape[1] if hasattr(selected,'regressor_') else selected.named_steps['features'].transform(x.iloc[:1]).shape[1]),
              'grouped_cv_MAE_mean_hours':float(np.mean(grouped_scores)),
              'grouped_cv_MAE_std_hours':float(np.std(grouped_scores,ddof=1)),
              'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'scikit_learn':sklearn.__version__,
              'caveats':['The frozen test was already visible at midterm.','Search CV scores are selection estimates, not nested CV.',
                         'Current metadata with October labels gives retrospective catalog prediction, not launch-day prediction.',
                         'Long-duration labels remain in the full-cohort evaluation.','Missing raw predictors stay missing; model-only imputation is fitted in folds.']}
    write_json(OUT/'experiment_manifest.json',manifest)
    print(comparison[['model','train_cv_MAE_hours','test_MAE_hours','test_R2']].to_string(index=False),flush=True)
    return comparison

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--add-classical-baselines',action='store_true')
    args=parser.parse_args()
    append_classical_baselines() if args.add_classical_baselines else run()
