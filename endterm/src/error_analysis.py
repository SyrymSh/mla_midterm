"""Descriptive errors across models and train-fitted clusters, without selection."""
import json
import numpy as np
import pandas as pd
from common import OUT, MIDTERM, CLEAN, utc_now, write_json


def summarize(frame, groups):
    return frame.groupby(groups, observed=True).agg(
        rows=('absolute_error','size'), MAE_hours=('absolute_error','mean'),
        median_AE_hours=('absolute_error','median'), bias_hours=('residual','mean')).reset_index()


def run():
    prediction=pd.read_csv(OUT/'test_predictions.csv',dtype={'game_id':str})
    comparison=pd.read_csv(OUT/'model_comparison.csv')
    selected=json.loads((OUT/'model_selection.json').read_text())['winner']
    old=pd.read_csv(MIDTERM/'outputs/test_predictions.csv',dtype={'game_id':str})
    prediction['Archived Midterm Ridge']=old.set_index('game_id').predicted_hours.reindex(prediction.game_id).to_numpy()
    assert prediction['Archived Midterm Ridge'].notna().all()
    names=comparison.model.tolist()+['Archived Midterm Ridge']
    prediction['duration_band']=pd.cut(prediction.actual_hours,[0,5,20,50,100,np.inf],
        labels=['0-5 h','>5-20 h','>20-50 h','>50-100 h','>100 h'])
    identifiers=['game_id','title','actual_hours','primary_genre','duration_band']
    long=prediction[identifiers+names].melt(id_vars=identifiers,
        value_vars=names,var_name='model',value_name='predicted_hours')
    long['residual']=long.actual_hours-long.predicted_hours
    long['absolute_error']=long.residual.abs()
    long.rename(columns={'residual':'residual_hours'}).to_csv(OUT/'all_model_residuals.csv',index=False)
    summarize(long,['model','duration_band']).to_csv(OUT/'all_model_error_by_duration.csv',index=False)
    genre=summarize(long,['model','primary_genre'])
    genre.loc[genre.rows.ge(10)].to_csv(OUT/'all_model_error_by_genre.csv',index=False)
    examples=[]
    ridge_error=(prediction.actual_hours-prediction['Archived Midterm Ridge']).abs()
    for name in names:
        e=(prediction.actual_hours-prediction[name]).abs()
        choices=[('largest error',e.idxmax())]
        ordinary=prediction.actual_hours.between(5,100)
        choices.append(('largest error among 5-100-hour games',e.loc[ordinary].idxmax()))
        if name!='Archived Midterm Ridge':
            gain=ridge_error-e
            choices.extend([('largest gain over archived Ridge',gain.idxmax()),
                            ('largest loss against archived Ridge',gain.idxmin())])
        for reason,i in choices:
            row=prediction.loc[i]
            examples.append({'model':name,'selection':reason,'game_id':row.game_id,'title':row.title,
                             'actual_hours':row.actual_hours,'predicted_hours':row[name],
                             'absolute_error':e.loc[i]})
    pd.DataFrame(examples).to_csv(OUT/'all_model_error_examples.csv',index=False)
    clusters=pd.read_csv(OUT/'cluster_assignments.csv',dtype={'game_id':str})
    clusters=clusters.loc[clusters.partition.eq('test'),['game_id','cluster']]
    grouped=long.merge(clusters,on='game_id',validate='many_to_one')
    assert len(grouped)==len(long)
    summarize(grouped,['cluster','model']).to_csv(OUT/'cluster_error_summary.csv',index=False)
    cases=prediction.merge(clusters,on='game_id',validate='one_to_one')
    metadata=pd.read_csv(CLEAN/'games_enriched.csv',dtype={'game_id':str})
    cases=cases.merge(metadata[['game_id','igdb_themes','igdb_game_modes']],on='game_id',validate='one_to_one')
    cases['midterm_absolute_error']=(cases.actual_hours-cases['Archived Midterm Ridge']).abs()
    cases['selected_absolute_error']=(cases.actual_hours-cases[selected]).abs()
    # One largest earlier-model failure and one near-median-duration example per cluster.
    chosen=[]
    for label,group in cases.groupby('cluster'):
        for reason,i in [('largest archived Ridge error',group.midterm_absolute_error.idxmax()),
                         ('near cluster median duration',(group.actual_hours-group.actual_hours.median()).abs().idxmin())]:
            row=cases.loc[i]
            chosen.append({'cluster':int(label),'selection':reason,'game_id':row.game_id,'title':row.title,
                           'primary_genre':row.primary_genre,'igdb_themes':row.igdb_themes,
                           'igdb_game_modes':row.igdb_game_modes,'igdb_single_player':row.igdb_single_player,
                           'actual_hours':row.actual_hours,'Archived Midterm Ridge':row['Archived Midterm Ridge'],
                           'selected_model_prediction':row[selected],
                           'midterm_absolute_error':row.midterm_absolute_error,
                           'selected_absolute_error':row.selected_absolute_error})
    pd.DataFrame(chosen).to_csv(OUT/'cluster_error_examples.csv',index=False)
    neural=json.loads((OUT/'neural_network_diagnostics.json').read_text())
    peak=int(np.argmax(neural['internal_validation_R2_curve']))
    neural.update(best_validation_iteration=peak+1,
                  best_internal_validation_R2=neural['internal_validation_R2_curve'][peak],
                  final_internal_validation_R2=neural['internal_validation_R2_curve'][-1],
                  loss_at_best_validation_iteration=neural['loss_curve'][peak],
                  final_training_loss=neural['loss_curve'][-1],
                  interpretation='After the validation peak, decreasing training loss with weaker validation indicates overfitting. Early stopping restores the peak weights; reaching a plateau does not establish a globally optimal network.')
    write_json(OUT/'neural_network_diagnostics.json',neural)
    write_json(OUT/'error_analysis_manifest.json',{
        'completed_at_utc':utc_now(),'models':names,'selected_model':selected,'test_rows':len(prediction),
        'cluster_fit':'Predictors from training partition only; no duration used for fitting or k selection.',
        'purpose':'Held-out descriptive error analysis. Does not select features, clusters, settings or winner.',
        'bias_definition':'Actual minus predicted; positive values indicate underprediction.',
        'examples':'Deterministic largest errors and gains/losses, plus near-median-duration cluster examples.',
        'midterm_reference':'Exact archived Midterm Ridge predictions, separate from current original-feature refit.'})
    print('Cross-model and cluster error analyses saved.',flush=True)
    return grouped


if __name__=='__main__':
    run()
