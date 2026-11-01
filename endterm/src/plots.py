"""Plot feature coverage, model results, residuals and clustering diagnostics."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import CLEAN, OUT
from train import load_data

TEAL='#176B68'
GOLD='#B66A3D'
GRAY='#7E8C98'

def save(name,fig):
    folder=OUT/'figures'
    folder.mkdir(exist_ok=True)
    fig.savefig(folder/(name+'.png'),dpi=170,bbox_inches='tight')
    plt.close(fig)

def run():
    plt.rcParams.update({'figure.figsize':(9,5),'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.right':False,'axes.spines.top':False,'axes.titleweight':'bold',
                         'figure.facecolor':'white','axes.facecolor':'white','axes.labelcolor':'#263744'})
    _,train,_=load_data()
    comparison=pd.read_csv(OUT/'model_comparison.csv')
    pred=pd.read_csv(OUT/'test_predictions.csv')
    coverage=pd.read_csv(CLEAN/'feature_coverage.csv').set_index('column')
    manifest=json.loads((OUT/'experiment_manifest.json').read_text())
    cluster_info=json.loads((OUT/'clustering_manifest.json').read_text())
    nn=json.loads((OUT/'neural_network_diagnostics.json').read_text())
    selection=pd.read_csv(OUT/'cluster_selection.csv')
    assignments=pd.read_csv(OUT/'cluster_assignments.csv')
    labels=['Release year','Genres','Gameplay themes','Gameplay keywords','Player perspectives','Game modes','Steam user tags','USD regular price']
    fields=['release_year','genres_text','igdb_themes','igdb_keywords','igdb_player_perspectives','igdb_game_modes','steam_tags','steam_regular_price_usd']
    before=[float(coverage.loc[c,'midterm_coverage_percent']) for c in fields]
    after=[float(coverage.loc[c,'coverage_percent']) for c in fields]
    fig,ax=plt.subplots(figsize=(10,5.5)); yy=np.arange(len(labels))
    ax.barh(yy-.17,before,height=.32,label='Midterm',color=GRAY)
    ax.barh(yy+.17,after,height=.32,label='Endterm',color=TEAL)
    ax.set(yticks=yy,yticklabels=labels,xlabel='Rows with observed values (%)',xlim=(0,109),title='More observed feature types on the same 5,229 games')
    for i,v in enumerate(after): ax.text(v+1,i+.17,f'{v:.1f}%',va='center',fontsize=9)
    ax.invert_yaxis(); ax.legend(loc='lower right');fig.tight_layout();save('feature_coverage',fig)
    fig,ax=plt.subplots();ax.hist(train.main_story_hours,bins=np.geomspace(.1,1000,45),color=TEAL,edgecolor='white')
    ax.set(xscale='log',xlabel='Main-story hours (logarithmic axis)',ylabel='Training games',title='The target has a long right tail')
    ax.axvline(train.main_story_hours.median(),color=GOLD,label=f'Training median {train.main_story_hours.median():.1f} h');ax.legend();save('target_distribution',fig)
    genres=train.groupby('primary_genre').main_story_hours.agg(['size','median']).query('size >= 30').sort_values('median').tail(12)
    fig,ax=plt.subplots();ax.barh(genres.index,genres['median'],color=TEAL);ax.set(xlabel='Median main-story hours in training data',title='Genre is informative, but broad');fig.tight_layout();save('genre_duration',fig)
    known=train.steam_regular_price_usd.notna()
    fig,ax=plt.subplots();ax.scatter(train.loc[known,'steam_regular_price_usd'],train.loc[known,'main_story_hours'],s=12,alpha=.35,color=TEAL)
    ax.set(xscale='symlog',yscale='log',xlabel='Observed US regular price (USD, symlog axis)',ylabel='Main-story hours (log axis)',title=f'Price and duration vary together, with large spread (n={int(known.sum()):,})');save('price_duration',fig)
    names=comparison.model.str.replace('original features','original',regex=False).str.replace('expanded IGDB','IGDB',regex=False)
    order=np.arange(len(comparison))
    fig,ax=plt.subplots(figsize=(11,7.5));ax.barh(order-.17,comparison.train_cv_MAE_hours,height=.32,xerr=comparison.train_cv_MAE_std_hours,color=GRAY,label='Train CV mean and SD',capsize=2)
    ax.barh(order+.17,comparison.test_MAE_hours,height=.32,color=TEAL,label='Frozen test')
    ax.set(yticks=order,yticklabels=names,xlabel='MAE in hours (lower is better)',title='Model selection uses training CV');ax.invert_yaxis();ax.legend();fig.tight_layout();save('model_comparison',fig)
    index=comparison.set_index('model')
    ablation={model:[float(index.loc[f'{model} {suffix}','test_MAE_hours']) for suffix in ['original features','expanded IGDB','IGDB + Steam']] for model in ['Ridge','ExtraTrees']}
    fig,ax=plt.subplots();xx=np.arange(3)
    for offset,(name,values) in zip([-.17,.17],ablation.items()):
        ax.bar(xx+offset,values,width=.32,label=name,color=TEAL if name=='Ridge' else GOLD)
        for i,v in enumerate(values):ax.text(i+offset,v+.12,f'{v:.2f}',ha='center',fontsize=10)
    ax.set(xticks=xx,xticklabels=['Original features','Expanded IGDB','IGDB + Steam'],ylabel='Frozen-test MAE (hours)',title='Feature ablation keeps the estimator fixed',ylim=(0,max(max(v) for v in ablation.values())*1.18));ax.legend();save('feature_ablation',fig)
    fig,ax=plt.subplots();ax.scatter(pred.actual_hours,pred.predicted_hours.clip(lower=.1),s=13,alpha=.4,color=TEAL)
    ax.plot([.1,1000],[.1,1000],color=GOLD,ls='--',label='Perfect prediction');ax.set(xscale='log',yscale='log',xlim=(.1,1100),ylim=(.1,1100),xlabel='Actual hours (log axis)',ylabel='Predicted hours (log axis)',title=f'Frozen-test predictions: {manifest["best_cv_model"]}');ax.legend();save('predictions',fig)
    fig,ax=plt.subplots();ax.scatter(pred.actual_hours,pred.residual_hours,s=13,alpha=.4,color=TEAL);ax.axhline(0,color=GOLD)
    ax.set(xscale='log',yscale='symlog',xlabel='Actual hours (log axis)',ylabel='Actual minus predicted hours (symlog axis)',title='Large positive residuals show underprediction of long games');save('residuals',fig)
    duration=pd.read_csv(OUT/'error_by_duration.csv')
    fig,ax=plt.subplots();ax.bar(duration.duration_band,duration['mean'],color=TEAL)
    for i,r in duration.iterrows():ax.text(i,r['mean']*1.1,f'{r["mean"]:.1f} h\nn={r["count"]}',ha='center',fontsize=10)
    ax.set(yscale='log',ylim=(.5,duration['mean'].max()*2.2),ylabel='Mean absolute error (hours, log axis)',xlabel='Actual completion-time band',title='Long games remain difficult on the full cohort');save('error_by_duration',fig)
    shown=assignments[assignments.partition.eq('train')]
    fig,ax=plt.subplots()
    for label,group in shown.groupby('cluster'):
        ax.scatter(group.PC1,group.PC2,s=9,alpha=.45,label=f'Cluster {int(label)}')
    ax.set(xlabel='Principal component 1',ylabel='Principal component 2',title=f'Training-only PCA and KMeans (k={cluster_info["selected_k"]})');ax.legend();save('pca_clusters',fig)
    fig,ax=plt.subplots();ax.plot(selection.k,selection.training_silhouette,'o-',color=TEAL);ax.set(xticks=selection.k,xlabel='Number of clusters k',ylabel='Sampled training silhouette',title='Choose k without completion-time labels');save('cluster_selection',fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4));axs[0].plot(np.arange(1,len(nn['loss_curve'])+1),nn['loss_curve'],color=TEAL)
    axs[0].set(xlabel='Training iteration',ylabel='Loss in standardized log-target space',title='Neural network training')
    axs[1].plot(np.arange(1,len(nn['internal_validation_R2_curve'])+1),nn['internal_validation_R2_curve'],color=GOLD)
    axs[1].set(xlabel='Training iteration',ylabel='Internal validation R² (transformed target)',title='Early stopping uses training rows only');fig.tight_layout();save('neural_learning',fig)
    importance=pd.read_csv(OUT/'permutation_importance.csv').head(10).iloc[::-1]
    fig,ax=plt.subplots();ax.barh(importance.feature,importance.validation_MAE_increase_hours,xerr=importance.std_hours,color=TEAL)
    ax.set(xlabel='Validation MAE increase after permutation (hours)',title='Importance from one training-validation fold');fig.tight_layout();save('permutation_importance',fig)
    print('Plots saved:',OUT/'figures',flush=True)

if __name__=='__main__':
    run()
