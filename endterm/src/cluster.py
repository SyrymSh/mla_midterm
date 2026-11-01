"""PCA and KMeans fit ONLY on training predictors, with no duration input."""
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from threadpoolctl import threadpool_limits
from common import CLEAN, OUT, SEED, utc_now, write_json
from features import preprocessing, input_columns, MultiHot
from train import load_data

def analyze(coverage_diagnostic=False):
    df,train,test=load_data()
    destination=OUT/'coverage_diagnostic' if coverage_diagnostic else OUT
    destination.mkdir(parents=True,exist_ok=True)
    if coverage_diagnostic:
        columns=input_columns('full')
        pre=preprocessing('full')
    else:
        # Remove store coverage confounding found in the training-only diagnostic.
        columns=['genres_text','igdb_themes','igdb_keywords','igdb_game_modes','igdb_player_perspectives']
        semantic=ColumnTransformer([(f'semantic{i}',MultiHot(100 if c=='igdb_keywords' else 40,3),c)
                                    for i,c in enumerate(columns)],sparse_threshold=0)
        # Equalize descriptor-vector length so annotation volume cannot dominate.
        pre=Pipeline([('semantic',semantic),('row_normalize',Normalizer(norm='l2'))])
    with threadpool_limits(limits=2):
        x=pre.fit_transform(train[columns])
        pca=PCA(n_components=.85,svd_solver='full')
        z=pca.fit_transform(x)
        rows=[]
        models={}
        for k in range(2,7):
            model=KMeans(n_clusters=k,n_init=20,random_state=SEED)
            labels=model.fit_predict(z)
            score=silhouette_score(z,labels,sample_size=min(1500,len(z)),random_state=SEED)
            rows.append({'k':k,'training_silhouette':float(score),'inertia':float(model.inertia_),
                         'smallest_cluster_rows':int(np.bincount(labels).min())})
            models[k]=model
        scores=pd.DataFrame(rows)
        k=int(scores.sort_values('training_silhouette',ascending=False).iloc[0].k)
        model=models[k]
        labels=model.labels_
        check=KMeans(n_clusters=k,n_init=20,random_state=SEED+1).fit(z)
        stability=adjusted_rand_score(labels,check.labels_)
        allz=pca.transform(pre.transform(df[columns]))
    assignments=df[['game_id','title','partition','primary_genre','main_story_hours','steam_metadata_available']].copy()
    assignments['cluster']=model.predict(allz)
    assignments['PC1']=allz[:,0]
    assignments['PC2']=allz[:,1]
    assignments.to_csv(destination/'cluster_assignments.csv',index=False)
    profiles=[]
    train_assigned=train.copy()
    train_assigned['cluster']=labels
    for label,group in train_assigned.groupby('cluster'):
        profile={'cluster':int(label),'training_rows':len(group),
                 'median_duration_hours_for_description_only':float(group.main_story_hours.median()),
                 'steam_coverage_percent':float(group.steam_metadata_available.mean()*100),
                 'single_player_listed_percent':float(group.igdb_single_player.mean()*100),
                 'median_release_year':float(group.release_year.median()),
                 'median_observed_keyword_count':float(group.igdb_keywords_count.median()),
                 'median_observed_theme_count':float(group.igdb_themes_count.median()),
                 'top_genres':', '.join(group.primary_genre.value_counts().head(3).index)}
        for field in ['igdb_game_modes','igdb_themes','steam_tags']:
            tokens=group[field].dropna().str.split('|').explode().value_counts().head(4)
            profile['top_'+field]=', '.join(tokens.index)
        profiles.append(profile)
    pd.DataFrame(profiles).to_csv(destination/'cluster_profiles.csv',index=False)
    scores.to_csv(destination/'cluster_selection.csv',index=False)
    explained=pd.DataFrame({'component':np.arange(1,len(pca.explained_variance_ratio_)+1),
                             'explained_variance_ratio':pca.explained_variance_ratio_,
                             'cumulative_explained_variance':np.cumsum(pca.explained_variance_ratio_)})
    explained.to_csv(destination/'pca_variance.csv',index=False)
    joblib.dump({'preprocessor':pre,'pca':pca,'kmeans':model},destination/'cluster_model.joblib',compress=3)
    write_json(destination/'clustering_manifest.json',{
        'completed_at_utc':utc_now(),'fit_rows':len(train),'pca_input_features':x.shape[1],
        'pca_components':z.shape[1],'retained_variance':float(pca.explained_variance_ratio_.sum()),
        'pc1_pc2_variance':float(pca.explained_variance_ratio_[:2].sum()),
        'selected_k':k,'selection':'Highest sampled training silhouette over k=2..6; no targets.',
        'selected_silhouette':float(scores.set_index('k').loc[k,'training_silhouette']),
        'adjusted_rand_index_second_seed':float(stability),
        'target_used_for_fit_or_selection':False,
        'feature_scope':'All full predictors including coverage indicators' if coverage_diagnostic else 'L2-normalized IGDB semantic categories only; no Steam, prices, numeric counts or missingness flags',
        'input_columns':columns,
        'interpretation':'Descriptive clusters, not ground-truth classes. Durations added after fitting. A separate full-feature diagnostic demonstrates coverage confounding.'})
    print(scores.to_string(index=False),flush=True)
    print('Selected k:',k,'PCA components:',z.shape[1],flush=True)
    return assignments

def run():
    analyze(coverage_diagnostic=True)
    return analyze(coverage_diagnostic=False)

if __name__=='__main__':
    run()
