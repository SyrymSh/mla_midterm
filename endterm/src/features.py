"""Fit all vocabularies, missing-value transforms and studio counts in folds."""
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_NUM=['release_year']
BASE_CAT=['primary_genre','developer_catalog_tier']
BASE_MULTI=['genres_text']
IGDB_NUM=['igdb_platforms_count','igdb_game_modes_count','igdb_player_perspectives_count',
          'igdb_themes_count','igdb_developer_count','igdb_publisher_count',
          'igdb_single_player','igdb_multiplayer','igdb_coop']
IGDB_CAT=['igdb_game_type']
IGDB_MULTI=['igdb_themes','igdb_keywords','igdb_game_modes','igdb_player_perspectives','igdb_platforms','igdb_game_engines']
STEAM_NUM=['steam_regular_price_usd','steam_current_price_usd','steam_discount_percent',
           'steam_tag_count','steam_single_player','steam_multiplayer','steam_coop',
           'steam_online_coop','steam_achievements_listed','steam_workshop','steam_cloud',
           'steam_full_controller_support','steam_windows','steam_mac','steam_steamos_linux','steam_language_count']
STEAM_CAT=['steam_item_type']
STEAM_MULTI=['steam_tags','steam_categories']
GROUPS={
    'midterm':(BASE_NUM,BASE_CAT,BASE_MULTI),
    'igdb':(BASE_NUM+IGDB_NUM,BASE_CAT+IGDB_CAT,BASE_MULTI+IGDB_MULTI),
    'full':(BASE_NUM+IGDB_NUM+STEAM_NUM,BASE_CAT+IGDB_CAT+STEAM_CAT,BASE_MULTI+IGDB_MULTI+STEAM_MULTI),
}
DISALLOWED_DURATION_TOKENS={'short','long','short game','long game','short games','long games','short playtime','long playtime'}

class CatalogTier(TransformerMixin,BaseEstimator):
    def fit(self,X,y=None):
        self.counts_=pd.DataFrame(X)['primary_developer'].fillna('Unknown').value_counts().to_dict()
        return self
    def transform(self,X):
        df=pd.DataFrame(X).copy()
        counts=df.primary_developer.fillna('Unknown').map(self.counts_).fillna(0)
        df['developer_catalog_tier']=pd.cut(counts,[-1,0,1,4,14,np.inf],labels=['unseen','one game','2-4 games','5-14 games','15+ games']).astype(str)
        return df

class MultiHot(TransformerMixin,BaseEstimator):
    def __init__(self,max_features=80,min_count=3):
        self.max_features=max_features
        self.min_count=min_count
    @staticmethod
    def tokens(value):
        return sorted({x.strip().lower() for x in str(value).split('|') if x.strip() and x.strip().lower() not in DISALLOWED_DURATION_TOKENS}) if pd.notna(value) else []
    def fit(self,X,y=None):
        counts=Counter(token for value in X for token in self.tokens(value))
        self.vocabulary_={t:i for i,t in enumerate(sorted((t for t,n in counts.items() if n>=self.min_count),key=lambda t:(-counts[t],t))[:self.max_features])}
        return self
    def transform(self,X):
        values=list(X)
        result=np.zeros((len(values),max(1,len(self.vocabulary_))),dtype=np.float64)
        for i,value in enumerate(values):
            for token in self.tokens(value):
                if token in self.vocabulary_: result[i,self.vocabulary_[token]]=1.0
        return result
    def get_feature_names_out(self,input_features=None):
        return np.array(list(self.vocabulary_) or ['no_observed_values'],dtype=object)

def preprocessing(feature_set='full'):
    nums,cats,multis=GROUPS[feature_set]
    numeric=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())])
    categorical=Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='Unknown')),('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=4,sparse_output=False))])
    transforms=[('numeric',numeric,nums),('categorical',categorical,cats)]
    for i,field in enumerate(multis):
        limit=100 if field in ['igdb_keywords','steam_tags'] else 40
        transforms.append((f'multi{i}',MultiHot(limit,3),field))
    return Pipeline([('catalog',CatalogTier()),('encode',ColumnTransformer(transforms,remainder='drop',sparse_threshold=0))])

def input_columns(feature_set='full'):
    nums,cats,multis=GROUPS[feature_set]
    return nums+[x for x in cats if x!='developer_catalog_tier']+['primary_developer']+multis
