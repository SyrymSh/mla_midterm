"""Critical model/data contracts. Toy fixtures never enter the project dataset."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from features import CatalogTier, MultiHot, input_columns
from common import CLEAN, MIDTERM, sha256

class LeakageContracts(unittest.TestCase):
    def test_all_predictors_exclude_label_identifiers_and_descriptions(self):
        forbidden={'main_story_hours','description','title','game_id','steam_appid',
                   'matched_steam_appid','time_submissions','hastily','normally','completely'}
        for feature_set in ['midterm','igdb','full']:
            self.assertFalse(set(input_columns(feature_set)) & forbidden)

    def test_developer_counts_are_fitted_only_on_training(self):
        transformer=CatalogTier().fit(pd.DataFrame({'primary_developer':['A','A','B']}))
        actual=transformer.transform(pd.DataFrame({'primary_developer':['A','C','C']}))
        self.assertEqual(actual.developer_catalog_tier.tolist(),['2-4 games','unseen','unseen'])
        self.assertEqual(transformer.counts_,{'A':2,'B':1})

    def test_tags_from_validation_do_not_expand_vocabulary(self):
        enc=MultiHot(10,1).fit(pd.Series(['Puzzle|Adventure','Puzzle']))
        before=enc.vocabulary_.copy()
        encoded=enc.transform(pd.Series(['Unknown tag','Puzzle|Unknown tag',np.nan]))
        self.assertEqual(enc.vocabulary_,before)
        self.assertEqual(encoded.sum(axis=1).tolist(),[0,1,0])

    def test_obvious_duration_tags_are_excluded(self):
        self.assertEqual(MultiHot.tokens('Short|Puzzle|Long|Story Rich'),['puzzle','story rich'])

    def test_split_is_byte_identical(self):
        self.assertEqual(sha256(CLEAN/'fixed_split.csv'),sha256(MIDTERM/'data/clean/fixed_split.csv'))

    def test_target_and_game_cohort_unchanged(self):
        old=pd.read_csv(MIDTERM/'data/clean/games_clean.csv',dtype={'game_id':str}).set_index('game_id')
        new=pd.read_csv(CLEAN/'games_enriched.csv',dtype={'game_id':str}).set_index('game_id')
        self.assertEqual(old.index.tolist(),new.index.tolist())
        np.testing.assert_array_equal(old.main_story_hours.to_numpy(),new.main_story_hours.to_numpy())

    def test_prices_use_explicit_usd_and_are_nonnegative(self):
        df=pd.read_csv(CLEAN/'games_enriched.csv')
        known=df.steam_regular_price_usd.notna()
        self.assertTrue(df.loc[known,'steam_price_currency'].eq('USD').all())
        self.assertTrue(df.loc[known,'steam_regular_price_usd'].ge(0).all())
        self.assertGreater(int(known.sum()),0)

    def test_steam_unmatched_has_no_invented_price_or_tags(self):
        df=pd.read_csv(CLEAN/'games_enriched.csv')
        unmatched=df.matched_steam_appid.isna()
        self.assertTrue(df.loc[unmatched,'steam_regular_price_usd'].isna().all())
        self.assertTrue(df.loc[unmatched,'steam_tags'].isna().all())

if __name__=='__main__':
    unittest.main(verbosity=2)
