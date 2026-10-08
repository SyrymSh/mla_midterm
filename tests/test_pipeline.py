"""Small synthetic unit tests, NOT real project measurements or project data."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from train import DeveloperTier, MultiValueEncoder, INPUT, models, metrics
from prepare import parse_record


def test_clean_record_uses_correct_units():
    src = {'id': 1, 'name': 'Example', 'description': 'An adventure exploring fictional areas with a story',
           'release_date': '2021-01-02', 'hltb_main_hours': 14.2, 'price_usd': 1499,
           'genres': ['Adventure'], 'tags': ['Story Rich'], 'developers': ['Example Games']}
    row = parse_record(src)
    assert row['main_story_hours'] == 14.2
    assert abs(row['price_usd'] - 14.99) < 1e-6
    assert row['release_year'] == 2021


def test_multivalue_encoder_folds_only():
    obj = MultiValueEncoder(max_features=3, min_freq=2).fit(pd.Series(['Action|RPG', 'Action|Puzzle', 'Action']))
    assert obj.transform(pd.Series(['RPG|Unseen', 'Action'])).shape == (2, 1)


def test_dev_tier_unknown_games():
    train = pd.DataFrame({'primary_developer': ['Studio A', 'Studio A', 'Studio B'], 'primary_genre': ['RPG'] * 3, 'genres_text': ['RPG'] * 3, 'tags_text': ['Story'] * 3})
    tier = DeveloperTier().fit(train)
    result = tier.transform(pd.DataFrame({'primary_developer': ['Studio A', 'Missing'], 'primary_genre': ['RPG'] * 2, 'genres_text': ['RPG'] * 2, 'tags_text': ['Story'] * 2}))
    assert list(result.developer_tier) == ['2-4 games', 'unseen']


def test_candidates_can_fit_toy_rows():
    n = 75
    df = pd.DataFrame({
        'release_year': [2000 + i % 22 for i in range(n)],
        'price_usd': [0 if i % 3 else 9.99 for i in range(n)],
        'price_missing': [0] * n,
        'primary_genre': ['Action' if i % 2 else 'RPG' for i in range(n)],
        'primary_developer': [f'Studio {i % 6}' for i in range(n)],
        'genres_text': ['Adventure|RPG' if i % 2 else 'Action' for i in range(n)],
        'tags_text': ['Singleplayer|Story Rich' if i % 2 else 'Action|Combat' for i in range(n)],
    })
    y = np.array([2 + i % 15 + (i % 2) * 3 for i in range(n)], dtype=float)
    for name, model in models().items():
        model.fit(df[INPUT], y)
        pred = model.predict(df[INPUT])
        assert np.isfinite(pred).all(), name
        assert 'MAE_hours' in metrics(y, pred)


def test_igdb_style_without_tags_or_prices():
    """Official IGDB collection may lack both fields; pipeline must still work."""
    n = 70
    df = pd.DataFrame({
        'release_year': [2003+i%20 for i in range(n)],
        'price_usd': [np.nan]*n,
        'price_missing': [1]*n,
        'primary_genre': ['Adventure' if i%2 else 'Action' for i in range(n)],
        'primary_developer': [f'Developer{i%20}' for i in range(n)],
        'genres_text': ['Action' if i%2 else 'Adventure' for i in range(n)],
        'tags_text': ['']*n,
    })
    from sklearn.linear_model import Ridge
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline
    from train import preprocessing
    fitted = TransformedTargetRegressor(regressor=Pipeline([
        ('data', preprocessing()), ('reg', Ridge(alpha=20, solver='lsqr'))]),
        func=np.log1p, inverse_func=np.expm1)
    fitted.fit(df[INPUT], np.linspace(3, 75, n))
    predictions = fitted.predict(df[INPUT])
    assert len(predictions)==n and np.isfinite(predictions).all()
