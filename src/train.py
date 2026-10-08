"""Leakage-conscious, reproducible structured-feature regression experiments.

NLP of game description intentionally reserved for Final. No text feature enters
this midterm model. Multi-valued genres and game tags are structured metadata.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, make_scorer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from common import CLEAN, OUT, SEED, hash_file, utc_now, write_json

NUMERIC = ['release_year', 'price_usd', 'price_missing']
CATEGORICAL = ['primary_genre', 'developer_tier']
TEXT_META = ['genres_text', 'tags_text']
INPUT = NUMERIC + ['primary_genre', 'primary_developer'] + TEXT_META


class DeveloperTier(BaseEstimator, TransformerMixin):
    """Calculate a developer *catalog frequency* proxy from each training fold only.

    A frequent name is not proof of AAA/large studio status. When applied to
    validation/test rows, unseen developer names have tier 'unseen'.
    """
    def fit(self, X, y=None):
        x = pd.DataFrame(X)
        self.developer_counts_ = x['primary_developer'].fillna('Unknown').value_counts().to_dict()
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        def tier(dev):
            n = self.developer_counts_.get(dev, 0)
            if n == 0: return 'unseen'
            if n == 1: return 'one game'
            if n < 5: return '2-4 games'
            if n < 15: return '5-14 games'
            return '15+ games'
        df['developer_tier'] = df['primary_developer'].fillna('Unknown').map(tier)
        df['primary_genre'] = df['primary_genre'].fillna('Unknown').replace('', 'Unknown')
        for col in TEXT_META:
            df[col] = df[col].fillna('').astype(str)
        return df


class MultiValueEncoder(BaseEstimator, TransformerMixin):
    """Train-only top-k vocabulary binarization for pipe-delimited categorical tags."""
    def __init__(self, max_features=60, min_freq=2):
        self.max_features = max_features
        self.min_freq = min_freq

    @staticmethod
    def _tokens(value):
        if value is None or pd.isna(value):
            return []
        return sorted(set(s.strip().lower() for s in str(value).split('|') if s.strip()))

    def fit(self, X, y=None):
        counts = Counter()
        for val in X:
            counts.update(self._tokens(val))
        common = sorted((name for name, n in counts.items() if n >= self.min_freq),
                        key=lambda name: (-counts[name], name))[:self.max_features]
        # A fully absent tag field still needs one all-zero column for sklearn.
        self.features_ = common if common else ['__no_observed_values__']
        self.index_ = {name: idx for idx, name in enumerate(common)}
        return self

    def transform(self, X):
        rows, cols = [], []
        vals = list(X)
        for i, val in enumerate(vals):
            for token in self._tokens(val):
                j = self.index_.get(token)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        return csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)),
                          shape=(len(vals), len(self.features_)), dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f'multi_{x}' for x in self.features_], dtype=object)


def preprocessing():
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=False, keep_empty_features=True)),
                        ('scale', StandardScaler())])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=4))])
    column = ColumnTransformer(
        transformers=[
            ('num', numeric, NUMERIC),
            ('cat', categorical, CATEGORICAL),
            ('genre_multi', MultiValueEncoder(35, 1), 'genres_text'),
            ('tags_multi', MultiValueEncoder(80, 3), 'tags_text'),
        ], sparse_threshold=.9,
    )
    return Pipeline([('developer_tier', DeveloperTier()), ('features', column)])


def models():
    regressors = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=20, solver='lsqr'),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, random_state=SEED),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=25, weights='distance', n_jobs=-1),
    }
    candidates = {'Dummy (median)': Pipeline([('input', preprocessing()),
                                               ('model', DummyRegressor(strategy='median'))])}
    for name, reg in regressors.items():
        base = Pipeline([('input', preprocessing()), ('model', reg)])
        candidates[name] = TransformedTargetRegressor(regressor=base,
                                                       func=np.log1p, inverse_func=np.expm1,
                                                       check_inverse=False)
    return candidates


def clipped_mae(y_true, y_pred):
    return mean_absolute_error(y_true, np.maximum(0, np.asarray(y_pred, dtype=float)))


def metrics(y_true, y_pred):
    a = np.asarray(y_true, dtype=float)
    b = np.maximum(0, np.asarray(y_pred, dtype=float))
    smape = np.mean(2 * np.abs(a - b) / np.maximum(a + b, 1e-9)) * 100
    return {
        'MAE_hours': float(mean_absolute_error(a, b)),
        'RMSE_hours': float(np.sqrt(mean_squared_error(a, b))),
        'R2': float(r2_score(a, b)),
        'MedAE_hours': float(median_absolute_error(a, b)),
        'MAPE_percent': float(np.mean(np.abs((a - b) / np.maximum(a, 1e-9))) * 100),
        'sMAPE_percent': float(smape),
    }


def fixed_split(df, reset=False):
    """Persist split IDs so later stages use exactly the same held-out games."""
    path = CLEAN / 'fixed_split.csv'
    ids = df.game_id.astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise ValueError('game_id duplicates remain: split cannot be stable')
    if path.exists() and not reset:
        split = pd.read_csv(path, dtype={'game_id': str})
        if set(split.game_id) != set(ids) or len(split) != len(ids):
            raise RuntimeError('Dataset IDs changed since fixed_split.csv was created. Restore original data, or use --reset-split deliberately.')
    else:
        train_idx, test_idx = train_test_split(np.arange(len(ids)), test_size=.2, random_state=SEED)
        test_ids = set(df.iloc[test_idx].game_id.astype(str))
        split = pd.DataFrame({'game_id': ids, 'partition': ['test' if gid in test_ids else 'train' for gid in ids]})
        split.to_csv(path, index=False)
    train_ids = set(split.loc[split.partition == 'train', 'game_id'])
    test_ids = set(split.loc[split.partition == 'test', 'game_id'])
    if train_ids & test_ids or not test_ids or not train_ids:
        raise ValueError('Fixed split is invalid')
    return df[df.game_id.astype(str).isin(train_ids)].copy(), df[df.game_id.astype(str).isin(test_ids)].copy()


def set_plot_defaults():
    plt.rcParams.update({'figure.figsize': (9, 5), 'axes.spines.right': False,
                         'axes.spines.top': False, 'font.size': 11,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.facecolor': 'white'})


def save_chart(path, fig):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight', dpi=175)
    plt.close(fig)


def plot_eda(df, folder):
    fig, ax = plt.subplots()
    ax.hist(df.main_story_hours.clip(upper=df.main_story_hours.quantile(.98)), bins=38,
            color='#15667d', edgecolor='white')
    ax.set(title='Main-story duration, capped at the 98th percentile', xlabel='Hours', ylabel='Games')
    ax.grid(alpha=.15)
    save_chart(folder / 'target_distribution.png', fig)
    counts = df.primary_genre.value_counts().head(10).sort_values()
    fig, ax = plt.subplots()
    counts.plot(kind='barh', ax=ax, color='#15667d')
    ax.set(title='Most common primary genres', xlabel='Games', ylabel='Genre')
    save_chart(folder / 'genre_counts.png', fig)


def plot_result(results, predictions, folder):
    ordered = results.sort_values('MAE_hours', ascending=False)
    fig, ax = plt.subplots()
    ax.barh(ordered.model, ordered.MAE_hours, color=['#927ad9' if name == 'Dummy (median)' else '#15667d' for name in ordered.model])
    ax.set(xlabel='Test MAE (hours; lower is better)', title='Held-out test error by model')
    for i, v in enumerate(ordered.MAE_hours):
        ax.text(v + max(.08, ordered.MAE_hours.max() * .01), i, f'{v:.2f}', va='center')
    ax.set_xlim(0, ordered.MAE_hours.max() * 1.2)
    save_chart(folder / 'model_mae.png', fig)
    f, ax = plt.subplots()
    ax.scatter(predictions.actual_hours, predictions.predicted_hours, s=17, alpha=.37, color='#15667d')
    high = float(max(predictions.actual_hours.quantile(.99), predictions.predicted_hours.quantile(.99)))
    ax.plot([0, high], [0, high], ls='--', c='#bb5b53', lw=1.5)
    ax.set(xlim=(0, high), ylim=(0, high), xlabel='Actual time (hours)', ylabel='Predicted time (hours)',
           title='Best CV model: predicted versus actual')
    save_chart(folder / 'prediction_vs_actual.png', f)
    bins = pd.cut(predictions.actual_hours, [0, 5, 15, 30, 60, 1000],
                  labels=['0-5h', '5-15h', '15-30h', '30-60h', '60h+'], include_lowest=True)
    group = predictions.groupby(bins, observed=True).absolute_error.agg(['median', 'count'])
    fig, ax = plt.subplots()
    ax.bar(group.index.astype(str), group['median'], color='#15667d')
    ax.set(xlabel='Actual completion-time bin', ylabel='Median absolute error (hours)',
           title='Error grows for longer games?')
    save_chart(folder / 'error_by_time.png', fig)


def run(csv_path: Path = CLEAN / 'games_clean.csv', reset_split=False, skip_cv=False,
        min_rows=3000):
    if not csv_path.exists():
        raise FileNotFoundError('Clean dataset not available. Run collection and src/prepare.py first.')
    df = pd.read_csv(csv_path, dtype={'game_id': str})
    if len(df) < min_rows:
        raise ValueError(f'Only {len(df)} rows; need at least {min_rows} for the project.')
    for name in INPUT + ['main_story_hours', 'game_id', 'title', 'description']:
        if name not in df:
            raise ValueError(f'Missing required column {name}')
    if df.main_story_hours.isna().any() or (df.main_story_hours <= 0).any():
        raise ValueError('Target must be known and strictly positive, without missing values')
    OUT.mkdir(parents=True, exist_ok=True)
    folder = OUT / 'figures'
    folder.mkdir(exist_ok=True)
    set_plot_defaults()
    plot_eda(df, folder)
    train, test = fixed_split(df, reset=reset_split)
    xtrain, ytrain = train[INPUT].copy(), train.main_story_hours
    xtest, ytest = test[INPUT].copy(), test.main_story_hours
    cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    candidates = models()
    for name, model in candidates.items():
        print('Fitting:', name, flush=True)
        cv_mae = np.nan
        if not skip_cv:
            scores = cross_val_score(model, xtrain, ytrain, cv=cv,
                                     scoring=make_scorer(clipped_mae, greater_is_better=False), error_score='raise')
            cv_mae = -float(scores.mean())
        model.fit(xtrain, ytrain)
        predicted = np.maximum(0, model.predict(xtest))
        row = {'model': name, 'train_cv_MAE_hours': cv_mae, **metrics(ytest, predicted)}
        rows.append(row)
        fitted[name] = (model, predicted)
        print('  Test MAE:', round(row['MAE_hours'], 3), '| R2:', round(row['R2'], 3))
    result = pd.DataFrame(rows)
    winner = (result.sort_values('train_cv_MAE_hours').iloc[0].model if not skip_cv
              else 'Ridge Regression')
    result.to_csv(OUT / 'model_comparison.csv', index=False)
    model, pred = fitted[winner]
    predictions = test[['game_id', 'title', 'primary_genre', 'main_story_hours']].copy()
    predictions['predicted_hours'] = pred
    predictions['signed_error'] = predictions.predicted_hours - predictions.main_story_hours
    predictions['absolute_error'] = predictions.signed_error.abs()
    predictions = predictions.rename(columns={'main_story_hours': 'actual_hours'})
    predictions.to_csv(OUT / 'test_predictions.csv', index=False)
    genre_error = predictions.groupby('primary_genre').absolute_error.agg(['count', 'mean', 'median'])
    genre_error = genre_error[genre_error['count'] >= 10].sort_values('mean', ascending=False)
    genre_error.to_csv(OUT / 'error_by_genre.csv')
    plot_result(result, predictions, folder)
    joblib.dump(model, OUT / 'best_structured_model.joblib')
    info = {'completed_at_utc': utc_now(), 'source_csv_sha256': hash_file(csv_path),
            'fixed_split_sha256': hash_file(CLEAN / 'fixed_split.csv'),
            'seed': SEED, 'split': '80% train / 20% untouched held-out test',
            'training_rows': len(train), 'test_rows': len(test),
            'cv_folds': 0 if skip_cv else 3, 'selection': 'minimum train CV MAE (no test-set selection)' if not skip_cv else 'Ridge prespecified, CV disabled',
            'best_cv_model': winner,
            'best_model_test_metrics': result.loc[result.model == winner].iloc[0].drop('model').to_dict(),
            'features': INPUT, 'uses_game_description': False,
            'developer_tier_definition': 'Computed within each training fold from developer catalogue count; not verified company size',
            'notes': 'Other candidate test metrics shown for comparison only. Select final model using CV.',
        }
    write_json(OUT / 'experiment_manifest.json', info)
    print('Chosen model based on CV:', winner)
    print('Saved measured results and figures to', OUT)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=CLEAN / 'games_clean.csv')
    parser.add_argument('--reset-split', action='store_true', help='Deliberately overwrite the persistent split')
    parser.add_argument('--skip-cv', action='store_true', help='Fast development run; CV not performed')
    parser.add_argument('--min-rows', type=int, default=3000)
    args = parser.parse_args()
    run(args.input, args.reset_split, args.skip_cv, args.min_rows)
