"""Build the Endterm notebook from explicit, reviewable cells; then execute it."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[2]
cells=[]
def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))

md('''# Predicting video game completion time
Syrym Shadiyarbek and Yerdaulet Akhanov · SE2424 · Machine Learning Algorithms Endterm

Can real game metadata help a player choose a game that fits a time budget? We predict IGDB main-story hours on the same 5,229 games and saved split as the Midterm. The Endterm tests additional observed feature types, tuned random forest and boosting, PCA/clustering and an MLP. Text modelling belongs to the Final plan.

Restart the kernel and Run All. `RETRAIN=True` below recomputes every experiment offline from the supplied snapshots. Live collection is separate because new API observations would change this benchmark. The ZIP contains the actual collection code and responses, all analysis code, this notebook, the report and slides. GitHub access is unnecessary.''')
code('''from pathlib import Path
import sys, json
ROOT = next(p for p in [Path.cwd(), Path.cwd().parent] if (p / 'src/train.py').exists())
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from IPython.display import display, Markdown, Image, Code
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 100)
RETRAIN = True
from common import CLEAN, OUT, MIDTERM, sha256
from prepare import prepare
from audit_data import run as audit_sources
prepare()
audit_sources()
from train import load_data
from features import input_columns, DISALLOWED_DURATION_TOKENS
df, training, test = load_data()
quality = json.loads((CLEAN / 'enrichment_report.json').read_text())
collection = json.loads((ROOT / 'data/raw/collection_manifest.json').read_text())
print('Games:', len(df), 'Train:', len(training), 'Test:', len(test))
print('Metadata collection completed UTC:', collection['completed_at_utc'])
print('Target: IGDB hastily seconds / 3600, unchanged since 8 October 2026')''')
md('''## 1 Changes responding to the teacher
The feedback supplied from 2 November identified missing collection code, absent conclusions in the notebook and presentation, empty columns, and a requirement to upload the complete project as a ZIP. It also commented on the defence ("защита получше"). The change log maps the file problems to fixes and gives a seven-minute defence preparation plan.''')
code("display(Markdown((ROOT / 'docs/CHANGELOG.md').read_text(encoding='utf-8')))")
md('''## 2 Collection code and source checks
One row represents one IGDB game. The original target is `game_time_to_beats.hastily / 3600`: reaching credits without notable extras. Although the old raw field name mentions HLTB, its recorded source is IGDB. No HLTB targets are mixed in.

`src/collect.py` collects additional IGDB metadata and explicit linked Steam app metadata, tags and current US USD prices. Both the original IGDB collector and the enrichment collector are included in the ZIP. The next cell displays their full source for inspection. It does not send requests or expose credentials. To recollect, supply your own environment variables and follow README. Saved endpoint responses, timestamps, source-check snapshots and hashes allow offline verification.

IGDB uses authenticated developer API access. The saved terms and source checks document the collection; no individual user profiles or reviews enter the data. Steam IDs come from explicit source links, with ambiguous matches excluded. Prices are current quotes, so this is retrospective catalog prediction. API availability and terms can change.''')
code('''for relative in ['src/collect.py','reference/midterm_source/src/collect_igdb.py']:
    display(Markdown('### ' + relative))
    display(Code(filename=str(ROOT / relative), language='python'))
display(pd.DataFrame([{'raw_source': name, 'sha256': entry['sha256']}
    for name, entry in collection['files'].items()]))''')
md('''## 3 Meaningful columns and preparation
Historical `price_usd` and `tags_text` contain no observations and remain audit fields. They are excluded from the models. New observed price/tag columns and other metadata supply additional feature types. A missing price remains missing in the stored dataset. Fold-fitted predictor imputation is a computational step, and never changes observed values or targets.''')
code('''model_data = pd.read_csv(CLEAN / 'games_model_inputs.csv')
assert model_data[input_columns('full')].notna().any().all()
empty_predictors = [c for c in input_columns('full') if model_data[c].isna().all()]
print('Main modelling dataset:', model_data.shape, 'Entirely empty predictors:', empty_predictors)
display(model_data[['game_id','title','main_story_hours','partition','igdb_themes',
    'igdb_game_modes','steam_regular_price_usd','steam_tags']].head(6))
dictionary = pd.read_csv(CLEAN / 'data_dictionary.csv')
print('Stored columns:', len(dictionary), 'Raw model input fields:', len(input_columns('full')))
display(dictionary[['field','stored_type','meaning','model_input','observed_rows']])
display(pd.read_csv(CLEAN / 'cleaning_steps.csv'))
display(pd.read_csv(CLEAN / 'cleaning_audit.csv').query('not kept')[['game_id','title','reason']].head(12))
display(Image(filename=str(OUT / 'figures/feature_coverage.png'), width=850))''')
md('''The inherited filter accepts finite targets between 0.1 and 1,000 hours. All 28 excluded durations are below six minutes; they are real source observations outside that inclusion rule. Five descriptions have fewer than eight words, one title fewer than two characters, and ten normalized-title/year entries are duplicates. These text and deduplication rules can exclude short games or different editions. Endterm preserves these choices for stage comparability.

IDs, titles, descriptions, source URLs, timestamps, other time-to-beat fields and explicit duration tags are excluded from predictors. Imputers, scalers, catalog-frequency tiers and encoders fit inside each training fold. Twenty-three test rows share Steam app IDs with training rows; grouped CV and an overlap-free subset document the limitation.''')
code('''original = pd.read_csv(MIDTERM / 'data/clean/games_clean.csv', dtype={'game_id':str})
assert original.game_id.tolist() == df.game_id.tolist()
np.testing.assert_array_equal(original.main_story_hours, df.main_story_hours)
assert sha256(CLEAN / 'fixed_split.csv') == sha256(MIDTERM / 'data/clean/fixed_split.csv')
assert not set(input_columns('full')) & {'main_story_hours','description','title','game_id','hastily','normally','completely'}
print('Cohort, target and split verified. Excluded duration tags:', sorted(DISALLOWED_DURATION_TOKENS))''')
md('''## 4 Training-only exploration
The target median is 5 hours, its 95th percentile 40 hours and 99th percentile 100 hours. This tail motivates MAE in original hours as the primary metric and log-target reference models. RMSE and R2 retain evidence of large errors.

Recorded RPG genre has median 14.25 hours over 246 training games, compared with 3 hours for Indie over 374 games. Broad genres motivate more detailed themes, modes and gameplay tags.

Among 2,305 training games with observed USD prices, price and duration have Spearman correlation about 0.618. We therefore test price jointly with metadata, report coverage, and use fixed-estimator feature comparisons. This association is descriptive and does not establish a causal price effect.''')
code('''for filename in ['target_distribution.png','genre_duration.png','price_duration.png']:
    display(Image(filename=str(OUT / 'figures' / filename), width=760))
display(training.main_story_hours.quantile([.25,.5,.75,.95,.99,1]).rename('Training hours').to_frame())''')
md('''## 5 Recompute the supervised and unsupervised experiments
The split stays at 4,183 training and 1,046 test games. All candidates use the same three shuffled training folds with seed 42. The median dummy and OLS/Ridge/tree/scaled KNN references provide basic comparisons. Fixed Ridge and ExtraTrees ablations isolate original, expanded IGDB and IGDB-plus-Steam features.

Random forest and gradient boosting each search eight configurations on training CV. The MLP searches four architecture/regularization settings. Parameter choices and reasons are recorded in the displayed model specifications. Select the expanded winner by training CV MAE before scoring the saved test. Earlier Midterm test exposure remains a limitation, and selected search-CV scores are not nested-CV estimates.

The code below performs the full fit by default. `RETRAIN=False` only inspects the saved outputs and is explicitly a faster review mode.''')
code('''if RETRAIN:
    from train import run as train_models
    from cluster import run as cluster_models
    from plots import run as plot_results
    train_models()
    cluster_models()
    from error_analysis import run as analyze_errors
    analyze_errors()
    plot_results()
manifest = json.loads((OUT / 'experiment_manifest.json').read_text())
assert manifest['dataset_sha256'] == sha256(CLEAN / 'games_enriched.csv')
assert manifest['split_sha256'] == sha256(CLEAN / 'fixed_split.csv')
selection = json.loads((OUT / 'model_selection.json').read_text())
comparison = pd.read_csv(OUT / 'model_comparison.csv')
print('Fit completed UTC:', manifest['completed_at_utc'])
print('Selected by training CV:', selection['winner'])
display(pd.read_csv(OUT / 'model_specifications.csv'))
display(comparison[['model','feature_set','train_cv_MAE_hours','train_cv_MAE_std_hours',
    'test_MAE_hours','test_RMSE_hours','test_R2','test_MedAE_hours']].round(3))
ranked = comparison.loc[comparison.feature_set.eq('full')].sort_values('train_cv_MAE_hours')
gap = ranked.iloc[1].train_cv_MAE_hours-ranked.iloc[0].train_cv_MAE_hours
print(f'Top two expanded CV means differ by {gap:.5f} h. Compare this with fold SD; a small gap is not evidence of clear superiority.')''')
code('''for filename in ['random_forest.csv','gradient_boosting.csv','neural_network.csv']:
    search = pd.read_csv(OUT / 'search' / filename)
    display(Markdown('### ' + filename.replace('_',' ').replace('.csv','')))
    display(search[['params','mean_test_score','std_test_score','rank_test_score']].sort_values('rank_test_score'))
print('Search scores are negative MAE. The least negative value is best.')
print('Search-export SD uses sklearn population SD; the comparison reports sample SD across the same three folds.')
display(Image(filename=str(OUT / 'figures/model_comparison.png'), width=900))''')
md('''## 6 What changed when features were added?
Ridge and ExtraTrees keep their respective settings constant across the three feature sets. This separates adding data types from changing the estimator. The paired bootstrap resamples the same test games 5,000 times. Its intervals describe these fitted models on this cohort, and do not cover all training variability. The exact archived Midterm Ridge comparison is distinguished from the current original-feature Ridge refit.''')
code('''display(Image(filename=str(OUT / 'figures/feature_ablation.png'), width=800))
paired = json.loads((OUT / 'paired_bootstrap.json').read_text())
display(pd.DataFrame([{'comparison': name, 'MAE reduction hours': v['MAE_reduction_hours'],
    'lower 95%': v['lower_95_hours'], 'upper 95%': v['upper_95_hours']} for name,v in paired.items()]).round(3))
display(Image(filename=str(OUT / 'figures/permutation_importance.png'), width=820))''')
md('''## 7 Errors across models
All candidates and the exact archived Ridge are examined by actual-duration band and genre. Positive residual means underprediction. Named cases show the largest error, the largest error among 5-100-hour games, and gains/losses against archived Ridge. These post-evaluation examples explain behaviour and do not select settings.

Linear and distance models smooth broad metadata relationships. Forests combine nonlinear interactions, boosting fits raw-hour absolute error, and the MLP learns a regularized transformed-target representation. Their trade-offs must be read alongside the measured subgroup errors: an aggregate improvement does not imply an improvement for each game.''')
code('''duration_errors = pd.read_csv(OUT / 'all_model_error_by_duration.csv')
display(duration_errors.pivot(index='model', columns='duration_band', values='MAE_hours').round(2))
display(pd.read_csv(OUT / 'all_model_error_examples.csv').round(2))
display(pd.read_csv(OUT / 'all_model_error_by_genre.csv').round(2))
cases = pd.read_csv(OUT / 'all_model_error_examples.csv')
for name in ['Archived Midterm Ridge','Random forest','Tuned gradient boosting','Tuned neural network']:
    bands = duration_errors.loc[duration_errors.model.eq(name)].set_index('duration_band')
    case = cases.loc[cases.model.eq(name) & cases.selection.eq('largest error among 5-100-hour games')].iloc[0]
    display(Markdown(f"{name}: MAE **{bands.loc['0-5 h','MAE_hours']:.2f} h** for 0-5-hour games "
        f"and **{bands.loc['>100 h','MAE_hours']:.2f} h** above 100 hours. Its largest 5-100-hour error "
        f"is **{case.title}**, actual {case.actual_hours:.2f} h, predicted {case.predicted_hours:.2f} h."))
for filename in ['predictions.png','residuals.png','cross_model_residuals.png','cross_model_duration_errors.png']:
    display(Image(filename=str(OUT / 'figures' / filename), width=820))
display(pd.read_csv(OUT / 'sensitivity_subsets.csv').round(3))
print('Steam-ID-grouped training CV MAE:', round(manifest['grouped_cv_MAE_mean_hours'],3),
    '+/-', round(manifest['grouped_cv_MAE_std_hours'],3), 'h')''')
md('''## 8 PCA, clustering and earlier-model failures
The full-feature diagnostic mostly separated Steam availability. The main analysis therefore uses L2-normalized IGDB semantic categories. Normalization reduces the influence of annotation volume. PCA retains at least 85% of predictor variance, and k=2..6 KMeans uses training silhouette. A second seed checks stability. Duration never enters fitting or selection.

Cluster profiles use actual themes, genres and modes. The examples link game names and metadata to the earlier Ridge's errors. The cluster error table compares held-out errors of archived Ridge and every Endterm model, so the unsupervised structure directly describes where the earlier model struggles. This descriptive analysis does not choose the winner. Low silhouette and the limited variance in a two-component map constrain the interpretation.''')
code('''cluster_manifest = json.loads((OUT / 'clustering_manifest.json').read_text())
display(pd.DataFrame([{k:v for k,v in cluster_manifest.items() if k not in ['interpretation','completed_at_utc']}]).T)
display(pd.read_csv(OUT / 'cluster_selection.csv').round(4))
display(pd.read_csv(OUT / 'cluster_profiles.csv'))
cluster_errors = pd.read_csv(OUT / 'cluster_error_summary.csv')
display(cluster_errors.loc[cluster_errors.model.isin(['Archived Midterm Ridge',selection['winner']])].round(3))
display(pd.read_csv(OUT / 'cluster_error_examples.csv').round(3))
for filename in ['cluster_selection.png','pca_clusters.png','cluster_model_errors.png']:
    display(Image(filename=str(OUT / 'figures' / filename), width=820))''')
md('''## 9 MLP learning and the best earlier model
The architecture search compares one 32-unit layer with two layers of 64/32 units and two L2 penalties. Small architectures limit complexity for 4,183 training objects. Train-fitted numeric scaling supports optimization. ReLU/Adam and an internal 15% training holdback control optimization and early stopping. The target is log1p hours followed by train-fitted standardization, and predictions return to hours before evaluation.

Training loss uses standardized log-target space. Internal validation R2 is dimensionless and computed on the same transformed target. Falling training loss accompanied by declining validation R2 indicates overfitting after the validation peak. Early stopping restores the best validation weights; the last iteration is not the best iteration. Training stops by validation progress, not test MAE.''')
code('''neural = json.loads((OUT / 'neural_network_diagnostics.json').read_text())
curve = np.asarray(neural['internal_validation_R2_curve'])
best_iteration = int(np.argmax(curve)) + 1
display(pd.DataFrame([{k:neural[k] for k in ['hidden_layer_sizes','alpha','iterations','max_iter',
    'early_stopping','validation_fraction','target_transform']}]).T)
display(Image(filename=str(OUT / 'figures/neural_learning.png'), width=900))
print(f'Validation R2 peaks at iteration {best_iteration}: {curve.max():.3f}; final iteration: {curve[-1]:.3f}.')
earlier = comparison.loc[(comparison.feature_set=='full') & (comparison.model!='Tuned neural network')].sort_values('train_cv_MAE_hours').iloc[0]
nnrow = comparison.set_index('model').loc['Tuned neural network']
display(comparison.loc[comparison.model.isin([earlier.model,'Tuned neural network','Ridge original features'])].round(3))
print(f'MLP minus best earlier ({earlier.model}) test MAE: {nnrow.test_MAE_hours-earlier.test_MAE_hours:+.3f} h.')''')
md('''## 10 Conclusions
The following answer uses the recomputed values. Real metadata can improve rough catalog estimates, while very long and open-ended games remain difficult. Incomplete Steam coverage, current prices, earlier test exposure, overlapping clusters and uncertain online-game labels limit generalization. The original range filter also excludes real games shorter than six minutes.''')
code('''selected = comparison.set_index('model').loc[selection['winner']]
median = comparison.set_index('model').loc['Median baseline']
gain = paired['selected_vs_submitted_midterm_ridge']
display(Markdown(f"The selected model is **{selection['winner']}**. On the same 1,046 test games, "
    f"MAE is **{selected.test_MAE_hours:.2f} hours**, RMSE **{selected.test_RMSE_hours:.2f} hours**, "
    f"R2 **{selected.test_R2:.3f}**, and median absolute error **{selected.test_MedAE_hours:.2f} hours**. "
    f"The MAE reduction against submitted Midterm Ridge is **{gain['MAE_reduction_hours']:.2f} hours**, "
    f"with conditional paired 95% interval **{gain['lower_95_hours']:.2f} to {gain['upper_95_hours']:.2f} hours**. "
    f"MAE is **{100*(median.test_MAE_hours-selected.test_MAE_hours)/median.test_MAE_hours:.1f}%** lower than the median baseline."))
pred = pd.read_csv(OUT / 'test_predictions.csv')
tail = pred.actual_hours.gt(100)
print(f'{tail.sum()} test games above 100 hours contribute {100*pred.loc[tail,"absolute_error"].sum()/pred.absolute_error.sum():.1f}% of total absolute error.')
print('Conclusions: additional observed metadata improves this stage comparison; large-tail errors and weak R2 remain.')''')
md('''## 11 Final source and experiment plan
Use the real IGDB descriptions already stored for these games, and the official `/v4/games` API `summary`/`storyline` fields if further text is needed. Link text by the same IGDB `game_id`. Access uses developer credentials kept in environment variables. Retain source attribution and terms evidence, observe rate limits, and preserve snapshots; current source terms must be checked before any new collection.

Descriptions can distinguish campaign structure and open-ended online games within broad genres. Before encoding, check descriptions for explicit duration statements and remove those phrases to avoid target disclosure. Fit TF-IDF vocabulary, IDF and a regularized Ridge text model inside training folds only. Compare structured-only, text-only and combined representations on the saved split, using the same three training folds and hour MAE. Select vocabulary/regularization by training CV, score the locked comparison on the held-out set, and inspect the long-game and cluster errors. A local app can then load the selected pipeline. This is a testable plan; no text-model improvement is claimed at Endterm.

## 12 Team review responsibilities
Syrym is assigned collection/preparation, exploration and PCA/clustering review. Yerdaulet is assigned model/tuning, MLP and evaluation review. Both must review conclusions, rehearse the seven-minute talk and answer questions across the project. These assignments require each student's factual confirmation after review.

`docs/CONTRIBUTIONS.md` records the review tasks. Human rehearsal, personal understanding and Moodle upload remain student actions.''')

nb=nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
target=ROOT/'notebooks/Video_Game_Endterm.ipynb'
nbf.write(nb,target)
print('Notebook built:',target)
