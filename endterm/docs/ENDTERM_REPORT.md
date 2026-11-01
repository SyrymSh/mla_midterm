# Predicting video game completion time

**Machine Learning Algorithms Endterm**

**Syrym Shadiyarbek and Yerdaulet Akhanov**

## Research question

Can a game's structured metadata help a player estimate the time needed to reach its credits? The project predicts a positive number of main-story hours. The endterm tests whether real additional feature types and the Week 5-8 methods improve the midterm model.

The selected model is gradient boosting. On the original 1,046 test games, its mean absolute error is **10.26 hours**, compared with **11.82 hours** for the submitted midterm Ridge model and **13.01 hours** for the median baseline. Median absolute error is **2.39 hours**. R2 is only **0.023**, and the largest errors remain substantial. The result supports rough catalog estimates, especially for ordinary-length games, rather than precise estimates for every title.

## Data and collection

The same 5,229 games are retained. Their target is IGDB's `hastily` time in seconds divided by 3,600, recorded on 8 October 2026. It measures reaching the credits without notable extras. The original raw field is named `hltb_main_hours`, but its source record explicitly identifies IGDB. No HLTB labels are mixed into this study.

On 2 November 2026 in Asia/Qyzylorda, the metadata was expanded using the official IGDB API and public Steam endpoints. UTC timestamps are saved beside the responses. IGDB adds themes, gameplay keywords, modes, perspectives, platforms, engines and game type. Explicit linked Steam app IDs add top user tags, store categories, platform/language information and current prices. Matching does not rely on similar titles.

The expanded snapshot contains **2,861 observed USD prices** and **3,333 rows with Steam tags**. Prices are current US-store quotes, not launch prices. Missing quotes remain missing. Imputed model inputs are kept separate from observations, and targets are never imputed. All observed regular prices and tag names were checked against the saved raw responses.

The data card defines all 70 stored columns and distinguishes the 39 raw predictor fields from IDs, provenance and historical fields. The fitted full pipeline creates 473 encoded features. Real descriptions are retained for Final-stage text experiments and are excluded from these models.

IGDB documents non-commercial API use under the Twitch Developer Service Agreement. Steam's API terms and robots rules were checked and saved. The IGDB website robots request returned 403; collection used the authenticated API instead of website scraping. No private user profiles or individual reviews were collected. Developer credentials are not stored in the submission. API access, source attribution and current metadata limitations are documented in DATA_CARD.md.

## Cleaning and checks

The audit reconstructs the original cleaning decisions from the raw snapshot. Every raw row has a recorded outcome in `data/clean/cleaning_audit.csv`.

| step | rows_before | removed | rows_after |
| --- | --- | --- | --- |
| Raw snapshot | 5273 | 0 | 5273 |
| Invalid target / insufficient text / unusable title | 5273 | 34 | 5239 |
| Duplicate normalized title and release year | 5239 | 10 | 5229 |
| Duplicate game ID | 5229 | 0 | 5229 |

The original target rule accepts finite values from 0.1 to 1,000 hours. All 28 excluded target values are positive durations below 0.1 hours (six minutes); they are outside that inclusion rule, not proven incorrect labels. Five descriptions have fewer than eight words and one title has fewer than two characters. These text filters retain a usable title and description for the later text stage. Ten repeated normalized-title/year records are removed to reduce duplicate entries; no additional repeated game IDs remain. The row-level audit records every outcome.

These inherited choices can exclude real very short games or collapse different editions with the same title and year. They are retained for comparability with the midterm. The reproduced game IDs and target values match the supplied clean dataset. Endterm uses a left join: it changes no target, adds no game and removes no further game. The split file is byte-identical to the midterm file.

## Exploration and decisions

Three training-only summaries guide the modelling choices.

| Evidence | Interpretation | Decision |
| --- | --- | --- |
| Training duration median 5 h, 95th percentile 40 h, 99th percentile 100 h | A few long labels can dominate squared error. | Use MAE as the main metric, keep RMSE/R2 visible, and test log-target models. |
| Recorded RPG genre median 14.25 h (246 training games), compared with 3 h for Indie (374 games) | Broad genres carry information but hide variation within a genre. | Retain all genres and add themes, modes, perspectives and gameplay tags. |
| 2,305 training games have observed prices; price/duration Spearman correlation is 0.618 | Price is associated with duration in the observed store-linked subset, but cannot explain it alone. | Test price jointly with other metadata, measure coverage and use a controlled feature comparison. |

The target, genre and price figures in the notebook use training rows only. These associations do not establish that changing a price changes a game's duration.

## Models and validation

The original split contains 4,183 training and 1,046 test games. Three shuffled training folds use seed 42. Numeric imputation and scaling, categorical encodings, tag vocabularies and the developer catalog tier are fitted inside each training fold. The catalog tier means frequency in the training sample, not verified studio size.

Other completion-time fields, explicit duration tags such as Short/Long, game IDs, titles, descriptions, URLs and collection timestamps are excluded. Twenty-three test rows share a Steam app ID with a training row. This limitation is reported rather than hidden: the study includes a test subset without that overlap and a Steam-ID-grouped training-CV check.

| Model family | Why it is included | Main settings |
| --- | --- | --- |
| Median baseline | A simple reference for absolute error | Training-label median |
| Ordinary linear regression | Tests linear relationships without regularization | OLS on the log target |
| Ridge | Regularizes the linear reference | alpha 20, lsqr solver |
| Decision tree | Tests nonlinear thresholds | depth 10, at least 10 rows per leaf |
| KNN | Tests local similarity | 25 distance-weighted neighbours; numeric scaling |
| ExtraTrees | Tests randomized tree interactions and feature-set gains | 200 trees, leaf minimum 3, feature fraction 0.8 |
| Random forest | Reduces single-tree variance through averaging | 250 bootstrap trees, leaf minimum 3, feature fraction 0.7 |
| Gradient boosting | Fits the hour-scale absolute-error objective | 250 iterations; learning rate 0.07; eight leaf/min-leaf/L2 settings |
| Neural network | Tests learned nonlinear combinations | Four architecture/L2 settings; early stopping |

The classical references use the original feature family and the same split/folds. Ridge and ExtraTrees also repeat unchanged settings with expanded IGDB and then IGDB plus Steam features. This separates feature gains from changing the estimator. The full parameter record is in `outputs/model_specifications.csv`.

OLS, Ridge, tree, KNN and forest models fit log1p hours. The neural network also standardizes that log target. Boosting uses absolute-error loss on raw hours. All reported metrics use predictions converted back to hours. The tree and KNN settings are fixed reference choices; they were not selected on the test score.

The expanded winner is selected by mean training CV MAE. Boosting searches eight configurations and the neural network searches four. Tuning CV is not nested CV, so its selected score can be optimistic. The separately dated classical-reference supplement does not change the locked expanded-model selection. The test was already observed at midterm and is a stage-comparison benchmark, not a new independent test.

## Results

All errors below are in hours. CV spread is the sample standard deviation across the three validation folds, not a confidence interval. Original refers to the midterm feature family; IGDB and full refer to the expanded inputs.

| Model and features | CV MAE +/- SD | Test MAE | RMSE | R2 | Median AE |
| --- | --- | --- | --- | --- | --- |
| Median baseline | 11.07 +/- 0.72 | 13.01 | 52.17 | -0.041 | 4.00 |
| OLS (original) | 10.10 +/- 0.64 | 11.83 | 51.27 | -0.006 | 3.28 |
| Ridge (original) | 10.07 +/- 0.68 | 11.81 | 51.43 | -0.012 | 3.20 |
| Decision tree (original) | 10.19 +/- 0.80 | 11.81 | 50.96 | 0.006 | 3.26 |
| KNN (original) | 10.40 +/- 0.69 | 11.90 | 51.25 | -0.005 | 3.29 |
| Ridge (IGDB) | 9.25 +/- 0.66 | 10.69 | 50.36 | 0.029 | 2.55 |
| Ridge (IGDB + Steam) | 9.16 +/- 0.68 | 10.71 | 50.49 | 0.024 | 2.42 |
| ExtraTrees (original) | 10.01 +/- 0.66 | 11.70 | 51.12 | -0.000 | 3.39 |
| ExtraTrees (IGDB) | 9.22 +/- 0.70 | 10.82 | 50.53 | 0.023 | 2.50 |
| ExtraTrees (IGDB + Steam) | 8.77 +/- 0.74 | 10.49 | 50.54 | 0.022 | 2.37 |
| Random forest (full) | 8.67 +/- 0.66 | 10.33 | 50.51 | 0.024 | 2.29 |
| Tuned boosting (full) | 8.64 +/- 0.60 | 10.26 | 50.52 | 0.023 | 2.39 |
| Tuned MLP (full) | 9.02 +/- 0.64 | 10.41 | 50.18 | 0.036 | 2.29 |

The selected boosting model has CV MAE **8.64 +/- 0.60 h**. Steam-ID-grouped training CV gives **8.66 +/- 0.66 h**. The test MAE reduction against the submitted Ridge is **1.56 h**, with a paired 95% bootstrap interval of **1.29 to 1.81 h**. Against the median baseline, MAE is **21.2% lower**.

## Value of the new features

With fixed Ridge settings, extra IGDB fields lower test MAE from 11.81 to 10.69 h. Steam then gives 10.71 h, so it adds no clear further gain for Ridge. For fixed ExtraTrees settings, extra IGDB gives 10.82 h and Steam lowers it to 10.49 h. That Steam gain is 0.33 h, with a paired interval of 0.10 to 0.57 h. Useful columns therefore depend partly on the model and the information already present.

A permutation check on one training-validation fold ranks genre, regular price, game type, themes and Steam tags among useful fields. Permuting regular price increases MAE by about 0.58 h and tags by 0.26 h in that fold. Correlated features can substitute for each other, so these are predictive associations rather than causal effects.

Bootstrap comparisons resample the same test games 5,000 times in pairs. They describe conditional uncertainty for these fitted models on this cohort, not all training variability or future-game performance. The complete intervals are saved in `outputs/paired_bootstrap.json`.

## PCA and clustering

The full-feature diagnostic mainly split games by Steam metadata coverage. The main analysis therefore uses IGDB genres, themes, keywords, modes and perspectives only, with no Steam fields or explicit missingness flags. L2 normalization reduces the influence of annotation volume before PCA. Neither PCA nor KMeans uses duration labels.

PCA reduces 157 encoded semantic features to **65 components**, retaining **85.1%** of feature variance. KMeans compares k=2 through 6 on training rows. Among these settings, k=6 has the highest sampled silhouette, **0.078**. The second-seed adjusted Rand index is **0.970**.

| Cluster | Train games | Common themes | Median hours |
| --- | --- | --- | --- |
| 0 | 1007 | Horror, Mystery | 2.0 |
| 1 | 566 | Action, Fantasy | 3.0 |
| 2 | 561 | Action, Science fiction | 8.0 |
| 3 | 666 | Fantasy, Action | 7.5 |
| 4 | 779 | Action, Fantasy | 10.0 |
| 5 | 604 | Action, Science fiction | 5.0 |

Durations are added only after fitting to describe the groups. The low silhouette shows substantial overlap despite stable initialization. The first two principal components explain only **15.2%** of feature variance. These results do not establish six distinct natural game classes.

## Neural network

The best neural configuration has 64- and 32-unit hidden layers, ReLU/Adam and L2 penalty 0.1. It uses a 15% internal training holdback and stops after 30 of at most 350 iterations, restoring the best validation weights. The training and validation curves use transformed-target units, not hour MAE.

Its CV MAE is 9.02 h and test MAE 10.41 h. Boosting performs better on the primary MAE metric, so the neural network remains a benchmark. Its slightly better R2 does not replace the prespecified MAE selection rule.

## Errors and label quality

| Actual hours | Games | MAE | Median AE |
| --- | --- | --- | --- |
| 0-5 h | 505 | 1.91 | 1.01 |
| >5-20 h | 387 | 4.14 | 3.23 |
| >20-50 h | 112 | 14.7 | 13.19 |
| >50-100 h | 29 | 57.29 | 50.64 |
| >100 h | 13 | 373.86 | 344.14 |

Only 13 test games exceed 100 hours, but they contribute **45.3% of total absolute error**. These cases explain much of the gap between mean and median error.

| IGDB ID | Game | Actual | Predicted | Error |
| --- | --- | --- | --- | --- |
| 143228 | Gorilla Tag | 997.5 | 5.85 | 991.65 |
| 134710 | Grand Theft Auto Online | 513.0 | 12.69 | 500.31 |
| 7360 | Rainbow Six Siege | 500.0 | 7.72 | 492.28 |
| 79824 | Old School RuneScape | 486.5 | 21.21 | 465.29 |
| 373557 | HDOS | 486.0 | 22.55 | 463.45 |
| 119277 | Genshin Impact | 468.49 | 23.3 | 445.19 |

The largest failures include ongoing online titles such as Gorilla Tag and Grand Theft Auto Online. A main-story average is difficult to interpret for those games. Their source labels are retained; a large error alone does not justify inventing a correction or dropping a test row.

The report also checks genre errors, store coverage, games explicitly listed as single-player and the subset without repeated Steam IDs. These are descriptive sensitivity checks. They do not replace the full-cohort score or choose the model.

## Conclusions and Final plan

The added metadata improves the comparison with the midterm, and tuned boosting has the lowest training CV MAE among expanded candidates. The average test error is still 10.26 h and R2 is low. The method is more useful for rough estimates of ordinary-length games than for very long or open-ended titles.

The main limits are selective Steam coverage, current rather than launch prices, community labels of unequal reliability, repeated store IDs and earlier exposure to test outcomes. The clustering result also shows weak separation. Those limitations remain part of the conclusion even though some metrics improve.

| Midterm weakness | Endterm response | Result and next step |
| --- | --- | --- |
| Empty price/tag fields and coarse metadata | Real IGDB/Steam feature expansion and matched feature comparisons | Gains are measured by estimator; retain source and coverage audits. |
| Limited linear relationships | Forests, ExtraTrees and tuned boosting | Boosting improves MAE; tail errors still need better label/scope validation. |
| No view of game groupings | Normalized descriptors, PCA and KMeans | Groups overlap; use them for exploration, not ground-truth labels. |
| No learned nonlinear benchmark | Tuned MLP with early stopping | Tested on the same split; it does not beat boosting on MAE. |

For Final, add description features and compare them with the structured model, build the local prediction app, and seek a newly collected independent validation cohort if feasible. These are the Final-stage additions named in the supplied guide; no description features enter this Endterm model.

## Team roles and tools

Syrym is assigned collection, preparation, exploration and unsupervised analysis. Yerdaulet is assigned modelling, tuning and evaluation. Both should review the findings and practise the full defence. These are assigned responsibilities; individual completed work has not been confirmed.

OpenAI Codex assisted with code, authorized data collection, executed experiments, checks and preparation of the materials. The factual disclosure and review tasks are in CONTRIBUTIONS.md.

## Sources

- Supplied Project Guide and Midterm Rubric, retained under `reference/`.
- IGDB API documentation: https://api-docs.igdb.com/
- Steam API overview: https://partner.steamgames.com/doc/webapi_overview
- Steam API terms: https://steamcommunity.com/dev/apiterms
- Saved raw responses, collection manifests, model configurations and experiment outputs included with the project.
