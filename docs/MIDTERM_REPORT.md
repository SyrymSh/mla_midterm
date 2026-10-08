# Midterm technical report (Direction 42 — Video games)

## 1. Problem, motivation and goal

Players often decide which game to buy based on its approximate time investment. Game descriptions and tags hint at narrative structure, but completion duration varies greatly between short adventures and long role-playing games. The research question is: **How accurately can structured game metadata predict the typical main-story completion time, in hours?** Users include players comparing games, curated gaming platforms, and analysts exploring game segmentation. This midterm answers the structured-data part; testing the incremental usefulness of written descriptions is a Final-stage goal.

**Hypothesis:** A supervised model using genres, release year, current/list price, developer history proxy, and tags will outperform predicting a single median duration for every game. No claim is made until evaluated on held-out games.

## 2. Dataset

Collection code: `src/collect_igdb.py` for the official IGDB time-to-beat endpoint and optional Steam app details; fallback `src/collect_mirror.py` for a public third-party Game Gauntlets compiled snapshot containing HLTB main-story times. The data card details what a row and each feature means. Source identity and collection time are serialized into the manifest rather than assigned an invented collection date.

Required sample size: at least **3,000 cleaned records** for this team's stated goal (the guide's formal minimum is 1,000). Training cannot begin without the real cleaned dataset. All raw selected records are kept as line-delimited JSON; the downloaded mirror snapshot is also preserved where applicable.

> **Fill after actual collection:** Before cleaning: see `data/clean/cleaning_report.json` → `raw_rows`; after cleaning: `cleaned_rows`; missing price/tags/year from the same file. If the actual run does not meet the size target, do not claim it does.

## 3. Preparation and data leakage prevention

Cleaning removes missing targets and invalid records, strips HTML, converts USD cents to dollars (prices), converts IGDB seconds to hours (target), de-duplicates game IDs/title-year pairs and retains missing predictor values. Models receive only the structured columns — **not** the game's title, its description, the target, an alternate time-to-beat field or source links. Text description is retained for the Final but intentionally not trained during Midterm.

Rows are split once into 80% training and 20% test with random seed 42. The exact IDs are saved in `fixed_split.csv` for Endterm/Final comparability. Numerical imputation and scaling, genre one-hot / multi-hot encoding, tag vocabulary, and developer-catalog tiers are learned from the relevant training fold. This avoids learning statistics or vocabularies from validation/test rows.

## 4. Models and metrics

Baseline: constant median-hour DummyRegressor. Week 3–4 regression comparisons: ordinary Linear Regression, Ridge Regression (regularization), KNN Regression (local similarity) and Decision Tree Regression (nonlinear splits). Models receive log1p-transformed training targets to manage a skewed positive target and predictions are transformed back to hours. No ensembles or neural networks are claimed at the Midterm.

Comparison criteria: 3-fold cross-validated MAE on the training partition for model selection; held-out MAE and RMSE (hours), R², median absolute error and MAPE/sMAPE (percent) for reporting. MAE measures mean absolute number of hours missed, RMSE penalizes large mistakes, R² compares variance explained, and percentage errors allow relative comparisons but can be unstable for very short games. The project's primary optimization metric is held-out **MAE in hours** with CV used for model selection.

> **Fill after actual training:** `outputs/model_comparison.csv` contains measured model scores; `outputs/experiment_manifest.json` records the model selected by training CV. Do not fill numeric scores until execution.

## 5. Error analysis and limitations

Generated charts: target distribution, top genres, held-out MAE comparison, actual-versus-predicted scatter and absolute error by time-to-beat bucket. Grouped errors for sufficiently represented genres go into `outputs/error_by_genre.csv`. Investigate the largest residuals in `test_predictions.csv`: games with unusual story length, episodic editions, open-ended gameplay and very limited metadata are likely difficult cases, though a specific cause must be verified before reporting it.

Limitations: non-random selection into time-to-beat databases; community estimate noise; varying sample counts; imprecise developer tier proxy; availability of store price varies by source/region/date; randomness in train/test differs from the future-releases use case; possible franchise similarities across both partitions. Source licensing/ToS are the team's responsibility. IGDB `hastily` and HLTB `main` cannot be described as perfectly interchangeable.

## 6. Plans for later stages

**Endterm (Week 9):** Compare tuned Random Forest/Extra Trees/boosting with the fixed split; K-means clustering of scaled categorical/numerical embeddings to describe game segments (clusters are analyzed, not used as labels); MLP on structured features. Retain the original split and examine validation curves.

**Final:** Add TF-IDF description features as a separate text block, test `structured only`, `description only` and `combined` models with identical data partitions and algorithmic controls, report CV variance plus confidence intervals for the paired test errors, and create a small local prediction app. An apparent decrease in MAE must be tested for robustness rather than simply declared significant.

## 7. Contributions and AI tools — **complete before defense**

| Team member | Actual responsibility (fill from work performed) |
|---|---|
| Partner 1 — **add full name** | Validate sources / run collection / inspect raw fields / data card — adapt to actual work |
| Partner 2 — **add full name** | Run cleaning / modeling / charts / slides — adapt to actual work |
| Both | Inspect results, report limitations and rehearse explanation of each model |

AI assistance disclosure: ChatGPT was used to help prepare code scaffolding, notebook text, slide structure and methodological documentation. Real dataset downloading, source-permissions verification, model execution, measured results, interpretation and independent validation are required before submitting the completed experiment. Change this statement if your actual use differs; do not conceal AI use or invent contributions.

## 8. References

- Instructor, *Course Project: Guide*, Machine Learning Algorithms, pp. 1–3 (user-supplied assignment guide).
- IGDB, [API documentation](https://api-docs.igdb.com/), especially `game_time_to_beats`.
- Steam, [store details endpoint](https://store.steampowered.com/api/appdetails).
- HowLongToBeat, [website](https://howlongtobeat.com/), community-contributed game completion-time summaries.
- Beliaev / Game Gauntlets, [Steam Game Database repository](https://github.com/leinstay/steamdb), GPL-3.0 for compilation; underlying content belongs to respective rights holders.
- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, 2825–2830 (2011).
