# Changes since the Midterm

The teacher's feedback, supplied by the team from 2 November 2026, identified missing collection code, missing conclusions in the notebook and slides, empty columns, and the need to upload every required file in a ZIP because GitHub would not be checked. It also commented on the defence ("защита получше"). These points determine the submission contents and preparation plan.

| Feedback or observed weakness | Earlier evidence | Endterm response and evidence |
| --- | --- | --- |
| Collection code was not available in the assessed submission | The teacher could not inspect how the data had been collected. | `src/collect.py` is included with saved responses and manifests. `reference/midterm_source/src/collect_igdb.py` preserves the original target collector. README gives commands, credentials handling and offline reproduction steps. |
| Conclusions were missing from the notebook and presentation | The teacher reported no conclusion in those materials. | The notebook ends with an explicit answer to the research question, the selected model, measured error and limits. The final slide states the conclusion and Final source plan; the report explains the same result. |
| Empty columns gave little useful information | Historical price and tag fields have zero observations. | The same 5,229 games now have 2,861 observed USD prices and 3,333 rows with Steam tags, plus IGDB modes, themes, perspectives and other feature types. Coverage and raw-response checks distinguish observed values from model-only imputation. The main `games_model_inputs.csv` has no entirely empty predictors. Empty historical fields remain only in the enriched audit view and are excluded from models. |
| Defence feedback: "защита получше" | The teacher commented on the oral defence alongside the problems in the files. | The Endterm rubric requires a seven-minute talk. `DEFENSE_GUIDE.md` allocates exactly 420 seconds across nine slides and both speakers, followed by five minutes of questions. It includes a spoken script and method questions. Students still need a timed rehearsal. |
| GitHub would not be checked | A repository link cannot supply the assessed files. | The complete ZIP contains code, raw/clean data, split, executed notebook, figures, searches, error analysis, report and slides. README explains extraction and local execution. Both partners upload identical contents in the required filename format. |
| Random forest had fixed settings | Only boosting had a CV search among the ensembles. | Both random forest and boosting now use training-fold CV searches. Candidate settings, selected parameters and fold variability are saved under `outputs/search/` and in the comparison table. |
| Clustering did not explain model performance | Earlier profiles described durations and metadata only. | Cluster-level held-out errors compare the archived Midterm Ridge with the selected model. Named games and their raw metadata support the group interpretations. These descriptive checks do not select a model using test errors. |
| Error discussion concentrated on boosting | Other models had aggregate scores without comparable examples. | Duration/genre error tables and real examples cover every model. The report and notebook discuss differences between linear, forest, boosting and neural predictions. |
| Final text plan lacked a testable source proposal | The plan said to add description features. | The Final plan names IGDB descriptions, lawful API access, linkage by `game_id`, fold-fitted TF-IDF/Ridge, and a structured/text/combined comparison using the same split and MAE. |

## What stays fixed

The 5,229 game IDs, October target values and saved 4,183/1,046 train/test assignment remain unchanged. Source enrichment adds columns by a left join. No target is imputed or replaced, and no difficult test game is removed. The row-level cleaning audit reconstructs the original 5,273-to-5,229 preparation.

## Limits retained in the conclusions

The original test was already visible at Midterm. Current prices and metadata describe retrospective catalog prediction. Steam coverage is incomplete, 23 test rows repeat a Steam app ID present in training, and some online-game completion labels are hard to interpret. Grouped CV and subset checks describe those limits. Low clustering silhouette means the groups overlap. A higher score alone would not resolve these issues.
