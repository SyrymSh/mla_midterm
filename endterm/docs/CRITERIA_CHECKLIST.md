# Technical checks and defence preparation

This checklist follows `reference/Endterm Rubric.pdf`, including the seven-minute presentation and five-minute question period. The earlier Midterm checklist is superseded. The rubric awards criteria 1-7 to the team and criterion 8 separately to each student.

File checks can establish that code, data and analyses are present and consistent. Presentation timing, independent understanding, actual personal contributions and Moodle submission require student actions.

The final verification completed two full fits: command-line training and a fresh-kernel notebook run with `RETRAIN=True`. All 13 model metric rows matched exactly; the maximum difference in selected-model predictions was 2.49e-14 hours. All 13 notebook code cells completed without errors. Eight integrity tests and the source/artifact checks passed. The report has ten reviewed pages; the presentation has nine reviewed slides, five native editable charts and five embedded workbooks. Evidence is in `outputs/reproducibility_check.json`, `validation_summary.json` and `artifact_review.json`.

| Criterion | Requirement and evidence | Verification or remaining action |
| --- | --- | --- |
| 1. Continuity and fixes, 10 points | `CHANGELOG.md` maps the teacher's actual feedback to fixes and before/after evidence. `data/clean/fixed_split.csv` preserves 4,183 train and 1,046 test IDs; target values match the Midterm. Collection code, raw responses and dates are included. | `audit_data.py`, the integrity tests and `validate_submission.py` check continuity. Notebook Restart and Run All with `RETRAIN=True` must reproduce the delivered scores within numerical tolerance. |
| 2. Tuned ensembles, 15 points | Random forest and gradient boosting both use three-fold training CV, with all preprocessing inside the fitted pipeline. Search candidates and selected settings are saved in `outputs/search/random_forest.csv` and `gradient_boosting.csv`. `model_comparison.csv` gives CV mean/spread and test scores alongside the Midterm-family references. | The test set never selects search settings. The report explains the search choices. The archived submitted Midterm result is distinguished from a new refit. |
| 3. Unsupervised analysis, 10 points | IGDB semantic descriptors are L2-normalized. PCA retains the chosen variance fraction; KMeans compares k and checks a second seed. `cluster_profiles.csv` interprets metadata; `cluster_error_summary.csv` and `cluster_error_examples.csv` connect groups to the archived Midterm Ridge's errors and the selected model. | PCA, encoders and clusters fit training objects only. Group errors are descriptive held-out checks. Named games support the interpretation, and low separation remains a stated limitation. |
| 4. MLP against the best earlier model, 15 points | Architecture/L2 settings use training CV. `neural_network_diagnostics.json` and learning curves record training loss and internal validation progress. The report explains the validation peak, stopping, restored weights and comparison with the strongest earlier model under MAE. | All final scores use the same test IDs and predictions transformed back to hours. Training loss uses standardized log-target space; dimensionless validation R2 uses the transformed target. |
| 5. Fair comparison and errors, 15 points | One results table includes the dummy, Midterm-family models, ensembles and MLP with CV mean/spread and test scores. `all_model_error_by_duration.csv`, `all_model_error_by_genre.csv` and `all_model_error_examples.csv` provide comparable errors and real objects across models. | Main metric MAE fits the time-budget question. RMSE, R2 and median error describe trade-offs. Residual figures and the report explain failure patterns rather than only ranking average scores. |
| 6. Conclusions, limits and Final plan, 10 points | The notebook, report and final slide answer the research question and state current limits. The Final plan names IGDB descriptions, access through the official API, links by `game_id`, campaign/open-ended-game motivation, and a structured/text/combined test with fold-fitted TF-IDF/Ridge. | The plan keeps training CV selection and the saved test split. Source terms must be reviewed before any future fresh collection; cached source-access evidence is included. |
| 7. Presentation, 10 points | Nine readable slides follow the question, fixes, data, models, results, errors, unsupervised analysis, MLP and conclusion. `DEFENSE_GUIDE.md` assigns both speakers a total of exactly 420 seconds. | Rehearse aloud to meet seven minutes. The planned allocation does not prove actual delivery time. Allow a further five minutes for questions. |
| 8. Individual answers, 15 points each | `CONTRIBUTIONS.md` gives assigned responsibilities. The defence guide covers tuning, leakage, group interpretation, MLP learning, errors and the partner's areas. | Each student must complete the stated review work, describe their actual contribution and explain the whole project independently. Personal work and oral understanding have not been independently confirmed. |

## Archive completeness

| Required item | Included location |
| --- | --- |
| Collection code and saved source observations | `src/collect.py`, `reference/midterm_source/src/collect_igdb.py`, `data/raw/`, `reference/midterm/data/raw/` |
| Clean data, saved split and data card | `data/clean/games_model_inputs.csv`, enriched audit view, saved split, `docs/DATA_CARD.md` |
| Runnable code, visible notebook outputs and run instructions | `src/`, `tests/`, `notebooks/Video_Game_Endterm.ipynb`, `requirements.txt`, `run.ps1`, `README.md` |
| Searches, unsupervised/MLP checks, results and errors | `outputs/search/`, other `outputs/` tables and figures |
| Change log, conclusions and Final source plan | `docs/CHANGELOG.md`, report, notebook and final slide |
| Slides for the seven-minute talk | `docs/Endterm_Defense.pptx`, `docs/Endterm_Defense.pdf`, `docs/DEFENSE_GUIDE.md` |
| Contributions | `docs/CONTRIBUTIONS.md` |
| Supplied instructions | `reference/Endterm Rubric.pdf`, earlier rubric and guide |
| Artifact construction code | `tools/presentation/`, `tools/report/`, `tools/notebook/` |

Both partners upload the same complete project contents. The confirmed group is SE2424. The archives use the required filename pattern: `Endterm_Shadiyarbek_Syrym_SE2424.zip` and `Endterm_Akhanov_Yerdaulet_SE2424.zip`. The deadline is 23:59 on the day before the defence. The teacher said GitHub would not be checked, so no required evidence may depend on a repository link.

## Qualifications to keep visible

- The saved test was already seen at Midterm. Reusing it supports a stage comparison and limits claims about unseen future games.
- Stored source gaps remain missing. `games_model_inputs.csv` has no entirely empty predictors. Historical empty price/tag columns remain only in the audit view, excluded from model inputs; their new populated replacements have measured coverage.
- The original range and text filters exclude some real short games, and normalized-title/year deduplication can collapse editions. Endterm preserves them for continuity.
- Current prices, selective Steam coverage, repeated app IDs and uncertain online-game labels limit interpretation. Neither a large error nor a higher score licenses changing source labels.

A successful automated validation covers the file requirements it tests. It cannot certify a seven-minute performance, completed personal work, the teacher's final judgement or a Moodle upload.
