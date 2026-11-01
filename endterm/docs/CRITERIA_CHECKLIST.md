# Criteria checklist

This review uses the supplied **Midterm Rubric** and **Project Guide**. The team confirmed that no separate Endterm brief has been issued. The guide requires ensembles and tuning, unsupervised analysis, and a neural network at Endterm. It reserves text/image features and a local app for Final.

The statuses below describe evidence in the files, not a predicted mark. Oral performance and administrative actions cannot be verified by code.

| Criterion | File evidence | Status |
| --- | --- | --- |
| 1 Question and purpose | Report and notebook define main-story duration regression for players planning their time. | Covered |
| 1 Own collection code, raw snapshot and date | collect.py, archived midterm collection code, timestamped IGDB/Steam responses, collection manifests and source hashes. | Verified |
| 1 Data minimum and format | 5,229 distinct games; numeric/categorical features and a real description field retained for Final. | Verified |
| 1 Source and ethics check | DATA_CARD.md records sources, API access, robots/terms checks, dates, attribution and absence of private-person data. | Documented; source acceptance remains the teacher's judgement |
| 1 Registration and team eligibility | Same-practice-group membership and topic approval are not documented in these files. | Team confirmation needed |
| 2 Cleaning and before/after evidence | audit_data.py reproduces 5,273 raw rows, 34 range/text exclusions and ten duplicates, leaving the frozen 5,229 rows. cleaning_audit.csv gives every raw row's outcome. Reasons and the inherited short-game/edition limitations are explained. | Verified |
| 2 Real additional feature types | Saved-source verification checks the observed Steam prices and tags. Coverage and dictionary files distinguish raw inputs, provenance and derived fields. | Verified |
| 2 Leakage check | Time-to-beat alternatives, explicit duration tags, identifiers and descriptions are excluded. Encoders, imputers, scales and catalog tiers fit training folds only. Steam-ID overlap is disclosed and checked by sensitivity analysis. | Verified with the stated overlap limitation |
| 2 At least three interpreted plots | Target, genre and price plots have quantitative interpretations and lead to metric, feature and model choices in the report/notebook. | Covered |
| 3 Fixed split | Original 4,183/1,046 split is byte-identical, and targets/game IDs are unchanged. | Verified |
| 3 Dummy and at least three Week 3-4 models | Median dummy, OLS, Ridge, decision tree and scaled KNN are compared using the same original-feature family. | Verified |
| 3 Pipelines, CV and parameter reasons | Three training-only folds; model_specifications.csv explains every model and its main settings. | Verified |
| 3 Notebook with visible outputs | The delivered notebook is executed from top to bottom and checked for errors. Cached experiments are identified; RETRAIN=True reproduces model fitting. | Execution checked by validate_submission.py |
| Endterm ensembles and tuning | Random forest, ExtraTrees, and an eight-setting gradient-boosting search with saved fold results. | Verified |
| Endterm unsupervised analysis | Training-only normalized IGDB descriptors, PCA, k=2..6 KMeans, silhouette, stability and profiles. Weak separation is explained. | Verified |
| Endterm neural network | Four-setting MLP comparison, target transformation, early stopping, learning curves and CV/test results. | Verified |
| 4 Metric and complete comparison | One table contains all 13 comparisons, CV mean/spread and test scores. MAE is justified; RMSE, R2 and median error are also reported. | Verified |
| 4 Errors and examples | Residuals, genre/duration errors, source-linked long-label audit, actual failure cases and coverage/overlap sensitivity checks. | Verified |
| 5 Conclusions, limits and next stage | Report answers the question, connects Week 5-8 methods to midterm weaknesses, and gives a Final plan. | Covered |
| 5 Contributions and AI tools | CONTRIBUTIONS.md identifies assigned responsibilities and the AI tool and its uses. Actual completed individual work has not been confirmed. | Disclosure present; factual individual contributions need confirmation |
| 6 Slides | Nine readable slides, labelled figures/tables and speaker notes. The complete model table is in the report/notebook. | File checks complete |
| 6 Both speakers and 7-10 minutes | DEFENSE_GUIDE.md allocates a planned nine-minute talk to both partners. | Timed rehearsal needed |
| 7 Each partner understands the whole project | Method questions and cross-topic preparation are provided in DEFENSE_GUIDE.md. | Cannot verify without each student's answers |
| Submission | Both partners upload the same package by 23:59 on the day before the defence. | Student action needed |

## Important qualifications

- The fixed test was already observed at midterm. It supports a stage comparison, not a new independent test.
- Current prices and metadata support retrospective catalog prediction, not launch-day forecasting.
- The main result retains extreme and open-ended-game labels. Reported subset scores do not replace the full-cohort result.
- IGDB uses developer OAuth credentials. This is documented explicitly; the team should confirm the teacher accepts authenticated APIs under the guide's no-login wording.
- Keeping the AI disclosure is part of criterion 5. The rubric explicitly requires the AI tools used and their purpose.

The remaining items require real team actions. This checklist does not claim that a rehearsal, upload, registration approval or individual review has happened.
