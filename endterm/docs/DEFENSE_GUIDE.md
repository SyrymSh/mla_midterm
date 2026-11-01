# Defence guide

The course guide specifies a 7-10 minute talk followed by 3-5 minutes of questions. Either partner may be asked about any part of the project. The schedule below totals nine minutes, but it still needs an actual timed rehearsal.

| Slide | Speaker | Seconds | Main point |
| --- | --- | --- | --- |
| 1 | Syrym | 35 | Main-story duration estimates help players plan their time. |
| 2 | Syrym | 65 | The same games now have more observed feature types. |
| 3 | Syrym | 60 | Labels and split stay fixed; preprocessing fits training folds. |
| 4 | Yerdaulet | 65 | Fixed-estimator comparisons isolate feature-set effects. |
| 5 | Yerdaulet | 70 | Explain the classical models, ensembles, tuning and selection rule. |
| 6 | Yerdaulet | 65 | Explain MAE, uncertainty and long-game failures. |
| 7 | Syrym | 65 | Explain PCA, selection of k and weak cluster separation. |
| 8 | Yerdaulet | 65 | Explain the neural network and early stopping. |
| 9 | Both | 50 | State the conclusion, limitations and Final-stage plan. |

## Questions both partners should practise

**What exactly is the target?** IGDB `hastily` seconds divided by 3,600: reaching credits without notable extras. The October labels stay unchanged. The old raw field name mentions HLTB, but its recorded source is IGDB.

**Why the same split?** It makes the comparison with midterm meaningful: 4,183 training and 1,046 test games. The test was already observed at midterm, so it is a stage benchmark, not an independent new test.

**Why did the original data need work?** All prices were missing and all tags empty. The effective predictors were mostly year, genre and a developer catalog proxy. More rows would not supply the missing feature types.

**Where do prices and tags come from?** Explicit linked Steam app IDs lead to saved public API responses. Prices are current US USD quotes, not launch prices. Tag names are mapped from source IDs. The audit checks these against the saved responses.

**Does missing price mean free?** No. The stored value stays missing. Only the model's training-fold imputer supplies a computational median. It is not treated as an observed price. Targets are never imputed.

**What cleaning was done?** Of 5,273 raw rows, 28 durations are below the inherited six-minute minimum, five descriptions have fewer than eight words and one title has fewer than two characters. Ten normalized-title/year duplicates are removed. The audit reproduces all 5,229 retained games and records each reason. The range rule can exclude real short games, and matching titles can hide different editions. Endterm keeps these choices for comparability and changes no games or labels.

**What would leak the answer?** Other completion-time fields, explicit duration tags, or preprocessing fitted using test rows. These are excluded or fitted in folds. Repeated Steam IDs across partitions are disclosed and checked separately.

**Why MAE?** Its unit is hours, which fits the time-budget question. RMSE highlights large errors, R2 describes explained variation, and median absolute error describes a typical mistake when a few labels are extreme.

**What does CV estimate?** It holds back different parts of the training data for validation. The mean supports selection and the fold standard deviation shows variation. The tuning score is not nested CV, so it can be optimistic.

**Why OLS, Ridge, a tree and KNN?** They are the Week 3-4 references. OLS tests unrestricted linear relationships; Ridge regularizes them; a tree learns thresholds; KNN tests local similarity. All four use the original feature family. Numeric inputs are scaled inside each fold for KNN.

**Why those parameters?** The tree has depth at most ten and at least ten rows per leaf to limit complexity. KNN uses 25 distance-weighted neighbours to reduce reliance on one noisy game. Ridge uses the original alpha 20. Those reference choices were not selected using test error.

**How are the ensembles different?** Random forest averages bootstrap-trained trees. ExtraTrees adds more random split thresholds. Boosting builds trees sequentially; this experiment uses absolute-error loss in hours. The main settings and reasons are in model_specifications.csv.

**What was tuned?** Eight boosting configurations vary leaf count, minimum leaf rows and L2 penalty. Four MLP configurations vary architecture and L2 penalty. Selection uses training CV MAE.

**How was extra data tested fairly?** Ridge settings stay fixed across three feature sets, and ExtraTrees repeats the same comparison. Steam improves ExtraTrees here but gives no clear further gain to Ridge. A real feature can have little extra value after correlated fields are present.

**How does the neural network work?** The chosen hidden layers have 64 and 32 units, using ReLU/Adam and L2 penalty 0.1. It fits a standardized log target and converts predictions back to hours. A 15% internal training holdback controls early stopping. Training stops after 30 iterations and restores the best validation weights.

**Why can validation worsen while training loss falls?** The network can fit training-specific detail. Early stopping follows validation progress. The loss curve uses transformed-target units, not hour MAE.

**What does PCA explain?** Variation in encoded predictors, not duration. The main study uses normalized IGDB descriptors and excludes Steam coverage fields. It fits training rows only.

**Why k=6?** It has the highest sampled silhouette among the tested values 2-6. The score is only about 0.078, so groups overlap. Stable initialization does not prove good separation or six natural game classes.

**Why are some errors enormous?** Ongoing online titles can have very large labels that are hard to interpret as main-story completion. The full result retains them. The audit flags source records rather than inventing corrections or deleting difficult test cases.

**What does the bootstrap interval mean?** It resamples the same test games in pairs to compare absolute errors for fitted models. It does not cover all training uncertainty or repair earlier test exposure.

**What remains for Final?** Description features, a matched comparison with structured inputs, a local app and stronger independent validation where feasible. Those are Final-stage additions in the supplied guide.

**How were AI tools used?** Give the factual statement in CONTRIBUTIONS.md, then explain what each student personally reviewed, changed or verified.

## Before uploading

- Complete and confirm each person's assigned review tasks.
- Rehearse aloud with a timer, with both partners speaking.
- Practise questions on the other person's area as well as your own.
- Save the presentation and use the same archive for both uploads.
- Both partners upload by 23:59 on the day before the defence.
