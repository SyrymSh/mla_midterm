# Defence guide

The Endterm defence has seven minutes for the presentation and five minutes for questions. Either partner may be asked about any part of the project. The allocations below total exactly 420 seconds. They are a rehearsal plan; actual delivery still needs a timer.

| Slide | Speaker | Seconds | Cumulative time | Topic |
| --- | --- | --- | --- | --- |
| 1 | Syrym | 25 | 0:25 | Question and beneficiary |
| 2 | Syrym | 50 | 1:15 | Feedback fixes and source coverage |
| 3 | Syrym | 45 | 2:00 | Fixed target, split and pipelines |
| 4 | Yerdaulet | 50 | 2:50 | Controlled feature comparison |
| 5 | Yerdaulet | 60 | 3:50 | Tuned ensembles and model selection |
| 6 | Yerdaulet | 55 | 4:45 | Errors across models |
| 7 | Syrym | 50 | 5:35 | PCA, clustering and group errors |
| 8 | Yerdaulet | 45 | 6:20 | MLP learning and comparison |
| 9 | Both | 40 | 7:00 | Conclusion and Final source plan |

The spoken script leaves time to point at the figures. Use the exact selected settings and measured values in the delivered slides. Do not explain every row in the full results table during the seven-minute talk. Keep it available in the notebook for questions.

## Spoken script

### Slide 1: Question and beneficiary

Syrym, 25 seconds.

Our question is whether game metadata can help a player choose a game that fits their time budget. We predict main-story hours using the same Midterm games and split, with additional real data and the required Endterm methods.

### Slide 2: Feedback fixes and source coverage

Syrym, 50 seconds.

The teacher identified missing collection code, missing conclusions and empty columns. The ZIP now contains both collectors, raw responses, collection dates, the notebook conclusion and slides. All 5,229 games remain. The old price and tag fields had no observations. Their replacements contain 2,861 USD prices and 3,333 rows with Steam tags. Themes, perspectives and modes add other feature types. Source gaps stay missing in the saved data; imputation belongs inside model training.

### Slide 3: Fixed target, split and pipelines

Syrym, 45 seconds.

The target is IGDB's hastily seconds divided by 3,600: time to credits without notable extras. October labels and the 4,183 training and 1,046 test games stay fixed. Encoders, imputation, scaling and developer frequencies fit inside training folds. We exclude alternate completion-time fields and explicit duration tags. The test was already seen at Midterm, which limits generalization claims. Offline reproduction needs no live API access.

### Slide 4: Controlled feature comparison

Yerdaulet, 50 seconds.

We hold each estimator's settings fixed while adding feature families. Ridge and ExtraTrees each compare original features, expanded IGDB, and IGDB plus Steam. Added IGDB descriptors help both comparisons. Steam's extra value depends on the estimator. Prices are current US-store quotes, and Steam covers only part of the cohort. These limits remain visible in the interpretation.

### Slide 5: Tuned ensembles and model selection

Yerdaulet, 60 seconds.

MAE in hours fits the time-budget question. Both forest and boosting now have eight-setting searches using three training folds. Forest varies tree count, leaf size and feature fraction. Boosting varies tree complexity and regularization. The MLP has a separate four-setting search. We select the lowest training CV MAE before calculating test scores. The selected model is random forest, with CV MAE 8.636 and test MAE 10.37 hours. Forest and boosting are nearly tied in CV. Boosting has lower test MAE, but the training-CV rule selects forest. The full table also reports RMSE, R-squared and median error.

### Slide 6: Errors across models

Yerdaulet, 55 seconds.

We now compare duration and genre errors for every model and inspect the same real cases across models. For games up to five hours, forest MAE is 1.80 hours, versus boosting at 1.91. For games over five and up to twenty hours, boosting does better: 4.14 versus 4.30 hours. Very long and ongoing online titles remain difficult. Their completion labels can be hard to interpret as a main story. We keep these games and source labels in the result. RMSE exposes extreme failures, while median error describes a more typical mistake.

### Slide 7: PCA, clustering and group errors

Syrym, 50 seconds.

We normalize IGDB semantic descriptors before PCA so heavily annotated games do not dominate distances. PCA and KMeans fit training objects only. We check explained variance, several values of k and a second seed. Low silhouette means the groups overlap. Their held-out errors now compare the archived Midterm Ridge and selected model. Ridge MAE is 3.62 hours in cluster zero and 19.76 in cluster five; forest reduces both. Old School RuneScape in cluster three is a fantasy MMO: actual 486.5 hours, forest prediction 20.81. These descriptive errors do not choose the models or k.

### Slide 8: MLP learning and comparison

Yerdaulet, 45 seconds.

The MLP searches two layer structures and two L2 penalties. It predicts a standardized log target and converts back to hours for evaluation. Training loss falls, while internal validation peaks around iteration 9. Training stops at 30 and restores the best validation weights. This controls overfitting. Curve units differ from hour MAE. The MLP's test MAE is 10.41 hours. Its CV MAE is higher than the selected forest's.

### Slide 9: Conclusion and Final source plan

Both, 40 seconds.

The additional metadata improves this stage comparison, but very long or open-ended games remain unreliable. For Final, we will test IGDB descriptions linked by the same game IDs. We will review API terms, fit TF-IDF inside training folds, and compare structured, text-only and combined inputs using the same CV metric. Campaign wording may distinguish story progression from repeatable online play. The comparison will test whether it helps.

## Questions both partners should practise

### Data and continuity

What is the target? IGDB `game_time_to_beats.hastily` seconds divided by 3,600, meaning time to the credits without notable extras. The October target values remain unchanged. The original field name mentions HLTB, but its source record identifies IGDB.

Where is the collection code? `src/collect.py` fetches the added IGDB and Steam metadata. `reference/midterm_source/src/collect_igdb.py` preserves the original target collector. Both are inside the ZIP, with raw observations and collection manifests.

Why did the Midterm need more data? The old price and tag fields had no observations. The Endterm adds populated feature types to the same games. It measures coverage and tests added value with fixed-estimator ablations.

Does a missing price mean free? No. The stored value remains missing. The model fits a predictor imputer inside each training fold. That computational value does not become an observed price, and the target is never imputed.

Why keep the same split? The saved 4,183/1,046 assignment makes stage results comparable. The test was already observed at Midterm, so it is a benchmark with earlier exposure, not an independent new evaluation.

What cleaning happened? The inherited target rule accepts finite labels from 0.1 to 1,000 hours. All 28 duration exclusions are below six minutes. Five descriptions fail the eight-word minimum and one title fails the two-character minimum. Ten normalized-title/year duplicates are removed. These filters can exclude real short games or different editions; the Endterm retains them for continuity.

What protects against leakage? Source identifiers, titles, descriptions, alternate completion-time fields and explicit duration tags are excluded from structured model inputs. Encoders, imputers, numeric scales and developer catalog counts fit training folds. Repeated Steam IDs across partitions remain a disclosed limitation, checked with grouped CV and overlap-free subsets.

### Tuning and comparison

Why MAE? Its unit is hours, matching the player's time-budget question. RMSE makes extreme failures more visible; median absolute error describes a typical error despite the long label tail. R2 is also reported, including weak values.

What does CV estimate? Each fold holds back different training games for validation. The mean supports selection and the sample standard deviation describes fold variation. The selected tuning score is not nested CV and can be optimistic.

What was tuned in the forest? The eight combinations vary tree count, minimum leaf rows and feature fraction. Tree count concerns ensemble stability and runtime; larger leaf sizes regularize trees; a smaller feature fraction decorrelates them. Selection uses training CV MAE.

What was tuned in boosting? Eight combinations vary maximum leaves, minimum leaf rows and L2 regularization. These control the complexity and regularization of the sequential model. The objective is absolute error in hours.

How do forest and boosting differ? A random forest averages bootstrap-trained trees. Boosting fits trees sequentially to improve the current predictions. ExtraTrees adds random split thresholds and serves as a fixed-setting feature-ablation comparator.

How was the model selected? By mean training CV MAE among full-feature candidates. Test scores describe the chosen comparison and do not select search settings. The selection manifest records the rule and the dataset/split hashes.

What is the Midterm reference? The submitted best model was Ridge under its training-CV rule. Its archived predictions remain available. The Endterm also refits the original-feature references using the current pipeline; those rows are labelled separately so small refit differences are visible.

Why test fixed-estimator feature sets? Keeping estimator settings fixed isolates the observed change when feature families are added. Comparing a new forest with an old linear model alone mixes feature and estimator effects.

### Unsupervised analysis and errors

What does normalization do here? It gives each multi-hot semantic descriptor vector equal L2 length before PCA. This reduces domination by games with more annotations. It does not use duration labels.

What does PCA explain? Variance in the encoded predictors, not duration. The retained-variance figure describes the full PCA space; the first two plotted components contain only part of that information.

Why choose that k? KMeans compares the declared values using sampled training silhouette. The second seed checks assignment stability. A stable assignment can still have weak separation, so low silhouette remains a limitation.

How are the groups relevant to prediction? Their held-out error summaries compare the archived Midterm Ridge and selected model, with named games and features. This uses the rubric's error-explanation route. Cluster labels are not added as predictors, and held-out group errors do not select k or the supervised model.

How do errors differ across models? Use `all_model_error_by_duration.csv` and `all_model_error_by_genre.csv` to compare matched subsets. Then open the same games in `all_model_error_examples.csv`. Explain a measured contrast using actual predictions rather than assuming one family wins everywhere.

Why are some errors huge? Ongoing online games can have very large labels that are difficult to interpret as main-story completion. The full-cohort result retains them. Raw labels are audited, and a large error alone is not a reason to change the source value or remove a test game.

What does the paired bootstrap interval mean? It resamples the same test games in pairs to compare the absolute errors of already fitted models. It describes conditional uncertainty on that cohort, without covering all training uncertainty or repairing earlier test exposure.

### MLP and Final plan

What does the MLP learn? Nonlinear combinations of the encoded structured inputs. Two layer layouts and two L2 penalties form the declared search. Numeric transforms, feature encodings and target transforms fit the training data within each CV fold.

What do the two learning curves show? Training loss and internal validation R2 use the transformed target. A falling training loss alongside weaker validation shows that fitting training details can stop helping validation. Early stopping restores the best validation weights. These curve units are different from final hour MAE.

Why compare the MLP with the best earlier model? The rubric asks whether it improves on the strongest preceding approach, including tuned ensembles. Architecture complexity and a higher R2 do not override the main MAE selection rule.

What exactly will be added at Final? IGDB description/summary text for the same game IDs. Start with the saved descriptions. A refresh would use the official API, local credentials, saved timestamps and a terms review. Compare structured, text-only TF-IDF/Ridge and combined inputs with fold-fitted vocabularies and the same CV metric/split. Campaign wording may explain information hidden by broad genres. Audit direct duration statements before fitting text.

Who did what? Each partner should describe tasks they actually completed and evidence they checked. The assigned review areas are in `CONTRIBUTIONS.md`; they should not be described as completed personal work until the students confirm them.

Which AI tools were used? OpenAI Codex assisted with code, runs, verification and preparation of the materials. Humanizer helped edit wording. Explain the resulting work independently and identify personal review or changes honestly.

## Five-minute question practice

Take turns asking one question from each of these areas: tuning, unsupervised interpretation, MLP learning, errors, and the partner's assigned work. Give a direct answer, then point to the relevant saved output. Practise navigating from the notebook to the raw record for one game without internet access.

## Before uploading

- Complete and record each person's actual review tasks in `CONTRIBUTIONS.md`.
- Rehearse aloud with both speakers. Stop at seven minutes, then practise five minutes of questions.
- Extract the final ZIP into a new folder and open the report, slides and executed notebook there.
- Check that the collection code, data, split and all shown results are inside that archive.
- Use `Endterm_Shadiyarbek_Syrym_SE2424.zip` and `Endterm_Akhanov_Yerdaulet_SE2424.zip`, containing the same complete project.
- Both partners upload the same complete contents to Moodle by 23:59 on the day before the defence.
