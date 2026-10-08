# Defense notes — 7–10-minute midterm (9 slides)

**All statements about achieved scores must be rewritten using actual `outputs/model_comparison.csv`. Do not defend a slide still marked "pending".**

1. **Slide 1 (30 seconds):** “Our project predicts how long the main storyline of a video game takes. The output is a number of hours, not a genre or review category; this makes it a regression task.”
2. **Slide 2 (50 seconds):** “Players use expected duration to choose games. We ask how much predictive information comes from genre, developer history, price, release year and tags. Text descriptions will be studied at the Final stage so the comparison is fair.”
3. **Slide 3 (60 seconds):** “Our target comes from [STATE ACTUAL USED SOURCE]. If IGDB: `hastily` is seconds to credits, converted to hours. If mirror: it uses HLTB `hltb_main_hours` through a public compilation. We retained the collection date and raw records, and distinguish original API collection from a third-party dataset.”
4. **Slide 4 (60 seconds):** “Here are the actual row counts, missing-value shares and filtering rules. Each row represents one game. We removed games without a usable main-story target, and kept descriptions for later NLP.”
5. **Slide 5 (60 seconds):** “The time distribution is asymmetric, with many short games and some lengthy ones. [Only say this if the chart supports it.] Genres are not evenly represented; this matters for error analysis.”
6. **Slide 6 (75 seconds):** “80% training and 20% test, fixed random seed 42; IDs persist for later stages. We benchmark a median predictor against linear regression, Ridge, a tree and KNN. All data transformations fit only on the training folds. Selection uses cross-validation MAE, not the test results.”
7. **Slide 7 (75 seconds):** “The best cross-validated model was [NAME]. Its held-out MAE was [ACTUAL] hours, RMSE [ACTUAL] hours and R² [ACTUAL]. Compare its MAE directly with the baseline. Only report measured results and distinguish selection metric from final test metric.”
8. **Slide 8 (75 seconds):** “The largest mistakes come from [SUPPORT USING TEST_PREDICTIONS.CSV]. Long games may produce bigger absolute errors. Missing prices and low tag coverage limit what the model can learn. Source differences and audience reporting behavior mean these are estimates, not ground truth for every player.”
9. **Slide 9 (60 seconds):** “Next, we will tune ensembles and cluster games for Endterm; for Final we will compare structured-only to structured-plus-description TF-IDF with the same test split, then deploy a local demonstration app. Our individual contributions and AI use are listed here.”

## Questions the teacher may ask

**Why regression?** Continuous positive target in hours; classification would discard the numeric duration.

**What is the baseline?** Median completion time learned only from training data. A model should beat its held-out MAE to show useful structure.

**Why `hastily`, not `normally`?** IGDB `normally` includes optional activities, while `hastily` is closest to a minimal main-credit story. HLTB's `main` is a separate, related metric.

**Why log-transform the target?** Completion time is right-skewed; fitting on log1p reduces sensitivity to exceptionally long games. MAE/RMSE are computed *after* converting predictions back to hours.

**How is leakage avoided?** Imputation, scaling, developer-frequency and tag vocabulary all fit on training folds only. Duration fields other than target, descriptions, titles and source URLs are excluded from predictors.

**How can you reuse the split?** `data/clean/fixed_split.csv` stores each game ID's partition and remains unchanged unless explicitly reset.

**Is developer tier AAA versus indie?** No. It is a proxy based on the number of records for that developer in the training sample; no claim about workforce or budget is made.

**How do you know NLP helps?** We do not yet; the Final will hold splits and the regression model constant, compare paired errors, and use uncertainty estimates.

**Does the mirror route count as own scraping?** We scripted a download and sampling from a third-party public compilation, not an original HLTB or IGDB crawl; disclose this and choose the authenticated IGDB route if original collection is required.
