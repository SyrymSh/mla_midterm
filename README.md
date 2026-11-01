## Endterm project

The completed Endterm project, report, slides, executed notebook and real IGDB/Steam data are in [endterm/](endterm/README.md).

# Video Game Completion Time Prediction — Midterm (Direction 42)

**Task:** Regression | **Target:** main-story completion time in **hours** | **Minimum:** 3,000 real, distinct games (1,000 is the formal guide minimum; this project's registration goal is 3,000+)  
**Guiding question:** How accurately can we predict a game's main-story completion duration using only structured metadata: genres, a developer-catalog tier proxy, initial/current USD list price, release year and player tags?  
**Reserved for Final:** Does TF-IDF/other NLP on game descriptions improve accuracy over structured metadata alone?  
**Reserved for Endterm:** Clustering, ensembles, hyperparameter optimization and neural networks.

This repo is a **working reproducible submission package**, **not a fabricated finished experiment**. Real data downloads require Internet access. Results/slides are generated from actual rows after running the collection commands; no test scores are invented. A notebook, script, data card, evaluation plots, split, data and slides belong in your Moodle upload **after running the pipeline**.

## Start here — recommended direct collection

**IGDB developer credentials are required for the official IGDB API** (Twitch OAuth client credentials), contrary to the assumption that IGDB can be scraped without authentication. Register an application in the Twitch developer console and keep credentials out of uploaded files and source control.

```powershell
cd video_games_midterm
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
$env:IGDB_CLIENT_ID = 'YOUR_ID'
$env:IGDB_CLIENT_SECRET = 'YOUR_SECRET'
python src/collect_igdb.py --desired 6500 --min-rows 3000 --max-pages 50
python src/prepare.py --min-rows 3000
python src/train.py
python src/build_slides.py
jupyter notebook notebooks/Video_Game_Midterm.ipynb
```

On macOS/Linux, replace the activate command with `source .venv/bin/activate` and `export IGDB_CLIENT_ID=...`, `export IGDB_CLIENT_SECRET=...`.

The official IGDB `game_time_to_beats.hastily` represents seconds to main credits without significant side content, converted to hours by dividing by 3,600. `normally` *includes extras* and is not the selected main-story target. Some IGDB game records lack Steam links, prices or community tags, and the script does **not** fabricate them. `--steam-enrich-limit 500` optionally queries up to 500 linked Steam games at a respectful request rate. Availability, source access, and final eligible sample size must be checked by running the code.

## Credential-free fallback — publicly compiled snapshot

If you do not have IGDB credentials, download the [Game Gauntlets / Steam Game Database compilation](https://github.com/leinstay/steamdb) with code:

```powershell
python src/collect_mirror.py --desired 6500 --min-rows 3000
python src/prepare.py --min-rows 3000
python src/train.py
python src/build_slides.py
```

The fallback downloads the published compressed JSON snapshot and **reservoir-samples actual matched records** with known `hltb_main_hours`. Upstream blends data originally from HowLongToBeat, IGDB, Steam, etc. This is **not a direct HLTB scraper or direct IGDB collection**, and the teacher may interpret the "collected with your own scraping/API code" rule to require direct-source requests. Use the IGDB route for the strictest interpretation and disclose the method in your defense. Upstream states **GPL-3.0** for the compiled dataset; underlying publisher descriptions have separate rights. We cannot confirm third-party scraping permissions on your behalf.

If the large GitHub LFS download fails, save the repository's `steamdb.min.json.gz` manually, then run:

```powershell
python src/collect_mirror.py --source-file "C:\path\to\steamdb.min.json.gz" --desired 6500
```

The full downloaded upstream `.gz` is a sizeable raw snapshot; if too large for Moodle, upload the selected unmodified `games_raw.jsonl`, `collection_manifest.json` and a stable source/download link, or provide a link to your raw full snapshot (per project guide).

## What is generated

| Artifact | Description |
|---|---|
| `data/raw/games_raw.jsonl` | Unmodified upstream source fields for sampled games; actual observations |
| `data/raw/collection_manifest.json` | Collection date (UTC), URL, SHA256, original/eligible/retained counts, source provenance |
| `data/clean/games_clean.csv` | Clean game-level data, includes original description field for Final |
| `data/clean/cleaning_report.json` | Exclusion counts, cleaned rows and missingness |
| `data/clean/fixed_split.csv` | Persistent, seeded train/test game IDs (80/20, seed 42) |
| `outputs/model_comparison.csv` | **Measured**, held-out MAE, RMSE, R², MedAE, MAPE, sMAPE + train CV MAE |
| `outputs/test_predictions.csv` | Actual and predicted hours for held-out games |
| `outputs/error_by_genre.csv` | Genre-level held-out error (n ≥ 10) |
| `outputs/experiment_manifest.json` | Reproducible run metadata and selected model |
| `outputs/best_structured_model.joblib` | Structured-feature regression pipeline |
| `outputs/figures/*.png` | Data distribution, genre distribution, model MAE, observed-vs-predicted and errors |
| `notebooks/Video_Game_Midterm.ipynb` | Study notebook with data card, EDA, training and interpretations |
| `slides/Video_Game_Midterm_Defense.pptx` | 9-slide editable defense (rebuilt from actual metrics when available) |
| `docs/DATA_CARD.md` | Data dictionary + provenance, limitations |
| `docs/MIDTERM_REPORT.md` | Methods, model choice, threat analysis, next stage |
| `docs/DEFENSE_NOTES.md` | Speaking notes and anticipated defense questions |

### Modeling method

The frozen test set is 20% of the cleaned rows. Preprocessors fit on each training fold only, avoiding categorical-imputation, normalization, developer-frequency and tag-vocabulary leakage. The median dummy model is the baseline. Ordinary least squares, Ridge regression, KNN and a limited-depth decision tree are fitted to log1p of the target and decoded to hours for evaluation. Model **selection** uses 3-fold shuffled CV MAE on training rows only, not test MAE. Models and baselines are evaluated in original hours. All predictors are structured game metadata; **game descriptions are held for the Final**.

A developer's "tier" is defined *only* as its **frequency of game appearances in the training catalog**, not actual studio headcount or financial AAA/indie status. Price is USD list price in cents in upstream raw data and becomes dollars after dividing by 100. Missing price/year are imputed within each training fold; targets are **never imputed**. Dataset row counts are enforced at cleaning and model stages; errors stop the run, rather than replacing missing games with made-up observations.

To re-run models on the exact same collection:

```powershell
python src/train.py
python src/build_slides.py
```

If the collection itself changes, saved IDs will no longer match. Preserve the original collection for comparison with Endterm/Final. Use `--reset-split` **only if deliberately abandoning** the old test set.

### Responsible sourcing and limitations

- Check **the current IGDB API agreement / Steam API rules and robots.txt** before collecting. The group's reported check is not independent verification. API developer credentials are different from logging into individual game pages.
- Direct source target: `hastily` is the IGDB main-credits average; mirror target: HowLongToBeat `hltb_main_hours`. They are related but not perfectly identical definitions. **Do not silently combine them** into one target.
- HLTB averages are user-submitted, may have unequal sample sizes and favor popular games. Multiple editions/releases may create near duplicates; cleaning deduplicates title/release year and game ID.
- Store price is time/region dependent, not historical launch price. Some games lack price/tags, especially on the IGDB route. An all-missing feature adds no predictive information.
- Random game-level split evaluates unseen *games* within the data pool; it does not establish accuracy on genuinely future releases or entirely new publishers.
- Do not claim statistical significance for NLP gains until the Final's matched-split experiments and appropriate confidence intervals are performed.

### Self-check

```powershell
pytest -q tests
```

The unit tests use small, synthetic rows **solely to test code**. They are not the project dataset, do not appear in any model results or slides, and prove nothing about prediction accuracy.

## Moodle checklist

1. Run collection, verify **3,000+ cleaned real games** and source compliance.
2. Execute the notebook completely (`Kernel > Restart & Run All`).
3. Rebuild slides so results are numeric rather than marked "pending".
4. Fill the two team names, group, collection verification, task split and AI-use notes.
5. Both team members submit the **same** notebook, collection code, raw & clean data (or source link if large) and slides by 23:59 the day before the defense; be ready to defend any step of the project.

**Guide:** `Project Guide.pdf`, pp. 1–2. The referenced Midterm grading rubric was *not* provided, so no exact rubric score can be guaranteed.
