# Video game completion time prediction Endterm

Syrym Shadiyarbek and Yerdaulet Akhanov, SE2424. Direction 42, regression.

Can game metadata help a player choose a game that fits a time budget? This project predicts main-story duration using the same 5,229 games, target values and saved train/test split as the Midterm. Real IGDB and Steam observations add feature types that were missing before. The report gives the measured result and its limits.

## Open the submission

1. Read [the report](docs/ENDTERM_REPORT.md) or [its PDF](docs/Endterm_Report.pdf).
2. Open [the executed notebook](notebooks/Video_Game_Endterm.ipynb). Its conclusion answers the project question. With `RETRAIN=True`, Restart and Run All repeats the supervised and unsupervised experiments from the included source snapshots.
3. Open [the slides](docs/Endterm_Defense.pptx) or [slide PDF](docs/Endterm_Defense.pdf). Use [the seven-minute defence script](docs/DEFENSE_GUIDE.md) to rehearse.
4. Check [the feedback change log](docs/CHANGELOG.md), [Endterm criteria](docs/CRITERIA_CHECKLIST.md), [data card](docs/DATA_CARD.md), and [responsibilities and AI use](docs/CONTRIBUTIONS.md).

Everything needed to inspect the work is in this folder. The teacher does not need GitHub, API credentials or a live data connection. A first-time environment setup needs the Python packages listed in `requirements.txt`.

## Collection code and real data

The Endterm collector is [src/collect.py](src/collect.py). It requests IGDB metadata for the original game IDs, follows explicit Steam app links and saves timestamped raw responses. The original target-collection code is included at [reference/midterm_source/src/collect_igdb.py](reference/midterm_source/src/collect_igdb.py), with its environment requirements. It records IGDB `hastily` seconds divided by 3,600. Its legacy raw field name mentions HLTB; its recorded source is IGDB.

Start with `data/clean/games_model_inputs.csv` for the modelling dataset: it contains the 39 raw predictor fields with game ID, title, target and split for reference, and has no entirely empty predictor columns. `data/clean/games_enriched.csv` keeps the full 70-column audit view. The historical `price_usd` and `tags_text` fields remain empty only in that audit view. Models exclude them. Their replacements contain 2,861 observed USD prices and 3,333 games with Steam tags. Additional IGDB fields include themes, gameplay keywords, modes, perspectives, platforms and game type. The data card distinguishes all 70 stored fields from the 39 raw model inputs. Missing source values remain missing in the stored CSV; the model fits its computational imputation inside each training fold.

Prices are current US-store quotes. They do not describe launch prices. Explicit duration tags and alternate time-to-beat fields are excluded from predictors. The source checks, collection timestamps, response hashes and row-level cleaning audit are included.

## Reproduce from the saved snapshots

From the extracted project folder, use Python 3.12 and install the pinned dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/prepare.py
python src/audit_data.py
python -m unittest discover -s tests -v
python src/train.py
python src/cluster.py
python src/error_analysis.py
python src/plots.py
python src/validate_submission.py
```

The model run uses three training-only folds, seed 42 and MAE in hours. Both random forest and boosting have saved CV searches. The MLP uses a separate architecture/regularization search and internal validation for early stopping. PCA and KMeans fit training objects only. The fixed test has already been observed at Midterm, so results describe a stage comparison.

Open the notebook using the same environment. Leave `RETRAIN=True` for the full Restart and Run All check. Set it to `False` for a quicker review of the included results. The experiment manifest records package versions and dataset hashes. Training can take several minutes; CSV files under `outputs/search/` record all search candidates. Rerunning the models changes measured outputs, so regenerate the report and slides before preparing a new archive.

`run.ps1 -Stage Train` provides the same preparation, fitting and plotting sequence. `run.ps1 -Stage Validate` runs the integrity tests, source audit and submission checks. Both commands accept `-PythonExecutable` for an explicit interpreter path.

## Live collection, if required

Keep the submitted raw snapshots for comparable results. To inspect or resume the original collection, set `IGDB_CLIENT_ID` and `IGDB_CLIENT_SECRET` in the local process and run:

```powershell
python src/collect.py
```

Existing cached responses are reused. To make a new snapshot, work in a separate copy and retain the submitted version. The collector reads credentials from environment variables and does not save tokens or secrets. Credentials are unnecessary for offline reproduction. The original target collector and its CLI options are retained under `reference/midterm_source/`; rerunning it against the live API can return a different cohort and must not overwrite the saved benchmark.

## Plan for the Final

Use the real IGDB description/summary text retained for each `game_id`. Start with the included descriptions, which already link one-to-one to the same games. If a refresh is needed, request only these IDs through the official IGDB API with local developer credentials, retain request dates and the raw snapshot, and review the current developer terms before collection. The existing source-access evidence is in `data/raw/source_checks/`. No private profiles or individual reviews are needed.

Broad genres do not describe campaign structure well, and the current models underpredict long or open-ended games. Description terms may distinguish story campaigns from repeatable online play. Test this idea with three matched configurations: structured features, TF-IDF description features with Ridge, and combined structured plus text features. Fit the TF-IDF vocabulary, document frequencies and all other transforms inside each training fold. Select vocabulary and Ridge settings with the same training CV MAE. Compare the selected configurations on the saved test once, report CV spread and MAE/RMSE/median error, and inspect the same long-game and cluster error groups. A gain is an empirical question. Description text can also contain duration claims, so audit and remove direct time-to-complete statements before fitting. The local app will show a duration estimate with the observed limitations.

## Files to upload

Both partners must upload the same complete project contents to Moodle by 23:59 on the day before the defence. Use `Endterm_Shadiyarbek_Syrym_SE2424.zip` for Syrym and `Endterm_Akhanov_Yerdaulet_SE2424.zip` for Yerdaulet. These archives contain identical project contents. GitHub is an additional copy and does not replace the ZIP.

- `src/`: collection, preparation, model training, clustering, scientific plots and validation code.
- `notebooks/`: the executed notebook, including conclusions and visible outputs.
- `data/raw/`: real saved API responses, source checks and collection/hash manifests.
- `data/clean/`: the main `games_model_inputs.csv`, full enriched audit view, field definitions, saved split, coverage and cleaning/source audits.
- `outputs/`: searches, CV/test results, predictions, models, cross-model and cluster errors, and figures.
- `docs/`: report, slides, seven-minute script, criteria review, change log and contribution disclosure.
- `reference/`: Endterm rubric, supplied earlier requirements, original submission and original collection/modelling code.
- `tools/`: presentation, report and notebook construction scripts, kept separately from analysis code.
- `requirements.txt`, `run.ps1` and this README: environment and run instructions.

Large local dependency folders, caches and actual credentials are excluded. Both students should extract the final ZIP into a new folder, open its report/slides/notebook, and practise questions about each other's assigned areas.
