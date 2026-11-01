# Video game completion time prediction Endterm

Syrym Shadiyarbek and Yerdaulet Akhanov. Real IGDB and Steam data. Direction 42, regression.

The project preserves all **5,229 midterm games**, their completion-time labels and fixed split. It expands **feature types**, rather than adding rows. The expanded model selected by training CV is **Tuned gradient boosting**, with frozen-test **MAE 10.26 h**, **RMSE 50.52 h**, and **R² 0.023**. Raw responses and actual experiment outputs are included.

## Start here

1. Read [the endterm report](docs/ENDTERM_REPORT.md) or [its PDF](docs/Endterm_Report.pdf).
2. Open [the executed notebook](notebooks/Video_Game_Endterm.ipynb). It shows stored, verified experiment outputs by default; set RETRAIN=True to recompute models.
3. Rehearse using [the defence guide](docs/DEFENSE_GUIDE.md), [team roles and AI use](docs/CONTRIBUTIONS.md), and [editable slides](docs/Endterm_Defense.pptx) or [slide PDF](docs/Endterm_Defense.pdf).
4. Consult [the data card](docs/DATA_CARD.md) and [criteria checklist](docs/CRITERIA_CHECKLIST.md).

## What changed

The old price and tag columns had no observations. The new snapshot has **2,861 explicit USD prices** and **3,333 rows with Steam tags**, plus IGDB themes, keywords, perspectives, modes, platforms and game type. Price is current US-store price, not historical launch price. Model inputs exclude explicit duration tags such as Short and all other time-to-beat fields. Missing values are retained in the stored data. No rows, prices, tags or metrics were invented.

The 13-model comparison includes the median baseline, ordinary linear regression, Ridge, a decision tree, scaled KNN, controlled Ridge/ExtraTrees feature comparisons, random forest, tuned gradient boosting and a neural network. PCA/KMeans uses no target labels. Model selection uses three-fold training CV MAE. The same 1,046 test games support comparison with the midterm; their outcomes were already known from that stage.

## Reproduce offline

Use Python 3.12.14 (the recorded environment) and the pinned requirements. Internet and API credentials are not required to rerun models from the included snapshots.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/prepare.py
python src/audit_data.py
python -m unittest discover -s tests -v
python src/train.py
python src/cluster.py
python src/plots.py
python src/validate_submission.py
```

Open the notebook in your notebook editor using this Python environment. It already contains the experiment outputs; set `RETRAIN=True` to rerun the experiments. The report is provided as Markdown and PDF, and the presentation as editable PPTX and PDF. If you rerun experiments, update the report and slides from the resulting CSV files.

`run.ps1` provides a local convenience command with an explicit PythonExecutable option. Its Validate stage runs the integrity tests, checks cleaning and source values, and validates the submission files; Train reruns preparation, experiments, clustering and plots. It uses `.venv` or Python on your PATH. Keep raw snapshots unchanged when comparing stages.

## Recollect live sources

Use environment variables IGDB_CLIENT_ID and IGDB_CLIENT_SECRET set locally in your terminal. Never put their actual values in a notebook, repository or submission. Then run:

```powershell
python src/collect.py
```

The collector resumes from saved responses. Existing snapshots are intentionally reused; to create a different collection, copy the project to a new directory and preserve this version first. It fetches only the original game cohort and explicit Steam matches. Credentials are not written to disk. API rate errors trigger bounded backoff. Review current source terms before recollection. See the saved source checks for the original access evidence and limitations.

## Submission contents

- `src/` contains collection, preparation, models, plots and validation.
- `data/raw/` contains real timestamped source responses, source checks and a hash manifest.
- `data/clean/` contains the expanded CSV, fixed split, field dictionary, coverage and audit files.
- `outputs/` contains CV/test results, search tables, predictions, models, figures and validation results.
- `notebooks/` contains the executed endterm notebook.
- `docs/` contains the report, slides, data card, defence guide, criteria checklist and contribution disclosure.
- `reference/` contains the supplied requirements, original submission and archived midterm collection and modelling code.

Both partners should review their assigned responsibilities, rehearse together and be ready to explain the entire pipeline. No commits or pushes were performed.
