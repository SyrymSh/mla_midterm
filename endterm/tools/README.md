# Material build scripts

These scripts build the submitted materials from the source files and actual experiment outputs. The ZIP includes them for inspection. Reading the report or opening the editable PPTX does not require these tools.

Install the analysis environment first. For report and notebook authoring, run `python -m pip install -r tools/requirements.txt` from the project root.

- `python tools/notebook/build_notebook.py` creates the notebook with `RETRAIN=True`. This replaces the notebook file and clears its displayed outputs.
- `python tools/notebook/execute_notebook.py` runs a fresh kernel, performs a complete offline fit, and compares it with the existing experiment outputs. Run a command-line fit first using README. It saves the executed notebook and `outputs/reproducibility_check.json`.
- `python tools/report/build_report.py` builds the Markdown report and PDF using the saved results. Run after the experiments, clustering and error analysis.
- `presentation/README.md` documents the editable presentation builder, its Artifact Tool dependency, rendering and the PDF export.

Presentation scripts use a bundled Artifact Tool runtime. Their temporary `.build` files and Python caches are excluded from Git and the submission archive. The scripts and reference deck stay in the archive. No script contains API credentials.
