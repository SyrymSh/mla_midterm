# Presentation authoring sources

These files rebuild the nine-slide Endterm presentation from the saved experiment CSV/JSON files. They are kept with the project so the figures and conclusions can be updated after a rerun. The previous deck is retained under `reference/` to document the source design.

- `build_deck.mjs` uses the Artifact Tool JavaScript API, native tables and native charts. Each chart receives an embedded workbook containing the exact plotted data.
- `build_pdf.py` places the reviewed slide renders into a nine-page PDF. It requires ReportLab and produces the same visible slides as the PPTX build.
- `build.ps1` accepts explicit runtime locations. It runs both scripts and stops on an error.
- `inspect_source.mjs` records and renders the previous deck for comparison.

The model experiments must run before this build. Required outputs include `model_comparison.csv`, `all_model_error_by_duration.csv`, `cluster_error_summary.csv`, `neural_network_diagnostics.json` and the saved search plans. Speaker notes come from `docs/defense_notes.json`.

Use the bundled Artifact Tool Node runtime and presentation validation tools, or an environment with equivalent API support. Pass its paths to `build.ps1`; the project does not store a machine-specific runtime path. Set `PYTHONPATH` to the Python packages installed for the project when needed. ReportLab is an optional authoring dependency (`pip install reportlab`), separate from model reproduction.

The `.build/` directory contains temporary exports, slide PNGs and the validation receipt and is excluded from Git and submission packaging. The scripts and source deck are included.
Chart-only numeric values are rounded to 12 significant digits for Excel workbook compatibility. Displayed comparisons use fewer decimals, and the original experiment CSV/JSON files retain full numeric precision.
