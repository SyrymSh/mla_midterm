# Data card — Video game completion time prediction

**Project:** Direction 42 — Video games | **Task:** Regression | **Target:** main-story time (hours) | **Observation:** One distinct video game / released game edition (after deduplication).

## Collection and primary sources

**Strict/direct route:** [IGDB API](https://api-docs.igdb.com/) — `game_time_to_beats` and `games`. IGDB needs Twitch OAuth developer credentials. `hastily / 3600` gives average seconds to credits without side content, expressed as hours. Optional [Steam Store API](https://store.steampowered.com/api/appdetails) gives descriptions, genres and list price where a linked Steam app ID is available.

**Credential-free fallback:** [Steam Game Database](https://github.com/leinstay/steamdb) from Game Gauntlets, which compiles Steam / IGDB / HLTB and other feeds. We fetch the published compressed JSON file via our own Python script; selected games use the **HLTB** `hltb_main_hours` field. **Disclosure:** Compiled data does not represent our own direct scraper of IGDB or HLTB; use the direct route if the instructor requires original source fetching. A source reference and origin fields are retained in the raw subset and provenance manifest. The upstream repository states GPL-3.0 for compilation, with third-party rights to underlying descriptions.

**Collection date, downloaded URL, record counts, snapshot SHA256, and sampling method:** *generated on the actual run* in `data/raw/collection_manifest.json`. The project code does not assert that collection took place unless the raw snapshot exists.

**Source permissions:** The students reported checking robots.txt and terms before registration. This notebook's tooling cannot independently certify that all source access/redistribution terms still permit bulk collection. Recheck the current source terms and report the exact URLs and dates. IGDB requires authentication; no-login assumptions about its official API are inaccurate.

## Field dictionary

| Field in cleaned CSV | Data type | Meaning / treatment |
|---|---|---|
| `game_id` | String ID | Stable game identifier for splitting and deduplication; **not a model feature** |
| `steam_appid` | Nullable integer | Steam store app ID, provenance/join key; not trained |
| `title` | Text | Game title, identification / error analysis only |
| `main_story_hours` | Positive float | **Target**: HLTB `hltb_main_hours` or IGDB `hastily / 3600` |
| `release_year` | Nullable numeric | Year of first/published release (out-of-range dates excluded to missing) |
| `price_usd` | Nullable float | US dollar list price. Raw `price_usd` is in cents; converted to dollars |
| `price_missing` | 0/1 | Was price absent in the raw metadata? |
| `primary_genre` | Category | Alphabetically first reported genre; other genres also retained |
| `genres_text` | Pipe-delimited categories | Multi-label game genres, binarized for **structured** model |
| `tags_text` | Pipe-delimited categories | Community/genre tags, binarized for **structured** model |
| `primary_developer` | Category | Alphabetically first reported developer; used to derive developer frequency tier |
| `description` | Text | Game description; **not used in Midterm training** (for Final NLP) |
| `description_word_count` | Integer | EDA/quality, **not a model feature** |
| `source_url` | String | IGDB / HLTB / Steam link for provenance, **not a model feature** |

**Derived within fitted pipeline:** `developer_tier` (one game, 2–4 games, 5–14 games, 15+ games, or unseen). This is a dataset-catalog frequency proxy and must not be presented as a validated AAA/indie classification.

## Cleaning and validation

- Reject missing/non-positive target, invalid title and blank/very short descriptions.
- Guard implausible target values (outside 0.1–1,000 hours), based on robust operational selection rather than population truth.
- Remove duplicate `game_id` and near-duplicate normalized title + release year. Loss of some legitimate editions is a limitation.
- Strip raw description HTML to keep a readable text field for the Final.
- Preserve missing numeric feature values, imputed only within training folds, not before splitting.
- Assert at least **3,000 real cleaned rows**; never impute a missing target or synthesize games.
- **Actual counts and missingness:** written to `data/clean/cleaning_report.json` after running collection.

## Representativeness, target validity and ethics

Observations reflect games with recorded completion-time metrics; casual/multiplayer/recent releases and obscure games may be underrepresented. HLTB and IGDB completion times use related but not identical definitions, so separate dataset runs cannot be directly compared as if they had the same outcome definition. Prices can vary with time and region, and human game descriptions can include language or misleading duration phrases. The table describes products/games, not private individuals. Do not redistribute raw publisher-provided description text beyond permitted uses.

## Reproducibility

All fetches are scripted. Use `data/raw/collection_manifest.json` for collection timestamp, SHA256 hashes and source method; `data/clean/fixed_split.csv` for frozen train/test rows (seed 42); `outputs/experiment_manifest.json` for features and evaluation details. Raw target and source fields remain in the JSONL. The prepared CSV is an analytical derivative.
