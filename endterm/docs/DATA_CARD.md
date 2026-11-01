# Video game completion time data card

One row describes one of the **5,229 real, distinct games** in the submitted midterm cohort. The beneficiary is a player choosing a game that fits a time budget. The task is retrospective catalog-level regression, not individual player behavior or prediction at launch.

The target is **IGDB `game_time_to_beats.hastily / 3600`**, to the credits without notable extras. It is frozen from **8 October 2026**. The original raw file unfortunately calls its field `hltb_main_hours`; its recorded `time_source` explicitly says IGDB, and no HLTB targets are mixed into this project. Descriptions remain real source text for the Final and are not encoded at Endterm.

## Collection and provenance

New metadata was collected on **2 November 2026 in Asia/Qyzylorda** (raw timestamps are UTC). The exact time for each response is stored beside it. The current collection contains 5,229 IGDB game responses, 3,334 rows with successful Steam metadata, 2,861 rows with explicit USD prices and 3,333 rows with Steam tags. See `data/raw/collection_manifest.json` for hashes, endpoints and request counts.

IGDB metadata comes from the authenticated official `/v4/games` API. Steam data comes from public `IStoreBrowseService/GetItems/v1`, `IStoreService/GetTagList/v1`, and batched Store `appdetails` with `filters=price_overview`. Matching uses IGDB's explicit source IDs/URLs, not guesses from similar titles. The returned top 20 Steam tags are saved both as source IDs and mapped names. Steam Browse and store endpoints are public access surfaces; their availability is not a promise of permanent supported API access.

Prices are **current US-region USD quotes**, converted from cents. They are not launch prices. A missing quote does not mean a free game and is never replaced by zero in the stored dataset. Missing raw values remain missing. Model-only predictor medians are fitted inside training folds and do not become observed data. No target is inferred or imputed.

## Cleaning and cohort preservation

The supplied midterm cleaning report records 5,273 raw rows, 34 invalid/incomplete exclusions and 10 duplicates removed, leaving 5,229 games. Endterm is a left join onto those exact rows: **zero new games, zero removed games, byte-identical split file and unchanged target values**. Original fields are preserved alongside clearly named new fields. The split remains 4,183 training and 1,046 test games. Cross-partition repeated Steam IDs affect 23 test rows; the overlap audit and sensitivity scores are supplied.

The inherited target rule requires finite values in 0.1–1,000 hours. The 28 excluded durations are all below six minutes; this is an inclusion rule, not evidence that these real source values are wrong. Five descriptions have fewer than eight words and one title has fewer than two characters, so they fail the text-readiness rules. Ten repeated normalized-title/year entries are removed to reduce duplication. These rules can exclude real short games or different editions. Endterm preserves the choices so the cohort remains comparable. `data/clean/cleaning_audit.csv` records every raw row's outcome; `cleaning_steps.csv` records before/after counts. `source_verification.json` records the source checks for every observed price and tag row.

## Field dictionary

There are 70 stored columns, including provenance and historical audit fields. **39 raw columns enter the full model pipeline**, which creates 473 numeric encoded features. These are different counts. Source IDs, URLs, titles, descriptions, timestamps and label-quality counts are not predictors.

| field | stored_type | meaning | model_input | observed_rows |
| --- | --- | --- | --- | --- |
| game_id | int64 | Frozen IGDB game identifier; one game per row. | False | 5229 |
| steam_appid | float64 | Steam ID stored in the October midterm submission. | False | 3343 |
| title | str | Game name in the frozen midterm snapshot. | False | 5229 |
| main_story_hours | float64 | Regression target: IGDB hastily seconds / 3600, frozen on 8 October 2026. | False | 5229 |
| release_year | float64 | Release year retained from midterm IGDB first release date. | True | 5227 |
| price_usd | float64 | Historical midterm price field, entirely missing; retained only for audit. | False | 0 |
| primary_genre | str | First genre in the original sorted IGDB genre list. | True | 5229 |
| genres_text | str | Pipe-separated original IGDB genre categories. | True | 5149 |
| tags_text | float64 | Historical midterm tag field, entirely empty; retained only for audit. | False | 0 |
| primary_developer | str | Original primary studio name; converted to a fold-fitted catalog frequency tier. | True | 5228 |
| description | str | Original real game description, reserved for Final and excluded at Endterm. | False | 5229 |
| description_word_count | int64 | Description length for provenance and audit, excluded from inputs. | False | 5229 |
| price_missing | int64 | Historical midterm price-missing indicator, constant and excluded. | False | 5229 |
| source_url | str | Original public IGDB game page. | False | 5229 |
| matched_steam_appid | float64 | Validated explicit Steam link for enrichment, never used as predictor. | False | 3372 |
| steam_match_method | str | Explicit external ID/URL or unmatched/ambiguous; no fuzzy title matching. | False | 5229 |
| steam_old_id_conflict | int64 | Whether old Steam ID disagrees with all current explicit candidates. | False | 5229 |
| igdb_metadata_available | int64 | Whether current IGDB metadata was retrieved for the game. | False | 5229 |
| steam_metadata_available | int64 | Whether Steam Browse returned success=1 for the matched app. | False | 5229 |
| igdb_collected_at_utc | str | UTC retrieval time for IGDB metadata. | False | 5229 |
| steam_collected_at_utc | str | UTC retrieval time for Steam Browse metadata. | False | 3372 |
| steam_price_collected_at_utc | str | UTC retrieval time for Steam price response. | False | 3372 |
| igdb_game_type | str | IGDB game type such as Main Game, DLC or Expansion. | True | 5229 |
| igdb_parent_game_id | float64 | IGDB parent game reference, audit only. | False | 998 |
| igdb_version_parent_id | float64 | IGDB version parent reference, audit only. | False | 192 |
| steam_name | str | Steam product name for manual link verification, excluded from predictors. | False | 3334 |
| steam_item_type | float64 | Source Steam Browse product type code; successful protobuf default is 0. | True | 3334 |
| steam_regular_price_usd | float64 | Current US regular/list price from explicit USD initial cents divided by 100. Not launch price. | True | 2861 |
| steam_current_price_usd | float64 | Current US selling price from explicit USD final cents divided by 100. | True | 2861 |
| steam_discount_percent | float64 | Source discount_percent for the same US USD quote. | True | 2861 |
| steam_price_currency | str | Source quote currency. Model prices accepted only when explicitly USD. | False | 2861 |
| steam_price_region | str | Requested Steam store country US, not the student location. | False | 3372 |
| steam_source_url | str | Public store page for the matched app. | False | 3372 |
| igdb_themes | str | Pipe-separated source names of IGDB themes. | True | 4749 |
| igdb_themes_count | float64 | Number of entries returned for IGDB themes. Missing array stays missing. | True | 4749 |
| igdb_keywords | str | Pipe-separated source names of IGDB keywords. | True | 4094 |
| igdb_keywords_count | float64 | Number of entries returned for IGDB keywords. Missing array stays missing. | False | 4094 |
| igdb_game_modes | str | Pipe-separated source names of IGDB game modes. | True | 5092 |
| igdb_game_modes_count | float64 | Number of entries returned for IGDB game modes. Missing array stays missing. | True | 5092 |
| igdb_player_perspectives | str | Pipe-separated source names of IGDB player perspectives. | True | 4651 |
| igdb_player_perspectives_count | float64 | Number of entries returned for IGDB player perspectives. Missing array stays missing. | True | 4651 |
| igdb_platforms | str | Pipe-separated source names of IGDB platforms. | True | 5229 |
| igdb_platforms_count | int64 | Number of entries returned for IGDB platforms. Missing array stays missing. | True | 5229 |
| igdb_game_engines | str | Pipe-separated source names of IGDB game engines. | True | 3122 |
| igdb_game_engines_count | float64 | Number of entries returned for IGDB game engines. Missing array stays missing. | False | 3122 |
| igdb_dlcs_count | float64 | Number of source-linked dlcs. Missing array stays missing; audit only. | False | 527 |
| igdb_expansions_count | float64 | Number of source-linked expansions. Missing array stays missing; audit only. | False | 216 |
| igdb_standalone_expansions_count | float64 | Number of source-linked standalone expansions. Missing array stays missing; audit only. | False | 79 |
| igdb_developers | str | Source company names explicitly marked developer. Audit only. | False | 4800 |
| igdb_developer_count | float64 | Count of source companies explicitly marked developer. | True | 4874 |
| igdb_publishers | str | Source company names explicitly marked publisher. Audit only. | False | 4710 |
| igdb_publisher_count | float64 | Count of source companies explicitly marked publisher. | True | 4874 |
| igdb_single_player | float64 | Binary source listing for single player. Missing mode/category list stays missing. | True | 5092 |
| igdb_multiplayer | float64 | Binary source listing for multiplayer. Missing mode/category list stays missing. | True | 5092 |
| igdb_coop | float64 | Binary source listing for coop. Missing mode/category list stays missing. | True | 5092 |
| steam_tags | str | Up to 20 ordered Steam community tags, mapped from source IDs using the saved dictionary. | True | 3333 |
| steam_tag_count | float64 | Number of returned top Steam tags, capped by the request at 20; not total lifetime tags. | True | 3333 |
| steam_categories | str | Source category ID set from player, feature and controller categories. | True | 3333 |
| steam_single_player | float64 | Binary source listing for single player. Missing mode/category list stays missing. | True | 3334 |
| steam_multiplayer | float64 | Binary source listing for multiplayer. Missing mode/category list stays missing. | True | 3334 |
| steam_coop | float64 | Binary source listing for coop. Missing mode/category list stays missing. | True | 3334 |
| steam_online_coop | float64 | Binary Steam category listing for online coop. Missing category data stays missing. | True | 3334 |
| steam_achievements_listed | float64 | Steam lists the achievements category. This is a binary listing flag, not achievement count. | True | 3334 |
| steam_workshop | float64 | Binary Steam category listing for workshop. Missing category data stays missing. | True | 3334 |
| steam_cloud | float64 | Binary Steam category listing for cloud. Missing category data stays missing. | True | 3334 |
| steam_full_controller_support | float64 | Binary Steam category listing for full controller support. Missing category data stays missing. | True | 3334 |
| steam_windows | float64 | Binary platform field from successful Steam platform data. Omitted protobuf booleans in a returned platform object mean false. | True | 3334 |
| steam_mac | float64 | Binary platform field from successful Steam platform data. Omitted protobuf booleans in a returned platform object mean false. | True | 3334 |
| steam_steamos_linux | float64 | Binary platform field from successful Steam platform data. Omitted protobuf booleans in a returned platform object mean false. | True | 3334 |
| steam_language_count | float64 | Number of returned language entries explicitly marked supported. | True | 3333 |
