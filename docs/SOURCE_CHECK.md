# Data access and source verification checklist

The registration states that the team reviewed robots.txt and source terms and that the subject is video games rather than personal information. **This is a team-reported check, not an independently verified access authorization.** Keep the actual evidence dated for your defense.

| Source | URL | What to verify before collection |
|---|---|---|
| IGDB API | https://api-docs.igdb.com/ | Twitch developer OAuth requirement; API terms and non-commercial limitations; appropriate request rate |
| HowLongToBeat | https://howlongtobeat.com/robots.txt | robots restrictions, data-use terms; this package **does not directly scrape HLTB** |
| Steam Store | https://store.steampowered.com/robots.txt | Automated-request rules and rights for store descriptions/price |
| Game Gauntlets compiled dataset | https://github.com/leinstay/steamdb | Upstream compilation license (states GPL-3.0), third-party rights, repository freshness and origin of each feature |

To fill after your verification:

- Verification date (local/UTC): **[fill]**
- Source route used (IGDB authenticated or mirror download): **[fill]**
- Robots/terms URL opened and conclusion: **[fill exact statements; do not copy unverified claims]**
- Does access require a developer credential? IGDB API: **yes**. Third-party mirror file: download public; not equivalent to original source scraping.
- Permission to republish the whole raw dataset or store descriptions in Moodle? **[check current license and course rules]**
- Actual dataset size / raw snapshot SHA: generated in `data/raw/collection_manifest.json`.

Avoid scraping around authentication or anti-bot mechanisms. Prefer documented IGDB API requests.
