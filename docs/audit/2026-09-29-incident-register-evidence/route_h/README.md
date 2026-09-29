# Route H — record drafts (W1.5 Phase 1, plan rev 4 Task 7)

Repository-only evidence: commits, repo documents, and the typed deploy-run table in `../deploy/deploy_runs.json`. Nothing here reads box data or `data/`.

| Path | What |
|---|---|
| `commits_code.txt`, `commits_docs_only.txt` | The 1,172 commits since 3/29, split into 806 code/config and 366 docs-only. |
| `commit_classification.tsv` | Each code/config commit's class (feature, preship_fix, incident_fix, latent_fix and so on), live dates, incident class, and a verbatim evidence quote from its message. Quoted outcome statements are operational commit text under exposure row X-01. The runtime-closure negatives' diff reviews are in `../negatives/`. |
| `episodes.md` | Working table of candidate episodes (E01–E111, L01–L09, EX1–EX3). Each has a preliminary disposition, tier and sources. |
| `DRAFTING_GUIDE.md` | The instructions the four drafting readers worked under. The paths in it point to the session scratch directory where the drafting ran. |
| `packs/<id>.json` | Per-episode evidence packs given to the readers. Each holds the episode row, plus for every cited commit: the authored time, subject, body, changed files, log-bound deploy interval and first successful candidate deploy run. |
| `drafts/<id>.json` | One register record per episode, validated by `records.validate(..., publish=False)` (draft mode). Fixture fields are empty until the Task 5/6 acceptance objects fill them. |
| `drafts/exclusions.json` | Episodes and former records that are out of scope, each with the reason. |
| `drafts/batch*.notes.md`, `drafts/batch*.ids` | The readers' notes and batch membership. |

## Lead consolidation (2026-09-29)
119 drafts became 115 records.

- **Merged:**
  - I-044 into I-095;
  - I-096 into I-047.
- **Excluded:**
  - I-026 and I-076: model quality.
  - I-208 (L08): not a deployed path.
  - I-103 and I-104: retired infrastructure, with no watchdog or restore relevance.
  - Episodes E70, E81, EX2 and EX3 are excluded too, with notes in `exclusions.json`.
- **Split:**
  - I-113 from I-106: the R2 restore bundle lacked `mdp_policy.npz`, observed on the Fly shadow host.
  - I-114 and I-115 from I-059: links 18 and 2, whose firings are reported.
- **Re-disposed under ruling 6 (continuing conditions):**
  - I-063 is now continuing from 6/11;
  - I-064 and I-074 are now observed incidents.
- **Install bounds:** every log-basis install bound was recomputed from the deploy timeline's observation points as `(not_live_before, live_by]`. There were 92.
- **Reports on fix steps:** operator reports that describe a fix step are cited by that step (ruling 6, fix-step clause).
- **Occurrence `links`:** I-050's two occurrences name the links they fire (7 and 4). Under ruling 6, a continuing occurrence ends at its earliest end event, including the fix installs of its own links.
- **Built fixtures:** the notes of I-203 (L03) and I-204 (L04) record their new strict expected-failure fixtures (Task 1).

Result: 0 draft-mode errors.
