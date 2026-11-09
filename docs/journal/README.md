# Project journal

Store new durable project records in this directory, including experiment reports, review notes, decision records, handoffs, and long-running roadmaps.

- Do not recreate or write new records under the retired predecessor directory.
- Preserve historical results, including failed gates; add a new record instead of rewriting an earlier outcome.
- Name records `yyyy-mm-dd-unixtimestamp-topic.txt`, using lowercase, hyphen-separated topic words. `README.md` is exempt.
- For future records, use the actual record creation time in Unix seconds and the matching calendar date in `Asia/Hong_Kong` (UTC+08:00). Do not use the migration time or a later edit time.
- Keep stable reference documentation, such as `docs/ROADMAP.md` and `docs/AGENT_VERSIONS.md`, in their existing locations.

## Filename migration provenance

The four existing records were renamed without changing their historical prose. The two timestamp-prefixed records retain their existing Unix timestamps, `1793769600` and `1793794946`; both resolve to `2026-11-04` in `Asia/Hong_Kong`.

For the two date-only records, the migration uses the first-add git author Unix timestamp as a fallback because their filenames did not contain Unix seconds: `1794238038` for `gv20-pv21-500-results` and `1794227013` for `gv20-pv21-exploratory`. Both resolve to `2026-11-09` in `Asia/Hong_Kong`. These fallback timestamps document repository provenance, not verified original record creation times. This fallback applies to these migrated records; future records must use actual record creation Unix seconds and the matching `Asia/Hong_Kong` date as required above.
