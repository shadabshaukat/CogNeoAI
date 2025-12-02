# Trigram Title Search — Fast Patterns and Indexing

If you see a Parallel Seq Scan with `similarity(md_title_lc, :q) > const`, you’re likely bypassing the GIN trigram index. Use the `%` operator with a reasonable `set_limit()` threshold and KNN ordering (`<->`) to get index-accelerated top‑K.

## Enable extension and index

```sql
-- 1) Ensure pg_trgm is available
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 2) Ensure md_title_lc has a trigram GIN index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_title_trgm
ON embeddings USING GIN (md_title_lc gin_trgm_ops);
```

Notes:
- `md_title_lc` should be a lowercased normalized title field.
- If you don’t have `md_title_lc`, create a stored/generated column or use an expression index: `GIN ((lower(md_title)) gin_trgm_ops)` (stored column is usually better).

## Use an index-friendly query

Instead of:
```sql
-- Often forces a seq scan and is too strict at 0.7–0.8
SELECT DISTINCT md_title, similarity(md_title_lc, :q) AS score
FROM embeddings
WHERE similarity(md_title_lc, :q) > 0.7
ORDER BY score DESC
LIMIT 10;
```

Use:
```sql
-- 1) Choose a threshold; 0.35–0.45 is typical for recall
SELECT set_limit(0.40);

-- 2) Filter by % (uses the GIN index) and order by KNN distance (<->)
--    Also include md_type filter with an array param when available
SELECT md_title,
       similarity(md_title_lc, :q) AS score
FROM embeddings e
WHERE
  (:type_eq IS NULL OR e.md_type = ANY(:type_eq))
  AND e.md_title_lc % :q
ORDER BY e.md_title_lc <-> :q   -- KNN GIN ordering (top-K)
LIMIT :shortlist;                -- e.g., 200–1000
```

Parameters:
- `:q` should be normalized like `md_title_lc` (lowercased, trimmed).
- `:shortlist` is the top candidate pool (e.g., 200–1000). You can then re-rank or dedupe if needed.

Why this is faster:
- `%` with a set `set_limit()` uses the trigram GIN index to find similar strings.
- `ORDER BY md_title_lc <-> :q` uses KNN GIN to sort nearest neighbors efficiently.
- Plain `similarity()` predicates generally don’t trigger the same index access path.

## Practical thresholds

- Start with `set_limit(0.40)`. If recall is too low, reduce to `0.35`. If precision is needed, increase to `0.45–0.50`.
- Values like `> 0.7` are often too strict and return empty sets unless titles are nearly identical.

## DISTINCT vs GROUP BY and de-dup

`DISTINCT md_title` can be expensive. Instead, limit first, then dedupe:
```sql
WITH cte AS (
  SELECT e.md_title, e.md_title_lc
  FROM embeddings e
  WHERE e.md_title_lc % :q
  ORDER BY e.md_title_lc <-> :q
  LIMIT :shortlist
)
SELECT md_title
FROM (
  SELECT md_title,
         ROW_NUMBER() OVER (PARTITION BY md_title ORDER BY md_title_lc <-> :q) AS rn
  FROM cte
) s
WHERE rn = 1
LIMIT 10;
```

## Example end-to-end

```sql
-- Lowercase your query in app code or do it in SQL:
-- :q_norm = lower(:q_original)

SELECT set_limit(0.40);

SELECT e.md_title,
       similarity(e.md_title_lc, :q_norm) AS score
FROM embeddings e
WHERE
  (:type_eq IS NULL OR e.md_type = ANY(:type_eq))
  AND e.md_title_lc % :q_norm
ORDER BY e.md_title_lc <-> :q_norm
LIMIT 500;
```

## Troubleshooting checklist

- Confirm the trigram extension and index exist.
- Ensure your query string is normalized similarly to `md_title_lc`.
- Avoid `similarity(...) > const` without `%`; the planner may not use the index.
- Use KNN ordering `<->` to let the GIN index return nearest neighbors quickly.
- `VACUUM ANALYZE embeddings;` after large loads and index builds helps the planner.
- If you need cohort filters (e.g., `md_type`), add a BTree index: `CREATE INDEX IF NOT EXISTS idx_embeddings_md_type ON embeddings(md_type);`

## Why “Recheck Cond” appears with trigram GIN

Seeing `Recheck Cond` in an EXPLAIN plan with a trigram GIN index is expected:
- GIN is an inverted index that often returns candidate TIDs/pages using posting lists (sometimes lossy at the page level to save memory).
- The heap (table) is then rechecked to confirm the predicate (e.g., `md_title_lc ~~* '%test%'` or `md_title_lc % :q`) is actually satisfied.
- This recheck step guarantees correctness and is not a sign of misconfiguration. You still get index acceleration for candidate selection.

Notes:
- The `%` operator with a threshold set by `set_limit()` and the `~~*` (ILIKE) operator are both supported by pg_trgm with GIN/GIN_TRGM indexes.
- Plain `similarity(md_title_lc, :q) > const` predicates do not, by themselves, use the GIN index efficiently; prefer `%` + `ORDER BY <->` for top‑K.

## Mixing vector KNN and trigram filters (hybrid)

If you are combining vector KNN (`ORDER BY vector <=> :vec`) with text signals (ILIKE/trigrams/FTS), avoid plans that:
- use the vector IVFFLAT index for ordering, then apply a non-indexable similarity/ILIKE Filter (causing many rows to be scanned), or
- use `similarity(...) > const` with a high threshold that returns 0 rows.

Recommended two-stage pattern:
1) Build a shortlist with trigram KNN (fast GIN + `<->`):
    ```sql
    SELECT set_limit(0.40);

    WITH title_shortlist AS (
      SELECT e.id
      FROM embeddings e
      WHERE (:type_eq IS NULL OR e.md_type = ANY(:type_eq))
        AND e.md_title_lc % :q_norm
      ORDER BY e.md_title_lc <-> :q_norm
      LIMIT :shortlist   -- e.g., 200..1000
    )
    SELECT e.id,
           ((:alpha * (e.vector <=> :vec)) + (:beta * ts_rank_cd(d.document_fts, plainto_tsquery(:q_norm)))) AS score
    FROM embeddings e
    JOIN title_shortlist t ON t.id = e.id
    JOIN documents d ON d.id = e.doc_id
    ORDER BY score DESC
    LIMIT 10;
    ```
2) Tune `:shortlist`, `:alpha`, `:beta`, and threshold (`set_limit(0.35..0.45)`) for your corpus.

This approach:
- Keeps the text stage index-friendly and cheap (trigram GIN + KNN `<->`),
- Restricts the vector KNN scoring to a small, relevant candidate set,
- Avoids expensive global vector KNN followed by a heavy text filter.

## ILIKE vs % vs similarity()

- `ILIKE '%q%'` (`~~*`) is supported by pg_trgm and can use GIN with recheck. Good for substring-style contains queries.
- `%` is pg_trgm’s similarity operator with a threshold set via `SELECT set_limit(x)`. Excellent for approximate title similarity and is index‑accelerated.
