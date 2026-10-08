# MOPS Monthly-Revenue Parser Benchmark — 2026-10-08

## Dataset

- 274 committed MOPS OTC monthly HTML files
- 200 files contain TUC (6274)
- 74 earlier files do not contain ticker 6274

## Before

The original parser called `pandas.read_html()` on every table in every file.

On the same local repository and machine, a full 274-file parse did not finish within **180 seconds**.

## After

The primary parser now:

1. scans the raw HTML text for the exact ticker cell;
2. slices only the surrounding `<tr>`;
3. extracts the ticker, company name and current-month revenue directly;
4. uses `pandas.read_html()` only as a compatibility fallback when ticker 6274 appears but the direct row schema is not recognized.

Measured results:

- parse all 274 files: **1.56 seconds**
- rebuild `monthly_revenue.csv`: **1.53 seconds**
- average direct parsing cost: about **5.7 ms/file**

This is at least **100× faster** than the previous implementation based on the 180-second timeout.

## Failure behavior

- `MopsTickerNotFound`: valid source file but ticker 6274 is absent.
- `MopsSchemaError`: ticker/schema is present but monthly revenue cannot be safely extracted.
- Schema errors include the source path and parser diagnostics instead of silently returning `None`.

Tests cover both an older 2013 layout and the 2026 layout, plus alternate-layout fallback and malformed-schema behavior.
