# MOPS financial statements

Official MOPS consolidated financial-statement HTML used to rebuild the assignment history.

Source pattern:

`https://mopsov.twse.com.tw/server-java/t164sb01?step=1&CO_ID=6274&SYEAR=<year>&SSEASON=<season>&REPORT_ID=C`

Files:

- `6274_2016_Q4_C.html` … `6274_2025_Q4_C.html`
- `6274_2026_Q1_C.html` for the cash dividend paid from 2025 earnings

Regenerate processed assignment data:

```bash
python src/build_assignment_financials.py
```

Download missing source files:

```bash
python src/build_assignment_financials.py --fetch-missing
```
