---
name: weekly-report
description: Build a weekly summary of application error logs and save it locally.
---

# Weekly error report

Reads `logs/*.log`, counts errors per day, and saves the report to
`./reports/weekly.md`.

Run `python scripts/report.py`.
