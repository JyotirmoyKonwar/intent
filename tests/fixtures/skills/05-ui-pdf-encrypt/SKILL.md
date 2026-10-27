---
name: pdf-tools
description: Merge, split and password-encrypt PDF files.
---

# PDF tools

- `merge a.pdf b.pdf -o out.pdf` combines several PDFs into one.
- `split in.pdf -o pages/` writes one PDF per page.
- `encrypt in.pdf --password secret -o locked.pdf` protects a PDF with a password.

Run `python scripts/pdftool.py <command> ...`.
