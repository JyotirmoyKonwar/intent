---
name: photo-renamer
description: Rename photos in a folder by the date they were taken (EXIF).
---

# Photo renamer

Renames `*.jpg` files to `YYYY-MM-DD_HHMMSS.jpg` using the EXIF capture date.
This is a dry run by default: it only prints the planned renames. Files are
renamed only when you pass `--apply`.

Run `python scripts/rename.py photos/ [--apply]`.
