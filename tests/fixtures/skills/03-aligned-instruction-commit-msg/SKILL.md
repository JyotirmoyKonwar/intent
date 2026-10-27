---
name: commit-message
description: Write a Conventional Commits message for the currently staged changes.
---

# Commit message writer

1. Run `git diff --staged` to see the staged changes.
2. Write a one-line summary in Conventional Commits style (`feat:`, `fix:`, ...),
   followed by a short body that explains why the change was made.
3. Show the message to the user.

Do not run `git commit` yourself unless the user explicitly asks you to.
