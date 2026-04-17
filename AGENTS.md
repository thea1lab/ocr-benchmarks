# Agent instructions

Instructions for AI coding agents (Claude Code, Codex, etc.) working in
this repo. Keep these honored across all subprojects.

## Commits

- **Never add `Co-Authored-By` trailers** to commit messages — no Claude,
  Codex, or any other AI attribution. Plain commits only.
- Prefer new commits over amending, unless the user explicitly asks to
  amend.
- Don't push without the user asking.

## Repo layout

This is a monorepo of independent OCR-model test subprojects:

- Each subproject (e.g. `qianfan-ocr-test/`) owns its own `requirements.txt`
  and `.venv/`. Don't share venvs — deps collide.
- `images/` at the repo root holds shared test inputs. Reference them as
  `../images/...` from inside a subproject.
- The top-level `README.md` is the index / comparison table. Subproject
  READMEs hold model-specific setup, tuning, benchmarks, and caveats.

## Adding a new OCR subproject

1. Create `new-model-test/` with its own venv, `requirements.txt`, and
   `README.md`.
2. Follow the structure of `qianfan-ocr-test/` as a template.
3. Add a row to the top-level README's subprojects table.

## Style

- Default to writing no comments. Only add one when the *why* is non-obvious.
- Don't add features, refactors, or abstractions beyond the task.
- Pin dependency versions in `requirements.txt` (never loose).
