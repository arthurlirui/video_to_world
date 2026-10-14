# Contributing Guide

This guide describes how to contribute to the Video-to-World project, including code changes, documentation updates, and release procedures.

---

## 1. Getting Started

1. Clone the repository and set up the environment following the [README](../README.md).
2. Read the [Architecture Overview](architecture.md) to understand the pipeline.
3. Browse the [Module Reference](modules/README.md) for the specific area you want to work on.

---

## 2. Making Code Changes

### Code Style

- The project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting (configured in `pyproject.toml`):
  - Target: Python 3.10
  - Line length: 120
  - Line ending: auto
- Run `ruff check .` and `ruff format .` before committing.

### Configuration Changes

All hyperparameters are defined as frozen dataclasses in `configs/`. When adding or modifying configuration:

1. **Adding a new field:** Always provide a sensible default value. This is a MINOR version change.
2. **Removing or renaming a field:** This is a MAJOR version change. Consider deprecating first with a warning.
3. **Document the field:** Add a docstring to the dataclass field explaining its purpose and valid values.

### Convention Metadata

If your change affects the checkpoint format or coordinate-frame convention:

1. Update `convention.json` handling in the producing stage.
2. Add validation in consuming stages.
3. This is a **MAJOR version change** — existing checkpoints will be incompatible.

---

## 3. Updating Documentation

Documentation is not optional — it is part of the deliverable.

### When to Update Documentation

| Change | Documentation to Update |
|--------|------------------------|
| New file added | `docs/modules/<relevant>.md` — add a new section |
| File deleted | `docs/modules/<relevant>.md` — remove the section |
| Function/class signature changed | `docs/modules/<relevant>.md` — update the entry |
| New pipeline stage | `docs/architecture.md` — add stage description; `docs/modules/README.md` — add script entry |
| New external dependency | `docs/architecture.md` — add to dependencies table; `README.md` — add install instructions |
| Module dependency graph changed | `docs/architecture.md` — update the graph |
| Bug fix | `docs/CHANGELOG.md` — add entry under `Fixed` |
| New feature | `docs/CHANGELOG.md` — add entry under `Added` |

### Documentation Style

- Write for a developer who is new to the project but experienced in 3D vision.
- Use complete sentences, not fragments.
- Include the "why" — not just "what" a function does, but why it exists.
- Reference other documentation files with relative Markdown links.
- Keep tables short and scannable; use prose for explanations.

---

## 4. Changelog Workflow

The changelog follows the [Keep a Changelog](https://keepachangelog.com/) format.

### Adding Entries

When making any change, add an entry to the `[Unreleased]` section in [`docs/CHANGELOG.md`](CHANGELOG.md):

```markdown
### Added
- New feature for doing X in `module_name.py`

### Fixed
- Fixed bug where Y would fail when Z (#123)
```

### Entry Guidelines

- **Be specific:** Name the file or module affected.
- **Be concise:** One line per entry, readable in a changelog summary.
- **Reference issues/PRs:** Append `(#NNN)` for traceability.
- **Use the right category:** `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`.

---

## 5. Release Procedure

Follow these steps to release a new version:

### Step 1: Determine the Version Number

Review the `[Unreleased]` changelog entries and determine the version bump:

- Any `Removed` or `Changed` entries that break compatibility → **MAJOR**
- Any `Added` entries (new features) → **MINOR**
- Only `Fixed` entries → **PATCH**

### Step 2: Update the Version File

Update [`VERSION`](../VERSION) at the project root:

```
X.Y.Z
```

### Step 3: Finalize the Changelog

In [`docs/CHANGELOG.md`](CHANGELOG.md):

1. Move entries from `[Unreleased]` to a new section:
   ```markdown
   ## [X.Y.Z] — YYYY-MM-DD
   
   ### Added
   - ...
   ```
2. Add a new empty `[Unreleased]` section.
3. Update the "Version History Summary" table.
4. Update the comparison links at the bottom.

### Step 4: Update Documentation

- Verify that `docs/INDEX.md` shows the correct version number.
- Verify that all new files/modules are documented in `docs/modules/`.
- Verify that `docs/architecture.md` reflects any structural changes.

### Step 5: Commit and Tag

```bash
git add VERSION docs/CHANGELOG.md docs/INDEX.md
git commit -m "release: vX.Y.Z"
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin main
git push origin vX.Y.Z
```

### Step 6: Verify

- Confirm the tag appears in the repository.
- Confirm the changelog and version file are consistent.
- Confirm the documentation renders correctly.

---

## 6. Checklist for Pull Requests

Before submitting a PR, verify:

- [ ] Code passes `ruff check .` and `ruff format .`
- [ ] Documentation is updated for all changed files
- [ ] Changelog entry added under `[Unreleased]`
- [ ] Version number in `VERSION` is **not** changed (only changed during release)
- [ ] New configuration fields have defaults and docstrings
- [ ] No breaking changes without prior discussion and MAJOR version justification
- [ ] Convention metadata updated if checkpoint format changed
- [ ] Architecture overview updated if module structure changed

---

## 7. Reporting Issues

When reporting a bug:

1. Include the exact command that failed.
2. Include the full error traceback.
3. Include the stage and configuration used.
4. Note the Python, CUDA, and key dependency versions.
5. If possible, include the scene directory structure (without large data files).

When requesting a feature:

1. Describe the use case.
2. Explain which pipeline stage it affects.
3. Suggest where the configuration would live (which `configs/` dataclass).
