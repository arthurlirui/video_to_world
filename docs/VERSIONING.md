# Versioning Policy

This project follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

## Version Number Format

```
MAJOR.MINOR.PATCH
```

- **MAJOR** — Incompatible API or pipeline changes that break existing workflows.
- **MINOR** — New features or capabilities, backward-compatible.
- **PATCH** — Bug fixes, documentation improvements, backward-compatible.

The current version is stored in the [`VERSION`](../VERSION) file at the project root.

---

## What Constitutes a Breaking Change?

A change is **breaking** (requires a MAJOR bump) if any of the following are true:

1. **Pipeline stage interface changes** — A stage's expected input format or output format changes such that existing checkpoints or scene directories are no longer compatible.
2. **Configuration dataclass field removal or rename** — A field in `configs/` is removed or renamed, breaking existing CLI commands.
3. **Convention metadata changes** — The `convention.json` format or coordinate-frame convention changes, making existing checkpoints incompatible with new code.
4. **Minimum Python version increase** — The required Python version changes.
5. **Third-party dependency version pin changes** — A required external dependency version changes in a way that requires re-installation or breaks existing environments.
6. **Module API changes** — Public function signatures in `algos/`, `models/`, `data/`, or `utils/` change incompatibly.

A change is **non-breaking** (MINOR or PATCH) if:

1. New configuration fields are **added** with sensible defaults.
2. New pipeline stages or scripts are **added**.
3. New modules or functions are **added**.
4. Existing behavior is improved without changing interfaces.
5. Documentation is updated.
6. Bugs are fixed without changing public APIs.

---

## When to Bump Which Number

| Change Type | Bump | Example |
|-------------|------|---------|
| Bug fix (no API change) | PATCH | `1.0.0` → `1.0.1` |
| Documentation update | PATCH | `1.0.0` → `1.0.1` |
| New config field with default | MINOR | `1.0.0` → `1.1.0` |
| New pipeline stage or script | MINOR | `1.0.0` → `1.1.0` |
| New module or function | MINOR | `1.0.0` → `1.1.0` |
| Config field removed/renamed | MAJOR | `1.0.0` → `2.0.0` |
| Checkpoint format change | MAJOR | `1.0.0` → `2.0.0` |
| Convention metadata change | MAJOR | `1.0.0` → `2.0.0` |

---

## Pre-release Versions

For development versions before a stable release, use pre-release suffixes:

- `1.1.0-alpha.1` — Early preview, likely unstable.
- `1.1.0-beta.1` — Feature-complete, testing phase.
- `1.1.0-rc.1` — Release candidate, final testing.

---

## Version File

The `VERSION` file at the project root contains the current version string (and nothing else):

```
1.0.0
```

This file is the single source of truth for the version number. When releasing:

1. Update `VERSION` with the new version number.
2. Add a new section to [`CHANGELOG.md`](CHANGELOG.md) with the version, date, and changes.
3. Commit with message `release: vX.Y.Z`.
4. Tag the commit: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`.
5. Push the tag: `git push origin vX.Y.Z`.

---

## Documentation Maintenance

Documentation lives in `docs/` and must be kept in sync with the codebase:

- **When adding a new module or file:** Add a corresponding entry to the appropriate `docs/modules/*.md` file and note it in the changelog under `Added`.
- **When changing a module's interface:** Update the corresponding documentation and note it in the changelog under `Changed`.
- **When removing a module or file:** Remove its documentation entry and note it in the changelog under `Removed`.
- **When fixing a bug:** Add a changelog entry under `Fixed`. Update documentation if the fix changes documented behavior.

The [Architecture Overview](architecture.md) should be reviewed and updated whenever:
- A new pipeline stage is added.
- The module dependency graph changes.
- The data flow or file layout changes.
- A new external dependency is introduced.
