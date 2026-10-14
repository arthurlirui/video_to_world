# patches/ — Third-Party Patches

Patch files applied to third-party dependencies to fix compatibility issues or add required functionality.

---

## `da3-export-trajectory.patch`

**Applied to:** [Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3) (at commit `2c21ea8`)

**Purpose:** Adds trajectory export functionality to DA3. Without this patch, DA3 does not export the camera trajectory (`_transforms.json`) that `eval_gs.py` needs to render novel views along the DA3 flythrough path.

**Application:**
```bash
git -C third_party/depth-anything-3 apply ../../patches/da3-export-trajectory.patch
```

---

## `romav2-dataclasses.patch`

**Applied to:** [RoMaV2](https://github.com/Parskatt/RoMaV2)

**Purpose:** Fixes a dependency-resolution issue where RoMaV2's `setup.py` declares `dataclasses>=0.8`, which conflicts with Python 3.10+'s built-in `dataclasses` module. The patch changes the dependency to just `dataclasses` (no version constraint), allowing installation under Python 3.10+.

See [Parskatt/RoMaV2#26](https://github.com/Parskatt/RoMaV2/issues/26) for the upstream issue.

**Application:**
```bash
git -C third_party/RoMaV2 apply ../../patches/romav2-dataclasses.patch
```

---

## Maintenance

When updating a third-party dependency to a newer commit, re-verify that these patches still apply cleanly. If a patch no longer applies:
1. Check if the upstream project has incorporated the fix natively.
2. If not, update the patch to apply against the new version.
3. Record the change in the [changelog](../CHANGELOG.md).
4. Update the pinned commit hash in the installation instructions.
