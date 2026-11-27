# Vendored third-party code

Each subdirectory is copied from an upstream project and is excluded from ruff and ty.
Every subdirectory has a `PROVENANCE.md` with: upstream URL, pinned commit, licence, and a list of local modifications.
First-party code that wraps vendored code lives outside this directory and is fully linted, type-checked and documented.
