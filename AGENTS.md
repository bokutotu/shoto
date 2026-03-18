## Environment

- Work inside `nix develop`.
- `lefthook` runs `./format.sh` and `cabal test all` on pre-commit for touched Haskell files.

## Commands

- Test: `cabal test all`
- Lint: `hlint .`
- Apply HLint refactors: `./hlint-refactor.sh`
- Format Haskell: `./format.sh`

## Haskell Style

- Prefer record-dot syntax (`x.field`) in new or edited code.

## Documentation

- Keep `AGENTS.md` and `CLAUDE.md` semantically synchronized.
- keep root ai docs focused on workflow and conventions, not code-discoverable details.
