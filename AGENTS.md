# Repository Guidelines

This guide explains how to develop, test, and contribute to the Shoto polyhedral-compiler library. Keep changes incremental, reproducible, and GPU-friendly.

## Project Structure & Module Organization
- `src/`: Library code (e.g., `Shoto`, `Runtime`, `FrontendIR`, `Tensor`, `Internal/FFI`).
- `test/`: Hspec tests (e.g., `ArithmeticSpec.hs`, `Spec.hs` for discovery).
- `playground/`: Scratch or experiment files (not part of the build).
- `shoto.cabal`: Package definition; library + `shoto-test` suite.
- `flake.nix`: Nix dev shell (GHC, tools, GCC 13, CUDA env setup).

## Build, Test, and Development Commands
- Enter dev shell: `nix develop` (sets GCC 13 and CUDA paths if available).
- Build library: `cabal build`
- Run tests: `cabal test`
- REPL for library: `cabal repl shoto`
- Format code: `fourmolu -i src/**/*.hs test/**/*.hs` and `stylish-haskell -i src/**/*.hs test/**/*.hs`
- Example focused test: `cabal test shoto-test --test-options='--match Addition'`

## Coding Style & Naming Conventions
- Indentation: 4 spaces; wrap at ~100 cols (Fourmolu), 80 for import formatting (Stylish Haskell).
- Use Fourmolu and Stylish Haskell; do not hand-format imports.
- Naming: Modules/Types `CamelCase` (e.g., `FrontendIR`), values/functions `camelCase`, constants use `camelCase`.
- Prefer explicit exports; group and align imports consistently.

## Testing Guidelines
- Framework: Hspec with discovery via `Spec.hs` and `*Spec.hs` files in `test/`.
- Keep tests deterministic; avoid relying on host GPU timing.
- Run all tests with `cabal test`. Focus with `--match` as shown above.
- Coverage: No enforced threshold yet; add meaningful examples and negative cases.

## Commit & Pull Request Guidelines
- Commits in history are short and imperative (often Japanese). Follow that style; explain the “why” if non-obvious.
- Prefer small, reviewable commits. Reference issues when applicable.
- PRs should include: clear summary, scope, screenshots/logs if relevant, test results, and any CUDA environment notes.

## Development Process Principles
- Implement in "small diffs" like 9cc. Keep each commit/PR minimal, always buildable, and with tests passing.
- Practice strict TDD in the Red → Green → Refactor cycle. Start with a failing test, make it pass with the smallest change, then refactor.

## Security & Configuration Tips
- CUDA is required at runtime: library links `cudart`, `cuda`, `nvrtc`, `stdc++` and expects `/usr/local/cuda` or `CUDA_PATH`. Ensure `LD_LIBRARY_PATH` includes the CUDA `lib64` path.
- In Nix shells, GCC 13 is selected; avoid mixing toolchains outside `nix develop`.
- Do not commit secrets or machine-specific paths; prefer shell/env setup in `flake.nix`.

## Agent-Specific Instructions
- Codex role: Acts as a pair-programming navigator and code reviewer (same policy as described in `CLAUDE.md`). It does not implement code directly; it provides reviews, feedback, design guidance, and next-step suggestions. Maintainers apply changes and run commands.
