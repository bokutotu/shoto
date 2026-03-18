# Shoto Commit Style Reference

Prefer the recent English header style used in the newest history for the first line, then always add `Why` and `What` sections. Use older bare subjects or Japanese phrasing only if the user explicitly asks to mirror that older style.

## Scope Map

- `shoto/src/Runtime*`, `shoto/test/Runtime*` -> `runtime`
- `shoto/src/FrontendIR*`, `shoto/test/FrontendIR*` -> `frontendir`
- `shoto/src/Polyhedral*`, `shoto/test/Polyhedral*` -> `polyhedral`
- `shoto/src/Codegen*`, `shoto/test/CodegenSpec.hs` -> `codegen`
- `isl/src/ISL*`, `isl/test/ISL*`, `isl/package.yaml`, `isl/CLAUDE.md` -> `isl`
- `.agents/skills/*`, `AGENTS.md`, `CLAUDE.md` -> `agents` with `docs` unless the change adds tooling behavior rather than guidance
- `flake.nix`, `cabal.project`, `lefthook.yml`, `format.sh`, `hlint-refactor.sh` -> usually `chore` with no scope unless one subsystem clearly dominates
- `shoto/src/Shoto.hs` or cross-pipeline compiler wiring -> use the dominant subsystem scope if clear; otherwise omit the scope
- dependency additions -> choose the module that needs the dependency, not the package manager file

## Type Hints

- Use `feat` for new lowering, scheduling, codegen, or runtime capabilities.
- Use `fix` for incorrect results, crashes, invalid generated code, or broken ABI handling.
- Use `refactor` for module splits, API reshaping, or ownership changes that should preserve behavior.
- Use `docs` for instruction files, design notes, or agent guidance.
- Use `test` when only specs or fixtures change.
- For dependency additions, explain in `Why` why the chosen dependency fits the affected module or workflow.

## Recent Examples

- `refactor(runtime): require explicit KernelSignature for CPU JIT`
- `feat(codegen): add initial C and CUDA backends`
- `refactor(frontendir,polyhedral): replace CheckedProgram flow with parse-based lowering`
- `feat(frontendir): add reduction statement lowering and validation`
- `docs(agents): translate AGENTS.md to English`
- `isl: add ScheduleNode bindings`

## Preferred Response Shape

- Start with one recommended commit message.
- Return only one recommended commit message, even when the diff is ambiguous.
- Prefer subjects that name the semantic effect instead of listing edited files.
- Return only the raw commit message text, with no intro, explanation, labels, or markdown fences.
