---
name: write-shoto-commit-message
description: Draft repository-specific commit messages for the Shoto polyhedral compiler and bundled ISL bindings. Use when Codex needs to turn staged or working-tree changes in this repo into a `git commit` subject/body, infer scopes from touched paths such as `runtime`, `frontendir`, `polyhedral`, `codegen`, `isl`, or `agents`, and match the repository's recent commit style.
---

# Write Shoto Commit Message

## Overview

Draft commit messages for this repo in a fixed template: one summary line plus `Why` and `What` sections.

## Use This Format

```text
<type>(<module>) <summary>

Why

- ...
- ...

What

- ...
- ...
```

Use `feat`, `fix`, `refactor`, `docs`, `test`, or `chore`. Use `chore`, not `chrore`.

## Draft The Message

1. Inspect staged changes first with `git diff --cached --stat`, `git diff --cached --name-only`, and `git diff --cached`.
2. If nothing is staged, inspect `git diff --stat`, `git diff --name-only`, and `git diff`, then say the draft is based on unstaged changes.
3. Pick one primary `module` from [commit-style.md](references/commit-style.md).
4. Write a one-line summary for the main effect. Keep it factual, imperative, and usually under 72 characters. Do not end it with a period.
5. Always include both sections:
   - `Why`: motivation, problem, or constraint
   - `What`: concrete code, API, or config changes
6. When adding a dependency, state in `Why` why this dependency or module was chosen for this code path.

## Keep It Tight

- Prefer the repo's recent English header style for the first line.
- Use code terms as written in the repo: `FrontendIR`, `Polyhedral`, `Runtime`, `Codegen`, `ISL`, `CPU JIT`, `KernelSignature`.
- Do not claim tests, formatting, or review work unless you actually verified them.
- If the diff is ambiguous, give one preferred draft first, then alternatives in the same template.

## Example

```text
refactor(runtime) require explicit KernelSignature for CPU JIT

Why

- stop relying on C parsing to recover runtime argument metadata
- keep the dispatch ABI explicit and stable

What

- require callers to pass `KernelSignature`
- update CPU JIT dispatch generation to consume the explicit signature
```
