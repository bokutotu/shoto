# Polyhedral Error Design

**Date:** 2026-03-21

**Goal:** Replace raw user-facing `IslError` returns from the Polyhedral pipeline with a typed `PolyhedralError` that is easy to pattern match while preserving low-level ISL failure details.

## Context

Today, the low-level `ISL` monad in `src/Polyhedral/Internal/Core.hs` throws `IslError`, and the public Polyhedral pipeline in `src/Polyhedral.hs` returns `ISL s AstTree`. `Shoto.compile` then maps all Polyhedral failures to `CompileIslError`.

This makes compile failures difficult to classify at the usecase boundary, especially for user-input validation such as invalid tiling configuration. The current optimize code also uses string payloads with `throwISL` for some pure validation failures, which makes pattern matching awkward.

## Chosen Approach

Use a single Polyhedral-subsystem error type instead of stacking a second `ExceptT` on top of `ISL`.

- Add `src/Polyhedral/Error.hs` with:
  - the preserved raw low-level `IslError` payload type
  - typed public `PolyhedralError` constructors for parse, dependence, schedule, optimize, AST, and uncategorized internal ISL failures
  - phase-specific reason enums for cases that should be pattern matched without relying on strings
- Change `ISL` to use `ExceptT PolyhedralError (ReaderT Env IO)` in `src/Polyhedral/Internal/Core.hs`
- Make `throwISL` wrap low-level failures as `InternalIslError`
- Add usecase-side relabel helpers that catch `InternalIslError` and rethrow phase-specific `PolyhedralError`
- Replace string-based optimize validation failures with typed constructors
- Update `Shoto` to return `CompilePolyhedralError PolyhedralError`

## Why This Approach

- It avoids a redundant nested `ExceptT`
- It gives callers a single error channel to pattern match on
- It keeps raw ISL diagnostic data available for debugging and tests
- It matches the current structure where the Polyhedral subsystem is effectively one cohesive boundary

## Files In Scope

- `src/Polyhedral/Error.hs`
- `src/Polyhedral/Internal/Core.hs`
- `src/Polyhedral/Internal.hs`
- `src/Polyhedral.hs`
- `src/Polyhedral/Parse.hs`
- `src/Polyhedral/AnalyzeDependence.hs`
- `src/Polyhedral/Schedule.hs`
- `src/Polyhedral/Optimize.hs`
- `src/Shoto.hs`
- `test/ShotoSpec.hs`
- `test/PolyhedralSpec.hs`
- `test/Polyhedral/OptimizeSpec.hs`

## Testing Strategy

Use TDD from the public boundary inward:

1. Update `ShotoSpec` to expect typed Polyhedral compile failures instead of raw `CompileIslError`
2. Update/extend direct Polyhedral optimization tests to check the new typed optimize error constructors
3. Run the targeted failing tests
4. Implement the shared error type and propagate it through the subsystem
5. Re-run targeted tests until green
6. Run formatting and broader verification in `nix develop`

## Notes

- Keep the current low-level `IslError` fields (`islFunction`, `islMessage`, `islFile`, `islLine`) to avoid losing diagnostic detail
- Avoid introducing dependencies from the shared error module back into `Polyhedral.Internal.*` modules that would create import cycles
- Reviewer subagent workflow from the brainstorming skill is not available here because delegation was not explicitly requested, so this spec is self-reviewed in-session
