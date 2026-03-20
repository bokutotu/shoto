# Polyhedral Error Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a typed `PolyhedralError` across the Polyhedral subsystem so compile failures are easy to pattern match and no longer surface as raw `IslError`.

**Architecture:** The existing `ISL` monad already owns the Polyhedral subsystem context and failure path. This plan replaces its error type with `PolyhedralError`, keeps raw ISL diagnostics as payload, and reclassifies failures at usecase boundaries in `Parse`, `AnalyzeDependence`, `Schedule`, `Optimize`, and `Polyhedral.synthesize`.

**Tech Stack:** Haskell, `ExceptT`, Cabal, Hspec, Nix dev shell, ISL FFI bindings

---

### Task 1: Drive The User-Facing Compile Error Red

**Files:**
- Modify: `test/ShotoSpec.hs`
- Test: `test/ShotoSpec.hs`

- [ ] **Step 1: Update `ShotoSpec` to expect `CompilePolyhedralError` and typed optimize failure constructors instead of `CompileIslError`**

- [ ] **Step 2: Run the targeted `Shoto compile` spec to verify it fails**

Run: `nix develop -c cabal test shoto-test --test-options='--match "Shoto compile"'`

Expected: FAIL because the production code still exports and returns `CompileIslError`/`IslError`

### Task 2: Drive The Direct Optimize Error Red

**Files:**
- Modify: `test/Polyhedral/OptimizeSpec.hs`
- Test: `test/Polyhedral/OptimizeSpec.hs`

- [ ] **Step 1: Add assertions for typed invalid tiling errors from `applyScheduleOptimizations`**

- [ ] **Step 2: Run the targeted optimize spec to verify it fails**

Run: `nix develop -c cabal test shoto-test --test-options='--match "Polyhedral schedule optimizations"'`

Expected: FAIL because optimize validation still throws raw string-based ISL failures

### Task 3: Introduce The Shared Polyhedral Error Type

**Files:**
- Create: `src/Polyhedral/Error.hs`
- Modify: `src/Polyhedral/Internal/Core.hs`
- Modify: `src/Polyhedral/Internal.hs`
- Test: `test/ShotoSpec.hs`
- Test: `test/Polyhedral/OptimizeSpec.hs`

- [ ] **Step 1: Add `Polyhedral.Error` with raw low-level `IslError`, typed public `PolyhedralError`, and the initial parse/optimize reason enums**

- [ ] **Step 2: Change `ISL` to throw `PolyhedralError` and make `throwISL` wrap low-level failures as `InternalIslError`**

- [ ] **Step 3: Update re-exports/imports so callers use `PolyhedralError` from the new shared module**

- [ ] **Step 4: Run the targeted `Shoto compile` and optimize specs**

Run: `nix develop -c cabal test shoto-test --test-options='--match "Shoto compile|Polyhedral schedule optimizations"'`

Expected: Some tests still fail because usecase modules and `Shoto` have not been fully relabeled yet

### Task 4: Reclassify Usecase Errors And Update Public APIs

**Files:**
- Modify: `src/Polyhedral/Parse.hs`
- Modify: `src/Polyhedral/AnalyzeDependence.hs`
- Modify: `src/Polyhedral/Schedule.hs`
- Modify: `src/Polyhedral/Optimize.hs`
- Modify: `src/Polyhedral.hs`
- Modify: `src/Shoto.hs`
- Modify: `test/ShotoSpec.hs`
- Modify: `test/PolyhedralSpec.hs`
- Modify: `test/Polyhedral/OptimizeSpec.hs`

- [ ] **Step 1: Add usecase-level error relabel helpers and convert optimize validation failures from strings to typed constructors**

- [ ] **Step 2: Update `Polyhedral.synthesize` and `Shoto.compile` to return typed Polyhedral errors**

- [ ] **Step 3: Update direct Polyhedral tests for the new error type where needed**

- [ ] **Step 4: Run the targeted Polyhedral and Shoto specs**

Run: `nix develop -c cabal test shoto-test --test-options='--match "Polyhedral synthesize|Polyhedral schedule optimizations|Shoto compile"'`

Expected: PASS

### Task 5: Full Verification

**Files:**
- Verify only

- [ ] **Step 1: Format the tree**

Run: `nix develop -c ./format.sh`

- [ ] **Step 2: Run the full test suite**

Run: `nix develop -c cabal test all`

Expected: PASS

- [ ] **Step 3: Run HLint**

Run: `nix develop -c hlint .`

Expected: PASS

## Review Notes

- Reviewer subagent workflow was not used because delegation was not explicitly requested in this session
- Execute inline in this session using `executing-plans`
