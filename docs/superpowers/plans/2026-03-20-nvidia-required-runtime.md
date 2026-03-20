# NVIDIA-Required Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the `CUDA_RUNTIME` conditional path so the repository always builds and tests the real NVIDIA runtime.

**Architecture:** The runtime already has a single real CUDA/NVRTC implementation plus a no-CUDA fallback branch selected by CPP. This plan deletes the fallback branch and its error constructor, then verifies the unguarded NVIDIA tests pass in the macro-free build.

**Tech Stack:** Haskell, Cabal via `package.yaml`/`hpack`, CUDA driver API, NVRTC, Hspec, Nix dev shell

---

### Task 1: Drive The Macro-Free Runtime Spec Red

**Files:**
- Modify: `package.yaml`
- Modify: `test/Runtime/NVIDIASpec.hs`
- Test: `test/Runtime/NVIDIASpec.hs`

- [ ] **Step 1: Remove `-DCUDA_RUNTIME` from `package.yaml` and remove the CPP guard from `test/Runtime/NVIDIASpec.hs`**

- [ ] **Step 2: Run the targeted runtime spec to verify it fails**

Run: `nix develop -c cabal test shoto-test --test-options='--match "Runtime.NVIDIA"'`

Expected: FAIL because the runtime still returns `ErrRuntimeCudaUnavailable` through the fallback path.

### Task 2: Remove The No-CUDA Fallback Implementation

**Files:**
- Modify: `src/Runtime/NVIDIA.hs`
- Modify: `src/Runtime/NVIDIA/Types.hs`
- Modify: `src/Runtime/Types.hs`
- Test: `test/Runtime/NVIDIASpec.hs`

- [ ] **Step 1: Remove CPP from the NVIDIA runtime modules and keep only the CUDA-backed implementation**

- [ ] **Step 2: Delete `ErrRuntimeCudaUnavailable` from `Runtime.Types`**

- [ ] **Step 3: Run the targeted runtime spec to verify it passes**

Run: `nix develop -c cabal test shoto-test --test-options='--match "Runtime.NVIDIA"'`

Expected: PASS

### Task 3: Full Verification

**Files:**
- Verify only

- [ ] **Step 1: Format the tree if needed**

Run: `nix develop -c ./format.sh`

- [ ] **Step 2: Run the full test suite**

Run: `nix develop -c cabal test all`

Expected: PASS

- [ ] **Step 3: Run HLint**

Run: `nix develop -c hlint .`

Expected: PASS

## Review Notes

- Reviewer subagent workflow was not used because delegation was not explicitly requested in this session.
- Execute inline in this session using `executing-plans`.
