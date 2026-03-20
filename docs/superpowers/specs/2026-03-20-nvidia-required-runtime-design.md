# NVIDIA-Required Runtime Design

**Date:** 2026-03-20

**Goal:** Make the NVIDIA runtime path unconditional in this repository by removing the `CUDA_RUNTIME` CPP gate and the `ErrRuntimeCudaUnavailable` fallback.

## Context

The project already assumes CUDA is present in the development environment and links against `cuda` and `nvrtc`. The remaining conditional compilation only preserves an alternate no-CUDA build path in:

- `package.yaml`
- `src/Runtime/NVIDIA.hs`
- `src/Runtime/NVIDIA/Types.hs`
- `test/Runtime/NVIDIASpec.hs`

That fallback no longer matches the project requirement that NVIDIA GPUs are mandatory.

## Chosen Approach

Remove the CPP-based split entirely and keep only the real NVIDIA implementation:

- Delete `cpp-options: -DCUDA_RUNTIME` from the library and test stanzas in `package.yaml`
- Remove `{-# LANGUAGE CPP #-}` and all `#ifdef CUDA_RUNTIME` / `#else` / `#endif` branches from the NVIDIA runtime modules and tests
- Delete `ErrRuntimeCudaUnavailable` from `Runtime.Types`
- Keep the current CUDA-backed runtime API and tests as the only supported path

## Why This Approach

- It makes the code match the actual platform requirement
- It deletes dead configuration and dead fallback code instead of preserving it behind a permanently-on macro
- It reduces maintenance burden in the runtime facade and tests

## Files In Scope

- `package.yaml`
- `src/Runtime/NVIDIA.hs`
- `src/Runtime/NVIDIA/Types.hs`
- `src/Runtime/Types.hs`
- `test/Runtime/NVIDIASpec.hs`

## Testing Strategy

Use TDD around the behavioral transition:

1. Remove the test-side CPP guard and package-level `CUDA_RUNTIME` macro first
2. Run the NVIDIA runtime spec and observe it fail because the code still routes through `ErrRuntimeCudaUnavailable`
3. Remove the fallback implementation and error constructor
4. Re-run the targeted runtime spec until it passes
5. Run the full project verification in `nix develop`

## Notes

- The worktree already contains unrelated runtime edits. This change must preserve those in-progress changes.
- Reviewer subagent workflow from the brainstorming skill is not available here because delegation was not explicitly requested, so this spec is self-reviewed in-session.
