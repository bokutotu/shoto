## What Is Shoto

Shoto is a polyhedral compiler. Its goal is to describe tensor computations and generate GPU kernel/SIMD code.
At the moment, it simply outputs C code.

## Package Structure

### shoto

The main compiler package.

### isl

Haskell bindings for Integer Set Library (ISL). It provides the mathematical foundation for polyhedral analysis. By separating it as an independent package from shoto, the ISL bindings can be reused on their own.

## Language and Style

- **GHC2024**: Use the latest Haskell language extension set
- **Leading comma**: Leading-comma style (easier to review diffs)
- **OverloadedRecordDot**: Prefer dot syntax for record field access (`x.field`) in new/edited Haskell code
- **4-space indentation**

## Prerequisites

Run inside the `nix develop` environment. `lefthook` automatically runs formatting and tests on commit.

## Codex Sandbox Notes

- In Codex sandbox runs, default Cabal paths under `~/.cache/cabal` and `~/.local/state/cabal` may be read-only.
- For build/test commands in Codex, use workspace-local cache/store paths:

```bash
XDG_CACHE_HOME=$PWD/.cache CABAL_DIR=$PWD/.cabal cabal --store-dir=$PWD/.cabal/store test all
```

- Replace `test all` with `build all` or other subcommands as needed.

## Test

```bash
cabal test all
```

## Lint

```bash
hlint .
```

Try automatic fixes:
```bash
./hlint-refactor.sh
```

## Formatting

```bash
./format.sh
```

## FrontendIR Tensor Rules

- `Program` must declare tensors explicitly via `tensors :: NonEmpty TensorDecl`.
- Tensor shape is symbolic (`[ParamName]`), and rank is `length shape`.
- `checkProgram` is the validation boundary (duplicates, undeclared tensors, rank mismatch, unknown shape params).
- `lowerToRaw` only lowers a checked program and does not perform validation.

# ISL Haskell Bindings

Haskell bindings for Integer Set Library (ISL).
A foundational library for polyhedral analysis.

## Module Structure

- `ISL.Core` - ISL computation monad (`runISL`) and error types
- `ISL.*` - Public APIs for each domain (Set, Map, Schedule, etc.)
- `ISL.Internal.FFI` - C FFI bindings
- `ISL.Internal.<Module>/Types.hs` - Type definitions
- `ISL.Internal.<Module>/Ops.hs` - Operation implementations

## Development Rules

### When Adding New Features

When adding new functions or modules, always add the corresponding tests at the same time.

1. Add `foreign import` to `ISL.Internal.FFI`
2. Implement wrappers in `Internal/<Module>/Ops.hs`
3. Re-export the public API from `ISL/<Module>.hs`
4. **Add tests in `test/ISL/<Module>Spec.hs`**

### How to Write Tests

Use assertions that verify exact matches:

- `shouldBe` - Exact value match
- `shouldMatchList` - List element match (order-insensitive)

As a rule, do not use partial-match assertions like `shouldContain`.

## Running Tests

```bash
cabal test isl-test
```

## ISL Monad (`ISL s`)

The ISL monad internally manages the ISL library's `isl_ctx`. Users do not need to manipulate `isl_ctx` directly.

- Run ISL computations with `runISL :: ISL s a -> IO (Either IslError a)`
- Allocation and deallocation of `isl_ctx` are handled automatically
- If an FFI function requires `isl_ctx`, you can obtain it via `ISL.Core.askEnv`

### Note: Functions Named `context`

ISL has multiple functions named "context":

- `isl_ctx_*`: Context for the entire ISL library (managed by the ISL monad)
- `isl_schedule_constraints_set_context`: Sets a parameter-constraint Set (type `isl_set`)
- `isl_ast_build_from_context`: Parameter constraints for AST generation (type `isl_set`)

The arguments to `isl_schedule_constraints_set_context` and `isl_ast_build_from_context` are `isl_set`, not `isl_ctx`.

## Documentation Updates

Each time planning (plan mode) is completed, update the following files as needed:

- `/CLAUDE.md` - Instructions for Claude Code
- `/AGENTS.md` - Shared instructions for multiple AI tools
- `/isl/CLAUDE.md` - ISL package-specific conventions

Keep shared sections in `AGENTS.md` and `CLAUDE.md` synchronized.
