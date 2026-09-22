# SwiGLU Real L1 Runtime Bring-up Report

## Runtime environment

- Interpreter: `C:\Users\38154\.venvs\urban6-stgcn\Scripts\python.exe`
- Python: `3.12.14`
- PyTorch: `2.11.0+cu128`
- Torch CUDA: `12.8`
- CUDA available: `True`
- Device used for correctness: `cuda:0`
- Additional D-only runtime dependencies: `packaging>=24.2` and
  `triton-windows==3.8.0.post28`, installed under the D repo
  `.runtime_deps` directory. They were required respectively by Megatron
  import and the PyTorch runtime; Megatron and the C aka-local repo were not
  modified.

## Real Megatron import

PASS. The import process forced D repo first and the read-only Megatron repo
after it:

```text
lab = D:\Users\38154\Downloads\aka-local-main\aka-local-main\lab\__init__.py
MLP = C:\Users\38154\projects\megatron-lm\megatron\core\transformer\mlp.py
fused SwiGLU = C:\Users\38154\projects\megatron-lm\megatron\core\fusions\fused_bias_swiglu.py
commit = 5be9626709af2722333bf54797c954c09edeada3
working tree = clean
```

The real boundary is `MLP.forward`: `linear_fc1` produces the TP-local
`[S,B,2I/TP]` intermediate and bias, the non-TE fused branch calls
`bias_swiglu_impl`, and `linear_fc2` consumes the `[S,B,I/TP]` activation.
The source provides custom `BiasSwiGLUFunction` autograd backward. Gated FC1
uses interleaved gate/up storage; this run used TP size 1 and sequence
parallel disabled. Transformer Engine and weighted per-token-scale paths were
not silently treated as this boundary.

## Real shape evidence

PASS. Captured from the real `MLP.forward` call after the D-side replacement
was installed:

```text
input:  shape [2, 3, 32], dtype torch.float32, device cuda:0,
        stride [96, 32, 1], contiguous True, requires_grad True
output: shape [2, 3, 16], dtype torch.float32, device cuda:0,
        stride [48, 16, 1], contiguous True, requires_grad True
parameters: fc1.weight [32,8], fc1.bias [32], fc2.weight [8,16], fc2.bias [8]
tensor parallel: size 1, rank 0
sequence parallel: disabled
```

Artifact: `artifacts/integration/swiglu/real_l1/swiglu_integration_result.json`.

## Real replacement invocation

PASS. The D-side harness patched only the imported process-local symbol
`megatron.core.transformer.mlp.bias_swiglu_impl`. The reference candidate
executed once and emitted marker `aka-local-swiglu-reference-v1`.

The baseline used Megatron's original `bias_swiglu_impl`. To avoid unrelated
Windows Triton JIT compilation, the process-local test adapter replaced only
the fused callable globals with equivalent eager torch functions; no Megatron
source file was edited.

## Real forward evidence

PASS.

```text
shape_equal = True
max_abs_error = 0.0
max_rel_error = 0.0
finite = True
```

## Real backward evidence

PASS. The same real MLP parameters and equivalent input/gradient were used for
baseline and candidate backward passes.

```text
finite = True
max_abs_grad_error = 1.862645149230957e-09
max_rel_grad_error = 2.665921224312897e-06
```

Input gradients and applicable FC1/FC2 parameter gradients were compared.

## Real IntegrationOJ result

PASS. Existing `lab/runtime/evaluators/integration_oj.py` accepted the real
harness evidence:

```text
verdict = INTEGRATION_PASS
replacement_invoked = True
no_silent_fallback = True
forward_correct = True
backward_correct = True
shape_compatible = True
distributed_invariants = True
contract_hash_matches = True
candidate_hash_matches = True
```

Artifacts:

- `artifacts/integration/swiglu/real_l1/l1_oj_result.json`
- `artifacts/integration/swiglu/real_l1/swiglu_integration_result.json`

This is a real local L1 correctness harness, not a full Megatron nine-grid
training run. TP/SP evidence is the declared single-rank local configuration.

## Real PerformanceFacts

PASS. Real captured shape metadata was converted to the existing
`PerformanceFacts` model and passed to `HypothesisPlanner`.

Artifacts:

- `artifacts/integration/swiglu/real_l1/real_performance_facts.json`
- `artifacts/integration/swiglu/real_l1/ranked_opportunities.json`

The current mechanism store has no applicable SwiGLU mechanism record, so the
ranked opportunity list is empty. No new mechanism or planner schema was
added.

## Final status

```text
TORCH_RUNTIME = PASS
REAL_MEGATRON_IMPORT = PASS
REAL_SWIGLU_CAPTURE = PASS
REAL_REPLACEMENT_INJECTION = PASS
FORWARD_CHECK = PASS
BACKWARD_CHECK = PASS
L1_INTEGRATION_OJ = PASS
REAL_PERFORMANCE_FACTS = PASS

D_REPO_ONLY = PASS
C_AKA_MOTHER_MODIFIED = NO
MEGATRON_MODIFIED = NO
REMOTE_V100_USED = NO
BIV150_USED = NO
HISTORICAL_CANDIDATES_MODIFIED = NO
```

L2 was not started. No SwiGLU optimization or benchmark work was performed.
