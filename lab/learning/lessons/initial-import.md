# Initial imported lessons

## What happened

The lab imports standalone RTX5060 CUDA experiments and remote BI-V150 Atrex canonical summaries without copying large artifacts.

## GPU concepts

- RMSNorm V2 is evidence for a scoped warp-reduction observation, not a universal rule.
- SwiGLU V3 shows why a weak single-run gain needs repeated ABBA before promotion.
- Residual RMSNorm records show launch count, atomics, row batching, and reduction structure can be coupled; attribution must follow the experiment record.
- A BI-V150 CoreX/Triton masked-2D miscompile is a backend quirk, not a general Triton rule.

## What is not proven

No cross-architecture rule is established. No formal Megatron integration is imported as complete.
