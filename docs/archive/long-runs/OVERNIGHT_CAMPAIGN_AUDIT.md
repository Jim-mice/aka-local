# Overnight Campaign Audit (checkpoint)

| target | canonical state | performance state | blocker |
|---|---|---|---|
| SwiGLU | frozen real-target work | historical, unchanged | none reopened |
| Vocab-Parallel CE | forward incumbent Episode 7; backward blocked | forward 2.4771907400291515x; backward no incumbent | frozen |
| RMSNorm | non-TE WrappedTorchNorm | R1/R2 stability blocked; no incumbent | frozen |
| Attention | native DotProductAttention local dense/no-mask/p=0 forward | A2 aggregate 1.338883x, mixed config behavior; no incumbent yet | A2 profile provenance |
| MoE | native SequentialMLP fallback boundary identified | no score | real replay/oracle/NSYS pending; TE blocked |

Promotion gates remain: contract/ABI → toolchain → semantic correctness → scope equivalence → reference stability → candidate stability → profile → incumbent → replay → integration. No score was imported from standalone operators.
