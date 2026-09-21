# Phase 20-B.1 — Attention Clean-Window Agent Campaign

## Invocation outcome

`ENVIRONMENT_NOT_READY`. This invocation ended before any scoring work. No
Phase 20 active reference baseline, V1 candidate, score, incumbent, paired
benchmark, or profile was created.

## Reconstructed starting state

The authoritative Phase 20-B.0 result remains
`ENVIRONMENT_NOT_READY_FOR_SCORING`. The separate scoring era is
`attention_vendor_gemm_hybrid_v1`; it preserves framework-owned vendor QK
(`torch.baddbmm`, `alpha=0.125`) and vendor PV (`torch.bmm`). The only future
candidate boundary is the local already-scaled FP16-score to FP16-probability
stage with FP32 row-softmax semantics. It must not receive Q/K/V, scale,
mask, dropout/RNG state, or a cuBLAS handle.

The required future local ABI remains
`attention_dense_softmax_fp16_stream(const __half* scores, __half*
probabilities, int64_t batch_heads, int64_t sequence_length, cudaStream_t
stream)`, as frozen in `vendor_gemm_local_softmax_abi.json`. No V1 episode
directory was present when this invocation began or ended.

## Worker and environment gate

The local campaign namespace contained only historical `episode_A1` through
`episode_A3`; no V1 existed. No local process was identifiable as a
Phase 20-B.1 worker. Remote worker-provenance inspection and the required
read-only environment capture were attempted twice through the configured
`v100-srv` alias. Both bounded attempts failed before connection because the
hostname could not be resolved.

Consequently no fresh physical-GPU identity, telemetry, CUDA-process snapshot,
or reference-only readiness probe was obtained. This is `UNKNOWN`, not
`READY`. The policy fails closed: no full baseline or Agent episode may begin.
No process was killed, no GPU setting was changed, and this is not a repeated
probe campaign.

## Scope and policy preserved

The already-approved scope is unchanged: preallocated Q/K/V; vendor QK;
reference or candidate local stage; vendor PV; output completion. It excludes
QKV projection, rotary, output projection, allocation outside the established
scope, backward, and unrelated synchronization. The frozen policy remains
`native_dot_product_attention_stability_v1`, hash
`692f9718d1e12d922561e6d8c69cef835a37cd9c0f6e087c2ab0b342e231ae41`, with
CV <= 0.20, retained raw samples, deterministic ABBA, and the existing
bootstrap/promotion rule. Phase 17 timings remain historical evidence only;
they were not used as a Phase 20 baseline.

## V1 authorization

Not authorized. The missing gates are a fresh `READY` environment result and
a complete new stability-qualified Phase 20 reference baseline. The frozen
local ABI and correctness-harness plan remain available, but no candidate may
be generated from them until both gates pass.

## Integrity

Frozen CE, RMSNorm, Phase 17 attention, and MoE campaign states were not
modified. Megatron was verified at
`5be9626709af2722333bf54797c954c09edeada3`; `git status --porcelain` produced
no output, so its working tree is clean.

## Next legal action

Wait for a later independently invoked Phase 20-B.1 readiness assessment with
a reachable remote host. It may make exactly one fresh read-only environment
capture and one reference-only readiness probe. It must not reuse this failed
connection as readiness evidence, reuse historical means as a baseline, or
generate V1 before a `READY` result and a complete stability-qualified
reference baseline.

## Phase 20-B.1R — Remote reachability recovery

This follow-up separates transport from scientific readiness. The Windows
OpenSSH client was found at `C:\\Windows\\System32\\OpenSSH\\ssh.exe`
(`OpenSSH_for_Windows_9.5p2`). `ssh -G v100-srv` retained `v100-srv` as the
effective hostname, with user `38154` and port `22`; there was no effective
`ProxyJump` or `ProxyCommand`.

The user SSH configuration exists, but has no `Host v100-srv` stanza and no
applicable `Include` directive. Existing identity-file paths were inspected as
metadata only; no private-key contents were read. This is precisely
`SSH_ALIAS_MISSING`, not a DNS failure for a configured remote host, a routing
failure, a TCP failure, an authentication failure, or evidence about GPU
noise. One bounded `BatchMode` confirmation failed during resolution before a
TCP connection or authentication could occur.

Accordingly, transport is `REMOTE_REACHABILITY_BLOCKED`; scientific
environment readiness remains `UNKNOWN`. No remote environment capture or
reference-only readiness probe was run. No Phase 20 reference baseline or V1
was created. The next legal action is an externally authorized restoration of
the intended SSH alias/configuration, followed by a later independent
reachability check and then the ordinary readiness gate.

## Direct-endpoint supersession

The previous `v100-srv` alias evidence is retained as historical transport
evidence but is `SUPERSEDED_BY_DIRECT_REMOTE_ENDPOINT`: the authorized endpoint
for this continuation is `<REMOTE_USER>@<REMOTE_HOST>`. No SSH configuration was
edited and no replacement alias was created.

Read-only `Test-NetConnection` established that TCP port 22 on the direct IP
is reachable from the local WLAN interface. A normal OpenSSH connection then
reached the interactive authentication prompt. It did not complete a remote
command because no credential was automated, persisted, written to an
artifact, or passed in a command line. The one locally owned, prompt-blocked
SSH process was terminated after its provenance was verified; no other SSH
process was touched.

The precise current transport state is
`AUTHENTICATION_INTERACTION_REQUIRED`, not DNS failure, TCP timeout,
connection refusal, or GPU readiness. Since no authenticated remote command
completed, no remote identity/toolchain/process inspection, GPU telemetry, or
reference-only readiness probe occurred. Scientific readiness remains
`UNKNOWN`; V1 remains absent and unauthorized. A future normal interactive SSH
session with securely supplied runtime authentication is required before the
ordinary environment and baseline gates can resume.
