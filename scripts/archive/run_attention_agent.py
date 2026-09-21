import json, sys
from pathlib import Path
from agent_backends.codex_appserver import run_one_turn

root=Path(__file__).resolve().parent
episode=sys.argv[1] if len(sys.argv)>1 else 'A1'
ep=root/'campaigns/targets/megatron_5be9626/megatron_native_dot_product_attention'/f'episode_{episode}'
ep.mkdir(parents=True,exist_ok=True)
target=root/'targets/megatron_5be9626/megatron_native_dot_product_attention'
c=json.loads((target/'optimization_contract.json').read_text())
r=json.loads((target/'replay_contract.json').read_text())
p=json.loads((target/'performance_contract.json').read_text())
b=json.loads((target/'baseline_manifest.json').read_text())
feedback = '' if episode == 'A1' else '''
=== A1 TOOLCHAIN FAILURE ===
A1 was a real Agent candidate and remains immutable. It never reached runtime. CUDA 11.8/sm_70 rejected CUDART_INF_F with: identifier "CUDART_INF_F" is undefined. Generic preflight now rejects this construct before nvcc. Do NOT use CUDART_INF_F. A CUDA 11.8 probe passed with a finite -FLT_MAX-equivalent literal and -INFINITY. This is toolchain feedback, not runtime profile evidence.
=== A1 RESULT ===
A1 proposed FULL_CORE_CUSTOM, but no performance or correctness conclusion is admissible because it did not compile.
=== NEXT SEARCH ===
Choose a fresh implementation strategy or repair the conceptual strategy in this new source. Preserve the exact ABI and semantics. Do not copy-edit A1 manually.
'''
if episode == 'A3':
 feedback += '''
=== A2 REAL NSYS PROFILE ===
A2 is correct and stable but mixed by official shape. It is FULL_CORE_CUSTOM_FP32_ROW_SOFTMAX with one observed candidate kernel and zero NCCL. Reference has six observed kernel instances. Candidate total GPU kernel time was approximately 37.3 us at S=16, 609.7 us at S=64, and 2258.0 us at S=128; reference totals were approximately 76.6, 100.7, and 139.2 us. A2 therefore wins S=16, is modestly faster S=64, and regresses severely S=128.
=== PROFILE DIAGNOSIS ===
Evidence supports FULL_CORE_KERNEL_SCALING and SERIAL_WORK_GROWTH. No NCU-level hardware claim is allowed.
=== A2 STABLE PERFORMANCE ===
S16 5.928x; S64 1.167x; S128 0.347x; geometric mean 1.338883x; CI95 [1.307531x,1.368066x].
=== NEXT SEARCH ===
Do not reproduce A2 blindly. Preserve exact semantics and ABI. Develop a shape-aware or tiled/vendor-GEMM-plus-custom-softmax strategy that covers all official shapes without excluding QK/PV. Shape dispatch across the explicit official configurations is permitted by the optimization contract. Do not add masks, dropout, TE, flash-attn, or backward.
'''
prompt=f'''You are a CUDA kernel Agent. Generate one standalone candidate only. Do not modify Megatron upstream.
=== REAL MEGATRON TARGET ===
Megatron commit {c['megatron_commit']}; selected backend is DotProductAttention.forward, native unfused local core.
=== EXACT FORWARD EQUATIONS ===
scores = Q K^T * 0.125; P = dense no-mask softmax with attention_softmax_in_fp32=true; dropout p=0 identity; O = P V.
=== REAL Q/K/V LAYOUT ===
Contiguous FP16 Q/K/V [S,B,16,64], output contiguous FP16 [S,B,1024]. Official shapes are S/B = 16/1, 64/2, 128/2.
=== EXACT ABI ===
extern "C" void dot_product_attention_forward_fp16_stream(const __half* q, const __half* k, const __half* v, __half* output, int64_t seq_len, int64_t batch, int64_t num_heads, int64_t head_dim, float scale, cudaStream_t stream);
Required marker exactly: // AKA_TARGET_CONTRACT: megatron_native_dot_product_attention_forward:{c['contract_hash']}
=== FP32 SOFTMAX SEMANTICS ===
Match Megatron attention_softmax_in_fp32=true and FP16 output tolerance max_abs <= 0.005. Do not change scale or numerical order without justification.
=== DENSE / NO-MASK / P=0 CONTRACT ===
Only dense attention_mask=None and dropout=0. No causal/padding/arbitrary mask and no RNG.
=== OFFICIAL CONFIGS ===
{json.dumps(r['official_configurations'])}
=== ACTIVE REFERENCE BASELINE ===
Hash {b['baseline_hash']}; means: {json.dumps(b['statistics'])}; reference stability PASS under policy {p['contract_hash']} and statistical policy 692f9718d1e12d922561e6d8c69cef835a37cd9c0f6e087c2ab0b342e231ae41.
=== REFERENCE KERNEL GRAPH ===
Native path uses Volta FP16 GEMM for QK and PV, Megatron softmax warp kernel, copies/conversions, and layout work; no NCCL.
=== WORKSPACE POLICY ===
Reference global score workspace is prepared outside timing. Candidate must not require a workspace argument or allocate during the timed call; it may stream/tile or use vendor GEMM only if the full boundary remains included.
=== CURRENT STREAM RULE ===
Enqueue all work on the passed cudaStream_t. Output is separate preallocated storage; Q/K/V are read-only and non-overlapping with output.
=== CUDA 11.8 / SM70 TOOLCHAIN ===
Use V100 sm_70 and CUDA 11.8-compatible code. Avoid unsupported modern CUDA constructs.
=== ZERO-COLLECTIVE INVARIANT ===
No NCCL, torch.distributed, or process-group operation in the local core.
=== FORBIDDEN TARGET EXPANSION ===
Do not implement causal attention. Do not implement arbitrary mask support. Do not implement dropout RNG. Do not implement TE or flash-attn. Do not include QKV/output projection GEMMs. Do not optimize backward. Do not modify Megatron.
Write candidate.cu, hypothesis.json, and AGENT.md in this episode directory and stop. Include the exact contract marker and complete semantic hypothesis. Do not benchmark or edit framework files.
{feedback}'''
(ep/'agent_prompt.txt').write_text(prompt,encoding='utf-8')
result=run_one_turn(prompt,ep)
(ep/'agent_response.txt').write_text(str(result),encoding='utf-8')
print(json.dumps({'episode':episode,'candidate':(ep/'candidate.cu').exists(),'path':str(ep)}))
