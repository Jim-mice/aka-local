import json, sys
from pathlib import Path
from agent_backends.codex_appserver import run_one_turn

root=Path(__file__).parents[3]
n=int(sys.argv[1])
ep=root/f'campaigns/targets/megatron_5be9626/swiglu_backward/episode_{n}'
ep.mkdir(parents=True,exist_ok=True)
c=json.loads((root/'targets/megatron_5be9626/swiglu/backward_optimization_contract.json').read_text())
profile=''
if n>1:
    profile='''
=== CURRENT PERFORMANCE ===
The prior backward candidate was evaluated only within saved-P backward scope.
=== REAL BACKWARD PROFILE EVIDENCE ===
See prior episode profile artifacts; preserve current-stream ABI and inspect launch count.
=== CURRENT DIAGNOSIS ===
MULTI_KERNEL_OVERHEAD if multiple PyTorch elementwise kernels remain; otherwise UNKNOWN.
=== PROFILE-GUIDED RECOMMENDATIONS ===
Fuse sigmoid, derivative, two products, and inverse packing in one kernel; preserve FP16 semantics.
=== WHAT WORKED ===
Exact gradient packing and FP16 tolerance.
=== WHAT FAILED ===
No assumption may expand into FC1/FC2.
=== NEXT RECOMMENDED SEARCH ===
Optimize only saved-intermediate plus grad-output to grad-intermediate.
'''
p=f'''You are a CUDA kernel Agent. Generate one standalone backward candidate only. Do not modify Megatron.
=== REAL MEGATRON TARGET ===
SwiGLU backward at commit {c['megatron_commit']}
=== BACKWARD REPLACEMENT BOUNDARY ===
Only backward of optional-bias-free activation boundary. FC1, FC2, forward, collectives, and bias gradient are excluded.
=== SAVED TENSORS AVAILABLE ===
intermediate [rows,width], grad_output [rows,width/2]. Both contiguous FP16.
=== ABI ===
{c['abi']}
Required marker exactly: {c['contract_marker']}
=== SEMANTICS ===
{c['mathematical_backward']}
=== SHAPES ===
[16,1,1024], [64,2,1024], [128,2,1024], rows=S*B,width=8192,offset=0.
=== FORBIDDEN ===
Do not change forward, do not include GEMMs, do not add bias semantics, do not synchronize, launch on passed stream.
{profile}
Write candidate.cu, hypothesis.json, AGENT.md in this episode directory and stop.'''
(ep/'agent_prompt.txt').write_text(p,encoding='utf-8')
r=run_one_turn(p,ep)
(ep/'agent_response.txt').write_text(str(r),encoding='utf-8')
print('episode',n,'candidate',(ep/'candidate.cu').exists())
