import json, torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig

torch.manual_seed(14)
device = torch.device('cuda')
S, B, H, I = 128, 2, 1024, 4096
config = TransformerConfig(num_layers=1, hidden_size=H, num_attention_heads=16, ffn_hidden_size=I, gated_linear_unit=True, activation_func=F.silu, add_bias_linear=False, bias_activation_fusion=False, use_te_activation_func=False, tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

class LocalLinear(nn.Module):
    def __init__(self, ins, outs, **kw):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(outs, ins, device=device, dtype=torch.float16) * 0.01)
    def forward(self, x):
        return F.linear(x, self.weight), None

subs = MLPSubmodules(linear_fc1=lambda *a, **k: LocalLinear(a[0], a[1]), linear_fc2=lambda *a, **k: LocalLinear(a[0], a[1]))
mlp = MLP(config, subs, input_size=H, ffn_hidden_size=I).to(device=device, dtype=torch.float16).eval()
cases = []
for S, B in ((16, 1), (64, 2), (128, 2)):
    x = torch.randn(S, B, H, device=device, dtype=torch.float16)
    with torch.no_grad():
        ref, bias = mlp(x)
        intermediate, b = mlp.linear_fc1(x)
        gate, up = torch.chunk(intermediate, 2, dim=-1)
        activation = F.silu(gate) * up
        replay, replay_bias = mlp.linear_fc2(activation)
    diff = (ref - replay).abs()
    case = {'shape': [S, B, H], 'dtype': str(x.dtype), 'x_stride': list(x.stride()), 'output_stride': list(ref.stride()), 'max_abs_error': float(diff.max()), 'replay_pass': bool(torch.allclose(ref,replay,atol=2e-3,rtol=2e-3))}
    if [S, B] == [128, 2]:
        for _ in range(10): mlp(x)
        torch.cuda.synchronize(); start, end = torch.cuda.Event(True), torch.cuda.Event(True); start.record()
        for _ in range(100): mlp(x)
        end.record(); torch.cuda.synchronize(); case['reference_latency_us'] = start.elapsed_time(end) * 1000 / 100
    cases.append(case)
payload = {'commit': '5be9626709af2722333bf54797c954c09edeada3', 'source_call': 'megatron.core.transformer.mlp.MLP.forward', 'config': {'hidden_size': H, 'ffn_hidden_size': I, 'tensor_parallel': 1, 'bias_activation_fusion': False, 'use_te_activation_func': False}, 'cases': cases}
print(json.dumps(payload, indent=2))
