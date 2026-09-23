#!/usr/bin/env python3
"""Reference-only outer-step stability calibration for Snapshot B/C diagnosis."""
import argparse, json, math, os, socket, statistics, time
import torch
import torch.distributed as dist

TARGETS_MS = [20, 50, 100, 200, 500, 1000]

def port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

def summary(values):
    ordered = sorted(values); mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    def pct(q):
        p = (len(ordered)-1)*q; lo, hi = math.floor(p), math.ceil(p)
        return ordered[lo] if lo == hi else ordered[lo]*(hi-p)+ordered[hi]*(p-lo)
    return {"n":len(values),"mean_ms":mean,"median_ms":statistics.median(values),"stdev_ms":sd,"cv":sd/mean if mean else float("inf"),"p10_ms":pct(.1),"p90_ms":pct(.9),"min_ms":min(values),"max_ms":max(values),"raw_ms":values}

def timed(step, k):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); start.record()
    for _ in range(k): step()
    end.record(); end.synchronize()
    return start.elapsed_time(end)

def choose_k(step, target):
    k = 1
    while k < 256:
        if timed(step, k) >= target: return k
        k *= 2
    return k

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--megatron-path", required=True); ap.add_argument("--output", required=True); ap.add_argument("--targets", default="20,50,100,200,500,1000"); ap.add_argument("--fixed-k", type=int, default=0); args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    import sys; sys.path.insert(0, args.megatron_path)
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port()}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(20260923); torch.manual_seed(20260923); torch.cuda.manual_seed_all(20260923)
    config=TransformerConfig(num_layers=2,hidden_size=1024,num_attention_heads=8,ffn_hidden_size=4096,normalization="RMSNorm",layernorm_epsilon=1e-5,params_dtype=torch.float16,fp16=True,use_cpu_initialization=True,hidden_dropout=0.0,attention_dropout=0.0,masked_softmax_fusion=False,bias_activation_fusion=False,bias_dropout_fusion=False)
    model=GPTModel(config=config,transformer_layer_spec=get_gpt_layer_local_spec(),vocab_size=2048,max_sequence_length=128,position_embedding_type="rope",parallel_output=False).cuda().half()
    tokens=torch.randint(0,2048,(2,128),device="cuda",dtype=torch.long); positions=torch.arange(128,device="cuda").unsqueeze(0).expand(2,128); mask=torch.triu(torch.ones(128,128,device="cuda",dtype=torch.bool),diagonal=1).view(1,1,128,128).expand(2,1,128,128); labels=tokens.clone(); opt=torch.optim.SGD(model.parameters(),lr=0.0)
    def step():
        opt.zero_grad(set_to_none=True); loss=model(input_ids=tokens,position_ids=positions,attention_mask=mask,labels=labels).float().mean(); loss.backward(); opt.step()
    results=[]
    for target in [int(x) for x in args.targets.split(",")]:
        k=args.fixed_k if args.fixed_k else choose_k(step,target)
        for _ in range(20): timed(step,k)
        values=[timed(step,k) for _ in range(30)]
        results.append({"target_ms":target,"k":k,"summary":summary(values),"warmup_windows":20,"measured_windows":30,"timing":"outer CUDA Event","sample_trimming":"NONE"})
    with open(args.output,"w",encoding="utf-8") as h: json.dump({"status":"PASS","gpu_uuid":"GPU-88be8e63-dd61-3d8f-2f44-e1f93204654d","results":results},h,indent=2)
    parallel_state.destroy_model_parallel(); dist.destroy_process_group()

if __name__ == "__main__": main()
