import os, statistics, torch, torch.distributed as dist
from remote_comparable_bench import run_max, run_prepare, candidate_functions

def measure(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(); fn(); end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0

def loss_soft(pred, den, ex, loss, soft):
    torch.log(den, out=loss); loss.sub_(pred); torch.div(ex, den[:, None], out=soft)

def backward_local(soft, grad, mask, tl, target, rank, lv):
    rows = target.numel()
    idx = (target - rank * lv).clamp(0, lv - 1)
    tl.copy_(idx)
    owned = (target >= rank * lv) & (target < (rank + 1) * lv)
    mask.zero_(); mask.view(rows, lv)[torch.arange(rows, device=target.device), idx] = owned.to(torch.uint8)
    grad.copy_(soft)
    grad[torch.arange(rows, device=target.device)[owned], idx[owned]] -= 1.0

def main():
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK']); mode = os.environ.get('MODE', 'reference')
    torch.cuda.set_device(local); dist.init_process_group('nccl')
    lib = candidate_functions(os.environ['SO']) if mode == 'candidate' else None
    rows, V, world = 64, 64, 2; lv = V // world
    torch.manual_seed(700 + rows); full = torch.randn(rows, V, device='cuda', dtype=torch.float16)
    target = torch.randint(0, V, (rows,), device='cuda', dtype=torch.long); x = full[:, rank*lv:(rank+1)*lv].contiguous()
    lm = torch.empty(rows, device='cuda', dtype=torch.float32); pred = torch.empty_like(lm); den = torch.empty_like(lm)
    ex = torch.empty(rows, lv, device='cuda', dtype=torch.float32); mask = torch.empty(rows*lv, device='cuda', dtype=torch.uint8)
    tl = torch.empty(rows, device='cuda', dtype=torch.long); loss = torch.empty_like(lm); soft = torch.empty_like(ex); grad = torch.empty_like(ex)
    stream = torch.cuda.current_stream()
    for _ in range(5):
        dist.barrier(); run_max(mode, lib, x, lm, rows, lv, stream); dist.all_reduce(lm, op=dist.ReduceOp.MAX)
        run_prepare(mode, lib, x, target, lm, pred, den, ex, mask, tl, rows, lv, rank, stream)
        dist.all_reduce(pred); dist.all_reduce(den); loss_soft(pred, den, ex, loss, soft); backward_local(soft, grad, mask, tl, target, rank, lv)
    torch.cuda.synchronize(); out=[]
    for iteration in range(10):
        dist.barrier(); torch.cuda.synchronize()
        tmax = measure(lambda: run_max(mode, lib, x, lm, rows, lv, stream)); dist.all_reduce(lm, op=dist.ReduceOp.MAX)
        tprepare = measure(lambda: run_prepare(mode, lib, x, target, lm, pred, den, ex, mask, tl, rows, lv, rank, stream))
        ttarget = measure(lambda: dist.all_reduce(pred)); tden = measure(lambda: dist.all_reduce(den))
        tloss_soft = measure(lambda: loss_soft(pred, den, ex, loss, soft))
        tback = measure(lambda: backward_local(soft, grad, mask, tl, target, rank, lv))
        out.append({'iteration': iteration, 'max_local_us': tmax, 'prepare_local_us': tprepare, 'sum_target_us': ttarget, 'sum_den_us': tden, 'loss_softmax_us': tloss_soft, 'backward_local_us': tback})
    print({'rank': rank, 'mode': mode, 'rows': out, 'means_us': {k: statistics.mean([r[k] for r in out]) for k in out[0] if k.endswith('_us')}}, flush=True)
    dist.barrier(); dist.destroy_process_group()

if __name__ == '__main__': main()
