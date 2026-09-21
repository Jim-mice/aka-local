import ctypes, os, statistics, torch, torch.distributed as dist

SHAPES = [(8, 1, 32), (32, 2, 64), (128, 2, 128)]

def reference_max(x, lm):
    lm.copy_(x.float().amax(dim=1))

def reference_prepare(x, target, lm, pred, den, ex, mask, tl, rank, lv):
    rows = x.shape[0]
    xf = x.float(); shifted = xf - lm[:, None]
    torch.exp(shifted, out=ex); den.copy_(ex.sum(dim=1))
    idx = (target - rank * lv).clamp(0, lv - 1); tl.copy_(idx)
    owned = (target >= rank * lv) & (target < (rank + 1) * lv)
    mask.zero_(); mv = mask.view(rows, lv)
    mv[torch.arange(rows, device=x.device), idx] = owned.to(torch.uint8)
    pred.zero_(); pred[owned] = shifted[torch.arange(rows, device=x.device)[owned], idx[owned]]

def candidate_functions(so):
    lib = ctypes.CDLL(so); lib.local_max_fp16_stream.restype = ctypes.c_int; lib.local_prepare_fp16_stream.restype = ctypes.c_int; return lib

def run_max(mode, lib, x, lm, rows, lv, stream):
    if mode == 'reference':
        reference_max(x, lm); return
    lib.local_max_fp16_stream(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(lm.data_ptr()), rows, lv, ctypes.c_void_p(stream.cuda_stream))

def run_prepare(mode, lib, x, target, lm, pred, den, ex, mask, tl, rows, lv, rank, stream):
    if mode == 'reference':
        reference_prepare(x, target, lm, pred, den, ex, mask, tl, rank, lv); return
    lib.local_prepare_fp16_stream(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(target.data_ptr()), ctypes.c_void_p(lm.data_ptr()), ctypes.c_void_p(pred.data_ptr()), ctypes.c_void_p(den.data_ptr()), ctypes.c_void_p(ex.data_ptr()), ctypes.c_void_p(mask.data_ptr()), ctypes.c_void_p(tl.data_ptr()), rows, lv, rank, ctypes.c_void_p(stream.cuda_stream))

def common_finalize_backward(pred, den, ex, mask, tl, loss, soft, grad, target, rank, lv):
    rows = target.numel(); torch.log(den, out=loss); loss.sub_(pred); torch.div(ex, den[:, None], out=soft); grad.copy_(soft)
    # Canonicalize the saved ownership state identically for both implementations.
    idx = (target - rank * lv).clamp(0, lv - 1); tl.copy_(idx); owned = (target >= rank * lv) & (target < (rank + 1) * lv)
    mask.zero_(); mask.view(rows, lv)[torch.arange(rows, device=target.device), idx] = owned.to(torch.uint8)
    grad[torch.arange(rows, device=target.device)[owned], idx[owned]] -= 1.0

def main():
    rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE']); local = int(os.environ['LOCAL_RANK']); mode = os.environ.get('MODE', 'reference')
    torch.cuda.set_device(local); dist.init_process_group('nccl'); lib = candidate_functions(os.environ['SO']) if mode == 'candidate' else None; output = []
    for S, B, V in SHAPES:
        rows = S * B; lv = V // world; torch.manual_seed(700 + rows)
        full = torch.randn(rows, V, device='cuda', dtype=torch.float16); target = torch.randint(0, V, (rows,), device='cuda', dtype=torch.long); x = full[:, rank * lv:(rank + 1) * lv].contiguous()
        lm = torch.empty(rows, device='cuda', dtype=torch.float32); pred = torch.empty_like(lm); den = torch.empty_like(lm); ex = torch.empty(rows, lv, device='cuda', dtype=torch.float32); mask = torch.empty(rows * lv, device='cuda', dtype=torch.uint8); tl = torch.empty(rows, device='cuda', dtype=torch.long); loss = torch.empty_like(lm); soft = torch.empty_like(ex); grad = torch.empty_like(ex); stream = torch.cuda.current_stream()
        for _ in range(5):
            dist.barrier(); run_max(mode, lib, x, lm, rows, lv, stream); dist.all_reduce(lm, op=dist.ReduceOp.MAX); run_prepare(mode, lib, x, target, lm, pred, den, ex, mask, tl, rows, lv, rank, stream); dist.all_reduce(pred); dist.all_reduce(den); common_finalize_backward(pred, den, ex, mask, tl, loss, soft, grad, target, rank, lv)
        fullf = full.float(); oracle_loss = torch.logsumexp(fullf, dim=1) - fullf[torch.arange(rows, device='cuda'), target]
        expected_grad = torch.softmax(fullf, dim=1)[:, rank * lv:(rank + 1) * lv]
        owned = (target >= rank * lv) & (target < (rank + 1) * lv)
        expected_grad[torch.arange(rows, device='cuda')[owned], (target[owned] - rank * lv)] -= 1.0
        loss_err = (loss - oracle_loss).abs().max().item(); soft_err = (soft - torch.softmax(fullf, dim=1)[:, rank * lv:(rank + 1) * lv]).abs().max().item(); grad_err = (grad - expected_grad).abs().max().item()
        torch.cuda.synchronize(); samples = []
        for _ in range(5):
            dist.barrier(); torch.cuda.synchronize(); start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True); start.record(); run_max(mode, lib, x, lm, rows, lv, stream); dist.all_reduce(lm, op=dist.ReduceOp.MAX); run_prepare(mode, lib, x, target, lm, pred, den, ex, mask, tl, rows, lv, rank, stream); dist.all_reduce(pred); dist.all_reduce(den); common_finalize_backward(pred, den, ex, mask, tl, loss, soft, grad, target, rank, lv); end.record(); torch.cuda.synchronize(); samples.append(start.elapsed_time(end) * 1000.0)
        output.append({'rows': rows, 'V': V, 'loss_max_abs': loss_err, 'saved_softmax_max_abs': soft_err, 'grad_max_abs': grad_err, 'correctness_pass': loss_err <= 2e-3 and soft_err <= 3e-3 and grad_err <= 3e-2, 'samples_us': samples, 'mean_us': statistics.mean(samples), 'std_us': statistics.stdev(samples), 'cv': statistics.stdev(samples) / statistics.mean(samples)})
    print({'mode': mode, 'rank': rank, 'world': world, 'rows': output}, flush=True); dist.barrier(); dist.destroy_process_group()

if __name__ == '__main__': main()
