# V100 RMSNorm Human Challenge — Snapshot B

This package is `V100_RMSNORM_E2E_SNAPSHOT_B`, derived from Snapshot A only to repair OJ and packaging gates.

The candidate ABI is a Python integration adapter: a Python file exporting `TritonRMSNorm(torch.nn.Module)`. The class name is frozen for runner compatibility; the implementation is not required to use Triton. A candidate may use Triton, a native CUDA extension, or a PyTorch custom op if it obeys the frozen contract, does not modify the OJ, and does not silently fall back to reference RMSNorm.

L0 checks operator forward/backward correctness and official timing. L1 checks real Megatron `WrappedTorchNorm` replacement, invocation, forward/backward completion, finite gradients, loss correctness, and RMSNorm-related gradient correctness. L2 checks the controlled GPTModel whole-step with paired reference/candidate outer CUDA-event timing and fail-closed stability/protocol gates.

This is Controlled Megatron E2E, not Nine-grid E2E. Human blind is intentionally `NO`: the human may know historical v28, but the comparison contract, runner, GPU, shapes, and gates are identical.
