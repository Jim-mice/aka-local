# A2 candidate

Standalone CUDA candidate for the exact native `DotProductAttention.forward`
boundary: dense attention, no mask, dropout zero, contiguous FP16 Q/K/V in
`[S,B,H,D]`, and contiguous FP16 output. One block owns each attention row.
Scores and softmax are FP32; PV is accumulated in FP32 before FP16 storage.
Dynamic shared memory is used without allocation during the call, and all work
is enqueued on the supplied stream.

The exact ABI and marker are in `candidate.cu`. It targets CUDA 11.8 / V100
`sm_70` and does not use `CUDART_INF_F`. This artifact has not been benchmarked
or validated for compilation, numerical tolerance, flashing, or hardware use.
