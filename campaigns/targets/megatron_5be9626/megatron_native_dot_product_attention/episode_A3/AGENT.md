# A3 candidate

Standalone implementation of the specified dense native DotProductAttention forward boundary. One 128-thread block handles each query row and performs QK, FP32 max-subtracted softmax, and PV in one CUDA kernel. Dynamic shared memory holds row scores and reduction scratch; no workspace argument or device allocation is used. The supplied stream is used for the launch.

Supports the official `S=16/64/128`, `B=1/2`, `H=16`, `D=64` configurations, accepts the caller scale, reads Q/K/V only, and writes contiguous `[S,B,1024]` FP16 output. No masks, causal mode, dropout, projections, collectives, or backward path are included. `-FLT_MAX` is used; `CUDART_INF_F` is intentionally absent.

No benchmark or framework modification was performed.
