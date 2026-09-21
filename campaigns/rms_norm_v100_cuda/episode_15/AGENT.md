# Episode 15 kernel summary

This candidate keeps the robust one-block-per-row reduction structure used by the incumbent, while making the main hidden-size path explicitly vectorized with `float4` loads. The vectorized path is enabled only when `hidden` is divisible by four; otherwise it falls back to scalar indexing, preserving correctness for general shapes.

Each of the 256 threads accumulates a coalesced subset of the row using FMA operations. Warp shuffle reduction combines lanes, and only eight warp totals are placed in shared memory before a final warp-level reduction. The resulting inverse RMS is computed once with `rsqrtf`, synchronized, and reused for the output pass. The output remains coalesced and applies the weight elementwise without an extra temporary buffer.

The file is standalone CUDA and exposes the required `extern "C" void launch_kernel(...)` entry point. No benchmark or compilation was run, per instructions.