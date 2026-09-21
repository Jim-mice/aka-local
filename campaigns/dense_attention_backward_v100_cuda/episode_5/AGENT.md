# AGENT.md

Implemented `candidate.cu` as a standalone CUDA C++ dense-attention backward kernel for V100/sm_70.

The implementation keeps the required C ABI and contract marker, computes all three gradients, and uses 256-thread blocks. Query-gradient rows use warp shuffle reduction with a small shared-memory staging area for the row normalization statistic. Value and key gradients use contiguous per-feature output assignments so threads write adjacent FP32 elements.

The requested constraints were followed: no Python, no PyTorch, no newer architecture target, no benchmark execution, and only the three requested files were created or updated. The kernel was not compiled or benchmarked, per instructions.
