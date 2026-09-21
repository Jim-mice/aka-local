# Candidate notes

This candidate implements the exact rank-local backward boundary in one direct
FP32 CUDA kernel. Each thread owns one `[row, local_vocab_column]` output,
reads the frozen softmax value, scales it by that row's gradient, and performs
the owner-row subtraction when the local target matches.

The launch uses the required `cudaStream_t` directly and does not synchronize,
allocate, communicate, cast, or invoke any forward-path logic. Empty dimensions
return without launching; positive dimensions use a 256-thread one-dimensional
grid.
