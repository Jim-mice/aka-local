# Candidate Agent Notes

This candidate targets the rank-local forward preparation stage for TP=2 vocab-parallel cross entropy.

- `local_max_fp16` computes the local FP32 row maximum. The caller must perform the required real MAX all-reduce.
- `local_prepare_fp16` computes local shifted exponent sums, target ownership/extraction, and local predicted/loss contributions. The caller must perform both required real SUM all-reduces in baseline order.
- No collective, process-group initialization, backward path, synchronization, or full-vocabulary reconstruction is included.
- Launches use the caller's current CUDA stream and expose asynchronous CUDA error behavior to the caller.
- This is a candidate only; no benchmark or runtime claim is made.
