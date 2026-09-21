# Candidate B2

This candidate implements only the required local FP32 vocab-parallel cross-entropy backward ABI.

It uses one CUDA kernel with a grid-stride traversal over `[rows, local_vocab]`. Each element computes `softmax * grad_output`; owner rows subtract `grad_output` at the local target column. Non-owning rows remain unchanged apart from the multiply.

The implementation preserves the supplied softmax buffer, supports general positive dimensions, launches only on the supplied `cudaStream_t`, and performs no device-wide synchronization or collective operation.
