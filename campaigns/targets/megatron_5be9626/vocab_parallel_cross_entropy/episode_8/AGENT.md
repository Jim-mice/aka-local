# Candidate notes

This candidate optimizes one local stage only. `local_prepare_fp16_stream` performs target masking, local target extraction, shifted exponentiation, local predicted-logit contribution, and local denominator accumulation in a single row-per-block kernel.

The caller remains responsible for the real MAX all-reduce, the two SUM all-reduces, loss, and softmax. The implementation uses the verified finite FP32 initializer and preserves both required exported C ABI symbols verbatim.
