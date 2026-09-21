# Learned CUDA Optimizations

## half2 vectorized fused SwiGLU on RTX 5060

Observed:

The copied V2b kernel loads and stores two FP16 values per thread with `half2`, computes the SiLU intermediates in float, and fuses the activation and multiply into one CUDA launch.

Evidence:

On RTX 5060 Laptop GPU (`sm_120`), same-process A/B/B/A timing passed for M=256, 1024, and 4096 with candidate speedups of 1.582x, 1.506x, and 1.673x respectively. `ptxas` reported 30 registers/thread and no spills. Nsight Compute successfully collected a filtered kernel profile.

Reason:

The fused implementation removes an intermediate global-memory write/read and launch boundary. `half2` reduces the number of FP16 memory transactions/instructions while retaining float arithmetic for the nonlinear calculation.

Applicable when:

Inputs and output are FP16, the element count is predominantly even and aligned, and the fused operation is large enough for launch overhead and memory traffic to matter.

Failure cases / caveats:

Odd tails must be handled explicitly. FP16 output comparison should use an appropriate absolute tolerance; relative error can look large near values close to zero. A result from one GPU architecture must not be assumed to transfer unchanged to V100 or another architecture without remeasurement.

## RMSNorm reduction candidate on RTX 5060

The official checkout had no RMSNorm backward or SwiGLU backward contract, so official RMSNorm forward was used as the reduction-oriented fallback. Luna proposed one fused CUDA kernel with one 256-thread block per token row, a shared-memory float32 tree reduction, and a second row pass for normalized output.

The candidate compiled for sm_120 and passed official correctness on all 56 workloads. Same-process A/B/B/A against the official eager reference used 20 warmups and 100 repetitions: arithmetic mean speedup 7.202740x, geometric mean 6.145711x, and total-time ratio 4.652903x. The slowest shape was 0.972903x and the fastest was 17.997631x.

cuobjdump reported 30 registers/thread, 2048 shared bytes/block, zero stack/local bytes, 256 threads/block, and one block per row. Decision: PROMOTE by mechanical policy; no automatic source replacement was performed and the candidate remains isolated. The <2% robustness trigger was not entered because the aggregate improvement was large.
