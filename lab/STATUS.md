# GPU Learning Lab Status

This is a separate knowledge/library layer inside aka-local. Existing experiments remain in their original locations.

## Platforms

- `rtx5060_laptop_sm120`: local Windows, CUDA 13.4, PyTorch 2.14+cu130, NCU available.
- `v100_sm70`: historical Crater/V100 evidence; profiler permission was unavailable.
- `biv150_corex`: historical Linux BI-V150 evidence; CoreX/Triton; exact CC/VRAM unknown.

## Imported experiment facts

Nine metadata records are imported: standalone RMSNorm V0/V1/V2, standalone SwiGLU V2b/V3, remote Residual RMSNorm V5, remote Bias SwiGLU V6, and remote MoE V0/V1. They point to original artifacts and do not copy binaries/logs.

## Knowledge counts

- OBSERVATION: 1
- HYPOTHESIS: 0
- SUPPORTED_RULE: 0
- ANTI_STRATEGY: 2
- BACKEND_QUIRK: 1
- HARDWARE_FACT: 0
- OPEN_QUESTION: 1

No abstract rule has been promoted across GPU architectures.

## Safety

No Agent, CUDA, Triton, Atrex, benchmark, NCU, remote job, or authentication action is run by this library's import/index tools.
