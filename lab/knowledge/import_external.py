"""CLI: ``python -m lab.knowledge.import_external``."""
from __future__ import annotations
from pathlib import Path
import json
from .importers.external import ExternalImporter

ROOT=Path(__file__).resolve().parents[2]

def main():
    importer=ExternalImporter(ROOT)
    performance=lambda path:any(word in path.lower() for word in ("profil","cuda","reduction","flash","triton","cutlass","cute","fused","rmsnorm","liger","tensor","memory","performance","kernel"))
    importer.results.append(importer.import_repo_markdown("modal_gpu_glossary","https://github.com/modal-labs/gpu-glossary.git",subdir="gpu-glossary"))
    importer.results.append(importer.import_repo_markdown("gpu_mode_resource_stream","https://github.com/gpu-mode/resource-stream.git",index_only=True))
    importer.results.append(importer.import_repo_markdown("gpu_mode_lectures","https://github.com/gpu-mode/lectures.git",include=performance,max_documents=250))
    for source_id,url,title,version in (
        ("nvidia_cuda_programming_guide","https://docs.nvidia.com/cuda/cuda-programming-guide/","NVIDIA CUDA Programming Guide",None),
        ("nvidia_cuda_best_practices","https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/","NVIDIA CUDA Best Practices Guide",None),
        ("nvidia_blackwell_tuning","https://docs.nvidia.com/cuda/blackwell-tuning-guide/","NVIDIA Blackwell Tuning Guide",None),
        ("nvidia_volta_tuning","https://docs.nvidia.com/cuda/volta-tuning-guide/","NVIDIA Volta Tuning Guide",None),
        ("cutlass_docs","https://docs.nvidia.com/cutlass/latest/","NVIDIA CUTLASS Documentation","latest"),
    ): importer.results.append(importer.import_web_page(source_id,url,title=title,version=version))
    manifest=importer.finish()
    # Windows PowerShell may still expose a legacy GBK stdout. The durable
    # manifest/report retain UTF-8; CLI status remains safe to print.
    print(json.dumps(manifest, ensure_ascii=True))

if __name__=="__main__":main()
