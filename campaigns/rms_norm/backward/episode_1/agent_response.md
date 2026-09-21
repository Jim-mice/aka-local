已完成一个隔离 CUDA 候选，未运行 CUDA、Atrex、benchmark、profiler 或远程任务。

写入文件：

- [candidate.py](C:/Users/38154/projects/aka-local/campaigns/rms_norm/backward/episode_1/candidate.py)
- [AGENT.md](C:/Users/38154/projects/aka-local/campaigns/rms_norm/backward/episode_1/AGENT.md)

候选仅采用一个优化方向：每个 RMSNorm token row 使用一个 256-thread CUDA block，通过共享内存进行 float32 tree reduction，并在同一个 kernel 中完成归一化与 bfloat16 写回。官方参考、评测器、baseline 及工作区外文件均未修改。