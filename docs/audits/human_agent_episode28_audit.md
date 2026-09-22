# Human Agent Episode 28 Audit

## 1. Executive summary

本报告只审查 `D:\Users\38154\Downloads\aka-local-main\aka-local-main`。该目录是 ZIP 解压副本，不含 `.git`；本轮没有连接 V100、运行 benchmark、生成 candidate 或修改任何生产代码 / `candidate.cu`。唯一新增文件是本报告。`C:\Users\38154\projects\aka-local` 未被读取为实现来源，也未被修改。

最重要的结论如下。

1. Episode 28 的源码确实把每个 row 的 `x` 从“两次 global-memory pass”改成了“首次读取后跨 reduction 保存在寄存器并复用”。这是已证实的结构差异，也是合理的主要性能候选原因；但它和 `float4`、完全展开、`hidden=4096` specialization 等改动同时发生，现有数据不能给出各项贡献比例。
2. Episode 28 的初次 `10.883x` 不能作为已稳定的 promotion 分数。后续复测仍证明它明显快于 v23（v28 常见约 `8.69–8.70x`，一轮 `7.521x`；v23 约 `4.08x`），但同一 candidate 出现约 `9 µs` 和约 `22 µs` 的双峰。旧 evaluator 还会让 `shapes` 和 `evidence` 取自不同 remote job，因此初始 artifact 内的 `8.98 µs` 与 `21.8 µs` 并不矛盾，却不应被混用。
3. 旧 V100 campaign 确实是“单次 ephemeral Agent 生成 → session 关闭 → 机械评测 → 新 episode / 新 Agent”。它没有把 compile、correctness、benchmark 或 profile 结果回送给同一个会话以修复实现。仓库另有 `LongHorizonRunner` / `CodexAgentSession` 的持久会话设计，但现有 D 盘文档明确标为未接入执行，且它服务的是本地 RTX5060 路径，不是这条旧 V100 campaign。
4. contract 身份已污染：Episode 28 的 `hypothesis.json` 写 `0a3d176112c36cee`，但 `episode_manifest.json` 写 `a69b1c9ffaa393a1`。D 盘旧 runner 实际产生后者的方法只是对 evaluation shape 列表做 SHA-256 截断，不是 semantic ABI contract hash；它也没有在 `remote_v100_campaign.run_evaluation_for_episode()` 前调用 contract validator。
5. 知识摘要把机制压缩为 strategy tag、短 lesson 和少量分数。它提到了 `float4`、warp shuffle、256 threads，却没有保留“x 跨 reduction 的生命周期 / 第二次 global load”这类实现级差异，也没有把 `REJECT_COMPILE`、`REJECT_CORRECTNESS` 和 framework failure 从“hypothesis 被否定”中严格分离。

## 2. v23 → v28 kernel delta

### Proven facts

来源：

- `campaigns/rms_norm_v100_cuda/episode_23/candidate.cu`
- `campaigns/rms_norm_v100_cuda/episode_28/candidate.cu`
- `campaigns/rms_norm_v100_cuda/episode_28/result.json`
- `campaigns/rms_norm_v100_cuda/episode_28/rechecks/summary.json`

| 项目 | v23 | v28 | 结论 |
|---|---|---|---|
| block / row mapping | 每 row 一个 256-thread block | 每 row 一个 256-thread block | 相同的高层映射 |
| `x` 首次读取 | scalar loop，`input[base + col]` | 每 thread 显式读取 `v0..v3`，共 4 个 `float4` / 16 floats | v28 的读取分块与 vector 类型改变 |
| reduction 后的 `x` | 第二个 loop 再读 `input[base + col]` | 直接使用仍活跃的 `v0..v3` | **v28 确实消除了第二次 global load of `x`** |
| sum-of-squares | source 显式 `fmaf(v, v, sum_sq)` | source 为显式 `sum_sq += v.* * v.*` | v28 源码没有显式 `fmaf`；是否被编译器融合未证明 |
| reduction | warp shuffle、8 个 shared warp sums、两个 barrier、shared `inv_rms` | warp shuffle、8 个 shared floats、两个 barrier，以 `warp_sum[0]` broadcast `inv_rms` | 两者的 reduction 拓扑接近；不是从大 shared reduction 改到小 shared reduction |
| hidden size | runtime `cols`，通用循环 | `hidden != 4096` 直接返回；固定 4096 | v28 是冻结 shape 专用实现 |
| output | scalar `input * scale * weight` store | 4 个 `float4` weight load、4 个 `float4` output store | v28 vectorized output |
| pointer / index | `base = row * cols`，循环地址计算 | `row * 4096`、固定四个 index group | specialization 同时减少 loop / 地址计算 |
| correctness | v23 artifact max error `0.0` | v28 artifact max error `4.76837158203125e-07` | 两者在当时 evaluator 的正确性检查中通过 |

v23 的第一 loop 读取 `input` 计算平方和；reduction 后第二 loop 再读取 `input` 写输出。v28 在 reduction 前将 row 的全部 4096 个元素分配为每 thread 四个 `float4`，在 reduction 后使用这些同一变量计算输出。因此“缓存 x 跨 reduction 并避免第二次 global read”不是描述性推测，而是源码可验证的事实。

### Likely effects, but not apportioned

- 少一次 `x` 的全局读取很可能降低 memory traffic，也消除了第二次 pass 的 load 指令和部分地址计算；它是最值得做消融验证的机制。
- `float4` load/store、固定 `4096`、显式展开和少量固定索引也都可能减少 loop/control/address 指令，或改善 memory transaction 形态。
- v28 把 16 个输入 float 保持跨两个 barrier。相较 v23，这通常会提升 register 使用量；实际 registers/thread、occupancy、spill 和 active warps 没有 ptxas 或 NCU 证据，不能声称“无 spill”或“occupancy 没受影响”。
- 两个实现都以一个 block 处理一个 row。小 batch 的并行度及 register pressure 对 occupancy 的影响需要编译产物或 profile 才能判断。

### Unproven hypotheses

- 当前没有单变量 ablation，因此不能证明 10.883x 中有多少来自缓存 `x`，多少来自 `float4`、展开或 specialization。
- 没有对照“保留 scalar、仅缓存 x”“保留二次读、仅 float4”“缓存 x 但使用 loop”的同环境测量。
- 对 `float4` 的对齐只是假设。v28 未在 launch 或 kernel 内验证 x/y/weight 16-byte alignment；冻结 allocator 可能满足它，但源码本身没有证明。
- 约 9/22 µs 双峰无法由源码归因给 register pressure、clock、context 或服务器争用；这些都是待测解释，不能从现有 artifact 选一个当结论。

## 3. Episode 1–27 search reconstruction

审查对象中不存在 `episode_1`、`episode_2`、`episode_4` 目录，所以无法从 D 盘副本重建它们。下表将“真正有效 Agent attempt”严格定义为同时具备 `candidate.cu` 与可审查的 `hypothesis.json` / `AGENT.md` 的 episode；这一定义下 1–27 有 **12 个**可审计的 Agent implementation attempts：7、8、11–17、23–25。不能据此断言其余候选一定不是 Agent 写的，只能说现有证据不足以归因。

| episode | hypothesis / implementation outcome | score（如有） | failure class | did it advance the search? |
|---:|---|---:|---|---|
| 1 | 目录缺失 | — | evidence absent | 无法判断 |
| 2 | 目录缺失 | — | evidence absent | 无法判断 |
| 3 | 单 shape legacy artifact；无 hypothesis / AGENT | `1.419` | incomplete provenance / single-shape legacy | 仅早期 baseline-like evidence |
| 4 | 目录缺失 | — | evidence absent | 无法判断 |
| 5 | 有 candidate/result，无 hypothesis / AGENT | `1.144` | incomplete provenance; performance reject | 不能可靠归因机制 |
| 6 | 有 candidate/result，无 hypothesis / AGENT | `1.202` | incomplete provenance; performance reject | 不能可靠归因机制 |
| 7 | one-block, coalesced scalar, FMA, warp/shared reduction | `2.474` | valid performance reject | 是，建立 row/block + shuffle 路线 |
| 8 | 同类 one-block/shuffle | `2.488` | valid performance reject | 有限增量 |
| 9 | 人为/明显的未定义符号与语法错误 | `0` | compile failure | 不应当作 optimization hypothesis failure |
| 10 | 3-shape accepted artifact，无 hypothesis / AGENT | `1.188` | incomplete provenance | 仅机械结果 |
| 11 | 明确 two-pass processing | `2.652` | valid accepted candidate | 是，但把 second pass 固化为常规做法 |
| 12 | reduction 后 second coalesced pass | `2.892` | valid accepted candidate | 是，性能提升；仍未试 x lifetime |
| 13 | coalesced / compact shared / fused output | `2.878` | valid performance reject | 小幅、无超越 incumbent |
| 14 | `float4` vectorization；仅 primary shape 正确 | legacy result `3.507`，最终 `1` | correctness failure（batch indexing 缺失） | 是，暴露 all-shape correctness gate 必要性 |
| 15 | guarded `float4` two-pass | `3.047` | valid accepted candidate | 是，恢复 vector path 的 correctness |
| 16 | coalesced `float4` two-pass | `3.650` | valid accepted candidate | 是 |
| 17 | vector loads/stores, FMA, two-pass | `3.871` | valid accepted candidate | 是，接近 v23 |
| 18 | 与 21/22/26/27 相同的 reference-like source，含 `cudaDeviceSynchronize()` | `1.159` | duplicate fixture / noncompetitive | 不构成新搜索 |
| 19 | `name 'operator' is not defined`；无 evaluation result | — | framework failure | 不应算 hypothesis failure |
| 20 | `unexpected keyword argument 'operator'`；无 evaluation result | — | framework failure | 不应算 hypothesis failure |
| 21 | 与 18 相同 candidate hash / source | `1.126` | duplicate fixture / noncompetitive | 不构成新搜索 |
| 22 | 与 18 相同 source，约 `2614 µs` | `0.038` | duplicate fixture + pathological result | 不构成新搜索；应标环境/测量异常待审 |
| 23 | scalar two-pass、256 thread、warp/shared reduction、`rsqrtf` | `4.077` | valid accepted incumbent | 是，历史 best |
| 24 | `float4` sum pass，仍有 second global read；`uintptr_t` 未定义 | `0` | compile failure | 机制未得到可运行实现验证 |
| 25 | four-way unrolled scalar two-pass | `3.775` | valid performance reject | 是，说明仅展开不足以超过 v23 |
| 26 | 与 18 相同 source，约 `2614 µs` | `0.038` | duplicate fixture + pathological result | 不构成新搜索 |
| 27 | 与 18 相同 source，约 `2614 µs` | `0.038` | duplicate fixture + pathological result | 不构成新搜索 |

`episode_18`、21、22、26、27 的 `candidate.cu` SHA-256 相同，且都在 `launch_kernel()` 内调用 `cudaDeviceSynchronize()`；它们不是五个独立的 optimization ideas。22/26/27 的近乎相同 `2613–2614 µs` 结果尤其不能被压缩成“该 hypothesis 五次失败”。

### Was the key mechanism proposed before episode 28?

在 1–27 可见的 `hypothesis.json`、`AGENT.md` 和 `candidate.cu` 中，没有发现“把完整 x segment 保留在 registers **跨 reduction**，并消除 second global load”的有效实现。11、12、14、16、17 和 25 明确记录了 second pass / second read；24 甚至在 risk 中写出 `second global read of the input row`，但其 proposed code 仍然二次读取，且先因缺少 `uintptr_t` 定义而 compile failed。

`float4`、256-thread row block、warp shuffle、small shared warp-total、vectorized output 已被多次提出；缺失的是把它们组合成明确的 **data-lifetime transformation**，并验证该 transformation 的实现。

## 4. Why the automated Agent missed v28

### Information and prompt

`lab/runtime/evaluators/remote_v100_campaign.py::_build_v100_prompt()` 提供了固定 strategy tags，例如 `register_optimization` 和 `vectorized_loads`，但这些 tag 不表达变量生命周期、跨 barrier live range、是否消除第二个 global pass，或要做哪一种可区分的 ablation。提示也要求“create exactly three files”后停止，因此 Agent 被设计成一次性产物生成器，不是可迭代实现者。

### Context continuity and feedback granularity

旧 V100 campaign 的 `main()` 对每个 episode：创建 `cx.thread_start(..., ephemeral=True)`，只调用一次 `t.run(prompt, ...)`，立即 `cx.close()`；之后才调用 `run_evaluation_for_episode()`。compile、correctness、benchmark 和 profile 结果从未送回同一个 thread。下一 episode 又是新 ephemeral thread，只收到压缩后的 knowledge context。

所以旧路径的真实结构是：

```text
Hypothesis/prompt → one candidate implementation → session ends
→ remote compile/correctness/benchmark/profile → decision
→ next episode, new Agent session
```

这意味着一个 hypothesis 在旧路径中事实上一轮只有一次 implementation attempt。CE 或 WA 只会转化为下一轮的短 lesson，而不会让同一个实现者看到具体 compiler line、失败 shape、`max_error` 和当前 source 后立即修补。因此 CE / WA 不能等价为“机制已经被反驳”：它们常常只是实现失败或 evaluator / framework failure。

### Knowledge compression

`build_knowledge_summary()` 只收集有限数量的 `lesson[:150]`、短 strategy name、分数和通用 recommendations；`build_knowledge_context()` 又进一步截断并只显示最近少量失败。当前 RMSNorm summary 的 recommendations 是泛化的 `float4`、warp/shared reduction、256 thread，而没有 candidate lineage、source diff、two-pass state、x lifetime、failure shape、compiler diagnostics 或重复测量分布。

此外，accepted experience card 记录的 `speedup` 取 primary-shape backward-compatible 字段，而不是 aggregate score：例如 episode 23 card 的 `average_gain` 是 `3.02`，但 aggregate 是 `4.077`；episode 28 card 是 `8.302`，但 aggregate 是 `10.883`。这会让 prompt 内的“what worked”同时丢失机制和混淆指标。

### Existing long-horizon design is not the old V100 campaign

`lab/runtime/agent/long_horizon.py` 与 `lab/runtime/agent/codex_session.py` 的设计允许一次 `session.start()` 后在 while loop 中多次 `_plan()`、`_edit()`、compile、correctness、development benchmark 和 `_gate()`；`CodexAgentSession.run_experiment_turn()` 可在同一 thread 继续调用。这比旧 V100 path 更接近“hypothesis 不等于一次 implementation”。

不过本副本的 `docs/ENTRYPOINT_GUIDE.md` 和 `lab/USER_GUIDE.md` 都将 multi-experiment long-horizon、candidate automatic modification、authoritative ABBA / robustness 标为“未接通”，并把当前完整路径标注为 RTX5060 local。旧 V100 campaign 没有调用 `LongHorizonRunner`。因此不能声称现有实际 V100 campaign 已有该能力。

## 5. Benchmark reliability audit

### Proven code path for the inconsistent result

`lab/runtime/evaluators/remote_v100.py::RemoteV100Evaluator.compile()` 运行：

```text
bash eval.sh <candidate> --shape <shape> --op <operator> --no-profile -o result.json
```

从返回 JSON 读取 `compile`、correctness 和 timing。因此该函数名为 `compile()`，但实际 remote operation 是 compile + correctness + benchmark 的完整 evaluator run。

`evaluate_candidate_multi_shape()` 接着：

1. 用 `primary_shape = shapes[0]` 创建 `ev` 并调用 `ev.compile()`；这已产生一次 primary-shape timing。
2. 进入 `for shape in shapes`，又为每个 shape 新建 `sev`，包括 primary shape，再调用 `sev.compile()`；primary shape 因此被完整执行两次。
3. `result["shapes"][0]` 使用第二个 job (`sev`) 的 measurement；`result["evidence"]` 使用第一个 job (`ev`) 的 `static_evidence()` 和 `ev.benchmark()`。

这直接解释 Episode 28 原始 `result.json`：

- `shapes[0]` (`1,4096`) 是第二个 job 的 `8.98 µs`；
- `evidence.latency_us` 是最初 primary compile/evaluate job 的 `21.8 µs`。

两者不是同一次运行，因而不能放在同一 score 解释中。该问题同样存在于 episode 23 及复测 JSON；只是 v23 两次结果都约 24 µs，视觉上不明显。

### What the available rechecks prove

`episode_28/rechecks/summary.json` 保存五次 aggregate：`8.699, 8.687, 8.700, 8.703, 7.521`。`v23_vs_v28` 交错文件中 v23 各 shape 约 `24.33–24.75 µs`；v28 通常约 `8.98–9.67 µs`，但存在 `32x4096 = 22.57 µs` 和 `8x4096 = 22.16 µs` 的慢点。

因此可以确认：

- v28 优于 v23 的相对改进没有被单次初始 fast sample 完全制造；
- 当前结果仍不满足“单次 geometric mean 即可 promotion”的可靠性要求；
- 双峰是 artifact 事实，原因尚未被代码或 profile 证实。

### What cannot be concluded offline

不能从本地静态代码证明慢模式是 CUDA context initialization、dlopen、Python 固定开销、GPU clock、shared-server contention、remote job 创建，还是其他 stream / evaluator 行为。代码证明的是每个 shape 都创建新 remote directory、上传 source、运行新的 remote `eval.sh` process；而 remote `eval.sh` 本身不在 D 盘，因此 warmup / iteration 是否严格一致、CUDA context 是否计时，必须在后续以其 source 和 per-iteration samples 核验。

`evaluation.json` 声明 warmup `50`、iterations `200`，但 `remote_v100.py` 没有把这些数值作为 CLI 参数传给 `eval.sh`。故 D 盘只能证明它们是声明的 contract，不能证明每个 remote job 实际采用了这些设置。

### Promotion risk

旧 `run_evaluation_for_episode()` 仅要求 compile pass、correctness pass 和 `aggregate_score > incumbent_score`。没有 repeated-run median、ABBA ordering、CV / bimodality detection、minimum sample count 或同 source/hash repetition gate。因此一次 fast-mode lucky run 可以直接改写 incumbent manifest。这是 P0 promotion-policy 缺口。

## 6. Contract / Python import contamination

### Contract findings

Episode 28 的两个 hash 语义不同且未被强制连接：

| artifact | value | 实际来源 / 含义 |
|---|---|---|
| `episode_28/hypothesis.json` | `0a3d176112c36cee` | Agent prompt 中的旧 semantic contract value；D 盘不能在当前实现内追溯其生成器 |
| `episode_28/episode_manifest.json` | `a69b1c9ffaa393a1` | `phase8d.build_episode_manifest()` 对 `json.dumps({"shapes": contract_shapes}, sort_keys=True)` 的 SHA-256 前 16 位 |

因此，D 盘旧 runner 的“正确 hash”是 **`a69b1c9ffaa393a1`，但它只是 evaluation-shape fingerprint，不是完整 ABI / semantic contract hash**。D 盘中 `operators/rms_norm_v100_cuda/metadata.json` 有 `contract_schema`，但 `lab/core/contract.py` 不存在；`remote_v100_campaign.py` 虽尝试 import `lab.core.contract` 来构建 prompt，实际 evaluation path 也未调用 `phase9.validate_candidate_contract()`。`phase9.run_unattended_campaign()` 有 validator 调用，但它不是旧 V100 `main()` / `run_evaluation_for_episode()` 的共同硬门禁，并且该 validator在本副本中依赖缺失模块。

这解释了为什么 hypothesis hash mismatch 能通过 evaluation：当前 hypothesis schema validator 只检查 tags / effects / risk，不检查 `contract_version` 或 `contract_hash`；remote V100 runner 则直接 benchmark。

### Import isolation findings

本轮使用受控系统 Python 前设置 D 盘 `PYTHONPATH` 并检查 `lab.__path__`。它只包含：

```text
D:\Users\38154\Downloads\aka-local-main\aka-local-main\lab
```

`lab.__file__` 是 `None`，因为这是 namespace package；不能用该字段做来源断言。D 盘副本没有 `.venv`，而 `lab.ps1` 却硬编码 `$ROOT\.venv\Scripts\python.exe`，所以现有 launcher 不能提供自包含、可验证的解释器隔离。namespace package 加上缺失的 `lab/core` 还会使来自 C 盘 editable install / `sys.path` 的同名 `lab.*` 子模块有机会被拼接或解析，正是本次污染风险的技术根源。

### Required remediation

1. 将 `lab` 变为普通 package（添加受控 `lab/__init__.py`），并让入口先 assert `Path(lab.__file__).resolve()` 落在 D repo root；对 namespace package 则检查所有 `lab.__path__` entries。
2. 建立一个唯一 launcher，例如 `scripts/run_lab.py`：解析自身父目录为 `PROJECT_ROOT`，删除/拒绝外部 `PYTHONPATH`，将 root 放在 `sys.path[0]`，拒绝已经加载且不在 root 下的 `lab` modules。
3. `lab.ps1` 只调用该 launcher，并在启动前验证解释器与 `lab` path；若 `.venv` 缺失，明确失败，不回退到全局 editable install。
4. 以单一 canonical contract JSON 生成 full SHA-256，内容至少包含 operator、version、entry、ordered arguments/type/role、数学语义、shape set、dtype、tolerance 和 required marker；evaluation fingerprint 应为不同字段 / 不同名称。
5. evaluator 在 upload / compile 前比较 candidate marker、hypothesis hash 和 manifest hash；不一致必须写 `REJECT_CONTRACT`，不得连接远程。

## 7. Atrex / KDA / CAKE attribution

本节只依据 D 盘已有文档和实现，不把本地问题推给上游。

### Evidence available in this repository

- `lab/ARCHITECTURE.md` 明确说 long-horizon design 受只读 survey 的 Atrex Kernel Agent `campaign.py`、`session.py`、`journal.py`、`protocol.py`、`store.py`、`git_episode.py` 启发，并明确“不复制 BI-V150 gateway 或 remote orchestration”。
- `docs/ENTRYPOINT_GUIDE.md` 描述 LongHorizonRunner / supervisor / recovery 的目标结构，同时声明多 experiment loop、evaluator、ABBA、robustness 和 remote sync 未接通。
- `lab/knowledge_sources/import_manifest.json` 记录的是 CUDA / GPU-mode / NVIDIA 文档来源；未发现可审计的 KDA 或 CAKE source note、接口适配层、commit 或执行入口。

### Attribution matrix

| 问题 | 归类 | 证据与理由 |
|---|---|---|
| long-horizon、journal、recovery 的设计目标 | A: 上游设计 tradeoff / inspiration | Atrex survey 被明确记录为设计来源；这不是 bug |
| 旧 V100 一次 Agent = 一次 candidate | C: aka-local 本地 legacy implementation | `remote_v100_campaign.py` 自行创建 ephemeral session 并关闭；不是 Atrex 的必然约束 |
| long-horizon 未接入旧 V100 evaluator | B: 融合不完整 | D 盘同时存在较丰富的 local LongHorizonRunner 和单次 V100 runner，但没有 adapter / entrypoint wiring |
| CE/WA 被压缩成 short lesson | C: aka-local knowledge policy | `_derive_reusable_rule()` 与 summary 明确将失败简化；framework failure 也没有被统一排除 |
| contract hash 漂移与 validator 未接线 | C: aka-local implementation / policy | `phase8d` shape hash、`phase9` validator 和 `remote_v100_campaign` evaluation 未形成同一门禁 |
| result/evidence 取自不同 job | C: aka-local evaluator implementation | `evaluate_candidate_multi_shape()` 中 primary job 与 loop job 的可见代码路径直接造成 |
| external Atrex canonical memory 被导入但只保留摘要 | D: standalone / import legacy | `lab/experiment_archive/remote-*` 是只读导入摘要；它不应被当作本地 V100 campaign 的实时执行能力 |
| KDA Hypothesis → Experiment → Evidence | B: 无法证明完整迁移 | 本副本有同名概念（Hypothesis、Evidence、Diagnostic），但未发现 KDA source note 或可追溯 integration；不能归因给 KDA |
| CAKE diagnostics / verifier / cost model | B: 无法证明完整迁移 | 本副本有 NSYS / NCU diagnostic skeleton 和 Evidence classes，但未找到 CAKE source note、IR、cost model 或闭环 executor；不能声称已迁移 |

结论：Atrex-inspired long-horizon / recovery 的理念本身不应背锅；当前“一次 hypothesis = 一次实现”来自旧 V100 runner 的本地选择。KDA / CAKE 是否完整迁移在 D 盘没有足够 provenance；正确表述是“未证实 / 未接线”，不是“上游没有该能力”。

## 8. Recommended architecture changes

### P0 — correctness and promotion safety

1. **Hypothesis != implementation attempt。** 为一个 hypothesis 建立 stable ID；允许多次 `implementation_attempt`，每次有 parent candidate hash、diff、compile/correctness/benchmark outcome。CE、WA、framework error 分别记录，不能自动终结 hypothesis。
2. **同一 session repair loop。** 对 compile / correctness 失败，将结构化错误（compiler lines、failed shapes、max error、current candidate hash）发送回原 session；只在 Agent 明确放弃 hypothesis、预算耗尽或人工中断时结束。性能诊断亦应在同 session 返回，但 promotion 仍由 deterministic supervisor 决定。
3. **strict contract validation。** 在任何 SSH / upload 之前比较 canonical contract full hash、candidate marker、hypothesis hash、manifest hash；shape/evaluation fingerprint 与 semantic hash 分字段保存。
4. **确定的 D-repo import isolation。** 统一 launcher，普通 package，启动 assert，拒绝 C 盘 / editable-install module path；没有 D `.venv` 时 fail closed。
5. **repeated benchmark gate。** 首次 run 只能标 `PROVISIONAL`。promotion 必须有 predeclared repeated statistics：同一 persistent worker / process、每 shape 多次、raw samples、median、CV / outlier policy、bimodality flag、ABBA 或随机化顺序。任何双峰或 result/evidence job mismatch 均阻止 promotion。

### P1 — search quality and observability

1. 保存 structured profiler feedback：ptxas registers/spills、occupancy、kernel timing、memory traffic 和其证据等级；没有 profile 时写 UNKNOWN，而不是猜测。
2. knowledge record 除 tag 外必须包含 mechanism、data lifetime、loads/stores、shared/register state、candidate/parent hash、failure shape、compiler diagnostic、benchmark distribution、是否为 implementation failure。
3. 建立 candidate lineage graph；同一 hypothesis 的 attempt 共享 hypothesis ID，而非只保留 episode 名称。
4. benchmark harness 输出完整 per-iteration samples、remote process / device metadata、clock / temperature / competing-process snapshot；自动检测 multimodality 与 primary/evidence provenance mismatch。

### P2 — evidence-guided optimizer integration

1. 若要声称 CAKE integration，先引入可审计的 IR、verifier 和 cost-model interface，并把 verifier 结果、适用范围和 cost prediction 写入实验记录。
2. 将 diagnostic 的 suggested actions 映射到明确的 mechanism / experiment design，而不是只拼接 prompt text。
3. 在确认 licensing、source provenance 和 exact API 后，再考虑扩展 Atrex / KDA / CAKE adapters；不要以名字替代可执行闭环。

## 9. Minimal patch plan

| priority | file | function / area | current behavior | proposed behavior | risk | tests needed |
|---|---|---|---|---|---|---|
| P0 | `lab/runtime/evaluators/remote_v100.py` | `evaluate_candidate_multi_shape` | primary `ev.compile()` 完整执行一次，loop 又执行 primary；`shapes` 与 `evidence` 来自不同 job | 拆分真正的 compile-only preflight，或只保留每 shape 一次 run；用 job ID / source hash / run ID 绑定 `shapes`、`evidence` 与 samples | 改变 legacy result schema | fake SSH evaluator test：primary only once；所有 fields 指向同一 run ID |
| P0 | `lab/runtime/evaluators/remote_v100.py` | `RemoteV100Evaluator.compile` / benchmark protocol | `compile()` 名称掩盖完整 eval；warmup/iterations 未从 contract 传递 | 返回显式 `compile_and_measure` result；把 contract 参数传给 remote harness；持久 worker 输出 raw samples | remote eval.sh CLI 兼容性 | contract-to-command unit test；fixture parse test；无网络 mock |
| P0 | `lab/runtime/evaluators/remote_v100_campaign.py` | `run_evaluation_for_episode` | 只比较单次 aggregate 与 incumbent，立即 ACCEPT | 新建 `PROVISIONAL`；调用 repeated/ABBA qualification，检查 median/CV/bimodality/provenance 后才 promotion | 旧历史结果需明确迁移状态 | synthetic fast/slow mixture must block; stable improvement promotes |
| P0 | `lab/runtime/evaluators/phase8d.py` | `build_episode_manifest` | `contract_hash` 实为 shape-list hash | 改名 `evaluation_fingerprint`；调用 canonical contract serializer 写 `semantic_contract_sha256` | manifest consumers 需要迁移 | fixed JSON canonicalization / hash golden test |
| P0 | `lab/runtime/evaluators/remote_v100_campaign.py` 与 `phase9.py` | before `run_evaluation_for_episode` | 旧 main 未调用 validator；hyp schema 不比较 hash | 统一 `validate_episode_contract()`，检查 marker、hypothesis、manifest、metadata contract，失败前不 SSH | missing legacy fields | mismatch test must produce `REJECT_CONTRACT` and assert evaluator not called |
| P0 | `lab/`、`scripts/run_lab.py`、`lab.ps1` | package / launcher | namespace package、`.venv` 不存在、无 import-root assertion | 添加 `lab/__init__.py`；launcher 用 `Path(__file__)` 固定 root，assert all module paths under root，fail closed | 现有 direct script commands | subprocess test with injected C-style `PYTHONPATH`; expected rejection |
| P1 | `lab/runtime/evaluators/remote_v100_campaign.py` | Agent-driven `main` | one `ephemeral=True` thread and one `t.run` | 将 V100 adapter 接到持久 session controller；使 compile/WA structured feedback 进入 `continue_episode` | cost / agent runaway | fake session test with CE→repair→pass; bounded attempt budget |
| P1 | `lab/runtime/agent/long_horizon.py` | optimization failure branches | CE/WA reset to baseline，不向 same Agent 提供 repair turn | 加 attempt-level repair state、source snapshot、failure payload；保持 hypothesis ID，限制修复次数 | source mutation safety | candidate lineage and reset/retry tests |
| P1 | `lab/runtime/evaluators/remote_v100_campaign.py` | `build_knowledge_summary`, `_derive_reusable_rule` | 截断 lesson、混合 primary score / aggregate、错误分类过粗 | 保存 structured mechanism/failure/provenance；framework errors exclude from anti-strategy; include metric scope | prompt length | knowledge serialization / context rendering tests |
| P2 | new `lab/integrations/cake_*` + schemas | verifier / IR / cost model | 无可验证 CAKE execution integration | 先定义 versioned adapter contract，再接入执行 loop | provenance/licensing/API drift | static verifier and mock cost-model tests |

## Appendix: audit boundaries and evidence limits

- 本报告没有重新计算 benchmark，也没有把 9/22 µs 的任一模式归因给特定硬件原因。
- `eval.sh` 不在 D 盘副本，故 warmup/iteration 实际值、context initialization 计时方式与服务器状态只能列为未验证项。
- D 盘内没有 KDA / CAKE 的可审计 source note 或完整 adapter；相关归因已明确使用“未证实”，没有把本地缺口归咎于上游。
- 发现的 C 盘绝对路径仅存在于历史文档示例；本轮未按其执行，也未从 C 盘复制实现。
