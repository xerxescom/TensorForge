# TensorForge 功能/性能优化检查（2026-04-17）

> 目标：先给出可落地的优化点，不直接改业务逻辑，方便你按优先级决定后续改动。

## 高优先级（建议先做）

1. **LLM 的 TTFT 采样方式会引入额外 CPU 开销和潜在阻塞**
   - 现状：`_single_run` 用 `proc.stdout.read(1)` 按字符轮询首 token，频繁系统调用，且对长输出有额外开销。
   - 影响：在低延迟模型上会放大测量噪声，TTFT 稳定性下降。
   - 建议：
     - 改为按行或按块读取（如 `readline()` 或固定 chunk），并用单独时间戳记录首个非空输出。
     - 若 Ollama 提供结构化流式输出，可优先解析事件时间而不是字符轮询。

2. **并发压测的基线阶段 + 并发阶段会重复初始化模型，耗时较高**
   - 现状：`ConcurrentStressTest.run()` 先 `_run_baselines()`，再并发跑任务；而 diffusion 子进程每次都 `from_pretrained(...).to("cuda")`。
   - 影响：并发测试总时长被“加载模型”放大，不完全反映 steady-state 吞吐。
   - 建议：
     - 增加“冷启动模式 / 稳态模式”开关：稳态模式在子进程内先 warmup 后计时。
     - 并发阶段与基线阶段都记录 `model_load_s` 与 `inference_only_s`，便于分离分析。

3. **模型缓存大小控制的目录扫描是 O(N 文件数)，大缓存场景会慢**
   - 现状：`_enforce_cache_size_limit()` 会对每个模型目录 `rglob` 统计体积，且可能重复计算。
   - 影响：模型多、文件碎片多时，下载前后的 housekeeping 开销明显。
   - 建议：
     - 维护轻量 metadata（目录大小、上次访问时间），下载后增量更新。
     - 清理时先按 metadata 排序，必要时再 fallback 实时扫描校正。

## 中优先级

4. **GPU 采样统计每个维度都单独建 list + sort，样本大时内存和 CPU 有冗余**
   - 现状：`summarize()` 对 utilization/power/temp/mem/clock 分别构建列表并排序。
   - 建议：
     - 若只需要 mean/max/min/p95，可考虑一次遍历维护 mean/max/min；p95 用 `numpy.percentile` 或近似分位算法（例如 t-digest）。
     - 对超长运行可按窗口聚合，避免最终一次性处理全量样本。

5. **配置覆盖（--set）对 list/dict 类型支持弱，容易把值当字符串**
   - 现状：`apply_overrides()` 按“当前值类型”做简单转换，list 等复杂类型无法从 CLI 可靠覆盖。
   - 建议：
     - 支持 `json:` 前缀（如 `--set llm_prompts=json:["a","b"]`）或提供 `--set-file`。
     - 对关键字段增加类型校验和错误提示，避免 silent wrong config。

6. **CV benchmark 暴露了 `batch_size` 参数，但 worker 未实际使用**
   - 现状：`CVBenchmark` 保存了 `batch_size`，但 `_build_script()` 中每次只推理单帧。
   - 影响：参数语义与实际行为不一致，影响报告可解释性。
   - 建议：
     - 实现真正批量输入（stack N 帧）并输出 batch 吞吐；或删除该参数避免误导。

## 功能完善（可选）

7. **ASR 的 WER 计算方式是近似值，不是标准词错率**
   - 现状：使用 `SequenceMatcher` 匹配块估算误差。
   - 建议：
     - 若你关注评测可比性，换成标准 Levenshtein WER（S/D/I）实现，并保留当前值作为 `wer_approx` 兼容。

8. **错误可观测性可再增强**
   - 现状：部分子进程异常只返回 `error`，缺少结构化分类。
   - 建议：
     - 统一输出 `error_type / error_stage / short_trace`，并在 final_report 做聚合计数（下载失败、OOM、依赖缺失、超时）。

## 推荐执行顺序（ROI）

1. LLM TTFT 采样改造（准确性收益最高）
2. 并发压测拆分“加载时间 vs 推理时间”
3. 缓存限额逻辑元数据化
4. CV `batch_size` 语义修正
5. CLI 覆盖复杂类型能力

---

如果你愿意，我下一步可以按“低风险、可快速回归”的方式先提交一个 **Phase-1**：
- 只改观测和指标层（不改模型推理结果），优先让报告更可信、更可解释。
