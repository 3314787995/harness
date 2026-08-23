# 当前实现与实验状态

状态日期：2026-08-23。

这份文档是“现在究竟做到哪了”的交接快照。机器可读的脱敏摘要位于
[results/evidence30_summary.json](../results/evidence30_summary.json)；本地完整 trace 位于
被 Git 忽略的 `runs/`。

## 工程状态

- 四条推理路径可从统一 CLI 选择：direct、tools、coarse-to-fine、active-tree；
- Video-MME 指定题运行、JSON/JSONL trace 和 active-tree HTML replay 已接通；
- Evidence30 v0.2.0 有 30 条冻结 AI-assisted internal reference：18 dev、12 locked；
- dev/locked 三路评测、冻结、断点续跑、聚合和密封流程已实现；
- 统一答题头 Oracle 消融与 dev-only core-density 诊断已实现；
- 79 个控制逻辑测试通过；
- Ruff 与 compileall 通过；
- GitHub Actions 只运行轻量控制逻辑，不下载模型、数据或运行 GPU 评测。

## Evidence30 dev v6

- 协议：`evidence30-relaxed-debug/1.0`
- 运行签名：`9eff46c5ac1ea0962fe8621ba495592bcd42b1b8d3a2c3ae5f335c104bd8c93f`

54/54 项完成，engineering pass 为 true。

| 策略 | Accuracy | Relaxed grounded | Mean slot coverage | Mean model calls |
|---|---:|---:|---:|---:|
| direct | 6/18（33.3%） | 61.1% | 70.4% | 1.0 |
| coarse-to-fine | 5/18（27.8%） | 50.0% | 62.5% | 3.28 |
| active-tree | 9/18（50.0%） | 44.4% | 62.0% | 10.94 |

active-tree 的 verified rate 只有 2/18（11.1%）。因此 9/18 的答案准确率不能解释为
“9 条都形成了可验证证据链”。它在这组调试样本上答对更多，但代价是更高模型调用和
上下文开销，且大多数终止原因仍是缺槽、修复耗尽或预算耗尽。

## Evidence30 locked v1

- 协议：`evidence30-relaxed-debug/1.0`
- 运行签名：`20790ae17d28d3dece407354cb4077bceadc859f9263cb7a6919f36e9243e9c2`
- 冻结 ID：`c9dbacd585332ef0d516c2b2c9aa0895442309cf4eda07479ad9e0d14adde44f`

36/36 项有结果，但 engineering pass 为 false：coarse-to-fine 的 12 条中有 1 条发生
degraded direct fallback。没有 fatal、trace invalid 或预算越界。

| 策略 | Accuracy | Relaxed grounded | Verified | Degraded |
|---|---:|---:|---:|---:|
| direct | 6/12（50.0%） | 50.0% | 不适用 | 0/12 |
| coarse-to-fine | 6/12（50.0%） | 41.7% | 不适用 | 1/12 |
| active-tree | 6/12（50.0%） | 41.7% | 1/12（8.3%） | 0/12 |

active-tree 相对 direct 是 1 win / 1 loss / 10 ties；相对 coarse-to-fine 也是
1 win / 1 loss / 10 ties。locked v1 不支持“active-tree 优于基线”的结论，并且该版本已被
执行；继续据此调参后不能再把它当作未见确认集。

## 统一答题头消融 v2

- 协议：`evidence30-oracle-unified-head/1.1`
- 运行签名：`9aab79d4f207170f0d9c9aa929def01c01e6ad7021329677e2cdae338ff679bc`

只使用 18 条 dev，不读取或重跑 locked。54/54 模型调用完成，engineering pass 为 true。

| Packet | Accuracy | Relaxed grounded | Slot coverage | Mean frames |
|---|---:|---:|---:|---:|
| direct replay | 6/18（33.3%） | 61.1% | 70.4% | 16.0 |
| active-tree replay | 6/18（33.3%） | 38.9% | 60.6% | 13.7 |
| Oracle context | 9/18（50.0%） | 100.0% | 100.0% | 14.3 |

这个结果支持两个同时存在的瓶颈：

1. 当前 active selector 在匹配预算的 packet 上没有优于 uniform direct，搜索/选择仍弱；
2. 即使 Oracle 时间上下文覆盖所有 required slot，2B 一次答题头仍只对 9/18，说明宽时间段
   内的关键帧采样、视觉识别和组合推理也有限。

完整解释见 [统一答题头消融报告](evidence30_ablation_v2_report.md)。

## core-density v2

- 协议：`evidence30-oracle-core-density/1.1`
- 运行签名：`53b619f428c04c78653e3bc26e89cb72dae439ab1ae649fe954ccec55886f5eff`

只选择 Oracle context 仍答错的 9 条 dev。loader 明确不打开 locked。18/18 项完成，
engineering pass 为 true，所有 packet 的 core item recall 为 100%。

| Variant | Accuracy on 9 failures | Mean selected frames | Mean input tokens |
|---|---:|---:|---:|
| core_16 | 1/9（11.1%） | 14.6 | 1,407 |
| core_32 | 2/9（22.2%） | 24.6 | 1,505 |

core_32 相对 core_16 为 2 wins / 1 loss / 6 ties，说明更密集的时间观察有真实但不稳定的作用。
两路 total-pixel 上限同为 2,097,152；core_32 把单帧上限从 131,072 降到 65,536，因此是
“更多低分辨率时间帧”的联合干预。大部分错误仍未恢复，后续应优先检查 OCR/细节分辨率、
跨槽顺序合成、关键瞬间定位和 2B 表征能力。

早期 policy 1.0 v1 保持了 131,072 的 core_32 单帧上限，实际 input tokens 明显高于
core_16，不满足 matched-budget 设计；它只保留作预算语义排错，不进入正式结论。完整逐题
解释见 [core-frame 密度诊断报告](evidence30_core_dense_v2_report.md)。

## 已证实能工作的机制

- 同一模型适配器可支撑四条路径；
- 视频帧磁盘缓存、字幕区间对齐和 contact sheet 可复用；
- coarse-to-fine 能执行粗路由、窗口排序、细化、投票和降级；
- active-tree 能构建场景树、编译证据契约、执行 option-hidden breadth、合法动作、
  原子落账、双验证和显式 unverified；
- Evidence30 能验证 Schema/哈希/split，标准化不同策略 exposure 并断点续跑；
- 统一答题头诊断能隔离 packet 来源；
- core-density loader 和 prompt 防泄漏测试能保证只读 dev、不给模型 gold 文本。

## 仍未解决

- active-tree 的 verified rate 很低，严格证据闭环尚未稳定；
- 当前搜索在匹配预算下没有优于 direct replay；
- Oracle 100% 时间覆盖不等于关键像素可见或被 2B 正确理解；
- sequence、OCR、宽 core interval 和长视频组合仍是主要风险；
- locked v1 工程门未通过，且已消费；
- Evidence30 没有独立人工双人复核；
- GPU 端到端 smoke 不在 CI；
- 大控制器尚未拆分。

## 不得对外宣称

- “复现了 VideoChat-A1 或其他论文训练/指标”；
- “Evidence30 是人工 gold”；
- “active-tree 已在 held-out 上显著优于 direct/coarse-to-fine”；
- “verified 与最终正确等价”；
- “Oracle context 是语义 Oracle 或准确率上界”；
- “core_32 证明更多帧完全无效”——这里只验证 9 条 dev、固定总像素和一种模型。

## 建议的下一步

1. 为 9 条 Oracle 失败做按题型的像素级检查，区分关键帧遗漏、OCR 不清、视觉误识别和组合错误；
2. 在 dev 上改进 evidence selector，以 matched-budget active replay 超过 direct replay 为首个门槛；
3. 保持统一答题头和固定像素预算，一次只改变一个搜索变量；
4. 新建独立 held-out split 后再做效果确认；
5. 在行为冻结后逐步拆分 active-tree 和 evaluation 大文件；
6. 论文级使用前升级为真实人工复核标注。
