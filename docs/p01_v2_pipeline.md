# P01 局部直接取证 Pipeline v2

> 状态：设计评审候选。本文描述当前代码实际执行的流程，不代表老师已经确认这些设计
> 取舍。待确认项与现有实跑证据见 [P01 v2 设计评审清单](p01_v2_design_review.md)。

## 目标

P01 v2 面向已知属于 P01 的单视频题，保留 training-free、纯视觉和单连续局部证据段，
同时把 benchmark 输出覆盖率设为硬约束：有效 MCQ 必须返回一个合法选项，G39 必须返回
非空 best-effort 描述。证据不再是二元拒答门，而是局部决策的纠错输入、一次 bounded
rescue 的触发信号和结果诊断信息。

v2 不包含 P 类路由、训练、字幕/音频、benchmark runtime 或裸视频 DirectPass。

## 主流程

```text
question
  -> choice-blind ObservationCompiler
  -> choice-blind adaptive Locator
  -> at most 2 independent choice-blind CandidateScouts
  -> deterministic single-span selection
  -> label-free HypothesisCompiler
  -> optional one routine local refinement
  -> InitialDecision(local media + EvidencePacket + full options)
  -> deterministic evidence grade / rescue trigger
  -> optional one bounded rescue
       global overview localization only (<=128 frames)
       + at most 2 discriminator-aware local scouts
       + FinalDecision
  -> mandatory legal MCQ label
```

G39 走相同的盲定位、scout、refinement 和可选 rescue，随后 Composer 同时读取冻结的
局部媒体与 EvidencePacket。若 Composer 为空或失败，控制器用 EventFact 声明生成非空
回退文本。

## 不变量

- 首轮 ObservationCompiler、Locator 和 CandidateScout 永远看不到选项。
- rescue 定位和 scout 只看去标签的原子判别声明，不看 A/B/C/D 或 option-to-claim 映射。
- 只有 InitialDecision/FinalDecision 看完整选项；它们只能查看当前连续局部段的媒体。
- 不存在全视频直接答题调用。全局 rescue 的最多 128 帧只允许定位，不允许回答。
- 一次 routine refinement 不消耗 rescue；每题最多一个 rescue episode。
- 相交的 initial/rescue span 仅在不超过 mode 上限时合并；不相交的段只能竞争，不能拼接证据。
- G42 严格排除显式区间外帧。长区间的 chunk 只是同一 canonical span 的传输分块。
- “Cannot be determined” 是普通语义选项；pipeline 失败永远不能自动选择它。
- evidence 的 `strong | partial | weak | none` 等级不抑制答案输出。

## 必答与降级链

中间结构化阶段允许一次纯文本 JSON repair；仍失败时记录 degradation 并继续使用确定性
协议回退。最终 MCQ 决策按以下顺序保证合法输出：

1. 解析 ChoiceDecision JSON；
2. 一次无媒体 JSON repair；
3. 从原始/修复文本提取合法 option ID 或 benchmark label；
4. 按 EvidencePacket 与去标签判别声明做确定性支持分数；
5. 完全并列时按原选项顺序稳定打破平局。

`max_model_calls=20`，非终局调用始终为 Decision/repair 预留 2 次调用。CUDA OOM 会被写入
资源账本，并以安全像素预算重试同一调用一次；不会静默切换 attention backend。

## 默认曝光预算（4090D 24GB）

| 环节 | v2 默认值 |
|---|---:|
| 导航索引 | 2 fps，最长边 768 |
| Locator | 6 tiles/page，单 sheet <= 1,048,576 pixels |
| Global rescue overview | <=128 frames，仅定位 |
| Static scout | 9 frames，最多 16 |
| Dynamic scout / refine | 6 fps / 8 fps，72 / 96 frames |
| OCR search / retain / crop | 8 fps / 16 frames / 6 crops |
| Caption scout / refine | 4 fps / 6 fps，96 / 112 frames |
| Initial/FinalDecision | 事实引用帧优先＋均匀上下文，最多 64 frames；4090D safe 为 48 |
| 自动 span 上限 | static 20s；dynamic 40s；OCR 20s；caption 50s |
| 普通视频像素 | min 65,536；max 262,144；total 12,582,912 |
| 普通独立图片 | min 131,072；max 262,144 |
| detail crop | max 1,048,576 |
| G42 chunk | <=8，自适应覆盖完整显式区间 |

当规则帧序列与 crop 同时出现时，媒体按“一个 frame-list video + 独立 detail images”混合
提交，避免因为加入 crop 而把完整动作序列退化成大量互不关联的图片。

## 输出协议

v2 主字段为：

- `prediction`：MCQ benchmark label 或 G39 文本；
- `decision_source`：`initial | rescued | terminal_fallback | composer | none`；
- `support_level`：非门控证据等级；
- `pipeline_outcome`：正常、rescue 或 degradation 完成状态；
- `decision`、`evidence_grade`、`evidence`、`trace`、`resources`。

`verified_answer`、`forced_prediction`、`prediction_kind` 和 `--force-choice` 保留一个版本的
兼容读取；其中 `verified_answer` 恒为 null，`forced_prediction` 是 `prediction` 的别名。

## 当前工程验收

- P01 单元/FakeModel 测试覆盖盲定位、一次 refinement、OCR consensus/crop、G39、G42、
  bounded rescue、终局解析回退、mixed media 和 OOM safe retry。
- 租卡 25 题的硬门槛是 20/20 MCQ 有合法预测、5/5 G39 有非空预测。
- 同时报告准确率、support 分布、rescue 使用率、OOM、调用数和媒体曝光；25 题结果不用于
  声明泛化提升。
