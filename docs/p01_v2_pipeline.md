# P01 局部直接取证 Pipeline v2

> 状态：设计评审候选。本文描述当前代码实际执行的流程，不代表老师已经确认这些设计
> 取舍。待确认项与现有实跑证据见 [P01 v2 设计评审清单](p01_v2_design_review.md)。

## 目标

P01 v2 面向已知属于 P01 的单视频题，保留 training-free、纯视觉和单连续局部证据段，
同时把 benchmark 输出覆盖率设为硬约束：有效 MCQ 必须返回一个合法选项，G39 必须返回
非空 best-effort 描述。证据不再是二元拒答门，而是局部决策的纠错输入、一次 bounded
rescue 的触发信号和结果诊断信息。

v2 不包含 P 类路由、训练、字幕/音频、benchmark runtime 或裸视频 DirectPass。

本评审版面向 [18 类原生题型的设计范围](../qwen3vl_agent/p01/README.md#18-类设计范围)，
不是逐类适配或验证完成的声明。Video-MME · Action Recognition 已从范围中移除，
内部 `dynamic_action` 模式保留；Vision-Guided Audio Description 的音频回答部分尚未实现。
其余类别也须逐题满足单连续局部证据边界，不能仅按类别名称认定整类可用。

## 版本与运行边界

2026-09-04 发布整理沿用“当前 v2 双卡版重测”的核心流程与媒体预算，来源为本地快照
`3ad03b9`。六个核心流程文件与 2026-09-02 服务器快照一致，不合入 v3 或新的推理策略。
旧评审提交 `27967aa` 增加的统一 64/48 帧决策截断已移除；90–96 帧调用仍由原取帧逻辑
按需产生，而不是每次固定取满。

普通运行使用 `configs/p01_8b.yaml`：Qwen3-VL-8B、`device: auto`、`bfloat16`、
`flash_attention_2`。不要求固定显卡型号、数量或逐卡显存额度；运行者自行准备兼容依赖和资源。
通用模型分片与输入设备处理保留，但不带入双卡七题专用执行脚本。没有新增音频接口或题型路由。

若沿用旧评审版的自定义 YAML，请删除已退役的 `p01.decision_max_frames` 字段；当前配置
按原服务器的分模式上限校验，不接受该额外字段。请勿在旧结果目录混用不同版本或配置。

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

## 原 v2 默认曝光预算

下表是流程预算，不是显卡规格要求或显存安全保证。普通配置的 `p01` 参数与重测快照一致。

| 环节 | v2 默认值 |
|---|---:|
| 导航索引 | 2 fps，最长边 768 |
| Locator | 6 tiles/page，单 sheet <= 1,048,576 pixels |
| Global rescue overview | <=128 frames，仅定位 |
| Static scout | 9 frames，最多 16 |
| Dynamic scout / refine | 6 fps / 8 fps，72 / 96 frames |
| OCR search / retain / crop | 8 fps / 16 frames / 6 crops |
| Caption scout / refine | 4 fps / 6 fps，96 / 112 frames |
| Initial/FinalDecision | 事实引用帧优先＋均匀上下文；分模式上限：static 16、dynamic 96、OCR 22、caption 112，无额外统一截断 |
| 自动 span 上限 | static 20s；dynamic 40s；OCR 20s；caption 50s |
| 普通视频像素 | min 65,536；max 262,144；total 12,582,912 |
| 普通独立图片 | min 131,072；max 262,144 |
| detail crop | max 1,048,576 |
| 显式区间短 / 中 / 长档 | 6 / 3 / 1.5 fps，72 / 90 / 112 frames |
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

CLI 的 `--trace-output` 写入完整 metadata，结果位于 `p01` 字段；`trace` 保留各阶段的控制流程，
`resources.calls[]` 保留实际调用的 `prompt`、`raw_response` 及帧信息。
模型调用 metadata 另外记录视觉 token 数、实际设备映射及
显存快照；这些是诊断信息，不是新的证据门控。保存方法见
[导览中的回答与证据说明](../qwen3vl_agent/p01/README.md#保存回答与检查证据)。

## 检查状态与历史验收边界

- 本次发布仅做源码一致性、配置、语法、文档与提交范围的静态检查，不主动运行单测、模型或显存
  压力预检。原有 GitHub CI 保持不变；静态检查不等于已在导师设备上运行通过。
- 仓库保留 P01 单元/FakeModel 测试，涉及盲定位、一次 refinement、OCR consensus/crop、G39、
  G42、bounded rescue、终局解析回退、mixed media 和 OOM safe retry；本次未重新执行这些测试。
- 历史租卡 25 题的输出门槛是 20/20 MCQ 有合法预测、5/5 G39 有非空预测，不代表 18 类覆盖。
- 同时报告准确率、support 分布、rescue 使用率、OOM、调用数和媒体曝光；25 题结果不用于
  声明泛化提升。
