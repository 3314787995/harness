# VideoMME-Evidence30 AI-assisted 证据参考规范

- 规范 ID：`videomme-evidence30`
- 版本：`0.2.0`
- 规范状态：`0.2.0` 已冻结
- 数据状态：v0.2 已完成 30 条 AI-assisted internal reference；不是独立人工 gold
- 适用范围：Qwen3-VL 长视频 coarse-to-fine / active-tree 调试与机制评测
- 机器格式：`schemas/videomme_evidence_annotation.schema.json`

本文定义 30 道 Video-MME 调试题的证据参考格式与 v0.2 已锁定集合。它只辅助判断答案所需证据是否被系统实际观察和正确落账，不替代 Video-MME 官方准确率评价，也不能被称为独立人工 gold。v0.2 保留单一标注者字段与事实性 Codex provenance；工程跑通采用 [`evidence30-relaxed-debug/1.0`](evidence30_debug_protocol.md) 的宽松时间暴露评分，本文中的 strict/core/fact/relation 规则继续作为诊断信息，而非工程硬门槛。

文中的“必须”“不得”为硬约束；“应”“建议”为默认规则，偏离时必须记录理由。

## 1. 标注目标与边界

标注要回答三个彼此独立的问题：

1. 原始视频、官方字幕或画面文字中，是否存在足以区分正确答案的证据？
2. 哪些最小证据项共同构成一套充分证据？
3. 推理流水线是否真的把这些像素、帧、字幕 cue 送入模型，并将其转成正确原子事实？

允许的证据来源只有：

- `visual`：视频画面；
- `subtitle`：Video-MME 提供的字幕；
- `ocr`：画面内可读文字；
- 上述来源的组合。

以下内容不得作为充分证据：

- 仅存在于音轨、且字幕未包含的信息；
- 必须联网或依赖视频外专门知识才能确定的结论；
- 视频标题、文件名、数据集标签或模型既有记忆；
- 模型生成的 query-agnostic caption 或自由文本总结；
- 只与题目相关、但不能区分选项的片段。

## 2. 30 题抽样与划分

### 2.1 硬约束

- 30 题必须来自 30 个不同视频。
- 每题的视频、题目、选项和官方答案必须可读取。
- 每题必须能由允许模态形成至少一套充分证据。
- 发现歧义、错误答案、媒体损坏或不支持模态时，原则上移出主集合并用新视频题替换；若实施版明确保留该候选，必须在题目备注和 manifest 中记录风险，不得静默改写官方答案或证据结论。
- 最终划分为 18 题 `dev` 和 12 题 `locked`，视频级完全隔离。
- 不得依据 v0.1、v0.2 或任一模型是否答对来选题。

### 2.2 分布目标

以下是抽样目标，不满足时必须在集合 manifest 中解释：

- `short / medium / long` 各约 10 题；
- 五种主要证据拓扑各约 6 题；
- 至少 10 题可由纯视觉证据解决；
- 至少 10 题实质需要字幕；
- 至少 5 题需要视觉与字幕联合；
- 域和 task type 不应被单一类别主导。

主要拓扑必须唯一；可使用多个 `secondary_topologies` 描述交叉性质。

### 2.3 Smoke 子集

三题 smoke 从 `dev` 中选择，分别以 `local`、`sequence`、`multi_set` 为主要拓扑，并尽量覆盖纯视觉、字幕、混合三种模态。Smoke 题应有清晰、低歧义的充分证据，避免把模型常识能力误当作流水线连通性。

### 2.4 Locked 管理

- `locked` 题在冻结前完成全部证据标注和单标注者检查；v0.2 不要求 A/B 双人标注、第三人裁决或独立复核包。
- `review.mode` 必须为 `single_annotator`；`annotator_a` 记录中性标注者 ID 和完成时间，`annotator_b`、`adjudicator` 为空。该状态不等同于人工双人 gold。
- 调参者在正式比较前不得查看 locked 题的逐题 gold interval、证据事实或模型 trace。
- 评测器可返回聚合指标；需要调试逐题失败时，必须先宣布该 locked 版本失效并重新划分。
- 锁定文件必须记录内容 SHA-256；任何改动均升级版本并重跑全部方法。

## 3. 题目有效性

`validity.status` 取值：

- `pending`：尚未裁定；
- `valid`：允许模态中存在可复核的充分证据，且正确选项唯一；
- `ambiguous`：两项以上均可被合理支持，或问题语义不唯一；
- `incorrect_official_answer`：官方答案与可观察证据冲突；
- `insufficient_observable_evidence`：视频内容不足以区分答案；
- `unsupported_modality`：决定性信息只存在于当前未支持模态；
- `broken_media`：视频、字幕或时间轴损坏。

只有 `valid` 记录可以进入 `dev` 或 `locked` 主集合。不得静默改写官方答案；错误题必须保留官方答案、写明理由，然后排除并替换。

## 4. 选项规范化

每个选项同时保存：

- `option_id`：稳定内部 ID，固定为 `O1`、`O2`、`O3`、`O4`；
- `benchmark_label`：Video-MME 的 `A`、`B`、`C`、`D`；
- `text`：去除或保留前缀均可，但全集合必须一致。

证据标注只引用 `option_id`。最终评分时再由映射转换为 benchmark label，防止答案内容正确但字母映射错误。

## 5. 证据拓扑

| primary_topology | 定义 | 最低充分性要求 |
|---|---|---|
| `local` | 一个连续局部事件、对象、人物或状态 | 至少一个直接支持或排他性反驳证据项 |
| `sequence` | 两个以上事件的先后、因果、转变或持续关系 | 所有事件锚点及显式关系均命中 |
| `multi_set` | 多个时间上不连续的事实共同决定答案 | 同一充分集合中的所有必需成员均命中 |
| `global` | 主旨、频率、重复次数、全程属性或不存在性判断 | 满足明确的全局覆盖契约，不允许用单一负样本代替 |
| `exclusion` | 需要排除一个或多个强竞争选项 | 直接正证据，或规定的竞争选项反驳集合全部完成 |

拓扑按“最小充分证据的结构”判定，而不是按 Video-MME 原始 task type 判定。

## 6. 证据契约

### 6.1 Evidence slot

`evidence_slots` 描述回答前必须解决的可观察问题。每个槽位包含：

- 稳定 `slot_id`；
- 选项中立的 `description`；
- `required`；
- 可选的时间关系或计数约束。

示例槽位应写成“确认人物明确讨论由谁担任伴郎”，而不是“证明 O2 正确”。

### 6.2 充分证据集合

- 一道题允许存在多套 `sufficient_evidence_sets`。
- 集合之间为 OR：命中任意一套即可。
- 集合内部 `required=true` 的证据项为 AND：缺少任一项即不充分。
- `required=false` 只作为背景，不进入严格 Recall。
- 每套集合必须能让不了解官方答案的复核者唯一确定正确选项。

### 6.3 原子事实

每个 `evidence_item.atomic_fact` 必须：

- 直接来自指定时间范围与模态；
- 尽量只表达一个可核验事实；
- 不包含隐藏推理过程；
- 不以 benchmark label 代替内容；
- 明确列出它支持或反驳的 `option_id`。

“人物说出 best man”是原子事实；“因此 O2 正确”不是。

### 6.4 关系

跨证据关系存放在 evidence set 的 `relations` 中，可取：

- `before`
- `after`
- `overlaps`
- `causes`
- `changes_to`
- `same_event`
- `different_occurrence`

关系必须通过 `source_evidence_ids` 引用至少两个证据项。

## 7. 时间区间和观察要求

### 7.1 时间表示

- 时间单位统一为秒，保留三位小数。
- `core_interval` 是最小充分证据范围。
- `context_interval` 是方便理解的上下文范围，必须覆盖 core。
- 局部题若 core 超过视频总长的 10%，必须在 `annotation_notes` 解释。
- 区间边界以可观察事件或字幕 cue 为准，不以模型采样帧为准。

### 7.2 Observation requirement

| requirement | 命中条件 |
|---|---|
| `point_frame` | 至少一张实际送入模型的帧位于 core 内并呈现目标事实 |
| `ordered_frames` | 实际输入包含标注要求的多个有序锚点，顺序可恢复 |
| `short_clip` | 送入模型的连续或准连续观察覆盖指定动态过程 |
| `subtitle_cues` | 所有必需 cue ID 或等价完整文本实际进入模型上下文 |
| `readable_ocr` | core 内对应画面以足够分辨率进入 detail/OCR 观察，文字可读 |
| `representative_coverage` | 达到人工规定的代表性分段覆盖条件 |
| `exhaustive_coverage` | 所有指定范围均被观察，适用于严格全局或不存在性判断 |

仅选择一个包含 core 的粗节点不算命中；要求的像素、帧或字幕必须真正进入模型输入。

### 7.3 模态细则

- `visual`：记录关键帧时间戳；动态事实不得只用一张静态帧证明。
- `subtitle`：优先记录 cue ID、起止时间和必要文本，不用任意大段字幕替代。
- `ocr`：记录文字所在帧和目标文本；若普通分辨率不可读，必须要求 `detail_ocr`。
- `mixed` 证据拆为多个 item，通过同一 evidence set 连接，不把画面和字幕揉成不可审计的一条总结。

### 7.4 全局覆盖契约

`global` 题必须定义 `global_coverage_contract`：

- 覆盖对象和范围；
- 必须观察的时间段；
- 可允许的最大未观察间隙；
- 是代表性覆盖还是穷尽覆盖；
- 若为不存在性判断，必须说明可能反例会出现在哪里，以及为何当前覆盖足以排除。

## 8. Hard negative

每题应标注至少一个高价值 hard negative；确实不存在时写明理由。Hard negative 是与题目或某个错误选项高度相关、但不足以完成证据契约的片段。

每条包含：

- 时间范围；
- 模态；
- 它为何看起来相关；
- 它为何不充分；
- 它容易诱导的 `option_id`。

Hard negative 不进入充分证据 Recall，但用于统计错误分支访问率和错误落账率。

## 9. 标注流程

### 9.1 机械预处理

工具可以提前生成：

- 视频时长和哈希；
- 场景边界候选；
- StoryBoard；
- 字幕 cue 时间轴；
- OCR 候选时间点。

工具不得根据模型答案自动生成 gold 原子事实、充分性判断或正确选项解释。

### 9.2 单标注者证据标注

1. 主标注者查看问题、选项、完整视频和字幕，完成有效性、主要拓扑、证据槽、充分集合、区间和 hard negatives。
2. 标注者不得根据模型答案自动生成 gold 原子事实或正确选项解释；如果工作区已有 trace，必须作为风险事实记录，不得据此补造证据。
3. `review.mode=single_annotator` 时，完成时间、内容哈希和锁定备注必须填写；不填写虚构的 B 或 adjudicator。
4. 正式集合仍必须通过 Schema、视频级隔离、证据引用和哈希检查。

### 9.3 单人一致性检查

至少检查：

- 有效性与正确选项；
- primary topology；
- required modalities；
- 原子事实语义；
- 是否存在至少一套语义等价的充分证据；
- 局部区间是否具有共同事件锚点；
- sequence / global 契约是否一致。

区间 IoU 可作为局部证据诊断阈值，默认 `0.5`，但字幕 cue、瞬时事件和全局契约以语义锚点为准，不机械依赖 IoU。发现自相矛盾时修订证据或移入 excluded，并在变更日志记录原因；v0.2 不伪造第三人裁决。

## 10. 文件组织与版本控制

建议目录：

```text
annotations/videomme_evidence30/
  manifest.json
  dev.jsonl
  locked.jsonl
  excluded.jsonl
  artifacts/<video_id>/...
```

一行 JSONL 对应一题，并通过机器 Schema 校验。`manifest.json` 记录：

- 规范版本；
- 数据集源文件哈希；
- 抽样规则与偏差说明；
- 18/12 的 question_id 和 video_id；
- 标注文件哈希；
- 锁定日期；
- 变更日志。

禁止在绝对路径中编码标注语义；源文件位置由运行配置提供，标注只存稳定 ID 和内容哈希。

## 11. 评价指标

### 11.1 Evidence item exposure

证据项 exposure 命中必须同时满足：

1. 观察输入与 gold core/锚点匹配；
2. 模态匹配；
3. observation requirement 满足；
4. 所需原始帧、像素或 cue 确实被送入模型。

### 11.2 Strict Exposure Set Recall

对问题 `q`：

```text
Hit_exposure(q) = 1
```

当且仅当至少一套 gold evidence set 的全部 required items 均 exposure 命中。集合中只命中部分项目计 0。

```text
Strict Exposure Set Recall = mean_q Hit_exposure(q)
```

这是判断搜索机制是否找到证据的主指标。

### 11.3 Strict Grounded Set Recall

在 exposure 命中之外，Observer 还必须生成与 gold `atomic_fact` 语义一致、时间和模态引用正确的证据账本项。该指标用于区分“看到了但没理解”和“根本没看到”。

### 11.4 辅助指标

必须同时保存但不替代主指标：

- 最终答案准确率；
- 单项 evidence recall；
- hard-negative visit / grounding rate；
- verified / unverified 比例；
- 协议合法率和格式修复次数；
- 唯一时间戳、视觉 token、文本 token、模型调用次数和耗时；
- zoom-out、shift、backtrack 次数。

## 12. Evidence30 调试与冻结门槛

先在 18 题 dev 上比较 `direct`、`coarse_to_fine` 与 `active_tree`，共 54 次推理。允许反复查看逐题结果、证据分数和回放。冻结只要求三路 54/54 正常完成、选项合法、trace/停止原因/资源账本完整，并且无 fatal、degraded fallback 或预算越界。准确率、verified rate 与 relaxed grounding 只报告，不设硬阈值。

达标后冻结源码、prompt、完整配置、评分规则、标注、模型权重和环境，再在 12 题 locked 上一次性完成三路 36 次推理。首次只查看聚合结果；逐题结果与 trace 密封保存。任何解封或基于 locked 继续调参都会消费该 locked 版本，后续效果结论必须另建 held-out 集。完整流程见 [`docs/evidence30_debug_protocol.md`](evidence30_debug_protocol.md)。

## 13. v0.2 锁定验收检查表

- [x] 30 个不同 `video_id`
- [x] 18 dev / 12 locked，无视频泄漏
- [x] 所有主集合题 `validity.status=valid`
- [x] 每题至少一套充分证据集合
- [x] 每个 required item 都有 core、模态、观察要求和原子事实
- [x] sequence / multi-set / global 契约完整
- [x] hard negative 已标注或解释缺失原因
- [x] 选项只通过 canonical `option_id` 引用
- [x] `review.mode=single_annotator` 且主标注者、完成时间已记录
- [x] 未伪造 B 标注或 adjudicator；已有 provenance 事实未被删除
- [x] JSONL 通过 Schema 校验
- [x] manifest 和内容哈希已生成
