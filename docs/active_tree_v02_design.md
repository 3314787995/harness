# Active Evidence Tree v0.2 设计与验证说明

- 状态：debug implementation
- 基线模型：Qwen3-VL-2B-Instruct
- 目标：用可审计、可消融的主动观察流程验证长视频 coarse-to-fine 机制
- 非目标：复现论文训练过程、宣称论文指标、替代 Video-MME 官方准确率

## 1. 第一性原理

长视频问答不是“把更多帧塞进上下文”，而是四个彼此独立的问题：

1. **路由**：哪些时间段可能含有答案所需信息？
2. **观测**：应以多大时间尺度、哪种模态重新查看？
3. **充分性**：当前证据是否覆盖问题要求的全部事实和关系？
4. **决策**：只有证据充分时，哪个选项被唯一支持？

因此 v0.2 强制以下不变量：

- 粗览只负责路由，不能直接完成 evidence slot。
- 模型写出的置信度不是证据充分性。
- 观察到相关对象不等于观察到目标动作；静态状态不能证明时序转变。
- 选项在根层粗览之后才揭示，避免最早阶段按答案措辞复制事实。
- 每条有效视觉证据必须引用实际输入帧；字幕证据必须回落到 SRT 原文。
- 缺少 required slot 时不调用答案验证器猜测，预算耗尽时明确返回 unverified。
- gold label 和 question ID 不参与路由、证据匹配、时间合成或停止判断。

## 2. 与方法论文的关系

| 来源 | 采用的原则 | 本实现的差异 |
| --- | --- | --- |
| [VideoChat-A1](https://arxiv.org/abs/2506.06097) | 1 fps 粗览；Shot Selection、Shot Partition、Shot Reflection；不确定时继续细看 | 不使用论文的 fine-tuned LongCLIP；不以单模型自报 confidence 作为停止条件；增加证据契约、严格账本和 abstain |
| [VideoTree](https://arxiv.org/abs/2405.19209) | coarse-to-fine 层次表示、相关区域分配更多观测预算 | 树在问题到来前由视觉变化和字幕间隔构造，避免选项影响分段；查询只控制遍历 |
| [VideoAgent](https://arxiv.org/abs/2403.11481) | 结构化记忆和工具化检索 | 不预生成整段 caption/object 数据库；只持久化帧缓存、场景索引和本题原子证据 |
| [VideoAgent2](https://arxiv.org/abs/2504.04471) | plan-adjust、工具噪声意识、信息不足时调整检索 | 用 missing slots、停滞检测和 repair directive 替代启发式自报不确定性 |
| [LVAgent](https://arxiv.org/abs/2503.10200) | perception/action/reflection、多视角审查 | 本地 debug 仍使用同一个 2B 模型，但以 completeness verifier 和 shuffled skeptic 隔离角色；不声称多模型独立性 |
| [LensWalk](https://arxiv.org/abs/2603.24558) | reason-plan-observe；宽扫描、局部聚焦、跨片段验证；预算自适应 | 实现为有类型且受 Python 合法性检查的动作；控制器而非模型拥有预算和停止权 |

VideoChat-A1 论文采用首轮 6 个 shot、后续候选二分、最多 3 轮，并用 0–3 的模型置信度决定是否继续。v0.2 保留“逐轮选择并细分”的核心，但不复刻这些超参数或置信度停止逻辑。

## 3. 系统结构

### 3.1 离线、问题无关层

1. 视频以 1 fps 解码并缓存 JPEG。
2. 相邻帧视觉变化与字幕 gap 共同产生候选边界。
3. 边界被组织为有父子关系的时间树，并持久化 scene index。
4. 每个节点保存时间范围、层级、子节点、变化峰值和边界分数。

这层不读取问题和选项，同一视频可跨题复用。

### 3.2 在线主动观察层

1. **Question-only compiler**：只看问题，输出 `local / sequence / multi_set / global / exclusion` 拓扑和 required slots。
2. **Option-hidden breadth**：查看所有根子节点 storyboard；结果标记为 `breadth`，只能生成观察提案。
3. **Option reveal + discriminator**：规范化 O1–O4，定义每个选项可观察的 support/refute test，不直接作答。
4. **Planner slate**：模型给出最多三个候选动作；Python 控制器检查节点、模态、重复、预算和树可见性。
5. **Observer**：只查看动作允许的帧/字幕，输出原子事实。
6. **Evidence ledger**：去重、绑定 slot、记录时间、模态、帧 ID、字幕引用和 observation mode。
7. **Verifier pair**：完整性验证器与选项乱序 skeptic 独立读取有效账本和原始媒体。
8. **Controller adjudication**：仅允许双验证一致、严格词面对齐审计或严格时间合成审计三种 verified 停止。

### 3.3 动作集合

- 导航：`expand`、`zoom_out`、`shift`
- 观察：`observe`、`compare`
- 观察模式：`overview`、`inspect`、`motion`、`event_verify`、`detail_ocr`、`subtitle`
- 决策请求：`verify`、`answer`

`event_verify` 是控制器保留动作：先用 target-blind 粗观察找到线索，再在短 shot 中只揭示一个目标事件，以带标签的 before/after contact sheet 核验视觉转变。完整问题和选项仍隐藏。

## 4. 不同证据拓扑

### 4.1 Local / 对话题

- 利用原始字幕做 option-aware **检索提案**，只决定应观察哪个节点。
- Observer 必须重新读取该节点 SRT，提案本身不能进入账本。
- 高精度 lexical alignment 只接受 option-unique phrase，并要求完整 slot 与至少一个模型验证器审计。
- 可用 `allow_alignment_adjudication: false` 做消融。

### 4.2 Sequence / 顺序题

1. 口语事件用字幕落账；可见动作使用 `motion`。
2. 粗视觉 Observer 是 target-blind，只转写可见动作/状态变化。
3. 相关但不充分的事实进入 `*_routing`，不计 slot coverage。
4. routing clue 命中可细分节点时，下钻到子 shot 并执行 `event_verify`。
5. 有效动作证据必须满足：
   - 与某个 slot 有足够且有 margin 的实体词面重合；
   - 动作谓词兼容；
   - 至少两个不同时刻的引用帧；
   - 静态“拿着/盖着”只作为 clue。
6. 时间合成器按每个 required slot 的最早有效证据排序，映射到 `(a)(b)(c)` 一类选项。
7. 时间候选仍需至少一个模型验证器引用合成所用证据后才能 verified。

### 4.3 Multi-set / Global / Exclusion

- 合同和通用主动观察路径已经支持这些拓扑。
- `multi_set` 允许多个 slot 与 compare；`global/exclusion` 要求覆盖而非局部命中。
- 当前尚无真实 2B smoke 通过记录，不能视为完成实证验证。

## 5. 账本状态

- `breadth`：根层路由事实，不参与充分性。
- `*_routing`：局部相关线索，可触发下钻，不参与充分性。
- 其他 observation mode：通过 grounding gate 的 active evidence。

字幕事实不会保存 Observer 的解释性复述，而保存实际命中的 SRT 行；时间范围由引用行解析。视觉/OCR 事实的时间范围由引用帧时间决定，不信任模型自填时间。

## 6. 停止规则

Verified：

- `dual_verifier_agreement`
- `grounded_alignment_adjudication`
- `grounded_temporal_adjudication`

Unverified：

- required slot 在搜索上限后仍缺失；
- verification repair 用尽；
- 剩余调用不足以完成双验证；
- 模型调用硬预算耗尽；
- 协议/控制器错误。

输出总会保留 best-effort benchmark label 以兼容评测器，但只有 `trace.verified=true` 才表示机制通过。

## 7. 可消融配置

- `allow_alignment_adjudication`
- `allow_temporal_adjudication`
- `enable_subtitle_retrieval_proposal`
- `alignment_min_phrase_tokens`
- `alignment_min_margin`
- `temporal_min_gap_seconds`
- 搜索、修复、导航和模型调用预算
- 各观察模式帧数与字幕字符预算

对比实验至少应报告：答案准确率、verified rate、strict exposure/grounded set recall、调用数、唯一帧数、累计帧查看数、输入/输出 token 和墙钟时间。

## 8. 当前真实验证状态

### 8.1 已通过：字幕 local 题 102-2

- active-tree：预测 B，正确，`verified=true`
- 停止：`grounded_alignment_adjudication`
- 6 次模型调用，约 53.3 秒
- direct 与旧 coarse-to-fine 2B 都预测 C
- 产物：`runs/active_tree_102-2_v8.json`、JSONL trace 和 HTML replay

这只证明一题上的主动检索与证据审计链有效，不证明总体提升。

### 8.2 未通过：sequence 题 496-1

- gold D：`(b)(a)(c)`
- v9：`verified=false`；仅 S2 字幕成为 active evidence；调用预算正常耗尽
- 控制器正确路由：
  - S1 候选：`L1-N0002 → L0-N0012`
  - S3 候选：`L1-N0003 → L0-N0018`
- 局部 v11 证明 2B 把 139 秒无羽毛帧也描述为“有羽毛”，因此 S3 被保守拒绝

结论是“树与路由连通，但 2B fine perception 未通过”，不能写成 sequence pipeline 已跑通。

### 8.3 自动验证

- 当前全量 pytest 通过（具体数量以冻结前测试输出为准）
- Ruff 通过
- compileall 通过
- Evidence30 v0.2.0 已有 30 条 AI-assisted internal reference（18 dev / 12 locked），不能称为独立人工 gold
- 无模型 preflight 已验证 Schema、固定哈希、parquet 映射、媒体与字幕完整性

## 9. 下一阶段门槛

在租卡跑全量 8B 前：

1. 使用本地 2B 在 18 题 dev 上依次完成 `direct`、旧 `coarse_to_fine`、`active_tree` 三路共 54 次推理。
2. 仅以执行完整性作为冻结门槛：合法选项、完整 trace/停止原因/资源账本、无 fatal/degraded/预算越界；效果指标不设阈值。
3. 冻结源码、prompt、显式配置、relaxed scorer、标注、模型权重和运行环境，生成唯一 `freeze_id`。
4. 用同一冻结版本在 12 题 locked 上一次性完成 36 次推理，首次只查看聚合；逐题产物密封。
5. 根据三路配对结果判断 active-tree 是否值得进入 8B 小规模实验；若基于 locked 调参，另建 held-out 集。

局部验证命令见 `scripts/smoke_event_validator.py`；它不读取答案标签，可单独测试已路由 shot 的视觉事件判别。
