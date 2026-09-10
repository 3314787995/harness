# R1–R9 Pipeline 与 Benchmark 具体题型对应说明

整理日期：2026-09-10。分类基线：[`benchmark_pipeline_reclassification_v04.xlsx`][W]，并吸收 2026-09-07 至 09-09 的题型讨论与 R3 重设计材料。

本文回答“哪种题该用哪条 pipeline”。**先看题目要求保存和处理什么证据，再看 benchmark 原生标签。** 一个原生标签可能包含多种执行子型，代表题的归属也不能代替整个标签的归属。

## 1. 阅读口径

| 标记 | 含义 |
|---|---|
| **通常对应** | 工作簿建议的默认方向；仍需核对实际题意、范围和输入模态，不是整类硬绑定。 |
| **条件对应** | 只有满足所列条件时才使用该 pipeline；主流程由最终查询决定，其他 pipeline 可提供取证组件。 |
| **待确认** | 计数单位、证据是否明示、标签与题意或观察截止尚有疑点，不能仅凭标签完成分流。 |

来源范围为 11 个 benchmark、127 个原生题型**入口**。127 不是题目数，也不表示数据覆盖率；本文精选主要对应关系，不逐项展开全部入口。名称保留工作簿的英文原生标签，Video-MME 与 Video-MME-v2 分开列示。[W]

**映射不等于实现或评测结果。** 本文不宣称所有列出的子型都已有数据适配器、可运行实现或实测效果。R6 已有 `r6-evidence-relations/1.0` 实现及 CPU 验收记录；真实 GPU smoke 尚未执行，详见[当前运行说明](pipelines/r6.md)。[原设计稿][D6P]保留为分类依据。涉及字幕、语音、音乐的题型须满足原评测输入协议；纯视觉实现不能据此宣称支持音频题。

## 2. 九类 Pipeline 精简对应总表

同一行内的“条件”只作用于其后的入口和子型。未展开的映射及 Source ID 可按“Benchmark＋原生题型”到工作簿的 `127题型重分类` 工作表反查。[W]

| Pipeline | 解决的问题与核心证据 | 通常对应：benchmark → 原生题型／具体任务 | 条件对应与边界 | 依据 |
|---|---|---|---|---|
| **R1 定向定位与局部取证** | 找到有限片段，读取足以区分答案的直接事实；局部不等于单帧。 | **Video-MME** → `OCR Problems`（特定文字）、`Object Recognition`（普通目标识别）、`Action Recognition`（局部动作）、`Attribute Perception`（单一属性）、`Spatial Perception`（同框背景关系）。<br>**LongVideoBench** → `Text-referred Object`、`Text-referred Event`、`Scene-referred Object Attribute`（由字幕／场景锚点定位后读取）。<br>**MLVU** → `Needle QA`、`Plot QA`（局部事实）、`Sub-Scene Captioning`（有限子场景描述）。<br>**LVBench** → `Key Information Retrieval`（局部年份／关键词）。 | **LVBench / Temporal Grounding**：给定区间，只问发生什么 → R1。<br>**EgoLifeQA / EntityLog、EventRecall**：直接回忆地点、价格明示值、某个事实 → R1。<br>**Video-MME-v2 / Visual Recognition**：直接可见 → R1；持续遮挡追踪 → R2。`Audio-Guided Visual Description`、`Vision-Guided Audio Description` 的同期事实绑定通常为 R1，但依赖合法音频／转写与画面对齐。<br>全范围否定／清单转 R4；第 N 个事件选择转 R3。 | [W]、[D145] |
| **R2 连续动态与实体状态追踪** | 保留同一实体的身份、状态、位置和有序动态，判断变化、方向、遮挡对应或相位。 | **MVBench** → `Action Antonym`、`Fine-grained Action`、`Fine-grained Pose`、`Moving Direction`、`Object Shuffle`。<br>**TVBench** → `Action Antonym`、`Moving Direction`、`Object Shuffle`。<br>**TOMATO** → `Direction`、`Rotation`、`Shape & Trend`、`Velocity & Frequency`。<br>**LongVideoBench** → `Scene-referred Object Attribute Change`、`Text-referred Object Attribute Change`。<br>**Video-MME-v2** → `Entity Attribute Change Detection`、`Entity Persistence Tracking`、`Fine-Grained Action Recognition`、`Motion Properties Analysis`、`Motion Trajectory Estimation`。 | **MVBench / Moving Attribute、Object Existence**：必须先判定哪个对象在运动时用 R2；普通静态属性／存在性可 R1。<br>**Video-MME / Action Recognition**：需区分细过程、动作正反时用 R2。<br>**Video-MME-v2 / Temporal Periodicity Detection**：稳定周期及下一相位用 R2，开放未来预测转 R7。<br>数“多少只不同的鸟离开”主 R4；数“离开发生几次”主 R3。 | [W]、[D2] |
| **R3 事件账本与时序归约**<br>新版执行设计：**查询驱动时间证据** | 围绕当前查询取得事件／区间证据，计算次数、顺序、第 N 个、首次／末次、邻接、时长或共现频次。 | **MVBench** → `Action Count`、`Action Localization`、`Action Sequence`、`Character Order`、`Scene Transition`。<br>**MLVU** → `Action Count`、`Action Order`。<br>**TVBench** → `Action Localization`、`Action Sequence`、`Egocentric Sequence`、`Scene Transition`、`Unexpected Action`（定位有趣片段）。<br>**TOMATO** → `Action Count`、`Visual Cues`（可见演奏动作的先后／时机）。<br>**Video-MME-v2** → `Repetitive Action Counting`、`Event Sequence Ordering`、`Object Appearance Ordering`、`Temporal Action Localization`。<br>**VSI-Bench** → `Appearance Order`。 | **Video-MME / Temporal Perception**：视频内事件邻接／时长／顺序 → R3；`Counting Problem` 的事件次数 → R3；`Object Recognition` 中第 N 个制作事件的成品 → R3。<br>**LongVideoBench / Event before/after Event、Event before/after Text、Sequence of Scenes**：需时间选择／排序时用 R3。<br>**LVBench / Temporal Grounding**：求事件区间、时长、顺序 → R3。<br>**EgoLifeQA / HabitInsight、RelationMap**：统计活动／共同参与频次 → R3；`EventRecall` 的锚点后首个事件 → R3。 | [W]、[D3]、[D3P] |
| **R4 清单构建与集合归约** | 建立范围内的唯一实体、类别、文字或事实清单，做去重计数、并差集、最多／最少或缺失项核验。 | **Video-MME** → `Counting Problem` 的实例／类别数量比较。<br>**Video-MME-v2** → `Basic Counting` 的实例数量。<br>**MVBench** → `Moving Count`（符合运动条件的不同对象）。<br>**TVBench** → `Object Count`（指定时段运动实体清点）。<br>**VSI-Bench** → `Object Count`（跨视角的房间实例去重）。<br>**LongVideoBench** → `Scene-referred Object Existence`（限定场景未出现项）。<br>**EgoLifeQA** → `TaskMaster` 的计划清单减已完成清单。 | **Video-MME / Object Recognition、Action Recognition、OCR Problems**：问未出现物品／活动、完整文字集合／未提及关键词 → R4，取证模态须匹配“看见”或“提及”。<br>**LVBench / Entity Recognition**：问实例数量 → R4；`Key Information Retrieval` 的完整关键词集合 → R4。<br>**Video-MME-v2 / Entity Existence Change Detection**：数不同离开实体 → R4，可借用 R2 追踪。<br>**TVBench / Action Count**：计数单位待确认，不纳入确定的 R4 核心入口。 | [W]、[D4]、[D145] |
| **R5 分段全局综合** | 覆盖相关范围，合并多段事实，形成整体活动、主线、体裁或事实摘要。 | **Video-MME** → `Information Synopsis`（事实性整体内容／体裁）。<br>**MLVU** → `Topic Reasoning`、`Video Summarization`。<br>**EgoSchema** → `Long-form Video QA` 的整体行为概括。<br>**LVBench** → `Summarization` 的整体综合。 | **LVBench / Event Understanding**：若问整体流程而非局部事件 → R5。<br>**MLVU / Sub-Scene Captioning**：若需综合一个较大子场景的多段事实 → R5，有限片段直接描述仍 R1。<br>**Video-MME-v2 / Symbolic/Metaphorical Interpretation**：仅复述剧情／体裁的子题可 R5，推断隐喻寓意为 R6。<br>长视频、长答案本身都不构成 R5 条件。 | [W]、[D145] |
| **R6 证据约束关系推理** | 用事实及关系证据比较原因、动机、人际关系、叙事反差或跨模态语义，核验解释。 | **Video-MME** → `Action Reasoning`、`Object Reasoning`、`Spatial Reasoning`、`Temporal Reasoning` 的非明示关系推断。<br>**LVBench** → `Reasoning` 的原因／关系子型。<br>**MVBench** → `Episodic Reasoning`。<br>**Video-MME-v2** → `Causal Reasoning`、`Collective Dynamics Analysis`、`Dyadic Interaction Dynamics`、`Cross-Modal Semantic Consistency`、`High-Order Narrative Deconstruction`、`Narrative Turning Point Detection`、`Symbolic/Metaphorical Interpretation`。 | 答案在允许的画面／台词中已明示时可 R1；事实摘要 R5；反事实或未见未来 R7；明确数学约束 R8。<br>**EgoSchema / Long-form Video QA**：隐含动机／社交解释 → R6。<br>**EgoLifeQA / HabitInsight、RelationMap**：解释原因／隐含关系 → R6，单纯频次统计仍 R3。<br>**MLVU / Anomaly Recognition、MVBench / Unexpected Action**：解释异常／笑点才用 R6，直接识别事件可 R1。<br>**当前实现：1.0；已有 CPU 验证，真实 GPU smoke 尚未执行。** | [W]、[D6]、[D6P] |
| **R7 假设与未来推演** | 固定已观察事实与截止，施加干预或推演未见未来，比较替代结果。 | **MVBench** → `Action Prediction`、`Counterfactual Inference`。<br>**Video-MME-v2** → `Future Event Prediction`、`Counterfactual Reasoning`。 | **LVBench / Reasoning、Video-MME / Temporal Reasoning**：确实预测未观察未来时 → R7。<br>**EgoLifeQA / TaskMaster**：推演尚未发生的后果 → R7，计划减已完成仍 R4。<br>**Video-MME-v2 / Temporal Periodicity Detection**：开放式未来行为才转 R7；稳定周期下一相位通常 R2。<br>观察范围必须按协议确认，不能把完整视频上的“接下来发生了什么”自动改成预测题。 | [W]、[D7] |
| **R8 视觉符号与约束求解** | 从证据绑定变量、单位和公式，进行可复算的数值／符号运算，并回代核验。 | **Video-MME-v2** → `Numerical Calculation`（公式、变量关系、明确数值计算）。 | **Video-MME-v2 / Professional Knowledge Acquisition**：数学图形面积、明确公式求解 → R8；明示知识 R1、概念关系应用 R6。<br>**EgoLifeQA / EntityLog**：由总价和数量求单价 → R8；只读标价 R1。<br>**LVBench / Key Information Retrieval**：读取数值后必须计算 → R8。<br>**Video-MME / Temporal Perception**：比较内容中给出的年代数值 → R8；读明示答案 R1、视频事件排序 R3。 | [W]、[D8] |
| **R9 空间场景建模与查询** | 绑定对象及参考系，建立足够解题的几何／拓扑关系，求方向、距离、尺度、朝向或路线。 | **VSI-Bench** → `Absolute Distance`、`Object Size`、`Relative Direction`、`Relative Distance`、`Room Size`、`Route Plan`。<br>**MVBench** → `Egocentric Navigation`。<br>**Video-MME-v2** → `Spatial Understanding` 的参考系、跨视角关系和导航子型。 | **Video-MME / Spatial Perception、Spatial Reasoning**：跨视角、非同框空间关系或参考系转换 → R9；直接背景事实 R1、原因解释 R6。<br>**VSI-Bench / Object Count** 主 R4，`Appearance Order` 主 R3，不因属于空间 benchmark 而全走 R9。<br>真实房间面积依赖尺度与边界 → R9；给定数学图形约束求面积 → R8。<br>`Spatial Understanding` 中标签与题意不符的样本须单独复核，见第 5 节。 | [W]、[D9] |

## 3. 最容易混淆的分流规则

| 判别点 | 具体分流 | 代表入口／例子与来源 |
|---|---|---|
| **次数、实例数、类别数** | 同一动作独立发生几次 → R3；多少个不同对象／多少种类别 → R4；直接读取视频已经给出的数量 → R1。 | Video-MME / `Counting Problem`；Video-MME-v2 / `Basic Counting` 的浇花次数为 R3、货运车厢数量为 R4。[W]、[D4] |
| **运动条件不改变计数主目标** | “多少只鸟飞出”维护鸟的唯一集合 → R4，可用 R2 追踪；“同一只鸟飞出几次”维护发生事件 → R3；只问进出状态变化 → R2。 | Video-MME-v2 / `Entity Existence Change Detection`。[W]、[D2] |
| **给定区间与求解区间** | 已给时间段，只问该段事实 → R1；需要求事件位置、时长、先后或相邻事件 → R3。 | LVBench / `Temporal Grounding`；“01:58–02:46 发生什么”是 R1。[W] |
| **事件顺序中的物体识别** | 先选第 N 个制作事件再认成品 → R3＋局部识别；只认已指定片段里的物体 → R1。 | Video-MME / `Object Recognition`，225-3“第二个纸动物” → R3。[D3] E02 |
| **否定项需要范围证据** | 单处普通识别 → R1；“全片／限定场景内没有出现哪项” → R4；“哪项陈述不正确”须逐项取证，只有需要关系推断时才主 R6。 | Video-MME / `Action Recognition`，251-1“未展示的日常活动” → R4；`Attribute Perception` 的陈述核验不能一律归 R6。[D4] E17、[D6] E04 |
| **概括内容与解释含义** | 跨片段事实综合 → R5；解释动机、寓意、笑点或反差 → R6。 | EgoSchema / `Long-form Video QA`；Video-MME-v2 / `Symbolic/Metaphorical Interpretation`。[W]、[D6] |
| **next／will 不等于预测** | 允许的视频中已发生事件的后继／出现顺序 → R3；周期下一相位 → R2；观察截止后未见的行为 → R7。 | VSI-Bench / `Appearance Order` 是 R3；Video-MME-v2 / `Future Event Prediction` 须先确认可用观察范围。[W]、[D7] |
| **空间词不等于场景建模** | 同框可见关系 → R1；连续物体运动 → R2；跨视角参考系／距离／路线 → R9；解释物体为何飞起 → R6。 | Video-MME / `Spatial Perception`、`Spatial Reasoning`；Video-MME-v2 / `Spatial Understanding`。[W]、[D9] |
| **数字不等于 R8** | 读数字 → R1；事件计次 → R3；集合基数／最多类别 → R4；公式、单位、变量约束求解 → R8；真实场景尺度恢复 → R9。 | EgoLifeQA / `EntityLog` 的单价计算；VSI-Bench / `Room Size`。[W]、[D8]、[D9] |
| **相同英文标签可能不同任务** | 按各 benchmark 的实际问题与答案接口分流，不能跨 benchmark 直接复制映射。 | MVBench / `Unexpected Action` 通常识别意外事件 → R1，解释笑点可 R6；TVBench 同名入口要求定位片段 → R3。TVBench / `Action Count` 待确认。[W] |

“完整范围”是题目限定的范围，不总是整段视频；混合题可调用其他类的组件，但主类仍按最终查询所需的核心证据与运算决定。

## 4. Video-MME：12 个原生题型专项对照

“默认”严格保留 v04 工作簿的建议默认 R，**不是整类最终归属**。条件列吸收较新原题材料；新增分流在第 5 节说明。下表只针对 Video-MME，不与 Video-MME-v2 的 33 个原生入口混用。[W]

| 原生题型 | 默认 | 具体子型 → Pipeline | 待确认事项／代表题提示 |
|---|---|---|---|
| `Action Recognition` | R1 | **通常 R1**：定位后识别局部动作。**条件 R2**：细粒度动作过程／正反。**条件 R4**：范围内未展示的活动、完整活动类别清单。 | 251-1 问未展示的日常活动 → R4；这是较新原题材料对 v04 候选 R1／R2 的补充。[D4] E17 |
| `Object Recognition` | R1 | **通常 R1**：普通物体识别。**条件 R4**：未出现／未讨论物品、类别清单。**条件 R3**：按第 N 个事件选择其物体／成品。 | 225-3“第二个纸动物” → R3；“墓葬哪些物品未被讨论” → R4。后者还需匹配讨论证据的输入模态。[D3] E02、[W] |
| `Attribute Perception` | R1 | **通常 R1**：直接属性读取。**条件 R4**：穷举属性／缺失项。**条件 R6**：须结合多条事实推出陈述是否成立。 | 工作簿将“萨拉热窝事件哪项不正确”代表题列 R6；较新讨论指出逐项事实核验也可能足够，需确认是否确有关系推断。[D6] E04 |
| `OCR Problems` | R1 | **通常 R1**：定位特定文字并读取。**条件 R4**：全范围唯一文字／关键词集合、未出现或未提及项。 | 062-2 问未提及关键词；不能仅凭 OCR 标签把“提及”改写成“屏幕出现”。[D4] E19 |
| `Counting Problem` | R4 | **通常 R4**：不同实例／类别数量及最多比较。**条件 R3**：重复事件次数。**条件 R1**：读取视频明示数量。 | 先固定单位是事件、实例还是类别。圣诞装饰物哪类最多属于实例分组计数和比较 → R4。[W] |
| `Spatial Perception` | R1 | **通常 R1**：同框背景、直接空间事实。**条件 R9**：非同框关系、跨视角或参考系转换。 | 空瓶出现时背景有什么 → R1；不是看到 Spatial 就启动空间重建。[W] |
| `Temporal Perception` | R3 | **通常 R3**：视频内顺序、邻接、时长。**条件 R1**：内容中明示的年代／阶段事实。**条件 R8**：读取年代后进行数值比较。 | 251-3 清洁地板后的动作、556-3 停留最久地点 → R3；“最早人类演化阶段”须区分语义年代与视频出现顺序。[D3] E03／E23、[W] |
| `Action Reasoning` | R6 | **通常 R6**：结合行为与背景解释动机。**条件 R1**：视频已明确给出原因。 | 考古人员为何发掘墓葬：需检查解说是否直接说明，不能仅按 Reasoning 定类。[W]、[D6] |
| `Object Reasoning` | R6 | **通常 R6**：证据约束的对象／知识关系推断。**条件 R1**：解说明示的关联事实。 | 食物对应哪个国家的代表题仍有明示性与关系措辞疑点，暂保留条件性 R6，不凭常识直接代替取证。[W]、[D6] E05 |
| `Spatial Reasoning` | R6 | **通常 R6**：涉及空间对象的原因／关系解释。**条件 R1**：直接空间事实。**条件 R9**：几何、参考系或场景关系查询。 | “屋顶物体为何飞起”是原因问题 → R6，并非距离／方向运算。[W]、[D9] B02 |
| `Temporal Reasoning` | R6 | **通常 R6**：时间语境中的原因／关系解释。**条件 R1**：明示历史事实；**条件 R3**：真实事件先后；**条件 R7**：未见未来推演。 | 节日传统传到美国后发生什么，若是解说中的直接事实 → R1；工作簿默认 R6 与代表题 R1 并不矛盾。[W]、[D6] B13 |
| `Information Synopsis` | R5 | **通常 R5**：整体活动、事实主线、体裁。**条件 R4**：完整实体／内容列表。**条件 R6**：隐喻、深层寓意或动机解释。 | “视频是什么体裁”通常 R5；摘要长度不决定分类，输出为选择题也可需要全局综合。[W]、[D145] |

因此，“R1 对应 Video-MME 哪些题型”的简明答案是：**通常对应 OCR Problems、Object Recognition、Action Recognition、Attribute Perception、Spatial Perception 中可通过局部直接事实回答的子题**；其他标签下的明示事实题也可能归 R1。上述五类中的否定穷举、事件排序、动态追踪和跨视角空间问题须另行分流。

## 5. 新材料补充、冲突与待确认项

| 项目 | v04 基线或旧表述 | 本文采用的处理 |
|---|---|---|
| **R3 执行设计更新** | 主类名称为“事件账本与时序归约”。 | 保留 R3 分类身份，同时注明 09-09 新设计为“查询驱动时间证据”：按当前运算组织必要观察、轻量候选和程序归约，**不要求每题建立完整全片账本**。这是执行设计更新，不是新增类别。[D3P] |
| **Video-MME / Object Recognition** | 条件候选为 R1／R4。 | 新材料 E02 的 225-3 要选第二个制作事件，补充 **条件 R3**；保留原生 Object Recognition 标签。[D3] |
| **Video-MME / Action Recognition** | 条件候选为 R1／R2。 | 新材料 E17 的 251-1 要查未展示活动，补充 **条件 R4**；不是将整类改成清单任务。[D4] |
| **TVBench / Action Count** | 默认及代表题暂列 R4，候选 R3／R4；模板涉及 sets／distinct repeated actions。 | **待确认**：复核具体视频和标注中的次数、动作种类或组数口径。表中不把暂定 R4 当作已确定结论；不能照搬 MVBench / Action Count → R3。[W]、[D145] §8.3 |
| **Video-MME-v2 / Basic Counting** | 默认 R4；工作簿记录论文定义与官方代表题存在计数语义差异。 | 保留原生标签，按实例数 R4、事件次数 R3 拆分；002-1 浇花次数与 004-1 货运车厢数量是两个不同执行子型。[W] |
| **Video-MME-v2 / Spatial Understanding** | 候选 R1／R9。 | 较新材料 B05 的 451-4 主要问运动轨迹，补充条件 R2，必要时结合 R9；B06 的 449-4 跨项目优缺点更接近 R5／R6，列为**标签与机制不吻合、待复核**，不把整类扩成无条件通用入口。[D9] |
| **R6 与明示事实的边界** | 部分代表题暂列 R6。 | 原因、立场、身份、陈述核验若由允许的证据直接给出，可 R1。尤其 Attribute Perception 代表题需复核实际推理负担；不靠题干中的 Reasoning 或“不正确”决定。[D6] |
| **R7 的观察截止** | 工作簿列出预测／反事实入口。 | 较新材料指出 Video-MME-v2 的相关标注没有逐题观察截止字段；必须按数据协议另行确认，不能从标签猜 cutoff。MVBench 的观察窗口也不必从 0 秒开始。[D7] §1.3 |

本次依据已有工作簿和本地讨论材料整理，没有新增逐题视频审阅、模型推理或准确率测量。材料中的官方标注定位用于追溯，**不等于本文已确认每题的实际最小证据需求**；上述待确认项保持待确认。

## 6. 来源与查证方式

| 来源 | 用途与定位 |
|---|---|
| [v04 重分类工作簿][W] | `9类Pipeline`：九类定义；`127题型重分类`：Benchmark、原生题型、Source ID、默认／代表题／条件候选 R、理由、论文和数据链接；`关键边界与反例`：边界案例。 |
| [R1／R3／R4／R5 实施讨论][D145] | R1 视觉基线边界、R5 事实综合边界；§8 给出候选入口及 TVBench Action Count 待审说明。 |
| [R2 特征与原题材料][D2] | 连续动态、状态、运动筛选、遮挡身份与周期子型；§2 原生入口表。 |
| [R3 特征与原题材料][D3]、[R3 查询驱动重设计][D3P] | 前者 E02／E03／E23 支持新增／细化的 Video-MME 子型；后者 §0、§3 说明新版执行设计。 |
| [R4 特征与原题材料][D4] | E17：Action Recognition 的否定活动清单；E19：OCR 关键词与“提及”模态边界；其他案例支持实例、类别和集合运算。 |
| [R6 特征与原题材料][D6]、[R6 设计稿][D6P] | 原因、社会关系、叙事及跨模态推理与直接事实的边界；设计稿记录设计阶段状态；当前实现状态以本仓库运行说明及验证记录为准。 |
| [R7 特征与原题材料][D7] | 未来与反事实入口，§1.3 观察窗口／截止字段，及周期相位边界。 |
| [R8 特征与原题材料][D8] | §1.2 五个相关入口；直接读数、数学求解与专业知识标签的区别。 |
| [R9 特征与原题材料][D9] | §1.2 原生入口；B05／B06 补充 Spatial Understanding 的异质样本；空间测量与参考系边界。 |

查某个映射时，先在工作簿按 **Benchmark＋原生题型** 找到 Source ID 和判定理由，再打开对应讨论材料的例题编号核对具体题意。本文只保留这一份当前说明，不复制旧版 pipeline 或实验结果。

[W]: sources/benchmark_pipeline_reclassification_v04.xlsx
[D145]: sources/r1,r3,r4,r5.md
[D2]: sources/R2_特征与原题讨论材料.md
[D3]: sources/R3_特征与原题讨论材料.md
[D3P]: sources/R3_查询驱动时间证据_pipeline重设计.md
[D4]: sources/R4_特征与原题讨论材料.md
[D6]: sources/R6_特征与原题讨论材料.md
[D6P]: sources/R6_Qwen3VL8B_TrainingFree_Pipeline.md
[D7]: sources/R7_特征与原题讨论材料.md
[D8]: sources/R8_特征与原题讨论材料.md
[D9]: sources/R9_特征与原题讨论材料.md
