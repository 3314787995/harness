> 分类来源材料：保留原材料的设计语境；当前实现与验证状态请看 [运行文档](../pipelines/) 和 [发布验证](../validation.md)。

# R6：面向 Qwen3-VL-8B-Instruct 的证据约束关系推理 Pipeline

## 设计状态与适用范围

本文以《R6_特征与原题讨论材料(1).md》为任务依据，结合文末六篇论文提出工程设计。主模型固定为 `Qwen/Qwen3-VL-8B-Instruct`。不更新模型参数，不进行 SFT、LoRA、RL，不训练路由器、验证器、奖励模型或检索器，也不以更强闭源模型承担最终推理。

这是一份待实现、待验证的设计，不是已经运行的系统报告。附件中的原题、答案和边界分析属于用户提供材料；本文的中间表示、控制器、提示词和预算属于工程建议。没有观看这些原题的完整视频，因此不提供虚构的逐帧证据、时间戳或最小充分帧集。参数是开发起点，不是论文结论或模型能力上限。

R6 的目标不是让模型写出更长的推理文本，而是：用可追溯的观察约束关系判断，找出候选解释之间尚未解决的区别，并针对区别获取证据。

## 1. 文献依据与迁移边界

| 论文 | 可借鉴的机制 | 本方案的具体用法 | 不直接照搬的部分 |
|---|---|---|---|
| MoReVQA，CVPR 2024 [P1] | 事件解析、定位、推理分阶段；共享外部记忆；训练无关调用 | 将任务拆成短上下文调用，所有阶段读写同一份受控状态 | 不要求 8B 在第一次调用时生成整题完整程序；不能假定模块越多越好 |
| VideoAgent，ECCV 2024，Xiaohan Wang 等 [P2] | 根据已有观察继续获取缺失信息 | 采用迭代观察；将笼统信心反馈改造成显式证据缺口 | 不直接使用模型自报“充分”作为停止真值，也不迁移原模型的性能数字 |
| VideoTree，CVPR 2025 [P3] | 问题相关、自适应粒度的视频表示 | 相关区间加密，必要时回查前后文 | 不为每个短视频建立完整层次树 |
| Video Question Answering with Procedural Programs，ECCV 2024 [P4] | 语义模块和程序化操作组合 | 语义交给 Qwen；时间、集合、否定逻辑和标签映射由代码完成 | 不默认执行模型自由生成的 Python |
| Chain-of-Verification，Findings of ACL 2024 [P5] | 将核验问题与初稿分离，独立回答核验问题 | 不向关键核验调用展示先前选中的字母及其解释 | 同模型另一次调用不等于独立证据，也不保证纠错 |
| CRITIC，ICLR 2024 [P6] | 用工具反馈支持批评与修正 | 回到原帧、原对话或同期音频检查缺口 | 不把“重新想一遍”包装成外部验证 |

MoReVQA 同时提醒我们：简单的帧描述后回答基线可能很强，单阶段复杂程序未必更好。因此本方案必须与简单基线做受控比较，而不是预设结构化流程必然获胜。

## 2. 总体架构

```text
冻结输入协议
    ↓
查询编译：问题 + 完整选项 → 待证命题与判别需求
    ↓
初始定位与局部观察
    ↓
直接证据已足够？──是──→ 完整命题核验 → 原生答案
    │ 否
    ↓
维护事实表 + 关系表
    ↓
候选比较：事实成立情况 + 对题目目标的契合情况
    ↓
缺口明确？──是──→ 选择合法取证动作 → 新观察 → 更新状态
    │ 否 / 候选已可区分
    ↓
关键关系独立核验
    ↓
充分则回答；不充分则继续取证或以资源性停止结束
```

同一套 Qwen 权重承担 Compiler、Observer、Relation Checker、Verifier 四种调用角色。它们不是四个独立部署的 agent，更不是四个经过训练的模型。

程序控制器负责范围、来源、预算、动作执行、缓存、逻辑归约和输出格式；模型负责语言解析、视觉观察与有来源的语义判断。控制器不假装能够仅凭 JSON 格式正确，就证明模型说的事实是真的。

S1—S6 只配置“这次要检查哪种关系、需要什么证据”；不实现六套独立 solve()。已有 R1、R3、R4、R5 能力通过底层模块复用，而不是嵌套启动多个完整求解器。

## 3. 输入协议与查询编译

### 3.1 程序先冻结 Protocol

最少保存：

```text
video_id / media_hash
allowed_intervals
allowed_modalities
history_cutoff
subtitle_policy
cross_question_cache_policy
answer_protocol
budget_profile
```

这些字段由数据适配层提供，不能由模型自行放宽。答案键、离线人工参考时间、组内其他题的标准答案不进入求解输入。外部网页检索默认关闭；本文研究论文时的联网，不等于评测时允许检索视频剧情或答案。

需区分：

- reference_scope：题目问的是哪个阶段、人物状态或事件。
- evidence_scope：评测实际允许读取哪些输入。

“开头两人关系如何”不自动表示只能看开头，也不表示可以用结尾和解覆盖开头关系。后来对白可以在允许范围内澄清早期身份，但角色在早期的信念不能被后验结论改写。

上下文扩展、字幕、音频、缓存、检索索引都必须受同一个协议检查。只对最终选帧检查范围，而让全文字幕或全片摘要提前越界，并不合规。

### 3.2 Compiler 看问题和全部选项，但不产生视频事实

建议输出：

```json
{
  "answer_operator": "best_explanation",
  "target_description": "本题要求解释的事件或关系",
  "reference_scope": "原题限定的时期或事件",
  "entities": [],
  "option_claims": [],
  "discriminators": [],
  "initial_evidence_requests": [],
  "ambiguities": []
}
```

选项保留原文、原始编号、否定词、比较词、时间和人物限定。复合选项拆成若干原子命题，但保存逻辑结构，不能把所有句子都不加区分地视为独立证据。

E09 的选项应拆出人物、动作/表情、音乐性质、视觉基调和一致/冲突关系；只留下“悲伤、一致”会抹掉 F/H 的人物差异。E08 要保留“依赖资源与魅力，而不是报价本身”，不能简化为“提到资源”。

Compiler 得到的是待检验假设，不是观察结果。程序不得把 option_claims 直接写入事实表。

## 4. 三张轻量逻辑表

### 4.1 事实表 F：原始来源支持了什么

```text
fact_id                    程序分配
source_ids                 指向实际交给模型的帧、文本段或音频片段
entity_ids                 已绑定实体；未确定时保留临时 ID
story_time                 事实发生时段
source_time                对应来源所在时段
kind                       observation / attributed_statement
predicate                  最小观察或陈述命题
speaker / referred_entity  发言人及被谈论对象
quote_or_paraphrase         逐字引用或转述，明确区分
quality                    clear / partial / unreadable / unresolved
coverage_notes             遮挡、未采样、未提及等情况
```

Source manifest 保存媒体哈希、实际帧号/时间、裁剪区域、输入模态、解码和采样参数。模型只能引用本次真正收到的 source_id；新 ID 由程序生成，不让模型编造。

“某人说原因是 X”属于 attributed_statement，不自动变成“X 是全部真实动机”。“她以为饼干属于自己”也要有行为、对白等证据；对其信念的推断放关系表，并标注信念主体和当时时段。

相同源片段被重复描述三次不算三个独立来源。文本相似也不能作为跨场景事实去重的唯一依据。

### 4.2 关系表 R：事实怎样支持解释

```text
relation_id
claim
relation_type              causal_support / stated_reason / motive /
                           social_relation / reveals / theme_mapping /
                           semantic_consistency / rule_violation
premise_fact_ids
premise_relation_ids       可选；必须能递归回到原始来源且无循环
bridge                     1—2 句可核验的连接说明
support_state              supported / contradicted / unknown
missing_premises
strong_alternatives
conflict_flag
```

这里的 supported 表示当前证据支持，而不是形式逻辑系统证明了故事的唯一真实解释。关系判断仍可能出错。

三个命题必须区分：事件 B 在 A 之后；A 与 B 有共同背景；A 促成 B。前两者不能自动转换为第三者。也不能把关系模型自己的推断写回 F，再把它当新事实支持自己。

### 4.3 控制表 C：现在缺什么、下一步能做什么

```text
query_spec
active_candidates
candidate_assessments
pending_gaps
coverage_state
action_history
remaining_budget
reserved_verification_budget
stop_reason
```

不必引入图数据库。只有确实存在多人物、跨阶段依赖时，按相关关系记录临时展开小的来源依赖图即可。

## 5. 初始定位与观察

### 5.1 从最便宜、最可能相关的证据开始

已有合法时间锚点时，优先看锚点附近；没有锚点时，才做粗粒度时间概览或检索允许的字幕。

局部原因题先定位结果，再查相关前因与必要背景。全片主题题需要覆盖主要情节。多场合题需要建立完整的相关场合清单。这些是同一控制器下不同的覆盖需求，不由视频长度决定是否启用 R6。

粗采样只用于定位；没有在粗采样中看到某事件，不证明该事件不存在。尤其不能将 32 帧概览宣称为完整事件枚举。

### 5.2 起始窗口配置

作为尚未调优的起点，可采用：核心窗口 8 秒，前后各带最多 2 秒合法上下文；一般互动先用 2 fps，快速动作或短暂反应可升到 4 fps。对应约 24/48 帧，过长窗口分批。预处理可能有补帧或再次采样，因此实际输入帧数仍需记录。

动作关系需要时间密度；细小物体、文字、身份线索需要空间清晰度。不要每次不确定都只加帧。根据缺口分别选择扩时间、加采样、裁剪放大或换到所需模态。

### 5.3 Observer 不看先前预测

推荐起点：Observer 接收问题中的目标和由选项提炼的中性观察需求，不接收全部故事性候选和当前答案。它看见“记录人物、动作、表情、环境及其时间”，而不是“请证明女孩忧郁，因此选 H”。

这种中性需求仍受到选项影响，不能声称完全 option-blind。需做三种版本的消融：全部选项、仅问题、中性判别需求。

观察输出只写看见、读到或由真实音频工具听到的内容。不要求 Observer 同时给全片解释和最终答案。每次可先限制到 6—10 条相关记录，超过则拆分处理，而不是静默丢弃关键事实。

## 6. 直接证据通道

初次观察后检查：是否已经取得能回答完整问题的明确证据，且实体、时间、指代、否定或比较条件匹配。

E02 在允许且实际输入相应字幕时，动机可能已经明示。此时可以直接进行命题核验并回答，不强行运行多轮动机推断。

这不是另训练一个路由器，也不必跳到另一套完整 R1。只是在 R6 内允许关系层为空或极简，复用取证能力提前结束。直接通道仍检查完整选项，而不是见到一个共同关键词就停止。

## 7. 候选比较：事实成立与问题契合分开

每个候选至少保存两个维度：

```text
factual_status = supported / contradicted / unknown
answer_target_fit = complete / partial / off_target / unresolved
```

E14 中，H 可能正确描述“包里有饼干”的揭示，G 描述由揭示引起的归属认识改变。不能为了选 G 而把 H 硬判为假。两者要在“最大叙事转折”的目标上比较。

对有明确合取结构的选项：一个必要原子命题被反驳，可反驳该选项；必要命题尚有未知，不能记成全部成立；全部成立也不必然最切合问题目标。

“None of the other options are correct”必须结合题目正在选择正确描述、错误操作还是修改建议来编译，不能无条件简化成所有其他事件都未发生。对于 E11 一类角色不清楚的候选，先保留 ambiguity，不用代码制造虚假的逻辑确定性。

先轻量审视全部候选，再对最有竞争力的 2—3 个展开细查。不能因初次低信心把其他候选永久删除；淘汰必须有具体冲突或目标不匹配依据。

### 7.1 六类关系的核验义务

| 子型 | 必须建立的连接 | 不成立的捷径 |
|---|---|---|
| 因果解释 | 目标结果、具体前因/条件、连接依据、竞争原因 | 发生得更早、题材常识 |
| 动机/社会关系 | 人物与时期、决策或互动、陈述/行为证据、替代解释 | 提及资源等于主要动机；一次冲突等于长期敌意 |
| 叙事线索/转折 | 先前理解、后续揭示、被改变的解释、相关来源 | 最后一幕等于最大转折 |
| 隐喻/幽默 | 铺垫或预期来源、实际事件、反差/主题对应 | 看到猫就证明理解幽默；一个身份词就概括主题 |
| 跨模态语义 | 同期独立音频事实、视觉事实、语义对应及完整候选 | 用画面猜音乐，再用猜测自证一致 |
| 规则/陈述核验 | 明确比较基准、条件、实际步骤/属性、否定逻辑 | 没提到就当没执行；通用常识替换视频示范 |

对“主要原因”“rather than”“最大转折”等比较结构，不宜用支持事实的数量打分。重复镜头和冗长选项都会让简单计数失真。应核验候选能否解释题目指定的关系，以及哪一项证据真正区分竞争解释。

## 8. 缺口驱动的补看控制器

### 8.1 Gap 是下一次调用的最小任务

```json
{
  "gap_id": "由程序分配",
  "type": "relation_bridge",
  "predicate": "仍未确定的具体命题",
  "affected_candidates": ["候选原始 ID"],
  "entity_scope": [],
  "time_scope": "待定位或合法区间",
  "required_modality": "visual",
  "proposed_action": "observe_clip",
  "desired_observation": "什么可观察结果能帮助解决该分歧"
}
```

缺口类型至少区分定位、身份、局部事实、时间范围、关系连接、竞争解释、覆盖、模态缺失、题意歧义。只有“信心不够”不构成有效 Gap。

例如：E09 音乐尚未观察，调用音频；F/H 人物不同，补看人物而不是再听音乐；E14 归属连接缺失，回查前文而不是一直放大包内饼干。

### 8.2 有限动作库

```text
search_allowed_text
observe_clip
expand_context
inspect_source_frame_or_crop
resolve_entity
analyze_audio_if_available
reduce_by_code
```

每个动作必须说明针对哪个 Gap。程序检查时间、模态、预算、参数和是否重复。由模型提议 1—3 个动作，控制器执行一个或小批量合法动作；不让模型自由生成执行代码。

### 8.3 无训练的优先级

采用可审计的字典序规则，而不是虚构概率：先看是否阻塞合法作答，再看能否区分当前竞争候选，然后看是否有机会得到不同来源或更清晰观察，最后比较成本。

这只是启发式控制，不是已校准的期望信息增益，也不是最优 POMDP 策略。若后来需要比较优化收益，可在独立开发集上调阈值，但不在测试集按答案调策略。

### 8.4 无进展检测

相同来源、相同参数、相同目标的重复请求默认阻止。提高分辨率、改变合法时间窗口或引入确实缺失的模态属于不同观察动作。

实质进展包括新取得相关原始来源、消除身份/时间冲突、解决关键未知命题、完成必要覆盖；更长解释、同一来源的另一次改写或更高自报信心不计进展。

允许针对新的原始观察修正旧记录；修改必须保留版本与冲突来源，不能删除不利证据让表面一致。

## 9. 关键关系核验

采用类似 CoVe 的上下文隔离：Verifier 看到中性核验问题、必要的原始证据和待区分关系，但看不到上一轮选中的字母、自报信心和完整辩护文本。

每题优先核查 1—3 个最关键前提或关系跳步。例如：

- 这段话表达的是人物自己的理由，还是别人对他的猜测？
- 是否有证据把当前物品与此前那次互动绑定？
- 这项资源陈述是在解释决策，还是只是一般自我介绍？
- 两个模态是否来自同一个时间段？

核验结论必须带来源；来源不足就输出 unknown 和具体缺口。核验失败后返回控制器，不以多数票覆盖冲突。

同一模型隔离上下文只能减少一种答案诱导，不能变成独立真值。代码能验证来源存在、范围合法、依赖无循环，但语义是否真的被支持仍需模型判断与离线人审。

## 10. 停止与原生输出

### 10.1 证据充分停止

以下条件同时满足才标记 evidence_sufficient：目标候选的必要命题有合法来源，关键连接没有未解决前提，影响胜负的强竞争解释已被针对性处理，且不存在会改变答案的身份、时间或模态冲突。

不要求证明所有其他选项绝对为假，因为“最佳解释”选项可能并不互斥。充分性检查也是系统判断，不能当成数学保证。

### 10.2 资源性与能力性停止

保留以下原因：

```text
EVIDENCE_SUFFICIENT
BUDGET_EXHAUSTED
NO_PROGRESS
MODALITY_UNAVAILABLE
INPUT_AMBIGUITY
TOOL_FAILURE
```

预算耗尽不等于证据充分。缺音频不等于视频中无音乐。模型找不到答案不等于选项“Cannot be determined”成立。

如果原评测要求强制选择，仍按原协议输出一个候选，同时在诊断日志保存 forced_choice、证据缺口和停止原因；不得自创新的拒答标签。最终字母映射由代码完成。

## 11. 原题走查：仅展示应如何取证

以下不声称已观察到原视频，官方答案仅用于离线核对，不进入控制器。

### 11.1 E08：多场合动机

先定位实际相关场合并按真实时间编号，确认清单覆盖情况。选项出现“第五次”不证明视频有第五次。

每个场合记录决策、报价条件、资源/能力陈述、说话人、上下文和竞争理由。语义判断分别回答：是否仅提到资源？是否将资源作为这次决策的依据？是否比报价本身更符合问题要求？

将每个实际场合分成 T（满足条件）、F（不满足条件）、U（未定）。在场合全集 N 已可靠枚举的条件下，一个候选集合 S 仍可能精确成立的必要条件是：

```text
T ⊆ S ⊆ T ∪ U，且 S ⊆ N。
```

这只是一条候选兼容性过滤规则，不是证明。只要仍有会影响集合的未知，不能把其默认为否。

假设第三场尚未确定，而其结果区分“仅第二场”和“第二、三场”，下一次调用就应查第三场的决策依据。若最后独立取证确实得到 {2,3}，代码再映射到原选项 F。不能把附件给出的 F 当作取得场合证据的依据。

### 11.2 E14：饼干转折

需要获取三类证据：前文围绕饼干归属的行为/陈述，双方相关互动，后文揭示及反应。是否足以确定原先归属，需要看实际片段。

H 对应包内饼干的发现；G 对应对早先饼干归属的重新理解。第一类物品事实不能独自证明第二类解释。

如果只观察到包内饼干，应将 H 相关事实记为有支持，将 G 的归属连接保留未定，并回查前文。若前后证据确实支持对先前互动的重新解释，G 才更切合“最大转折”。H 不必因此被判为假。

无法取得归属证据时，诚实保留缺口，不能用熟悉的饼干故事模板补齐未看见的情节。

### 11.3 E09：音画一致

至少区分三个实验版本：V（视觉），V+T（视觉与允许转写），V+T+A（再加真实音频分析）。

Qwen3-VL-8B-Instruct 是视觉—语言模型，不能把输入带音轨的视频文件等同于模型消费了音频。普通 ASR 不足以判断背景音乐性质。

允许使用冻结工具时，可以增加 Qwen2-Audio-7B-Instruct 等音频模型作为观察工具，而非另一个最终答题模型。工具只接收对应时间的真实音频及中性描述请求，输出速度、力度、音色和基调等有不确定性标注的描述，再交由 Qwen3-VL 比较。该设置增加工具、算力和可能的感知错误，必须单列报告。

视觉侧独立核验人物、动作、表情与环境。两侧来源对齐后才比较一致性，最后检查完整选项，特别是 F/H 的男孩/女孩差别。没有音频能力时将音乐命题保留 unknown，不由画面预测音乐。

## 12. Qwen3-VL-8B 的实现约束

### 12.1 短上下文调用

不让一次调用同时看全片、写人物关系网、输出长解释、规划所有工具并选答案。每个调用只接收当前子任务必要的证据。全量记忆留在程序，模型只获得相关切片。

编译器和关系检查可使用同一 Qwen 的文本输入模式，不必另加纯文本模型。对需要核验的关键结论，不能只给二手摘要；必要时重新交付原始图像或局部视频。

### 12.2 正确处理视频和来源

以固定版本的官方 Qwen3-VL 处理器与 qwen-vl-utils 为准。官方工具示例使用 image_patch_size=16，并传递 video_metadata；不能直接照抄 Qwen2.5-VL 的旧尺寸假设。

若 qwen-vl-utils 已完成 resize，后续处理应避免重复缩放，按官方示例配置 do_resize=False。不要让封装层偷偷再次均匀采样，导致 source_id 指向的帧与模型实际看到的帧不一致。

不等间隔抽帧不能伪装成等间隔视频序列。保存真实时间戳和帧号；若当前适配器不能正确传递非均匀时间，改为多个合法短片或带显式来源时间的独立图像输入，不虚构 fps。裁剪后保留与原帧的对应。

帧数不是完整成本。还需计入分辨率、实际视觉 token、重复编码、补帧、文本输入和输出。以处理器输出和实际模型调用为准，不用统一“每帧若干 token”替代测量。

### 12.3 起始配置（待调优）

```yaml
model_id: Qwen/Qwen3-VL-8B-Instruct
training: false
external_web_at_inference: false

observation:
  initial_overview_frames: 32
  core_window_seconds: 8
  context_seconds_each_side: 2
  fps_default: 2
  fps_for_fast_events: 4
  max_frames_per_visual_call: 48
  crop_frames_per_request: 2
  max_relevant_facts_per_call: 10

context_budget:
  max_total_input_tokens_per_call: 16384
  max_visual_tokens_per_call: 8192
  max_new_tokens_compiler: 1024
  max_new_tokens_observer: 1024
  max_new_tokens_relation_checker: 1536
  max_new_tokens_verifier: 1024

controller:
  max_model_calls_total: 16
  reserve_model_calls_for_verification: 2
  max_refinement_rounds: 4
  max_identical_observation_requests: 1
  schema_repair_attempts_per_call: 1

decoding:
  do_sample: false
```

以上 token 上限是初始工程预算，不是原生上下文上限，也不保证适配某块 GPU。工具模型调用和失败重试也计入总预算。窗口与帧数服从 token 和显存预算；超限应显式拆分或降采样，不能无记录地截断。do_sample=false 用作复现基线，不声称一定比采样准确。

## 13. 提示词契约

这些是提示词骨架；需要与实际 JSON Schema、来源清单和服务接口绑定。它们不是经过验证的最优提示词，也不要把本文原题答案作 few-shot 示例混入测试。

### 13.1 Compiler

```text
任务：把原问题及全部候选编译成待核验的命题与观察需求，不回答问题。
输入：原题、原选项、不可修改的输入协议。
保留：实体、时间、否定、量词、比较结构及原始候选编号。
区分：候选提到的事情与视频已观察到的事情；当前没有视频事实。
输出：answer_operator、target_description、option_claims、
      discriminators、initial_evidence_requests、ambiguities。
不得：将选项写成事实；放宽允许范围；猜测答案；重写含糊题目使其迎合某答案。
```

### 13.2 Observer

```text
任务：只从本次实际提供的来源中记录与观察目标相关的事实或有归属陈述。
输入：中性观察目标、源帧/片段/允许文本、source_id 清单。
每条输出：来源、实体、时段、命题、发言归属、清晰度及未观察原因。
只允许引用本次输入提供的 source_id。
区分：未采样、遮挡、看不清、未提及、明确相反。
不得：选择答案；把常识补全为观察；从候选情节反推事实；把人物自述当成客观真相。
无法确定时返回 unknown，不强制给出完整故事。
```

### 13.3 Relation Checker

```text
任务：使用给定事实和来源比较候选，不新增无来源的事实。
先检查完整候选的必要命题，再检查其是否切合问题目标。
每条关系输出：premises、bridge、support_state、missing_premises、strong_alternatives。
bridge 最多两句，必须指出事实怎样支持关系，不能仅复述结论。
时间先后不能单独证明原因；共现不能单独证明动机或人际关系。
多个候选可能部分或全部为真，不得强行为获胜候选把其他真命题改为假。
缺口必须转成明确的可取证目标；不要只输出“信心不足”。
```

### 13.4 Verifier

```text
任务：回答本次中性核验问题，而不是支持任何先前答案。
输入：核验命题、合法原始证据、来源清单、必要比较语境。
输出：supported / contradicted / unknown，来源引用，尚缺什么。
独立检查人物、时间、陈述归属和关系连接。
没有足够来源就保留 unknown，不用流畅解释填补。
不得：自行获取范围外输入；将同一来源的重复描述视为独立证据。
```

## 14. 控制器伪代码

以下表示控制顺序，不是可直接部署的完整 Python 实现。所有函数应有类型、参数检查、统一成本记账和失败状态。

```text
solve(item, protocol, model, tools):
    validate_protocol_and_remove_forbidden_metadata(item, protocol)
    state = initialize_state(protocol)
    spec = compile_query_with_fixed_model(item.question, item.options)
    validate_compiler_output(spec)
    state.set_spec(spec)

    execute_initial_legal_observation_plan(state)

    while state.has_budget_for_next_step():
        assessment = assess_all_candidates(state.relevant_evidence())
        validate_refs_and_dependencies(assessment)
        state.update_assessments(assessment)

        if assessment.has_sufficient_direct_evidence():
            if direct_claim_check_passes(state):
                return serialize_native_answer(state, EVIDENCE_SUFFICIENT)

        if assessment.ready_for_critical_verification():
            if not state.has_reserved_verification_budget():
                break
            result = verify_without_previous_answer(state.raw_evidence_packet())
            state.update_with_verification(result)
            if evidence_obligations_satisfied(state):
                return serialize_native_answer(state, EVIDENCE_SUFFICIENT)

        gaps = collect_explicit_unresolved_gaps(state)
        proposed_actions = propose_whitelisted_actions(gaps)
        legal_actions = filter_scope_modality_cost_and_repetition(proposed_actions)

        if legal_actions is empty:
            state.stop_reason = classify_blockage(state)
            break

        action = pick_by_lexicographic_priority(legal_actions)
        result = execute_and_account_for_cost(action)
        validate_then_update_sources_and_facts(state, result)

        if no_material_progress_under_remaining_legal_actions(state):
            state.stop_reason = NO_PROGRESS
            break

    return serialize_native_forced_choice_or_allowed_abstention(
        state,
        evidence_status="insufficient",
        stop_reason=state.stop_reason_or_budget_exhausted()
    )
```

预算扣减应发生在实际调用处，失败也扣减。给 verifier 的预算是预留，而不是最后才发现没有额度。原生强制选择阶段不能再偷偷进行一个未计费的大模型调用；复用最近的候选评估或事先保留终局选择预算。

## 15. 开发顺序与复用

第一版优先实现三张表、查询协议、四个提示词角色、完整候选核验、显式缺口和合法观察动作。先以 E08、E10、E14 类结构明确的问题检查机制。

第二步加入直接证据通道与资源性停止，使用 E02、E04、E12 等检查是否过度推理。题意存疑的 E05、E11 单列审查，不能为提高开发集准确率反向编造规则。

第三步再加入真实音频工具和更复杂的跨片段索引。音频是能力扩展，不是简单多写一个 prompt；长视频树结构是规模扩展，不是 R6 的定义。

附件描述的复用关系：R1 提供定位和局部来源；R3 提供场合账本与顺序；R4 提供三值集合运算；R5 提供必要的全局事实覆盖。它们共同使用同一 scope、source manifest 和 budget，不各自维护一套。这里未检查实际仓库代码，不能据文档声明模块已经可直接运行。

## 16. 实验与错误归因

### 16.1 切分与泄漏控制

附件 32 题是有目的选取的讨论材料，不是随机测试集。正式实验按视频或故事划分开发与测试，避免同片不同题跨集合；不能把在这些原题上修改提示词后的结果称为独立泛化成绩。

同视频题间的缓存复用必须服从协议。共享只能是合法获取的事实与来源，不是标准答案、人工推理链或其他题的已知真值。

### 16.2 核心消融

| 实验 | 固定条件 | 比较 |
|---|---|---|
| 中间表示 | 相同模型、相同观察结果与输入模态 | 自由综合 vs 事实表 + 关系表 |
| 主动观察 | 相同初始证据与总预算 | 均匀加帧 vs 明确缺口定向补看 |
| 关键核验 | 相同成本预算 | 只支持当前答案 vs 检查竞争解释及关键跳步 |
| 选项诱导 | 相同问题与观察预算 | 全选项可见 vs 仅问题 vs 中性判别需求 |
| 简单题保护 | 人工确认的直接证据样本 | 直接通道 vs 强制多轮关系推理 |
| 模态能力 | 单独报告输入条件 | 视觉 vs 允许转写 vs 真实音频增强 |

另保留单次 Qwen 直接回答和“逐段描述后统一回答”两类简单基线。单次基线不必伪装成与多调用系统同样调用次数，应同时报告成本曲线；机制消融则严格控制观察或计算变量。

可以增加只给问题/选项的诊断基线，检查数据捷径，但其得分不能证明视频关系机制有效。

### 16.3 指标

结果：原生准确率、各子型分子/分母、直接证据与真正关系样本分层、强制选择比例。Video-MME-v2 的官方分组分数须使用完整组和官方公式，不能拿抽出的 R6 单题代替。

机制：引用是否真支持命题、关键事实覆盖、人物/说话人绑定、关系跳步可支持性、关键竞争解释处理、补看指定缺口的解决率。官方没有解释真值时，使用盲审标准，不能声称计算了官方推理链准确率。

成本：模型/工具调用、视觉与文本 token、独特源帧和重复编码帧、裁剪数量、音频秒数、失败重试、端到端延迟，以及预算耗尽/无进展比例。

建议人审把“最终选项正确”与“证据及关系正确”分开打分。E12 尤其要区分识别猫伤人与解释幽默反差。

### 16.4 错误归因顺序

题目/标注问题 → 解析问题 → 范围/模态问题 → 定位与感知问题 → 事实保留与绑定问题 → 关系判断问题 → 反证与停止问题 → 集合归约/答案映射问题。

当前置事实都不可靠时，不应把错误统称为“8B 推理能力差”；反过来，事实完备仍判断错，也不能期待无限补看自动修复。

## 17. 核心结论

这套设计把额外计算放在三个地方：取得能区分候选的证据，保存从事实到关系的缺口，核查最关键的一步连接。它不依靠增加自由推理长度来冒充能力提升。

最值得先验证的不是大图或多 agent，而是：在固定证据下关系表是否有帮助；在固定预算下缺口取证是否优于均匀加帧；独立核验是否减少没有证据支撑的答案。只有这些实验成立，才有依据继续增加系统复杂度。

## 参考文献与模型资料

[P1] Min, J., et al. **MoReVQA: Exploring Modular Reasoning Models for Video Question Answering.** CVPR 2024. 方法全文：https://arxiv.org/html/2404.06511v1 。正式论文页面：https://openaccess.thecvf.com/content/CVPR2024/html/Min_MoReVQA_Exploring_Modular_Reasoning_Models_for_Video_Question_Answering_CVPR_2024_paper.html 。

[P2] Wang, X., Zhang, Y., Zohar, O., and Yeung-Levy, S. **VideoAgent: Long-form Video Understanding with Large Language Model as Agent.** ECCV 2024. https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10325.pdf 。注意不是其他同名 VideoAgent 工作。

[P3] Wang, Z., et al. **VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos.** CVPR 2025. https://arxiv.org/abs/2405.19209 。

[P4] Choudhury, R., et al. **Video Question Answering with Procedural Programs.** ECCV 2024. https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05543.pdf 。

[P5] Dhuliawala, S., et al. **Chain-of-Verification Reduces Hallucination in Large Language Models.** Findings of ACL 2024. https://aclanthology.org/2024.findings-acl.212/ 。

[P6] Gou, Z., et al. **CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing.** ICLR 2024. https://openreview.net/forum?id=Sx038qxjek ；https://arxiv.org/abs/2305.11738 。

[M1] Qwen3-VL-8B-Instruct 官方模型卡：https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct 。

[M2] Qwen3-VL 官方使用说明：https://github.com/QwenLM/Qwen3-VL 。

[M3] Transformers Qwen3-VL 文档：https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl 。

[M4] 可选冻结音频工具示例，Qwen2-Audio-7B-Instruct：https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct 。

[U1] 用户附件《R6_特征与原题讨论材料(1).md》：第 1 节定义和边界；第 3 节 S1—S6；第 4 节 E01—E16；第 6 节概念流程和复用背景；第 7 节诊断实验与评估限制。原题答案属于离线对照，不是运行时证据。
