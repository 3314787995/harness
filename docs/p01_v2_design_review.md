# P01 v2 设计评审清单

这份文档用于和老师确认“P01 应该怎样定义和执行”，不是优化方案定稿。下面先区分三类
内容：当前代码事实、已有实跑现象、待确认的设计选择。

## 当前代码事实

1. 上游已经知道题目属于 P01；v2 不负责路由。
2. ObservationCompiler、首轮 Locator 和 CandidateScout 对选项盲。
3. 选定一个局部 span 后，HypothesisCompiler 才读取选项并生成去标签判别声明。
4. routine refinement 最多一次；InitialDecision 后至多再执行一次 bounded rescue。
5. rescue 可以看去标签判别声明，用全局 overview 重定位，再对最多两个局部候选做 scout。
6. 不相交的局部段只能竞争，不能拼接为证据。
7. MCQ 无论证据等级如何都必须返回合法选项；必要时走确定性回退。

## 现有 25 题诊断结果

2026-09-02 在 Qwen3-VL-8B 上完成了一轮内部 smoke：25/25 题完成，20/20 MCQ 有合法
输出，5/5 G39 有非空输出；MCQ 为 11/20。这个样本只覆盖 Video-MME、MLVU、
LVBench 中人工选出的五类探针，不能当正式 benchmark 或泛化结果。

该轮有 7 道 MCQ 出现 CUDA OOM。为避免把资源故障混入流程判断，下面只分析另外 13 道
无 OOM 的 MCQ：其中 10 道正确、3 道错误。16/20 MCQ 的 HypothesisCompiler 最终退化为
确定性 option wrapper；在无 OOM 的 13 道中仍有 11 道发生该退化，说明它不是显存问题。

原始结果保存在本地归档，不提交 Git，因为逐题 trace 含绝对服务器路径且体积很大。
本评审分支相对实跑快照新增了 Decision 输入帧数上限（默认 64，safe profile 为 48）以及
发布整理，但尚未据此重跑，因此上面的数字不是本分支的新准确率声明。

## 三道无 OOM 错题说明了什么

| 题目 | 实际发生的流程 | 暴露的问题 |
|---|---|---|
| `Video-MME::599-1`（闹钟时间，OCR） | 首轮被判为 `dynamic_action`；rescue overview 在约 2.50s 已写出可见时间，但后续 OCR scout 没保住这个锚点，最终得到空 facts 并选“未提及” | 全局定位观察不能直接成为证据；OCR 的二次采样/排序会丢掉已经命中的小字帧 |
| `Video-MME::573-3`（人在袜子上行走时狗的动作） | 首轮候选未见目标；rescue 定位文本已出现“dog lying”，但 weak candidate 被强制选中，最终回答“sitting” | 跨阶段语义冲突没有进入统一账本；`no_local_anchor` 后仍允许弱包进入 FinalDecision |
| `Video-MME::197-1`（衣服图案） | 定位和主体都正确，证据只描述绿、红、金等外观；HypothesisCompiler 退化后，decision 把这些泛化成“tree”，并给 evidence `strong` | “看清主体/颜色”被误当成“看清选项判别属性”；证据等级只检查槽位覆盖，不检查是否真的区分选项 |

三题共同指向同一个结构性缺口：早期阶段负责“找什么、看什么”，但直到局部段冻结后才
知道选项之间究竟差在哪；一旦早期采样没有覆盖那些判别细节，后面的 refinement/rescue
往往只能在错误或过宽的观察目标上继续。

## 建议直接问老师的七个问题

1. **选项盲的边界在哪里？** 是否要求 Locator 和首次 Scout 都严格看不到选项，还是只需
   Locator 盲、局部 Scout 可以读取去标签判别条件？
2. **P01 是否必须是单连续局部段？** 如果两个不相交时刻共同决定答案，应继续判为 P01，
   还是应交给另一类 pipeline？
3. **选项判别应何时进入？** 当前在候选 span 选定后才编译 discriminants；是否应该在
   ObservationSpec 阶段就生成“要区分的视觉属性”，但仍隐藏 option label？
4. **弱证据是否仍必须作答？** 当前 benchmark 覆盖率优先，`weak/none` 也必须输出；老师
   是否需要区分“系统预测”和“证据支持的预测”？
5. **rescue 的输入能否继承全局定位观察？** 当前 overview 只能定位，不能把已读到的文字或
   动作写进 EvidencePacket；这条隔离是否是硬要求？
6. **evidence grade 的语义是什么？** 它应只表示目标和槽位可见，还是必须证明已经覆盖各
   选项的关键判别属性？
7. **协议退化是否可接受？** HypothesisCompiler 大量退化为 deterministic wrapper 时，
   应继续执行、重写 prompt/schema，还是把该题标记为 pipeline degradation？

## 老师确认前不建议混在一起改的内容

- 不先把 v3 的高预算、多卡或其他题型逻辑并入 v2；
- 不用扩大采样量掩盖“判别条件进入太晚”的结构问题；
- 不把 OOM 与三道无 OOM 错题归为同一种失败；
- 不把 25 题准确率写成模型或方法效果结论。

老师确认上述定义后，再决定是保持严格 choice-blind 设计并加强证据闭环，还是调整选项
信息进入时机。这样后续改动才是在实现要求，而不是凭三道题反向拟合。
