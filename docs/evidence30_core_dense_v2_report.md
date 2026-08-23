# Evidence30 Oracle-failure core-frame 密度诊断（v2）

## 结论

本实验只在官方 Oracle-context ablation 仍答错的 9 条 dev 上运行，不读取 `locked.jsonl`。
Qwen3-VL-2B 的 18 次推理全部正常完成；两路 relaxed grounded rate、reference-level core
item recall 和 chosen-set core recall 都是 100%。

更密集的 core-frame 观察有真实但不稳定的作用：

- `core_16` 从 9 条既有 Oracle 错题中恢复 1 条（11.1%）。
- 匹配视觉预算的 `core_32` 恢复 2 条（22.2%）。
- `core_32` 对 `core_16` 的逐题配对是 2 胜、1 负、6 平；不是单调改进。
- 平均实际帧数从 14.56 增至 24.56（+68.7%），平均 input tokens 仅从 1407.2
  增至 1504.7（+6.9%），平均推理耗时增加 2.1%。

所以，当前失败不能只归因于搜索没有命中时间区间；在证据时间覆盖已由 Oracle 控制后，
2B 的细粒度视觉/OCR、跨槽关系合成和证据干扰仍是主要限制。与此同时，`730-1` 的恢复
证明采样与槽位分配机制确实至少解释了一部分失败。

## 实验定义

选择规则固定读取 `runs/evidence30/ablations_v2/items.jsonl` 中满足以下条件的记录：

1. `packet_source == oracle_context`；
2. 策略为官方 `evidence30-oracle-unified-head/1.1`；
3. 题号属于 manifest 的 18 条 dev；
4. 推理完成、relaxed grounded、core item recall 为 100%；
5. `correct == false`。

代码断言结果必须恰好为 9 条。dev-only loader 只打开 `manifest.json` 和 `dev.jsonl`，不打开
`locked.jsonl`。

两路使用同一个确定性 2B 模型和统一答题头：

- `core_16`：最多 16 帧，单帧最多 131,072 像素。
- `core_32`：最多 32 帧，单帧最多 65,536 像素。
- 两路最大源帧像素预算均为 2,097,152；字幕、prompt 和生成参数相同。
- 视觉/OCR 与字幕使用各自的 core interval。帧按 required evidence item 均衡分配，
  不先把重叠区间合成一个大窗口，避免 global 槽淹没局部槽。

这里的 `core_32` 是“固定视觉预算下，更多低分辨率时间帧”的联合干预，不是无限制增加算力。

## 正式 v2 结果

| question | answer | Oracle context | core-16 | core-32 | frames 16→32 | 解释 |
|---|---:|---:|---:|---:|---:|---|
| 102-1 | B | C | C | **B** | 7→7 | 帧集合不增，只降低空间分辨率后翻对；是预处理/分辨率敏感，不是时间密度收益 |
| 434-2 | A | B | B | B | 16→24 | OCR/技能识别持续失败 |
| 496-1 | D | B | C | C | 16→32 | 预测会随采样变化，但始终不能合成正确事件顺序 |
| 604-2 | C | B | B | B | 16→19 | 字幕 core 已完整暴露，仍选择竞争选项 |
| 647-2 | B | A | A | A | 16→31 | 更多局部视觉帧未压过错误语义/先验 |
| 700-1 | D | C | C | C | 16→32 | 同一错误稳定存在 |
| 730-1 | B | C | C | **B** | 16→32 | local 制作片段与 global 排除槽均获独立配额后恢复 |
| 845-1 | B | C | **B** | A | 16→32 | 16 帧恢复，但更多帧反而引入序列合成干扰 |
| 884-2 | A | B | B | B | 12→12 | 1 fps core 内只有 12 个 OCR 候选帧，密度实际上未增加 |

| variant | complete | recovered | grounded | core recall | mean frames | mean input tokens | mean wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| core-16 | 9/9 | 1/9 | 9/9 | 100% | 14.56 | 1407.2 | 0.807 s |
| core-32 | 9/9 | 2/9 | 9/9 | 100% | 24.56 | 1504.7 | 0.824 s |

## 因果拆分

### 1. 机制问题确实存在，但不是唯一瓶颈

`730-1` 在预算不匹配的探索版 core-32 和正式匹配预算的 core-32 中都从 C 恢复为 B。
这条题含一个局部制作过程槽和一个全片排除槽；按 item 保留独立配额、增加时间覆盖后结果稳定
改善。因此，原先把区间合并并按总时长采样会稀释局部证据，这是可定位的机制问题。

### 2. “多看帧”会产生证据干扰

`845-1` 在 core-16 中从 C 恢复为 B，但两个 core-32 版本都变为 A。四个珠宝阶段本身都被
完整覆盖，问题出在模型如何把更多局部观测合成为一个顺序，而不是是否接触过标注区间。
这说明下一版不应简单把更多帧一次性塞给答题头；应先按 slot 独立形成小结，再做受约束的关系合成。

### 3. 分辨率与时间密度存在交互

`102-1` 的 core-16 和 core-32 都只有同样 7 帧；高分辨率探索版仍答 C，正式低分辨率版却答 B。
这个翻转不能算作“更多帧”的功劳，只能说明 2B 对视觉 tokenization/尺度很敏感。OCR 题尤其
不能默认用降低空间分辨率换取更多帧，应该把关键 OCR 帧作为高分辨率 still image 单独输入。

### 4. full temporal exposure 不等于模型读懂证据

`434-2`、`604-2`、`647-2`、`700-1`、`884-2` 在 Oracle context、core-16、两种 core-32
构造中都稳定答错；`496-1` 虽然选项振荡，也始终错误。这些记录已经把“时间区间没找对”的
解释大幅压缩，剩余候选主要是：

- 2B 无法识别细粒度动作、HUD/OCR 或人物属性；
- 能看到各槽但不能正确合成先后、排除或跨片段关系；
- 模型先验压过视频证据；
- 个别 benchmark 题/AI-assisted reference 自身存在歧义。

relaxed grounding 只证明对应模态的帧或字幕时间段被输入，不证明像素清晰、OCR 可读或语义已被
模型正确抽取，因此不能把 100% grounding 解读为 100% semantic evidence。

## v1 作废说明

第一次探索运行使用 `core_32.max_pixels=131072`。虽然 YAML 中两路 `total_pixels` 相同，
`qwen-vl-utils` 对帧列表使用 temporal `FRAME_FACTOR=2`，导致 core-32 平均 input tokens 达
2187.3，而 core-16 只有 1407.2。该运行放大了视觉预算，违反原定 matched-budget 设计，保留在
`runs/evidence30/core_dense_v1` 仅用于定位预算语义，不作为正式结论。正式策略升级为
`evidence30-oracle-core-density/1.1` 并完整重跑。

## 完整性与产物

- 正式 run signature：`53b619f428c04c78653e3bc26e89cb72dae439ab1ae649fe954ccec55886f5eff`
- preflight SHA-256：`5b63a3a42193d34b28993fc34c752d4ba65e77ba10b8fd88a608f79ec755793f`
- items SHA-256：`44559636dda3bf33a641c06af424a0e4d13c029b7da527223e9bc6cea9f1170e`
- summary SHA-256：`51ea5dd26a97917a213450cf38b449e9389196f541c14293ebb2f038657d02ac`
- run manifest SHA-256：`f9b46abec6fec94fe0ba2aa18ffd377a7c1697d59ad567ca04ee3d38dcbbe156`
- baseline ablation items SHA-256：`04eb488115d94c13125ab0e15b48a986cc9394d173876afdb635ee44d95a0726`
- dev reference SHA-256：`ec39b9e486c111e5cad8b77ec71388feda48af54f8e589d9157043a8e3f0f784`
- 完成：18/18；fatal error：0；degraded fallback：0；locked records read：false。

本实验是对已知 9 条错误的条件诊断，不是独立准确率估计；标注仍是 AI-assisted internal
reference，不是人工 gold。
