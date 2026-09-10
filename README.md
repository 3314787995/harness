# Harness：九类视频推理 Pipeline

R1–R9 的当前代码、共享依赖、配置、运行脚本、示例、测试和文档统一放在 **[pipelines/](pipelines/)**。

- **[安装与运行](pipelines/README.md)**：模型与媒体配置、命令、结果和 trace。
- **[Benchmark 题型对应](pipelines/docs/benchmark_mapping.md)**：11 个 benchmark、Video-MME 全部 12 类及条件分流。
- **[验证记录](pipelines/docs/validation.md)**：CPU 检查、已知失败和验证范围。

## 九类入口

| Pipeline | 版本 | 主要任务 | CLI 策略 |
|---|---|---|---|
| [R1](pipelines/docs/pipelines/r1.md) | 3.1 | 定向定位与局部取证 | `r1` |
| [R2](pipelines/docs/pipelines/r2.md) | 1.5 | 连续动态与实体状态追踪 | `r2` |
| [R3](pipelines/docs/pipelines/r3.md) | 5.4 | 查询驱动时间证据 | `r3` |
| [R4](pipelines/docs/pipelines/r4.md) | 5.7 | 清单构建与集合归约 | `r4` |
| [R5](pipelines/docs/pipelines/r5.md) | 3.1 | 分段全局综合 | `r5` |
| [R6](pipelines/docs/pipelines/r6.md) | 1.0 | 证据约束关系推理 | `r6` |
| [R7](pipelines/docs/pipelines/r7.md) | 1.0 | 假设与未来推演 | `r7` |
| [R8](pipelines/docs/pipelines/r8.md) | 1.0 | 视觉符号与约束求解 | `r8` |
| [R9](pipelines/docs/pipelines/r9.md) | 1.0 | 空间场景建模与查询 | `r9` |

## Benchmark 题型速查

| Pipeline | 主要原生入口及具体子型（精选） |
|---|---|
| R1 | Video-MME / OCR Problems（局部文字）、Object Recognition（普通目标）、Attribute Perception（直接属性） |
| R2 | MVBench / Action Antonym、Object Shuffle；TOMATO / Rotation、Direction |
| R3 | MVBench / Action Count、Character Order；Video-MME / Temporal Perception（事件顺序、时长）；Video-MME-v2 / Repetitive Action Counting |
| R4 | Video-MME / Counting Problem（实例数）、Object Recognition（未讨论项）；VSI-Bench / Object Count |
| R5 | Video-MME / Information Synopsis（事实综合）；MLVU / Video Summarization；EgoSchema / Long-form Video QA（整体行为） |
| R6 | Video-MME / Action Reasoning（非明示原因）；Video-MME-v2 / Causal Reasoning、Symbolic/Metaphorical Interpretation |
| R7 | MVBench / Action Prediction、Counterfactual Inference；Video-MME-v2 / Future Event Prediction、Counterfactual Reasoning |
| R8 | Video-MME-v2 / Numerical Calculation；EgoLifeQA / EntityLog（由总价和数量求单价） |
| R9 | VSI-Bench / Relative Direction、Relative Distance、Route Plan；Video-MME-v2 / Spatial Understanding（参考系、跨视角子型） |

以上是任务机制对应，需按具体证据需求分流，不代表已完成所有 benchmark 的适配或效果验证。完整条件与来源见[题型对应说明](pipelines/docs/benchmark_mapping.md)。

## 开始使用

```bash
git clone https://github.com/3314787995/harness.git
cd harness/pipelines
python -m venv .venv
# 激活虚拟环境后，按 pipelines/README.md 安装依赖并运行。
```

后续安装、运行和测试命令都在 `pipelines/` 中执行。`r1` 与 `r1-v3` 同为 R1 3.1，其余策略为 `r2` 至 `r9`。

```text
harness/
├── README.md                 仓库导航
├── pipelines/                九类 pipeline 的全部内容
│   ├── README.md             安装与运行总说明
│   ├── qwen3vl_agent/        R1–R9 与必要共享代码
│   ├── configs/              当前配置
│   ├── docs/                 各类说明、题型映射及来源
│   ├── scripts/              运行脚本
│   ├── examples/             小型示例
│   ├── tests/                回归测试
│   ├── tools/                检查工具
│   ├── pyproject.toml        安装入口
│   ├── requirements-*.txt    依赖清单
│   └── release_manifest.json 版本与文件校验值
└── .github/workflows/        GitHub CI
```

当前 CPU 回归结果为 **938 通过、60 项已知失败**；全量 lint 也存在已记录问题。未执行真实 GPU smoke 或完整 benchmark 测评，详见[验证记录](pipelines/docs/validation.md)。旧研究内容保留在 Git 历史。
