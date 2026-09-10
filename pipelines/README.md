# Harness：九类视频推理 Pipeline

基于冻结的 Qwen3-VL-8B-Instruct，按题目需要的证据结构组织视频取证与推理。仓库提供 **R1–R9 的当前实现、运行配置、CPU 回归测试和 benchmark 具体题型映射**。不训练模型，也不包含权重或 benchmark 视频。

这里的 R1–R9 是项目的执行分类，并非九种官方 benchmark 标签。**先判断实际题意，再选择策略；没有自动将整个原生标签硬路由到某一类的分类器。**

## 当前版本

| Pipeline | 版本 | 主要任务 | CLI 策略 |
|---|---|---|---|
| [R1](docs/pipelines/r1.md) | 3.1 | 定向定位与局部取证 | `r1` |
| [R2](docs/pipelines/r2.md) | 1.5 | 连续动态与实体状态追踪 | `r2` |
| [R3](docs/pipelines/r3.md) | 5.4 | 查询驱动时间证据 | `r3` |
| [R4](docs/pipelines/r4.md) | 5.7 | 清单构建与集合归约 | `r4` |
| [R5](docs/pipelines/r5.md) | 3.1 | 分段全局综合 | `r5` |
| [R6](docs/pipelines/r6.md) | 1.0 | 证据约束关系推理 | `r6` |
| [R7](docs/pipelines/r7.md) | 1.0 | 假设与未来推演 | `r7` |
| [R8](docs/pipelines/r8.md) | 1.0 | 视觉符号与约束求解 | `r8` |
| [R9](docs/pipelines/r9.md) | 1.0 | 空间场景建模与查询 | `r9` |

`r1` 与 `r1-v3` 都运行 **R1 3.1**，沿用 `r1_v3` 配置／trace 命名空间。其余旧版独立策略已从统一入口移除。源码中的 `r1/`、`r1_v2/`、`p01/`、`coarse_to_fine/` 保留的是当前实现仍需的基类和工具，不是额外发布版本。版本与代码校验值见 [发布清单](release_manifest.json)。

## Pipeline 对应哪些 benchmark 题型

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

例如，**R1 对应 Video-MME 的局部 OCR、普通物体识别、直接动作或属性读取**；但 Object Recognition 中“第二个制作的纸动物”需要事件排序，主 R3；“哪些物品没有被讨论”需要范围清单与否定核验，主 R4。

Counting Problem 也必须细分：**事件次数 → R3，不同实例／类别数量 → R4，读取视频明示数量 → R1**。同一原生标签可能存在多个执行子型。

完整说明见 [九类与 benchmark 题型对应说明](docs/benchmark_mapping.md)，其中包含 11 个 benchmark 的主要对应关系、Video-MME 全部 12 类专项对照、通常／条件／待确认标记及来源。Video-MME 与 Video-MME-v2 分开处理；映射不代表已完成全部数据适配或效果验证。

## 安装

本文及各类运行说明中的命令均在仓库的 `pipelines/` 目录执行。

统一使用 **Python 3.11**。真实模型推理需要按配置准备 CUDA GPU 和 Qwen3-VL 权重；CPU 可运行控制逻辑、合成媒体和入口检查。

```bash
git clone https://github.com/3314787995/harness.git
cd harness/pipelines
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,eval,r8]"
```

仅做 CPU 工程验证、避免安装模型推理依赖时：

```bash
python -m pip install -e . --no-deps
python -m pip install -r requirements-cpu.txt
python -m pytest -q
python tools/check_release.py
```

各类模型配置保留现有 attention 与预算设置。配置使用 FlashAttention 2 时，需另行准备与本机 CUDA／PyTorch 匹配的环境；改用 `sdpa` 时在本地配置中明确设置并记录，不会在本文中宣称两种条件等价。

## 配置模型与运行

模型路径优先使用已有本地快照，机器路径通过环境变量传入。下面是 PowerShell 示例：

```powershell
$env:QWEN3VL_MODEL_PATH = "D:/models/Qwen3-VL-8B-Instruct"
$env:QWEN3VL_CACHE_DIR = "D:/cache/harness"
qwen3vl-agent --strategy r1 --config configs/r1_8b.yaml --video path/to/video.mp4 --query "What word is displayed on the sign?" --choice "OPEN" --choice "CLOSED" --trace-output runs/r1/trace.json
```

Linux 可用 `export QWEN3VL_MODEL_PATH=/path/to/model` 设置同一变量。不设置时使用配置中的模型 ID；首次真实运行可能下载权重。R6–R9 等配置若指定模型 revision，本地快照也应与指定版本对应。

九类均通过 `--strategy r1` 至 `r9` 与对应的 `configs/rN_8b.yaml` 选择；题目、原选项及媒体由运行者提供。每类的输入权限、补充参数与批量入口见上方运行文档。不存在必须安装 benchmark 视频才能运行的隐含项目绝对路径。

```bash
python -m qwen3vl_agent.cli --help
python -m qwen3vl_agent.r6.evaluate --help
python -m qwen3vl_agent.r8.evaluate --help
```

**时间范围和模态是输入协议的一部分。** `allowed_scope`／`allowed_intervals` 限制能看的媒体，`query_scope`／`reference_scope` 指所问事件的范围，二者不可混用。字幕、ASR 必须按原任务授权和时间对齐；已有转写不能替代音乐证据。未来预测必须确认观察截止，不得从题型标签猜测截止时刻。

## 答案、证据与 Trace

统一 CLI 打印答案；`--trace-output` 保存 metadata，`--show-metadata` 额外显示完整记录。各类沿用自身结果字段和命名空间，详细字段见对应运行文档与公开 Request／Result 类型。

答案、证据充分性、执行停止原因分别记录。非空预测或合法选项不代表语义已验证；预算耗尽、缺失模态、未知证据和协议失败不能当作零次、空集合或不存在。批量评测中参考答案放在评分侧，不作为推理请求的证据。

## 验证状态与能力边界

发布版的实测检查、原始快照对照和已知失败见 [发布验证记录](docs/validation.md)。CI 在 Python 3.11 中运行当前测试并如实报告失败，不把原有失败改成跳过来制造通过结果。

R6 已有 1.0 实现及 CPU 验收记录，**真实 GPU smoke 尚未执行**，真实音频 Provider 尚未启用。本次发布不启动付费 GPU 或完整 benchmark 评测；其他版本的历史小样本成绩也不作为本快照的真实效果结论。

## 目录

```text
qwen3vl_agent/           九类实现与必需内部公共代码
configs/                当前配置（r1_v3 配置为 r1 的兼容别名）
docs/pipelines/          每类一份当前运行文档
docs/benchmark_mapping.md
docs/sources/           分类来源；不是额外运行版本
examples/               小型开发请求，媒体另行提供
tests/                  当前机制回归与共享测试辅助代码
tools/check_release.py  无模型发布完整性检查
release_manifest.json   协议版本、发布基点、代码 SHA-256
```

旧研究内容可从 Git 历史查阅。当前目录不包含旧上传包、视频、模型权重或运行日志。
