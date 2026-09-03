# Qwen3-VL Video Agent

[![CI](https://github.com/3314787995/harness/actions/workflows/ci.yml/badge.svg)](https://github.com/3314787995/harness/actions/workflows/ci.yml)

这是一个面向长视频推理机制研究的 Qwen3-VL 工程仓库。它把同一个本地
Qwen3-VL 模型封装成五条可比较的推理路径，并提供 Video-MME 小样本评测、
Evidence30 证据标注、冻结协议、消融实验和可回放 trace。

当前重点是验证“如何找到、记录并验证长视频证据”，不是训练模型，也不宣称复现
任何论文的训练过程或公开指标。模型权重、Video-MME 数据和批量运行产物不在仓库中。

## 先看结论

| 路径 | 解决的问题 | 实现位置 |
|---|---|---|
| `direct` | 单次模型回答；评测时可用有界均匀帧基线 | `qwen3vl_agent/cli.py`、`evaluation/runtime.py` |
| `tools` | 先规划工具调用，再把工具结果作为补充证据回答 | `qwen3vl_agent/agent.py` |
| `coarse_to_fine` | 先粗看全片，再定位窗口并逐轮细化 | `qwen3vl_agent/coarse_to_fine/` |
| `active_tree` | 用场景树、证据契约、原子账本和双验证器主动搜索 | `qwen3vl_agent/active_tree/` |
| `p01` | 对已知 P01 单视频题做选项盲定位、单连续局部取证和一次有界补救 | `qwen3vl_agent/p01/` |

Evidence30 不是线上推理策略，而是一套机制诊断工具：

- 30 条 AI-assisted internal reference，18 条 dev、12 条 locked；
- direct / coarse-to-fine / active-tree 的统一运行和证据暴露评分；
- 统一答题头、Oracle context 与 core-frame 密度诊断；
- 配置、标注、运行签名和冻结/密封约束。

截至 2026-08-23，控制逻辑有 79 个不加载模型的单元测试。当前真实实验结论、失败项和
不可宣称内容见 [当前状态](docs/current_status.md)。

## 五分钟上手

要求 Python 3.10+。真实推理通常需要 CUDA GPU；CPU 只适合静态检查、单元测试和
无模型 preflight。

```powershell
git clone https://github.com/3314787995/harness.git
cd harness
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,eval]"
```

默认配置使用 Hugging Face 模型 ID `Qwen/Qwen3-VL-2B-Instruct`。如果模型已下载到
本地，推荐通过环境变量覆盖，而不是修改版本化配置：

```powershell
$env:QWEN3VL_MODEL_PATH = "D:\models\Qwen3-VL-2B-Instruct"
$env:VIDEOMME_ROOT = "D:\videomme"
$env:QWEN3VL_CACHE_DIR = "D:\videomme\cache\qwen3vl_agent"
Copy-Item configs\local.example.yaml configs\local.yaml
```

`configs/local.yaml` 已被 Git 忽略，适合保存机器专属路径。未设置
`VIDEOMME_ROOT` 时，评测命令默认按下面的仓库相对布局查找：

```text
data/videomme/
├── videomme/test-00000-of-00001.parquet
├── videos/<video_id>.mp4
└── subtitle/<video_id>.srt
```

先跑无需模型的质量门：

```powershell
python -m ruff check qwen3vl_agent tests scripts tools examples
python -m compileall -q qwen3vl_agent
python -m pytest -q
```

再做一次真实 direct 推理：

```powershell
qwen3vl-agent `
  --config configs\local.yaml `
  --strategy direct `
  --video path\to\example.mp4 `
  --query "What happens in this video?"
```

## 常用命令

工具调用基线：

```powershell
qwen3vl-agent `
  --config configs\local.yaml `
  --strategy tools `
  --video path\to\example.mp4 `
  --query "How long is this video?" `
  --show-metadata
```

粗到细多选推理：

```powershell
qwen3vl-agent `
  --config configs\local.yaml `
  --strategy coarse_to_fine `
  --video path\to\example.mp4 `
  --subtitle path\to\example.srt `
  --query "What are the people arguing about?" `
  --choice "Option A" `
  --choice "Option B" `
  --choice "Option C" `
  --choice "Option D" `
  --trace-output runs\manual_trace.json
```

Active Evidence Tree 推理与 HTML 回放：

```powershell
qwen3vl-agent `
  --config configs\local.yaml `
  --strategy active_tree `
  --video path\to\example.mp4 `
  --subtitle path\to\example.srt `
  --query "What happens before the final event?" `
  --choice "Option A" `
  --choice "Option B" `
  --choice "Option C" `
  --choice "Option D" `
  --trace-output runs\active_tree_trace.json `
  --replay-output runs\active_tree_replay.html
```

P01 v2 局部直接取证（纯视觉、单视频、无字幕）：

```powershell
qwen3vl-agent `
  --config configs\p01_8b.yaml `
  --strategy p01 `
  --video path\to\example.mp4 `
  --query "What time is shown when the alarm is turned off?" `
  --choice "8:24" `
  --choice "9:24" `
  --choice "6:24" `
  --choice "Not shown" `
  --trace-output runs\p01_trace.json `
  --show-metadata
```

不传 `--choice` 时按 G39 自由文本处理；有明确时间范围时可加
`--given-interval START END`。P01 v2 对有效 MCQ 必须给出一个合法选项，证据等级只用于
诊断和触发最多一次 bounded rescue，不作为拒答门。完整返回结构位于
`metadata["p01"]`。

P01 当前是供设计评审的 v2 候选版本。先读
[P01 v2 导览](qwen3vl_agent/p01/README.md)，再看
[完整流程与不变量](docs/p01_v2_pipeline.md)；给老师确认的关键分歧和脱敏诊断结果集中在
[设计评审清单](docs/p01_v2_design_review.md)。租卡复现步骤见
[4090D 运行手册](docs/p01_rental_runbook.md)。

Video-MME 指定题目评测：

```powershell
qwen3vl-videomme `
  --config configs\local.yaml `
  --question-id 102-2 `
  --strategy active_tree `
  --with-subtitles `
  --output runs\active_tree_debug.json `
  --trace-jsonl runs\active_tree_debug.jsonl `
  --replay-dir runs\active_tree_replays
```

Evidence30 的完整命令、冻结顺序和 locked 约束见
[调试协议](docs/evidence30_debug_protocol.md)。三个诊断入口分别是：

```powershell
qwen3vl-evidence30 preflight
qwen3vl-evidence30-ablate preflight
qwen3vl-evidence30-core-dense preflight
```

## 仓库地图

```text
qwen3vl_agent/             Python 包
├── models/                Qwen3-VL 模型适配层
├── tools/                 工具协议、注册表和默认工具
├── coarse_to_fine/        粗到细检索、缓存、字幕对齐和预算
├── active_tree/           场景树、证据账本、主动观察与验证
├── p01/                   P01 v2 局部定位、类型化证据、决策与 bounded rescue
└── evaluation/            Video-MME、Evidence30、消融与 core-density

configs/                   可版本化实验配置与本地配置模板
annotations/               冻结的 Evidence30 参考记录
schemas/                   标注 JSON Schema
tests/                     不加载真实模型的控制逻辑测试
scripts/                   小范围人工 smoke 工具
tools/                     标注构建等维护脚本
docs/                      架构、接手说明、协议和实验报告
results/                   去除本机路径后的精选机器可读摘要
```

详细职责和调用关系见 [架构说明](docs/architecture.md)。新接手者建议按
[上手与维护指南](docs/onboarding.md) 的阅读顺序走一遍。

## 版本化与本地产物边界

仓库会提交：

- 源码、配置、测试、Schema 和设计文档；
- `annotations/videomme_evidence30/0.2.0` 冻结参考；
- `results/` 中去除绝对路径后的精简结果。

仓库不会提交：

- 模型权重、Video-MME 视频/字幕/原始 parquet；
- `runs/`、`tmp/`、帧缓存、HTML 回放和逐题大 trace；
- P01 的服务器原始结果、绝对路径、运行日志和租卡环境信息；
- `annotations/_unreviewed_ai_drafts`；
- `configs/local.yaml`、环境变量或凭据。

## 重要边界

- Evidence30 记录是 AI-assisted internal reference，不是独立人工 gold。
- 当前样本规模很小，准确率差异只用于机制诊断，不是稳健效果结论。
- Active-tree 的 verified rate 仍低，不能把“最终选项正确”等同于“证据链已验证”。
- P01 v2 的 25 题结果仅用于流程诊断；它不是正式 benchmark，也不能用于声明泛化提升。
- locked v1 已执行，且 coarse-to-fine 出现一次 degraded fallback；详见状态报告。
- `active_tree/agent.py` 仍是大文件。首版发布保留行为稳定性，暂不做高风险拆分。

## 许可证

本仓库目前没有开源许可证。公开可见不等于获得复制、修改或再分发许可；复用前请联系
仓库所有者确认授权。
