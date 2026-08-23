# 新接手者上手与维护指南

目标是让你在第一天完成三件事：知道项目在研究什么、能跑完控制逻辑测试、能定位一次真实
推理失败发生在“检索、协议、感知、验证”中的哪一层。

## 第一天的建议顺序

1. 阅读 [README](../README.md) 的项目边界和四条路径。
2. 阅读 [当前状态](current_status.md)，先知道哪些结果有效、哪些结论不能说。
3. 阅读 `models/base.py` 和 `models/qwen3vl.py`，理解模型接口。
4. 阅读 `cli.py`、`agent.py`，理解 direct/tools。
5. 按 `config → types → cache/prompts → agent` 的顺序阅读
   `coarse_to_fine/`。
6. 先读 [Active Tree 设计](active_tree_v02_design.md)，再读
   `active_tree/types.py`、`scene_tree.py` 和 `agent.py`。
7. 最后读 `evaluation/` 和 [Evidence30 调试协议](evidence30_debug_protocol.md)。

不要从 `active_tree/agent.py` 第一行硬啃到最后一行。先掌握 TaskContract、
SceneTree、EvidenceLedger 和 ResourceLedger 四个对象，再跟 `generate → _run`
主流程。

## 1. 本地环境

### Python 与虚拟环境

项目要求 Python 3.10+，当前本地验证使用 Python 3.11。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,eval]"
```

Linux/macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,eval]"
```

基础依赖包含 Torch、Transformers、qwen-vl-utils、PyAV 和 Accelerate；`eval` extra
增加 PyArrow 与 JSON Schema；`dev` extra 增加 pytest 与 Ruff。

### 本机配置

版本化配置支持 `${NAME:-fallback}`。推荐设置三个环境变量：

| 变量 | 示例 | 说明 |
|---|---|---|
| `QWEN3VL_MODEL_PATH` | `D:\models\Qwen3-VL-2B-Instruct` | 本地模型或 Hugging Face ID |
| `VIDEOMME_ROOT` | `D:\videomme` | 数据集根目录 |
| `QWEN3VL_CACHE_DIR` | `D:\videomme\cache\qwen3vl_agent` | 帧与场景树缓存 |

PowerShell：

```powershell
$env:QWEN3VL_MODEL_PATH = "D:\models\Qwen3-VL-2B-Instruct"
$env:VIDEOMME_ROOT = "D:\videomme"
$env:QWEN3VL_CACHE_DIR = "D:\videomme\cache\qwen3vl_agent"
Copy-Item configs\local.example.yaml configs\local.yaml
```

Bash：

```bash
export QWEN3VL_MODEL_PATH=/models/Qwen3-VL-2B-Instruct
export VIDEOMME_ROOT=/data/videomme
export QWEN3VL_CACHE_DIR=/data/cache/qwen3vl_agent
cp configs/local.example.yaml configs/local.yaml
```

`configs/local.yaml` 不提交。实验配置应提交，但机器路径必须写成环境变量或相对路径。

### 数据布局

`VIDEOMME_ROOT` 下默认需要：

```text
<VIDEOMME_ROOT>/
├── videomme/test-00000-of-00001.parquet
├── videos/
│   └── <video_id>.mp4
└── subtitle/
    └── <video_id>.srt
```

模型权重和数据集都不在 Git 仓库中。不要把它们复制进项目目录后强制提交。

## 2. 先验证控制逻辑

```powershell
python -m ruff check qwen3vl_agent tests scripts tools examples
python -m compileall -q qwen3vl_agent
python -m pytest -q
```

发布时基线是 79 passed。测试全部使用 FakeModel、临时帧或小型 parquet，不加载真实模型。
GitHub Actions 运行同一组轻量门禁。

测试与模块对应关系：

| 测试 | 保护的行为 |
|---|---|
| `test_model_messages.py` | 多模态消息注入 |
| `test_tools.py` / `test_agent.py` | 工具注册、计划、执行与降级 |
| `test_coarse_to_fine.py` | 窗口、预算、投票、缓存和协议 |
| `test_active_tree.py` | 场景树、证据槽、动作合法性、落账和验证 |
| `test_annotation_schema.py` / `test_evidence30_artifacts.py` | Schema、哈希、split 和标注一致性 |
| `test_evidence30_evaluation.py` | exposure、冻结、三路运行与聚合 |
| `test_evidence30_ablations.py` | 统一答题头与 Oracle packet |
| `test_evidence30_core_dense.py` | dev-only core-density、采样与防 gold 泄漏 |
| `test_config.py` | 环境变量、默认路径和可移植配置 |

修改停止原因、metadata 字段或 trace schema 时，先搜索对应断言。它们是调试和实验报告的
外部接口，不应当作普通内部字段随意改名。

## 3. 第一次真实 smoke

先用短视频和 direct，确认模型/processor/CUDA 能加载：

```powershell
qwen3vl-agent `
  --config configs\local.yaml `
  --strategy direct `
  --video path\to\short.mp4 `
  --query "Summarize the video." `
  --show-metadata
```

再跑同一题的 coarse-to-fine，最后才跑 active-tree。这样能区分：

- direct 也失败：模型、媒体解码、环境或题目本身；
- direct 成功、coarse-to-fine 失败：路由/窗口/协议；
- coarse-to-fine 成功、active-tree 失败：契约、动作、落账或验证；
- 最终答案正确但 verified=false：答案与证据闭环是两回事。

Active-tree smoke 推荐同时保存 JSON trace 和 HTML replay：

```powershell
qwen3vl-agent `
  --config configs\local.yaml `
  --strategy active_tree `
  --video path\to\short.mp4 `
  --query "Your question" `
  --choice "A" --choice "B" --choice "C" --choice "D" `
  --trace-output runs\smoke.json `
  --replay-output runs\smoke.html
```

## 4. 如何读一次失败

按这个顺序查 trace：

1. `fatal / degraded`：是否先发生异常或回退；
2. `resource ledger`：是否提前耗尽模型调用或帧预算；
3. `selected windows / scene frontier`：是否看到了正确时间范围；
4. `aligned subtitles / shown frames`：决定性像素或 cue 是否真的进入模型；
5. `raw protocol output`：模型 JSON 是否非法、截断或被修复；
6. `evidence ledger`：看到了但没有形成正确 atomic fact，还是根本没看到；
7. `missing slots`：哪个 required slot 未覆盖；
8. `completeness + skeptic`：验证为何拒绝；
9. `final adapter`：内容正确但选项标签映射是否错误。

不要只看最终 accuracy。项目的主要价值是把失败拆成 selection、exposure、grounding、
reasoning、verification 和 protocol 六类。

## 5. 常见修改落点

### 改模型加载或多模态消息

修改 `models/qwen3vl.py`，并先补 `test_model_messages.py`。不要把真实
模型对象泄漏到控制器。

### 添加工具

实现接收 `ToolContext` 的函数，通过 `ToolRegistry.tool` 注册，并为参数
提供 JSON Schema。默认 CLI 工具放在 `tools/defaults.py`，领域专用工具可以在调用方
构造 registry。

### 调 coarse-to-fine

- 预算和默认值：`coarse_to_fine/config.py`；
- 时间窗、帧预算：`types.py`；
- 视频/字幕：`cache.py`；
- prompt 和协议解析：`prompts.py`；
- 状态机：`agent.py`。

新增配置字段必须进入 dataclass，未知字段会被拒绝。

### 调 active-tree

先判断修改属于：

- 结构：`scene_tree.py`；
- 数据契约/账本/预算：`types.py`；
- prompt/协议：`prompts.py`；
- 对齐/时序裁决：`alignment.py`、`temporal.py`；
- 状态转移/回退：`agent.py`；
- 可视化：`replay.py`。

不要同时改 prompt、预算和评分再根据一个题判断效果；那样无法归因。

### 调 Evidence30

- 标注规范：`docs/videomme_evidence30_annotation_standard.md`；
- Schema：`schemas/`；
- exposure/grounding：`evaluation/evidence30.py`；
- 运行聚合：`evaluation/evidence30_runner.py`；
- 统一头：`evaluation/ablations.py`；
- core-density：`evaluation/core_dense.py`。

## 6. 实验纪律

### dev 与 locked

- dev 可以反复查看逐题 trace 并调试；
- locked 只允许冻结后一次性运行并先看聚合；
- 查看 locked 逐题记录或基于它调参，会消费该版本；
- 当前 locked v1 已运行，且整体 engineering pass 为 false，不能当作干净的最终确认集；
- 下一轮效果结论需要新建 held-out/locked 版本，而不是继续使用 v1。

### 防止 gold 泄漏

Oracle 与 core-density 是诊断，不是线上可用策略。prompt 中不能出现 official answer、
atomic fact、hard negative 文本、参考区间编号或前序模型答案。相关防泄漏断言已在测试中。

### 一次只改一个主变量

保留：

- 完整 YAML；
- policy ID 与 run signature；
- 源标注/基线 items 的 SHA-256；
- 代码提交 ID；
- 模型与依赖环境；
- 原始 runs（本地或外部制品库，不进 Git）；
- 去除本机路径后的精选 `results/`。

## 7. 什么可以提交

应提交：

- 源码和对应测试；
- 可移植 YAML、Schema、设计/实验文档；
- 冻结 annotations；
- 精简、脱敏的聚合结果。

不应提交：

- `runs/`、`tmp/`、`.cache/`；
- 模型、视频、字幕、parquet；
- HTML replay、抽帧图、逐题大 JSONL；
- `configs/local.yaml`、token、代理地址或凭据；
- `annotations/_unreviewed_ai_drafts`。

提交前执行：

```powershell
git status --short
git diff --check
python -m ruff check qwen3vl_agent tests scripts tools examples
python -m compileall -q qwen3vl_agent
python -m pytest -q
```

## 8. 已知风险与建议

- Active-tree 主控制器很大。拆分前先用现有特征测试锁定行为，优先提取纯函数和状态对象；
- GPU smoke 不在 CI 中。涉及模型输入、像素预算或 qwen-vl-utils 的改动必须本地实跑；
- 当前配置没有严格的顶层 Schema，但策略 dataclass 会拒绝未知字段；
- 相对缓存路径依赖启动时工作目录，团队环境建议显式设置 `QWEN3VL_CACHE_DIR`；
- Evidence30 标注没有独立双人复核，任何论文级结论前必须升级数据质量；
- 仓库没有许可证，外部复用前需要所有者授权。
