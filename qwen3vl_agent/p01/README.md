# P01 Pipeline v2：18 类设计范围与导师验收

P01 v2 是一条面向“答案可由单个连续局部视频片段直接支持”的纯视觉推理路径。它假定
题目已经被上游路由为 P01；本包本身不负责题型路由，也不读取字幕、音频或外部知识。

当前状态是 **design-review candidate**：这是面向以下 18 类原生题型的 P01 v2 设计评审版本，
不是 18 类已全部适配或验证通过的声明。这里的 `v2` 是 pipeline 版本，不是模型或 benchmark 版本。

## 18 类设计范围

清单来自原计划候选范围，移除了 **Video-MME · Action Recognition**。这是本项目的设计范围，
不是各 benchmark 官方给出的统一 pipeline 分类。

| # | Benchmark | 原生题型 |
|---|---|---|
| 1 | LVBench | Entity Recognition |
| 2 | LVBench | Event Understanding |
| 3 | LVBench | Key Information Retrieval |
| 4 | LVBench | Temporal Grounding |
| 5 | MLVU | Sub-Scene Captioning |
| 6 | MVBench | Action Antonym |
| 7 | MVBench | Fine-grained Action |
| 8 | MVBench | Fine-grained Pose |
| 9 | MVBench | Moving Attribute |
| 10 | MVBench | Object Existence |
| 11 | TVBench | Action Antonym |
| 12 | Video-MME | Attribute Perception |
| 13 | Video-MME | Object Recognition |
| 14 | Video-MME | OCR Problems |
| 15 | Video-MME | Spatial Perception |
| 16 | Video-MME-v2 | Fine-Grained Action Recognition |
| 17 | Video-MME-v2 | Vision-Guided Audio Description |
| 18 | Video-MME-v2 | Visual Recognition |

**第 17 项目前尚未实现音频回答部分**。它保留在设计范围中供导师讨论，但不能计入当前已实现能力；
纯视觉输入不能替代该任务要求的声音证据。本次不增加音频、字幕或跨模态接口。

其他 17 项也不表示所有样本都适用：上游仍须确认答案能由单个连续局部片段直接支持。
跨不相交片段聚合、依赖外部知识等题目不因类别在表中就自动符合 P01。
当前没有原生题型自动路由器，也没有按本清单拦截请求的分类器。

移除的是 Video-MME 的一个原生类别，**不是删除动作理解能力**。内部 `dynamic_action`
模式继续服务于保留的细粒度动作、事件和子场景题。原 smoke 题单和失败案例作为历史记录保留，
不作为这 18 类的覆盖证明。

## 本次版本基线

- 核心流程与模型适配层取自本地重测分支 `p01-v2-dual4090d` 的 `3ad03b9`。
- 六个核心流程文件与 2026-09-02 服务器快照一致；保留原 prompt、取证、refinement、rescue 和预算。
- 旧评审提交 `27967aa` 中额外增加的统一 64/48 帧决策截断已移除；动态决策仍最多 96 帧。
  原流程可产生 90–96 帧调用，但不是每次强制取这些帧，其他模式仍使用各自上限。
- 保留通用模型分片、输入设备处理和运行记录能力，不带入七题专用 runner、双卡配置或租卡部署包。
- 本次发布只做静态检查，不主动运行单测、模型推理或压力预检；原有 GitHub CI 配置保持不变。

## 一句话流程

```text
问题
  -> 选项盲 ObservationCompiler
  -> 选项盲分层 Locator
  -> 最多两个选项盲 CandidateScout
  -> 选出一个连续局部段
  -> 引入去标签选项判别条件
  -> 最多一次常规局部 refinement
  -> InitialDecision
  -> 必要时最多一次 bounded rescue
  -> FinalDecision / 确定性合法选项回退
```

G39 自由文本题复用定位和取证阶段，最后由 Composer 生成非空 best-effort 描述。

## 固定边界

- 输入必须是一个视频；不支持图片、字幕或音频。
- 首轮 ObservationCompiler、Locator、CandidateScout 看不到选项。
- 只允许一个 canonical local span；不把两个不相交片段拼成证据。
- 有显式区间时，区间外帧不得进入定位、上下文或决策。
- 每题最多一次 routine refinement 和一次 bounded rescue。
- 全局 overview 只用于 rescue 定位，不能直接回答。
- MCQ 必须输出合法选项；`weak/none` 证据不会触发拒答。
- 默认模型调用预算为 20，并为终局 decision/repair 预留 2 次。

完整协议、媒体预算、降级链与输出字段见
[docs/p01_v2_pipeline.md](../../docs/p01_v2_pipeline.md)。

## 快速运行

以下是普通单题入口，不需要原来的 25 题数据包或服务器目录。要求 Python 3.10+、
Qwen3-VL-8B 权重，以及相互兼容的 PyTorch/CUDA/FlashAttention 2 环境。
默认保留 `bfloat16` 与 `flash_attention_2`；请按自己的系统和硬件准备这些依赖，
仓库不指定 CUDA 安装包、显卡型号、卡数或显存容量，也不承诺任意设备可运行或零 OOM。

以下命令在已经准备好兼容 PyTorch 的环境中执行；已有完整环境时跳过重复安装。
先取得本评审分支（不是默认 `main`）：

```bash
git clone --branch p01-v2-review https://github.com/3314787995/harness.git
cd harness
python -m pip install -e .
# 如果尚未安装 FlashAttention 2，在兼容的编译环境中安装：
python -m pip install packaging psutil ninja
python -m pip install flash-attn --no-build-isolation
```

不需要安装 dev extra 或先跑单测、smoke、显存压力预检。FlashAttention 2 不在基础依赖中，
不能把仅安装本项目视为已准备好该后端；本版本不会在失败时静默改用 SDPA。

设置自己的模型与缓存路径（以下为 Bash 示例）：

```bash
export QWEN3VL_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export QWEN3VL_CACHE_DIR=/path/to/qwen3vl-cache
```

PowerShell 中对应使用 `$env:QWEN3VL_MODEL_PATH = "<模型目录>"` 和
`$env:QWEN3VL_CACHE_DIR = "<缓存目录>"`。未设置模型变量时，配置使用
`Qwen/Qwen3-VL-8B-Instruct`，加载时可能从 Hugging Face 下载；已有权重时应显式指定本地目录。

普通配置 `configs/p01_8b.yaml` 使用 `device: auto`。它不设置 `CUDA_VISIBLE_DEVICES`、
`required_cuda_devices` 或逐卡 `max_memory`；模型放置由加载器结合可见设备决定。
如需手动控制设备或内存，请复制为被 Git 忽略的 `configs/local.yaml`，按自己的环境显式配置；
这不意味着自动放置能保证运行时显存充足，模型分片也不是启动多份 pipeline 并行答题。
若复制的是旧评审版配置，先删除已退役的 `p01.decision_max_frames` 字段；其他流程预算保持原值。

### 选择题

下面的问题和选项仅示范 CLI 格式，请替换为自己视频中的真实问题：

```bash
qwen3vl-agent \
  --config configs/p01_8b.yaml \
  --strategy p01 \
  --video /path/to/video.mp4 \
  --query "What color is the person's jacket near the entrance?" \
  --choice "Red" \
  --choice "Blue" \
  --choice "Green" \
  --choice "White" \
  --trace-output runs/p01-example.json \
  --show-metadata
```

`--choice` 按原选项顺序重复传入，MCQ 支持 2–26 个选项。`--force-choice` 仅为旧调用兼容保留，
对 v2 结果没有影响。

### 显式时间区间

时间单位为秒，边界是原视频的绝对时间，不是裁剪后重新从零计时：

```bash
qwen3vl-agent \
  --config configs/p01_8b.yaml \
  --strategy p01 \
  --video /path/to/video.mp4 \
  --query "What happens from 118 to 166 seconds?" \
  --given-interval 118 166 \
  --choice "A person enters the room." \
  --choice "A person leaves the room." \
  --trace-output runs/p01-interval.json
```

该参数适用于 MCQ 和自由文本题；区间外帧不会进入本题取证或决策。

### 自由文本

不传 `--choice` 时复用 G39 子场景描述路径，例如：

```bash
qwen3vl-agent \
  --config configs/p01_8b.yaml \
  --strategy p01 \
  --video /path/to/video.mp4 \
  --query "Describe what the person does after placing the bag on the table." \
  --trace-output runs/p01-caption.json \
  --show-metadata
```

G39 是项目内部路径名，不是额外增加的第 19 类 benchmark 题型。

### 保存回答与检查证据

终端输出最终回答；`--show-metadata` 另外打印完整 metadata。`--trace-output` 把 metadata
保存为 JSON，P01 结果位于其中的 `p01` 字段。为每次运行指定不同文件名，避免覆盖旧记录。

- `prediction`、`decision_source`：最终选项或文本，以及 initial/rescue/fallback/composer 来源。
- `trace`：定位、候选、区间变化、rescue 和协议退化等流程记录。
- `resources.calls[]`：实际调用的 `role`、`prompt`、`raw_response`、帧信息和 `model_metadata`。
  **逐次原始模型回答在 `p01.resources.calls[].raw_response`**，不是只有最终选项。
- `evidence`、`evidence_grade`：所选局部段、证据和支持等级；合法答案不等于证据充分或答案正确。
- `resources` 及调用中的模型 metadata：调用次数、媒体曝光、OOM 与模型放置/显存记录。

完整字段语义见[输出协议](../../docs/p01_v2_pipeline.md#输出协议)。CLI、请求和返回结构保持兼容，
模型 metadata 新增的设备与显存信息是诊断字段，不改变答案规则。原始 trace 可能含本机绝对路径，
请保存在自己的运行目录，脱敏前不要直接提交 GitHub。

## 历史 smoke 与部署资料

旧 25 题 runner、safe 配置和[旧租卡手册](../../docs/p01_rental_runbook.md) 仅供复查历史实验，
不是上述普通运行的前置步骤，也不代表当前 18 类覆盖情况。原题单中保留的 Action Recognition
题不属于本次对外设计范围。模型、原视频、题单数据与原始结果不随仓库提交。

## 代码地图

| 文件 | 职责 |
|---|---|
| `agent.py` | 主状态机、模型调用预算、refinement、rescue 和终局回退 |
| `config.py` | v2 预算与不变量校验 |
| `media.py` | 低分辨率导航索引、源帧抽取、OCR/crop 媒体准备 |
| `control.py` | 确定性区间、候选排序、证据合并和 option 解析 |
| `prompts.py` | 各角色 prompt、JSON parser 和协议回退 |
| `types.py` | 请求、证据、决策、资源账本和返回结构 |
| `smoke.py` | 25 题单进程 runner、断点续跑和汇总 |
| `preflight.py` | 配置、数据、依赖、GPU 与模型加载检查 |

## 当前评审重点

当前实跑暴露出的核心问题不是“完全找不到视频”，而是早期取证与后期选项判别之间的
信息闭环不够强：选项差异进入较晚、弱候选仍可被强制送入终局、跨阶段矛盾没有成为
硬门控。具体证据和建议向老师确认的问题见
[docs/p01_v2_design_review.md](../../docs/p01_v2_design_review.md)。

原始逐题 trace、模型权重、视频、服务器路径和凭据不进入 Git。
