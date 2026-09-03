# P01 Pipeline v2

P01 v2 是一条面向“答案可由单个连续局部视频片段直接支持”的纯视觉推理路径。它假定
题目已经被上游路由为 P01；本包本身不负责题型路由，也不读取字幕、音频或外部知识。

当前状态是 **design-review candidate**：代码保留现行 v2 行为，方便先让老师确认流程
取舍。这里的 `v2` 是 pipeline 版本，不是模型版本。

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

要求 Python 3.10+、Qwen3-VL-8B 权重和可用 CUDA 环境。先安装项目：

```bash
python -m pip install -e ".[dev]"
```

设置模型与缓存路径：

```bash
export QWEN3VL_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export QWEN3VL_CACHE_DIR=/path/to/qwen3vl-cache
```

运行一条 MCQ：

```bash
qwen3vl-agent \
  --config configs/p01_8b.yaml \
  --strategy p01 \
  --video /path/to/video.mp4 \
  --query "What happens after the person opens the door?" \
  --choice "He sits down." \
  --choice "He leaves the room." \
  --choice "He closes the window." \
  --choice "It is not shown." \
  --trace-output runs/p01-example.json \
  --show-metadata
```

有 benchmark 给定区间时加 `--given-interval START END`。不传 `--choice` 时走 G39
自由文本路径。`--force-choice` 仅为旧调用兼容保留，对 v2 结果没有影响。

## 25 题 smoke runner

数据不随仓库提交。准备符合 runner manifest 的本地数据后：

```bash
python scripts/p01_gpu_preflight.py \
  --config configs/p01_8b.yaml \
  --data-root /path/to/data \
  --work-dir /path/to/runtime

python scripts/run_p01_smoke.py \
  --config configs/p01_8b.yaml \
  --data-root /path/to/data \
  --output-dir /path/to/runtime/runs/p01-v2
```

runner 在一个进程内只加载一次模型，逐题原子写入，并默认根据 `items/*.json` 断点续跑。
服务器部署、模型下载、五题 gate 和 safe profile 见
[docs/p01_rental_runbook.md](../../docs/p01_rental_runbook.md)。

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
