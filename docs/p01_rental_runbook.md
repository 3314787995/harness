# P01 4090D 租卡运行手册

这个上传包已经固定为 25 题、20 个视频，数据仍按 Video-MME、MLVU、LVBench 三个原 benchmark 分开存放。以下命令假定压缩包解到了数据盘，并且当前目录是上传包根目录。

## 1. 解包与完整性校验

示例把数据盘设为 /root/autodl-tmp；若平台路径不同，只改第一行。

    export DATA_MOUNT=/root/autodl-tmp
    cd "$DATA_MOUNT"
    tar -xf p01-smoke-v2-final-upload.tar
    cd p01-smoke-v2-final
    python code/tools/verify_p01_upload_bundle.py --bundle-root .

校验必须显示 25 questions、20 videos 且 ready=true。失败时不要开始计费跑题，优先重传损坏文件。

## 2. 环境与模型目录

模型、缓存和运行结果全部放数据盘，避免占满 30 GB 系统盘。

    export BUNDLE="$DATA_MOUNT/p01-smoke-v2-final"
    export RUNTIME="$DATA_MOUNT/p01-runtime"
    export HF_HOME="$DATA_MOUNT/huggingface"
    export QWEN3VL_MODEL_PATH="$HF_HOME/Qwen3-VL-8B-Instruct"
    export QWEN3VL_CACHE_DIR="$DATA_MOUNT/qwen3vl-cache"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export CUDA_VISIBLE_DEVICES=0
    mkdir -p "$RUNTIME" "$HF_HOME" "$QWEN3VL_CACHE_DIR"
    cd "$BUNDLE/code"

建议选择平台的 PyTorch 2.x、Python 3.11、CUDA 12.x 镜像。宿主机显示 CUDA 13.0 不要求安装 CUDA 13 工具链；PyTorch 自带的 CUDA runtime 只需与驱动兼容。

安装项目及 FlashAttention 2：

    python -m pip install -U pip packaging psutil ninja huggingface_hub
    python -m pip install -e ".[dev]"
    MAX_JOBS=4 python -m pip install flash-attn --no-build-isolation

下载模型到数据盘：

    hf download Qwen/Qwen3-VL-8B-Instruct --local-dir "$QWEN3VL_MODEL_PATH"

两个配置会自动读取 QWEN3VL_MODEL_PATH，无需改写配置文件。

## 3. GPU preflight

先做完整静态、依赖、数据、GPU 检查：

    python scripts/p01_gpu_preflight.py \
      --config configs/p01_8b.yaml \
      --data-root "$BUNDLE/data" \
      --work-dir "$RUNTIME" \
      --json-output "$RUNTIME/preflight.json"

再做一次真实模型加载。只有报告 ready=true 才开始跑题：

    python scripts/p01_gpu_preflight.py \
      --config configs/p01_8b.yaml \
      --data-root "$BUNDLE/data" \
      --work-dir "$RUNTIME" \
      --load-model \
      --json-output "$RUNTIME/preflight-model-load.json"

如果 FlashAttention 2 安装失败，不要让程序静默切换。可显式把两个配置中的 attn_implementation 改为 sdpa，并在结果目录旁保存修改后的配置；这会成为单独的实验条件。

## 4. 先跑五类各一题

P01 v2 不需要 `--force-choice`：每道有效 MCQ 都必须输出一个选项，G39 必须输出非空描述。题目顺序是：G01 静态、G01 动作、G07 OCR、G39、G42。

    python scripts/run_p01_smoke.py \
      --config configs/p01_8b.yaml \
      --data-root "$BUNDLE/data" \
      --output-dir "$RUNTIME/runs/v2-gate-5" \
      --question-id 058-2 \
      --question-id 007-1 \
      --question-id 007-3 \
      --question-id sub_scene:33 \
      --question-id 3343

检查 `summary.json`、`items/*.json` 和每题 trace。五题都应有非空 prediction；同时检查 `decision_source`、`support_level`、`bounded_rescue`、`oom_retries` 和采样曝光。确认没有未恢复的 OOM 或明显采样异常后再跑全量。

## 5. 跑完整 25 题

Runner 在同一进程只加载一次模型，逐题原子落盘并默认断点续跑。重新执行同一条命令会跳过已有题目。
MCQ 首轮定位和 scout 保持选项盲。选中局部段后，模型编译去标签判别声明并执行局部 DecisionPass；若证据或选项判别较弱，最多再执行一次 bounded rescue 和 FinalDecision。独立 Verifier 已从 v2 主路径删除。

    python scripts/run_p01_smoke.py \
      --config configs/p01_8b.yaml \
      --data-root "$BUNDLE/data" \
      --output-dir "$RUNTIME/runs/v2-full-25"

显式时间区间会只建立区间内导航指标；区间外画面不会进入观察、context 或关键帧排序。超过 60 秒的区间自适应分成不超过 8 个重叠 chunk，并保持整个逻辑区间的连续 canonical span。

本轮工程验收先看覆盖率：20/20 MCQ 有合法选项，5/5 G39 有非空文本。准确率仍需完整报告，但不作为代码是否完成的 gate，也不能据这 25 题宣称泛化提升。

## 6. 4090D fallback

默认配置的每次模型调用遇到 CUDA OOM 时，会记录失败、清理缓存并以低像素预算重试一次。只有仍出现可复现 OOM 时才整体启用 safe 配置。它降低像素预算，但不改变单连续 canonical span、一次 refinement、一次 bounded rescue 和 MCQ 必答协议。

    python scripts/run_p01_smoke.py \
      --config configs/p01_4090d_safe.yaml \
      --data-root "$BUNDLE/data" \
      --output-dir "$RUNTIME/runs/v2-4090d-safe-full-25"

不要在原结果目录中更换配置续跑。default 与 4090d-safe 必须保留为两个独立实验条件。

## 7. 关机前下载

至少下载整个 $RUNTIME 目录，其中应包含：

- preflight.json 与 preflight-model-load.json；
- 每次运行的 run_plan.json、environment.json、summary.json；
- results.jsonl 和 items/ 下的逐题结果；
- 每题 trace、资源账本、异常栈和峰值显存信息。

确认结果已下载并可打开后再释放实例。
