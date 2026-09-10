# R6 开发输入

`requests.jsonl` 是可直接交给 preflight/run 的无答案输入。`answers.jsonl` 只供离线 score 使用。`media_catalog.json` 提供原视频链接、原生媒体名称和标注来源。

把对应原始媒体放到本目录的 `media/E02.mp4` 等位置，或在请求清单中修改 `video_path`。未随交付下载视频。E02 使用 MVBench 原生片段；其他题使用对应完整 benchmark 视频，不用标准答案或人工参考时间裁片。

默认清单只允许视觉；需要测试许可转写时，在复制的运行清单中增加文本路径、对应 allowed_modalities 和 subtitle_policy=aligned_only。读取范围由运行协议决定，不能因调试答案而扩大。

这九题只用于开发与机制讨论。E05、E11 的题意问题另行审查，不把原标注当成语义真值。完整使用方式见 ../../docs/r6_pipeline.md。
