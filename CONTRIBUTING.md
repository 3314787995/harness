# Contributing

开始修改前先阅读 [架构说明](docs/architecture.md)、[上手指南](docs/onboarding.md) 和
[当前状态](docs/current_status.md)。

## 质量门

每个提交至少运行：

```powershell
python -m ruff check qwen3vl_agent tests scripts tools examples
python -m compileall -q qwen3vl_agent
python -m pytest -q
git diff --check
```

涉及模型消息、帧/像素预算、视频解码或 qwen-vl-utils 的改动，还必须用真实模型做至少一个
direct smoke 和一个受影响策略 smoke；GPU smoke 目前不在 CI 中。

## 改动原则

- 新行为必须有测试；修 bug 时先增加能复现问题的测试；
- 不随意改 trace 字段、停止原因、policy ID 或结果 Schema；
- Python 控制器负责合法动作和预算，模型输出不能绕过约束；
- prompt、搜索预算和评分规则不要在同一实验中同时变化；
- 新配置字段进入对应 dataclass，并保持未知字段报错；
- 大文件重构应先保持行为不变，再单独提交语义改动。

## 实验与数据

- 只在 dev 上调试；
- locked 运行必须遵守冻结协议；
- Oracle/core-density 诊断不得向模型泄漏 gold；
- 提交聚合结果时保留 run signature 和哈希，并去除绝对路径；
- 不提交模型、Video-MME 数据、`runs/`、缓存、回放或逐题大 trace；
- 冻结 annotations 只能通过版本升级修改，不能原地悄悄重写。

## 安全与许可证

不要提交 token、密码、代理凭据、私有数据路径或第三方受限内容。仓库当前没有开源许可证；
外部复用和再分发需要所有者明确授权。
