# 发布验证记录

日期：2026-09-10。验证对象是独立发布目录的冻结代码，Python **3.11.16**，Windows；GitHub CI 使用 Python 3.11／Ubuntu。基于远端 `116c514dc1975f71e65119859da5952078e17714` 正常提交，原工作目录未被发布整理覆盖。

## 检查结果

| 检查 | 结果 |
|---|---|
| 冻结源快照、同组当前测试 | 919 passed / 61 failed，共 980 项 |
| 最终发布版全部测试 | **938 passed / 60 failed，共 998 项** |
| 新增公开接口测试 | 18 passed：九类配置／API、R1 最新别名、旧入口拒绝和 trace 兼容 |
| 发布版相对源快照的新增失败 nodeid | **0** |
| 无模型导入、九类配置与版本、CLI／安装入口 | 通过 |
| compileall | 通过 |
| R6 九题 preflight | 9 条 media_unavailable，全部 model_loaded=false；原媒体未随仓库分发 |
| 全量 Ruff | **234 条诊断，未通过**；完整选择规则固定在 pyproject.toml |
| 新增公共入口与发布检查工具的限定 Ruff | 通过 |
| GPU smoke／完整 benchmark 测评 | 未执行 |

**本仓库当前不是全绿状态。** CI 的 lint 与当前回归测试仍会如实失败，不使用 continue-on-error、xfail 或跳过有效失败来制造通过结果。发布完整性、编译和测试是独立步骤，lint 失败不会阻止后续测试执行。

## 已有失败与解释

保留的失败均可在冻结源代码的对应测试中复现。失败名称虽可能带历史版本号，但仍检查当前共享代码、证据、预算、恢复或协议行为，因此保留；没有仅按版本号删除这些测试。

| 测试文件 | 失败数 |
|---|---|
| `test_r3_query_v5.py` | 7 |
| `test_r4_v5.py` | 3 |
| `test_r4_v5_1_compile.py` | 7 |
| `test_r4_v5_2.py` | 10 |
| `test_r4_v5_3.py` | 1 |
| `test_r4_v5_4.py` | 5 |
| `test_r4_v5_5.py` | 8 |
| `test_r4_v5_6.py` | 10 |
| `test_r4_v5_composition.py` | 1 |
| `test_r4_v5_edges.py` | 8 |

具体 nodeid 和错误分类见 [已知测试失败](known_test_failures.json)。分类统计：`{"existing_result_budget_or_state_assertion": 41, "existing_prompt_or_example_contract_mismatch": 5, "existing_runtime_contract_error": 10, "existing_schema_or_fixture_contract_mismatch": 4}`。

- R3 的一部分测试仍固定旧的调用次数／复核次数预期，与当前 5.4 的局部复核流程不一致；本次保留断言与失败，未把这种差异自动视为算法正确。
- R4 同时存在旧观察 schema／示例约定不一致、结果与预算断言失败，以及真实的异常路径问题。例如编译失败后 `spec=None` 进入 `map_answer`，触发 `choice_values` 属性错误。它们不是发布裁剪新增的问题，本次没有改写推理逻辑来消除它们。
- 提示词／模拟输出契约漂移与当前运行缺陷分别记录；对未能仅凭现有证据定性的结果断言，保留为待处理回归，而不是宣称全部属于过时测试。

## 本次整理修复及测试迁移

1. R1 的公开 `r1` 与 `r1-v3` 都指向 3.1；相应默认入口断言更新，不再要求旧 solver。保留原 `r1_v3` 配置／trace 命名空间。
2. CLI 保持可不传配置的程序默认路径，推荐实际使用对应配置；现有 R2 原选项分发回归通过。
3. 初始文件大小筛选遗漏三份 R3 回放夹具，已从源文件补充冻结。发布 JSON 仅压缩空白及移除机器路径，没有删去回归证据；相关 102 项定向检查通过。
4. R5 的历史文件锁定原先包含已被其他 R 类更新的共享模型／runner，现收窄到 R5 所有代码与配置，公共接口由当前测试验证。9 个原 AST 哈希在原 Python 3.13 下与当前函数全部匹配；迁移至 Python 3.11 的 AST 序列化，保持 R5 函数语义不变。这解释了源快照中那条旧冻结测试的失败。
5. 只移除独立旧 solver 的测试函数和未引用夹具；仍被当前测试导入的 fake builder／fixture 保留。新接口增加 18 项测试，没有设置跳过或预期失败。

## 范围与复现

测试使用 FakeModel、合成媒体和必要的已记录协议回放。通过意味着被测试的工程行为成立，不等于真实视频理解准确率或性能收益。R6 1.0 已实现，但真实 GPU smoke 未执行，音频 Provider 仍未启用。此前其他版本的小样本成绩不作为本快照的实验结论。

```bash
python -m pip install -e . --no-deps
python -m pip install -r requirements-cpu.txt
python tools/check_release.py
python -m compileall -q qwen3vl_agent
python -m ruff check qwen3vl_agent tests scripts tools
python -m pytest -q --tb=short
```

发布文件的协议版本与 SHA-256 见 [发布清单](../release_manifest.json)。本地详细原始日志保留在发布工作目录之外，不随仓库上传。对不同操作系统的额外差异以 GitHub CI 的实际输出为准。
