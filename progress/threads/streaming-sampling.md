# Streaming Sampling Thread

- Status: closed
- Scope: 为本地 checkpoint 采样补充终端实时 streaming 输出

## Log

### 2026-03-28 - Start streaming sampling support

- Goal: 让采样命令在终端里逐步输出生成文本，而不是全部生成完成后一次性打印
- Change:
  - 在 `lite_llm_pretraining/common.py` 中新增 `sample_text_stream()`，基于 UTF-8 增量解码逐步输出文本
  - 保留原有 `sample_text()`，改为复用 streaming 生成逻辑后再拼接完整文本
  - 在 `lite_llm_pretraining/sample.py` 中新增 `--stream` 参数和 `stream_from_checkpoint()` 路径
  - 更新 `README.md`，补 streaming 命令示例
- Validation:
  - `source .venv/bin/activate && python -m compileall lite_llm_pretraining`
  - `source .venv/bin/activate && python -m lite_llm_pretraining.sample --checkpoint_dir checkpoints/tinyshakespeare-byte-smoke/best --prompt 'ROMEO:\n' --max_new_tokens 80 --stream`
- Result:
  - 终端已能基于现有 `best` checkpoint 实时输出生成文本
  - checkpoint 元信息会先打印，再进入 streaming 输出
- Issue: 暂无
- Conclusion: 当前仓库已经具备基础 streaming 采样能力
- Next: 如继续推进，可把 `run_local` 的最终采样也改成可选 streaming 模式
