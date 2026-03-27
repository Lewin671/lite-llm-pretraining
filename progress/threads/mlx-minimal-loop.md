# MLX Minimal Loop Thread

- Status: closed
- Scope: 在当前 M1 Pro 16GB 电脑上落地一个可复现、可验证的最小预训练闭环

## Log

### 2026-03-28 - Build and verify the first runnable loop

- Goal: 让仓库从纯文档状态进入“可以准备数据、启动训练、保存 checkpoint、重新加载并采样”的可运行状态
- Change:
  - 新增 `requirements.txt`，固定最小依赖为 `mlx` 与 `numpy`
  - 新增 `.gitignore`，忽略本地环境、数据与 checkpoint 产物
  - 新增 `configs/tinyshakespeare-byte-smoke.json` 作为第一阶段配置
  - 新增 `lite_llm_pretraining/prepare_tiny_shakespeare.py`
  - 新增 `lite_llm_pretraining/common.py`
  - 新增 `lite_llm_pretraining/train.py`
  - 新增 `lite_llm_pretraining/sample.py`
  - 更新 `README.md`，把当前真实能力、命令入口和产物路径写清楚
- Validation:
  - `python3 -m compileall lite_llm_pretraining`
  - `python -m lite_llm_pretraining.prepare_tiny_shakespeare`
  - `python -m lite_llm_pretraining.train`
  - `python -m lite_llm_pretraining.sample --checkpoint_dir checkpoints/tinyshakespeare-byte-smoke/best --prompt 'ROMEO:\n' --max_new_tokens 120`
- Result:
  - 数据准备成功，生成：
    - `data/tinyshakespeare-byte/train.bin`
    - `data/tinyshakespeare-byte/val.bin`
    - `data/tinyshakespeare-byte/meta.json`
  - 训练成功完成 200 step smoke stage
  - 生成：
    - `checkpoints/tinyshakespeare-byte-smoke/best/weights.npz`
    - `checkpoints/tinyshakespeare-byte-smoke/latest/weights.npz`
    - `checkpoints/tinyshakespeare-byte-smoke/metrics.jsonl`
    - `checkpoints/tinyshakespeare-byte-smoke/samples/step-0100.txt`
    - `checkpoints/tinyshakespeare-byte-smoke/samples/step-0200.txt`
  - 基础验证结果：
    - `val_loss: 5.1821 -> 2.4444`
    - `val_ppl: 178.05 -> 11.52`
  - 采样脚本可以从 `best` checkpoint 重新加载并输出文本
- Issue:
  - 首次运行时 `count_parameters()` 误把嵌套参数字典当成平面结构，已改为 `tree_flatten` 统计
- Conclusion: 当前仓库已满足“本机最小预训练闭环”目标
- Next: 如果继续推进，优先扩展 tokenizer 和数据规模，而不是重做训练骨架
