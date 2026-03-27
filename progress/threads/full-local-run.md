# Full Local Run Thread

- Status: closed
- Scope: 把当前仓库整理成一条可直接执行的数据准备、训练、checkpoint、采样本地完整运行链路

## Log

### 2026-03-28 - Start full local run workflow

- Goal: 让“整个 LLM 在本地运行成功”变成仓库里的明确入口，而不是依赖人工串命令
- Change:
  - 为现有脚本补充可复用函数接口：
    - `prepare_dataset()`
    - `train_from_config()`
    - `sample_from_checkpoint()`
  - 新增 `lite_llm_pretraining/run_local.py`，把数据准备、训练、checkpoint、重载采样串成单命令入口
  - 修正训练重跑时 `metrics.jsonl` 会累加旧结果的问题，改为每轮开始先重置指标文件
  - 更新 `README.md`，把单命令入口和产物路径写清楚
- Validation:
  - `source .venv/bin/activate && python -m compileall lite_llm_pretraining`
  - `source .venv/bin/activate && python -m lite_llm_pretraining.run_local --force_prepare --max_new_tokens 120`
- Result:
  - 统一入口已在当前电脑上完整跑通
  - 产物包括：
    - `data/tinyshakespeare-byte/meta.json`
    - `checkpoints/tinyshakespeare-byte-smoke/best/weights.npz`
    - `checkpoints/tinyshakespeare-byte-smoke/latest/weights.npz`
    - `checkpoints/tinyshakespeare-byte-smoke/local_run_summary.json`
    - `checkpoints/tinyshakespeare-byte-smoke/final_sample.txt`
  - 本轮端到端结果：
    - `loaded_step = 200`
    - `best_val_loss = 2.4435`
- Issue: 暂无
- Conclusion: 当前仓库已经有“单命令本地完整运行”入口，用户不需要手动串准备数据、训练和采样
- Next: 如果继续推进，优先把 tokenizer 和数据规模从 smoke 级别提升到更接近真实 lite LLM 预训练
