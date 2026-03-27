# TinyStories Full Training Thread

- Status: in_progress
- Scope: 接入 `TinyStories` 全量数据集，配置约 `50M` 参数模型，并用现有训练/采样链路补充自动化质量验证

## Log

### 2026-03-28 - Start TinyStories full-dataset training work

- Goal: 让仓库从 `Tiny Shakespeare` smoke 级闭环，扩展到 `TinyStories` 全量训练的下一阶段闭环
- Plan:
  - 新增 `TinyStories` 数据准备脚本，支持全量下载、切分与元信息写入
  - 新增约 `50M` 参数的训练配置，并调整训练超参数到更适合该数据集
  - 补一个基于现有 checkpoint 采样链路的自动化质量验证入口
  - 跑通必要验证，确认训练、采样和验证入口能工作
- Validation: 待执行
- Issue: 当前环境尚未安装 `mlx`
- Conclusion: 待完成
- Next: 检查环境并开始实现 `TinyStories` 数据准备与配置

### 2026-03-28 - TinyStories pipeline landed and training verified

- Goal: 在当前仓库里真正落地 `TinyStories` 全量训练闭环，而不是停留在配置和文档层
- Change:
  - 新增 `lite_llm_pretraining.prepare_tinystories`，直接下载官方 `TinyStories-train.txt` / `TinyStories-valid.txt`
  - 对 byte-level 数据改为按 `uint8` 落盘，避免 `TinyStories` 全量数据无意义放大一倍
  - 训练读取逻辑改为按 `meta.json` 的 `token_dtype` 自动加载 memmap
  - 新增 `configs/tinystories-byte-50m.json`
    - `context_size=256`
    - `dim=640`
    - `num_layers=10`
    - `num_heads=8`
    - 参数量约 `49.7M`
  - `run_local` 支持按配置分发不同数据准备逻辑，并在训练后产出 `validation_report.json`
  - 新增 `lite_llm_pretraining.validate_checkpoint`，对采样文本做基础质量门槛检查并可附带 val loss 估计
  - 更新 `README.md`，补充 `TinyStories` 入口和 `50M` 配置说明
- Validation:
  - `./.venv/bin/python -m compileall lite_llm_pretraining`
  - `./.venv/bin/python -m lite_llm_pretraining.prepare_tinystories`
  - `./.venv/bin/python -m lite_llm_pretraining.train --config /tmp/tinystories-byte-50m-bringup.json`
  - `./.venv/bin/python -m lite_llm_pretraining.train --config /tmp/tinystories-byte-50m-sanity.json`
  - `./.venv/bin/python -m lite_llm_pretraining.run_local --config /tmp/tinystories-byte-50m-500step.json`
  - `./.venv/bin/python -m lite_llm_pretraining.run_local --config /tmp/tinystories-byte-50m-2000step.json`
  - `./.venv/bin/python -m lite_llm_pretraining.validate_checkpoint --checkpoint_dir checkpoints/tinystories-byte-50m-2000step/best --data_dir data/tinystories-byte --seed 123`
- Issue:
  - `2000` 步训练后，`val_loss` 已降到 `1.4707`，但在固定 `seed=123` 的严格验证下仅 `1/3` prompt 通过；人工查看 sample 仍明显带有 byte-level 早期模型的伪词和坏拼写，当前还不能称为“高质量模型”
- Conclusion:
  - `TinyStories` 全量数据和 `49.7M` 模型配置已经在本机真实跑通
  - 当前配置在 `batch_size=4` 下可以稳定训练，不需要立刻下调参数量
  - 仅靠 `2000` 步和 byte-level tokenizer，质量已经显著优于 smoke，但还不足以满足“高质量模型”目标
- Next:
  - 继续拉长训练步数，并优先评估是否要切换到更合适的 tokenizer，否则样本文本质量会被 byte-level 方案持续拖住
