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

### 2026-03-28 - Start first optimization pass

- Goal: 在不引入重型 tokenizer 依赖的前提下，先处理当前输出里最明显的质量问题
- Plan:
  - 优先清理 `TinyStories` 原文中的字面量 `<|endoftext|>` 标记，避免模型把它当普通文本学进去
  - 把采样与验证温度显式配置化，减少“训练还行但默认 sample 太散”的噪音
  - 用短训练对比清理前后的样本质量是否改善
- Validation: 待执行
- Issue: 待确认仅清理 `<|endoftext|>` 是否足以明显改善当前 byte-level 输出
- Conclusion: 待优化
- Next: 实现数据清理和可配置采样温度

### 2026-03-28 - First optimization pass result

- Goal: 验证“清理 `<|endoftext|>` + 降低采样温度”是否能带来实质改善
- Change:
  - `prepare_tinystories` 默认改为清理字面量 `<|endoftext|>`，替换成空行
  - 新增采样温度参数传递链路：`sample.py`、`train.py`、`run_local.py`
  - `configs/tinystories-byte-50m.json` 默认将 `sample_temperature` 和验证温度调整为 `0.6`
  - 基于现有数据做出本地清理对照集 `data/tinystories-byte-clean/`
- Validation:
  - `./.venv/bin/python -m compileall lite_llm_pretraining`
  - `./.venv/bin/python -m lite_llm_pretraining.run_local --config /tmp/tinystories-byte-clean-50m-500step.json`
  - 对比 `checkpoints/tinystories-byte-50m-500step/` 与 `checkpoints/tinystories-byte-clean-50m-500step/`
- Issue:
  - 清理 `<|endoftext|>` 的确去掉了样本中的字面量分隔符，但在 `500` 步量级上，整体文本质量提升有限，核心瓶颈仍然是 byte-level tokenizer
- Conclusion:
  - 这是值得保留的低成本优化，因为它消除了一个明确的脏信号
  - 但它不足以单独解决“伪词和坏拼写”问题
- Next:
  - 第二轮优化优先级应上移到 tokenizer，而不是继续只调温度或只做轻量清洗

### 2026-03-28 - SentencePiece comparison result

- Goal: 验证 tokenizer 升级是否比继续在 byte-level 上打补丁更有效
- Change:
  - 新增显式 tokenizer 抽象与 checkpoint tokenizer 资产保存/加载
  - 新增 `lite_llm_pretraining.prepare_tinystories_sentencepiece`
  - 新增 `configs/tinystories-spm-32m.json`
  - `requirements.txt` 增加 `sentencepiece`
  - 用清理后的 `TinyStories` 训练 `SentencePiece unigram vocab_size=2048`
  - 生成 `data/tinystories-spm/`
    - `train_tokens=518,648,348`
    - `val_tokens=5,220,813`
  - 跑通 `33.7M` 参数的对照训练 `checkpoints/tinystories-spm-32m-1000step/`
- Validation:
  - `./.venv/bin/python -m lite_llm_pretraining.prepare_tinystories_sentencepiece --source_data_dir data/tinystories-byte-clean --out_dir data/tinystories-spm`
  - `./.venv/bin/python -m lite_llm_pretraining.run_local --config /tmp/tinystories-spm-32m-1000step.json`
- Issue:
  - 当前 `SentencePiece` 路线的样本里仍偶尔出现 `⁇`，说明下一轮还可以继续处理 unk / byte fallback
- Conclusion:
  - 即使只训练 `1000` 步，`SentencePiece 32M` 的样本已经明显比 `byte-level 49.7M` 更像自然故事
  - 这一轮优化证明：当前最值得继续投入的是 tokenizer 路线，而不是继续在 byte-level 上硬堆步数
- Next:
  - 优先继续优化 `SentencePiece` 路线，下一步可评估 `byte_fallback`、更合适 vocab 或更长训练
