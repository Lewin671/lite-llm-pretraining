# TinyStories Optimization Sweep Thread

- Status: in_progress
- Scope: 以 `TinyStories` 为主线，连续进行不少于 20 个优化方案尝试，并逐轮记录结果与提交

## Goal

- 不再停留在单一配置讨论
- 用可比的短训练实验筛出更优路线
- 优先提升真实 sample 质量，而不是只看 loss

## Sweep Rules

- 每个尝试使用唯一 `Attempt ID`
- 每个尝试至少记录：配置变化、验证方法、主要结果、结论
- 先用短训练筛方向，再把最好路线继续拉长
- 尽量只改变少量变量，避免无法归因
- 从本轮开始，`validate_checkpoint` 额外把 `⁇`、`<unk>`、`<|endoftext|>` 记为未知标记并参与 pass/fail

## Tooling

- 已新增 `python -m lite_llm_pretraining.run_sweep_attempt`
- 每个新 attempt 会把配置、核心指标和 sample 摘要写到 `progress/artifacts/tinystories-sweep/`

## Attempts

### A01

- Change: 现有 `SentencePiece unigram 2048 / 32M / 1000step` checkpoint，验证温度改为 `0.5`
- Validation: `validate_checkpoint`，3 个固定 prompt，`max_new_tokens=120`，`seed=123`
- Result: `3/3` 通过；`val_loss=3.353845`
- Conclusion: 文本最稳，仍有重复，但怪词和 `⁇` 最少

### A02

- Change: 同一 checkpoint，验证温度改为 `0.6`
- Validation: 同 A01
- Result: `3/3` 通过；`val_loss=3.353845`
- Conclusion: 也很稳，但比 `0.5` 略多重复和 `⁇`

### A03

- Change: 同一 checkpoint，验证温度改为 `0.7`
- Validation: 同 A01
- Result: `3/3` 通过；`val_loss=3.353845`
- Conclusion: 可读性下降，开始稳定出现更奇怪的词形

### A04

- Change: 同一 checkpoint，验证温度改为 `0.8`
- Validation: 同 A01
- Result: `3/3` 通过；`val_loss=3.353845`
- Conclusion: 形式上通过，但主观质量继续下降，不适合作为后续 sweep 默认温度

### Pending Batch

- A05: `SentencePiece unigram 2048` 重建统一 tokenizer 基线数据集
- A06: `SentencePiece unigram 4096`
- A07: `SentencePiece BPE 2048`
- A08: `SentencePiece BPE 4096`
- A09: `SentencePiece unigram 2048 + byte_fallback`
- A10: `SentencePiece BPE 2048 + byte_fallback`
- A11: `SentencePiece unigram 2048 + smaller context`
- A12: `SentencePiece unigram 2048 + larger context`
- A13: `SentencePiece unigram 2048 + smaller model`
- A14: `SentencePiece unigram 2048 + larger model`
- A15: `SentencePiece unigram 2048 + lower learning rate`
- A16: `SentencePiece unigram 2048 + higher learning rate`
- A17: `SentencePiece unigram 2048 + smaller batch`
- A18: `SentencePiece unigram 2048 + larger batch`
- A19: `SentencePiece unigram 2048 + longer warmup`
- A20: `SentencePiece unigram 2048 + shorter warmup`
- A21: `Best short-run winner + longer run`
- A22: `Best overall carry-forward run`

## Validation

- 当前短结论：
- `SentencePiece` 默认温度先收敛到 `0.5`
- 高温度会放大当前 checkpoint 的伪词与不稳定输出
- 需要继续看 tokenizer 变量和训练超参数，确认最优路线是否仍是 `unigram 2048`

## Conclusion

- 第一轮先不继续调高采样温度
- 后续短训练对照默认使用 `temperature=0.5`
