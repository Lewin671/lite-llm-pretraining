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
- Result: `0/3` 通过；`val_loss=3.353845`；未知标记计数 `[2, 3, 2]`
- Conclusion: 旧校验口径太宽，严格计入 `⁇` 后这条不再通过

### A02

- Change: 同一 checkpoint，验证温度改为 `0.6`
- Validation: 同 A01
- Result: `1/3` 通过；`val_loss=3.353845`；未知标记计数 `[2, 1, 0]`
- Conclusion: 在当前 `32M / 1000step` 基线上，`0.6` 比 `0.5/0.7/0.8` 更接近可用

### A03

- Change: 同一 checkpoint，验证温度改为 `0.7`
- Validation: 同 A01
- Result: `0/3` 通过；`val_loss=3.353845`；未知标记计数 `[2, 2, 2]`
- Conclusion: 温度升高会稳定放大 `⁇` 和坏词形

### A04

- Change: 同一 checkpoint，验证温度改为 `0.8`
- Validation: 同 A01
- Result: `0/3` 通过；`val_loss=3.353845`；未知标记计数 `[5, 1, 2]`
- Conclusion: `0.8` 继续恶化，后续 sweep 默认不再用高温

### A05

- Change: `context_size=128`，其余沿用 `SentencePiece unigram 2048` 结构，`200 step`
- Validation: `run_sweep_attempt`，固定 3 prompt，`temperature=0.5`
- Result: `best_val_loss=4.4526`；严格校验 `0/3`；未知标记总数 `10`
- Conclusion: 短窗口里 sample 比更大 context 更完整，但仍远未达标

### A06

- Change: `context_size=384`，其余不变，`200 step`
- Validation: 同 A05
- Result: `best_val_loss=4.1247`；严格校验 `0/3`；未知标记总数 `12`
- Conclusion: token loss 更低，但 sample 更差且训练更慢，不适合作为当前主线

### A07

- Change: 更小模型 `dim=384 / layers=8 / heads=6`，`200 step`
- Validation: 同 A05
- Result: `best_val_loss=4.2479`；严格校验 `0/3`；未知标记总数 `17`
- Conclusion: 小模型没有换来更干净的文本，退化明显

### A08

- Change: 更大模型 `dim=640 / layers=12 / heads=10 / batch=4`，`200 step`
- Validation: 同 A05
- Result: `best_val_loss=4.5742`；严格校验 `0/3`；未知标记总数 `15`
- Conclusion: 在短训练预算下，大模型反而更差，不值得当前阶段继续加大

### A09

- Change: 以 `A05` 的 `context=128` 为基座，学习率降到 `1.5e-4`，`200 step`
- Validation: 同 A05
- Result: `best_val_loss=4.5820`；严格校验 `0/3`；未知标记总数 `10`
- Conclusion: 降学习率没有带来更好 sample，短窗内只是在更慢地学

### A10

- Change: 以 `A05` 为基座，学习率升到 `4.0e-4`，`200 step`
- Validation: 同 A05
- Result: `best_val_loss=4.4488`；严格校验 `0/3`；未知标记总数 `16`
- Conclusion: 更高学习率稍微压低了 loss，但文本更脏，不能只看 loss

### A11

- Change: 以 `A05` 为基座，`batch_size=4`，`200 step`
- Validation: 同 A05
- Result: `best_val_loss=4.9558`；严格校验 `0/3`；未知标记总数 `17`
- Conclusion: 更小 batch 明显退化，当前主线不考虑

### A12

- Change: 以 `A05` 为基座，`batch_size=12`，`200 step`
- Validation: 同 A05
- Result: `best_val_loss=4.2541`；严格校验 `0/3`；未知标记总数 `16`
- Conclusion: 这是目前短窗里最好的 loss，但 sample 仍不干净，只能保留为备选

### Pending Batch

- A13: `SentencePiece unigram 2048 + longer warmup`
- A14: `SentencePiece unigram 2048 + shorter warmup`
- A15: `SentencePiece unigram 2048 + no gradient checkpointing`
- A16: `Best short-run checkpoint + lower temperature`
- A17: `Best short-run checkpoint + higher temperature`
- A18: `Byte-level 50M / 2000step under strict validator`
- A19: `Byte-level clean 50M / 500step under strict validator`
- A20: `Best overall carry-forward run`

## Validation

- 当前短结论：
- 严格校验后，旧 `32M / 1000step` 基线只在 `temperature=0.6` 下有 `1/3` 通过
- 在当前 `200 step` 窗口里，结构 sweep 没有出现比基线更强的明显赢家
- `context=384` 和 `batch=12` 都能降低 loss，但还没有转化成更干净的文本

## Conclusion

- 当前主线先不再继续放大模型或 context
- 下一批重点转到学习率、batch 和 warmup，而不是结构扩张
