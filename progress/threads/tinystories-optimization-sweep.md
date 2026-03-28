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

### A13

- Change: 以 `A05` 为基座，`warmup_steps=600`，`200 step`
- Validation: 同 A05
- Result: `best_val_loss=4.6493`；严格校验 `0/3`；未知标记总数 `10`
- Conclusion: 更长 warmup 过慢，短窗里明显吃亏

### A14

- Change: 以 `A05` 为基座，`warmup_steps=50`，`200 step`
- Validation: 同 A05
- Result: `best_val_loss=4.5479`；严格校验 `1/3`；未知标记总数 `15`
- Conclusion: 这是当前短训里第一个 `1/3` 通过的配置，值得继续做温度复核和拉长训练

### A15

- Change: 以 `A05` 为基座，关闭 `gradient_checkpointing`，`200 step`
- Validation: 同 A05
- Result: `best_val_loss=4.4524`；严格校验 `0/3`；未知标记总数 `13`
- Conclusion: 不开 gradient checkpointing 没带来质量收益，只是提供一个速度/显存参考点

### A16

- Change: 对 `A14` best checkpoint 降验证温度到 `0.4`
- Validation: `validate_checkpoint`，固定 3 prompt，`max_new_tokens=120`，`seed=123`
- Result: `0/3` 通过；`val_loss=4.4379`；未知标记总数 `14`
- Conclusion: 继续降温没有帮助，只是让输出更钝

### A17

- Change: 对 `A14` best checkpoint 升验证温度到 `0.6`
- Validation: 同 A16
- Result: `0/3` 通过；`val_loss=4.4379`；未知标记总数 `10`
- Conclusion: 未知标记更少，但通过数没提升，`A14@0.5` 仍是更直接的 carry-forward 候选

### A18

- Change: 现有 `byte-level 50M / 2000step` checkpoint 按新严格口径复核
- Validation: 同 A16，温度 `0.6`
- Result: `0/3` 通过；`val_loss=1.4395`；未知标记总数 `0`
- Conclusion: 它的问题不是未知标记，而是大量伪词；严格口径下依然不够好

### A19

- Change: 现有 `byte-clean 50M / 500step` checkpoint 按新严格口径复核
- Validation: 同 A16，温度 `0.6`
- Result: `1/3` 通过；`val_loss=2.2684`；未知标记总数 `0`
- Conclusion: 清洗 `<|endoftext|>` 的收益是真实的，但 byte-level 主线的文本质量上限仍然偏低

### A20

- Change: 把 `A14` 的 `context=128 + warmup=50` SentencePiece 路线拉长到 `600 step`
- Validation: 同 A16，温度 `0.5`
- Result: `best_val_loss=3.9462`；严格校验 `0/3`；未知标记总数 `14`
- Conclusion: 单纯继续训练能显著降 loss，但没有把文本质量一起拉上来，当前瓶颈更像 tokenizer/解码质量而不是步数不够

## Validation

- 当前短结论：
- 严格校验后，旧 `32M / 1000step` 基线只在 `temperature=0.6` 下有 `1/3` 通过
- 在当前 `200 step` 窗口里，结构 sweep 没有出现比基线更强的明显赢家
- `context=384` 和 `batch=12` 都能降低 loss，但还没有转化成更干净的文本
- 到目前为止，唯一在短训里真正改善通过数的是 `warmup=50`
- byte-level 清洗线可以达到 `1/3`，但人类可读性仍然不如 SentencePiece 线
- A20 说明继续堆当前 `SentencePiece 2048` 路线的步数，收益主要体现在 loss，而不是真实 sample 质量
- 新 tokenizer 路线 `u4096 + byte_fallback` 已经把未知标记稳定压到 `0`
- 当前最佳 checkpoint 来自 `A24` 的 `step 1200`

## Conclusion

- 当前主线先不再继续放大模型或 context
- A14 经过温度复核后仍保留为最佳 SentencePiece carry-forward 候选
- 本轮 20 个 attempt 已完成；下一轮最值得做的是 tokenizer 路线重开，而不是继续在当前配置上硬拉步数
- tokenizer 路线已经重开并验证成功，当前最佳主线是 `u4096 + byte_fallback + context128 + warmup50`

## Next Phase

- 继续使用 `TinyStories` 全量数据集
- 优先尝试 `SentencePiece + byte_fallback`
- 同时放开 `vocab_size` 和 `max_sentence_length`
- 新一轮实验目标是先消掉 `⁇ / <unk>`，再看 sample 是否真正提升
- 当前新增问题：TUI / chat prompt 格式与 TinyStories 分布不一致，需要同步修正推理入口与验证相关性
- 当前新增问题：plain-story LM 已会写故事，但还没充分学会“围绕给定开头继续写”，需要把数据形态也对齐到 continuation 任务

### Inference / Eval Alignment

- TUI 已切到 TinyStories 自动 story mode，避免继续用 chat transcript 包装 prompt
- `validate_checkpoint` 默认 prompt 已切到完整 TinyStories 风格起始句，并新增 prompt keyword 命中检查
- 新口径复核结果：
- `A24`: `best_val_loss=3.5205`，相关性严格校验 `1/3`
- `A25`: `best_val_loss=3.5658`，相关性严格校验 `1/3`
- 新结论：之前的 `3/3` 更像是“文本像故事”，不是“文本继续了这个 prompt 的故事”
- 当前 plain-story 最强相关性基线是 `2/3`
- 下一轮 sweep 需要优先提升条件跟踪能力，而不是只继续压低 token loss

### A21

- Change: `SentencePiece unigram 4096 + byte_fallback + max_sentence_length=16384`，训练配置沿用 `context=128 + warmup=50`，`300 step`
- Validation: `run_sweep_attempt`，固定 3 prompt，`temperature=0.5`
- Result: `best_val_loss=4.2806`；严格校验 `3/3`；未知标记总数 `0`
- Conclusion: tokenizer 方向命中根因，当前最优主线已经从旧 `spm-2048` 切换到 `u4096 + byte_fallback`

### A22

- Change: 继续使用 `A21` 的 tokenizer 路线，把训练拉长到 `600 step`
- Validation: `run_sweep_attempt`，固定 3 prompt，`temperature=0.5`
- Result: `best_val_loss=3.9471`；严格校验 `3/3`；未知标记总数 `0`
- Conclusion: 更长训练没有破坏文本质量，当前最佳主线已经稳定

### A23

- Change: 继续使用 `A22` 的 tokenizer 路线，把训练拉长到 `1000 step`
- Validation: `run_sweep_attempt`，固定 3 prompt，`temperature=0.5`
- Result: `best_val_loss=3.6301`；严格校验 `3/3`；未知标记总数 `0`
- Conclusion: 当前最优路线继续单调改善，说明这条 tokenizer 主线还没到头

### A24

- Change: 继续使用 `A23` 的 tokenizer 路线，把训练拉长到 `1500 step`
- Validation: `run_sweep_attempt`，固定 3 prompt，`temperature=0.5`
- Result: `best_val_loss=3.5205`；严格校验 `3/3`；未知标记总数 `0`
- Conclusion: 这是目前最优 checkpoint；质量继续改善，但固定学习率在后段开始轻微抖动

### A25

- Change: 维持 `A24` 的 tokenizer 路线不变，只加入 cosine decay，`min_learning_rate=2.5e-5`
- Validation: `run_sweep_attempt`，固定 3 prompt，`temperature=0.5`
- Result: `best_val_loss=3.5658`；严格校验 `3/3`；未知标记总数 `0`
- Conclusion: decay 没有赢过常数学习率，当前最佳仍然是 `A24`

### A26

- Change: 维持 `u4096 + byte_fallback` 路线不变，把 `context_size` 提到 `256`，`300 step`
- Validation: `run_sweep_attempt`，完整 TinyStories prompt，`temperature=0.5`
- Result: `best_val_loss=4.0589`；相关性严格校验 `2/3`
- Conclusion: 更长 context 比继续压 loss 更有利于 prompt 跟踪，plain-story 线的相关性基线提升到 `2/3`

### A27

- Change: 维持 `u4096 + byte_fallback` 路线不变，把模型放大到约 `64M`，`300 step`
- Validation: `run_sweep_attempt`，完整 TinyStories prompt，`temperature=0.5`
- Result: `best_val_loss=4.6838`；相关性严格校验 `1/3`
- Conclusion: 在当前训练预算下，直接放大模型没有带来条件跟踪收益，当前更值得押注任务对齐而不是盲目增参

### A28

- Change: 沿着 `A26` 的 `context=256` 路线继续拉长到 `600 step`
- Validation: `run_sweep_attempt`，完整 TinyStories prompt，`temperature=0.5`
- Result: `best_val_loss=3.8493`；相关性严格校验 `1/3`
- Conclusion: 单纯继续训练会继续压低 loss，但会再次损伤 prompt 相关性；plain-story 线已经出现“更会写故事，但不更会续写”的分叉

### A29

- Change: 沿着 `A28` 的 `context=256` 路线加入 cosine decay，`min_learning_rate=2.5e-5`
- Validation: `run_sweep_attempt`，完整 TinyStories prompt，`temperature=0.5`
- Result: `best_val_loss=3.9371`；相关性严格校验 `0/3`
- Conclusion: decay 没能修复相关性回落，说明问题不只是后段学习率过高

### A30

- Change: 维持 `context=256` 路线，tokenizer 改为 `SentencePiece unigram 8192 + byte_fallback`，`300 step`
- Validation: `run_sweep_attempt`，完整 TinyStories prompt，`temperature=0.5`
- Result: `best_val_loss=4.1106`；相关性严格校验 `2/3`
- Conclusion: 更大词表在不增大太多模型参数的前提下提升了 plain-story 路线的综合表现，当前最佳 plain-story checkpoint 从 `A26` 切换到 `A30`

### A31

- Change: 切到显式 `Prompt/Continuation` 训练格式，`u4096 + byte_fallback + context128`，`300 step`
- Validation: `run_sweep_attempt`，通过 `Prompt: ... / Continuation:` 模板做推理，`temperature=0.3`
- Result: `best_val_loss=4.1340`；相关性严格校验 `0/3`
- Conclusion: 任务对齐方向本身没错，但在当前短训练窗口里这条线会先塌到高重复输出；下一轮要优先解决 repetition，而不是立刻判定这条路线失败
