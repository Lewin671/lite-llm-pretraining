# Research Mac 16GB Thread

- Status: closed
- Scope: 调研 Mac 16GB 上可运行的 lite LLM 预训练或持续预训练最小闭环方案，并选出当前仓库的最小可行路线

## Log

### 2026-03-28 - Start Mac 16GB feasibility research

- Goal: 在实现前先收集真实可跑经验，优先确认哪些框架和配置更接近当前机器上的最小闭环
- Plan:
  - 优先尝试从 `x.com` 搜索带配置、资源占用和结果的经验贴
  - 对候选方案用 GitHub、Hugging Face、官方文档或技术博客交叉验证
  - 形成至少 3 个候选方案表，并明确 1 个继续采用的最小可行方案
- Validation: 待执行
- Issue: 暂无
- Conclusion: 待调研
- Next: 开始外部调研并把候选方案、失败模式和取舍记录到本线程

### 2026-03-28 - Candidate survey and route selection

- Goal: 基于外部资料和本机环境，把 Mac 16GB 上可跑的训练路线收敛到一个当前仓库可落地的最小方案
- Change:
  - 尝试通过浏览器打开 `https://x.com/search?q=mac%2016gb%20llm%20pretraining&src=typed_query&f=live`
  - 记录到两个事实：
    - 浏览器页面出现 JavaScript 降级提示，`wait --load networkidle` 超时
    - 该精确关键词在当前公开搜索页直接返回 `No results`
  - 回退到 GitHub、官方文档和公开社区资料继续交叉验证
  - 补充本机约束检查：
    - 当前机器为 `MacBookPro18,1 / Apple M1 Pro / 16 GB`
    - `python3` 为 `3.14.2`
    - `mlx==0.31.1` 与 `torch==2.11.0` 都存在 `cp314 arm64` wheel，可直接安装
- Validation:
  - `agent-browser` 可打开 `x.com` 搜索页，但网络空闲等待超时，且页面正文显示 JavaScript 降级提示
  - `python3 -m pip download --no-deps --only-binary=:all: mlx==0.31.1`
  - `python3 -m pip download --no-deps --only-binary=:all: torch==2.11.0`

## Candidate Table

| 方案名 | 训练框架 | 是否适合 Mac 16GB | 主要风险 | 最小可验证实验 | 结论 |
| --- | --- | --- | --- | --- | --- |
| MLX 官方 `transformer_lm` 缩小版 | `MLX + Metal` | 适合。MLX 明确面向 Apple silicon，官方例子直接包含 decoder-only LM 训练。默认配置过大，但缩小后非常接近当前仓库目标。 | 官方样例默认 `context_size=1024`、`dim=1024`、`num_blocks=12`、`batch_size=2`，对 16GB 最小闭环偏重；需要缩小模型和上下文，否则首次实验成本高。 | 用 byte-level 或极小文本集，`context=128`、`dim=256`、`layers=4`、`heads=4`、`batch=16` 左右，先验证 loss 下降、checkpoint 和 sample。 | 继续采用 |
| 社区 `gpt2-mlx` / `mlx-gpt2` 最小 GPT 路线 | `MLX + Metal` | 基本适合。`gpt2-mlx` 明确写到可在 M1 Pro 16GB 上实时跑 GPT-2 XL 推理，并支持从零训练自定义 GPT；`mlx-gpt2` 给出更小的字符级训练脚本。 | 社区实现质量不一，部分仓库偏演示代码；直接照搬容易把 toy 代码带进仓库。 | 吸收其 checkpoint、sample 和小模型训练方式，但自己实现更小、更干净的脚本。 | 暂缓，吸收思路不直接采用 |
| `nanoGPT` 缩小版 + PyTorch MPS | `PyTorch + MPS` | 可跑。`nanoGPT` README 明确给出 “I only have a macbook” 的小配置，并建议 Apple Silicon 使用 `--device=mps`；PyTorch 官方也确认 `mps` 可在 macOS 上做 GPU 训练。 | MPS 仍有后端不完整、算子回退和稳定性不确定的问题；仓库当前从零起步，直接走 PyTorch 会引入更多依赖和兼容面。 | 复刻 `train_shakespeare_char` 的缩小配置，先看 100~200 step 的 loss 和吞吐。 | 暂缓，作为备选 |
| tinygrad / Metal 自搭路线 | `tinygrad + METAL` | 理论可跑，官方 README 明确支持 `METAL`。 | 数据、tokenizer、checkpoint、采样周边都要自己补，当前阶段实现风险高，信息增益不如 MLX。 | 只有在 MLX 与 PyTorch 都被环境或稳定性卡住时，再做最小对照实验。 | 放弃当前阶段采用 |

## Key Evidence

- `x.com` 搜索阶段：
  - 当前公开搜索页对 `mac 16gb llm pretraining` 没给出直接结果，且页面正文显示 JavaScript 降级提示，因此本轮没有从 `x.com` 提取到足够高信号候选。
- MLX 官方信号：
  - `ml-explore/mlx` README 明确说明 MLX 是 “machine learning on Apple silicon”，并强调 unified memory；`mlx-examples` README 列出官方 `Transformer language model` 训练例子。
  - `mlx-examples/transformer_lm/main.py` 默认参数显示这条路线已经覆盖 `context_size`、`batch_size`、训练步数、验证与 test ppl 输出，只需要为 16GB 缩小配置。
- MLX 社区信号：
  - `gpt2-mlx` README 明确写到 “GPT-2 XL 1.5B real-time text generation on M1 Pro 16GB”，同时支持 train custom GPT-style models from scratch。
  - `gpt2-mlx/train.py` 使用 `batch_size=2 + grad_accumulation_steps=4 + checkpoint 保存`，说明作者已经按小显存/统一内存思路组织训练。
  - `mlx-gpt2/train.py` 给出更小的字符级脚本，包含 train/val 和 completions 输出，适合作为最小闭环形态参考。
- PyTorch/MPS 信号：
  - PyTorch 官方 `MPS backend` 文档确认 `mps` 设备可在 macOS 上做 GPU 训练。
  - `nanoGPT` README 明确说 character-level Shakespeare 适合 “macbooks and such”，并额外建议 Apple Silicon 使用 `--device=mps`，可带来约 `2-3X` 加速。
  - 同一份 README 也说明在低资源机器上必须显著减小 `block_size`、`batch_size`、`n_layer`、`n_head` 和 `n_embd`。
- 16GB 经验边界：
  - 一则 MLX 本地微调公开讨论里提到 `16GB is minimum`，在 `16GB M1 MacBook Air` 上可训练但较慢，说明 16GB 路线成立，但实验必须先小后大。

## Decision

- 当前最小可行方案：
  - 采用 `MLX + Metal`
  - 不直接照抄官方默认大配置，而是做一个“官方 `transformer_lm` 思路 + 仓库内极小 byte-level 数据闭环”的缩小实现
  - 首轮目标不是追求 tokenizer/BPE 完整度，而是优先证明：
    - 本机可稳定启动训练
    - loss 有下降趋势
    - 能保存并加载 checkpoint
    - 能做基础 sample
- 为什么不是先上 PyTorch：
  - 当前仓库还没有任何训练代码，优先选 Apple silicon 原生、依赖更少、环境确定性更高的 MLX 更符合最小闭环目标
- 为什么不是先上 tinygrad：
  - 当前主要不确定性是“能否最短时间形成稳定闭环”，不是“能否进一步缩到最少代码”；tinygrad 在这个阶段实现成本偏高

## Sources

- `https://x.com/search?q=mac%2016gb%20llm%20pretraining&src=typed_query&f=live`
- `https://github.com/ml-explore/mlx`
- `https://github.com/ml-explore/mlx-examples`
- `https://raw.githubusercontent.com/ml-explore/mlx-examples/main/transformer_lm/main.py`
- `https://raw.githubusercontent.com/dx-dtran/gpt2-mlx/main/README.md`
- `https://raw.githubusercontent.com/dx-dtran/gpt2-mlx/main/train.py`
- `https://raw.githubusercontent.com/pranavjad/mlx-gpt2/main/train.py`
- `https://docs.pytorch.org/docs/stable/notes/mps.html`
- `https://raw.githubusercontent.com/karpathy/nanoGPT/master/README.md`
- `https://raw.githubusercontent.com/karpathy/nanoGPT/master/config/train_shakespeare_char.py`
- `https://www.reddit.com/r/LocalLLaMA/comments/191s7x3/a_simple_guide_to_local_llm_finetuning_on_a_mac/`

- Issue:
  - `x.com` 公开搜索结果质量不足，当前不能把核心判断建立在 X 单一来源上
- Conclusion: 当前最优路线是仓库内实现一个缩小版 MLX decoder-only LM 闭环，再用本机实际训练结果验证这条判断
- Next: 调研线程完成，后续若要扩展到 BPE 或更大数据集，再开新线程继续对比方案
