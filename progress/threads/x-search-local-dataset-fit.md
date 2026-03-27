# X Search Local Dataset Fit Thread

- Status: closed
- Scope: 用浏览器在 X 上搜索 Apple M1 Pro 16GB 本机适合的 LLM 训练数据集范围，并结合当前仓库目标给出建议

## Log

### 2026-03-28 - Start X search for dataset fit

- Goal: 基于 X 上公开讨论，确认当前这台 `Apple M1 Pro 16GB` 机器更适合什么规模和类型的数据集
- Plan:
  - 用浏览器在 X 搜索与 `M1 Pro 16GB`、`MLX`、`TinyStories`、`Shakespeare`、`fine-tune`、`pretrain` 相关的公开内容
  - 记录可提取的高信号经验，包括机器配置、数据集类型、训练目标和约束
  - 结合当前仓库的本地最小闭环路线，收敛成数据集建议
- Validation: 待执行
- Issue: 暂无
- Conclusion: 待搜索
- Next: 打开 X 搜索并提取可复用经验

### 2026-03-28 - X public search result and conclusion

- Goal: 从 X 公开搜索中提取对当前机器和仓库阶段真正有帮助的数据集建议
- Change:
  - 用浏览器打开多组 X 搜索，包括：
    - `M1 Pro 16GB LLM pretraining dataset`
    - `apple silicon tinystories`
    - `ml explore mlx tinystories`
  - 页面在当前浏览器环境下多次显示 `JavaScript is not available`，且部分搜索请求直接返回 `ERR_ABORTED`
  - 在可读取的公开搜索结果里，`apple silicon tinystories` 命中一条高信号内容：
    - `2026-03-09`，`@karpathy` 回复里明确表示：`TinyStories is the right thing to train on for very small models / Apple Silicon`
    - 同一条回复还指向 `karpathy/tinystories-gpt4-clean` 数据集，并称其是更干净的版本
  - 其它更宽泛或更具体的搜索词未稳定返回更多高信号样本，公开页结果密度有限
- Validation:
  - `agent-browser open 'https://x.com/search?q=apple%20silicon%20tinystories&src=typed_query&f=live'`
  - `agent-browser get text body`
- Issue:
  - X 公开搜索在当前浏览器环境下存在 JavaScript 降级和偶发 `ERR_ABORTED`，因此本轮结论建立在少量可见公开结果上
- Conclusion:
  - 对当前这台 `Apple M1 Pro 16GB` 机器，如果目标是从零训练或持续预训练一个小模型，最适合的是 `TinyStories` 这类干净、小规模、短文本、易收敛的数据集
  - 当前仓库继续使用 `Tiny Shakespeare` 做 smoke 闭环是合理的；如果往上走一阶段，优先切到 `TinyStories`，而不是直接上更杂、更大的网页语料
- Next:
  - 如需落地，实现一个 `TinyStories` 子集数据准备和训练配置线程
