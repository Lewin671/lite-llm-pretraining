# Dataset Research 50M Thread

- Status: closed
- Scope: 用浏览器调研适合当前 `M1 Pro 16GB` 机器训练约 `50M` 参数模型的数据集选择，并给出可落地建议

## Log

### 2026-03-28 - Start 50M dataset research

- Goal: 为当前仓库下一阶段找出适合 `50M` 参数本地训练的数据集，不只追求能跑，还要兼顾收敛、清洁度和落地成本
- Plan:
  - 用浏览器打开候选数据集和相关项目页面，优先查官方说明或数据集主页
  - 重点比较 `TinyStories`、小型文学/故事集，以及更大网页语料的轻量子集是否适合 `50M` 模型
  - 结合当前机器和仓库约束，收敛成推荐数据集与阶段化路线
- Validation: 待执行
- Issue: 暂无
- Conclusion: 待调研
- Next: 打开候选数据集页面并提取关键信息

### 2026-03-28 - Browser findings and recommendation

- Goal: 用浏览器确认 `50M` 模型应该优先选哪类数据集，而不是只凭经验拍脑袋
- Change:
  - 打开 `roneneldan/TinyStories` 数据集页，提取到：
    - `2.14M` 行
    - 下载体积约 `1 GB`
    - 数据集是“synthetically generated short stories that only use a small vocabulary”
    - 页面列出的已训练模型覆盖 `1M/3M/8M/28M/33M` 等小模型尺度
    - 数据集页还说明存在更大的 `TinyStoriesV2-GPT4-train.txt`
  - 打开 `TinyStories` 论文页，提取到：
    - 该数据集专门用于训练远小于主流规模的 LM
    - 文中明确说可用于 `below 10 million total parameters` 的模型，并让小模型产出连贯英文
  - 打开 `nanoGPT` README，提取到：
    - `Tiny Shakespeare` 只是 `1MB` 的 quick start toy 数据
    - `OpenWebText` 对应复现 `GPT-2 124M`，并明确写到需要 `8xA100 40GB`
    - 对 Macbook，只建议显著缩小 block size、batch 和模型规模
  - 打开 `OpenWebText2` 文档页，提取到：
    - 清洗版规模约 `17,103,059` 文档
    - 约 `65.86 GB` 原始文本，`28 GB` 压缩包
  - 打开 DeepMind 关于 Chinchilla 的文章，提取到：
    - 训练预算取决于模型大小和 token 数
    - 同等计算预算下，更小模型配更多数据可能更优
- Validation:
  - `agent-browser open 'https://huggingface.co/datasets/roneneldan/TinyStories'`
  - `agent-browser open 'https://arxiv.org/abs/2305.07759'`
  - `agent-browser open 'https://raw.githubusercontent.com/karpathy/nanoGPT/master/README.md'`
  - `agent-browser open 'https://openwebtext2.readthedocs.io/en/latest/'`
  - `agent-browser open 'https://deepmind.google/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/'`
- Issue:
  - 部分 Hugging Face 页面在当前浏览器环境里偶发 `ERR_ABORTED`，但已从可稳定打开的页面提取到足够结论
- Conclusion:
  - 对当前这台 `M1 Pro 16GB` 和当前仓库阶段，`50M` 模型最合适的数据集不是 `Tiny Shakespeare`，也不是整套 `OpenWebText2`
  - 第一推荐应为 `TinyStories` 或其更干净的 GPT-4 变体
  - 如果后续要提升语域覆盖面，应在第二阶段引入“更大网页语料的小型清洗子集”，而不是一开始就上全量网页语料
  - 从数据量角度，`50M` 模型不该只喂 `1MB` 级 toy 数据；首轮至少应进入“几亿 token”量级的数据子集训练
- Next:
  - 如进入实现阶段，优先新增 `TinyStories` 数据准备脚本和一个 `50M` 级训练配置
