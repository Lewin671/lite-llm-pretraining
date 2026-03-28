# PROGRESS.md

本文件是当前项目进度入口，只保留当前上下文和主题索引。

## Current Status

- Current Task: 用新的 TinyStories prompt suite 作为北极星继续做条件续写优化
- Latest Progress: 已落 `24` 条结构化评测集和 suite 汇总脚本；`A36` 与 `A40` 在新口径下都还是 `0/24`，但 `A40` 的前段锚点命中率从 `0.0` 升到 `0.0417`
- Issues: 当前模型几乎完全学不会稳定保留名字、物体和场景锚点；同时 `A40` 的 example-aligned 窗口只有 `1091` 条，短训很可能还没把截短路线训透
- Next Step: 提交新的评测集与基线结果，然后直接把 `A40` 这条截短 continuation 路线拉长训练，继续用 suite 复核

## How To Read

- 默认先读本文件
- 需要细节时，只打开相关主题线程
- 不把长历史直接堆在本文件里

## Recording Rules

- `Issues` 只记录当前仍未解决的阻塞、风险或待确认的关键决策
- `Issues` 非空时，相关事项必须在 `Open Threads` 中有对应主题线程
- 主题线程负责记录背景、验证、问题、结论和后续动作

## Open Threads

- [progress/threads/tinystories-full-training.md](/Users/qingyingliu/Code/lite-llm-pretraining/progress/threads/tinystories-full-training.md)
- [progress/threads/tinystories-optimization-sweep.md](/Users/qingyingliu/Code/lite-llm-pretraining/progress/threads/tinystories-optimization-sweep.md)
- [progress/threads/tinystories-eval-suite.md](/Users/qingyingliu/Code/lite-llm-pretraining/progress/threads/tinystories-eval-suite.md)

## Closed Threads

- [progress/threads/dataset-research-50m.md](/Users/qingyingliu/Code/lite-llm-pretraining/progress/threads/dataset-research-50m.md)
- [progress/threads/chat-tui-layering.md](/Users/qingyingliu/Code/lite-llm-pretraining/progress/threads/chat-tui-layering.md)
- [progress/threads/x-search-local-dataset-fit.md](/Users/qingyingliu/Code/lite-llm-pretraining/progress/threads/x-search-local-dataset-fit.md)
- [progress/threads/streaming-sampling.md](/Users/qingyingliu/Code/lite-llm-pretraining/progress/threads/streaming-sampling.md)
- [progress/threads/full-local-run.md](/Users/qingyingliu/Code/lite-llm-pretraining/progress/threads/full-local-run.md)
- [progress/threads/mlx-minimal-loop.md](/Users/qingyingliu/Code/lite-llm-pretraining/progress/threads/mlx-minimal-loop.md)
- [progress/threads/research-mac-16gb.md](/Users/qingyingliu/Code/lite-llm-pretraining/progress/threads/research-mac-16gb.md)
- [progress/threads/readme-sync-rule.md](/Users/qingyingliu/Code/lite-llm-pretraining/progress/threads/readme-sync-rule.md)
- [progress/threads/agent-workflow.md](/Users/qingyingliu/Code/lite-llm-pretraining/progress/threads/agent-workflow.md)
- [progress/threads/progress-issue-tracking.md](/Users/qingyingliu/Code/lite-llm-pretraining/progress/threads/progress-issue-tracking.md)
- [progress/threads/collaboration-docs.md](/Users/qingyingliu/Code/lite-llm-pretraining/progress/threads/collaboration-docs.md)
