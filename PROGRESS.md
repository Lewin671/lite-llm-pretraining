# PROGRESS.md

本文件是当前项目进度入口，只保留当前上下文和主题索引。

## Current Status

- Current Task: 执行 `TinyStories` 连续优化 sweep，至少完成 20 个方案尝试
- Latest Progress: 已补齐 attempt runner、SentencePiece prepare 兼容参数和更严格的未知标记校验；正在重建统一 tokenizer 数据集进入短训练矩阵
- Issues: 当前最佳样本仍不够稳定，需要通过 sweep 收敛真正有效的改进项
- Next Step: 先完成 `u2048/u4096/b2048/b4096/u1024` 统一数据集准备，再按固定短训练 protocol 跑 A05 之后的结构与超参数对照

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
