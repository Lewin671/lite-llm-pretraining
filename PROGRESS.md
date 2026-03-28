# PROGRESS.md

本文件是当前项目进度入口，只保留当前上下文和主题索引。

## Current Status

- Current Task: 基于 `TinyStories` 全量数据集继续做任务对齐优化，提升“按给定开头续写”的相关性
- Latest Progress: 已确认 committed `A30 (context=256 + unigram8192 + byte_fallback)` 仍是当前最稳的 best；prompted continuation 路线在 `0.4/0.5` 下能恢复到 `2/3`，但新跑的 `A34 (context256 + 600step)` 又掉回 `0/3`
- Issues: 两条主线都还没拿到稳定 `3/3`；prompted continuation 对训练长度和解码温度非常敏感，稳定性不足
- Next Step: 继续围绕 prompted continuation 做稳态优化，优先找出为什么 `300 step @ 0.5` 能到 `2/3`，但 `600 step @ 0.5` 会再次失稳

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
