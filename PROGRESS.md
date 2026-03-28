# PROGRESS.md

本文件是当前项目进度入口，只保留当前上下文和主题索引。

## Current Status

- Current Task: 沿 `Dolly Q/A` 主线继续优化，目标先收紧成“简单问题能稳定答出来”
- Latest Progress: 已补 `Dolly simple QA` 评测、简单问答子集过滤和多条聚焦配置；当前最优是 `4.2M compact + context=128 + loss_window` 路线，`simple QA dev token_f1=0.0297`，`holdout token_f1=0.0126`
- Issues: 当前 best 仍然是 `strict_pass_rate=0.0 / exact_match=0.0`，真实采样还会退化成伪词和重复短语；说明现在虽然比原始 smoke 有进展，但离“简单问题回答好”还差很远
- Next Step: 继续围绕当前 compact simple-QA 主线优化输出质量，优先考虑更强的答案对齐，而不是再扩大数据范围

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
- [progress/threads/qa-mode.md](/Users/qingyingliu/Code/lite-llm-pretraining/progress/threads/qa-mode.md)

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
