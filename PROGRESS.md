# PROGRESS.md

本文件是当前项目进度入口，只保留当前上下文和主题索引。

## Current Status

- Current Task: 已落地 `Dolly 15k -> Q/A 训练 -> SQuAD 风格评测 -> Q/A TUI` 主线，下一步转到 `Q/A` 质量优化
- Latest Progress: 新增了 `prepare_dolly_qa.py`、`prepare_squad_qa_eval.py`、`evaluate_qa_suite.py`、通用 `evaluate_suite.py`，并把 `sample.py`、`tui_chat.py`、训练内 sample 和 `run_local.py` 全部接到真正的 `Question / Context / Answer` 模式；`configs/dolly-qa-spm-smoke.json` 已完成一次本机 smoke 闭环
- Issues: 当前 `Dolly Q/A smoke` 只能证明链路打通，质量仍然很弱；当前 `best_suite` 在 `SQuAD dev/holdout v1` 上都是 `strict_pass_rate=0.0`，`token_f1` 仅约 `0.0281 / 0.0340`
- Next Step: 以当前 `Q/A` 主线为基线开始做质量优化，优先收紧训练数据形态与损失设计，而不是继续沿故事续写路线调参

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
