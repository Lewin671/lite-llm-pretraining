# QA Mode Thread

- Status: in_progress
- Scope: 为仓库新增 `Question -> Answer` 数据准备、评测和推理入口，先以 `Dolly 15k` 为训练集，补充 `SQuAD 2.0` 风格评测

## Goal

- 把当前以故事续写为中心的本地训练链路扩展到真正的 `Q/A` 任务
- 复用现有 `SentencePiece + loss mask + suite_eval + TUI` 基础设施，避免再起一套平行实现
- 让本机可以直接运行 `Dolly 15k -> 训练 -> QA 评测 -> TUI`

## Changes

- 新增 [prepare_dolly_qa.py](/Users/qingyingliu/Code/lite-llm-pretraining/lite_llm_pretraining/prepare_dolly_qa.py)
  - 下载 `Databricks Dolly 15k`
  - 统一编码成 `Question / Context / Answer`
  - 支持 `SentencePiece`、`byte_fallback`、answer-only loss mask、prompt/head 权重
- 新增 [prepare_squad_qa_eval.py](/Users/qingyingliu/Code/lite-llm-pretraining/lite_llm_pretraining/prepare_squad_qa_eval.py)
  - 从官方 `SQuAD 2.0 dev` 生成本地 `dev/holdout` 紧凑评测集
- 新增 [evaluate_qa_suite.py](/Users/qingyingliu/Code/lite-llm-pretraining/lite_llm_pretraining/evaluate_qa_suite.py) 和 [evaluate_suite.py](/Users/qingyingliu/Code/lite-llm-pretraining/lite_llm_pretraining/evaluate_suite.py)
  - 支持 `strict_pass_rate / exact_match / token_f1`
  - 训练内 `suite_eval` 可直接用于 `Q/A`
- 更新 [story_inference.py](/Users/qingyingliu/Code/lite-llm-pretraining/lite_llm_pretraining/story_inference.py)、[sample.py](/Users/qingyingliu/Code/lite-llm-pretraining/lite_llm_pretraining/sample.py)、[tui_chat.py](/Users/qingyingliu/Code/lite-llm-pretraining/lite_llm_pretraining/tui_chat.py)、[app/chat.py](/Users/qingyingliu/Code/lite-llm-pretraining/lite_llm_pretraining/app/chat.py)
  - 新增 `qa` 推理模式
  - TUI 支持 `question || context` 单行输入
- 更新 [train.py](/Users/qingyingliu/Code/lite-llm-pretraining/lite_llm_pretraining/train.py)、[run_local.py](/Users/qingyingliu/Code/lite-llm-pretraining/lite_llm_pretraining/run_local.py)、[run_sweep_attempt.py](/Users/qingyingliu/Code/lite-llm-pretraining/lite_llm_pretraining/run_sweep_attempt.py)
  - 训练内 sample、`best_suite` 选模、本地闭环和 sweep artifact 都兼容 `Q/A`
- 新增 [dolly-qa-spm-smoke.json](/Users/qingyingliu/Code/lite-llm-pretraining/configs/dolly-qa-spm-smoke.json)
- 更新 [README.md](/Users/qingyingliu/Code/lite-llm-pretraining/README.md)

## Validation

- `py_compile`
  - `lite_llm_pretraining/story_inference.py`
  - `lite_llm_pretraining/app/chat.py`
  - `lite_llm_pretraining/sample.py`
  - `lite_llm_pretraining/tui_chat.py`
  - `lite_llm_pretraining/validate_checkpoint.py`
  - `lite_llm_pretraining/evaluate_qa_suite.py`
  - `lite_llm_pretraining/evaluate_suite.py`
  - `lite_llm_pretraining/train.py`
  - `lite_llm_pretraining/run_local.py`
  - `lite_llm_pretraining/prepare_dolly_qa.py`
  - `lite_llm_pretraining/prepare_squad_qa_eval.py`
- `prepare_dolly_qa` smoke
  - `/tmp/dolly-qa-smoke`
  - `train_examples=14260`
  - `val_examples=751`
  - `train_tokens=6980095`
  - `val_tokens=360930`
- `prepare_squad_qa_eval`
  - 已生成 [squad_qa_dev_v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/prompts/squad_qa_dev_v1.json)
  - 已生成 [squad_qa_holdout_v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/prompts/squad_qa_holdout_v1.json)
- `evaluate_qa_suite`
  - 已用临时 `qa` profile 跑通真实生成路径
- `run_local --config configs/dolly-qa-spm-smoke.json --force_prepare`
  - 已完整跑通 prepare / train / suite_eval / final sample
  - `params=17,428,992`
  - `best_val_loss=6.3553`
  - `best_suite` 停在 `step 50`
- `sample --mode qa`
  - 已验证 `--context` 参数和 `qa` 模板路径可用

## Current Result

- `Dolly Q/A` 主线已经可运行，但质量仍明显不够
- 当前 smoke 基线：
  - `SQuAD dev v1`: [dolly-qa-smoke-dev-v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/progress/artifacts/qa-mode/dolly-qa-smoke-dev-v1.json)
    - `strict_pass_rate=0.0`
    - `token_f1=0.0281`
    - `exact_match=0.0`
  - `SQuAD holdout v1`: [dolly-qa-smoke-holdout-v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/progress/artifacts/qa-mode/dolly-qa-smoke-holdout-v1.json)
    - `strict_pass_rate=0.0`
    - `token_f1=0.0340`
    - `exact_match=0.0`
- 结论：这次迭代完成的是“把 Q/A 管线打通并留下可复现基线”，不是“已经做出高质量 Q/A 模型”

## Next

- 基于当前 `Dolly` 主线做质量优化
- 优先看：
  - 更窄的训练子任务范围
  - 更强的 answer-only / prompt weighting 设计
  - 更贴近 `Q/A` 的 sample 与 checkpoint 选择策略
  - 单独补一套 `Dolly simple QA` 评测，避免只用 `SQuAD` 这种偏抽取式口径

## Simple QA Optimization

- 目标收紧成“简单问题能稳定答出来”，不再只看 `SQuAD`
- 数据集分析：
  - `Dolly 15k` 中真正接近简单问答的主要是 `open_qa / closed_qa / information_extraction`
  - 这些类别里，满足“单行短答案”的样本约 `2181` 条
- 新增：
  - [prepare_dolly_qa_eval.py](/Users/qingyingliu/Code/lite-llm-pretraining/lite_llm_pretraining/prepare_dolly_qa_eval.py)
    - 从 `Dolly` 原始或预处理 JSONL 生成本地 `simple QA dev/holdout` suite
  - [prepare_dolly_qa.py](/Users/qingyingliu/Code/lite-llm-pretraining/lite_llm_pretraining/prepare_dolly_qa.py)
    - 新增 `min/max_answer_words`、`max_question_words`、`require_single_line_answer`
  - [common.py](/Users/qingyingliu/Code/lite-llm-pretraining/lite_llm_pretraining/common.py)
    - 新增 `loss_window_start_positions`
  - [train.py](/Users/qingyingliu/Code/lite-llm-pretraining/lite_llm_pretraining/train.py)
    - 新增 `batch_sampling_mode=loss_window`
    - 新增 `init_checkpoint_dir`
- 新配置：
  - [dolly-qa-simple-spm-screen.json](/Users/qingyingliu/Code/lite-llm-pretraining/configs/dolly-qa-simple-spm-screen.json)
  - [dolly-qa-simple-spm-c128-screen.json](/Users/qingyingliu/Code/lite-llm-pretraining/configs/dolly-qa-simple-spm-c128-screen.json)
  - [dolly-qa-simple-spm-c128-compact.json](/Users/qingyingliu/Code/lite-llm-pretraining/configs/dolly-qa-simple-spm-c128-compact.json)
  - [dolly-qa-simple-ft-screen.json](/Users/qingyingliu/Code/lite-llm-pretraining/configs/dolly-qa-simple-ft-screen.json)

## Optimization Results

- `simple QA suite`
  - [dolly_qa_simple_dev_v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/prompts/dolly_qa_simple_dev_v1.json)
  - [dolly_qa_simple_holdout_v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/prompts/dolly_qa_simple_holdout_v1.json)
  - `32 / 32`，答案平均约 `5-6` 个词
- `17M, context=256, from scratch`
  - best `dev token_f1=0.0283`
  - `holdout token_f1=0.0117`
  - 问题：明显过拟合，sample 退化成重复模板
- `17M, context=128, loss_window`
  - best `dev token_f1=0.0141`
  - `holdout token_f1=0.0057`
  - 结论：修采样是对的，但大模型仍然不够适合这份小数据
- `4.2M compact, context=128, loss_window`
  - best `dev token_f1=0.0297`，出现在 `step 250`
  - `holdout token_f1=0.0126`
  - 当前最优报告：
    - [dolly-qa-simple-compact-dev-v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/progress/artifacts/qa-mode/dolly-qa-simple-compact-dev-v1.json)
    - [dolly-qa-simple-compact-holdout-v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/progress/artifacts/qa-mode/dolly-qa-simple-compact-holdout-v1.json)
- `17M from full Dolly Q/A checkpoint`
  - best `dev token_f1=0.0148`
  - 没有超过 compact 从零训练

## Current Conclusion

- 当前最好的路不是“大模型 + 更长步数”，而是：
  - 更窄的简单问答子集
  - 只采样真正带监督的窗口
  - 更小、更容易学会短答案的模型
- 但当前 best 仍然达不到“简单问题回答好”：
  - `strict_pass_rate=0.0`
  - `exact_match=0.0`
  - 手工简单题采样仍会生成伪词或重复短语
