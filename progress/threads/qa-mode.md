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
  - `Dolly 15k` 中更接近简单问答的主要是 `open_qa / closed_qa / information_extraction`
  - 在这几个类别里，加上 `max_answer_words=12`、`max_question_words=24`、`require_single_line_answer=true` 后，剩余 `2181` 条样本；按 `train_split=0.95` 落地后得到 `2071 train / 110 val`
- 新增：
  - [prepare_dolly_qa_eval.py](/Users/qingyingliu/Code/lite-llm-pretraining/lite_llm_pretraining/prepare_dolly_qa_eval.py)
    - 从 `Dolly` JSONL 生成本地 `simple QA dev/holdout` suite
  - 新配置：
    - [dolly-qa-simple-spm-screen.json](/Users/qingyingliu/Code/lite-llm-pretraining/configs/dolly-qa-simple-spm-screen.json)
    - [dolly-qa-simple-spm-c128-screen.json](/Users/qingyingliu/Code/lite-llm-pretraining/configs/dolly-qa-simple-spm-c128-screen.json)
    - [dolly-qa-simple-spm-c128-cw32-screen.json](/Users/qingyingliu/Code/lite-llm-pretraining/configs/dolly-qa-simple-spm-c128-cw32-screen.json)
    - [dolly-qa-simple-spm-random-cw32-screen.json](/Users/qingyingliu/Code/lite-llm-pretraining/configs/dolly-qa-simple-spm-random-cw32-screen.json)
  - 新评测集：
    - [dolly_qa_simple_dev_v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/prompts/dolly_qa_simple_dev_v1.json)
    - [dolly_qa_simple_holdout_v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/prompts/dolly_qa_simple_holdout_v1.json)
    - [dolly_qa_simple_cw32_dev_v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/prompts/dolly_qa_simple_cw32_dev_v1.json)
    - [dolly_qa_simple_cw32_holdout_v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/prompts/dolly_qa_simple_cw32_holdout_v1.json)

## Optimization Results

- `simple QA` 基线
  - 旧 `dolly-qa-spm-smoke` 在 `cw32 dev` 上只有 `token_f1=0.0237`
  - 在 `simple holdout` 上只有 `token_f1=0.0098`
- `17M, context=256, example_start`
  - 配置：[dolly-qa-simple-spm-screen.json](/Users/qingyingliu/Code/lite-llm-pretraining/configs/dolly-qa-simple-spm-screen.json)
  - `best_suite` 出现在 `step 200`
  - `dev token_f1=0.0324`
  - `holdout token_f1=0.0117`
  - 结论：比原始 smoke 更对齐简单问答，但 sample 仍明显退化
- `17M, context=128, example_start`
  - 配置：[dolly-qa-simple-spm-c128-screen.json](/Users/qingyingliu/Code/lite-llm-pretraining/configs/dolly-qa-simple-spm-c128-screen.json)
  - 虽然 `val_loss` 更低，但 `suite_f1` 长时间为 `0`
  - 结论：单独压短 context 不够，很多样本虽然更短，但仍会被 `example_start` 和长 context 挤掉答案
- `17M, context=128, context_word_limit=32, random`
  - 配置：[dolly-qa-simple-spm-random-cw32-screen.json](/Users/qingyingliu/Code/lite-llm-pretraining/configs/dolly-qa-simple-spm-random-cw32-screen.json)
  - `best_suite` 出现在 `step 300`
  - `cw32 dev token_f1=0.0425`
  - `cw32 holdout token_f1=0.0070`
  - 结论：这是当前 `dev` 上最强的一条，但泛化仍然很差

## Key Findings

- `example_start` 对短 QA 数据不友好
  - `context_size=256` 时，`example-aligned windows` 只有 `195`
  - 压到 `128` 后虽然更多，但许多样本依然太短或答案太晚进入窗口
- `context_word_limit` 明显影响答案能否进入训练窗口
  - 在当前 simple QA 数据上，`context_word_limit=32` 时，答案能在前 `128` token 内出现的样本约 `1909 / 2071`
- 单纯把 `val_loss` 压低不等于简单问答变好
  - 当前最优 `dev` 指标来自 `random + cw32`
  - 但真实采样仍然会退化成 `The the ...`

## Current Conclusion

- 当前最好的路不是继续沿 `SQuAD` 或全量 Dolly 硬堆步数，而是：
  - 更窄的简单问答子集
  - 让答案更早进入训练窗口
  - 避免 `example_start` 把短样本直接排除
- 但当前 best 仍然达不到“简单问题回答好”：
  - `strict_pass_rate=0.0`
  - `exact_match=0.0`
  - 手工简单题采样仍会生成重复短语

## Next Optimization

- 下一轮主线切到 `open_qa-only`
- 目标：
  - 先把无 `context` 的短事实问答做对
  - 不再让 `closed_qa` 和 `information_extraction` 混进主训练口径
- 计划：
  - 生成 `open_qa-only` 的 `dev/holdout` suite
  - 跑一条小模型 `open_qa-only + short answer + loss_window` 基线
  - 只要 `exact_match` 或 `strict_pass_rate` 首次变成非零，就优先沿这条线继续

## Open QA-Only

- 数据集分析：
  - `open_qa` 原始样本 `3742`
  - 加上 `max_answer_words=12`、`max_question_words=24`、`require_single_line_answer=true` 后，剩余 `1173`
  - 平均答案长度约 `5.43` 词，平均问题长度约 `9.03` 词
- 新增：
  - [dolly_open_qa_short_dev_v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/prompts/dolly_open_qa_short_dev_v1.json)
  - [dolly_open_qa_short_holdout_v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/prompts/dolly_open_qa_short_holdout_v1.json)
  - [dolly-open-qa-short-c096-compact.json](/Users/qingyingliu/Code/lite-llm-pretraining/configs/dolly-open-qa-short-c096-compact.json)
  - [prepare_dolly_qa_eval.py](/Users/qingyingliu/Code/lite-llm-pretraining/lite_llm_pretraining/prepare_dolly_qa_eval.py) 现在支持 `suite_name_prefix`

## Open QA Results

- 配置：
  - `open_qa-only`
  - `context_size=96`
  - `dim=256`
  - `num_layers=4`
  - `num_heads=4`
  - `batch_sampling_mode=loss_window`
- 结果：
  - `best_suite` 出现在 `step 500`
  - `dev token_f1=0.1077`
  - `holdout token_f1=0.0949`
  - `exact_match=0.0`
  - `strict_pass_rate=0.0`
- 报告：
  - [dolly-open-qa-short-dev-v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/progress/artifacts/qa-mode/dolly-open-qa-short-dev-v1.json)
  - [dolly-open-qa-short-holdout-v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/progress/artifacts/qa-mode/dolly-open-qa-short-holdout-v1.json)

## Open QA Conclusion

- `open_qa-only` 明显比混合 `simple QA` 更对齐目标
- 这是当前第一次把 `holdout token_f1` 推近 `0.1`
- 但当前瓶颈已经从“任务不对齐”变成“答案表面形态不稳定”：
  - 模型会生成看起来像答案的短语
  - 但仍然经常是伪词、错实体或半对半错
