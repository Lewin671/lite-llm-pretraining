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
