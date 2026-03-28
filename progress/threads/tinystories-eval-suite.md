# TinyStories Eval Suite

## Goal

- 设计一个比当前 `3` 条 prompt 更稳健的 TinyStories 条件续写评测集
- 评测目标对齐“高要求 tiny LLM”：既保留 prompt 中的人名和关键物体，也要生成自然、低重复、像故事的续写

## Current Plan

- 落一个 `20-50` 条的 prompt suite，覆盖名字、动物、物体、地点、状态变化、目标行为
- 给每条 prompt 增加结构化锚点，方便自动打分
- 增加评测脚本，输出通过率、锚点命中率、前段命中率和质量约束汇总
- 用新评测复核当前 TinyStories checkpoints，作为后续训练优化的新北极星

## Implemented

- 新增 [prompts/tinystories_eval_suite_v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/prompts/tinystories_eval_suite_v1.json)，共 `24` 条 prompt
- 每条样本都带有 `tags` 和 `anchors`，锚点按 `name / object / setting / animal` 等分组
- 新增 [evaluate_prompt_suite.py](/Users/qingyingliu/Code/lite-llm-pretraining/lite_llm_pretraining/evaluate_prompt_suite.py)，输出：
  - 严格通过率
  - 锚点组命中率
  - 前 `40` 词早段锚点命中率
  - 按 tag 聚合的分项统计

## Baseline

- `A36` 报告已保存到 [a36-suite-v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/progress/artifacts/eval-suite/a36-suite-v1.json)
- `A40` 报告已保存到 [a40-suite-v1.json](/Users/qingyingliu/Code/lite-llm-pretraining/progress/artifacts/eval-suite/a40-suite-v1.json)
- 当前新基线：
  - `A36`: `strict_pass_rate=0.0`，`anchor_group_hit_ratio=0.0127`，`early_anchor_hit_rate=0.0`
  - `A40`: `strict_pass_rate=0.0`，`anchor_group_hit_ratio=0.0127`，`early_anchor_hit_rate=0.0417`

## Conclusion

- 这套 suite 比原来的 `3` 条 prompt 更能区分“看起来像故事”和“真的保留 prompt 锚点”
- 当前模型在名字、物体、场景三类锚点上仍然几乎完全失效
- `A40` 至少在早段锚点命中上出现了微弱改善，所以后续优化可以继续沿这条截短 continuation 路线追踪
