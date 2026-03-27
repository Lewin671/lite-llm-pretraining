# Agent Workflow Thread

- Status: closed
- Scope: supervisor 工作流设计、浏览器调研优先级、持续推进与失败处理规则

## Log

### 2026-03-28 - Add supervisor prompt

- Goal: 为持续运行的 AI agent 工作流落一版可直接使用的 supervisor 提示词
- Change: 新增 `prompts/supervisor.md`，固化使命、成功标准、X 调研优先级、失败处理与线程规则
- Validation: 复核提示词是否覆盖持续推进、失败后继续探索、不允许放弃、先用浏览器调研 `x.com`
- Issue: 暂无
- Conclusion: 当前仓库已经有一版可直接落地的 supervisor 提示词基础版
- Next: 按需继续补 `worker`、`reviewer` 提示词和任务线程模板
