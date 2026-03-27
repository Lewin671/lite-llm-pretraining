# README Sync Rule Thread

- Status: closed
- Scope: 明确何时更新 `README.md`，以及 `README.md` 与 `AGENTS.md` 的职责边界

## Log

### 2026-03-28 - Clarify README sync expectations

- Goal: 在实现推进过程中，必要时同步更新 `README.md`，同时避免 `README.md` 与 `AGENTS.md` 大段重复
- Change: 在 `AGENTS.md` 中补充 README 同步规则，并明确 `README.md` 面向仓库现状与使用入口，`AGENTS.md` 面向协作约束
- Validation: 复核更新后的 `AGENTS.md`，确认已覆盖“必要时同步更新 README”与“避免 README/AGENTS 重复说明”两点
- Issue: 当前规则只限制 README 内容范围，但没有明确要求在能力变化时同步维护，也没有写清与 `AGENTS.md` 的边界
- Conclusion: README 维护时机和两份文档的边界已经更明确，后续实现推进时更容易保持文档一致
- Next: 按新规则在后续功能落地时视情况同步更新 `README.md`
