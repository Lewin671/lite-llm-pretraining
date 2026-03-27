# Collaboration Docs Thread

- Status: closed
- Scope: 仓库协作文档、进度记录规则、归档方式

## Log

### 2026-03-28 - Topic-based progress threads

- Goal: 找到比按月归档更适合 LLM 检索的组织方式
- Change: 放弃月度归档，改为 `PROGRESS.md` 入口索引加 `progress/threads/` 主题线程
- Validation: 复核 `PROGRESS.md` 与 `AGENTS.md` 是否都指向主题线程模式
- Issue: 暂无
- Conclusion: 按主题拆分比按时间拆分更利于 review 和后续续写
- Next: 后续新任务按主题新建线程文件，而不是继续扩展单一历史文件

### 2026-03-28 - Progress archival structure

- Goal: 避免 `PROGRESS.md` 随长期迭代持续膨胀
- Change: 将 `PROGRESS.md` 收缩为当前上下文文件，并引入历史归档思路
- Validation: 复核 `AGENTS.md` 与归档路径约定是否一致
- Issue: 按月切分虽然控制了大小，但相关性仍然不够好
- Conclusion: 时间分桶不如主题分桶适合后续 agent 检索
- Next: 改为主题线程模式

### 2026-03-28 - Progress format simplification

- Goal: 把进度记录改成更适合长期维护的格式
- Change: 删除独立模板区，改为固定状态头部加倒序工作日志
- Validation: 复核 `PROGRESS.md` 与 `AGENTS.md` 的记录规则是否一致
- Issue: 单文件虽然更轻，但长期仍会变大
- Conclusion: 还需要继续降低单文件的增长速度
- Next: 引入更细粒度的拆分方式

### 2026-03-28 - Progress tracking unification

- Goal: 合并当前状态和历史尝试记录入口
- Change: 删除 `STATUS.md` 与 `ATTEMPTS.md`，统一改为 `PROGRESS.md`
- Validation: 检查旧文件引用已全部切换到 `PROGRESS.md`
- Issue: 合并后初版格式仍偏重，不利于长期维护
- Conclusion: 统一入口是对的，但格式还需要继续简化
- Next: 改成更轻的记录结构

### 2026-03-28 - Collaboration docs initialization

- Goal: 为仓库建立最小协作约束和记录机制
- Change: 新增 `AGENTS.md`、补充 `README.md`，并建立进度记录文件
- Validation: 复核文档内容，并确认相关改动已独立提交
- Issue: 将当前状态和历史尝试拆成两个文件，使用上偏重
- Conclusion: 更适合合并成一个统一入口文件
- Next: 使用单一入口管理进度记录
