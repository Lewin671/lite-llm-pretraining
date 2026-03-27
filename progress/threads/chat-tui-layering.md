# Chat TUI Layering Thread

- Status: closed
- Scope: 为本地模型增加模型层 / 应用层分离，并提供交互式 TUI 对话入口

## Log

### 2026-03-28 - Add model/app split and interactive TUI

- Goal: 在现有本地 checkpoint 采样基础上，补一个可维护的分层结构，让应用层负责对话，模型层负责生成，并提供交互式终端入口
- Change:
  - 新增 `lite_llm_pretraining/model/engine.py`
  - 新增 `lite_llm_pretraining/app/chat.py`
  - 新增 `lite_llm_pretraining/tui_chat.py`
  - 将现有采样逻辑改为复用模型层
  - 在公共生成逻辑中增加 `include_prompt`，让 CLI 采样与对话生成复用同一底层流式路径
- Validation:
  - `source .venv/bin/activate && python -m compileall lite_llm_pretraining`
  - `source .venv/bin/activate && python - <<'PY' ... ChatApplication(...).stream_reply(...) ... PY`
  - 启动 `python -m lite_llm_pretraining.tui_chat --checkpoint_dir checkpoints/tinyshakespeare-byte-smoke/best --max_new_tokens 40`
  - 在 TUI 内发送 `hi`，确认 assistant 开始流式回复，再发送 `/quit`
- Result:
  - 模型层现在只负责：
    - 加载 checkpoint
    - 非流式生成
    - 流式生成
  - 应用层现在只负责：
    - 维护消息历史
    - 组织对话 prompt
    - 驱动 assistant 回复流
  - TUI 已可在当前 checkpoint 上工作，支持：
    - 输入消息
    - assistant 流式回复
    - `/clear`
    - `/quit`
- Issue:
  - 当前模型仍是 `Tiny Shakespeare + byte-level` smoke 模型，因此“能对话”更多是交互能力验证，不代表语义对话质量已经足够好
- Conclusion: 仓库现在已经具备“模型层 + 应用层 + TUI 应用入口”的最小结构
- Next: 如果继续推进，优先把聊天底层模型升级到更适合对话的 tokenizer 和数据
