# lite-llm-pretraining

一个面向当前电脑的轻量级 LLM 预训练实验仓库。

当前仓库已经落地一条可在 Apple Silicon Mac 16GB 上直接运行的最小闭环：

- 框架：`MLX + Metal`
- 数据：`Tiny Shakespeare` / `TinyStories`
- tokenizer：`UTF-8 byte-level`
- 模型：decoder-only Transformer
- 验证目标：训练、验证、checkpoint 保存/加载、基础采样
- 交互入口：streaming 采样与交互式 TUI 对话

## 范围

- 单机优先，不做分布式和集群部署设计
- 小模型优先，先保证能训练、能收敛、能生成
- 文档和脚本保持精简，避免过早抽象

## 当前能力

- 下载并准备 `Tiny Shakespeare` 字节级数据集
- 下载并准备 `TinyStories` 全量字节级数据集
- 在本机用 `MLX` 启动一个最小 decoder-only LM 训练
- 周期性输出 train/val loss，并写入 `metrics.jsonl`
- 保存 `best` / `latest` checkpoint
- 从保存的 checkpoint 重新加载并采样
- 生成基于 sample 和 val loss 的基础质量验证报告
- 提供模型层 / 应用层分离的对话结构
- 提供终端交互式 TUI 聊天入口

## 环境

- 已验证机器：`Apple M1 Pro / 16 GB`
- Python：`3.14`
- 依赖见 `requirements.txt`

## 快速开始

1. 创建环境并安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

2. 一条命令跑完整本地链路

```bash
python -m lite_llm_pretraining.run_local --force_prepare
```

这条命令会顺序执行：

- 准备 `Tiny Shakespeare` 字节级数据
- 启动训练
- 保存 `best` / `latest` checkpoint
- 从 `best` checkpoint 重新加载并生成最终 sample
- 写出 `local_run_summary.json`

3. 如果只想拆开执行，也可以手动跑各阶段

准备 `Tiny Shakespeare` 数据：

```bash
python -m lite_llm_pretraining.prepare_tiny_shakespeare
```

准备 `TinyStories` 全量数据：

```bash
python -m lite_llm_pretraining.prepare_tinystories
```

启动默认 smoke 训练：

```bash
python -m lite_llm_pretraining.train
```

默认配置在 `configs/tinyshakespeare-byte-smoke.json`，当前 smoke stage 为：

- `context_size=128`
- `dim=256`
- `num_layers=4`
- `num_heads=4`
- `batch_size=16`
- `max_steps=200`

如果要跑 `TinyStories` 的约 `50M` 参数配置：

```bash
python -m lite_llm_pretraining.run_local \
  --config configs/tinystories-byte-50m.json \
  --force_prepare
```

这份配置的核心参数为：

- `context_size=256`
- `dim=640`
- `num_layers=10`
- `num_heads=8`
- `gradient_checkpointing=true`
- 参数量约 `49.7M`

从 checkpoint 采样：

```bash
python -m lite_llm_pretraining.sample \
  --checkpoint_dir checkpoints/tinyshakespeare-byte-smoke/best \
  --prompt "ROMEO:\n" \
  --max_new_tokens 120
```

如果希望终端里实时 streaming 输出：

```bash
python -m lite_llm_pretraining.sample \
  --checkpoint_dir checkpoints/tinyshakespeare-byte-smoke/best \
  --prompt "ROMEO:\n" \
  --max_new_tokens 120 \
  --stream
```

如果希望进入交互式 TUI 对话：

```bash
python -m lite_llm_pretraining.tui_chat \
  --checkpoint_dir checkpoints/tinyshakespeare-byte-smoke/best \
  --max_new_tokens 120
```

TUI 内置命令：

- `/clear` 清空当前会话
- `/quit` 退出界面

## 分层结构

当前交互能力按两层组织：

- 模型层：`lite_llm_pretraining/model/`
  - 负责 checkpoint 加载、一次性生成和流式生成
- 应用层：`lite_llm_pretraining/app/`
  - 负责对话历史、prompt 组织和 assistant 回复流

终端入口：

- 单次采样：`lite_llm_pretraining/sample.py`
- 交互式对话：`lite_llm_pretraining/tui_chat.py`

## 训练产物

- 数据输出：`data/tinyshakespeare-byte/`
- 数据输出：`data/tinystories-byte/`
- 训练输出：`checkpoints/tinyshakespeare-byte-smoke/`
- 指标日志：`checkpoints/tinyshakespeare-byte-smoke/metrics.jsonl`
- 采样结果：`checkpoints/tinyshakespeare-byte-smoke/samples/`
- 本地完整运行摘要：`checkpoints/tinyshakespeare-byte-smoke/local_run_summary.json`
- 最终 sample：`checkpoints/tinyshakespeare-byte-smoke/final_sample.txt`
- 质量验证报告：`validation_report.json`

## 推荐目录

随着代码补齐，建议逐步整理成下面的结构：

```text
data/        原始数据和处理中间产物
configs/     训练和数据配置
src/         核心训练代码
scripts/     可直接执行的脚本
checkpoints/ 本地训练产物
logs/        训练日志
```

## 开发原则

- 先可运行，再优化速度
- 先本地闭环，再扩展规模
- 优先可读性，不引入不必要框架

## 当前结果

- 已在当前电脑上跑通 200 step smoke stage
- `val_loss` 从 `5.1821` 降到 `2.4444`
- 已生成可加载 checkpoint，并完成一次基础采样
- 已验证 `python -m lite_llm_pretraining.run_local --force_prepare` 可直接完成本地完整链路
- 已补齐 `TinyStories` 全量数据准备、约 `50M` 参数配置和自动化质量验证入口

## 下一步

如果要继续扩展，不再优先补“有没有闭环”，而是优先补“闭环之后的第二阶段能力”：

1. 把 `TinyStories` `50M` 配置从探路阶段推进到更长训练阶段
2. 引入更接近 LLM 习惯的 tokenizer 方案
3. 补更清晰的 eval / sample 对比标准
4. 再考虑是否引入更大语料的小型清洗子集
