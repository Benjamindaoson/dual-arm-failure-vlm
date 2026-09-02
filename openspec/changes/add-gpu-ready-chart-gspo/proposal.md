# GPU-ready chart GSPO trainer

## Why

项目当前只具有评测逻辑与 GPU 预检，不能启动 GSPO 后训练。

## What Changes

- 新增图表 JSONL 与图像路径审计。
- 新增 profile 约束的 TRL 多模态 GRPO/LoRA 训练入口。
- 新增训练后 JSONL 评测命令与 CPU dry-run 测试。

## Impact

影响多模态项目的升级实现、配置、脚本、测试和 README；课程 notebook 保持原样。
