# 多模态图表 GSPO GPU-Ready 设计

## 目标

把课程 Qwen3-VL GSPO notebook 升级为可审计的数据适配、可组合奖励、LoRA 后训练、checkpoint 恢复和训练前后评测链路。

## 架构与边界

输入 JSONL 每行包含 `image_path`、`question`、`reference_answer`、`task`。CPU 适配器验证字段、图像存在性和答案可解析性，产出接受/拒绝审计。奖励函数复用现有正确性和格式评分，并在训练中作为组合 reward。图像、数据集和模型权重均不提交。

训练提供 `single_v100_smoke`（FP16、较小 VLM）和 `full_chart_gspo`（Qwen3-VL-8B、BF16/多卡或高显存）两个 profile。单卡 V100 不允许误用 8B 正式 profile。实际入口在非 dry-run 后使用 TRL 多模态 GRPO 与 PEFT LoRA，保存 checkpoint 及最终模型。

## 验收标准

- CPU 可验证图表数据、输出拒绝审计和组合 reward 参数。
- CPU dry-run 不导入 torch/TRL，且会拒绝 V100 + BF16、V100 + 8B 正式配置、或占位数据路径。
- 非 dry-run 在图像路径、GPU 和可选依赖均可用后启动多模态训练。
- 训练后评测按 task 输出 accuracy、format rate、平均 reward 和解析失败率。

## 非目标

不在本地 CPU 下载 Qwen3-VL、处理完整 MathVista 或执行真实 GSPO rollout。
