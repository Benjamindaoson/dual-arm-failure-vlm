# 简历项目表述（真实运行版）

**双臂机器人精密装配失败识别与多模态大模型后训练**｜独立研究项目，基于公开 REBOOT 数据

面向装配对位偏差、滑移和卡滞等异常，构建基于历史多视角画面与机器人状态的执行状态判断和时间定位评测；研究对象是失败感知，不是机器人动作控制。

- 审计 60 个真实操作 episode、53,886 条 frame rows，隔离 7 个标注或边界异常 episode；冻结 42/5/6 个 episode 的训练、验证、测试划分，构建 469 个因果观测窗口和 84 条失败/恢复时点样本，避免同一轨迹跨集合泄漏。
- 在 RTX 4090 D 上完成 Qwen2.5-VL-3B 的 NF4 QLoRA 2-epoch SFT，并按统一 54 样本冻结测试集评估 Base、SFT、100-step GRPO 和探索性 GSPO；统一数据、seed、4 generations 与结果级 verifier，保存逐样本预测、配置、日志和运行回执。
- SFT 将严格 JSON 有效率从 0 提到 100%，State Macro-F1 达到 0.4012；同时测得 Failure Recall 为 0/18、Failure K=2 稳定检出为 0/6。进一步发现 GRPO、GSPO 和未经专门训练的 Trace-Text 均未改善失败检出，据此保留负结果并限定项目能力边界。

技术栈：Python、PyTorch、Transformers、TRL、PEFT、Qwen2.5-VL、NF4 QLoRA、GRPO、GSPO、LeRobot、PyAV。

## 一行短版

基于 REBOOT 公开机器人轨迹完成 60 episode / 53,886 帧审计与防泄漏评测，在 4090 D 上对比 Qwen2.5-VL 的 SFT、GRPO 和探索性 GSPO；SFT State Macro-F1 为 0.4012，但 Failure Recall 仍为 0，明确记录失败识别尚未解决。

## 表述边界

本项目不是佐治亚理工或企业官方合作，不含企业内部数据；没有真实机器人闭环控制、跨任务泛化或生产部署结果。不能写“Failure Recall 提升”“GSPO 优于 GRPO”或把 Recovery Recall 1.0 脱离 12/18 次提前恢复误报单独展示。全部数字对应 [`实验报告`](EXPERIMENT_REPORT_CN.md) 和 [`最终比较表`](../artifacts/eval/final_model_comparison.json)。
