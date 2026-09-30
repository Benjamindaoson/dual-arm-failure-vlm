# 简历表述（证据限定版）

## 推荐主版本

- 独立研究项目：围绕公开 REBOOT 双臂机器人精密装配轨迹，设计 failure-aware 多模态执行 Critic，将多视角时序视觉与机器人 state/action trace 映射为阶段、执行状态和失败模式的严格 JSON 诊断。
- 建立数据优先的证据链：固定公开数据 revision，审计 60 个 recovery episodes、53,886 条 frame rows、4 路 RGB 与 14-D state/action；识别并隔离 7 个异常 episode，生成 episode-level 42/5/6 train/val/test 划分和 SHA-256 收据。
- 实现 Base → QLoRA SFT → RLVR 的自动 evidence gate，并区分 TRL 中 token-level GRPO 与 sequence-level GSPO；格式正确仅作为 validity gate，不作为正奖励。
- 实现 469 个因果时序窗口（372/43/54）与 84 条 onset-relative timing 样本，覆盖 A0–A3 输入消融、per-episode bootstrap CI 与 failure detection delay 协议。
- 完成 AutoDL 离线运行链路：真实下载并验证 2.7 GB 数据与 7.1 GB 模型快照，成功解码/物化公开 RGB 视频；目标 SSH 容器未暴露 CUDA，因此所有模型效果、显存与训练耗时均明确报告为 N/A。

## 更短版本

- 基于公开 REBOOT 精密装配轨迹构建 failure-aware VLM post-training 管线，审计 60 条真实轨迹并隔离 7 条异常，生成 469 个 episode 防泄漏时序窗口及 Base/SFT/RLVR 证据门控；GPU 结果因容器未挂载 CUDA而未执行、未写入简历数字。

## 绝对不要写

- “提升 failure recall X%”或“GSPO 优于 GRPO”：尚无 GPU prediction receipt。
- “使用奇瑞内部机器人数据/在奇瑞落地”：数据是公开 REBOOT。
- “Georgia Tech 官方研究”：这是 independent research project。
- “支持 RGB-D Pilot”：当前固定 sample schema 只有 4 路 RGB，没有 depth 字段。
- “证明跨 connector 泛化”：full-suite holdout 尚未执行。
