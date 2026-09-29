# 深挖问答

## 1. 为什么选 REBOOT？

它把 failure 当成一等信号，提供真实双臂精密装配轨迹、共享五阶段、失败模式、失败注入点与恢复起点，适合研究“何时失败、为什么失败、何时恢复”，而不是只做终局成功率。项目先用 60-episode sample 做端到端审计，避免直接在 full suite 上放大数据错误。

## 2. 为什么不是普通 CV？

目标不是框、mask 或静态类别，而是从执行历史判断阶段和状态转移。misalignment、slip、jamming 与 recovery 都依赖任务语义和时间上下文；传统检测可以成为底层能力，但不是本项目的学习目标。

## 3. 为什么 VLM？

输入同时包含任务指令、多图历史和可选 trace，输出是可被后续 planner 消费的结构化语义状态。VLM 适合把视觉证据与任务阶段、failure taxonomy 对齐；是否优于更小的专用模型仍需 Base/ablation 结果，而不是预设结论。

## 4. 为什么 temporal context？

单帧无法稳定区分“正在接近目标”“已经偏离”“正在重新对齐”。时序窗口提供运动方向、接触前后变化和 recovery transition。A0 与 A1/A2 的 held-out 差异直接验证它是否有增益。

## 5. 为什么 action trace？

视觉外观相似时，state/action 可以揭示停滞、反向修正或持续施力。第一版只把窗口压缩成 state delta 与 latest action，避免把 14-D 序列冗长地文本化。A3 不优于 A2 就删除 trace，而不是为故事保留。

## 6. 为什么 SFT？

只有 Base 在 failure recall 或 state macro-F1 上留下明显 headroom 时才做。SFT 主要学习数据 taxonomy、阶段边界和严格短输出，不假设它一定改善视觉识别。

## 7. 为什么 RL？

结构化标签允许 outcome-level 可验证 reward。只有 SFT 明显改善但仍有稳定 outcome error 时，RLVR 才有存在资格；否则 RL 只会增加成本和 reward hacking 面。

## 8. 为什么 GRPO？

它可以对同一 prompt 的多条短诊断进行组内相对优化，不需要单独训练 value model。这里把它作为 RLVR baseline，而不是项目卖点。

## 9. 为什么 GSPO？

诊断 reward 是 sequence-level，GSPO 的 sequence-level importance sampling 与这个 credit unit 更一致。但“更一致”不是“更好”；必须与 token-level GRPO 在同一数据、checkpoint、reward 和 test split 上比较。

## 10. reward hacking 怎么处理？

JSON parse 只做 gate，不给正奖励；有效但错误的 JSON 不会因为格式漂亮得分。reward component 单独记录，nominal 不奖励 taxonomy 猜测，未知状态得零。最终仍用 held-out task metrics，而不是训练 reward 证明能力。

## 11. 为什么 JSON format 不能算核心 reward？

格式提升只说明模型更会遵循协议，不能说明它更会识别 failure。JSON valid rate 独立报告；能力结论依赖 failure recall、state/phase/failure-mode F1 与 timing。

## 12. 如何防 leakage？

先固定 episode split，再让所有 frame/window 继承所属 episode。split receipt 写出 episode ids、分布和 SHA-256；消融比较检查四组 test episode 集合完全一致。禁止 frame-level random split。

## 13. 为什么 episode split？

同一 episode 的相邻帧高度相关。frame random split 会让模型在 test 看到几乎相同的场景、操作者和轨迹，虚高指标。episode 是这个数据中最低合理独立单位。

## 14. vision encoder 是否需要微调？

不知道，所以比较 `SFT-Language` 与 `SFT-VisionLanguage`。如果 language-only 已解决大部分错误，瓶颈更可能是 label/状态解释；如果 vision-language 稳定改善且显存成本可接受，才有证据支持视觉适配。

## 15. failure detection delay 有什么意义？

普通 accuracy 不告诉系统多久才发现失败。工业恢复越晚，越可能继续施力、损伤零件或把可恢复状态推成不可恢复。delay 与 false alarm 一起衡量“快且不过度报警”。

## 16. 如果 SFT 比 RL 好怎么办？

保留 SFT，停止 RL。RL 不是必须阶段；更简单、稳定、可复现的方案就是更好的工程答案。

## 17. 如果 GSPO 不如 GRPO 怎么办？

如实报告并分析 reward variance、KL、completion length 与 rare-class slice。项目名称和结论不会预设 GSPO 优胜；sequence-level IS 只是待检验机制。

## 18. 真实汽车制造有什么迁移价值？

公开数据中的 connector、fastener、插接和精密定位与汽车 harness/connector/fastener 工位共享 failure 形态：偏轴、打滑、卡滞、未完全啮合、提前释放。可迁移的是 failure-aware Critic、评测协议和 recovery gate，不是对某家车企产线效果的直接声明。

## 19. 和过去 Agent trace/recovery 工作有什么关系？

共同点是把长任务失败建模为可观测状态与可恢复转移：Agent 侧读取 tool/action trace，机器人侧读取视觉与 state/action trace。差异是机器人具有连续物理状态、传感误差和接触风险，因此必须强调 causal window、false alarm、delay 与真实数据 annotation audit。

## 20. 当前哪些数字可以说？

可以说：固定 revision、60 个 observed episodes、53 valid、7 quarantined、53,886 条 frame rows、30 FPS、4 RGB cameras、14-D state/action、42/5/6 episode split、469 个因果窗口、84 条 timing-manifest rows，以及本地和远端各通过 39 项测试。

不能说：Base/SFT/GRPO/GSPO 的任何准确率、F1、recall、提升比例、显存、耗时或跨任务泛化。完整数据与模型虽已下载，但目标容器没有 GPU 设备节点，因而没有正式 GPU run receipt。
