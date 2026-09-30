# 深挖问答：REBOOT 真实 GPU 实验

## 1. 项目研究什么？

把公开双臂精密装配轨迹中的历史视觉和可选机器人 trace 映射为阶段、执行状态、失败模式三个结构化字段，并测量失败是否在正确时间被发现。没有训练机器人动作策略。

## 2. 为什么选 REBOOT？

它提供真实操作视频、共享装配阶段、失败模式及失败/恢复起点，允许按 episode 做时序评测。本轮只使用 60 个 episode 的公开 cylinder-install sample。

## 3. 为什么先隔离 7 个 episode？

全量审计发现越界帧、非法 originating phase 和 recovery 早于 failure。将有问题的完整轨迹隔离，保留原因与哈希，避免模型被矛盾标签监督。

## 4. 为什么按 episode 划分？

同一轨迹的相邻帧高度相关。frame-level random split 会让 train/test 共享几乎相同的观察；本项目固定 42/5/6 个完整 episode，测试集为 `02, 08, 18, 20, 47, 54`。

## 5. 为什么输入必须因果？

在线诊断不能提前看到 failure 或 recovery 之后的画面。469 个窗口和 84 条 onset 样本只采样当前及过去时间点；未来信息不进入 prompt。

## 6. A0/A1/A2 有何区别？

A0 为当前单帧，A1 为同一相机四个历史时间点，A2 为四个时间点×两个相机。Base 在三组严格 JSON 评测中均为 0，因而没有证据说时序或第二相机带来增益。

## 7. Base 全零是否说明视觉没有信号？

不能。Base 常输出带 Markdown 围栏的 JSON，还出现非法阶段标签；严格解析后任务分数为 0。这同时包含格式和语义失败，我们没有事后改解析规则。

## 8. 为什么还选 A2 做 SFT？

Base 严格分数同为 0，无法凭它挑出视觉最优输入。A2 保留完整的多视角历史，下一步用一次受限的 SFT 检验该输入是否能学到状态信号；这不是基于 Base 显著优势的选择。

## 9. SFT 具体改善了什么？

2-epoch NF4 QLoRA 后，54 样本 test 的 JSON Valid Rate 为 100%，State Macro-F1 为 0.4012，Recovery Recall 为 0.6111；结果级配对有 7 条由 Base 错转为 SFT 对。

## 10. SFT 没解决什么？

Failure Recall 仍为 0/18，K=2 failure timing 仍为 0/6 episode 检出。State Macro-F1 的提升不能替代 primary metric。

## 11. 为什么 JSON 有效率单独报告？

协议正确只表明输出可解析。reward 中无“格式正确”正奖励，解析失败只使任务 reward 为零；冻结测试集还必须看状态、阶段、failure mode 和时点。

## 12. Verifier reward 如何避免明显投机？

phase、state、适用 failure mode 分开给分；nominal 不因猜 failure taxonomy 获奖，未知标签没有分。最终选择仍依赖 held-out 指标，不依赖训练 reward 曲线。

## 13. RL Gate 为什么允许 GRPO？

已实现 gate 依据 SFT 的 State Macro-F1 相对 Base 提高 0.4012、仍有 47/54 个结果级错误、verifier 可计算而输出 `RUN_RLVR`。Failure Recall 的增益为 0，因此只是有总体 headroom 的机制实验。

## 14. GRPO 与 GSPO 的真实配置差异？

两者同源 SFT adapter、同一数据/seed、4 generations、100 steps、同一 reward 和 `loss_type=grpo`。GRPO 使用 token-level importance sampling；GSPO 使用 sequence-level。`sequence + dr_grpo` 不被称为本项目的 GSPO。

## 15. 为什么 GSPO 标为探索性？

GRPO 的 held-out State Macro-F1 从 SFT 的 0.4012 降为 0.1667，Failure Recall 仍为 0，未满足“GRPO 有真实增益才进入正式 GSPO”的更严格门槛。GSPO 只为同预算算法对照运行。

## 16. RL 最终结果如何？

GRPO State Macro-F1 0.1667、Recovery Recall 1.0、Failure Recall 0；探索性 GSPO 分别为 0.1619、0.9444、0。两者结果级全对样本均为相同的 4/54，不能宣称 GSPO 优于 GRPO。

## 17. GRPO Recovery Recall 1.0 是否代表恢复判断很好？

不代表。K=2 recovery timing 仅检出 3/6 episode，且在 18 个恢复 onset 前窗口中误报 12 次；test 状态分布中它几乎把所有窗口预测为 recovery。

## 18. Failure Detection Delay 为何是 N/A 而非 0？

首次连续两次预测 failure 才产生 delay。四个模型在 6 个失败 episode 上都没有稳定检出，故没有观测到可计算的 delay；0 秒会伪造“刚好及时发现”。

## 19. Trace-Text 为什么变差？

同一 SFT adapter 在 A2 Visual-only 的 State Macro-F1 为 0.4012，直接加 14-D state/action 的因果文字摘要后 A3 为 0.1667；adapter 没有针对新 prompt 再训练，因此只能说该附加方法在本测试集退化，不能说 trace 信息本身无用。

## 20. 有哪些可用和不可用的结论？

可说：4090 D 上完成 Base/SFT/GRPO/探索性 GSPO 的真实运行，40 项单测通过，保留逐样本预测及运行回执；SFT 是本轮 State Macro-F1 最好的 checkpoint。不可说：已可靠识别失败、RL 改善 primary metric、跨装配任务泛化、真实机器人闭环成功、佐治亚理工或企业官方合作。完整结果见 [`实验报告`](EXPERIMENT_REPORT_CN.md) 与 [`最终比较表`](../artifacts/eval/final_model_comparison.json)。
