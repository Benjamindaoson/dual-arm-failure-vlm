# V2 研究计划与证据边界

研究问题：后训练能否同时提高机器人多模态 Critic 的输出协议遵循，却削弱失败感知？研究对象是 REBOOT 圆柱装配样本上的 Qwen2.5-VL-3B 结构化执行状态判断器，不是动作策略或闭环恢复系统。

## 已观察的发现证据

V1 六个 episode、54 个窗口属于已反复分析的诊断集。严格裸 JSON 解析下 Base 的 State Macro-F1 和 Failure Recall 均为零；机械剥离最外层 Markdown fence 后，54/54 输出可解析，Base 的 Failure Recall 为 6/18、State Macro-F1 为 0.300。SFT 严格格式有效率升至 1.0，但 Failure Recall 为 0/18；历史 GRPO 和探索性 GSPO 同样为零，并偏向 recovery。这些是发现性相关现象，不能单独证明 SFT 或奖励函数造成了退化。

## 机制假设与判别

| 假设 | 相互竞争的解释 | 判别实验与主要终点 | 可支持的结论 |
| --- | --- | --- | --- |
| H1 格式合规不等于失败感知 | Base 只是 Markdown 格式错；语义内容可能不同 | 同一原始输出同时给 Strict 与固定 Semantic 评分；记录 Failure Recall、JSON Valid Rate、状态分布 | 可描述该样本上的分离；不能把诊断指标冒充 V1 正式分数 |
| H2 当前 SFT 配方抑制 failure signal | 摄像头看不到、稀疏标签、类别失衡、多任务输出干扰 | state-only C0/C1/C2；同预算 sparse/dense；匹配训练 A2/A3；只在 state-only 有信号时做 full-schema 对照 | 只有匹配控制且新冻结集复现时，才可归因到某一因素 |
| H3 可加奖励允许状态错误仍获分 | V1 RL 训练差异和数据影响；未必是 reward shortcut | 先执行失败优先 RL Gate；若通过，同一个 SFT 起点、数据、种子和更新预算比较 additive 与 state-gated GRPO | 若 gate 未通过，H3 仍只是代码层结构风险，不能写“干预有效” |

## 冻结协议

1. V1 旧测试 episode 02、08、18、20、47、54 仅作诊断，不能再次称为 untouched final。
2. V2 选择只看 validation：优先 Failure Recall，随后 State Macro-F1，固定 tie order。必须同时报告 Failure Precision、Nominal/Recovery Recall、虚警和 Collapse Ratio，以识别“全预测 failure”的退化解。
3. 所有时序样本满足最大采样帧不晚于 anchor。Trace-Text 只来自同时刻及历史的 14-D state、14-D action。
4. 正式 RL Gate 要求 SFT failure recall 非零且不低于同协议 Base，State Macro-F1 不退化、虚警不增加、严格 JSON 有效率至少 0.95、仍有错误余量且 verifier 通过验证。门失败则不做正式 RL。
5. 新任务优先；官方新任务若缺少可操作的 onset 标签或在远端不可获取，按预定 Tier B 冻结新 episode split，所有最终模型从 Base 重训。Tier B 仅支持单任务证据，并须承认其候选 episode 曾属于方法开发池。
6. 统计单位为 episode。报告样本量、各类别 support、点估计、95% episode-bootstrap CI，主要成对比较给出差值 CI；不强行报告不适用的显著性检验。

## 执行顺序

状态单任务与相机筛选 → 密集监督 → 匹配 Trace → 条件性多任务隔离 → RL Gate → 条件性 additive/gated GRPO → 新冻结集重训与评测 → 图表、论文和复核。所有 V2 结果必须有预测、指标、配置、环境、时间/显存和 run receipt；未执行项目写 N/A，不补猜测值。
