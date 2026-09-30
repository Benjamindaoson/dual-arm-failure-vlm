# 面试叙事：从真实数据到负结果

## 30 秒版本

我用公开 REBOOT 双臂装配轨迹研究机器人能否及时识别失败。先审计 60 个 episode、53,886 条帧记录，隔离 7 个异常 episode，再以完整轨迹固定 42/5/6 的训练、验证、测试划分。我在 RTX 4090 D 上运行了 Qwen2.5-VL-3B 的 Base、QLoRA SFT、GRPO 和探索性 GSPO。SFT 的严格 JSON 有效率达到 100%、State Macro-F1 达到 0.4012，但 Failure Recall 仍是 0/18，时间定位中 6/6 个失败 episode 也都未稳定检出。这个项目的关键结论是：格式学习和更高训练 reward 没有解决真实失败感知。

## 2 分钟版本

精密装配的异常常发生在成功率数字看不到的时刻：对位略偏、夹持滑移或插入卡滞。我把任务收窄为执行状态 Critic：只看当前和历史画面，加可选的机器人 state/action trace，输出当前阶段、nominal/failure/recovery 状态和 failure mode，不训练动作策略。

首先解决数据可信性。REBOOT sample 有 60 个真实 episode、53,886 条 frame rows、4 路 RGB、14 维 state 和 14 维 action。审计发现越界帧、非法 originating phase 和 recovery 早于 failure 等问题，隔离 7 个 episode，不擅自修标签。剩下 53 个 episode 按完整轨迹固定为 42/5/6，生成 469 个因果窗口；另取 failure/recovery onset 前后七个时间点形成 84 条样本。6 个 test episode 在所有实验中相同。

在 4090 D 上，Base 的 A0 单帧、A1 单相机时序、A2 双相机时序各跑完 54 条测试，但都输出带代码围栏的 JSON，按预先固定的严格 schema 计分为零。我继续用 A2 做两轮 QLoRA SFT：JSON 有效率升到 100%，State Macro-F1 0.4012，Recovery Recall 0.6111，但最重要的 Failure Recall 仍是 0/18。结果级配对中 Base 错而 SFT 对 7 条，说明模型确实学到一部分状态和格式，仍没有学会识别失败。

我们又用连续两次预测的 K=2 定义检查“何时发现失败”。Base、SFT、GRPO、GSPO 都是 0/6 个失败 episode 稳定检出，因此 Failure Detection Delay 是 N/A。Trace-Text 在同一 SFT adapter 上由 0.4012 降到 0.1667。证据门控允许尝试 GRPO，因为 SFT 对整体状态有增益且存在大量错误；100-step GRPO 的 State Macro-F1 降到 0.1667，Failure Recall 仍为 0。GSPO 随后只做探索性对照，得到 0.1619 与 0。结论是当前最好的状态诊断 checkpoint 是 SFT，但没有模型满足失败识别目标，不能上线作安全判断。

## 5 分钟追问路径

### 为什么先做数据审计？

同一 episode 的相邻帧高度相关，随机按帧切分会制造泄漏。更基础的是标签本身存在边界异常；如果不隔离，时点监督甚至可能要求模型预测“恢复早于失败”。我保留原始 revision、异常原因和 split hash。`duration_frames` 在该快照是含终点 frame index，正常 0..897 对应 898 行；只有 6 个 episode 多出越界 frame 898。这一点曾导致早期审计口径错误，后来用 parquet 索引与元数据交叉验证修正。

### 为什么 Base 的零分不能简单说明视觉模型完全不会？

严格解析器要求直接返回恰好三个字段的 JSON；Base 常加 Markdown 代码围栏，还输出不在 taxonomy 内的阶段文本，所以正式任务分数为零。我们没有事后放宽解析器。这个零分混合了格式遵循与语义判断问题，因此 A0/A1/A2 的相对语义能力不能从三组零分推断。

### SFT 到底改善了什么？

同一 54 样本上，JSON 有效率 0→1.0、State Macro-F1 0→0.4012、Recovery Recall 0→0.6111；但 18 个 failure 窗口一个也未识别。按 episode bootstrap 的 State Macro-F1 95% CI 为 `[0.3467, 0.4727]`，只有 6 个 test episode，因此不做显著性或外推声明。SFT 改善的是结构化输出及部分 nominal/recovery 判断，核心 failure 任务未解决。

### RL 为什么跑，结果怎样？

项目的 gate 在 SFT 的 State Macro-F1 增益、结果级错误余量和 verifier 可计算性三个条件上放行 GRPO。reward 的 JSON 解析只作有效性门，不给格式正奖励；phase、state 和适用 failure mode 才给分。GRPO 用 token-level importance sampling，100 步训练稳定，但 held-out State Macro-F1 0.1667、Failure Recall 0，模型偏向预测 recovery。GRPO 没有产生继续正式 GSPO 的收益证据；为了比较两种采样粒度，GSPO 明确标记为探索性，使用同一 SFT 起点、数据、seed、generations 和更新预算，结果仍未改善 failure。训练 reward 不能代替冻结测试集指标。

### 时间定位和 trace 给了什么额外信息？

K=2 时间定位揭示 6/6 个失败 episode 对所有模型都未检出；没有检出就没有可定义的平均 failure delay。GRPO 虽对 3/6 个 recovery episode 在 onset 后 0.5 秒稳定检出，却在 18 个 onset 前窗口中误报 12 次。Trace-Text 是把 14-D state/action 压缩成因果文字附加到已在 A2 上训练的 SFT adapter，结果变差；这只能否定该零样本附加方式，不能否定状态信息本身。

### 可以怎样概括项目价值？

我把数据审计、因果时序、冻结评测、后训练和门控串成可复核闭环，并记录一个有用的负结果：小样本公开装配数据上，VLM 可以学会输出协议，却仍漏掉全部 held-out failure。项目没有真实机器人动作控制或跨任务泛化结果，也不属于佐治亚理工或企业官方项目。详细数字见 [`实验报告`](EXPERIMENT_REPORT_CN.md)。
