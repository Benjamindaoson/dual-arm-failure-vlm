# REBOOT 精密装配失败识别：真实 GPU 实验报告

状态：2026-09-30，已执行；项目身份：基于公开 REBOOT 数据的独立研究。数据 revision `0633573d0438be1185bddebdf1a2c8f5505f7b2a`，Qwen2.5-VL-3B-Instruct 模型 revision `66285546d2b821cf421d4f5eb2576359d3770cd3`，seed 42。全部完整评测使用冻结的 test episodes `02, 08, 18, 20, 47, 54`，每个输入版本均为 54 个窗口；split receipt SHA-256 为 `4301609f83c471cb00abdac1b61855449ceb1872f02189fe986a82f1945b774c`。

主要结论：**没有一个模型识别出 held-out failure**。SFT 将严格 JSON 有效率从 0 提到 100%，State Macro-F1 达到 0.4012，但 Failure Recall 仍为 0/18。GRPO 和探索性 GSPO 都没有改善 Failure Recall，且总体 State Macro-F1 低于 SFT。该结果不支持把系统称为可靠的装配失败检测器。

## 【硬件】

新 AutoDL 实例实测 NVIDIA GeForce RTX 4090 D，设备节点 `/dev/nvidia*` 存在，`torch.cuda.is_available() == True`，BF16 可用，显存 25,252,724,736 bytes（约 23.52 GiB）；CPU 16 cores、RAM 62 GB、Python 3.10.8、PyTorch 2.6.0+cu124。正式训练全程没有 OOM，也没有降像素、帧数、相机数或 RL generations。原先无 GPU 的容器探针作为历史记录保留在 [`gpu_environment_pre_gpu.json`](../artifacts/environment/gpu_environment_pre_gpu.json)；当前探针和运行环境见 [`hardware_probe.txt`](../artifacts/environment/hardware_probe.txt) 与 [`gpu_environment.json`](../artifacts/environment/gpu_environment.json)。

## 【真实数据】

公开 sample 本地数据约 2.7 GB，模型快照约 7.1 GB；没有重复下载、重新审计或重新划分。全量审计覆盖 60 个 episode、53,886 条 frame rows、30 FPS、4 路 RGB、14-D state 和 14-D action。7 个异常 episode `00, 15, 16, 26, 34, 37, 38` 隔离后，有效 episode 53 个，按完整 episode 固定为 42 train / 5 val / 6 test。469 个因果窗口分为 372/43/54；时间定位集为 6 test episodes × 2 onsets × 7 offsets = 84 条。A0/A1/A2/A3 全部物化，图片缺失数为 0；A3 trace 字段缺失数为 0。物化回执见 [`artifacts/materialization`](../artifacts/materialization/A2/dataset_receipt.json)，审计与 split 见 [`DATA_RECEIPT.md`](../artifacts/data_audit/DATA_RECEIPT.md) 和 [`split_receipt.json`](../artifacts/splits/split_receipt.json)。

## 【Base A0/A1/A2】

模型采用 NF4 4-bit、BF16、确定性解码，`max_pixels=100352`、`max_new_tokens=96`。每组先跑 2 样本 smoke，再跑冻结测试集 54 样本。

| 输入 | 图片/窗口 | State Macro-F1 | Failure Recall | Recovery Recall | Phase Macro-F1 | Failure-mode Macro-F1 | JSON Valid Rate | 完整评测墙钟 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 单帧单相机 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 94.23 s |
| A1 四时点单相机 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 106.27 s |
| A2 四时点双相机 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 112.77 s |

三组 Base 输出大量带 Markdown 代码围栏的 JSON，未通过预先固定的严格解析器，因此正式任务分数为 0。不能把这些零分解释为视觉信息完全无用，也不能为了提高分数事后放宽解析协议。Base 消融结果与逐样本预测见 [`base_input_ablation.json`](../artifacts/eval/base_input_ablation.json) 和 [`artifacts/evidence`](../artifacts/evidence/base-full-A2-20260929T235818Z/predictions.jsonl)。因为三组严格指标同为零，A2 不是凭 Base 优势选出的；继续 A2 SFT 是为了测试多视角时序输入能否学到有效信号。

## 【SFT】

A2 使用 NF4 4-bit QLoRA，LoRA rank 16、alpha 32、dropout 0.05、batch 1、gradient accumulation 8、seed 42。5-step smoke 的 loss 与 gradient 均有限，adapter 保存并重载成功。正式训练 2 epochs、94 steps，可训练参数 37,152,768；最佳 validation checkpoint 为 `checkpoint-94`。训练 loss 0.2689、validation loss 0.1974；完整 run 墙钟 2,142.50 s，框架记录的峰值分配显存 5,644,279,296 bytes。

同一冻结测试集：State Macro-F1 **0.4012**（按 6 个 episode bootstrap 的 95% CI `[0.3467, 0.4727]`）；Failure Recall **0/18 = 0**（bootstrap CI `[0, 0]`）；Recovery Recall **11/18 = 0.6111**；Phase Macro-F1 **0.2938**；Failure-mode Macro-F1 **0.0857**；JSON Valid Rate **54/54 = 100%**。结果级全字段配对：Base 错而 SFT 对 7，Base 对而 SFT 错 0，双方均错 47。SFT 改善了结构化输出与部分状态判断，核心失败漏检仍未解决。证据见 [`base_vs_sft.json`](../artifacts/eval/base_vs_sft.json)。

## 【Failure Timing】

failure onset 和 recovery onset 各取 `-2, -1, -0.5, 0, +0.5, +1, +2` 秒的因果窗口；首次连续 K=2 个目标状态才算稳定检出。每个 onset 类型有 6 个 episode，onset 前共有 18 个窗口。

| 模型 | Failure K=2 检出 | Failure 检测延迟 | 提前 failure 误报 | Recovery K=2 检出 | 已检出 recovery 的平均延迟 | 提前 recovery 误报 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0/6 | N/A | 0/18 | 0/6 | N/A | 0/18 |
| SFT | 0/6 | N/A | 0/18 | 0/6 | N/A | 1/18 |
| GRPO | 0/6 | N/A | 0/18 | 3/6 | 0.5 s | 12/18 |
| GSPO（探索性） | 0/6 | N/A | 0/18 | 1/6 | 0.5 s | 3/18 |

所有模型的 failure 未检出率均为 6/6，故 Failure Detection Delay 不可计算，不能写成 0 秒。GRPO 的 recovery 检出增加伴随大量 onset 前误报；0.5 秒只针对已检出的 episode，不能外推至全部 6 个。逐 episode 结果见 [`failure_timing_base.json`](../artifacts/eval/failure_timing_base.json)、[`failure_timing_sft.json`](../artifacts/eval/failure_timing_sft.json)、[`failure_timing_grpo.json`](../artifacts/eval/failure_timing_grpo.json)、[`failure_timing_gspo.json`](../artifacts/eval/failure_timing_gspo.json)，恢复结果位于同目录的 `recovery_timing_*.json`。

## 【Trace Ablation】

对同一个 SFT adapter、同一 54 样本，比较 A2 Visual-only 与 A3 Visual + 14-D state/action 的因果 Trace-Text。A2 State Macro-F1 0.4012、Recovery Recall 0.6111；A3 分别为 **0.1667**、**0**。配对结果：Visual-only 对而 Trace 错 7，反向 1，双方均错 46。Base A3 的严格分数仍为 0。这只能说明本次 **未针对 Trace 再训练的文本附加方案**在该测试集退化，不能断言机器人状态/动作信息本身无价值。证据见 [`trace_ablation.json`](../artifacts/eval/trace_ablation.json)。

## 【RL Gate】

Base → SFT 的 State Macro-F1 增益为 +0.4012，SFT 仍有 47/54 个结果级错误，结构化 verifier reward 可计算且单测通过。因此项目的已实现 gate 输出 [`RUN_RLVR`](../artifacts/decisions/rl_gate.json)。这个门控依据是 State Macro-F1，**Failure Recall 的增益为 0**；进入 GRPO 是一次有 headroom 的机制试验，不代表安全指标已改善。

## 【GRPO】

从同一 SFT adapter 出发，TRL 0.29.1，token-level importance sampling、`loss_type=grpo`、4 generations、96 completion tokens、seed 42。5-step smoke 完成后，单个 100-step 正式 run 在 25/50/75/100 步保存检查点；每个检查点训练日志均无 NaN/Inf，最终 adapter 可保存。完整 run 墙钟 **6,100.28 s**，纯训练 **5,783.85 s**，峰值分配显存 **8,707,639,808 bytes**。训练 reward、reward std、KL、梯度范数和 completion length 均留存于 [`train_log.json`](../artifacts/evidence/grpo-full-100-20260930T023348Z/train_log.json)。

冻结测试集：State Macro-F1 **0.1667**、Failure Recall **0**、Recovery Recall **1.0000**、Phase Macro-F1 **0.0677**、Failure-mode Macro-F1 **0.1000**、JSON Valid Rate **100%**。相对 SFT 的配对结果为 SFT 对/GRPO 错 7，SFT 错/GRPO 对 4，双方均错 43。GRPO 学成了偏向 recovery 的预测，未改善 primary metric，State Macro-F1 下降 **0.2346**。训练 reward 的存在不等于 held-out 能力增益。

## 【GSPO】

GRPO 未产生真实总体增益，故 GSPO **只作为探索性算法消融**；它不是第二个通过 GRPO 增益门槛的正式阶段。它从相同 SFT checkpoint、数据、seed、4 generations、100-step 更新预算出发，使用 sequence-level importance sampling 与 `loss_type=grpo`，没有把 `dr_grpo` 冒充 GSPO。5-step smoke 与 100-step run 均完成；25/50/75/100 检查点可恢复，101 条最终训练日志无 NaN/Inf。完整 run 墙钟 **6,234.36 s**，纯训练 **5,900.02 s**，峰值分配显存 **8,707,639,808 bytes**。

冻结测试集：State Macro-F1 **0.1619**、Failure Recall **0**、Recovery Recall **0.9444**、Phase Macro-F1 **0.0965**、Failure-mode Macro-F1 **0.1043**、JSON Valid Rate **100%**。与 GRPO 的结果级完全正确样本相同，均为 4/54；两种 RL 方法都未提升 failure 检出。证据见 [`grpo_vs_gspo_exploratory.json`](../artifacts/eval/grpo_vs_gspo_exploratory.json)。

## 【最终模型比较】

| 模型 | 性质 | State Macro-F1 | Failure Recall | Recovery Recall | Phase Macro-F1 | Failure-mode Macro-F1 | JSON Valid Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| Base A2 | baseline | 0 | 0 | 0 | 0 | 0 | 0 |
| SFT A2 | 正式 | **0.4012** | **0** | 0.6111 | **0.2938** | 0.0857 | 1.0000 |
| GRPO A2 | 正式 | 0.1667 | 0 | **1.0000** | 0.0677 | 0.1000 | 1.0000 |
| GSPO A2 | 探索性 | 0.1619 | 0 | 0.9444 | 0.0965 | **0.1043** | 1.0000 |

若按 State Macro-F1 和 Phase Macro-F1 选，SFT 是本次最好的诊断 checkpoint；若按预设最优先的 Failure Recall，**不存在合格模型**。Recovery Recall 最高的 GRPO 同时造成 12/18 个 onset 前恢复误报，不能单看该数字选模型。机器可读总表见 [`final_model_comparison.json`](../artifacts/eval/final_model_comparison.json) 和 [`final_model_comparison.csv`](../artifacts/eval/final_model_comparison.csv)。

统计口径：每个方法只用 6 个 held-out episode、54 个评测窗口；95% CI 为按 episode 重采样 1,000 次的描述性区间，未计算成对差异 p 值，也不作统计显著性声明。Failure-mode 各类别 test support 仅 6–12，macro-F1 对单例预测敏感。所有原始预测、配置、环境、stdout、耗时、显存和回执在 [`artifacts/evidence`](../artifacts/evidence/sft-eval-A2-20260930T014920Z/run_receipt.json)；远端原始 adapter 的大小、路径和 SHA-256 见 [`adapter_inventory.json`](../artifacts/eval/adapter_inventory.json)。

## 【训练时间】与【峰值显存】

正式 run 墙钟：SFT 2,142.50 s（35 分 43 秒），GRPO 6,100.28 s（1 小时 41 分 40 秒），探索性 GSPO 6,234.36 s（1 小时 43 分 54 秒），合计 **14,477.15 s（4 小时 1 分 17 秒）**。三个 5-step smoke 另约 1,730.63 s（28 分 51 秒）。峰值框架已分配显存：SFT 5.64 GB、GRPO/GSPO 8.71 GB；`nvidia-smi` 观察到 RL 进程约 9.7 GiB，口径不同，不能混为一项。Base A2 评测 112.77 s；正式模型评测墙钟详见总表。

## 【失败实验】与【最重要发现】

1. Base A0/A1/A2 均没有通过严格 JSON schema，三个输入版本的正式任务指标全为 0。不能声称 temporal 或双相机已优于单帧。
2. SFT 的 JSON Valid Rate 100%、State Macro-F1 0.4012，但 Failure Recall 0/18；格式学习没有变成失败识别能力。
3. 所有模型在 failure onset 的 K=2 检测为 0/6；GRPO 的 recovery 检出伴随 12/18 提前误报。
4. 未再训练的 Trace-Text 将 State Macro-F1 从 0.4012 降到 0.1667。
5. GRPO 与探索性 GSPO 的 held-out State Macro-F1 都低于 SFT，Failure Recall 均为 0；没有 RL 增益证据。
6. 之前的无 CUDA 实例、frame 终点语义误判、TorchCodec ABI 与 LeRobot API 兼容问题均已保留在历史回执；本次 4090D run 没有 OOM、NaN 或 GPU 消失。

判断：保留 SFT 作为可复核的 pilot 基线，不把任何 checkpoint 用作安全关键失败报警。本轮不继续调 RL 超参数；下一轮研究需要更强的 failure 监督和更广的任务/episode 覆盖，这些尚未执行。

## 【可以写简历的真实数字】与【不能写简历的数字】

可以写：审计 60 个真实 episode / 53,886 条 frame rows，隔离 7 个异常 episode；冻结 42/5/6 episode split，469 个因果窗口与 84 条时间定位样本；在 RTX 4090 D 上完成 2-epoch QLoRA SFT、100-step GRPO 与探索性 GSPO 对照；54 样本 test 上 SFT State Macro-F1 0.4012、JSON Valid Rate 100%，但必须同句交代 **Failure Recall 0**。可写“发现 RL 与 Trace-Text 在该 pilot 上没有改善 primary metric”。

不能写：“failure recall 提升”“已经可靠检测装配失败”“GSPO 优于 GRPO”“机器人恢复策略成功”“跨任务或汽车产线验证”。不能把本独立研究写作佐治亚理工官方合作，也不能把 REBOOT 公开数据写作企业内部数据。GRPO 的 Recovery Recall 1.0 必须带上 12/18 提前恢复误报，不能单独作为成功数字。

## 【项目局限】

数据仅覆盖一个 16 mm cylinder-install sample，测试只有 6 个 episode，不能证明跨任务泛化或生产可用性。Base 严格解析失败使输入消融的语义比较受限。A3 是在 A2 训练的 adapter 上直接加 Trace-Text，未验证专门训练 trace 的潜力。当前模型不输出机器人动作，也未上真实机器人闭环。远端实验 bundle 没有 `.git`，run receipt 的 `git_commit` 为 `null`；revision、split hash、配置与输出哈希均保留，最终代码提交发生在运行之后。模型权重和 checkpoint 留在 `/root/autodl-tmp`，不进入 Git。

## 【复核与来源】

远端与本地均通过 40 项单元测试和 `compileall`。主证据入口：[`final_model_comparison.json`](../artifacts/eval/final_model_comparison.json)、[`base_vs_sft.json`](../artifacts/eval/base_vs_sft.json)、[`trace_ablation.json`](../artifacts/eval/trace_ablation.json)、[`rl_gate.json`](../artifacts/decisions/rl_gate.json)。原始训练曲线、预测及 run receipt 在 `artifacts/evidence/<run-id>/`，正式权重和四个 RL 检查点保留在远端数据盘。

## 【Git】

交付分支为 `reboot-precision-recovery`，现有 [PR #1](https://github.com/Benjamindaoson/multimodal-chart-gspo/pull/1) 保持未合并；本报告和证据随该分支提交。运行时远端 bundle 没有 `.git`，因此 run receipt 中的 `git_commit=null` 是已知来源限制，不能用后提交的 commit 冒充训练当时的源码哈希。
