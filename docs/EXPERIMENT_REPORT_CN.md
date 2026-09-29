# REBOOT Precision-Assembly Recovery 实验报告

状态日期：2026-09-30（Asia/Shanghai）  
项目身份：Independent Research Project  
数据集：`REBOOT26/sample_recovery-demonstration`  
数据 revision：`0633573d0438be1185bddebdf1a2c8f5505f7b2a`  
模型：`Qwen/Qwen2.5-VL-3B-Instruct`  
模型 revision：`66285546d2b821cf421d4f5eb2576359d3770cd3`

本报告只把真实执行过且有收据的结果写成数字。完整数据、模型和真实视频解码链路已经验证；模型推理与训练因目标 SSH 容器没有 NVIDIA 设备而被硬件门禁阻止。所有模型指标均为 **N/A**，不是 0。

## 【GPU】

| 项目 | 实测 |
|---|---|
| AutoDL 登录资源 | 0.5 CPU core / 2 GB RAM |
| NVIDIA 型号 | N/A |
| VRAM | N/A |
| `/dev/nvidia*` | 不存在 |
| `nvidia-smi` | `/usr/bin/nvidia-smi` 为 0-byte stub；没有设备输出 |
| Python | 3.10.8 |
| PyTorch | 2.6.0+cu124 |
| PyTorch CUDA runtime | 12.4 |
| cuDNN | 90100 |
| `torch.cuda.is_available()` | `False` |
| BF16 support | N/A |
| Transformers / TRL / PEFT | 4.57.6 / 0.29.1 / 0.21.1 |
| Accelerate / bitsandbytes | 1.15.0 / 0.49.2 |
| LeRobot / Datasets | 0.4.4 / 4.8.5 |
| 最终状态 | `BLOCKED_NO_CUDA` |

这是不可由 Python 包修复的平台挂载问题：CUDA 版 PyTorch 已安装，但容器没有 GPU 设备节点。项目因此没有在 CPU 上伪跑 Qwen2.5-VL-3B 推理或训练。

证据：`artifacts/environment/gpu_environment.json`、`artifacts/environment/hardware_probe.txt`。

## 【数据】

完整 pilot 已下载到 `/root/autodl-tmp/reboot_sample`，磁盘占用约 2.7 GB；本地校验发现 60 个 parquet data files、18 个 video files，metadata 和完整 payload 均存在。模型快照已下载到 `/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct`，磁盘占用约 7.1 GB。

| 项目 | 真实统计 |
|---|---:|
| Episodes declared / observed | 60 / 60 |
| Frame-table rows | 53,886 |
| FPS | 30 |
| RGB cameras | 4 |
| Depth keys | 0 |
| Robot state / action | 14-D / 14-D |
| Valid / quarantined episodes | 53 / 7 |
| Episode split | 42 train / 5 val / 6 test |
| Causal windows | 469 |
| Window split | 372 train / 43 val / 54 test |
| Window states | 157 nominal / 153 failure / 159 recovery |
| Timing samples | 84（6 test episodes × 2 onsets × 7 offsets） |
| Split leakage | 未发现 |

`duration_frames` 在该快照中是含终点的 frame index：终点 897 正常对应 0..897 共 898 行。episodes `00`、`15`、`26`、`34`、`37`、`38` 各多出一条越界 frame 898；episode `16` 和 `26` 的 `originating_phase` 越界；episode `37` 的 recovery 开始早于 failure。异常没有自动修复，而是按 episode 隔离。合并去重后共隔离 7 个 episode：`00, 15, 16, 26, 34, 37, 38`。

53 个有效 episode 的 failure taxonomy：

| Failure mode | Episodes |
|---|---:|
| misalignment | 22 |
| slip | 8 |
| premature_release | 6 |
| excessive_force | 5 |
| retention_failure | 3 |
| sub_mm_misalignment | 3 |
| jamming | 2 |
| delayed_start | 1 |
| freeze | 1 |
| poor_engagement | 1 |
| premature_grasp | 1 |

真实 LeRobot 读取已通过：使用 PyAV 从 MP4 解码 `cam_high` 与 `cam_low`。A0 已完整物化 469 张图和 469 条 JSONL（372/43/54）；A2 smoke 生成 2 张 JPEG 和 1 条训练 JSONL。A1 在确认硬件阻断后停止于 24 张图并保留为 partial，A3 未物化；这些不能写成完整 A1–A3 数据结果。

## 【Base】

| Variant | Held-out 输入 | State Macro-F1 | Failure Recall | Recovery Recall | Phase Macro-F1 | Failure-mode Macro-F1 | JSON Valid Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| A0 | single current frame + task | N/A | N/A | N/A | N/A | N/A | N/A |
| A1 | 4 timestamps × 1 camera + task | N/A | N/A | N/A | N/A | N/A | N/A |
| A2 | 4 timestamps × 2 cameras + task | N/A | N/A | N/A | N/A | N/A | N/A |

状态：`BLOCKED_NO_CUDA`。没有生成 Base predictions，因此不存在可计算的 Base gate。

## 【SFT】

预定且已通过配置/单元测试的训练契约是：Qwen2.5-VL-3B-Instruct、NF4 4-bit、LoRA rank 16、alpha 32、dropout 0.05、gradient checkpointing、batch size 1、gradient accumulation 8、seed 42、最多 2 epochs，并以 validation checkpoint selection 控制过拟合。

| 项目 | 结果 |
|---|---|
| 5-step smoke | N/A — BLOCKED_NO_CUDA |
| 正式 2-epoch SFT | N/A — BLOCKED_NO_CUDA |
| Adapter reload | N/A — 没有合法 SFT checkpoint |
| Trainable parameters | N/A |
| Peak VRAM | N/A |
| Wall clock / examples per second | N/A |
| Train / eval loss | N/A |
| Held-out metrics | N/A |

## 【Ablation】

四个输入契约共享同一组 test episodes：`02, 08, 18, 20, 47, 54`，split manifest SHA-256 为 `4cd35a1f76effe516d07577f1a86134e41c16430b9b7cab6092ea2bd5b2ae029`。

- A0：单帧、`cam_high`。
- A1：4 个因果时间点、`cam_high`。
- A2：4 个因果时间点、`cam_high + cam_low`，共 8 张图。
- A3：A2 + 只使用当前窗口过去信息的 state/action Trace-Text。

真实视频解码和 A0 完整物化、A2 smoke 链路已验证；A1 仅有 24 张 partial 图，A3 未物化。模型消融结果均为 **N/A — BLOCKED_NO_CUDA**。因此不能说 temporal、多相机或 trace 有增益。

## 【Failure Timing】

已从真实 annotation 生成 84 条因果 timing samples，覆盖 failure onset 和 recovery onset 的 `-2.0, -1.0, -0.5, 0, +0.5, +1.0, +2.0` 秒。检测定义是首次连续 `K=2` 个窗口预测 failure。

| 指标 | Base | SFT |
|---|---:|---:|
| Mean failure detection delay | N/A | N/A |
| Pre-failure false alarms | N/A | N/A |
| Undetected episode rate | N/A | N/A |

原因：没有 GPU predictions；manifest 数量不是模型 timing 结果。

## 【RL Gate】

状态：**未评估，不是“不通过”**。

RL gate 需要先证明：SFT 相比 Base 有清晰 held-out 增益、SFT 仍存在 outcome-level headroom、reward 可由 annotation 自动验证。第三项已经在代码和测试中满足，前两项因 Base/SFT 均未执行而没有证据。因此 RL 没有被授权启动。

## 【GRPO】

N/A — 没有合法 SFT checkpoint，RL gate 未评估，且容器没有 CUDA。已验证的配置合同为 token-level importance sampling、`loss_type="grpo"`、短 completion、4 generations、5-step smoke 后最多 50-step pilot；这只是实现合同，不是训练结果。

## 【GSPO】

N/A — 同上。已验证的配置合同为 sequence-level importance sampling、`loss_type="grpo"`；验证器明确拒绝把 `dr_grpo` 叫作 GSPO。没有证据表明 GSPO 优于 GRPO。

## 【最重要发现】

1. 完整 frame-table 审计改变了数据口径：可用 episode 从 metadata-only 的 57 降至 53；最终可信 split 是 42/5/6，不是旧的 45/5/7。
2. `duration_frames` 是含终点索引；只有 6 条 frame 898 是真正的越界额外行。若把 897 误当 row count，会错误隔离全部 60 个 episode。
3. 真实数据足以构成数百级时序 pilot：469 个因果窗口和 84 条 onset-relative timing samples，且没有 episode leakage。
4. 数据、模型、依赖和真实视频解码均已准备好；唯一阻断 P0 模型闭环的是目标容器没有 NVIDIA 设备挂载。
5. 目前没有任何模型效果证据，因而不能判断 temporal、多相机、trace、SFT、GRPO 或 GSPO 是否有价值。

## 【失败实验】

- GPU preflight 失败：登录横幅报告 `No devices were found`，没有 `/dev/nvidia*`，CUDA 版 PyTorch 仍返回 `False`。判定为不可在容器内修复的硬平台故障。
- 首轮 frame audit 曾把 `duration_frames=897` 错当成 row count，导致 60/60 全部被隔离；检查 parquet 的 0..897 实际索引和 `meta/episodes.length=898` 后修正为含终点语义，并增加回归检查。该错误结果已被覆盖，不能引用。
- LeRobot 0.4.4 不接受旧参数 `return_uint8`，且列选择位于内部 Hugging Face Dataset；物化器已按锁定版本修正。
- 默认 TorchCodec 因 FFmpeg shared-library ABI 不匹配无法加载；切换到 LeRobot 支持的 PyAV backend 后真实首帧解码通过。

## 【可用于简历的数字】

- 60 个 observed episodes、53 个 valid、7 个 quarantined。
- 53,886 条 frame rows、30 FPS、4 路 RGB、14-D state/action。
- 42/5/6 episode-level train/val/test split，split leakage 为 false。
- 469 个因果窗口：372 train、43 val、54 test。
- 84 条 failure/recovery onset timing samples。
- 完整验证 60 个 data files、18 个 video files、2.7 GB 数据和 7.1 GB 固定模型快照。
- 本地与远端各通过 39 项单元测试（以最终验证记录为准）。

## 【绝对不能对面试官说的数字】

- 任何 Base/SFT/GRPO/GSPO accuracy、F1、recall、提升比例或置信区间。
- 任何模型 failure detection delay、false-alarm rate 或 trace gain。
- 任何 GPU peak VRAM、训练时间、examples/sec、trainable parameter count 或 loss 曲线。
- “4090 已跑完实验”或“4090 可用”；当前收据证明相反。
- “GSPO 优于 GRPO”“SFT 提升 failure recall”或“temporal context 已证明有用”。
- “证明跨 connector/mechanism 泛化”或“已在汽车制造产线验证”。

## 【剩余风险】

- 目标容器必须重新挂载真实 RTX 4090；恢复标准是 `nvidia-smi` 可见设备、存在 `/dev/nvidia*`、PyTorch CUDA 为 true，三者同时满足。
- pilot 只有一个 16 mm cylinder-install task；即使后续模型指标良好，也不能外推 full-suite 泛化。
- 数据标签极不平衡，misalignment 占 53 个有效 episode 中的 22 个；failure-mode macro-F1 必须与 per-class support 一起解释。
- 两个有效 episode 的 failure interval 为零长度，因此其 failure windows 不存在；这是 source semantics，不应伪造。
- 物化图像、模型权重和 checkpoint 位于 `/root/autodl-tmp`，不进入 Git；远端数据盘仍是恢复实验的必要依赖。

## 【Git】

- 分支：`reboot-precision-recovery`
- 实现与证据 commit：`81da1f1924128feba4d9d9b9758e01dccf6ab471`
- 架构与交付状态：待最终架构验证、push、PR 和远端读回；最终状态以交付消息为准。
