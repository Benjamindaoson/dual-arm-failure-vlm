# 面试叙事

## 30 秒版本

我做的是一个公开真实机器人数据上的 failure-aware 多模态后训练项目。核心不是控制机器人，也不是做检测，而是让 VLM 根据多视角时序视觉和 robot trace 判断装配阶段、nominal/failure/recovery 状态以及 failure mode。我固定 REBOOT 数据 revision，对 53,886 条 frame rows 做全量审计，隔离 7 个异常 episode，再做 episode-level 42/5/6 划分，避免 frame leakage。模型路线是 Base → QLoRA SFT → RLVR，但每一阶段都由 held-out evidence gate 决定；目标 SSH 容器未挂载 CUDA，所以我只报告已验证的数据和运行链路，模型指标是 N/A。

## 2 分钟版本

这个项目来自一个实际问题：精密装配里，局部动作看起来都合理，但轻微 misalignment、slip、jamming 或 premature release 会让长任务失败。单帧很难判断“这是尚未对齐、已经失败，还是正在恢复”，所以我把任务定义为 Failure-Aware Execution Critic，而不是一开始就做 end-to-end action control。

我先处理数据真实性。对 REBOOT sample 固定 revision，读取真实 `info.json`、`phase.json` 和全部 parquet frame tables。审计确认有 60 个 episode、53,886 条 frame rows、30 FPS、4 路 RGB、14-D state 和 14-D action。`duration_frames=897` 在该快照中表示含终点索引，正常应有 898 行；有 6 个 episode 额外包含越界 frame 898，另有 originating phase 越界和 recovery 早于 failure 的注释问题。我的处理不是修数据，而是 quarantine 并保留原因、原始哈希和 split receipt。最终 53 个可用 episode 固定成 42/5/6，所有 469 个 window 都继承 episode split。

模型输出是严格 JSON：phase、state、failure_mode。评测主指标是 failure recall，同时报告 state macro-F1、recovery recall、phase/failure-mode F1、JSON valid rate、per-class 指标和 per-episode bootstrap CI。我还设计了 A0 单帧、A1 单相机时序、A2 双相机时序、A3 双相机加 Trace-Text 四组消融，并以 failure onset 为零点计算连续 K 个窗口正确后的 detection delay 和提前误报。

训练不是默认答案。Base 达到 failure recall 0.90 且 state macro-F1 0.85 就停止；否则才做 QLoRA SFT。SFT 如果没有干净增益，结论是回到数据或任务，而不是硬上 RL。只有 SFT 有增益但仍有 outcome error，才比较 GRPO 与 sequence-level GSPO。格式只是 validity gate，reward 来自 phase/state/failure correctness。我下载并验证了完整 2.7 GB 数据、7.1 GB 模型快照，真实解码了视频并验证物化链路；但 fresh SSH 登录、设备节点和 PyTorch 三重证据都表明目标容器没有 GPU。为避免 CPU 假跑，我没有写任何模型性能数字，所有 Base/SFT/RL 指标均为 N/A。

## 5 分钟版本

### 1. 问题定义

传统成功率把长任务压缩成一个 bit，看不到机器人在哪个阶段开始偏离。REBOOT 把装配统一成 Align(pick)、Engage(pick)、Transport、Align(place)、Engage(place) 五个阶段，并提供失败注入点、恢复开始点和 failure mode。这使“执行 Critic”成为可验证任务：输入历史，输出当前 phase、state 与 failure mode。

我刻意没有直接预测 action。原因是当前 sample 只有 60 条 recovery trajectories，先验证感知和状态解释是否成立，比让 3B VLM 直接控制接触丰富的机器人更稳妥。未来 recovery policy 可以消费这个结构化状态，但本阶段不凭空造 recovery action label。

### 2. 数据证据

项目最重要的工作不是模型参数，而是先证明数据契约。固定 Hugging Face revision 后，我同时检查 frame-table schema 与 phase annotations：camera key、RGB/depth、state/action shape、episode index 类型、phase boundary、originating phase、failure/recovery timing、description、重复和缺失字段。

审计发现 sample schema 没有 depth，尽管 full REBOOT 项目描述为 RGB-D；还发现 episode index 在 frame table 是 int64，在 annotation 是补零字符串。更关键的是两个 originating phase 越界、一个 recovery 时间早于 failure。它们进入 quarantine，不参加训练。所有统计、SHA 和异常原因写成 machine-readable JSON 与 Markdown receipt。

### 3. 评测设计

剩余 episode 先按 seed 42 做 80/10/10 episode split。然后每个 window 只属于一个 episode。A0–A3 必须共享同一组 test episode，否则消融差异可能只是数据差异。

除了常规 F1，我把 failure recall 放在第一位，因为漏检 failure 比格式错误更接近工业风险。failure timing 使用相对 onset 的七个时间点，输入窗口因果采样，不偷看未来帧；首次连续两个窗口预测 failure 的时刻作为 delay，同时统计 onset 前 false alarm。

### 4. 后训练与 reward

Base 模型是 Qwen2.5-VL-3B-Instruct，理由是 24 GB GPU 可承载多图和 4-bit 推理。SFT 用 NF4 QLoRA，比较 language-only 与 vision-language 两个适配范围，问题不是“视觉一定要微调”，而是瓶颈究竟在视觉表征还是状态解释。

RLVR reward 先解析严格 JSON；解析失败直接为零，但解析成功本身不加分。reward 由 phase、state、failure-mode correctness 构成；nominal 必须输出 none；failure mode 使用逆平方根权重缓解不平衡。GRPO 和 GSPO 使用相同 SFT checkpoint、数据和 reward，只改变 importance sampling 粒度。GSPO 配置是 sequence-level IS 加 `loss_type=grpo`；`sequence + dr_grpo` 会被配置验证器拒绝，避免错误命名。

### 5. 当前结论与下一步

当前可以确认的是数据与工程层：revision、60/53/7 episode、42/5/6 split、469 个 pilot windows、84 条 timing 样本、完整数据与模型快照、真实视频解码、39 项本地和远端测试。不能确认的是 Base/SFT/RL 指标，因为目标 AutoDL SSH 容器没有 `/dev/nvidia*` 且 PyTorch CUDA 不可用。

恢复条件很明确：让同一 SSH 端点真正挂载 4090 后，直接复用已验证的本地数据、模型和物化样本，跑 Base A0–A2；由 Base gate 决定是否 SFT，再由 SFT gate 决定是否 RLVR。这样面试时可以严格区分“实现了什么”“执行了什么”“证据支持什么”，而不是把计划包装成结果。
