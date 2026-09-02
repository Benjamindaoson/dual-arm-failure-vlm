# Chart GSPO GPU training requirements

## ADDED Requirements

### Requirement: Chart samples are validated before training

系统 SHALL 仅接受存在的图像路径、非空问题和可解析参考答案，并对拒绝行写出原因。

#### Scenario: Missing image is rejected

- **WHEN** 样本的 image_path 不存在
- **THEN** 样本不得进入训练 JSONL，拒绝原因是 `missing_image`。

### Requirement: Hardware profiles prevent infeasible runs

训练脚本 SHALL 将 `single_v100_smoke` 限制为 FP16 小模型，将 8B 正式后训练限制到 full profile。

#### Scenario: V100 cannot request full 8B profile

- **WHEN** profile 为 single_v100_smoke 且模型为 8B
- **THEN** dry-run 以配置错误失败。

### Requirement: Training uses composable rewards

GPU 训练 SHALL 使用正确性和答案格式 reward，并保存用于评测的配置副本。

#### Scenario: Correct malformed answer receives only correctness reward

- **WHEN** 回答答案值正确但未遵守 `<answer>` 格式
- **THEN** 正确性 reward 为正，格式 reward 为零。
