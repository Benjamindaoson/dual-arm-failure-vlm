# 多模态图表推理 GSPO 后训练系统

项目保留课程 Qwen3-VL GSPO notebook 与基线记录，并新增可解析答案、格式奖励、任务维度准确率评测。课程 notebook 属于 legacy 路径，升级评测代码位于 `upgraded_implementation/`。

CPU：`python scripts/prepare_course_materials.py` 后运行 `python -m unittest discover -s tests -v`。

## GPU 训练（单卡 V100 32GB smoke）

默认 profile 用 4-bit Qwen2.5-VL-3B、FP16、MathVista `testmini` 和 2 个候选生成，直接复用课件的 `FastVisionModel + GRPOTrainer` 训练路径；奖励函数改为本项目的答案正确性和 `<answer>` 格式奖励。

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-gpu.txt
python scripts/train_gspo.py --dry-run
python scripts/train_gspo.py
```

课件中的 Qwen3-VL-8B / BF16 配置保留在 `legacy_reproduction/`，不应在单张 V100 上直接运行；需要它时应换支持 BF16 且显存更大的 GPU。
