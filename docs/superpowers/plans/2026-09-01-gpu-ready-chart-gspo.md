# Chart GSPO GPU-Ready Implementation Plan

> **For agentic workers:** Execute task-by-task with one CPU test cycle per task.

**Goal:** Replace the chart GSPO guard with course-compatible data preparation and Unsloth/TRL GSPO LoRA training.

**Architecture:** A stdlib adapter validates image JSONL rows. The GPU entrypoint uses the course GRPOConfig GSPO settings and FastVisionModel only after CPU config validation passes.

### Task 1: Chart task preparation

- [ ] Add a JSONL/image-path preparation function and test missing images.
- [ ] Run test red, implement, run green, commit.

### Task 2: Profiled GSPO launcher

- [ ] Add V100/full-profile dry-run tests.
- [ ] Replace `train_gspo.py` guard with Unsloth + TRL GRPOConfig GSPO launcher and checkpoint resume.
- [ ] Run full CPU tests, compile, dry-run, commit.
