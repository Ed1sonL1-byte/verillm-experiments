# VeriLLM 实验项目进度

## 项目概述
VeriLLM 是一个可公开验证的去中心化 LLM 推理协议。本项目实现论文中的验证机制实验。

## 当前进度

### 实验1：同构硬件基线 ✅ 已完成
- **目标**：在相同硬件上进行推理和验证，建立误差基线
- **结果**：10 次实验已完成
- **数据位置**：`data/raw/exp1/`
- **统计结果**（记录在 `configs/experiments.yaml` 注释中）：
  ```
  Pe:          mean=0.008982, p90=0.015625
  Pm:          mean=0.602678, p50=0.842634
  Pw:          mean=0.397317, p50=0.157366
  Mean Error:  mean=0.001503, p95=0.004475
  Accept Rate: mean=0.9884, min=0.9607, max=0.9991
  ```

### 实验2：异构硬件验证 🔄 进行中
- **目标**：跨硬件平台（NVIDIA GPU ↔ Mac M系列）的推理和验证
- **设计**：分两步进行
  1. 步骤1（NVIDIA服务器）：运行推理，保存 hidden states
  2. 步骤2（Mac）：读取 hidden states，运行 prefill 验证

#### 步骤1：NVIDIA 推理 ✅ 已完成
- **数据位置**：`data/raw/exp2/inference/`
- **包含文件**：
  - `*_metadata.json` - 元数据（prompt, generated_tokens, timing 等）
  - `*_hidden_states.pkl` - hidden states 数据（pickle 格式）
  - `exp2_inference_summary.json` - 汇总信息
- **共 10 个 trials**

#### 步骤2：Mac 验证 ⏳ 待执行
需要在 Mac 上执行以下操作：

1. **下载数据**（如果还没下载）：
   ```bash
   mkdir -p /path/to/verillm-experiments/data/raw/exp2/inference
   scp -r nvserver:/home/edison/verillm-experiments/data/raw/exp2/inference/* ./data/raw/exp2/inference/
   ```

2. **运行验证脚本**：
   ```bash
   cd /path/to/verillm-experiments
   python scripts/exp2_step2_verification.py \
       --input-dir data/raw/exp2/inference \
       --device mps \
       --num-verifiers 3
   ```

3. **预期输出**：
   - `data/raw/exp2/verification/` 目录下的验证结果
   - 每个 trial 的 JSON 结果文件
   - 汇总统计（accept rate, overhead 等）

### 实验3-5：量化攻击检测 ⏳ 待执行
- 实验3：同构硬件 + 量化推理 vs 全精度验证
- 实验4：异构硬件 + 量化攻击
- 实验5：全精度推理 + 量化验证（懒惰验证者）

## 关键文件

### 脚本
- `scripts/exp2_step1_inference_only.py` - 实验2步骤1：NVIDIA 推理
- `scripts/exp2_step2_verification.py` - 实验2步骤2：Mac 验证
- `scripts/run_parallel_experiments.py` - 多 GPU 并行实验

### 配置
- `configs/experiments.yaml` - 实验配置和阈值
- `configs/prompts.yaml` - 测试 prompt 模板（30个）
- `configs/models.yaml` - 模型配置

### 实验代码
- `experiments/exp1_homogeneous.py` - 实验1
- `experiments/exp2_heterogeneous.py` - 实验2（原版，需要同时访问两种硬件）
- `experiments/base_experiment.py` - 基类

## 验证阈值

论文阈值 vs 实验阈值对比：

| 参数 | 论文阈值 | 实验1阈值 | 说明 |
|------|---------|-----------|------|
| Pe (指数位不匹配率) | ≤ 0.05 | ≤ 0.0156 | 实验更严格 |
| Pm (大尾数偏差) | ≥ 0.75 | ≥ 0.75 | 相同 |
| Pw (小尾数偏差) | ≥ 0.50 | ≥ 0.50 | 相同 |
| mean_epsilon | ≤ 0.01 | ≤ 0.0054 | 实验更严格 |

## 硬件信息

### NVIDIA 服务器 (nvserver)
- 3x NVIDIA GPU
- 用于推理和同构实验

### Mac 服务器
- Apple Silicon (M系列)
- 用于异构验证（实验2步骤2）

## 下一步工作
1. 在 Mac 上运行实验2步骤2验证
2. 分析异构硬件的验证结果
3. 运行实验3-5（量化攻击检测）
