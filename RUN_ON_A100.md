# Running VeriLLM Experiment 1 on 8×A100 GPUs

## 🎯 Quick Start (TL;DR)

```bash
# On your A100 server
cd /path/to/verillm-experiments

# Make script executable
chmod +x run_exp1_a100.sh

# Run experiment
./run_exp1_a100.sh
```

**Expected time**: ~30-45 minutes total (3 prompts × ~10-15 min each)

---

## 📋 Detailed Step-by-Step Guide

### Step 1: SSH to A100 Server

```bash
ssh your-username@your-a100-server

# Navigate to project
cd /path/to/verillm-experiments
```

### Step 2: Verify GPU Access

```bash
nvidia-smi

# Expected output: Should show 8× A100 GPUs
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |                               |                      |               MIG M. |
# |===============================+======================+======================|
# |   0  NVIDIA A100-SXM...  On   | 00000000:00:04.0 Off |                    0 |
# | N/A   30C    P0    56W / 400W |      0MiB / 81920MiB |      0%      Default |
# ...
# |   7  NVIDIA A100-SXM...  On   | 00000000:00:0B.0 Off |                    0 |
```

### Step 3: Activate Environment

```bash
# If venv doesn't exist, create it:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Otherwise, just activate:
source venv/bin/activate

# Verify CUDA in PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
# Expected: CUDA: True, GPUs: 8
```

### Step 4: Configure Model Download (First Time Only)

实验会自动从HuggingFace下载模型。有两种方式：

**Option A: 自动下载（推荐）**
```bash
# 模型会自动下载到 ~/.cache/huggingface/
# 第一次运行会比较慢（~20GB下载）
# 后续运行会使用缓存
```

**Option B: 预先下载到本地**
```bash
# 编辑 configs/models.yaml，确保 local_path 正确
# 如果已有模型文件，将它们放在 ./models/ 目录

mkdir -p models/qwen2.5-7b

# 从HuggingFace下载（需要git-lfs）
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct models/qwen2.5-7b
```

### Step 5: Run Experiment

**方式1：使用脚本（推荐）**
```bash
chmod +x run_exp1_a100.sh
./run_exp1_a100.sh
```

**方式2：直接运行Python**
```bash
cd experiments
python exp1_homogeneous.py
```

**方式3：使用nohup后台运行**
```bash
# 如果实验时间长，使用nohup防止SSH断开导致中断
nohup ./run_exp1_a100.sh > exp1_output.log 2>&1 &

# 查看进度
tail -f exp1_output.log

# 查看运行状态
ps aux | grep exp1
```

---

## 📊 实验运行过程

### 你会看到的输出

```
================================================================================
EXPERIMENT 1: Homogeneous Hardware Baseline
Model: qwen2.5-7b, Device: cuda:0
Number of verifiers per trial: 3
================================================================================

================================================================================
Trial 1: qwen2.5-7b on cuda:0
Prompt length: 245 chars
================================================================================

Loading model qwen2.5-7b to cuda:0...
Starting inference (Prefill + Decode)...
生成中: 100%|███████████████████████████| 856/1000 [02:07<00:00,  6.72it/s]

Inference complete: generated 856 tokens
Inference time: 127.34s

Starting verification 1/3 (Prefill only)...
Verification 1 complete: 0.98s

Starting verification 2/3 (Prefill only)...
Verification 2 complete: 1.02s

Starting verification 3/3 (Prefill only)...
Verification 3 complete: 0.95s

Verifier 1: Accept rate = 98.50%
Verifier 1: Overhead = 0.77%

Verifier 2: Accept rate = 98.52%
Verifier 2: Overhead = 0.80%

Verifier 3: Accept rate = 98.48%
Verifier 3: Overhead = 0.75%

Result saved to: ../data/raw/exp1/qwen2.5-7b_cuda:0_trial1.json
```

### 时间估计（单个A100）

| 阶段 | 时间 | 说明 |
|------|------|------|
| 模型加载 | ~30秒 | 首次下载可能需要5-10分钟 |
| 推理（Prefill+Decode） | ~2-3分钟 | 生成~800-1000 tokens |
| 验证 3次 | ~3秒 | 每次约1秒（仅Prefill） |
| 对比分析 | ~5秒 | 计算统计量 |
| **单个trial总计** | **~3-5分钟** | |
| **3个trials总计** | **~10-15分钟** | |

---

## 📁 实验结果

### 输出文件位置

```
data/raw/exp1/
├── qwen2.5-7b_cuda:0_trial1.json    # Trial 1结果
├── qwen2.5-7b_cuda:0_trial2.json    # Trial 2结果
├── qwen2.5-7b_cuda:0_trial3.json    # Trial 3结果
└── qwen2.5-7b_cuda:0_summary.json   # 汇总结果（最重要）
```

### 查看结果

```bash
# 查看汇总结果
cat data/raw/exp1/qwen2.5-7b_cuda:0_summary.json | jq '.aggregate'

# 输出示例：
# {
#   "avg_accept_rate": 0.9850,
#   "avg_overhead_percentage": 0.77,
#   "pass_rate": 1.0
# }

# 查看单个trial的详细统计
cat data/raw/exp1/qwen2.5-7b_cuda:0_trial1.json | jq '.verifiers[0].statistics.summary'

# 输出示例：
# {
#   "accept_rate": 0.985,
#   "accept_count": 2891,
#   "total_count": 2934,
#   "avg_mean_error": 0.0089,
#   "avg_Pe": 0.0482
# }
```

### 预期结果（对比论文Table 3）

| Metric | Expected (RTX 5090) | Your A100 | Status |
|--------|---------------------|-----------|--------|
| Accept Rate | > 95% | ~98.5% | ✅ |
| Overhead | ~0.8-1% | ~0.77% | ✅ |
| Mean ε | < 0.01 | ~0.009 | ✅ |
| Pe | < 0.05 | ~0.048 | ✅ |

---

## 🚀 多GPU并行运行（可选）

如果想利用全部8张A100并行跑不同prompts：

### 创建并行脚本 `run_exp1_parallel.sh`

```bash
#!/bin/bash
# Run 8 prompts in parallel on 8 A100s

for gpu_id in {0..7}; do
    echo "Starting on GPU $gpu_id..."
    CUDA_VISIBLE_DEVICES=$gpu_id python experiments/exp1_homogeneous.py \
        --model qwen2.5-7b \
        --device cuda:0 \
        --gpu-id $gpu_id \
        > logs/exp1_gpu${gpu_id}.log 2>&1 &
done

echo "All 8 GPUs started. Check progress:"
echo "  tail -f logs/exp1_gpu*.log"

wait
echo "All parallel jobs complete!"
```

**注意**：需要修改 `exp1_homogeneous.py` 支持命令行参数（可选优化）

---

## 🔧 故障排查

### 问题1: CUDA Out of Memory

```bash
# 解决方案1：减少生成长度
# 编辑 experiments/exp1_homogeneous.py
# 将 max_new_tokens=1000 改为 max_new_tokens=500

# 解决方案2：使用更小的模型
# MODEL_NAME = "qwen2.5-3b"  # 如果有的话
```

### 问题2: 模型下载失败

```bash
# 检查网络
ping huggingface.co

# 使用国内镜像（如果在中国）
export HF_ENDPOINT=https://hf-mirror.com
./run_exp1_a100.sh

# 或手动设置镜像
export HF_DATASETS_OFFLINE=1
```

### 问题3: 导入错误

```bash
# 确保在项目根目录运行
pwd  # 应该显示 .../verillm-experiments

# 重新安装依赖
pip install -r requirements.txt --force-reinstall
```

### 问题4: 找不到configs文件

```bash
# 检查目录结构
ls configs/
# 应该有: experiments.yaml  models.yaml  prompts.yaml

# 如果缺失，检查是否在正确目录
cd /path/to/verillm-experiments
```

---

## 📈 下一步

实验1成功后，你可以：

1. **分析结果**：对比你的A100数据与论文Table 3
2. **调整参数**：尝试不同的prompts、模型
3. **运行实验2-5**：按照 [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)

---

## 📞 需要帮助？

遇到问题可以：
1. 检查 `logs/exp1_homogeneous_*.log` 日志文件
2. 查看 [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)
3. 查看 [QUICKSTART.md](QUICKSTART.md)
