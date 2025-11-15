# 在三张NVIDIA Pro 6000上运行VeriLLM实验指南

## 硬件配置
- GPU: 3x NVIDIA Pro 6000 (每张24GB显存)
- 优势: 可以并行运行多个实验或在不同GPU上做异构测试

## 快速开始

### 1. 环境设置

```bash
# 克隆仓库
git clone https://github.com/Ed1sonL1-byte/verillm-experiments.git
cd verillm-experiments

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装量化支持（用于实验3-5）
pip install autoawq>=0.1.8
```

### 2. 验证环境（重要！）

```bash
# 检查GPU
nvidia-smi

# 验证PyTorch可以访问所有GPU
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# 运行环境验证脚本
python scripts/verify_experiments.py --test-inference
```

**期望输出：**
```
GPUs available: 3
GPU 0: NVIDIA RTX 6000 Ada Generation
GPU 1: NVIDIA RTX 6000 Ada Generation
GPU 2: NVIDIA RTX 6000 Ada Generation
```

### 3. 下载模型

```bash
# 下载所有需要的模型
bash scripts/download_models.sh

# 或手动下载
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
# 量化模型（实验3-5需要）
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct-AWQ')
"
```

---

## 实验运行策略

### 方案1: 按顺序运行（推荐新手）

**在一张GPU上顺序运行所有实验：**

```bash
cd experiments

# 实验1: 同构硬件验证（使用GPU 0）
python exp1_homogeneous.py
# 修改代码中的 DEVICE = "cuda:0"

# 实验2: 异构硬件（使用GPU 0和GPU 1模拟异构）
python exp2_heterogeneous.py
# 修改代码：
# INFERENCE_DEVICE = "cuda:0"
# VERIFICATION_DEVICE = "cuda:1"

# 实验3: 量化推理+同构验证（GPU 0）
python exp3_quantized_inference_homogeneous.py
# INFERENCE_DEVICE = "cuda:0"
# VERIFICATION_DEVICE = "cuda:0"

# 实验4: 量化推理+异构验证（GPU 0和1）
python exp4_quantized_inference_heterogeneous.py
# INFERENCE_DEVICE = "cuda:0"
# VERIFICATION_DEVICE = "cuda:1"

# 实验5: FP16推理+量化验证（GPU 0）
python exp5_full_inference_quantized_verification.py
# INFERENCE_DEVICE = "cuda:0"
# VERIFICATION_DEVICE = "cuda:0"
```

---

### 方案2: 并行运行（充分利用3张GPU，推荐）

**同时运行多个实验，节省时间：**

#### 修改实验配置

1. **实验1配置** (`exp1_homogeneous.py`):
```python
DEVICE = "cuda:0"  # 使用第一张卡
```

2. **实验2配置** (`exp2_heterogeneous.py`):
```python
INFERENCE_DEVICE = "cuda:0"
VERIFICATION_DEVICE = "cuda:1"  # 使用第二张卡做验证
```

3. **实验3配置** (`exp3_quantized_inference_homogeneous.py`):
```python
INFERENCE_DEVICE = "cuda:2"      # 使用第三张卡
VERIFICATION_DEVICE = "cuda:2"
```

4. **实验4配置** (`exp4_quantized_inference_heterogeneous.py`):
```python
INFERENCE_DEVICE = "cuda:1"
VERIFICATION_DEVICE = "cuda:2"
```

5. **实验5配置** (`exp5_full_inference_quantized_verification.py`):
```python
INFERENCE_DEVICE = "cuda:0"
VERIFICATION_DEVICE = "cuda:1"
```

#### 并行运行脚本

创建 `run_all_parallel.sh`:

```bash
#!/bin/bash

cd experiments

# 在后台运行实验1（GPU 0）
nohup python exp1_homogeneous.py > ../logs/exp1.log 2>&1 &
echo "Experiment 1 started on GPU 0 (PID: $!)"

# 等待5秒避免同时加载模型
sleep 5

# 在后台运行实验3（GPU 2）
nohup python exp3_quantized_inference_homogeneous.py > ../logs/exp3.log 2>&1 &
echo "Experiment 3 started on GPU 2 (PID: $!)"

# 等待实验1和3完成后，运行实验2
wait

# 运行实验2（GPU 0和1）
nohup python exp2_heterogeneous.py > ../logs/exp2.log 2>&1 &
echo "Experiment 2 started on GPU 0,1 (PID: $!)"

# 同时运行实验4（GPU 1和2）
sleep 5
nohup python exp4_quantized_inference_heterogeneous.py > ../logs/exp4.log 2>&1 &
echo "Experiment 4 started on GPU 1,2 (PID: $!)"

wait

# 最后运行实验5
python exp5_full_inference_quantized_verification.py > ../logs/exp5.log 2>&1
echo "Experiment 5 completed"

echo "All experiments completed!"
```

运行：
```bash
chmod +x run_all_parallel.sh
./run_all_parallel.sh
```

**监控进度：**
```bash
# 监控GPU使用情况
watch -n 1 nvidia-smi

# 查看实时日志
tail -f logs/exp1.log
tail -f logs/exp2.log
tail -f logs/exp3.log
```

---

### 方案3: 最大化性能（推荐经验用户）

**同时在3张卡上运行不同配置：**

```bash
# Terminal 1 - GPU 0
cd experiments
CUDA_VISIBLE_DEVICES=0 python exp1_homogeneous.py

# Terminal 2 - GPU 1
cd experiments
CUDA_VISIBLE_DEVICES=1 python exp3_quantized_inference_homogeneous.py

# Terminal 3 - GPU 2
cd experiments
CUDA_VISIBLE_DEVICES=2 python exp5_full_inference_quantized_verification.py
```

或使用tmux并行管理：

```bash
# 安装tmux（如果没有）
sudo apt-get install tmux

# 启动tmux会话
tmux new -s verillm

# 分割窗口（Ctrl+B 然后按 %）
# 在每个窗口中：
# 窗口1
cd experiments && CUDA_VISIBLE_DEVICES=0 python exp1_homogeneous.py

# 窗口2（Ctrl+B %）
cd experiments && CUDA_VISIBLE_DEVICES=1 python exp3_quantized_inference_homogeneous.py

# 窗口3（Ctrl+B %）
cd experiments && CUDA_VISIBLE_DEVICES=2 python exp5_full_inference_quantized_verification.py
```

---

## 阈值优化（推荐在运行实验后执行）

充分利用多GPU并行优化：

```bash
# 在GPU 0上优化实验1和2
CUDA_VISIBLE_DEVICES=0 nohup python scripts/optimize_thresholds.py --experiment exp1 --trials 20 --device cuda:0 > logs/opt_exp1.log 2>&1 &

# 在GPU 1上优化实验3和4
CUDA_VISIBLE_DEVICES=1 nohup python scripts/optimize_thresholds.py --experiment exp3 --trials 20 --device cuda:0 > logs/opt_exp3.log 2>&1 &

# 在GPU 2上优化实验5
CUDA_VISIBLE_DEVICES=2 nohup python scripts/optimize_thresholds.py --experiment exp5 --trials 20 --device cuda:0 > logs/opt_exp5.log 2>&1 &

# 查看进度
tail -f logs/opt_exp*.log
```

---

## GPU显存管理

### 预估显存使用（每个实验）

| 实验 | 推理阶段 | 验证阶段 | 总计（单张卡） |
|------|---------|---------|---------------|
| Exp 1 | ~18GB | ~18GB | ~18GB（同一张卡） |
| Exp 2 | ~18GB | ~18GB | 每张卡~18GB |
| Exp 3 | ~10GB（量化） | ~18GB | ~18GB |
| Exp 4 | ~10GB | ~18GB | 每张卡分别 |
| Exp 5 | ~18GB | ~10GB | ~18GB |

**Pro 6000 (24GB) 完全够用！**

### 如果遇到OOM (Out of Memory)

修改实验代码，减少生成长度：

```python
# 在实验文件中找到这行
max_new_tokens=1000

# 改为
max_new_tokens=500  # 或 750
```

---

## 快速检查清单

运行前确认：

- [ ] `nvidia-smi` 显示3张GPU
- [ ] `python scripts/verify_experiments.py` 通过所有检查
- [ ] 模型已下载（检查 `~/.cache/huggingface/` 或 `./models/`）
- [ ] 虚拟环境已激活
- [ ] 足够的磁盘空间（至少100GB）

---

## 预期运行时间（3张GPU并行）

| 策略 | 总时间 |
|------|--------|
| 顺序运行 | ~60-90分钟 |
| 并行运行（方案2） | ~30-40分钟 |
| 最大化并行（方案3） | ~20-30分钟 |

---

## 故障排查

### GPU不可见

```bash
# 检查CUDA
nvidia-smi

# 检查PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# 如果false，重装PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 端口/进程冲突

```bash
# 查看GPU使用情况
nvidia-smi

# 杀掉僵尸进程
pkill -f python
```

### 显存不足

```bash
# 清理缓存
python -c "import torch; torch.cuda.empty_cache()"

# 或重启
sudo systemctl restart nvidia-persistenced
```

---

## 推荐工作流程（最优）

```bash
# 1. 验证环境
python scripts/verify_experiments.py --test-inference

# 2. 下载模型
bash scripts/download_models.sh

# 3. 并行运行实验（使用方案2的脚本）
./run_all_parallel.sh

# 4. 监控进度
watch -n 1 nvidia-smi

# 5. 等待完成后，并行优化阈值
CUDA_VISIBLE_DEVICES=0 python scripts/optimize_thresholds.py --experiment exp1 --trials 20 &
CUDA_VISIBLE_DEVICES=1 python scripts/optimize_thresholds.py --experiment exp3 --trials 20 &
CUDA_VISIBLE_DEVICES=2 python scripts/optimize_thresholds.py --experiment exp5 --trials 20 &

# 6. 查看结果
for exp in exp1 exp2 exp3 exp4 exp5; do
  echo "=== $exp ==="
  cat data/raw/$exp/*_summary.json 2>/dev/null | jq '.aggregate' | head -20
done
```

---

## 联系支持

遇到问题请运行：
```bash
python scripts/verify_experiments.py > env_report.txt
nvidia-smi > gpu_report.txt
```

然后提供这两个文件的内容。
