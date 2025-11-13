# 实验3、4、5：量化模型验证实验

## 概述

这三个实验探讨了**量化模型**在VeriLLM验证机制中的表现，测试量化对验证可靠性的影响。

| 实验 | 推理模型 | 验证模型 | 硬件环境 | 预期Accept Rate |
|------|---------|---------|---------|----------------|
| **实验3** | 量化模型(INT4) | 正常模型(FP16) | 同构 | >80% |
| **实验4** | 量化模型(INT4) | 正常模型(FP16) | 异构 | >70% |
| **实验5** | 正常模型(FP16) | 量化模型(INT4) | 同构 | >80% |

---

## 实验3：量化推理 + 同构硬件验证

### 目标
测试使用**量化模型做推理**，**正常精度模型做验证**在同构硬件上的可靠性。

### 配置

**硬件组合**：
- 推理设备：CUDA:0 (量化模型)
- 验证设备：CUDA:1 或 CUDA:0 (正常模型)
- 或：MPS → MPS (需要加载/卸载模型)

**模型精度**：
- 推理：INT4/AWQ 量化模型
- 验证：FP16 正常精度模型

**其他参数**：
- 模型：Qwen2.5-7B-Instruct
- 输出长度：1000 tokens
- 验证者数量：3
- Hidden States 采样：每隔8层

### 运行方式

```bash
cd experiments
python exp3_quantized_inference_homogeneous.py
```

**修改配置**：
在 `exp3_quantized_inference_homogeneous.py` 中修改：
```python
MODEL_NAME = "qwen2.5-7b"
INFERENCE_DEVICE = "cuda:0"      # 量化模型
VERIFICATION_DEVICE = "cuda:1"   # 正常模型
NUM_VERIFIERS = 3
```

### 预期结果

- **Accept Rate**: 80-85%
- **Verification Overhead**: 0.01-0.02%
- **Pass Rate**: 100% (3/3 trials)

**关键发现**：
- 量化误差导致Accept Rate低于同构FP16实验（99%）
- 但仍然保持在可接受范围内（>80%）
- 验证开销极低

### 输出文件

```
data/raw/exp3/
├── qwen2.5-7b_quantized_cuda:0_to_fp_cuda:1_trial1.json
├── qwen2.5-7b_quantized_cuda:0_to_fp_cuda:1_trial2.json
├── qwen2.5-7b_quantized_cuda:0_to_fp_cuda:1_trial3.json
└── qwen2.5-7b_quantized_cuda:0_to_fp_cuda:1_summary.json
```

---

## 实验4：量化推理 + 异构硬件验证

### 目标
测试使用**量化模型做推理**，**正常精度模型做验证**在异构硬件上的可靠性。

### 配置

**硬件组合**：
- **场景1**：NVIDIA GPU (量化) → Mac M-series (FP16)
- **场景2**：Mac M-series (量化) → NVIDIA GPU (FP16)

**模型精度**：
- 推理：INT4/AWQ 量化模型
- 验证：FP16 正常精度模型

**其他参数**：
- 模型：Qwen2.5-7B-Instruct
- 输出长度：1000 tokens
- 验证者数量：3
- Hidden States 采样：每隔8层

### 运行方式

```bash
cd experiments
python exp4_quantized_inference_heterogeneous.py
```

**修改配置**：
在 `exp4_quantized_inference_heterogeneous.py` 中修改：
```python
MODEL_NAME = "qwen2.5-7b"

# 场景1: NVIDIA → Mac
INFERENCE_DEVICE = "cuda:0"      # 量化模型
VERIFICATION_DEVICE = "mps"      # 正常模型

# 场景2: Mac → NVIDIA
# INFERENCE_DEVICE = "mps"         # 量化模型
# VERIFICATION_DEVICE = "cuda:0"   # 正常模型

NUM_VERIFIERS = 3
```

### 预期结果

- **Accept Rate**: 70-80%
- **Verification Overhead**: 0.01-0.02%
- **Pass Rate**: 100% (3/3 trials)

**关键发现**：
- 量化误差 + 硬件异构性导致Accept Rate进一步降低
- 仍然可以通过放宽阈值（70%）来实现可靠验证
- 验证开销仍然极低

### 输出文件

```
data/raw/exp4/
├── qwen2.5-7b_quantized_cuda:0_to_fp_mps_trial1.json
├── qwen2.5-7b_quantized_cuda:0_to_fp_mps_trial2.json
├── qwen2.5-7b_quantized_cuda:0_to_fp_mps_trial3.json
└── qwen2.5-7b_quantized_cuda:0_to_fp_mps_summary.json
```

---

## 实验5：正常推理 + 量化模型验证

### 目标
测试使用**正常精度模型做推理**，**量化模型做验证**的可靠性（与实验3相反）。

### 配置

**硬件组合**：
- 推理设备：CUDA:0 (正常模型)
- 验证设备：CUDA:1 或 CUDA:0 (量化模型)
- 或：MPS → MPS

**模型精度**：
- 推理：FP16 正常精度模型
- 验证：INT4/AWQ 量化模型

**其他参数**：
- 模型：Qwen2.5-7B-Instruct
- 输出长度：1000 tokens
- 验证者数量：3
- Hidden States 采样：每隔8层

### 运行方式

```bash
cd experiments
python exp5_full_inference_quantized_verification.py
```

**修改配置**：
在 `exp5_full_inference_quantized_verification.py` 中修改：
```python
MODEL_NAME = "qwen2.5-7b"
INFERENCE_DEVICE = "cuda:0"      # 正常模型
VERIFICATION_DEVICE = "cuda:1"   # 量化模型
NUM_VERIFIERS = 3
```

### 预期结果

- **Accept Rate**: 80-85%
- **Verification Overhead**: 0.005-0.01% (量化模型更快)
- **Pass Rate**: 100% (3/3 trials)

**关键发现**：
- 量化验证者可以有效验证正常精度推理
- Accept Rate与实验3相似（80-85%）
- 验证开销可能更低（量化模型推理更快）

### 输出文件

```
data/raw/exp5/
├── qwen2.5-7b_fp_cuda:0_to_quantized_cuda:1_trial1.json
├── qwen2.5-7b_fp_cuda:0_to_quantized_cuda:1_trial2.json
├── qwen2.5-7b_fp_cuda:0_to_quantized_cuda:1_trial3.json
└── qwen2.5-7b_fp_cuda:0_to_quantized_cuda:1_summary.json
```

---

## 实验比较

### Accept Rate 对比

| 实验 | 推理精度 | 验证精度 | 硬件 | 预期Accept Rate |
|------|---------|---------|------|----------------|
| 实验1 | FP16 | FP16 | 同构 | ~99% |
| 实验2 | FP16 | FP16 | 异构 | 90-95% |
| **实验3** | **INT4** | **FP16** | **同构** | **80-85%** |
| **实验4** | **INT4** | **FP16** | **异构** | **70-80%** |
| **实验5** | **FP16** | **INT4** | **同构** | **80-85%** |

### 关键研究问题

1. **量化对验证的影响**：
   - 量化导致Accept Rate降低约10-15%
   - 但仍然可以通过调整阈值保持可靠性

2. **方向性差异**：
   - 实验3 (INT4→FP16) vs 实验5 (FP16→INT4)
   - 预期结果相似，但验证速度可能不同

3. **量化 + 异构的累积效应**：
   - 实验4测试最严苛场景
   - Accept Rate最低，但仍可用

---

## 环境要求

### 硬件
- NVIDIA GPU (推荐 RTX 4090/5090, A100)
- Mac M-series (M1/M2/M3/M4)
- 至少一个设备需要支持量化模型

### 软件依赖
```bash
# 基础依赖
pip install torch transformers accelerate

# 量化支持 (选择其一)
pip install autoawq>=0.1.8      # AWQ量化
pip install auto-gptq>=0.5.0    # GPTQ量化
```

### 模型准备
确保已下载对应的量化模型：
```bash
# 下载Qwen2.5-7B及其量化版本
python scripts/download_models.sh

# 或手动下载
python -c "
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct-AWQ')
"
```

---

## 结果分析

### 查看实验结果摘要

```bash
# 实验3结果
cat data/raw/exp3/*_summary.json | jq '.aggregate'

# 实验4结果
cat data/raw/exp4/*_summary.json | jq '.aggregate'

# 实验5结果
cat data/raw/exp5/*_summary.json | jq '.aggregate'
```

### 关键指标解读

```json
{
  "avg_accept_rate": 0.82,         // 平均接受率
  "avg_overhead_percentage": 0.01,  // 平均验证开销
  "pass_rate": 1.0                  // 试验通过率
}
```

### 可视化分析

可使用 `notebooks/` 中的Jupyter notebooks进行可视化分析：
- Accept Rate分布
- 各层Hidden States差异
- Pe, Pm, Pw 指标对比

---

## 故障排查

### 量化模型加载失败

**问题**：`ModuleNotFoundError: No module named 'awq'`

**解决**：
```bash
pip install autoawq>=0.1.8
# 或
pip install auto-gptq>=0.5.0
```

### CUDA内存不足

**问题**：`CUDA out of memory`

**解决**：
1. 使用不同的CUDA设备（cuda:0 和 cuda:1）
2. 减少生成长度：`max_new_tokens=500`
3. 清理缓存：`torch.cuda.empty_cache()`

### MPS内存不足

**问题**：`MPS backend out of memory`

**解决**：
```python
import torch.mps
torch.mps.empty_cache()
```

### 量化模型精度降低过多

**问题**：Accept Rate < 70%

**分析**：
- 可能是量化方法不适合该模型
- 尝试其他量化方法（AWQ vs GPTQ）
- 检查量化模型是否正确下载

---

## 下一步

完成实验3-5后，可以：
1. 对比不同量化方法（AWQ vs GPTQ）
2. 测试不同模型大小（7B vs 13B）
3. 分析量化误差的层级分布
4. 优化阈值参数以提高Accept Rate

## 参考

- VeriLLM Paper: Section 8 (Quantization Impact)
- 实验需求文档：`# 实验3-5`
- 模型配置：`configs/models.yaml`
