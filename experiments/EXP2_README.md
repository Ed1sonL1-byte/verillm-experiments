# Experiment 2: Heterogeneous Hardware Verification

## 目标

测试 VeriLLM 验证机制在**异构硬件**环境下的可靠性：
- 推理设备和验证设备使用不同硬件平台
- 评估跨设备浮点误差对验证准确性的影响

## 实验配置

### 硬件组合

| 场景 | 推理设备 | 验证设备 | 预期Accept Rate |
|------|---------|---------|----------------|
| **Scenario 1** | NVIDIA GPU (A100/3090/4090) | Mac M-series (M1/M2/M3/M4) | >90% |
| **Scenario 2** | Mac M-series | NVIDIA GPU | >90% |

### 软件配置

- **模型**: Qwen2.5-7B-Instruct (FP16)
- **输入长度**: ~300 tokens
- **输出长度**: 1000 tokens
- **验证者数量**: 3
- **Hidden States 采样**: 每隔8层（Layer 0, 8, 16, 24, 32）

### 阈值（针对异构硬件放宽）

```yaml
Pe: 0.05     # 指数位不匹配率 < 5%
Pm: 0.75     # 大尾数偏差比例 > 75%
Pw: 0.50     # 小尾数偏差比例 > 50%
mean_epsilon: 0.01  # 平均误差 < 0.01
```

**判决标准**: Accept Rate > **90%** (比同构硬件的95%稍宽松)

## 运行方式

### 方式1：运行单个场景

```bash
cd experiments

# Scenario 1: NVIDIA → Mac
python exp2_heterogeneous.py

# 或者在代码中修改配置运行 Scenario 2: Mac → NVIDIA
```

### 方式2：运行两个场景（推荐）

```bash
cd experiments
python run_exp2_both_scenarios.py
```

## 预期结果

### Accept Rate

| 场景 | 预期范围 | 说明 |
|------|---------|------|
| NVIDIA → Mac | 90-95% | 跨硬件浮点误差略高于同构 |
| Mac → NVIDIA | 90-95% | 同上 |

**对比实验1（同构）**:
- 实验1: 99.79% (A100 → A100)
- 实验2: 预计 90-95% (跨硬件)

### Verification Overhead

- **预期**: 0.01-0.02% (与实验1类似)
- **原因**: Prefill验证非常快，与硬件平台关系不大

### Pass Rate

- **预期**: 100% (3/3 trials)
- **说明**: 即使Accept Rate略低，仍应通过90%阈值

## 关键研究问题

1. **硬件差异影响**:
   - NVIDIA GPU 和 Mac M-series 的浮点运算实现不同
   - 是否会导致显著的 hidden states 差异？

2. **方向性差异**:
   - NVIDIA → Mac 和 Mac → NVIDIA 是否有不同表现？
   - 哪个方向的误差更大？

3. **误差累积**:
   - 跨硬件是否导致更多的指数位不匹配（Pe）？
   - 层级越深，误差累积是否更明显？

## 输出文件

```
data/raw/exp2/
├── qwen2.5-7b_cuda:0_to_mps_trial1.json
├── qwen2.5-7b_cuda:0_to_mps_trial2.json
├── qwen2.5-7b_cuda:0_to_mps_trial3.json
├── qwen2.5-7b_cuda:0_to_mps_summary.json
├── qwen2.5-7b_mps_to_cuda:0_trial1.json
├── qwen2.5-7b_mps_to_cuda:0_trial2.json
├── qwen2.5-7b_mps_to_cuda:0_trial3.json
└── qwen2.5-7b_mps_to_cuda:0_summary.json
```

## 结果分析

### 查看摘要

```bash
# Scenario 1 结果
cat data/raw/exp2/qwen2.5-7b_cuda:0_to_mps_summary.json | jq '.aggregate'

# Scenario 2 结果
cat data/raw/exp2/qwen2.5-7b_mps_to_cuda:0_summary.json | jq '.aggregate'
```

### 关键指标

```json
{
  "avg_accept_rate": 0.92,      // 平均接受率
  "avg_overhead_percentage": 0.01,  // 平均验证开销
  "pass_rate": 1.0              // 试验通过率
}
```

## 与论文对比

| 指标 | VeriLLM论文 | 实验2预期 | 说明 |
|------|------------|----------|------|
| **Accept Rate** | >90% | 90-95% | 符合预期 |
| **Overhead** | ~1% | 0.01% | 更优（得益于prefill-only设计） |
| **Pass Rate** | - | 100% | 稳定性验证 |

## 故障排查

### Mac M-series 相关

如果遇到 `RuntimeError: MPS backend out of memory`:

```python
# 在代码中添加
import torch.mps
torch.mps.empty_cache()
```

### CUDA 相关

如果遇到 `CUDA out of memory`:

```python
import torch
torch.cuda.empty_cache()
```

### 模型加载失败

确认 HuggingFace 模型已下载：

```bash
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"
```

## 下一步

完成实验2后，可以进行：
- **实验3**: 量化推理攻击检测（同构硬件）
- **实验4**: 量化推理攻击检测（异构硬件）
- **实验5**: Lazy Verifier 检测

## 参考

- VeriLLM Paper: Section 8.1 (Heterogeneous Hardware Results)
- 实验需求文档: `# 实验2：在异构硬件上校验推理`
