# 快速开始指南

## 1. 环境配置（5分钟）

```bash
# 解压项目
unzip verillm-experiments.zip
cd verillm-experiments

# 配置环境
bash scripts/setup_environment.sh

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

## 2. 测试环境（1分钟）

```bash
python test_setup.py
```

应该看到：
```
✅ CUDA可用 - RTX 5090
✅ 模型加载成功
✅ 推理成功
✅ 所有测试通过！
```

## 3. 下载模型（30分钟-1小时）

```bash
# 方法1：使用脚本（推荐）
bash scripts/download_models.sh

# 方法2：手动下载
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
"
```

## 4. 运行第一个实验（10-20分钟）

```bash
# 运行实验1
python experiments/exp1_homogeneous.py
```

**预期输出：**
```
[2024-11-06 10:00:00] 开始推理，最大tokens: 1000
[2024-11-06 10:05:00] 推理完成，生成 800 tokens
[2024-11-06 10:05:05] 验证完成，耗时: 0.5秒
[2024-11-06 10:05:05] 验证开销比例: 0.85%
[2024-11-06 10:05:06] 验证通过率: 98.5%
[2024-11-06 10:05:06] 最终判定: PASS
```

**结果文件：**
- `data/raw/exp1/qwen2.5-7b_cuda_run1.json` - 实验数据
- `logs/exp1_homogeneous_YYYYMMDD_HHMMSS.log` - 详细日志

## 5. 查看结果

```python
import json

# 加载结果
with open('data/raw/exp1/qwen2.5-7b_cuda_run1.json', 'r') as f:
    result = json.load(f)

# 关键指标
print(f"验证开销: {result['overhead']['percentage']:.2f}%")
print(f"验证通过率: {result['statistics']['summary']['accept_rate']*100:.2f}%")
print(f"判定: {result['verdict']}")
```

## 常见问题

### Q1: CUDA out of memory
**解决**：减少max_new_tokens
```python
# 在experiments/exp1_homogeneous.py中修改
inference_result = inferencer.generate_with_hidden_states(
    prompt=prompt,
    max_new_tokens=500,  # 从1000改为500
    ...
)
```

### Q2: 模型下载太慢
**解决**：使用国内镜像
```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/download_models.sh
```

### Q3: Mac上使用MPS
**修改**：在experiments/exp1_homogeneous.py中
```python
device = "mps"  # 改为mps
```

## 下一步

- 修改配置文件：`configs/experiments.yaml`
- 添加自定义提示词：`configs/prompts.yaml`
- 运行其他实验：`experiments/exp2_heterogeneous.py`

## 获取帮助

查看完整文档：`README.md`
