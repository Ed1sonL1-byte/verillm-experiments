# VeriLLM实验项目

基于论文《VeriLLM: Verifiable Decentralized Inference for LLMs》的验证机制实验复现项目。

## 项目概述

本项目旨在验证VeriLLM论文中提出的LLM验证机制，通过对比推理端和验证端的Hidden States来检测推理诚实性。

### 核心实验

1. **实验1**: 同构硬件基线验证
2. **实验2**: 异构硬件兼容性测试  
3. **实验3**: 量化攻击检测（同构）
4. **实验4**: 量化攻击检测（异构）
5. **实验5**: 懒惰验证者检测
6. **实验6**: 多验证者并行验证

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置检查

```bash
# 运行环境测试
python test_setup.py
```

### 3. 下载模型

```bash
# 使用提供的脚本下载模型
bash scripts/download_models.sh
```

### 4. 运行第一个实验

```bash
# 运行实验1
python experiments/exp1_homogeneous.py
```

## 项目结构

```
verillm-experiments/
├── configs/           # 配置文件
├── src/              # 源代码
│   ├── models/       # 模型管理
│   ├── inference/    # 推理端
│   ├── verification/ # 验证端
│   ├── analysis/     # 数据分析
│   └── utils/        # 工具函数
├── experiments/      # 实验脚本
├── data/            # 实验数据
├── models/          # 模型文件
└── logs/            # 日志文件
```

## 硬件要求

### 最低配置
- GPU: NVIDIA RTX 5090 (24GB) 或 Apple M3 Max (64GB+)
- RAM: 64GB
- 存储: 200GB

### 推荐配置
- GPU: NVIDIA RTX 5090 + Apple M3 Max
- RAM: 128GB
- 存储: 500GB

## 实验流程

### 实验1示例
```python
from src.models.model_loader import ModelLoader
from src.inference.inferencer import Inferencer
from src.verification.verifier import Verifier

# 加载模型
loader = ModelLoader()
model, tokenizer = loader.load_model("qwen2.5-7b", device="cuda")

# 推理
inferencer = Inferencer(model, tokenizer, "cuda", logger)
result = inferencer.generate_with_hidden_states(prompt)

# 验证
verifier = Verifier(model, tokenizer, "cuda", logger)
verification = verifier.verify_with_prefill(prompt, result['generated_tokens'])
```

## 数据输出

每次实验会生成：
- JSON结果文件（统计数据）
- 日志文件（详细过程）
- Hidden States（可选）

## 常见问题

### 1. CUDA out of memory
- 减少batch size或序列长度
- 使用量化模型

### 2. 模型下载失败
- 检查网络连接
- 使用国内镜像源

### 3. MPS设备不可用
- 确认macOS版本 >= 14.0
- 确认安装了Xcode Command Line Tools

## 论文引用

```bibtex
@article{verillm2024,
  title={VeriLLM: Verifiable Decentralized Inference for LLMs},
  author={...},
  year={2024}
}
```

## 联系方式

如有问题，请提交Issue或联系项目维护者。

## 许可证

MIT License
