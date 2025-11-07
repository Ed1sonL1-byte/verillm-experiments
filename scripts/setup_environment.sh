#!/bin/bash
echo "=== VeriLLM实验环境配置 ==="

echo "1. 创建Python虚拟环境..."
python3 -m venv venv
source venv/bin/activate || . venv/Scripts/activate

echo "2. 升级pip..."
pip install --upgrade pip

echo "3. 安装依赖包..."
pip install -r requirements.txt

echo "4. 创建目录结构..."
mkdir -p data/{raw,processed,results}/{exp1,exp2,exp3,exp4,exp5,exp6}
mkdir -p models logs

echo "5. 验证环境..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

if command -v nvidia-smi &> /dev/null; then
    echo "✅ 检测到NVIDIA GPU"
elif python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
    echo "✅ 检测到Apple Silicon (MPS)"
else
    echo "⚠️  仅CPU可用"
fi

echo "✅ 环境配置完成！"
