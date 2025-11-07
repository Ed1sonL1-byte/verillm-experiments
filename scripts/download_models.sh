#!/bin/bash
echo "=== 下载VeriLLM实验模型 ==="

echo "注意：这将下载约60GB的模型文件，确保有足够的磁盘空间和网络带宽。"
read -p "继续下载？ (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 1
fi

echo "1. 下载Qwen2.5-7B..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('下载Qwen2.5-7B-Instruct...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', cache_dir='./models')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', cache_dir='./models')
print('✅ Qwen2.5-7B下载完成')
"

echo "2. 下载量化版本（可选）..."
read -p "是否下载Qwen2.5-7B-AWQ量化版本？ (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('下载Qwen2.5-7B-Instruct-AWQ...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct-AWQ', cache_dir='./models')
print('✅ 量化版本下载完成')
"
fi

echo "✅ 模型下载完成！"
echo "模型位置: ./models/"
