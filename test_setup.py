"""快速环境测试脚本"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

def test_setup():
    print("=" * 50)
    print("VeriLLM环境测试")
    print("=" * 50)
    
    print(f"\n1. Python版本: {sys.version}")
    print(f"2. PyTorch版本: {torch.__version__}")
    
    print("\n3. 可用设备:")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA可用 - {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        print(f"   ✅ MPS可用 (Apple Silicon)")
    else:
        print(f"   ⚠️  仅CPU可用")
    
    print("\n4. 测试模型加载...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        print("   ✅ 模型加载成功")
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        return False
    
    print("\n5. 测试推理...")
    try:
        inputs = tokenizer("Hello!", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        print(f"   ✅ 推理成功 - Hidden States层数: {len(outputs.hidden_states)}")
    except Exception as e:
        print(f"   ❌ 推理失败: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ 所有测试通过！")
    print("=" * 50)
    return True

if __name__ == "__main__":
    test_setup()
