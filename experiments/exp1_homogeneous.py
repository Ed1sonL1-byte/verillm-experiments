"""实验1：同构硬件基线验证"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
import numpy as np
from src.utils.logger import setup_logger
from src.utils.data_saver import DataSaver
from src.models.model_loader import ModelLoader
from src.inference.inferencer import Inferencer
from src.verification.verifier import Verifier
from src.analysis.statistics import HiddenStateStatistics

def run_exp1_single(model_name, device, prompt, run_id, output_dir, logger):
    """运行单次实验1"""
    logger.info(f"=" * 60)
    logger.info(f"实验1 - 同构验证 ({device} → {device})")
    logger.info(f"模型: {model_name}, 运行: {run_id}")
    logger.info(f"=" * 60)
    
    with open("configs/experiments.yaml", 'r') as f:
        exp_config = yaml.safe_load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"加载模型到 {device}...")
    model_loader = ModelLoader()
    model, tokenizer = model_loader.load_model(model_name, device=device)
    
    inferencer = Inferencer(model, tokenizer, device, logger)
    logger.info("开始推理阶段...")
    inference_result = inferencer.generate_with_hidden_states(
        prompt=prompt, max_new_tokens=1000, temperature=0.7, 
        top_p=0.9, sample_layers_every=8
    )
    
    logger.info(f"推理完成，生成 {len(inference_result['generated_tokens'])} tokens")
    
    verifier = Verifier(model, tokenizer, device, logger)
    logger.info("开始验证阶段...")
    verification_result = verifier.verify_with_prefill(
        prompt=inference_result['prompt'],
        generated_tokens=inference_result['generated_tokens'],
        sample_layers_every=8
    )
    
    inference_time = inference_result['timing']['total_time']
    verification_time = verification_result['timing']['total_time']
    overhead_ratio = verification_time / inference_time
    
    logger.info(f"验证开销比例: {overhead_ratio*100:.2f}%")
    
    all_layer_stats = {}
    for token_idx in inference_result['hidden_states'].keys():
        for layer_idx in inference_result['hidden_states'][token_idx].keys():
            inf_hidden = inference_result['hidden_states'][token_idx][layer_idx]
            ver_hidden = verification_result['hidden_states'][token_idx][layer_idx]
            stats = HiddenStateStatistics.compute_float_diff_statistics(inf_hidden, ver_hidden)
            thresholds = exp_config['experiment']['thresholds']
            verdict = HiddenStateStatistics.check_thresholds(stats, thresholds)
            key = f"token_{token_idx}_layer_{layer_idx}"
            all_layer_stats[key] = {**stats, 'verdict': 'ACCEPT' if verdict else 'REJECT'}
    
    accept_count = sum(1 for s in all_layer_stats.values() if s['verdict'] == 'ACCEPT')
    total_count = len(all_layer_stats)
    accept_rate = accept_count / total_count
    
    logger.info(f"验证通过率: {accept_rate*100:.2f}%")
    
    result = {
        'experiment': 'exp1_homogeneous', 'model': model_name,
        'device': device, 'run_id': run_id,
        'inference': {'generated_length': len(inference_result['generated_tokens']),
                     'timing': inference_result['timing']},
        'verification': {'timing': verification_result['timing']},
        'overhead': {'ratio': overhead_ratio, 'percentage': overhead_ratio * 100},
        'statistics': {'per_layer': all_layer_stats,
                      'summary': {'accept_rate': accept_rate}},
        'verdict': 'PASS' if accept_rate > 0.95 else 'FAIL'
    }
    
    output_file = output_path / f"{model_name}_{device}_run{run_id}.json"
    DataSaver.save_json(result, str(output_file))
    logger.info(f"结果已保存到: {output_file}")
    return result

def main():
    logger = setup_logger("exp1_homogeneous")
    model_name = "qwen2.5-7b"
    device = "cuda"  # 或 "mps"
    prompt = "请详细解释区块链共识机制的演进历史，包括PoW、PoS、DPoS等主要共识算法。"
    result = run_exp1_single(model_name, device, prompt, 1, "data/raw/exp1", logger)
    logger.info(f"\n验证开销: {result['overhead']['percentage']:.2f}%")
    logger.info(f"最终判定: {result['verdict']}")

if __name__ == "__main__":
    main()
