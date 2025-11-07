"""统计分析模块"""
import numpy as np
from typing import Dict

class HiddenStateStatistics:
    @staticmethod
    def compute_float_diff_statistics(inf_hidden: np.ndarray, ver_hidden: np.ndarray) -> Dict:
        """计算浮点数差异统计量"""
        delta = np.abs(inf_hidden - ver_hidden)
        mean_error = np.mean(delta)
        max_error = np.max(delta)
        exact_match = np.sum(delta == 0)
        Pe = HiddenStateStatistics._compute_exponent_mismatch_rate(inf_hidden, ver_hidden)
        Pm = HiddenStateStatistics._compute_large_mantissa_ratio(delta)
        Pw = HiddenStateStatistics._compute_small_mantissa_ratio(delta)
        return {
            'mean_error': float(mean_error), 'max_error': float(max_error),
            'exact_match': int(exact_match), 'Pe': float(Pe),
            'Pm': float(Pm), 'Pw': float(Pw)
        }
    
    @staticmethod
    def _compute_exponent_mismatch_rate(arr1: np.ndarray, arr2: np.ndarray) -> float:
        arr1_f32 = arr1.astype(np.float32).flatten()
        arr2_f32 = arr2.astype(np.float32).flatten()
        arr1_bits = arr1_f32.view(np.uint32)
        arr2_bits = arr2_f32.view(np.uint32)
        exponent_mask = 0x7F800000
        arr1_exp = (arr1_bits & exponent_mask) >> 23
        arr2_exp = (arr2_bits & exponent_mask) >> 23
        mismatch = np.sum(arr1_exp != arr2_exp)
        return mismatch / arr1_exp.size
    
    @staticmethod
    def _compute_large_mantissa_ratio(delta: np.ndarray, threshold: float = 1e-5) -> float:
        return np.sum(delta > threshold) / delta.size
    
    @staticmethod
    def _compute_small_mantissa_ratio(delta: np.ndarray, threshold: float = 1e-7) -> float:
        return np.sum(delta < threshold) / delta.size
    
    @staticmethod
    def check_thresholds(stats: Dict, config: Dict) -> bool:
        Pe_pass = stats['Pe'] < config['Pe']
        Pm_pass = stats['Pm'] > config['Pm']
        Pw_pass = stats['Pw'] > config['Pw']
        mean_pass = abs(stats['mean_error']) < config['mean_epsilon']
        return Pe_pass and Pm_pass and Pw_pass and mean_pass
