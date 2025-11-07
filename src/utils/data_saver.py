"""数据保存模块"""
import json
import pickle
from pathlib import Path
from typing import Any, Dict
import numpy as np

class DataSaver:
    """数据保存工具"""
    
    @staticmethod
    def save_json(data: Dict, filepath: str):
        """保存JSON文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_json(filepath: str) -> Dict:
        """加载JSON文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_numpy(data: np.ndarray, filepath: str):
        """保存NumPy数组"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(filepath, data)
    
    @staticmethod
    def load_numpy(filepath: str) -> np.ndarray:
        """加载NumPy数组"""
        return np.load(filepath)
