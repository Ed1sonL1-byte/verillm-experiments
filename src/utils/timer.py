"""计时器模块"""
import time
from typing import Dict

class Timer:
    """精确计时器"""
    def __init__(self):
        self.times: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
    
    def start(self, name: str):
        """开始计时"""
        self.start_times[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """停止计时并返回耗时（秒）"""
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' was not started")
        elapsed = time.perf_counter() - self.start_times[name]
        self.times[name] = elapsed
        return elapsed
    
    def get(self, name: str) -> float:
        """获取已记录的时间"""
        return self.times.get(name, 0.0)
    
    def get_all(self) -> Dict[str, float]:
        """获取所有计时结果"""
        return self.times.copy()
    
    def reset(self):
        """重置所有计时"""
        self.times.clear()
        self.start_times.clear()
