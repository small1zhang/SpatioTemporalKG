#!/usr/bin/env python
"""从帧数据构建知识图谱 (v3 §8.6)."""
from stk.pipeline.runner import run_pipeline

if __name__ == "__main__":
    result = run_pipeline("S13", max_frames=6)
    print(f"Pipeline 完成: {len(result.get('results',[]))} 帧处理")