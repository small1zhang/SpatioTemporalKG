#!/usr/bin/env python3
"""fix_nan.py -- 修复 graph_*.json 中 JSON 非法的 NaN/Infinity 值

问题原因：
- Python 的 json.load 默认支持 NaN / Infinity（通过 parse_constant）
- 浏览器的 JSON.parse 不支持这些值，会抛出 SyntaxError

修复方式：
- NaN → null（丢失位置信息，但不会崩溃）
- Infinity → 极大值，-Infinity → 极小值（用于数值字段）
"""
from __future__ import annotations
import json, re
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_TOWNS = [
    "Town01_20min", "Town02_20min", "Town04_20min",
    "Town05_20min", "Town10HD_20min",
]
_VIZ = _REPO / "viz_output"


def fix_json_string(txt: str) -> str:
    """替换所有独立的 NaN/Infinity/-Infinity 为 JSON 合法值"""
    # 用正则匹配独立单词边界
    txt = re.sub(r'(?<!["\w])NaN(?![\w"])', 'null', txt)
    txt = re.sub(r'(?<!["\w])Infinity(?![\w"])', '1e308', txt)
    txt = re.sub(r'(?<!["\w])-Infinity(?![\w"])', '-1e308', txt)
    return txt


def fix_shard_file(path: Path) -> dict:
    """修复单个 shard 文件，返回统计信息"""
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()

    # 统计原始问题数量
    nan_count = len(re.findall(r'(?<!["\w])NaN(?![\w"])', raw))
    inf_count = len(re.findall(r'(?<!["\w])Infinity(?![\w"])', raw))
    ninf_count = len(re.findall(r'(?<!["\w])-Infinity(?![\w"])', raw))

    if nan_count == 0 and inf_count == 0 and ninf_count == 0:
        return {"ok": True, "nan": 0, "inf": 0, "ninf": 0}

    fixed = fix_json_string(raw)

    # 验证修复后能解析
    try:
        data = json.loads(fixed)
        # 写回
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        return {"ok": True, "nan": nan_count, "inf": inf_count, "ninf": ninf_count}
    except json.JSONDecodeError as e:
        return {"ok": False, "nan": nan_count, "inf": inf_count, "ninf": ninf_count, "error": str(e)}


def main():
    for town in _TOWNS:
        town_dir = _VIZ / town
        shards = sorted(town_dir.glob("graph_*_*.json"))
        if not shards:
            print(f"[skip] {town}: 无 shard 文件")
            continue
        print(f"[*] {town} — {len(shards)} 个 shard")
        for shard_path in shards:
            result = fix_shard_file(shard_path)
            if result["ok"]:
                if result["nan"] or result["inf"] or result["ninf"]:
                    print(f"  [{shard_path.name}] NaN={result['nan']} Inf={result['inf']+result['ninf']} → 已修复")
                else:
                    print(f"  [{shard_path.name}] OK (无需修复)")
            else:
                print(f"  [{shard_path.name}] FAIL: {result['error']}")


if __name__ == "__main__":
    main()
