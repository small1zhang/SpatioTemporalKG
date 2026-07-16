# free_gpu.py 使用说明

> 一键查询服务器上的 GPU 占用情况，自动挑选空闲卡给训练用。
> 文件位置：`scripts/remote/server_scripts/free_gpu.py`

---

## 1. 快速上手

登录服务器，跑：

```bash
python3 /home/aisecurity/01_ZHB/SpatioTemporalKG/scripts/remote/server_scripts/free_gpu.py
```

输出示例：

```
====================================================================================
Idx  Name                   Temp    Util    Memory Used / Total            Status
------------------------------------------------------------------------------------
0    NVIDIA GeForce RTX 5090   40C    27%     7113 / 32607   MiB  |######..........................|  21.8%  [BUSY]
1    NVIDIA GeForce RTX 5090   43C     0%       18 / 32607   MiB  |................................|   0.1%  [FREE]
2    NVIDIA GeForce RTX 5090   45C     0%       18 / 32607   MiB  |................................|   0.1%  [FREE]
3    NVIDIA GeForce RTX 5090   33C     0%       18 / 32607   MiB  |................................|   0.1%  [FREE]
====================================================================================

[OK] 可用 GPU: [1, 2, 3] (3 张)
[*] 推荐使用 GPU 3 (空闲 31.8GiB, 33C)
```

## 2. 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mem-threshold N` | 显存阈值（MiB），超过该值则认为卡不空闲 | 2048 |
| `--util-threshold N` | 利用率阈值（%），超过该值则认为卡不空闲 | 10 |
| `--exclude 0,2` | 排除指定的卡号（逗号分隔） | （无） |
| `--brief` | 只显示简表，不显示进程详情 | off |
| `--pick-one` | 只输出一张推荐的卡号（供脚本用） | off |
| `--pick-all` | 输出所有空闲卡号，逗号分隔 | off |
| `--json` | 输出 JSON 格式 | off |

## 3. 常用场景

**训练前手动选卡：**
```bash
python3 ~/01_ZHB/SpatioTemporalKG/scripts/remote/server_scripts/free_gpu.py
CUDA_VISIBLE_DEVICES=3 python train.py
```

**一行命令自动选卡训练（最推荐）：**
```bash
CUDA_VISIBLE_DEVICES=$(python3 ~/01_ZHB/SpatioTemporalKG/scripts/remote/server_scripts/free_gpu.py --pick-one) python train.py
```

**排除已知被占用的卡：**
```bash
CUDA_VISIBLE_DEVICES=$(python3 ~/01_ZHB/SpatioTemporalKG/scripts/remote/server_scripts/free_gpu.py --pick-one --exclude 0) python train.py
```

**在 Python 训练脚本内使用：**
```python
import json, os, subprocess
result = subprocess.run(
    ["python3", "/home/aisecurity/01_ZHB/SpatioTemporalKG/scripts/remote/server_scripts/free_gpu.py", "--json"],
    capture_output=True, text=True
)
data = json.loads(result.stdout)
free_gpus = data["free_indices"]
if free_gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, free_gpus))
```

**设置别名，以后直接 `free-gpu`：**
```bash
echo 'alias free-gpu="python3 ~/01_ZHB/SpatioTemporalKG/scripts/remote/server_scripts/free_gpu.py"' >> ~/.bashrc
source ~/.bashrc
```

## 4. 判断规则

一张卡被判为 **"空闲"**，需同时满足：
1. 显存占用 < 2048 MiB（≈ 2 GiB）
2. GPU 利用率 < 10%
3. 没有计算进程（不算 Xorg 等图形进程）

## 5. 调整阈值

```bash
python3 free_gpu.py --mem-threshold 5120 --util-threshold 30
```

## 6. 输出 JSON（供监控/告警系统用）

```bash
python3 free_gpu.py --json | jq -r '.free_indices[]'
```

## 7. 文件位置

- 脚本：`/home/aisecurity/01_ZHB/SpatioTemporalKG/scripts/remote/server_scripts/free_gpu.py`
- 本说明：`/home/aisecurity/01_ZHB/SpatioTemporalKG/scripts/remote/server_scripts/README.md`
- 软链接：`/home/aisecurity/bin/free_gpu`

软链接可在任意目录直接调 `free_gpu`，但需要把 `~/bin` 加入 PATH。如果不想改 PATH，就用完整路径。
