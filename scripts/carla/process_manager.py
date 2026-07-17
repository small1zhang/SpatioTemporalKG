#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""process_manager.py -- CARLA 进程级隔离管理器

诊断根因:
  CARLA 0.9.16 在 server 内仍有 actor stream 时调用 client.load_world()
  会触发 UE4 SIGSEGV (Signal 11). `del client + sleep(1) + 新 client`
  并不能从根上避免.

修复策略 (用户确认):
  彻底进程级隔离. 禁止运行时 load_world() 切图. 任何想换地图的路径
  必须走: 完全 kill 当前 server 进程 -> 等端口释放 -> cold-boot 新 server
  -> 健康检查 -> client 重新连接.

Py3.13 兼容补丁:
  原 venv 用 py3.13, CARLA 0.9.16 没有 cp313 wheel, 不能 import carla.
  本模块不再依赖当前 python 能 import carla, 改为探测可用 python 解释器
  (优先级: conda stk env -> miniconda stk env -> sys.executable), 用子进程
  + RPC 探测 CARLA server 健康.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import signal
import socket
import subprocess
import time
from typing import Optional, List

DEFAULT_CARLA_SH = "/home/aisecurity/Carla/CarlaUE4.sh"
DEFAULT_LOG_DIR = "/home/aisecurity"

import shutil

# Python 解释器探测候选 (按优先级排序)
_PYTHON_CANDIDATES = [
    "/home/aisecurity/miniconda3/envs/stk/bin/python",
    "/home/aisecurity/miniconda3/bin/python3.10",
    "/usr/bin/python3.10",
    "/usr/bin/python3",
]


def _log(msg: str) -> None:
    print(f"[carla-pm] {msg}", flush=True)


def is_port_free(host: str, port: int) -> bool:
    """True if TCP port is currently free."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.bind((host, port))
        s.close()
        return True
    except OSError:
        return False


def find_carla_python() -> str:
    """找能 import carla 的 python 解释器. 优先用 sys.executable, 不行就探测候选.

    Returns: python 可执行文件绝对路径.
    Raises: RuntimeError 如果所有候选都不能 import carla.
    """
    candidates: List[str] = []
    # 第一个候选 = 当前解释器, 可能就是 stk env
    candidates.append(os.path.realpath(sys_executable()))
    # 然后 _PYTHON_CANDIDATES
    for p in _PYTHON_CANDIDATES:
        rp = os.path.realpath(p)
        if rp not in candidates and os.path.exists(rp):
            candidates.append(rp)

    for py in candidates:
        if not os.path.exists(py):
            continue
        try:
            r = subprocess.run(
                [py, "-c", "import carla"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                _log(f"found carla-capable python: {py}")
                return py
        except Exception:
            continue

    raise RuntimeError(
        "no python interpreter on this host can `import carla`. "
        f"tried: {candidates}"
    )


def sys_executable() -> str:
    import sys
    return sys.executable


def kill_carla_on_port(port: int, grace_seconds: float = 3.0) -> bool:
    """Kill any process listening on port (SIGTERM grace then SIGKILL)."""
    try:
        out = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [p for p in out.stdout.split() if p.strip().isdigit()]
    except FileNotFoundError:
        try:
            out = subprocess.run(
                ["fuser", f"{port}/tcp"],
                capture_output=True, text=True, timeout=5,
            )
            pids = [p for p in out.stdout.split() if p.strip().isdigit()]
        except Exception:
            pids = []
    except Exception:
        pids = []

    if not pids:
        return False

    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass

    t0 = time.time()
    while time.time() - t0 < grace_seconds:
        if is_port_free("127.0.0.1", port):
            break
        time.sleep(0.3)

    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass

    return True


def wait_port_released(port: int, timeout: float = 30.0) -> bool:
    """Wait until TCP port is free to bind."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if is_port_free("127.0.0.1", port):
            return True
        time.sleep(0.5)
    return False


INI_PATH = "/home/aisecurity/Carla/CarlaUE4/Config/DefaultEngine.ini"
INI_BACKUP_PATH = "/home/aisecurity/Carla/CarlaUE4/Config/DefaultEngine.ini.bak_pm"
INI_SECTION = "[/Script/EngineSettings.GameMapsSettings]"


def _get_map_package_path(town):
    return f"/Game/Carla/Maps/{town}.{town}"


def set_default_map(town):
    if not pathlib.Path(INI_PATH).exists():
        _log("INI not found: " + INI_PATH)
        return False
    if not pathlib.Path(INI_BACKUP_PATH).exists():
        shutil.copy2(INI_PATH, INI_BACKUP_PATH)
        _log("backed up INI to " + INI_BACKUP_PATH)
    target_map = _get_map_package_path(town)
    with open(INI_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("EditorStartupMap="):
            new_lines.append("EditorStartupMap=" + target_map + chr(10))
            changed = True
        elif stripped.startswith("GameDefaultMap="):
            new_lines.append("GameDefaultMap=" + target_map + chr(10))
            changed = True
        elif stripped.startswith("ServerDefaultMap="):
            new_lines.append("ServerDefaultMap=" + target_map + chr(10))
            changed = True
        else:
            new_lines.append(line)
    if not changed:
        _log("[!] ServerDefaultMap not found, appending")
        found = False
        for i, line in enumerate(new_lines):
            if line.strip() == INI_SECTION:
                new_lines.insert(i + 1, "ServerDefaultMap=" + target_map + chr(10))
                found = True
                break
        if not found:
            new_lines.append(INI_SECTION + chr(10))
            new_lines.append("ServerDefaultMap=" + target_map + chr(10))
    target_map = _get_map_package_path(town)
    with open(INI_PATH, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    _log("set default map to " + town + " (" + target_map + ")")
    return True


def restore_default_map():
    if pathlib.Path(INI_BACKUP_PATH).exists():
        shutil.copy2(INI_BACKUP_PATH, INI_PATH)
        pathlib.Path(INI_BACKUP_PATH).unlink()
        _log("restored INI from backup")
        return True
    _log("no backup to restore")
    return False


def start_carla_with_map(port, town, carla_sh=None, log_path=None,
                          quality_level="Low", graphics_adapter=None,
                          extra_args=None):
    """Edit INI to set ServerDefaultMap=town, then cold-boot CARLA.
    IMPORTANT: INI backup remains in place. Caller MUST call restore_default_map()
    AFTER health check confirms CARLA has loaded the new map (CARLA reads the
    INI lazily during its startup load, so we cannot restore immediately).
    Returns the spawned CARLA pid on success, or False on INI failure.
    """
    if carla_sh is None:
        carla_sh = DEFAULT_CARLA_SH
    if not set_default_map(town):
        return False
    pid = start_carla(port, carla_sh=carla_sh, log_path=log_path,
                     quality_level=quality_level,
                     graphics_adapter=graphics_adapter,
                     extra_args=extra_args)
    return pid


def restart_carla_with_map(port, town, host="localhost", carla_sh=None,
                            log_path=None, cold_boot_timeout=120.0,
                            graphics_adapter=None, quality_level="Low",
                            extra_args=None, python=None):
    """Cold-boot CARLA server that directly loads `town` (no load_world()).
    Steps:
      1. set_default_map(town)  -- INI backup created
      2. start_carla() spawning subprocess
      3. health_check() wait until server ready AND loaded map == town
      4. restore_default_map()  -- INI reverted (only after CARLA already read it)
    Returns True on success. On failure also restores INI.
    """
    if not wait_port_released(port, timeout=30.0):
        _log("port %d not free, killing first" % port)
        kill_carla_on_port(port, grace_seconds=3.0)
        if not wait_port_released(port, timeout=30.0):
            _log("port %d still not free, abort" % port)
            restore_default_map()
            return False

    pid_or_fail = start_carla_with_map(port, town, carla_sh=carla_sh,
                                        log_path=log_path,
                                        quality_level=quality_level,
                                        graphics_adapter=graphics_adapter,
                                        extra_args=extra_args)
    if not pid_or_fail:
        restore_default_map()
        return False

    # Wait health; also probe that the loaded map is the target town.
    target = 'Carla/Maps/' + town
    t0 = time.time()
    last_err = ""
    if python is None:
        try:
            python = find_carla_python()
        except RuntimeError as e:
            _log(str(e))
            restore_default_map()
            return False
    script = (
        "import sys, time\n"
        "import carla\n"
        "c = carla.Client(sys.argv[1], int(sys.argv[2]))\n"
        "c.set_timeout(10.0)\n"
        "try:\n"
        "    ver = c.get_server_version()\n"
        "    w = c.get_world()\n"
        "    m = w.get_map().name\n"
        "    print(f'OK {ver} {m}')\n"
        "    sys.exit(0)\n"
        "except Exception as e:\n"
        "    print(f'ERR {type(e).__name__}: {e}')\n"
        "    sys.exit(1)\n"
    )
    while time.time() - t0 < cold_boot_timeout:
        try:
            r = subprocess.run(
                [python, "-c", script, host, str(port)],
                capture_output=True, text=True, timeout=15,
            )
            out = (r.stdout or "").strip()
            if r.returncode == 0 and out.startswith("OK "):
                # Verify map name matches target
                loaded_map = out[3:].split(' ', 1)[1] if ' ' in out[3:] else ''
                if loaded_map == target:
                    _log(f"health OK: {out[3:]} (matches {town})")
                    restore_default_map()
                    return True
                else:
                    _log(f"server up but map={loaded_map} != {target}; still booting?")
                    last_err = f"map mismatch: {loaded_map} vs {target}"
            else:
                last_err = out or f"rc={r.returncode}"
        except subprocess.TimeoutExpired:
            last_err = "subprocess timeout"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(3)
    _log(f"restart_carla_with_map FAILED after {cold_boot_timeout:.0f}s: {last_err}")
    restore_default_map()
    return False

def start_carla(port, carla_sh=None, log_path=None, quality_level="Low",
                graphics_adapter=None, extra_args=None):
    """Cold-boot a brand new CARLA server on port.
    Returns the PID of the spawned CARLA process.
    """
    if carla_sh is None:
        carla_sh = DEFAULT_CARLA_SH
    cmd = [carla_sh, "-RenderOffScreen", "-nosound",
           f"-carla-rpc-port={port}",
           f"-quality-level={quality_level}"]
    if graphics_adapter is not None:
        cmd.append(f"-graphics-adapter={graphics_adapter}")
    if extra_args:
        cmd.extend(extra_args)

    if log_path is None:
        log_path = f"{DEFAULT_LOG_DIR}/carla_pm_{port}.log"
    log_f = open(log_path, "w")
    env = os.environ.copy()
    if graphics_adapter is not None and "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = str(graphics_adapter)

    proc = subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
    _log(f"cold-boot CARLA pid={proc.pid} port={port} log={log_path}")
    return proc.pid


def health_check(host, port, timeout=90.0, python=None):
    """Probe CARLA server is fully ready.
    Uses a fresh carla.Client each iteration via subprocess so that this
    module does NOT require the current python interpreter to be able to
    `import carla` (Py3.13 vs cp310 wheel mismatch).
    """
    if python is None:
        try:
            python = find_carla_python()
        except RuntimeError as e:
            _log(str(e))
            return False

    # Probe script: try connect, get_server_version + get_world + get_map.
    # On success print "OK <ver> <map>"; on failure print "ERR <msg>".
    script = (
        "import sys, time\n"
        "import carla\n"
        "c = carla.Client(sys.argv[1], int(sys.argv[2]))\n"
        "c.set_timeout(10.0)\n"
        "try:\n"
        "    ver = c.get_server_version()\n"
        "    w = c.get_world()\n"
        "    m = w.get_map().name\n"
        "    print(f'OK {ver} {m}')\n"
        "    sys.exit(0)\n"
        "except Exception as e:\n"
        "    print(f'ERR {type(e).__name__}: {e}')\n"
        "    sys.exit(1)\n"
    )
    t0 = time.time()
    last_err = ""
    while time.time() - t0 < timeout:
        try:
            r = subprocess.run(
                [python, "-c", script, host, str(port)],
                capture_output=True, text=True, timeout=15,
            )
            out = (r.stdout or "").strip()
            if r.returncode == 0 and out.startswith("OK "):
                _log(f"health OK: {out[3:]}")
                return True
            last_err = out or f"rc={r.returncode} stderr={r.stderr.strip()[:200]}"
        except subprocess.TimeoutExpired:
            last_err = "subprocess timeout"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(2)
    _log(f"health check FAILED after {timeout:.0f}s: {last_err}")
    return False


def restart_carla_on_port(port, host="localhost", carla_sh=None, log_path=None,
                          cold_boot_timeout=90.0, keep_existing=False,
                          graphics_adapter=None, quality_level="Low",
                          extra_args=None, python=None):
    """High-level helper: kill -> wait port release -> cold-boot -> health-check.
    Returns True on success.
    if `keep_existing=True` and a healthy CARLA is already on port, skip restart.
    """
    if keep_existing and health_check(host, port, timeout=5.0, python=python):
        _log(f"reuse healthy CARLA on {port}")
        return True

    killed = kill_carla_on_port(port, grace_seconds=3.0)
    _log(f"killed existing on {port}: {killed}")

    if not wait_port_released(port, timeout=30.0):
        _log(f"port {port} not released after 30s, abort")
        return False
    _log(f"port {port} is free")

    start_carla(port, carla_sh=carla_sh, log_path=log_path,
                quality_level=quality_level, graphics_adapter=graphics_adapter,
                extra_args=extra_args)

    if health_check(host, port, timeout=cold_boot_timeout, python=python):
        return True
    return False


def connect_fresh(host, port, timeout_s=30.0):
    """Build a fresh carla.Client and return it.
    Caller must ensure CARLA healthy on port. Requires current python to
    have carla installed; otherwise use the subprocess-based health_check
    + spawn run_phases_1_5.py from the correct interpreter.
    """
    import carla
    c = carla.Client(host, port)
    c.set_timeout(timeout_s)
    return c


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--action", choices=["kill", "start", "restart", "health", "find-py"], required=True)
    p.add_argument("--graphics-adapter", type=int, default=None)
    args = p.parse_args()

    if args.action == "kill":
        killed = kill_carla_on_port(args.port)
        print(f"killed_existing={killed}")
    elif args.action == "start":
        start_carla(args.port, graphics_adapter=args.graphics_adapter)
    elif args.action == "restart":
        ok = restart_carla_on_port(args.port, graphics_adapter=args.graphics_adapter)
        print(f"ok={ok}")
    elif args.action == "health":
        ok = health_check("localhost", args.port, timeout=10.0)
        print(f"ok={ok}")
    elif args.action == "find-py":
        py = find_carla_python()
        print(f"carla_python={py}")
