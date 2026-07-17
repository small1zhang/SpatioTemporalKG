"""配置加载：从 config/ 目录加载 YAML 配置文件."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(name: str, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """加载 config/ 下的 YAML 配置文件.

    Args:
        name: 配置文件名 (如 'ontology.yaml')
        base_dir: 项目根目录, 默认为当前文件向上找 2 级

    Returns:
        解析后的配置字典
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "config" / name
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
