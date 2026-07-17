"""阶段 0 烟雾测试：验证项目骨架可导入、可运行."""

import pytest


def test_import_stk():
    """主包可以导入."""
    import stk
    assert hasattr(stk, "__version__")


def test_version_string():
    from stk import __version__
    assert isinstance(__version__, str)
    assert __version__ == "0.0.1"


def test_import_submodules():
    """所有子模块可以导入."""
    import stk.ontology
    import stk.scenario
    import stk.behavior
    import stk.rules
    import stk.rules.rss
    import stk.rules.traffic
    import stk.dynamic
    import stk.storage
    import stk.extraction
    import stk.pipeline
    import stk.viz


def test_load_ontology_config():
    """ontology.yaml 可加载."""
    from stk.config import load_config
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent
    cfg = load_config("ontology.yaml", base_dir=base)
    assert cfg["entity_types"]["VEHICLE"] == "Vehicle"
    assert len(cfg["entity_types"]) >= 14


def test_load_rss_config():
    """rss_rules.yaml 可加载."""
    from stk.config import load_config
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent
    cfg = load_config("rss_rules.yaml", base_dir=base)
    assert cfg["rho"] == 0.1
    assert "R13a" in cfg["rules"]


def test_load_traffic_config():
    """traffic_rules.yaml 可加载."""
    from stk.config import load_config
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent
    cfg = load_config("traffic_rules.yaml", base_dir=base)
    assert "R1" in cfg["rules"]
    assert "R18" in cfg["rules"]
    assert len(cfg["rules"]) >= 17


def test_load_neo4j_config():
    """neo4j.yaml 可加载."""
    from stk.config import load_config
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent
    cfg = load_config("neo4j.yaml", base_dir=base)
    assert cfg["uri"] == "bolt://localhost:7687"


def test_load_pipeline_config():
    """pipeline.yaml 可加载."""
    from stk.config import load_config
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent
    cfg = load_config("pipeline.yaml", base_dir=base)
    assert cfg["stages"]["extraction"] is True
