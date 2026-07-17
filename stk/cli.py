"""CLI 入口: 通过 'stk' 命令调用各模块."""

from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="stk",
    help="SpatioTemporalKG: 时空动态知识图谱 CLI",
    no_args_is_help=True,
)
console = Console()


# ==================== 根命令 ====================

@app.callback()
def callback():
    """时空动态知识图谱命令行工具."""
    pass


@app.command()
def version():
    """显示版本信息."""
    from stk import __version__
    console.print(f"stk v{__version__}")
    console.print("时空动态知识图谱 · SpatioTemporalKG")


@app.command()
def info():
    """显示项目配置与状态信息."""
    console.print("[bold]SpatioTemporalKG[/bold]")
    console.print("  设计文档: 时空动态知识图谱实验设计_方案A定版_v3.docx")
    console.print("  状态: 开发中")
    console.print("  模块:")
    console.print("    ontology    - 本体层")
    console.print("    scenario   - 场景层")
    console.print("    behavior   - 行为层")
    console.print("    rules      - 规则层")
    console.print("    dynamic    - 动态更新")
    console.print("    storage    - Neo4j 存储")
    console.print("    extraction - CARLA 提取")
    console.print("    pipeline   - 主流水线")
    console.print("    viz        - 可视化")


# ==================== ontology 子命令组 ====================

@app.group()
def ontology():
    """本体层 (v3 §1) 相关操作"""
    pass


@ontology.command()
def validate(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="本体配置文件路径 (默认 config/ontology.yaml)",
    ),
):
    """校验本体配置与枚举定义的一致性 (任务1.10)。

    加载 config/ontology.yaml，逐项核对实体类型、关系类型、
    命名空间前缀是否与 stk.ontology.types 中的枚举定义一致。
    """
    # 1. 定位并加载配置文件
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "ontology.yaml"
    if not config_path.exists():
        console.print(f"[red]✗ 配置文件未找到: {config_path}[/red]")
        raise typer.Exit(code=1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    console.print(f"[dim]本体配置文件: {config_path}[/dim]")
    console.print(f"[dim]加载完成: {len(cfg)} 个顶级配置项[/dim]")
    console.print()

    # 2. 引入代码中的枚举
    from stk.ontology.types import (
        BehaviorRelationType, CrossLayerRelationType,
        EntityType, RuleRelationType, SceneRelationType,
    )
    from stk.ontology.namespace import NAMESPACE_PREFIXES

    errors = []

    # 3a. 实体类型
    yaml_entities = set(cfg.get("entity_types", {}).keys())
    code_entities = set(et.name for et in EntityType)
    missing_entities = yaml_entities - code_entities
    extra_entities = code_entities - yaml_entities
    if missing_entities:
        errors.append(f"YAML 中的实体类型在代码中未定义: {missing_entities}")
    if extra_entities:
        errors.append(f"代码中的实体类型在 YAML 中未列出: {extra_entities}")
    console.print(f"  [bold]实体类型[/bold]      YAML={len(yaml_entities)}  代码={len(code_entities)}  "
                  f"{'[green]✓ 一致[/green]' if not missing_entities and not extra_entities else '[red]✗ 不一致[/red]'}")

    # 3b. 关系类型 (四类)
    relation_checks = [
        ("场景层关系",  "scene_relation_types",  SceneRelationType),
        ("行为层关系",  "behavior_relation_types", BehaviorRelationType),
        ("规则层关系",  "rule_relation_types",   RuleRelationType),
        ("跨层桥接关系", "cross_layer_relation_types", CrossLayerRelationType),
    ]
    for label, yaml_key, enum_cls in relation_checks:
        yaml_rels = set(cfg.get(yaml_key, {}).keys())
        code_rels = set(m.value for m in enum_cls)
        miss = yaml_rels - code_rels
        extra = code_rels - yaml_rels
        ok = not miss and not extra
        console.print(f"  [bold]{label}[/bold]  YAML={len(yaml_rels)}  代码={len(code_rels)}  "
                      f"{'[green]✓ 一致[/green]' if ok else '[red]✗ 不一致[/red]'}")
        if miss:
            errors.append(f"{label}: YAML 中有代码中没有的关系 — {miss}")
        if extra:
            errors.append(f"{label}: 代码中有 YAML 中没有的关系 — {extra}")

    # 3c. 命名空间前缀
    yaml_prefixes = set(cfg.get("namespace_prefixes", {}).keys())
    code_prefixes = set(NAMESPACE_PREFIXES.keys())
    prefix_ok = (yaml_prefixes == code_prefixes)
    console.print(f"  [bold]命名空间前缀[/bold]  YAML={len(yaml_prefixes)}  代码={len(code_prefixes)}  "
                  f"{'[green]✓ 一致[/green]' if prefix_ok else '[red]✗ 不一致[/red]'}")
    if not prefix_ok:
        errors.append(f"命名空间前缀差异 — YAML: {yaml_prefixes - code_prefixes}, 代码: {code_prefixes - yaml_prefixes}")

    # 4. 汇总
    console.print()
    if not errors:
        console.print("[bold green]✓ 本体校验通过，配置与枚举完全一致[/bold green]")
    else:
        for e in errors:
            console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
