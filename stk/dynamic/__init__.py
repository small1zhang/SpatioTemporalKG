"""动态更新: 三层语义的增量演化机制 (§5)."""
from .diff import DeltaGraph, DiffSet, compute_delta
from .incremental_updater import IncrementalEngine
from .snapshot_store import SnapshotStore
from .event_injector import inject_violation
from .version import VersionManager, AttrVersion
from .time_window import TimeWindowAggregator, SummaryEvent

__all__ = [
    "DeltaGraph", "DiffSet", "compute_delta",
    "IncrementalEngine",
    "SnapshotStore",
    "inject_violation",
    "VersionManager", "AttrVersion",
    "TimeWindowAggregator", "SummaryEvent",
]