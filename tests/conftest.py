"""pytest 共享夹具与配置."""

import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import yaml


@pytest.fixture
def sample_ontology_config() -> Dict[str, Any]:
    return {
        "entity_types": {
            "VEHICLE": "Vehicle",
            "PEDESTRIAN": "Pedestrian",
        },
        "namespace_prefixes": {
            "veh_": "VehicleEntity",
            "ped_": "PedestrianEntity",
        },
        "id_formats": {
            "VehicleEntity": "veh_<actor_id>",
            "PedestrianEntity": "ped_<actor_id>",
        },
    }


@pytest.fixture
def tmp_config_file(sample_ontology_config) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(sample_ontology_config, f)
        tmp = Path(f.name)
    yield tmp
    tmp.unlink(missing_ok=True)


@pytest.fixture
def sample_rss_params() -> Dict[str, float]:
    return {
        "rho": 0.1,
        "a_max_accel": 1.5,
        "a_min_brake_long": 4.0,
        "a_brake_long": 8.0,
        "mu": 0.5,
        "a_min_brake_lat": 1.0,
        "a_brake_lat": 3.0,
    }
