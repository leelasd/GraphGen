# tests/chemistry/test_configs.py
import yaml
from pathlib import Path

CONFIGS = [
    "chemistry/configs/chemistry_atomic_config.yaml",
    "chemistry/configs/chemistry_cot_config.yaml",
    "chemistry/configs/chemistry_multihop_config.yaml",
    "chemistry/configs/chemistry_aggregated_config.yaml",
]

def test_configs_exist():
    for path in CONFIGS:
        assert Path(path).exists(), f"Missing config: {path}"

def test_configs_valid_yaml():
    for path in CONFIGS:
        data = yaml.safe_load(Path(path).read_text())
        assert "global_params" in data
        assert "nodes" in data

def test_configs_have_required_nodes():
    # All configs skip build_kg (KGs are pre-loaded into graph storage)
    required_ops = {"read", "chunk", "partition", "generate"}
    for path in CONFIGS:
        data = yaml.safe_load(Path(path).read_text())
        op_names = {n["op_name"] for n in data["nodes"]}
        assert required_ops == op_names, f"{path} missing ops: {required_ops - op_names}"

def test_configs_point_to_graphml():
    for path in CONFIGS:
        data = yaml.safe_load(Path(path).read_text())
        read_node = next(n for n in data["nodes"] if n["op_name"] == "read")
        input_paths = read_node["params"]["input_path"]
        assert any(p.endswith(".graphml") for p in input_paths), \
            f"{path}: read node should point to a .graphml file"

def test_configs_correct_graphml_routing():
    """KG1 configs must use chemistry_rule_graph.graphml; KG2 configs must use molecule_graph.graphml."""
    kg1_configs = [
        "chemistry/configs/chemistry_atomic_config.yaml",
        "chemistry/configs/chemistry_cot_config.yaml",
    ]
    kg2_configs = [
        "chemistry/configs/chemistry_multihop_config.yaml",
        "chemistry/configs/chemistry_aggregated_config.yaml",
    ]
    for path in kg1_configs:
        data = yaml.safe_load(Path(path).read_text())
        read_node = next(n for n in data["nodes"] if n["op_name"] == "read")
        input_paths = read_node["params"]["input_path"]
        assert any("chemistry_rule_graph.graphml" in p for p in input_paths), \
            f"{path}: KG1 config must point to chemistry_rule_graph.graphml"
    for path in kg2_configs:
        data = yaml.safe_load(Path(path).read_text())
        read_node = next(n for n in data["nodes"] if n["op_name"] == "read")
        input_paths = read_node["params"]["input_path"]
        assert any("molecule_graph.graphml" in p for p in input_paths), \
            f"{path}: KG2 config must point to molecule_graph.graphml"
