"""Assemble KG1 from computed nodes and edges, export as GraphML."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import yaml

from chemistry.kg1_build.compute_nodes import build_all_fg_nodes
from chemistry.kg1_build.compute_edges import build_mmp_edges, build_cooccurrence_edges

logger = logging.getLogger(__name__)


def build_kg1(fg_defs: list[dict], molecules_df: pd.DataFrame) -> nx.Graph:
    """Build the full KG1 graph from FG definitions and molecule data."""
    G = nx.Graph()

    # --- Add FG nodes ---
    nodes = build_all_fg_nodes(fg_defs)
    for node in nodes:
        node_id = node.pop("id")
        G.add_node(node_id, **_stringify_attrs(node))

    # --- Add LogD effect nodes + MMP edges ---
    mmp_edges = build_mmp_edges(molecules_df, fg_defs)
    for edge in mmp_edges:
        target = edge["target"]
        if not G.has_node(target):
            effect_content = (
                f"LogD effect node for {edge['source']}. "
                f"Average δLogD = {edge['delta_logd']:+.2f} units "
                f"from {edge['n_pairs']} matched molecular pairs."
            )
            G.add_node(target, content=effect_content, node_type="logd_effect",
                       delta_logd=str(edge["delta_logd"]))
        G.add_edge(
            edge["source"], edge["target"],
            edge_type=edge["edge_type"],
            delta_logd=str(edge["delta_logd"]),
            n_pairs=str(edge["n_pairs"]),
            content=edge["content"],
        )

    # --- Add FG co-occurrence edges ---
    cooc_edges = build_cooccurrence_edges(molecules_df, fg_defs)
    for edge in cooc_edges:
        if G.has_node(edge["source"]) and G.has_node(edge["target"]):
            G.add_edge(
                edge["source"], edge["target"],
                edge_type=edge["edge_type"],
                count=str(edge["count"]),
                content=edge["content"],
            )

    logger.info(
        "KG1 built: %d nodes, %d edges",
        G.number_of_nodes(), G.number_of_edges()
    )
    return G


def _stringify_attrs(attrs: dict[str, Any]) -> dict[str, str]:
    """GraphML requires all attributes to be string/int/float — convert booleans."""
    result = {}
    for k, v in attrs.items():
        if isinstance(v, bool):
            result[k] = str(v)
        else:
            result[k] = v
    return result


def export_graphml(G: nx.Graph, output_path: Path) -> None:
    """Export NetworkX graph to GraphML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(output_path))
    logger.info("KG1 exported to %s (%d nodes, %d edges)",
                output_path, G.number_of_nodes(), G.number_of_edges())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fg_defs = yaml.safe_load(Path("chemistry/kg1_build/fg_smarts.yaml").read_text())["functional_groups"]
    df = pd.read_csv("test_logd.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    logd_col = next((c for c in df.columns if "logd" in c.lower()), None)
    if logd_col and logd_col != "logd_exp":
        df = df.rename(columns={logd_col: "logd_exp"})
    G = build_kg1(fg_defs, df)
    export_graphml(G, Path("chemistry/kg1_build/chemistry_rule_graph.graphml"))
    print(f"KG1: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
