from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Union

from graphgen.bases.base_reader import BaseReader
from graphgen.utils import logger

if TYPE_CHECKING:
    import ray
    from ray.data import Dataset


class GraphmlReader(BaseReader):
    """
    Reader for GraphML files that extracts nodes and edges as text documents.

    Uses Ray Data for distributed processing.
    """

    def __init__(self, *, text_column: str = "content", include_edges: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.text_column = text_column
        self.include_edges = include_edges

    def read(
        self,
        input_path: Union[str, List[str]],
    ) -> "Dataset":
        """
        Read GraphML file(s) using Ray Data.

        :param input_path: Path to GraphML file or list of GraphML files.
        :return: Ray Dataset containing extracted node/edge documents.
        """
        import ray

        if not ray.is_initialized():
            ray.init()

        if isinstance(input_path, str):
            input_path = [input_path]

        paths_ds = ray.data.from_items(input_path)

        include_edges = self.include_edges

        def process_graphml(row: Dict[str, Any]) -> List[Dict[str, Any]]:
            try:
                file_path = row["item"]
                return self._parse_graphml_file(Path(file_path), include_edges=include_edges)
            except Exception as e:
                logger.error(
                    "Failed to process GraphML file %s: %s", row.get("item", "unknown"), e
                )
                return []

        docs_ds = paths_ds.flat_map(process_graphml)
        docs_ds = docs_ds.filter(self._should_keep_item)
        return docs_ds

    @staticmethod
    def _read_graphml_nodes_only(file_path: Path):
        """Parse GraphML with iterparse, extracting only nodes (skips edges for large graphs)."""
        import xml.etree.ElementTree as ET
        import networkx as nx

        ns = "http://graphml.graphdrawing.org/graphml"
        key_map: dict = {}  # id -> (attr_name, attr_type, default)
        graph = nx.Graph()

        for event, elem in ET.iterparse(str(file_path), events=("start", "end")):
            if event == "end":
                tag = elem.tag.replace(f"{{{ns}}}", "")
                if tag == "key":
                    kid = elem.get("id")
                    aname = elem.get("attr.name", kid)
                    atype = elem.get("attr.type", "string")
                    default_el = elem.find(f"{{{ns}}}default")
                    default = default_el.text if default_el is not None else None
                    key_map[kid] = (aname, atype, default)
                    elem.clear()
                elif tag == "node":
                    node_id = elem.get("id")
                    attrs: dict = {}
                    for d in elem.findall(f"{{{ns}}}data"):
                        kid = d.get("key")
                        if kid in key_map:
                            aname, atype, _ = key_map[kid]
                            val = d.text or ""
                            if atype == "int":
                                try:
                                    val = int(val)
                                except (ValueError, TypeError):
                                    pass
                            elif atype in ("double", "float"):
                                try:
                                    val = float(val)
                                except (ValueError, TypeError):
                                    pass
                            elif atype == "boolean":
                                val = val.lower() == "true"
                            attrs[aname] = val
                    if node_id:
                        graph.add_node(node_id, **attrs)
                    elem.clear()
                elif tag == "edge":
                    elem.clear()  # skip edges entirely
        return graph

    def _parse_graphml_file(self, file_path: Path, include_edges: bool = True) -> List[Dict[str, Any]]:
        """
        Parse a GraphML file and extract node/edge documents.

        :param file_path: Path to GraphML file.
        :return: List of document dictionaries.
        """
        import networkx as nx

        if not file_path.is_file():
            raise FileNotFoundError(f"GraphML file not found: {file_path}")

        try:
            if include_edges:
                graph = nx.read_graphml(str(file_path))
            else:
                # nodes-only: parse via iterparse to avoid loading millions of edges
                graph = self._read_graphml_nodes_only(file_path)
        except Exception as e:
            raise ValueError(f"Cannot parse GraphML file {file_path}: {e}") from e

        docs: List[Dict[str, Any]] = []

        # Represent each node as a document
        for node_id, attrs in graph.nodes(data=True):
            parts = [f"Node: {node_id}"]
            for k, v in attrs.items():
                parts.append(f"{k}: {v}")
            doc = {
                "id": str(node_id),
                "type": "text",
                self.text_column: " | ".join(parts),
                "properties": attrs,
                "path": str(file_path),
            }
            docs.append(doc)

        # Represent each edge as a document
        if not include_edges:
            logger.info("GraphML file %s: skipping edge documents (include_edges=False)", file_path)
        for src, dst, attrs in (graph.edges(data=True) if include_edges else []):
            parts = [f"Edge: {src} -> {dst}"]
            for k, v in attrs.items():
                parts.append(f"{k}: {v}")
            doc = {
                "id": f"{src}__{dst}",
                "type": "text",
                self.text_column: " | ".join(parts),
                "properties": {"source": src, "target": dst, **attrs},
                "path": str(file_path),
            }
            docs.append(doc)

        logger.info(
            "GraphML file %s: %d nodes, %d edges -> %d documents",
            file_path,
            graph.number_of_nodes(),
            graph.number_of_edges(),
            len(docs),
        )

        if not docs:
            logger.warning("GraphML file %s contains no nodes or edges.", file_path)

        return docs
