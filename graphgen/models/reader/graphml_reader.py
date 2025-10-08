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

    def __init__(self, *, text_column: str = "content", **kwargs):
        super().__init__(**kwargs)
        self.text_column = text_column

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

        def process_graphml(row: Dict[str, Any]) -> List[Dict[str, Any]]:
            try:
                file_path = row["item"]
                return self._parse_graphml_file(Path(file_path))
            except Exception as e:
                logger.error(
                    "Failed to process GraphML file %s: %s", row.get("item", "unknown"), e
                )
                return []

        docs_ds = paths_ds.flat_map(process_graphml)
        docs_ds = docs_ds.filter(self._should_keep_item)
        return docs_ds

    def _parse_graphml_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Parse a GraphML file and extract node/edge documents.

        :param file_path: Path to GraphML file.
        :return: List of document dictionaries.
        """
        import networkx as nx

        if not file_path.is_file():
            raise FileNotFoundError(f"GraphML file not found: {file_path}")

        try:
            graph = nx.read_graphml(str(file_path))
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
                self.text_column: " | ".join(parts),
                "properties": attrs,
                "path": str(file_path),
            }
            docs.append(doc)

        # Represent each edge as a document
        for src, dst, attrs in graph.edges(data=True):
            parts = [f"Edge: {src} -> {dst}"]
            for k, v in attrs.items():
                parts.append(f"{k}: {v}")
            doc = {
                "id": f"{src}__{dst}",
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
