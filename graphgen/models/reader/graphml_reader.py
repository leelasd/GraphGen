import networkx as nx
from graphgen.bases.base_reader import BaseReader


class GraphmlReader(BaseReader):
    def read(self, file_path: str):
        """
        Read GraphML file and return empty content list since graph will be loaded separately
        """
        # Validate that the file is a valid GraphML
        try:
            graph = nx.read_graphml(file_path)
            # Return minimal content to satisfy the pipeline
            return [{"content": f"GraphML file with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"}]
        except Exception as e:
            raise ValueError(f"Invalid GraphML file: {e}")
