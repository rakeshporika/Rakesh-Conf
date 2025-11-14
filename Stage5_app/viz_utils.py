from __future__ import annotations
from typing import Iterable, Optional, Set, Tuple
import pandas as pd
from pyvis.network import Network
import networkx as nx

# Colors
C_BASE = "#93c5fd"   # blue-ish
C_HUB = "#60a5fa"    # stronger blue
C_ADDED = "#22c55e"  # green
C_REMOVED = "#ef4444"# red



def build_graph_html(
    tag: str,
    modules: pd.DataFrame,
    edges: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    added_nodes: Optional[Set[str]] = None,
    removed_nodes: Optional[Set[str]] = None,
    added_edges: Optional[Set[Tuple[str,str]]] = None,
    removed_edges: Optional[Set[Tuple[str,str]]] = None,
    height: str = "700px",
    width: str = "100%",
) -> str:
    """Return PyVis HTML for a per-tag dependency graph.

    Node size ∝ fan_in + fan_out (bounded). Hub nodes slightly darker.
    Added/removed highlights (if provided) are colored accordingly.
    """
    m = modules.copy()
    e = edges.copy()
    x = metrics.copy()

    # Slice to the tag
    m = m[m["tag"] == tag]
    e = e[e["tag"] == tag]
    x = x[x["tag"] == tag]

    # Build NetworkX graph
    G = nx.DiGraph()
    G.add_nodes_from(m["module"].tolist())
    G.add_edges_from(e[["src_module", "dst_module"]].itertuples(index=False, name=None))

    # Compute sizes
    deg_out = dict(G.out_degree())
    deg_in = dict(G.in_degree())
    size = {n: deg_in.get(n, 0) + deg_out.get(n, 0) for n in G.nodes()}
    max_size = max(size.values()) if size else 1

    # Map metrics
    x = x.set_index("module")[
        ["fan_in", "fan_out", "cyclomatic", "centrality_degree"]
    ].to_dict(orient="index") if not x.empty else {}

    # PyVis setup
    nt = Network(height=height, width=width, bgcolor="#ffffff", font_color="#111827", directed=True)
    nt.barnes_hut()

    for n in G.nodes():
        s = 15 + (25 * size.get(n, 0) / max_size)  # 15..40
        hub = size.get(n, 0) >= max(2, 0.2 * max_size)
        color = C_HUB if hub else C_BASE
        if added_nodes and n in added_nodes:
            color = C_ADDED
        if removed_nodes and n in removed_nodes:
            color = C_REMOVED
        meta = x.get(n, {"fan_in": 0, "fan_out": 0, "cyclomatic": 0, "centrality_degree": 0.0})
        title = (
            f"<b>{n}</b><br/>"
            f"fan_in={meta['fan_in']}, fan_out={meta['fan_out']}<br/>"
            f"cyclomatic={meta['cyclomatic']}, centrality={meta['centrality_degree']:.3f}"
        )
        nt.add_node(n, label=n.rsplit('.', 1)[-1], title=title, color=color, size=s)

    def edge_color(s, d):
        if added_edges and (s, d) in added_edges:
            return C_ADDED
        if removed_edges and (s, d) in removed_edges:
            return C_REMOVED
        return "#94a3b8"  # slate-300

    for s, d in G.edges():
        nt.add_edge(s, d, color=edge_color(s, d))

    return nt.generate_html(name="graph.html")