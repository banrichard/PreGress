import networkx as nx
import torch
from torch_geometric.utils import subgraph
from torch_geometric.data import Batch



def k_hop_induced_subgraph(graph, node, k=2) -> nx.Graph:
    queue = [(node, 0)]
    node_set = set()
    edge_set = set()
    while queue:
        current_node, distance = queue.pop(0)
        node_set.add(current_node)
        if distance >= k:
            continue
        for neighbor in graph.neighbors(current_node):
            edge_set.add((current_node, neighbor))
            queue.append((neighbor, distance + 1))
    subgraph = nx.Graph()
    subgraph.add_nodes_from(node_set)
    subgraph.add_edges_from(edge_set)
    for u in subgraph.nodes():
        subgraph.nodes[u]['x'] = graph.nodes[u]['x']
    for u, v in subgraph.edges():
        subgraph[u][v]["edge_attr"] = graph[u][v]["edge_attr"]

    return subgraph

def k_hop_induced_subgraph_edge(graph, edge, k=1) -> nx.Graph:
    node_list = []
    edge_list = []
    node_u = edge[0]
    node_v = edge[1]
    node_list.append(node_u)
    node_list.append(node_v)
    for neighbor in graph.neighbors(node_u):
        node_list.append(neighbor)
        edge_list.append((node_u, neighbor))
    for neighbor in graph.neighbors(node_v):
        node_list.append(neighbor)
        edge_list.append((node_v, neighbor))
    node_list = list(set(node_list))
    edge_list = [(u, v, {"edge_attr": graph.edges[u, v]["edge_attr"]})
                 for (u, v) in edge_list]
    subgraph = nx.subgraph(graph, node_list).copy()
    remove_edge_list = [edge for edge in subgraph.edges(data=True) if edge not in edge_list]
    subgraph.remove_edges_from(remove_edge_list)
    return subgraph


def graph_refinement(graph, y):
    """
    :param graph: pyg.graph
    :param y: tensor
    :return: selected_graph, selected label
    """
    # x_dim = graph.x.shape[1]
    mask = y.gt(0).squeeze(1)
    # mask = mask.expand(-1, x_dim)
    new_graph = Batch()
    new_graph = new_graph.to(y.device)
    try:
        new_graph.x = graph.x[mask, :]
    except IndexError:
        print(graph.x.shape[0])
    # graph.x = graph.x.resize(-1, x_dim)
    new_graph.edge_index, new_graph.edge_attr = subgraph(mask, graph.edge_index, graph.edge_attr, relabel_nodes=True)
    new_graph.num_nodes = new_graph.x.shape[0]
    new_graph.batch = torch.zeros(new_graph.num_nodes).to(new_graph.x.device)
    y = y[mask]
    return new_graph, y
