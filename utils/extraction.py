import networkx as nx
import re

def extract_items_from_string(input_string):
    # Use regex to find the first number
    first_number_match = re.search(r'-?\d+\.\d+|-?\d+', input_string)
    first_number = float(first_number_match.group()) if first_number_match.group().find('.') != -1 else int(
        first_number_match.group())

    # Extract the list using the previously used method
    list_match = re.search(r'\[(.*?)\]', input_string)
    extracted_list = eval(list_match.group()) if list_match else []

    # Remove the first number and the list from the string to simplify extracting the next number
    string_after_list = input_string[list_match.end():]
    next_number_match = re.search(r'-?\d+\.\d+|-?\d+', string_after_list)
    next_number = float(next_number_match.group()) if next_number_match.group().find('.') != -1 else int(
        next_number_match.group())

    # Return the four requested items
    return (first_number, extracted_list, next_number)
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
