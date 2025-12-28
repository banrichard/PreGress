import os

import torch

from data.data_load import centrality_cal
from torch_geometric.utils import negative_sampling, k_hop_subgraph, to_networkx

from data.dataset import MyReddit2, MyFlickr


def ogb_to_subgraphs(name):
    dataset = MyFlickr(root="/mnt/data/banlujie/dataset/flickr")

    # split_idx = dataset.get_idx_split()
    # train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    data = dataset[0]  # pyg graph object    return data
    graph = to_networkx(data, to_undirected=True)
    with open("/mnt/data/banlujie/flickr.txt", "w") as f:
        f.write(f"{graph.number_of_nodes()} {graph.number_of_edges()}\n")
        for edge in sorted(graph.edges):
            f.write(f"{edge[0]} {edge[1]}\n")


def get_negative_edges(pos_edge_index, num_nodes, num_neg_samples):
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples,
        method='normal'
    )
    return neg_edge_index


def get_backbone_name(file_path):
    file_name = os.path.basename(file_path)  # Returns 'best_epoch_GIN.pt'

    # Split the file name to isolate the backbone name
    backbone_name = file_name.split('_')[-1].split('.')[0]  # Splits by '_' and then by '.'

    return backbone_name  # Output: 'GIN'


def rename_files_with_labels(pattern_graph_dir, label_dir):
    # Extract the number of nodes from the pattern graph directory name
    directory_name = os.path.basename(pattern_graph_dir)
    number_of_nodes = ''.join(filter(str.isdigit, directory_name))

    unique_id = 1  # Start ID from 1
    # Iterate over all files in the pattern graph directory
    for filename in os.listdir(pattern_graph_dir):
        if filename.endswith(".txt"):
            # Open the pattern graph file to read the last line
            with open(os.path.join(pattern_graph_dir, filename), 'r') as file:
                lines = file.readlines()
                # Extract the orbit ID from the last line
                last_line = lines[-1].strip()
                orbit_id = last_line.split()[-1]  # The last number in the last line

            # Construct the new filename for the pattern graph file
            new_filename = f"{number_of_nodes}_{orbit_id}_{unique_id}.txt"

            # Rename the pattern graph file
            os.rename(os.path.join(pattern_graph_dir, filename), os.path.join(pattern_graph_dir, new_filename))
            print(f"Renamed {filename} to {new_filename} in pattern graph directory")

            # Find the corresponding label file in the label directory
            if os.path.exists(os.path.join(label_dir, filename)):
                label_file_path = os.path.join(label_dir, filename)
            else:
                continue
            if os.path.exists(label_file_path):
                # Rename the corresponding label file
                os.rename(label_file_path, os.path.join(label_dir, new_filename))
                print(f"Renamed {filename} to {new_filename} in label directory")
            else:
                print(f"Label file for {filename} not found in {label_dir}")

            # Increment the unique ID for the next file
            unique_id += 1


if __name__ == "__main__":
    rename_files_with_labels("/mnt/data/banlujie/dataset/wiki/chameleon/query_graph/5voc","/mnt/data/banlujie/dataset/wiki/chameleon/label/5voc")
