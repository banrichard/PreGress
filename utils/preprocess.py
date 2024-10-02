import os


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


# Set the directory paths
pattern_graph_dir = '/mnt/data/banlujie/dataset/gowalla/query_graph/7voc'
label_dir = '/mnt/data/banlujie/dataset/gowalla/label/7voc'

# Call the function to rename files and their corresponding labels
rename_files_with_labels(pattern_graph_dir, label_dir)
