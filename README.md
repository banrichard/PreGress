# PreGress:Pre-trained Subgraph Neural Network for Node-centric Graph Property Regression

-----------------
A PyTorch + torch-geometric implementation of PreGress.

Install the required packages by running
```
pip install -r requirements.txt
```
### Quick Start
Model Pre-training
```
python main_pretrain.py --dataset web-spam --dataset_name web-spam
```
Prompt Tuning
```
python non_meta/tuning.py --dataset web-spam --dataset_name web-spam
```

### Key Parameters

| name             | type   | description                                                     | 
|------------------|--------|-----------------------------------------------------------------|
| graph_net        | String | type of component GNN layer (GIN, GINE, GAT, GCN, SAGE)         |
| motif_net        | String | type of motif NN layer (GIN, GINE, GAT, GCN, SAGE, NNGINConcat) |
| num_layers       | Int    | number of GNN layers                                            |
| epochs           | Int    | number of training epochs                                       |
| batch_size       | Int    | mini-batch size for sgd                                         |
| k                | Int    | number of hops to extract ego nets                              |
| lr               | Float  | learning rate                                                   |
| weight_decay_var | Float  | Trade-off parameter for predicted mean and variance             |
| decay_factor     | Float  | Decay factor for ExponentialLR scheduler                        |
