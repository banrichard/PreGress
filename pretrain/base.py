import torch
from model.graphconv import Backbone
from torch.optim import Adam

class PreTrain(torch.nn.Module):
    def __init__(self, gnn_type='TransformerConv', dataset_name = 'QM9', hid_dim = 128, gln = 2, num_epoch=100):
        super().__init__()
        self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.num_layer = gln
        self.epochs = num_epoch
        self.hid_dim =hid_dim
       
        
    def initialize_gnn(self, input_dim, out_dim):
        if self.gnn_type == "GCN":
                self.gnn = Backbone("GCN",self.num_layer,input_dim,self.hid_dim,output_dim=out_dim)
        elif self.gnn_type == "GAT":
                self.gnn = Backbone("GAT",self.num_layer,input_dim,self.hid_dim,output_dim=out_dim)
        elif self.gnn_type == "SAGE":
                self.gnn = Backbone("SAGE",self.num_layer,input_dim,self.hid_dim,out_dim)
        elif self.gnn_type == "GIN":
                self.gnn = Backbone("GIN",self.num_layer,input_dim,self.hid_dim,out_dim)
        else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        self.gnn.to(self.device)
        self.optimizer = Adam(self.gnn.parameters(), lr=0.001, weight_decay=0.00005)

    def load_graph_data(self):
        self.input_dim, self.output_dim, _, _, _, self.graph_list= load4graph(self.dataset_name)
        
#     def load_node_data(self):
#         self.data, self.dataset = load4node(self.dataset_name, shot_num = self.shot_num)
#         self.data.to(self.device)
#         self.input_dim = self.dataset.num_features
#         self.output_dim = self.dataset.num_classes

