from itertools import product
from collections import defaultdict


class MetaConfig(object):
    def __init__(self):
        self.dataname = 'web-spam'
        self.adapt_lr = None
        self.meta_lr = None
        self.adapt_steps = None
        self.epoch = None
        self.adapt_steps_meta_test = None
        self.K_shot = None
        self.device = "cuda:0"
        self.exp_type = defaultdict(dict)
        self.para_set = None
        self.set_parameters()
        self.layer_num = 5

    def set_parameters(self):
        self.adapt_lr, self.meta_lr, self.adapt_steps, self.epoch, self.adapt_steps_meta_test, self.K_shot, self.exp_type = self.macro_pars()
        self.para_set = self.micro_pars()

    def macro_pars(self):
        adapt_lr = 0.00001
        meta_lr = 0.001
        adapt_steps = 2
        epoch = 200
        adapt_steps_meta_test = 40
        K_shot = 10
        exp_type = defaultdict(dict)

        # exp_type['global'] = {
        #     'meta_train_tasks': [4, 8],
        #     'meta_test_tasks': [16, 32]
        #
        # }
        #
        # exp_type['edge_level'] = {
        #     'meta_train_tasks': [41, 42, 43, 44],
        #     'meta_test_tasks': {
        #         'edge2edge': [80, 81],
        #         'edge2node': [39, 40]
        #     }
        # }

        exp_type['local'] = {
            'meta_train_tasks': [5, 6],
            'meta_test_tasks': [5, 6]
        }

        return adapt_lr, meta_lr, adapt_steps, epoch, adapt_steps_meta_test, K_shot, exp_type

    def micro_pars(self):
        para_set = set()
        pre_train_method = ['SimGRACE']  # , 'GraphCL', 'SimGRACE']
        with_prompt = [False]
        meta_learning = [True]
        gnn_type = ['GIN']

        pre_epoch = None

        for para_list in product(pre_train_method, with_prompt, meta_learning, gnn_type):
            pre_train_method, with_prompt, meta_learning, gnn_type = para_list
            if pre_train_method == 'None':
                with_prompt, meta_learning = False, False
                pre_train_path = None
            else:
                if pre_train_method == 'GraphCL' and gnn_type == 'GAT':
                    pre_epoch = 100
                elif pre_train_method == 'GraphCL' and gnn_type == 'GCN':
                    pre_epoch = 80
                elif pre_train_method == 'GraphCL' and gnn_type == 'TransformerConv':
                    pre_epoch = 90
                elif pre_train_method == 'SimGRACE' and gnn_type == 'GAT':
                    pre_epoch = 70
                elif pre_train_method == 'SimGRACE' and gnn_type == 'GCN':
                    pre_epoch = 100
                elif pre_train_method == 'SimGRACE' and gnn_type == 'TransformerConv':
                    pre_epoch = 90
                elif gnn_type == "GIN":
                    pre_epoch = 200
                pre_train_path = "/home/banlujie/metaCounting/saved_model/best_epoch_{:s}.pt".format(gnn_type)

            para_set.add((pre_train_method, with_prompt, meta_learning, gnn_type, pre_train_path))

        return sorted(para_set, reverse=True)


if __name__ == "__main__":
    args = MetaConfig()
    print(args.para_set)
