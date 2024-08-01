from copy import deepcopy

import pandas as pd
from torch_geometric.graphgym import optim
from torch_geometric.loader import DataLoader

from data.data_load import load4graph, meta_dataset_load, meta_motif_load
from pretrain.base import PreTrain
import torch
from random import shuffle
from sklearn.metrics import mean_squared_error
from meta_pipeline import model_components


def meta_test_adam(
        dataname,
        K_shot,
        seed,
        maml,
        adapt_steps_meta_test,
        lossfn,
        save_project_head=False,
        save_pickles=None, meta_test_task_id_list=None, device="cpu"):
    pre_train_method, with_prompt, meta_learning, gnn_type = save_pickles
    graph = meta_dataset_load(dataname)
    graph = graph.to(device)
    task_results = []
    # meta-testing
    if len(meta_test_task_id_list) < 2:
        raise AttributeError("\ttask_id_list should contain at leat two tasks!")

    shuffle(meta_test_task_id_list)

    task_pairs = [(meta_test_task_id_list[i], meta_test_task_id_list[i + 1]) for i in
                  range(0, len(meta_test_task_id_list) - 1, 2)]

    for task_1, task_2, support, query, label in meta_motif_load(dataname, task_pairs=task_pairs):
        support = support.to(device)
        query = query.to(device)
        test_model = deepcopy(maml.module)
        test_opi = optim.Adam(filter(lambda p: p.requires_grad, test_model.parameters()),
                              lr=0.001,
                              weight_decay=0.00001)

        test_model.train()
        train_loader = DataLoader(support, batch_size=1, shuffle=True)
        for _ in range(adapt_steps_meta_test):
            for i, data in enumerate(train_loader):
                support_preds = test_model(graph, data)
                support_loss = lossfn(support_preds.to(torch.float), data.y.to(torch.float))
                if _ % 5 == 0:
                    print('{}/{} training loss: {:.8f}'.format(_,
                                                               adapt_steps_meta_test,
                                                               support_loss.item()))
            test_opi.zero_grad()
            support_loss.backward()
            test_opi.step()

        test_model.eval()
        query_preds = test_model(graph, query)
        query_loss = lossfn(query_preds.to(torch.float), torch.FloatTensor(query.y).reshape(-1, 1).to(device))
        # mse_loss = mean_squared_error(query.y, query_preds)
        print("""\t MSE Loss: {:.4} """.format(query_loss))
        task_results.append([query_loss])
        if save_project_head:
            torch.save(test_model.project_head.state_dict(),
                       "/home/banlujie/saved_model/projection_head/{}.{}.{}.pth".format(dataname, pre_train_method,
                                                                                        gnn_type))
            print("project head saved! @./projection_head/{}.{}.{}.pth".format(dataname, pre_train_method, gnn_type))

    return task_results


def meta_train_maml(epoch, maml, lossfn, opt, dataname, adapt_steps, K_shot=10, meta_train_task_id_list=None,
                    device="cpu"):
    # meta-training
    graph = meta_dataset_load(dataname)
    graph = graph.to(device)
    shuffle(meta_train_task_id_list)
    task_pairs = [(meta_train_task_id_list[i], meta_train_task_id_list[i + 1]) for i in
                  range(0, len(meta_train_task_id_list) - 1, 2)]
    for ep in range(epoch):
        meta_train_loss = 0.0
        pair_count = 0
        # for local counting, load label from external .txt file
        for task1, task2, support, query, total_num, label in meta_motif_load(dataname, task_pairs=task_pairs):
            support = support.to(device)
            query = query.to(device)
            pair_count = pair_count + 1

            learner = maml.clone()
            train_loader = DataLoader(support, batch_size=1, shuffle=True)
            for _ in range(adapt_steps):  # adaptation_steps
                for i, data in enumerate(train_loader):
                    support_preds = learner(graph, data)
                    support_loss = lossfn(support_preds.to(torch.float), data.y.to(torch.float))
                    learner.adapt(support_loss)

            query_preds = learner(graph, query)
            query_loss = lossfn(query_preds.to(torch.float), query.y)
            meta_train_loss += query_loss

        print('\tmeta_train_loss at epoch {}/{}: {}'.format(ep, epoch, meta_train_loss.item()))
        meta_train_loss = meta_train_loss / pair_count
        opt.zero_grad()
        meta_train_loss.backward()
        opt.step()


if __name__ == '__main__':
    from config.config_meta import MetaConfig

    seed = 42
    args = MetaConfig()
    res_full = []
    res_per_source = []
    meta_train_task_id_list = args.exp_type['local']['meta_train_tasks']
    meta_test_task_id_list = args.exp_type['local']['meta_test_tasks']
    for paras in args.para_set:

        pre_train_method, with_prompt, meta_learning, gnn_type, pre_train_path = paras

        maml, opt, lossfn = model_components(args, pre_train_path=pre_train_path)
        maml = maml.to(args.device)
        print(maml)
        if meta_learning:
            # meta-training
            print("meta-training for {}.{}.{}.{}...".format(pre_train_method, with_prompt,
                                                            meta_learning, gnn_type))
            meta_train_maml(args.epoch, maml, lossfn, opt,
                            args.dataname, args.adapt_steps, K_shot=args.K_shot,
                            meta_train_task_id_list=meta_train_task_id_list, device=args.device)

            print("meta-test for {}.{}.{}.{}...".format(pre_train_method, with_prompt,
                                                        meta_learning, gnn_type))
            save_pickles = (
                pre_train_method, with_prompt, meta_learning, gnn_type)
            res = meta_test_adam(args.dataname, args.K_shot, seed, maml,
                                 args.adapt_steps_meta_test, lossfn=lossfn, save_project_head=False,
                                 save_pickles=save_pickles, meta_test_task_id_list=meta_test_task_id_list,
                                 device=args.device)

            res_per_source.append([pre_train_method,
                                   with_prompt, meta_learning, gnn_type] + res)

    res_per_source_pd = pd.DataFrame(res_per_source)

    res_per_source_pd.columns = ['PTM', 'prompt', 'meta',
                                 'gnn_type', 'mse']

    path_per_source = '/home/banlujie/metaCounting/saved_model/results/{}.csv'.format(args.dataname)

    res_per_source_pd.to_csv(path_per_source, header=True, index=False)

# res_full_pd = pd.DataFrame(res_full)
#
# res_full_pd.columns = ['source', 'target', 'PTM', 'prompt', 'meta',
#                        'gnn_type', 'task_1_id', 'task_2_id', 'acc', 'f1', 'auc']
# path_res_full = './results/{}.full.xlsx'.format(args.dataname)
#
# res_full_pd.to_excel(path_res_full, header=True, index=False)

print("ALL DONE!")
