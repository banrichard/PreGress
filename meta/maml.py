from copy import deepcopy

import pandas as pd
from torch_geometric.graphgym import optim

from data.data_load import load4graph, meta_dataset_load
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
        save_pickles=None):
    pre_train_method, with_prompt, meta_learning, gnn_type = save_pickles

    task_results = []
    # meta-testing

    for support, query, test in load4graph('yeast'):

        test_model = deepcopy(maml.module)
        test_opi = optim.Adam(filter(lambda p: p.requires_grad, test_model.parameters()),
                              lr=0.001,
                              weight_decay=0.00001)

        test_model.train()

        for _ in range(adapt_steps_meta_test):
            support_preds = test_model(support)
            support_loss = lossfn(support_preds, support.label)
            if _ % 5 == 0:
                print('{}/{} training loss: {:.8f}'.format(_,
                                                           adapt_steps_meta_test,
                                                           support_loss.item()))
            test_opi.zero_grad()
            support_loss.backward()
            test_opi.step()

        test_model.eval()
        query_preds = test_model(query)
        mse_loss = mean_squared_error(query.label, query_preds)
        print("""\t MSE Loss: {:.4} """.format(mse_loss))
        task_results.append([mse_loss])
        if save_project_head:
            torch.save(test_model.project_head.state_dict(),
                       "/home/banlujie/saved_model/projection_head/{}.{}.{}.pth".format(dataname, pre_train_method,
                                                                                        gnn_type))
            print("project head saved! @./projection_head/{}.{}.{}.pth".format(dataname, pre_train_method, gnn_type))

    return task_results


def meta_train_maml(epoch, maml, lossfn, opt, dataname, adapt_steps, K_shot=10):
    # meta-training
    graph = meta_dataset_load("yeast")
    for ep in range(epoch):
        meta_train_loss = 0.0
        pair_count = 0
        seed = 8964
        support, query, test = load4graph(dataname)
        pair_count = pair_count + 1

        learner = maml.clone()

        for _ in range(adapt_steps):  # adaptation_steps
            support_preds = learner(graph, support)
            support_loss = lossfn(support_preds, support.label)
            learner.adapt(support_loss)

        query_preds = learner(query)
        query_loss = lossfn(query_preds, query.label)
        meta_train_loss += query_loss

        print('\tmeta_train_loss at epoch {}/{}: {}'.format(ep, epoch, meta_train_loss.item()))
        meta_train_loss = meta_train_loss / pair_count
        opt.zero_grad()
        meta_train_loss.backward()
        opt.step()


if __name__ == '__main__':
    from config.meta_config import MetaConfig

    seed = 8964
    args = MetaConfig()
    res_full = []
    res_per_source = []

    for paras in args.para_set:

        pre_train_method, with_prompt, meta_learning, gnn_type, pre_train_path = paras

        maml, opt, lossfn = model_components(args, pre_train_path=pre_train_path)

        if meta_learning:
            # meta-training
            print("meta-training for {}.{}.{}.{}...".format(pre_train_method, with_prompt,
                                                            meta_learning, gnn_type))
            meta_train_maml(args.epoch, maml, lossfn, opt,
                            args.dataname, args.adapt_steps, K_shot=args.K_shot)

            print("meta-test for {}.{}.{}.{}...".format(pre_train_method, with_prompt,
                                                        meta_learning, gnn_type))
            save_pickles = (
                pre_train_method, with_prompt, meta_learning, gnn_type)
            res = meta_test_adam(args.dataname, args.K_shot, seed, maml,
                                 args.adapt_steps_meta_test, lossfn=lossfn, save_project_head=False,
                                 save_pickles=save_pickles)

            res_per_source.append([pre_train_method,
                                   with_prompt, meta_learning, gnn_type] + res)

    res_per_source_pd = pd.DataFrame(res_per_source)

    res_per_source_pd.columns = ['source', 'target', 'PTM', 'prompt', 'meta',
                                 'gnn_type', 'task_1_id', 'task_2_id', 'acc', 'f1', 'auc']

    path_per_source = '/home/banlujie/metaCounting/saved_model/results/.{}.xlsx'.format(args.dataname)

    res_per_source_pd.to_excel(path_per_source, header=True, index=False)

# res_full_pd = pd.DataFrame(res_full)
#
# res_full_pd.columns = ['source', 'target', 'PTM', 'prompt', 'meta',
#                        'gnn_type', 'task_1_id', 'task_2_id', 'acc', 'f1', 'auc']
# path_res_full = './results/{}.full.xlsx'.format(args.dataname)
#
# res_full_pd.to_excel(path_res_full, header=True, index=False)

print("ALL DONE!")
