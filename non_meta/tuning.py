import datetime
import gc
import json
import logging
import os
import random
import time
import warnings
import sys
from random import shuffle

from sympy.physics.units import force
from torch_geometric.datasets import ZINC
# from torchmetrics.functional import spearman_corrcoef
from torch_geometric.data import Batch, Data
from torch_geometric.loader import NeighborSampler, DataListLoader
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from thop import profile
from data.dataset import PretrainDataset, MyReddit2, MyFlickr, wiki_load

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.functional import huber_loss
from torch_geometric.loader import DataLoader, NeighborLoader, NodeLoader
from torch_geometric.datasets import Flickr
from torch.utils.tensorboard import SummaryWriter
from data.data_load import importance_graph_load, counting_graph_load, load_queries, pca
from non_meta.model_construction import ImportancePipeline, Pipeline,ClassificationPipeline

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
INF = float("inf")
finetune_config = {
    "base": 2,
    "cuda": True,
    "gpu_id": 2,
    "num_workers": 16,
    "epochs": 50,
    "batch_size": 256,
    "update_every": 1,  # actual batch_sizer = batch_size * update_every
    "print_every": 100,
    "init_emb": True,  # None, Normal
    "share_emb": False,  # sharing embedding requires the same vector length
    "share_arch": False,  # sharing architectures
    "dropout": 0.4,
    "dropatt": 0.2,
    "cv": False,
    "reg_loss": "HUBER",  # MAE, MSEl
    "bp_loss": "MSE",  # MAE, MSE
    "bp_loss_slp": "anneal_cosine$1.0$0.01",  # 0, 0.01, logistic$1.0$0.01, linear$1.0$0.01, cosine$1.0$0.01,
    # cyclical_logistic$1.0$0.01, cyclical_linear$1.0$0.01, cyclical_cosine$1.0$0.01
    # anneal_logistic$1.0$0.01, anneal_linear$1.0$0.01, anneal_cosine$1.0$0.01
    "lr": 0.01,
    "weight_decay": 0.0005,
    "trade_off": 1e-10,
    "weight_decay_film": 0.0001,
    "decay_factor": 0.1,
    "attr_ratio": 0.5,
    "decay_patience": 20,
    "max_grad_norm": 8,
    "model": "SAGE",  # Graphormer
    "emb_dim": 32,
    "activation_function": "relu",  # sigmoid, softmax, tanh, relu, leaky_relu, prelu, gelu
    # MeanAttnPredictNet, SumAttnPredictNet, MaxAttnPredictNet,
    # MeanMemAttnPredictNet, SumMemAttnPredictNet, MaxMemAttnPredictNet,
    # DIAMNet, CCANet
    "mem_len": 1,
    "predict_net_mem_init": "mean",
    # mean, sum, max, attn, circular_mean, circular_sum, circular_max, circular_attn, lstm
    "predict_net_recurrent_steps": 3,
    "edgemean_num_bases": 8,
    "edgemean_graph_num_layers": 3,
    "edgemean_pattern_num_layers": 3,
    "edgemean_hidden_dim": 32,
    "init_g_dim": 32,
    "init_e_dim": 4,
    "num_g_hid": 32,
    "num_e_hid": 32,
    "out_g_ch": 32,
    "graph_num_layers": 5,
    "queryset_dir": "queryset",
    "true_card_dir": "label",
    "dataset": "web-spam",
    "data_dir": "dataset",
    "dataset_name": "web-spam",
    "save_res_dir": "result",
    "save_model_dir": "saved_model",
    "task": "importance",
    "test_only": False,
    "hop_number": 1,
    "few_shot": False,
    "shot_num": 10,
    "ana_mode": False,
    "ablation": "no_subgraph",
    "no_subgraph": False,
    "flag": "full",
    "without_prompt": False,
    "without_snn": False,
    "gnn_name": "NGNN",
    "eval_term": "betweenness",  # pagerank, betweenness, eigenvector
    "class_num": 2,
}


def classfication_loss(logits, y):
    y = y.view(-1)
    loss = F.cross_entropy(logits, y)
    return loss

def classification_accuracy(logits, batch):
    # Root (seed) nodes only when using NeighborLoader
    if hasattr(batch, "batch_size"):
        root_sz = batch.batch_size
        logits_used = logits[:root_sz]
        target = batch.y[:root_sz].view(-1)
    else:
        logits_used = logits
        target = batch.y.view(-1)
    pred = logits_used.argmax(dim=-1)
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct, total


def train(model, optimizer, scheduler, data_type, data_loader, device, config, epoch, logger=None, writer=None,
          bottleneck=False, graph=None):
    global bp_crit, reg_crit
    total_bp_loss = 0
    total_cnt = 1e-6
    if config["reg_loss"] == "MAE":
        reg_crit = lambda pred, target: F.l1_loss(F.relu(pred), target)
    elif config["reg_loss"] == "MSE":
        reg_crit = lambda pred, target: F.mse_loss(F.relu(pred), target)
    elif config["reg_loss"] == "SMSE":
        reg_crit = lambda pred, target: F.smooth_l1_loss(pred, target)
    elif config["reg_loss"] == "HUBER":
        reg_crit = lambda pred, target: F.huber_loss(pred, target, delta=0.1)
    if config["bp_loss"] == "MAE":
        bp_crit = lambda pred, target: F.l1_loss(F.relu(pred), target)
    elif config["bp_loss"] == "MSE":
        bp_crit = lambda pred, target: F.mse_loss(pred, target)
    elif config["bp_loss"] == "SMSE":
        bp_crit = lambda pred, target: F.smooth_l1_loss(pred, target)
    elif config["bp_loss"] == "HUBER":
        bp_crit = lambda pred, target: F.huber_loss(pred, target, delta=0.1)
    # data preparation
    # config['init_pe_dim'] = graph.edge_attr.size(1)
    if bottleneck:
        model.load_state_dict(
            torch.load(
                os.path.join(
                    save_model_dir,
                    "best_epoch_{:s}_{:s}.pt".format(
                        finetune_config["predict_net"], finetune_config["graph_net"]
                    ),
                )
            )
        )
    epoch_step = len(data_loader)
    model.to(device)
    model.train()
    total_time = 0
    cnt = 0
    s = time.time()
    # split the graph into several small graphs
    if config['task'] == "importance" or config['task'] == "property":
        for (i, batch) in enumerate(data_loader):
            batch = batch.to(device)
            batch.x = batch.x.to(torch.float32)
            if config['eval_term'] == "pagerank":
                batch.y = batch.pagerank
            elif config['eval_term'] == "betweenness":
                batch.y = batch.betweenness_centrality
            elif config['eval_term'] == "eigenvector":
                batch.y = batch.eigenvector_centrality
            # batch.y_eigen = minmax(batch.y_eigen)
            pred = model(batch)
            # if i < 3:
            #     # print("FLOPS", FlopCountAnalysis(model.prompt, inputs=(torch.Tensor(size=(1, 64)).to("cuda:1"),)).total())
            #     flops, params = profile(model.prompt, inputs=(torch.Tensor(size=(1, 64)).to("cuda:1")))
            #     print(f"flops :{flops}\t params: {params}")
            importance_loss = bp_crit(pred, batch.y)
            importance_loss.backward()
            # logger.info(
            #     "data_type: {:<5s}\t\tCUDA cost: {:.3f} MB\t Memory cost: {:.3f} MB\t (epoch: {:0>3d})".format(
            #         "train", torch.cuda.memory.memory_allocated(device) / (1024 * 1024),
            #         float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)), epoch))
            bp_loss_item = importance_loss.item()
            total_bp_loss += bp_loss_item
            if writer:
                writer.add_scalar(
                    "%s/BP-%s" % (data_type, config["bp_loss"]),
                    bp_loss_item,
                    epoch * epoch_step + i,
                )

            if logger and (
                    (i / len(batch)) % config["print_every"] == 0
                    or i == epoch_step - 1
            ):
                logger.info(
                    "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}\tbp loss: {:.5f}\t".format(
                        int(epoch),
                        int(config["epochs"]),
                        data_type,
                        int(i / len(batch)),
                        int(epoch_step / len(batch)),
                        float(bp_loss_item),
                    )
                )
            if (i + 1) % config["batch_size"] == 0 or cnt == epoch_step - 1:
                if config["max_grad_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["max_grad_norm"]
                    )
                optimizer.step()
                optimizer.zero_grad()
        e = time.time()
        total_time += e - s
        total_cnt += 1

        mean_bp_loss = total_bp_loss / total_cnt
        if writer:
            writer.add_scalar(
                "%s/BP-%s-epoch" % (data_type, config["bp_loss"]), mean_bp_loss, epoch
            )
        if logger:
            logger.info(
                "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbp loss: {:.4f}".format(
                    epoch, config["epochs"], data_type, mean_bp_loss
                )
            )
    elif config['task'] == "localcounting":
        for (i, batch) in enumerate(data_loader):
            batch = batch.to(device)
            pred, reg = model(graph.to(device), batch)
            counting_loss = bp_crit(pred.mean(), batch.y)
            bp_loss = (counting_loss + finetune_config["trade_off"] * reg) / len(batch)
            bp_loss.backward(retain_graph=True)
            bp_loss_item = bp_loss.item()
            total_bp_loss += bp_loss_item
            if writer:
                writer.add_scalar(
                    "%s/BP-%s" % (data_type, config["bp_loss"]),
                    bp_loss_item,
                    epoch * epoch_step + i,
                )

            if logger and (
                    (i / len(batch)) % config["print_every"] == 0
                    or i == epoch_step - 1
            ):
                logger.info(
                    "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}\tbp loss: {:.5f}\t".format(
                        int(epoch),
                        int(config["epochs"]),
                        data_type,
                        int(i / len(batch)),
                        int(epoch_step / len(batch)),
                        float(bp_loss_item),
                    )
                )
            if (i + 1) % config["batch_size"] == 0 or cnt == epoch_step - 1:
                if config["max_grad_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["max_grad_norm"]
                    )
                optimizer.step()
                optimizer.zero_grad()
            e = time.time()
            total_time += e - s
            total_cnt += 1

            mean_bp_loss = total_bp_loss / total_cnt
        if writer:
            writer.add_scalar(
                "%s/BP-%s-epoch" % (data_type, config["bp_loss"]), mean_bp_loss, epoch
            )
        if logger:
            logger.info(
                "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbp loss: {:.4f}".format(
                    epoch, config["epochs"], data_type, mean_bp_loss
                )
            )
    elif config['task'] == "classification" or config['task'] == "lp" :
        total_correct = 0
        total_seen = 0
        for (i, batch) in enumerate(data_loader):
            batch = batch.to(device)
            batch.x = batch.x.to(torch.float32)
            # batch.y_eigen = minmax(batch.y_eigen)
            logits = model(batch)
            if hasattr(batch, "batch_size"):
                loss_logits = logits[:batch.batch_size]
                loss_target = batch.y[:batch.batch_size].view(-1)
            else:
                loss_logits = logits
                loss_target = batch.y.view(-1)
            # if i < 3:
            #     # print("FLOPS", FlopCountAnalysis(model.prompt, inputs=(torch.Tensor(size=(1, 64)).to("cuda:1"),)).total())
            #     flops, params = profile(model.prompt, inputs=(torch.Tensor(size=(1, 64)).to("cuda:1")))
            #     print(f"flops :{flops}\t params: {params}")
            importance_loss = classfication_loss(loss_logits, loss_target)
            importance_loss.backward()
            correct, seen = classification_accuracy(logits, batch)
            total_correct += correct
            total_seen += seen
            acc_now = total_correct / max(1, total_seen)
            # logger.info(
            #     "data_type: {:<5s}\t\tCUDA cost: {:.3f} MB\t Memory cost: {:.3f} MB\t (epoch: {:0>3d})".format(
            #         "train", torch.cuda.memory.memory_allocated(device) / (1024 * 1024),
            #         float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)), epoch))
            bp_loss_item = importance_loss.item()
            total_bp_loss += bp_loss_item
            if writer:
                writer.add_scalar("%s/loss" % data_type, bp_loss_item, epoch * epoch_step + i)
                writer.add_scalar("%s/acc" % data_type, acc_now, epoch * epoch_step + i)

            if logger and ((i) % config["print_every"] == 0 or i == epoch_step - 1):
                logger.info(
                    "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tstep: {:0>5d}/{:0>5d}\tloss: {:.5f}\tacc: {:.4f}".format(
                        int(epoch), int(config["epochs"]), data_type, i, epoch_step, bp_loss_item, acc_now
                    )
                )
            if (i + 1) % config["batch_size"] == 0 or cnt == epoch_step - 1:
                if config["max_grad_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["max_grad_norm"]
                    )
                optimizer.step()
                optimizer.zero_grad()
        e = time.time()
        total_time += e - s
        total_cnt += 1

        mean_bp_loss = total_bp_loss / total_cnt
        epoch_acc = total_correct / max(1, total_seen)
        if writer:
            writer.add_scalar("%s/epoch_loss" % data_type, mean_bp_loss, epoch)
            writer.add_scalar("%s/epoch_acc" % data_type, epoch_acc, epoch)
        if logger:
            logger.info(
                "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tloss: {:.4f}\tacc: {:.4f}".format(
                    epoch, config["epochs"], data_type, mean_bp_loss, epoch_acc
                )
            )
    gc.collect()
    return mean_bp_loss, total_time


def evaluate(model, data_type, data_loader, config, logger=None, writer=None, graph=None):
    epoch_step = len(data_loader)
    total_step = config["epochs"] * epoch_step
    total_var_loss = 0
    total_reg_loss = 0
    total_bp_loss = 0
    total_cnt = 1e-6

    evaluate_results = {"error": {"importance_loss": list(), "spearman corr": list(), "rmse": list()},
                        "time": {"avg": list(), "total": 0.0}}
    # val_graph_loader = DataLoader(graph, batch_size=512, shuffle=True)
    model.eval()
    total_time = 0
    with torch.no_grad():
        # for gbatch in val_graph_loader:
        #     gbatch = gbatch.cuda()

        if config['task'] == "localcounting":
            st = time.time()
            for batch_id, batch in enumerate(data_loader):
                batch = batch.cuda()
                # importance = minmax(importance)
                graph = graph.cuda()
                pred, reg = model(graph, batch)
                counting_loss = F.mse_loss(pred.mean(), batch.y)
                et = time.time()
                counting_loss_item = counting_loss.item() if config['task'] == "localcounting" else None
                bp_loss_item = counting_loss_item
                evaluate_results["error"]["importance_loss"].extend([bp_loss_item])
                evaluate_results["time"]["total"] += et - st
                avg_t = (et - st) / len(batch)
                evaluate_results["time"]["avg"].extend([avg_t])
                total_bp_loss += bp_loss_item
                et = time.time()
                total_time += et - st
                total_cnt += len(batch)
                mean_bp_loss = total_bp_loss / len(batch)

                if logger and batch_id == epoch_step - 1 and config["test_only"] is False:
                    logger.info(
                        "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:d}/{:d}\tbp loss: {:.4f}".format(
                            int(epoch),
                            int(config["epochs"]),
                            (data_type),
                            int(batch_id),
                            int(epoch_step),
                            float(bp_loss_item),
                        )
                    )
                # float(var), float(pred_var[0].item())))
                # "ground: {:.4f}\tpredict: {:.4f}"

            if logger and config["test_only"] is False:
                logger.info(
                    "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbp loss: {:.4f}".format(
                        epoch, config["epochs"], data_type, mean_bp_loss
                    )
                )
        elif config['task'] == "importance" or config['task'] == "property":
            for batch_id, batch in enumerate(data_loader):
                if config['eval_term'] == "pagerank":
                    batch.y = batch.pagerank
                elif config['eval_term'] == "betweenness":
                    batch.y = batch.betweenness_centrality
                elif config['eval_term'] == "eigenvector":
                    batch.y = batch.eigenvector_centrality
                st = time.time()
                batch = batch.cuda()
                importance = batch.y
                pred = model(batch)
                importance_loss = F.mse_loss(pred, importance)
                rmse_importance = torch.sqrt(importance_loss)
                bp_loss = importance_loss
                # correlation = spearman_corrcoef(pred.squeeze(1), importance)
                bp_loss_item = bp_loss.mean().item()
                evaluate_results["error"]["importance_loss"].extend([bp_loss_item])
                # evaluate_results["error"]["spearman corr"].extend([correlation.item()])
                evaluate_results["error"]["rmse"].extend([rmse_importance.item() * 10 ** 5])
                et = time.time()
                evaluate_results["time"]["total"] += et - st
                avg_t = (et - st) / len(batch)
                evaluate_results["time"]["avg"].extend([avg_t])
            total_bp_loss += bp_loss_item

            et = time.time()
            total_time += et - st
            total_cnt += len(batch)
            mean_bp_loss = total_bp_loss / total_cnt

            if logger and batch_id == epoch_step - 1 and config["test_only"] is False:
                logger.info(
                    "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:d}/{:d}\tbp loss: {:.4f}".format(
                        int(epoch),
                        int(config["epochs"]),
                        (data_type),
                        int(batch_id),
                        int(epoch_step),
                        float(bp_loss_item),
                    )
                )
            # float(var), float(pred_var[0].item())))
            # "ground: {:.4f}\tpredict: {:.4f}"

            if logger and config["test_only"] is False:
                logger.info(
                    "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbp loss: {:.4f}".format(
                        epoch, config["epochs"], data_type, mean_bp_loss
                    )
                )
        elif config['task'] == "classification":
            total_correct = 0
            total_seen = 0
            for batch_id, batch in enumerate(data_loader):
                st = time.time()
                batch = batch.cuda()
                logits = model(batch)
                if hasattr(batch, "batch_size"):
                    loss_logits = logits[:batch.batch_size]
                    loss_target = batch.y[:batch.batch_size].view(-1)
                else:
                    loss_logits = logits
                    loss_target = batch.y.view(-1)
                importance_loss = F.cross_entropy(loss_logits, loss_target)
                bp_loss_item = importance_loss.item()
                rmse_importance = torch.sqrt(importance_loss)
                bp_loss = importance_loss
                # correlation = spearman_corrcoef(pred.squeeze(1), importance)
                correct, seen = classification_accuracy(logits, batch)
                total_correct += correct
                total_seen += seen
                acc_now = total_correct / max(1, total_seen)

                evaluate_results["error"]["importance_loss"].append(bp_loss_item)
                evaluate_results["error"].setdefault("acc", []).append(acc_now)
                evaluate_results["error"]["rmse"].append(rmse_importance.item() * 10 ** 5)

                et = time.time()
                evaluate_results["time"]["total"] += et - st
                avg_t = (et - st) / max(1, seen)
                evaluate_results["time"]["avg"].append(avg_t)
                bp_loss_item = bp_loss.mean().item()
            total_bp_loss += bp_loss_item
            mean_bp_loss = total_bp_loss / max(1, total_seen)
            mean_acc = total_correct / max(1, total_seen)

            if logger and batch_id == epoch_step - 1 and config["test_only"] is False:
                logger.info(
                    "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:d}/{:d}\tbp loss: {:.4f}".format(
                        int(epoch),
                        int(config["epochs"]),
                        (data_type),
                        int(batch_id),
                        int(epoch_step),
                        float(bp_loss_item),
                    )
                )
            # float(var), float(pred_var[0].item())))
            # "ground: {:.4f}\tpredict: {:.4f}"

            if logger and config["test_only"] is False:
                logger.info(
                    "eval data_type: {:<5s}\tloss: {:.4f}\tacc: {:.4f}".format(
                        data_type, mean_bp_loss, mean_acc
                    )
                )
            evaluate_results["summary_acc"] = 1 - mean_acc
    gc.collect()
    return mean_bp_loss, evaluate_results, total_time


def model_test(save_model_dir, test_loaders, config, logger, writer, graph=None):
    total_test_time = 0
    model.load_state_dict(
        torch.load(
            os.path.join(
                save_model_dir,
                "best_epoch_{:s}_{:d}.pt".format(finetune_config["model"], finetune_config['graph_num_layers']),
            )
        )
    )
    # print(model)
    mean_bp_loss, evaluate_results, _time = evaluate(
        model=model.cuda(),
        data_type="test",
        data_loader=test_loaders,
        config=config,
        logger=logger,
        writer=writer,
        graph=graph
    )
    total_test_time += _time
    # if mean_reg_loss <= best_reg_losses['test']:
    #     best_reg_losses['test'] = mean_reg_loss
    # best_reg_epochs['test'] =
    logger.info(
        "data_type: {:<5s}\tbest mean loss: {:.3f}".format("test", mean_bp_loss)
    )
    save_path = os.path.join(
        save_model_dir,
        "%s_%s_pre_trained_%s_%s_%s_%s_%d_%s_%s_%s.json"
        % (
            finetune_config["model"],
            finetune_config["dataset_name"],
            finetune_config['task'],
            finetune_config['hop_number'],
            finetune_config['ablation'],
            finetune_config['flag'],
            finetune_config['graph_num_layers'],
            finetune_config['gnn_name'],
            finetune_config['eval_term'],
            finetune_config['task']
        )
    )
    with open(save_path, "w") as f:
        json.dump(evaluate_results, f)
        print("results are saved at " + save_path)
    return evaluate_results, total_test_time


def loader_construction(finetune_config):
    if finetune_config['no_subgraph'] or finetune_config['dataset'] in ['reddit', 'flickr', "wiki"]:
        if finetune_config['dataset'] == 'reddit':
            data = MyReddit2(root='/mnt/data/banlujie/dataset/Reddit')[0]
        elif finetune_config['dataset'] == "flickr":
            data = MyFlickr(root="/mnt/data/banlujie/dataset/flickr")[0]
            data.x = pca(data.x, 32)
        elif finetune_config['dataset'] == "wiki":
            data = wiki_load(root="/mnt/data/banlujie/dataset/wiki", name=finetune_config['dataset_name'])
        else:
            data = PretrainDataset(name=finetune_config['dataset'], filepath=finetune_config['dataset'])[0][0] if \
                finetune_config[
                    'dataset'] == "flixster" else \
                PretrainDataset(name=finetune_config['dataset'], filepath=finetune_config['dataset'],
                                force_reload=True)[0]
        kwargs = dict(
            data=data,
            num_neighbors=[-1, -1] * 2,
            batch_size=finetune_config["batch_size"],
            num_workers=8,
            pin_memory=True
        )
        if finetune_config['dataset'] in ["web-spam", "indochina", "brightkite", "facebook", "youtube", "reddit",
                                          'flickr']:
            train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
            val_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
            test_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
        elif finetune_config['dataset'] == "wiki":
            train_idx = data.train_mask.nonzero(as_tuple=False)
            val_idx = data.val_mask.nonzero(as_tuple=False)
            test_idx = data.val_mask.nonzero(as_tuple=False)
        else:
            train_idx = data.train_idx
            val_idx = data.val_idx
            test_idx = data.test_idx
        train_loader = NeighborLoader(
            input_nodes=train_idx,
            shuffle=True,
            **kwargs,
        )
        val_loader = NeighborLoader(
            input_nodes=val_idx,
            shuffle=False,
            **kwargs,
        )
        test_loader = NeighborLoader(
            input_nodes=test_idx,
            shuffle=False,
            **kwargs,
        )

    else:
        train_loader, val_loader, test_loader = importance_graph_load(
            finetune_config['dataset'], batch_size=finetune_config["batch_size"], train_ratio=0.8, val_ratio=0.1,
            few_shot=finetune_config['few_shot'], shot_num=finetune_config['shot_num'],
            k=finetune_config['hop_number'],
            ana_mode=finetune_config['ana_mode'], no_subgraph=finetune_config['no_subgraph']
        )
    return train_loader, val_loader, test_loader


def lsc_loader_construction(finetune_config):
    if finetune_config['no_subgraph']:
        data = PretrainDataset(name=finetune_config['dataset'], filepath=finetune_config['dataset'])[0][0] if \
            finetune_config[
                'dataset'] == "flixster" else \
            PretrainDataset(name=finetune_config['dataset'], filepath=finetune_config['dataset'])[0]
        # data.x = data.x.squeeze(1)
        train_graph = data
        val_graph = data
        test_graph = data
        _, _, _, motif_train_loader, motif_val_loader, motif_test_loader = counting_graph_load(
            finetune_config['dataset'], batch_size=finetune_config["batch_size"], train_ratio=0.8, val_ratio=0.1,
            k=finetune_config['hop_number'],
            ana_mode=finetune_config['ana_mode']
        )
    elif finetune_config['dataset'] in ['reddit', 'flickr']:
        data = MyFlickr("/mnt/data/banlujie/dataset/flickr")[0]
        data.x = pca(data.x, 32)
        train_graph, val_graph, test_graph = data, data, data
        query_load_path = os.path.join("/mnt/data/banlujie/dataset", finetune_config['dataset'], "query_graph")
        true_card_load_path = os.path.join("/mnt/data/banlujie/dataset", finetune_config['dataset'], "label")
        query_graphs, num_queries, size_num, pattern_num, _ = load_queries(
            query_load_path, true_card_load_path, dataname=finetune_config['dataset'], submode=False)
        motif_list = [item for sublist in list(query_graphs.values()) for item in
                      sublist]  # list(query_graphs.values())
        motif_batch = Batch.from_data_list(motif_list)
        motif_batch.edge_attr = motif_batch.edge_attr.to(torch.float32)
        num_instances = len(motif_list)
        random.shuffle(motif_list)

        train_sets = motif_batch[: int(num_instances * 0.8)]
        val_sets = motif_batch[int(num_instances * 0.8): int(num_instances * 0.85)]
        test_sets = motif_batch[int(num_instances * 0.85):]
        motif_train_loader, motif_val_loader, motif_test_loader = DataLoader(train_sets,
                                                                             batch_size=finetune_config[
                                                                                 'batch_size']), DataLoader(
            val_sets, batch_size=finetune_config['batch_size']), DataLoader(test_sets,
                                                                            batch_size=finetune_config['batch_size'])
    elif finetune_config['dataset'] == "wiki":
        data = wiki_load(root="/mnt/data/banlujie/dataset/wiki", name=finetune_config['dataset_name'])
        data.x = pca(data.x, 32)

        train_graph, val_graph, test_graph = data, data, data
        query_load_path = os.path.join("/mnt/data/banlujie/dataset", finetune_config['dataset'],
                                       finetune_config['dataset_name'], "query_graph")
        true_card_load_path = os.path.join("/mnt/data/banlujie/dataset", finetune_config['dataset'],
                                           finetune_config['dataset_name'], "label")
        query_graphs, num_queries, size_num, pattern_num, _ = load_queries(
            query_load_path, true_card_load_path, dataname=finetune_config['dataset_name'], submode=False)
        motif_list = [item for sublist in list(query_graphs.values()) for item in
                      sublist]  # list(query_graphs.values())
        motif_batch = Batch.from_data_list(motif_list)
        motif_batch.edge_attr = motif_batch.edge_attr.to(torch.float32)
        num_instances = len(motif_list)
        random.shuffle(motif_list)
        train_sets = motif_batch[: int(num_instances * 0.8)]
        val_sets = motif_batch[int(num_instances * 0.8): int(num_instances * 0.85)]
        test_sets = motif_batch[int(num_instances * 0.85):]
        motif_train_loader, motif_val_loader, motif_test_loader = DataLoader(train_sets,
                                                                             batch_size=finetune_config[
                                                                                 'batch_size']), DataLoader(
            val_sets, batch_size=finetune_config['batch_size']), DataLoader(test_sets,
                                                                            batch_size=finetune_config['batch_size'])

    else:
        train_graph, val_graph, test_graph, motif_train_loader, motif_val_loader, motif_test_loader = counting_graph_load(
            finetune_config['dataset'], batch_size=finetune_config["batch_size"], train_ratio=0.8, val_ratio=0.1,
            k=finetune_config['hop_number'],
            ana_mode=finetune_config['ana_mode']
        )

    return train_graph, val_graph, test_graph, motif_train_loader, motif_val_loader, motif_test_loader

def classification_loader_construction(finetune_config):

    if finetune_config['no_subgraph'] or finetune_config['dataset'] in ['reddit', 'flickr', "wiki"]:
        if finetune_config['dataset'] == 'reddit':
            data = MyReddit2(root='/mnt/data/banlujie/dataset/Reddit')[0]
        elif finetune_config['dataset'] == "flickr":
            dataset = Flickr(root="/mnt/data/banlujie/dataset/flickr_org")
            data = dataset[0]
            finetune_config['class_num'] = dataset.num_classes
            data.x = pca(data.x, 32)
            data.edge_attr = torch.ones_like(data.edge_index[0].view(-1, 1), dtype=torch.float32)
        elif finetune_config['dataset'] == "wiki":
            data = wiki_load(root="/mnt/data/banlujie/dataset/wiki", name=finetune_config['dataset_name'])
        else:
            data = PretrainDataset(name=finetune_config['dataset'], filepath=finetune_config['dataset'])[0][0] if \
                finetune_config[
                    'dataset'] == "flixster" else \
                PretrainDataset(name=finetune_config['dataset'], filepath=finetune_config['dataset'],
                                force_reload=True)[0]
        kwargs = dict(
            data=data,
            num_neighbors=[10, 5],
            batch_size=finetune_config["batch_size"],
            num_workers=8,
            pin_memory=True
        )
        if finetune_config['dataset'] in ["web-spam", "indochina", "brightkite", "facebook", "youtube", "reddit",
                                          'flickr']:
            train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
            val_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
            test_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
        elif finetune_config['dataset'] == "wiki":
            train_idx = data.train_mask.nonzero(as_tuple=False)
            val_idx = data.val_mask.nonzero(as_tuple=False)
            test_idx = data.val_mask.nonzero(as_tuple=False)
        else:
            train_idx = data.train_idx
            val_idx = data.val_idx
            test_idx = data.test_idx
        train_loader = NeighborLoader(
            input_nodes=train_idx,
            shuffle=True,
            **kwargs,
        )
        val_loader = NeighborLoader(
            input_nodes=val_idx,
            shuffle=False,
            **kwargs,
        )
        test_loader = NeighborLoader(
            input_nodes=test_idx,
            shuffle=False,
            **kwargs,
        )

    else:
        train_loader, val_loader, test_loader = importance_graph_load(
            finetune_config['dataset'], batch_size=finetune_config["batch_size"], train_ratio=0.8, val_ratio=0.1,
            few_shot=finetune_config['few_shot'], shot_num=finetune_config['shot_num'],
            k=finetune_config['hop_number'],
            ana_mode=finetune_config['ana_mode'], no_subgraph=finetune_config['no_subgraph']
        )
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    for i in range(1, len(sys.argv), 2):
        arg = sys.argv[i]
        value = sys.argv[i + 1]

        if arg.startswith("--"):
            arg = arg[2:]
        if arg not in finetune_config:
            print("Warning: %s is not surported now." % (arg))
            continue
        finetune_config[arg] = value
        try:
            value = eval(value)
            if isinstance(value, (int, float)):
                finetune_config[arg] = value
        except:
            pass

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_name = "%s_%s" % (finetune_config["model"], ts)
    save_model_dir = finetune_config["save_model_dir"]
    os.makedirs(save_model_dir, exist_ok=True)

    # save config
    with open(os.path.join(save_model_dir, "finetune_config.json"), "w") as f:
        json.dump(finetune_config, f)

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="[ %(asctime)s ] %(message)s", datefmt="%a %b %d %H:%M:%S %Y"
    )
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    local_time = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    logfile = logging.FileHandler(
        os.path.join(save_model_dir, "train_log_{:s}.txt".format(local_time)), "w"
    )
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    # set device
    device = torch.device(
        "cuda:%d" % finetune_config["gpu_id"] if finetune_config["gpu_id"] != -1 else "cpu"
    )
    if finetune_config["gpu_id"] != -1 and finetune_config["test_only"] == False:
        torch.cuda.set_device(device)

        # load data
        # os.makedirs(train_config["save_data_dir"], exist_ok=True)
    # decompose the query
    if finetune_config['task'] == "importance":
        train_loader, val_loader, test_loader = loader_construction(finetune_config)
        finetune_config["init_g_dim"] = next(iter(train_loader)).x.shape[1]
        model = ImportancePipeline(
            input_dim=finetune_config["init_g_dim"], layer_num=finetune_config['graph_num_layers'],
            pre_train_path=os.path.abspath(
                os.path.join("/home", "banlujie", "metaCounting", finetune_config['save_model_dir'],
                             "best_epoch_" + finetune_config[
                                 'model'] + f"_{finetune_config['graph_num_layers']}" + ".pt")),
            frozen_prompt=False, gnn_name=finetune_config['gnn_name']
        )
    elif finetune_config['task'] == "localcounting":
        train_graph, val_graph, test_graph, train_motif_loader, val_motif_loader, test_motif_loader = lsc_loader_construction(
            finetune_config)
        finetune_config['init_g_dim'] = (train_graph).x.size(1)
        model = Pipeline(gnn_type=finetune_config["model"],
                         input_dim=finetune_config["init_g_dim"], hid_dim=finetune_config['emb_dim'],
                         layer_num=finetune_config['graph_num_layers'],
                         pre_train_path=os.path.abspath(
                             os.path.join("/home", "banlujie", "metaCounting", finetune_config['save_model_dir'],
                                          "best_epoch_" + finetune_config[
                                              'model'] + f"_{finetune_config['graph_num_layers']}" + ".pt")),
                         without_prompt=finetune_config['without_prompt'],
                         without_snn=finetune_config['without_snn'], gnn_name=finetune_config['gnn_name']
                         )
    elif finetune_config['task'] == "property":
        train_set, val_set, test_set = ZINC(root="/mnt/data/banlujie/dataset/zinc", split='train'), ZINC(
            root="/mnt/data/banlujie/dataset/zinc", split='val'), ZINC(root="/mnt/data/banlujie/dataset/zinc",
                                                                       split='test')
        finetune_config["init_g_dim"] = train_set[0].x.size(1)
        train_loader, val_loader, test_loader = DataLoader(train_set,
                                                           batch_size=finetune_config["batch_size"],
                                                           shuffle=True), DataLoader(
            val_set, batch_size=finetune_config["batch_size"], shuffle=False), DataLoader(test_set,
                                                                                          batch_size=finetune_config[
                                                                                              "batch_size"],
                                                                                          shuffle=False)
        model = ImportancePipeline(
            input_dim=finetune_config["init_g_dim"], layer_num=finetune_config['graph_num_layers'],
            pre_train_path=os.path.join("..", finetune_config['save_model_dir'],
                                        "best_epoch_" + finetune_config[
                                            'model'] + f"_{finetune_config['graph_num_layers']}.pt"),
            frozen_prompt=False
        )
    # train_config.update({'init_g_dim': graph.x.size(1)})
    # construct the model
    elif finetune_config['task'] == "classification":
        train_loader, val_loader, test_loader = classification_loader_construction(finetune_config)
        finetune_config["init_g_dim"] = next(iter(train_loader)).x.shape[1]
        model = ClassificationPipeline(
            input_dim=finetune_config["init_g_dim"], layer_num=finetune_config['graph_num_layers'],
            pre_train_path=os.path.abspath(
                os.path.join("/home", "banlujie", "metaCounting", finetune_config['save_model_dir'],
                             "best_epoch_" + finetune_config[
                                 'model'] + f"_{finetune_config['graph_num_layers']}" + ".pt")),
            frozen_prompt=False, gnn_name=finetune_config['gnn_name'], num_classes=finetune_config['class_num']
        )
    else:
        raise NotImplementedError(
            "Currently, the %s model is not supported" % (finetune_config["model"])
        )
    # model = torch.compile(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"Parameter: {name}, Size: {param.size()}")
    logger.info(
        "num of parameters: %d"
        % (sum(p.numel() for p in model.parameters() if p.requires_grad))
    )

    # optimizer and losses
    writer = SummaryWriter()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), finetune_config['lr'])
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=finetune_config["decay_factor"]
    )
    best_bp_losses = INF
    best_bp_epochs = {"train": -1, "val": -1, "test": -1}
    torch.backends.cudnn.benchmark = True
    total_train_time = 0
    total_dev_time = 0
    total_test_time = 0
    cur_reg_loss = {}
    if finetune_config["test_only"]:
        evaluate_results, total_test_time = model_test(
            save_model_dir, test_loader if finetune_config['task'] in ["importance", "property"] else test_motif_loader,
            finetune_config, logger, writer,
            graph=test_graph if finetune_config['task'] == "localcounting" else None
        )
        exit(0)
    tolerance_cnt = 0
    for epoch in range(finetune_config["epochs"]):
        # if train_config['cv'] == True:
        #     cross_validate(model=model, query_set=QS, device=device, config=train_config, graph=graph, logger=logger,
        #                    writer=writer)
        # else:
        mean_bp_loss, _time = train(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_type="train",
            data_loader=train_motif_loader if finetune_config['task'] == "localcounting" else train_loader,
            device=device,
            config=finetune_config,
            epoch=epoch,
            logger=logger,
            writer=writer,
            bottleneck=False,
            graph=train_graph if finetune_config['task'] == "localcounting" else None
        )
        logger.info("data_type: {:<5s}\t\tCUDA cost: {:.3f} MB\t Memory cost: {:.3f} MB\t (epoch: {:0>3d})".format(
            "train", torch.cuda.memory.memory_allocated(device) / (1024 * 1024),
            float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)), epoch))
        total_train_time += _time
        if scheduler and (epoch + 1) % finetune_config["decay_patience"] == 0:
            scheduler.step()
            # torch.save(model.state_dict(), os.path.join(save_model_dir, 'epoch%d.pt' % (epoch)))
        mean_bp_loss, evaluate_results, total_time = evaluate(
            model=model,
            data_type="val",
            data_loader=val_motif_loader if finetune_config['task'] == "localcounting" else val_loader,
            config=finetune_config,
            logger=logger,
            writer=writer,
            graph=val_graph if finetune_config['task'] == "localcounting" else None
        )
        logger.info("data_type: {:<5s}\t\tCUDA cost: {:.3f} MB\t Memory cost: {:.3f} MB\t (epoch: {:0>3d})".format(
            "val", torch.cuda.memory.memory_allocated(device) / (1024 * 1024),
            float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)), epoch))
        if writer:
            writer.add_scalar(
                "%s/BP-%s-epoch" % ("val", finetune_config["bp_loss"]), mean_bp_loss, epoch
            )
            total_dev_time += total_time

        if best_bp_losses - mean_bp_loss > 1e-4:
            tolerance_cnt = 0
            best_bp_losses = mean_bp_loss
            # best_reg_epochs["val"] = epoch
            logger.info(
                "data_type: {:<5s}\t\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(
                    "val", mean_bp_loss, epoch
                )
            )
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_model_dir,
                    "best_epoch_{:s}_{:d}.pt".format(finetune_config["model"], finetune_config['graph_num_layers']),
                ),
            )
            with open(
                    os.path.join(save_model_dir, "%s_%d.json" % ("val", epoch)), "w"
            ) as f:
                json.dump(evaluate_results, f)
                # for data_type in data_loaders.keys():
                #     logger.info(
                #         "data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(data_type,
                #                                                                             best_reg_losses[data_type],
                #                                                                             best_reg_epochs[data_type]))
        tolerance_cnt += 1
        if tolerance_cnt >= 20:
            break
    print("data finish")
    evaluate_results, total_test_time = model_test(
        save_model_dir, test_loader if finetune_config['task'] == "importance" or finetune_config[
            'task'] == "property" or finetune_config['task'] == "classification" else test_motif_loader, finetune_config,
        logger, writer,
        graph=test_graph if finetune_config['task'] == "localcounting" else None
    )
    logger.info(
        "train time: {:.3f}, train time per epoch :{:.3f}, test time: {:.3f}, all time: {:.3f}".format(
            total_train_time,
            total_train_time / finetune_config["epochs"],
            total_test_time,
            total_train_time + total_dev_time + total_test_time,
        )
    )
