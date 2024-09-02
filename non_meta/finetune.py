import datetime
import gc
import json
import logging
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.functional import huber_loss
from torch.utils.tensorboard import SummaryWriter
from data.data_load import dataset_load, importance_graph_load
from non_meta.model_construction import ImportancePipeline
from pretrain.GIN_pretrain import GIN
from pretrain.graphormer_pretrain import Gphormer

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
INF = float("inf")

finetune_config = {
    "base": 2,
    "cuda": True,
    "gpu_id": 2,
    "num_workers": 16,
    "epochs": 200,
    "batch_size": 16,
    "update_every": 1,  # actual batch_sizer = batch_size * update_every
    "print_every": 10,
    "init_emb": True,  # None, Normal
    "share_emb": False,  # sharing embedding requires the same vector length
    "share_arch": False,  # sharing architectures
    "dropout": 0.4,
    "dropatt": 0.2,
    "cv": False,
    "reg_loss": "HUBER",  # MAE, MSEl
    "bp_loss": "HUBER",  # MAE, MSE
    "bp_loss_slp": "anneal_cosine$1.0$0.01",  # 0, 0.01, logistic$1.0$0.01, linear$1.0$0.01, cosine$1.0$0.01,
    # cyclical_logistic$1.0$0.01, cyclical_linear$1.0$0.01, cyclical_cosine$1.0$0.01
    # anneal_logistic$1.0$0.01, anneal_linear$1.0$0.01, anneal_cosine$1.0$0.01
    "lr": 0.00001,
    "weight_decay": 0.0005,
    "weight_decay_var": 0.1,
    "weight_decay_film": 0.0001,
    "decay_factor": 0.1,
    "attr_ratio": 0.5,
    "decay_patience": 20,
    "max_grad_norm": 8,
    "model": "GIN",  # Graphormer
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
    "init_g_dim": 11,
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
}


def minmax(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def train(
        model,
        optimizer,
        scheduler,
        data_type,
        data_loader,
        device,
        config,
        epoch,
        logger=None,
        writer=None,
        bottleneck=False,
):
    global bp_crit, reg_crit
    epoch_step = len(data_loader)
    total_step = config["epochs"] * epoch_step
    total_var_loss = 0
    total_reg_loss = 0
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
        bp_crit = lambda pred, target: F.l1_loss(F.leaky_relu(pred), target)
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
    model.to(device)

    model.train()
    total_time = 0
    for i, batch in enumerate(data_loader):
        batch = batch.to(device)
        batch.x = batch.x.to(torch.float32)
        batch.y_eigen = minmax(batch.y_eigen)
        s = time.time()
        if config['model'] == "Graphormer":
            pred, importance_loss = model(batch)
            total_loss_per_step = importance_loss
        else:
            pred = model(batch)
            importance_loss = bp_crit(pred, batch.y_eigen)
            importance_loss.backward()
        bp_loss_item = importance_loss.item()
        total_bp_loss += bp_loss_item

        if writer:
            writer.add_scalar(
                "%s/BP-%s" % (data_type, config["bp_loss"]),
                bp_loss_item,
                epoch * epoch_step + i,
            )

        if logger and (
                (i / config["batch_size"]) % config["print_every"] == 0
                or i == epoch_step - 1
        ):
            logger.info(
                "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}\tbp loss: {:.5f}\t".format(
                    int(epoch),
                    int(config["epochs"]),
                    data_type,
                    int(i / config["batch_size"]),
                    int(epoch_step / config["batch_size"]),
                    float(bp_loss_item),
                )
            )

        if (i + 1) % config["batch_size"] == 0 or i == epoch_step - 1:
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

    gc.collect()
    return mean_bp_loss, total_time


def evaluate(model, data_type, data_loader, config, logger=None, writer=None):
    epoch_step = len(data_loader)
    total_step = config["epochs"] * epoch_step
    total_var_loss = 0
    total_reg_loss = 0
    total_bp_loss = 0
    total_cnt = 1e-6

    evaluate_results = {"mean": {"importance": list(), "preds": list()},
                        "error": {"importance_loss": list(), "attr_loss": list()},
                        "time": {"avg": list(), "total": 0.0}}
    model.eval()
    model = model.to("cpu")
    total_time = 0
    preds = []
    with torch.no_grad():
        for batch_id, batch in enumerate(data_loader):
            batch.x = batch.x.to(torch.float32)
            st = time.time()
            importance = batch.y_eigen
            importance = minmax(importance)
            evaluate_results["mean"]["importance"].extend(importance.view(-1).tolist())
            if config['model'] == "Graphormer":
                pred, importance_loss = model(batch)
                bp_loss = importance_loss
            else:
                pred = model(batch)
                evaluate_results["mean"]["preds"].extend(pred.view(-1).tolist())
                importance_loss = huber_loss(pred, importance)
                bp_loss = importance_loss
            et = time.time()
            evaluate_results["time"]["total"] += et - st
            avg_t = et - st

            evaluate_results["time"]["avg"].extend([avg_t])
            bp_loss_item = bp_loss.mean().item()
            total_bp_loss += bp_loss_item
            evaluate_results["error"]["importance_loss"].extend(importance_loss.view(-1).tolist())
            et = time.time()
            total_time += et - st
            total_cnt += 1
        mean_bp_loss = total_bp_loss / total_cnt
        if logger and batch_id == epoch_step - 1 and config["test_only"] is False:
            logger.info(
                "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:d}/{:d}\tbp loss: {:.4f}\t".format(
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
                "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\t\tbp loss: {:.4f}".format(
                    epoch, config["epochs"], data_type, mean_bp_loss
                )
            )

    gc.collect()
    return mean_bp_loss, evaluate_results, total_time


def test(save_model_dir, test_loaders, config, logger, writer):
    total_test_time = 0
    model.load_state_dict(
        torch.load(
            os.path.join(
                save_model_dir,
                "best_epoch_{:s}.pt".format(finetune_config["model"]),
            )
        )
    )
    # print(model)
    mean_bp_loss, evaluate_results, _time = evaluate(
        model=model,
        data_type="test",
        data_loader=test_loaders,
        config=config,
        logger=logger,
        writer=writer,
    )
    total_test_time += _time
    # if mean_reg_loss <= best_reg_losses['test']:
    #     best_reg_losses['test'] = mean_reg_loss
    # best_reg_epochs['test'] =
    logger.info(
        "data_type: {:<5s}\tbest mean loss: {:.3f}".format("test", mean_bp_loss)
    )
    with open(
            os.path.join(
                save_model_dir,
                "%s_%s_%s_pre_trained.json"
                % (
                        finetune_config["model"],
                        "best_test",
                        finetune_config["dataset"],
                ),
            ),
            "w",
    ) as f:
        json.dump(evaluate_results, f)

    return evaluate_results, total_test_time


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
    with open(os.path.join(save_model_dir, "train_config.json"), "w") as f:
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
        "cuda:%d" % finetune_config["gpu_id"] if (
                finetune_config["gpu_id"] != -1 and finetune_config["cuda"] == True) else "cpu"
    )
    if finetune_config["gpu_id"] != -1 and finetune_config["cuda"] == True:
        torch.cuda.set_device(device)

        # load data
        # os.makedirs(train_config["save_data_dir"], exist_ok=True)
    # decompose the query
    train_loader, val_loader, test_loader = importance_graph_load(
        finetune_config['dataset'], batch_size=finetune_config["batch_size"],train_ratio=0.8,val_ratio=0.1
    )
    # config['init_g_dim'] = graph.x.size(1)
    # train_config.update({'init_g_dim': graph.x.size(1)})
    # construct the model
    finetune_config["init_g_dim"] = next(iter(train_loader)).x.shape[1]
    if finetune_config["task"] == "importance":
        model = ImportancePipeline(
            input_dim=finetune_config["init_g_dim"], layer_num=finetune_config['graph_num_layers'],
            pre_train_path=os.path.join("..", finetune_config['save_model_dir'],
                                        "best_epoch_" + finetune_config['model'] + ".pt")
        )
    elif finetune_config['task'] == "localcounting":
        model = Gphormer(finetune_config['graph_num_layers'],
                         finetune_config["init_g_dim"],
                         finetune_config["num_g_hid"],
                         finetune_config['init_e_dim'],
                         finetune_config['num_e_hid'],
                         output_dim=finetune_config["out_g_ch"],
                         pretrain=True)
    else:
        raise NotImplementedError(
            "Currently, the %s model is not supported" % (finetune_config["model"])
        )
    # model = torch.compile(model)
    logger.info(model)
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
        evaluate_results, total_test_time = test(
            save_model_dir, test_loader, finetune_config, logger, writer
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
            data_loader=train_loader,
            device=device,
            config=finetune_config,
            epoch=epoch,
            logger=logger,
            writer=writer,
            bottleneck=False,
        )
        total_train_time += _time
        if scheduler and (epoch + 1) % finetune_config["decay_patience"] == 0:
            scheduler.step()
            # torch.save(model.state_dict(), os.path.join(save_model_dir, 'epoch%d.pt' % (epoch)))
        mean_bp_loss, evaluate_results, total_time = evaluate(
            model=model,
            data_type="val",
            data_loader=val_loader,
            config=finetune_config,
            logger=logger,
            writer=writer,
        )
        if writer:
            writer.add_scalar(
                "%s/BP-%s-epoch" % ("val", finetune_config["bp_loss"]), mean_bp_loss, epoch
            )
            total_dev_time += total_time
            # cur_reg_loss[loader_idx] = mean_reg_loss
            # flag = True
            # for key1, key2 in zip(cur_reg_loss.keys(), best_reg_losses.keys()):
            #     if cur_reg_loss[key1] > best_reg_losses[key2]:
            #         flag = False
            # if flag:
            #     for key1, key2 in zip(cur_reg_loss.keys(), best_reg_losses.keys()):
            #         best_reg_losses[key2] = cur_reg_loss[key1]
            #     best_reg_epochs['val'] = epoch
        err = best_bp_losses - mean_bp_loss
        if err > 1e-5:
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
                    "best_epoch_{:s}.pt".format(finetune_config["model"]),
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
    evaluate_results, total_test_time = test(
        save_model_dir, test_loader, finetune_config, logger, writer
    )
    logger.info(
        "train time: {:.3f}, train time per epoch :{:.3f}, test time: {:.3f}, all time: {:.3f}".format(
            total_train_time,
            total_train_time / finetune_config["epochs"],
            total_test_time,
            total_train_time + total_dev_time + total_test_time,
        )
    )
