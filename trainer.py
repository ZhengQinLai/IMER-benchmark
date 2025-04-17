import sys
import logging
import torch
import os
import numpy as np
import time

from utils import factory
from utils.data_manager import IncrementalIndexGenerator, IncrementalDataloaderGenerator



def train(args):
    curve = _train(args)
    return curve


def _train(args):
    logs_name = f"log/logs_{args['log_file']}/{args['model_name']}/{args['split_mode']}"
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    time_ = str(int(time.time()))
    logfilename = f"log/logs_{args['log_file']}/{args['model_name']}/{args['split_mode']}/{args['seed']}_{args['backbone_type']}_{time_}"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    index_gen = IncrementalIndexGenerator(split_flag=args["split_mode"], up=args.get('up', False))
    dataloader_gen = IncrementalDataloaderGenerator(batch_size=args['batch_size'], shuffle='True', img_size=args['img_size'])
    cnn_fold = []
    classifier_fold = []
    all_y_pred, all_y_true = [[],[],[],[]], [[],[],[],[]]
    for fold in range(args['K']):
        model = factory.get_model(args["model_name"], args)
        model.Incregen = dataloader_gen
        cnn_list = []
        classifier_list = []
        for session in range(len(np.unique(list(index_gen.iData.d2i.values())))):
            train_idx, test_idx = index_gen.get_split(session, fold)
            model.cur_split = train_idx, test_idx
            train_loader, test_loader = dataloader_gen.get_dataloader(train_idx, test_idx, extra_data=model._get_memory())
            print(len(train_loader.dataset),len(test_loader.dataset))
            model.incremental_train(train_loader, test_loader)
            cnn_accy, classifier_accy, y_pred, y_true = model.eval_task(test_loader)
            # if session > 0:
            #     casme2_list.append(classifier_accy['grouped']['05-09'])
            # Since 'up' is never True in configs, this is always called
            model.after_task()
            if classifier_accy is not None:
                classifier_list.append(classifier_accy['top1'])
            else:
                cnn_list.append(cnn_accy['top1'])
            all_y_pred[session].append(y_pred)
            all_y_true[session].append(y_true)
        cnn_fold.append(cnn_list)
        if len(classifier_list) == 0:
            pass
        else:
            classifier_fold.append(classifier_list)
    for session in range(len(np.unique(list(index_gen.iData.d2i.values())))):
        all_y_pred[session] = np.concatenate(all_y_pred[session])
        all_y_true[session] = np.concatenate(all_y_true[session])


    # 确保目录存在
    os.makedirs(logs_name, exist_ok=True)

    print(all_y_pred,all_y_true)
    if len(classifier_fold) != 0:
        print(classifier_fold)
        return classifier_fold
    else:
        print(cnn_fold)
        return cnn_fold


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus

def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
