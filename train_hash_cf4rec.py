import argparse
import os


import random

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import now

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from torch.distributed.elastic.multiprocessing.errors import *




def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", default='/home/hexngroup/ym689/BinLLM/train_configs/hash_CF_binllm_movies.yaml', help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--local-rank")
    args = parser.parse_args()
    

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "rec_runner_base"))

    return runner_cls

@record
def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    args = parse_args()
    cfg = Config(args)
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    # cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    # cfg.model_cfg.get("user_num", "default")
    data_name = list(datasets.keys())[0]
    # data_dir = "/home/sist/zyang/LLM/datasets/ml-1m/"
    try: #  movie
        data_dir = cfg.datasets_cfg.movie_ood.path
    except: # amazon
        data_dir = cfg.datasets_cfg.amazon_ood.path
    print("data dir:", data_dir)
    # data_dir = "/data/zyang/datasets/ml-1m/"
    train_ = pd.read_pickle(data_dir+"train_ood2.pkl")
    valid_ = pd.read_pickle(data_dir+"valid_ood2.pkl")
    test_ = pd.read_pickle(data_dir+"test_ood2.pkl")
    
    #amazon book
    user_num = 22967
    item_num = 34154
    
    #user_num = max(train_.uid.max(),valid_.uid.max(),test_.uid.max())+1
    #item_num = max(train_.iid.max(),valid_.iid.max(),test_.iid.max())+1

    cfg.model_cfg.rec_config.user_num = int(user_num) #int(datasets[data_name]['train'].user_num)  #cfg.model_cfg.get("user_num",)
    cfg.model_cfg.rec_config.item_num = int(item_num) #int(datasets[data_name]['train'].item_num) #cfg.model_cfg.get("item_num", datasets[data_name]['train'].item_num)
    cfg.pretty_print()
    
    #`print(args.local_rank)
    model = task.build_model(cfg)
    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
