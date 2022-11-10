"""
::

    deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config last_config.json

"""
import argparse
import json
import logging
import os
import pickle
import pprint
import sys
import time
import warnings

warnings.simplefilter("ignore")

import deepspeed
import numpy as np
import pandas
import torch
import torch.distributed as dist
from datasets import list_datasets, load_dataset
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset, TensorDataset
from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer
from transformers.deepspeed import HfDeepSpeedConfig

from data_helper import CustomDataset
from exp_stats import get_stats
from gpt2_loss import GPT2Loss

WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))


def _main_deepspeed(model_name, cmd_args):
    logger = logging.getLogger("DeepSpeed")
    logger.setLevel(logging.WARNING)

    if model_name == "gpt2":
        input_dim = 768
    elif model_name == "gpt2-large":
        input_dim = 1280
    else:
        raise ValueError(f"Unexpected model name {model_name!r}.")

    with open(cmd_args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    train_batch_size = ds_config["train_batch_size"]
    local_rank = int(cmd_args.local_rank)
    scenario = os.path.split(os.path.splitext(cmd_args.deepspeed_config)[0])[-1]
    scenario = scenario.replace("last-config-", "")
    log_name = f"log/log-distri-{model_name}-{local_rank}.txt"

    def _log_(d):
        d["scenario"] = scenario
        with open(log_name, "a") as f:
            f.write(str(d))
            f.write("\n")

    if train_batch_size <= 1:
        raise ValueError(
            f"train_batch_size={train_batch_size} must be > 1\n{pprint.pformat(ds_config)}",
        )
    if local_rank >= WORLD_SIZE or local_rank < 0:
        raise ValueError(
            f"local_rank={local_rank} must be >= 0 and < WORLD_SIZE={WORLD_SIZE}",
        )

    torch.cuda.set_device(local_rank)
    # deepspeed.init_distributed()

    # loading the model and the data
    if model_name == "gpt2":
        input_dim = 768
    elif model_name == "gpt2-large":
        input_dim = 1280
    else:
        raise ValueError(f"Unexpected model name {model_name!r}.")
    model = GPT2Model.from_pretrained(model_name)
    name = f"data/encoded_tensors-{model_name}.pkl"
    with open(name, "rb") as f:
        [encoded_tensors, labels] = pickle.load(f)
    labels = labels.reshape((-1, 1, 5)).to(torch.float32)
    model = GPT2Loss(model, input_dim, 5)

    # data
    device = torch.device("cuda:%d" % local_rank)
    ds = CustomDataset(WORLD_SIZE, encoded_tensors, labels, device, local_rank)
    my_dataloader = DataLoader(ds)

    # onnxruntime
    if "-ort" in cmd_args.deepspeed_config:
        from onnxruntime.training.ortmodule import ORTModule

        model = ORTModule(model)

    # deepspeed initialization
    model, optimizer, _, __ = deepspeed.initialize(
        args=cmd_args,
        model=model,
        model_parameters=model.parameters(),
        # config_params=ds_config,
        # optimizer=optimizer,
    )

    print(
        f"[{WORLD_SIZE}({cmd_args.local_rank})-trainds] N={len(my_dataloader)}, {type(model)}, {type(optimizer)}",
    )
    if hasattr(optimizer, "partition_count"):
        print(
            f"[{WORLD_SIZE}({cmd_args.local_rank})-partition_count] {optimizer.partition_count!r}",
        )
    if hasattr(optimizer, "contiguous_gradients"):
        print(
            f"[{WORLD_SIZE}({cmd_args.local_rank})-contiguous_gradients] {optimizer.contiguous_gradients!r}",
        )
    if False:
        ngr = len(optimizer.optimizer.param_groups)
        for igr, gr in enumerate(optimizer.optimizer.param_groups):
            total = sum(np.prod(g[0].shape) for g in gr["params"])
            print(
                f"[{cmd_args.local_rank}/{ngr}-group{igr}] {list(gr)} - total={total}"
            )
        if hasattr(optimizer, "params_in_partition"):
            ngr = len(optimizer.params_in_partition)
            for igr, gr in enumerate(optimizer.params_in_partition):
                total = sum(np.prod(g.data.shape) for g in gr)
                print(f"[{cmd_args.local_rank}/{ngr}-in-group{igr}] total={total}")
            print(f"[{cmd_args.local_rank}] --")

    # training
    kinds = {}
    model.train()

    begin = time.perf_counter()

    for i, (x, y) in enumerate(my_dataloader):
        if i >= 2 * train_batch_size:
            break

        # optimizer.zero_grad()
        batch_loss = model(x["input_ids"], y)

        model.backward(batch_loss)

        model.step()

        if i == 0:
            kinds = get_stats(local_rank, model)
            # skipping the first iteration
            begin = time.perf_counter()

    end = time.perf_counter() - begin
    print(f"END local_rank={cmd_args.local_rank}/{WORLD_SIZE}, time={end}, N={i}")
    info = dict(WORLD_SIZE=WORLD_SIZE, N=i, time=end, time_per_img=end / (i - 1))
    info.update(kinds)
    _log_(info)


# nvitop -m
parser = argparse.ArgumentParser(description="My training script.")
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="local rank passed from distributed launcher",
)

# Include DeepSpeed configuration arguments
parser = deepspeed.add_config_arguments(parser)
cmd_args = parser.parse_args()
print(f"[WORLD_SIZE={WORLD_SIZE},LOCAL_RANK={cmd_args.local_rank}]")
print(cmd_args)
_main_deepspeed("gpt2-large", cmd_args)
