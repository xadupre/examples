import argparse
import json
import logging
import os
import pickle
import pprint
import time
import sys
import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas
from tqdm import tqdm
from datasets import list_datasets, load_dataset
import deepspeed
import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset, IterableDataset
from transformers import GPT2Tokenizer, GPT2Model
from transformers.deepspeed import HfDeepSpeedConfig

WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))


class GPT2Loss(torch.nn.Module):
    def __init__(self, model, input_dim: int, num_classes: int):
        super().__init__()
        # input_dim: 768 or 1280
        self.gpt2model = model
        self.fc1 = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x, y):
        gpt_out = self.gpt2model(x)
        linear_output = self.fc1(gpt_out.last_hidden_state)
        out = linear_output.sum(axis=2)
        return torch.abs(out - y).sum()


class CustomDataset(Dataset):
    def __init__(self, encoded_tensors, labels, device, local_rank=-1):
        if local_rank == -1:
            self.encoded_tensors = [e.to(device) for e in encoded_tensors]
            self.labels = labels.to(torch.float32).to(device)
        else:
            self.encoded_tensors = [e.to(device) for e in encoded_tensors]
            self.labels = labels.to(torch.float32).to(device)[local_rank::WORLD_SIZE]
        self.transform = None
        self.target_transform = None
        self.local_rank = local_rank

    def __len__(self):
        return len(self.encoded_tensors)

    def __getitem__(self, idx):
        return self.encoded_tensors[idx], self.labels[idx]


def _main_deepspeed(model_name, cmd_args):
    logger = logging.getLogger("DeepSpeed")
    logger.setLevel(logging.INFO)

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

    if train_batch_size <= 1:
        raise ValueError(
            f"train_batch_size={train_batch_size} must be > 1\n{pprint.pformat(ds_config)}"
        )
    if local_rank >= WORLD_SIZE or local_rank < 0:
        raise ValueError(
            f"local_rank={local_rank} must be >= 0 and < WORLD_SIZE={WORLD_SIZE}"
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
    name = f"encoded_tensors-{model_name}.pkl"
    with open(name, "rb") as f:
        [encoded_tensors, labels] = pickle.load(f)
    labels = labels.reshape((-1, 1, 5)).to(torch.float32)
    model = GPT2Loss(model, input_dim, 5)

    # onnxruntime
    # from onnxruntime.training.ortmodule import ORTModule
    # model = ORTModule(model)

    # data
    device = torch.device("cuda:%d" % local_rank)
    ds = CustomDataset(encoded_tensors, labels, device, local_rank)
    my_dataloader = DataLoader(ds)

    # deepspeed initialization
    model, optimizer, _, __ = deepspeed.initialize(
        args=cmd_args,
        model=model,
        model_parameters=model.parameters(),
        # config_params=ds_config,
        # optimizer=optimizer,
    )
    print(
        f"[{WORLD_SIZE}({cmd_args.local_rank})-trainds] {type(model)}, {type(optimizer)}"
    )
    print(
        f"[{WORLD_SIZE}({cmd_args.local_rank})-partition_count] {optimizer.partition_count!r}"
    )
    print(
        f"[{WORLD_SIZE}({cmd_args.local_rank})-contiguous_gradients] {optimizer.contiguous_gradients!r}"
    )
    ngr = len(optimizer.optimizer.param_groups)
    for igr, gr in enumerate(optimizer.optimizer.param_groups):
        total = sum(np.prod(g[0].shape) for g in gr["params"])
        print(f"[{cmd_args.local_rank}/{ngr}-group{igr}] {list(gr)} - total={total}")
    if hasattr(optimizer, "params_in_partition"):
        ngr = len(optimizer.params_in_partition)
        for igr, gr in enumerate(optimizer.params_in_partition):
            total = sum(np.prod(g.data.shape) for g in gr)
            print(f"[{cmd_args.local_rank}/{ngr}-in-group{igr}] total={total}")
        print(f"[{cmd_args.local_rank}] --")

    # training
    times = []
    model.train()

    for epoch_num in range(10 + 1):
        begin = time.perf_counter()

        for i, (x, y) in enumerate(my_dataloader):

            # optimizer.zero_grad()
            batch_loss = model(x["input_ids"], y)

            model.backward(batch_loss)

            model.step()

        end = time.perf_counter() - begin
        if epoch_num > 0:
            times.append(end)
        print(
            f"local_rank={cmd_args.local_rank}/{WORLD_SIZE}, epoch_num={epoch_num}, time={end}"
        )
    return times


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
