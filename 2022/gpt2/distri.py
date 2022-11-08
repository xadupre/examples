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

    # data
    device = torch.device("cuda:%d" % local_rank)
    ds = CustomDataset(encoded_tensors, labels, device, local_rank)
    my_dataloader = DataLoader(ds)

    # onnxruntime
    if True:
        from onnxruntime.training.ortmodule import ORTModule

        model = ORTModule(model)
        # one iteration to initialize the module
        print("INITIALIZATION ORT BEGIN")
        for x, y in my_dataloader:
            # it cannot run on GPU since the model may not hold in CPU memory
            batch_loss = model(x["input_ids"].to("cpu"), y.to("cpu"))
            break
        print("INITIALIZATION ORT DONE")
        # It works but it fails later during training. It seems that DLPack protocol is used by onnxruntime
        # to get some data and it tries to delete the tensor but fails (maybe the destructor is null,
        # maybe the tensor should remain). Stage 3 is expected to keep ownership.
        # 2022-11-08 15:10:48.214031399 [E:onnxruntime:, orttraining_partial_executor.cc:368 Execute] Non-zero status code returned while running ATen node. Name:'/_original_module/gpt2model/wpe/ATen' Status Message: The specified pointer resides on host memory and is not registered with any CUDA device.
        # Exception raised from getDeviceFromPtr at ../aten/src/ATen/cuda/CUDADevice.h:17 (most recent call first):
        # frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x3e (0x7fd05c06a86e in site-packages/torch/lib/libc10.so)
        # frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, char const*) + 0x60 (0x7fd05c035469 in site-packages/torch/lib/libc10.so)
        # frame #2: <unknown function> + 0x16054f (0x7fcff92c854f in site-packages/torch/lib/libtorch_cuda_cpp.so)
        # frame #3: at::TensorMaker::make_tensor() + 0xa30 (0x7fcfe04ce790 in site-packages/torch/lib/libtorch_cpu.so)
        # frame #4: at::fromDLPack(DLManagedTensor const*) + 0x696 (0x7fcfdfa39e36 in site-packages/torch/lib/libtorch_cpu.so)
        # frame #5: ATenOperator::ToIValueArgument(DLManagedTensor const*, unsigned long) const + 0x9c (0x7fcf7c068bac in onnxruntime/training/ortmodule/torch_cpp_extensions/aten_op_executor.cpython-38-x86_64-linux-gnu.so)
        # frame #6: ExecuteATenOperator(char const*, char const*, unsigned long, DLManagedTensor**, unsigned long, DLManagedTensor**) + 0x135 (0x7fcf7c0623e5 in onnxruntime/training/ortmodule/torch_cpp_extensions/aten_op_executor.cpython-38-x86_64-linux-gnu.so)
        # frame #7: <unknown function> + 0x8fdbee (0x7fcfa1618bee in onnxruntime/capi/onnxruntime_pybind11_

    # deepspeed initialization
    model, optimizer, _, __ = deepspeed.initialize(
        args=cmd_args,
        model=model,
        model_parameters=model.parameters(),
        # config_params=ds_config,
        # optimizer=optimizer,
    )

    if False:
        from onnxruntime.training.ortmodule import ORTModule

        model = ORTModule(model)

    print(
        f"[{WORLD_SIZE}({cmd_args.local_rank})-trainds] N={len(my_dataloader)}, {type(model)}, {type(optimizer)}"
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
    model.train()

    begin = time.perf_counter()

    for i, (x, y) in enumerate(my_dataloader):

        # optimizer.zero_grad()
        batch_loss = model(x["input_ids"], y)

        model.backward(batch_loss)

        model.step()

        if i >= 2 * train_batch_size:
            break

    end = time.perf_counter() - begin
    print(f"END local_rank={cmd_args.local_rank}/{WORLD_SIZE}, time={end}, N={i}")


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
_main_deepspeed("gpt2", cmd_args)
