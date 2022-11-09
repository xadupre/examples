import argparse
import json
import logging
import os
import pickle
import pprint
import sys
import time
import warnings

import deepspeed
import numpy as np
import pandas
import torch
from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer

from gpt2_loss import GPT2Loss
from data_helper import CustomDataset, DataLoader
from exp_config import get_deepspeed_config

warnings.simplefilter("ignore")


WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))


def startup(model_name, local_rank=-1):
    if model_name == "gpt2":
        input_dim = 768
    elif model_name == "gpt2-large":
        input_dim = 1280
    else:
        raise ValueError(f"Unexpected model name {model_name!r}.")
    print(f"[{local_rank}-start] input_dim={input_dim}")
    print(f"[{local_rank}-start-load-model]")

    model = GPT2Model.from_pretrained(model_name)
    print(f"[{local_rank}-start-run-tokenizer]")

    print(f"[{local_rank}-start-done]")
    name = f"data/encoded_tensors-{model_name}.pkl"
    if not os.path.exists(name):
        print(f"[{local_rank}-start-load-tokenizer]")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        print(f"[{local_rank}-start-loading-data]")
        df = pandas.read_csv("data/train.csv")
        print(
            f"[{local_rank}-start-done]",
            df.shape,
            df.columns,
        )
        labels = torch.from_numpy(pandas.get_dummies(df.category).values)
        model_input = df.text
        encoded_tensors = []
        for t in tqdm(model_input):
            # tens = torch.tensor([tokenizer.encode(t, add_special_tokens=True)])
            encoded_input = tokenizer(t, return_tensors="pt")
            for k in encoded_input:
                tens = encoded_input[k]
                if tens.shape[-1] > 1024:
                    tens = tens[:, :1024]
                    encoded_input[k] = tens
            encoded_tensors.append(encoded_input)
        print(
            f"[{local_rank}-start-pickle]",
            len(encoded_tensors),
            name,
        )
        with open(name, "wb") as f:
            pickle.dump([encoded_tensors, labels], f)
    else:
        print(f"[{local_rank}-start-unpickle]", name)
        with open(name, "rb") as f:
            [encoded_tensors, labels] = pickle.load(f)
    print(f"[{local_rank}-start-done]", len(encoded_tensors))

    if not os.path.exists(f"onnx/{model_name}.onnx"):
        if not os.path.exists("onnx"):
            os.mkdir("onnx")

        print(f"[{local_rank}-start-convert-onnx]")
        torch.onnx.export(
            model,
            encoded_tensors[0],
            f"onnx/{model_name}.onnx",
            verbose=False,
            input_names=["X"],
            output_names=["Y"],
            opset_version=15,
            dynamic_axes={"X": [0, 1]},
        )
        print(f"[{local_rank}-start-done]")

    model = GPT2Loss(model, input_dim, 5)
    return model, encoded_tensors, labels, f"onnx/{model_name}.onnx"


def train(
    model,
    model_name,
    epochs,
    encoded_tensors,
    labels,
    device,
    scenario=None,
    train_batch_size=None,
):
    use_ort = scenario is not None and "ort" in scenario.split("-")

    local_rank = 0

    print(
        f"[{local_rank}-train-scenario] scenario={scenario!r} use_ort={use_ort} model_name={model_name!r}"
    )
    print(f"[{local_rank}-train-device]", device, len(encoded_tensors))
    print(f"[{local_rank}-train-model-gpu]")
    print(f"[{local_rank}-train-dataset]")

    ds = CustomDataset(
        encoded_tensors,
        labels.reshape(
            (-1, 1, 5),
        ).to(torch.float32),
        device,
        -1,
    )
    my_dataloader = DataLoader(ds)
    print(
        f"[{local_rank}-train-done]",
        type(encoded_tensors[0]),
        labels.shape,
    )
    print(f"[{local_rank}-train-done]")
    params = list(model.parameters())
    print(
        f"[{local_rank}-params#] {len(params)} - total {sum([np.prod(p.shape) for p in params])}",
    )
    times = []

    if use_ort:
        from onnxruntime.training.ortmodule import ORTModule

        print(f"[{local_rank}-train-ORTModule]")
        model = ORTModule(model)
        print(f"[{local_rank}-train-done]")

    if scenario not in ("torch", "ort-torch", "torch-ort", "ort"):
        use_deepspeed = True
        logger = logging.getLogger("DeepSpeed")
        logger.setLevel(logging.WARNING)

        import deepspeed
        from deepspeed.runtime.zero.stage3 import (
            estimate_zero3_model_states_mem_needs_all_live,
        )
        from deepspeed.runtime.zero.stage_1_and_2 import (
            estimate_zero2_model_states_mem_needs_all_live,
        )

        print("------------------------------------")
        print(
            estimate_zero2_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=4,
                num_nodes=1,
            ),
        )
        print(
            estimate_zero3_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=4,
                num_nodes=1,
            ),
        )
        print("------------------------------------")

        # deepspeed.init_distributed()
        ds_config = get_deepspeed_config(WORLD_SIZE, scenario, train_batch_size)
        if not os.path.exists("config"):
            os.mkdir("config")
        with open(f"config/last-config-{scenario}.json", "w") as f:
            json.dump(ds_config, f, sort_keys=True, indent=4)
        train_batch_size = ds_config["train_batch_size"]
        pprint.pprint(ds_config)

        model, optimizer, _, __ = deepspeed.initialize(
            config_params=ds_config,
            model=model,
            model_parameters=model.parameters(),
            # optimizer=optimizer,
        )
        assert optimizer is not None
        object_step = model

        print(
            f"[{local_rank}-train-deepspeed] is_gradient_accumulation_boundary={model.is_gradient_accumulation_boundary()}",
        )
        print(
            f"[{local_rank}-train-deepspeed] custom_loss_scaler={getattr(optimizer, 'custom_loss_scaler', None)}",
        )

        model = model  # Wrapper(model)

    elif "torch" in scenario.split("-"):
        use_deepspeed = False
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        model = model.to(device)
        train_batch_size = 8

    else:
        raise ValueError(f"Unexpected value for scenario={scenario!r}.")

    print(f"[{local_rank}-train-type] {type(my_dataloader)}")
    print(f"[{local_rank}-train-type] {type(model)}")
    print(f"[{local_rank}-train-type] {type(optimizer)}")
    model.train()
    if False:
        for k, v in sorted(model.__dict__.items()):
            if v is not None:
                if isinstance(v, (int, float, bool, str)):
                    print(f"[{local_rank}-train] model.{k}={v!r}")
                elif isinstance(v, (list, set, dict)):
                    print(f"[{local_rank}-train] model.{k}: {len(v)}:{type(v)}")
                else:
                    print(f"[{local_rank}-train] model.{k}: {type(v)}")
        for k, v in sorted(optimizer.__dict__.items()):
            if v is not None:
                if isinstance(v, (int, float, bool, str)):
                    print(f"[{local_rank}-train] optimizer.{k}={v!r}")
                elif isinstance(v, (list, set, dict)):
                    print(
                        f"[{local_rank}-train] optimizer.{k}: {len(v)}:{type(v)}",
                    )
                else:
                    print(f"[{local_rank}-train] optimizer.{k}: {type(v)}")

    for epoch_num in tqdm(range(epochs + 1)):
        total_loss_train = 0
        begin = time.perf_counter()

        if not use_deepspeed:
            optimizer.zero_grad()

        for i, (x, y) in enumerate(my_dataloader):

            # optimizer.zero_grad()
            batch_loss = model(x["input_ids"], y)
            total_loss_train += batch_loss.to(float).item()

            if use_deepspeed:
                model.backward(batch_loss)

                if False and getattr(
                    optimizer,
                    "overflow",
                    False,
                ):  # only for Zero
                    lo = batch_loss.detach().float()
                    raise RuntimeError(
                        f"Overflow after step(), offload={optimizer.cpu_offload}, "
                        f"len(optimizer.bit16_groups)={len(optimizer.bit16_groups)}, "
                        f"len(optimizer.averaged_gradients)={len(optimizer.averaged_gradients)}, "
                        f"partition_gradients={optimizer.partition_gradients}, "
                        f"loss={lo!r}, total_loss_train={total_loss_train!r}, i={i}",
                    )

                model.step()
            else:
                batch_loss.backward()
                if i % train_batch_size == train_batch_size - 1:
                    optimizer.step()  # optimizer.step() or model.step() for deepspeed
                    optimizer.zero_grad()

        end = time.perf_counter() - begin
        if epoch_num > 0:
            times.append(end)
        print(
            f"epoch_num={epoch_num}, total_loss_train={total_loss_train}, time={end}",
        )
    return times, train_batch_size, len(my_dataloader)


def main(
    epochs=10,
    n_obs=100,
    scenario="ds0",
    train_batch_size=None,
    model_name="gpt2",
):
    """
    Trains a dummy model based on GPT-2. The model has no real meaning
    but to measure the training of a GPT-2 model.

    :param epochs: number of training iterations (+ 1),
        first one is not included in the average time
    :param n_obs: number of observations in the training set
    :param scenario: training to measure. See below.
    :param train_batch_size: training batch size
    :param model_name: model name `gpt2`, `gpt2-large`

    About scenario:

    * `torch`: pytorch training
    * `'ort'`: pytorch + ORTModule
    * `'ds0'`: pytorch + deepspeed, no zero
    * `'ds1'`: pytorch + deepspeed stage zero 1
    * `'ds2'`: pytorch + deepspeed stage zero 2
    * `'ds3'`: pytorch + deepspeed stage zero 3

    Building instructions for onnxruntime:

    ::

        python3 ./tools/ci_build/build.py
                --config Release
                --skip_tests
                --build_wheel
                --parallel
                --build_dir
                ./build/linux_cuda
                --build_shared_lib
                --use_cuda
                --cuda_home /usr/local/cuda-${CUDA_VERSION}/
                --cudnn_home /usr/local/cuda-${CUDA_VERSION}/
                --cuda_version=${CUDA_VERSION}
                --enable_training
                --enable_training_ops
                --enable_training_torch_interop
        python3 -m torch_ort.configure
    """
    scenario = "-".join(sorted(scenario.lower().split('-')))
    local_rank = -1
    print(
        f"[{local_rank}-train] get_device_capability()={torch.cuda.get_device_capability()}",
    )
    print(f"[{local_rank}-train] get_arch_list()={torch.cuda.get_arch_list()}")
    print(
        f"[{local_rank}-train] get_device_properties(...)={torch.cuda.get_device_properties(torch.device('cuda'))}",
    )
    print(
        f"[{local_rank}-train] epochs={epochs}, n_obs={n_obs}, scenario={scenario!r}, model_name={model_name!r}",
    )
    model, encoded_tensors, labels, model_name = startup(model_name)

    epochs = 10
    print()
    print(f"[{local_rank}-train]")
    device = torch.device("cuda:%d" % max(local_rank, 0))
    times, train_batch_size, N = train(
        model,
        os.path.splitext(os.path.split(model_name)[-1])[0],
        epochs,
        encoded_tensors[:n_obs],
        labels,
        device,
        scenario=scenario,
        train_batch_size=train_batch_size,
    )
    if not os.path.exists("results.txt"):
        with open("results.txt", "w") as f:
            f.write("program,scenario,N,train_batch_size,average,details\n")
    with open("results.txt", "a") as f:
        text = ":".join(map(str, times))
        f.write(
            f"perf.py,{scenario},{N},{train_batch_size},{sum(times) / len(times)},{text}\n"
        )
    print(
        f"[times] scenario={scenario}, N={N}, train_batch_size={train_batch_size}, average={sum(times) / len(times)}, details={times}"
    )
    print("[done]")


if __name__ == "__main__":
    # nvitop -m
    if any(map(lambda x: x.startswith("--local_rank"), sys.argv)):
        raise RuntimeError(f"Use distri.py.")
    elif any(map(lambda x: x.startswith("--scenario"), sys.argv)):
        print(f"[WORLD_SIZE={WORLD_SIZE}]")
        import fire

        fire.Fire(main)
    else:
        print(f"[WORLD_SIZE={WORLD_SIZE}-default]")
        main()
