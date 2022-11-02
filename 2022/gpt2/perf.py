import argparse
import logging
import os
import pickle
import pprint
import time
import sys
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
_origin = time.perf_counter()


def get_deepspeed_config(scenario_full, train_batch_size=10):
    gradient_acc_step = 1
    micro_batch_per_gpu = train_batch_size // (gradient_acc_step * WORLD_SIZE)
    if micro_batch_per_gpu * gradient_acc_step * WORLD_SIZE != train_batch_size:
        micro_batch_per_gpu += 1
        train_batch_size = micro_batch_per_gpu * gradient_acc_step * WORLD_SIZE
    scenario = scenario_full.split("-")[0]
    if scenario == "ds0":
        ds_config = {
            "zero_optimization": {
                "allgather_bucket_size": 5e8,
                "contiguous_gradients": False,
                "stage": 0,
                "overlap_comm": True,
            },
        }
    elif scenario == "ds1":
        ds_config = {
            "zero_optimization": {
                "allgather_bucket_size": 2e8,
                "allgather_partitions": True,
                "contiguous_gradients": False,
                "overlap_comm": True,
                "reduce_bucket_size": 2e8,
                "reduce_scatter": True,
                "stage": 1,
            },
            "zero_allow_untested_optimizer": True,
        }
    elif scenario == "ds2":
        ds_config = {
            "zero_optimization": {
                "allgather_bucket_size": 2e8,
                "allgather_partitions": True,
                "contiguous_gradients": False,
                # "offload_optimizer": { "device": "cpu", },
                "reduce_bucket_size": 2e8,
                "reduce_scatter": True,
                "overlap_comm": True,
                "stage": 2,
            },
            "zero_allow_untested_optimizer": True,
        }
    elif scenario == "ds3":
        ds_config = {
            "zero_optimization": {
                "allgather_bucket_size": 2e8,
                "allgather_partitions": True,
                "contiguous_gradients": False,
                "offload_param": {"device": "cpu", "pin_memory": True},
                "overlap_comm": True,
                "reduce_bucket_size": 2e8,
                "reduce_scatter": True,
                "stage": 3,
            },
            "zero_allow_untested_optimizer": True,
        }
    else:
        raise ValueError(f"Unknwon scenario {scenario!r}.")

    optimizer_config = {
        "type": "AdamW",
        "params": {
            "betas": [0.8, 0.999],
            "eps": 1e-8,
            "lr": 0.0001,
            "weight_decay": 3e-7,
        },
    }

    scheduler_config = {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
        },
    }

    ds_config.update(
        {
            # "gradient_clipping": 1.0,
            "micro_batch_per_gpu": micro_batch_per_gpu,
            "optimizer": optimizer_config,
            # "scheduler": scheduler_config,
            "train_batch_size": train_batch_size,
            "wall_clock_breakdown": False,
        }
    )
    if "-16" in scenario_full:
        ds_config.update(
            {
                "bf16": {"enabled": False},
                "fp16": {
                    "enabled": True,
                    "loss_scale": 0,
                    "min_loss_scale": 0,
                },
            }
        )
    elif "-b16" in scenario_full:
        ds_config.update(
            {
                "bf16": {"enabled": True},
                "fp16": {"enabled": False},
            }
        )
    else:
        ds_config.update(
            {
                "bf16": {"enabled": False},
                "fp16": {"enabled": False},
            }
        )

    return ds_config


def return_time():
    global _origin
    dt = time.perf_counter()
    res = dt - _origin
    _origin = dt
    return res


def load_dataset_train(path="data/wiki.train.tokens"):
    # see https://github.com/knagrecha/hydra/blob/main/examples/utils.py
    raw_text = ""
    with open(path, "r") as fp:
        raw_text += fp.read()
    raw_text += "<|endoftext|>"
    tokens = np.stack(tokenizer.encode(raw_text))
    return tokens


def get_data_loader_train(batch_size, context_length=512):
    data = lazy_load_train()[0]
    # Chunk data by context_length
    ds = Subset(
        data,
        [
            slice(i, i + context_length)
            for i in range(0, len(data) - (len(data) % context_length), context_length)
        ],
    )
    data_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    return data_loader


def startup(model_name, local_rank=-1):
    print(f"[{local_rank}-start]")
    print(f"[{local_rank}-start-load-model]", return_time())
    model = GPT2Model.from_pretrained(model_name)
    print(f"[{local_rank}-start-run-tokenizer]", return_time())

    print(f"[{local_rank}-start-done]", return_time())
    name = f"encoded_tensors-{model_name}.pkl"
    if not os.path.exists(name):
        # print(f"[{local_rank}-start-load-datasets]", return_time())
        # data = load_dataset("wikitext", "wikitext-2-raw-v1")
        print(f"[{local_rank}-start-load-tokenizer]", return_time())
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        print(f"[{local_rank}-start-loading-data]", return_time())
        df = pandas.read_csv("data/train.csv")
        print(f"[{local_rank}-start-done]", df.shape, df.columns, return_time())
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
        print(f"[{local_rank}-start-pickle]", len(encoded_tensors), return_time(), name)
        with open(name, "wb") as f:
            pickle.dump([encoded_tensors, labels], f)
    else:
        print(f"[{local_rank}-start-unpickle]", return_time(), name)
        with open(name, "rb") as f:
            [encoded_tensors, labels] = pickle.load(f)
    print(f"[{local_rank}-start-done]", len(encoded_tensors), return_time())

    if not os.path.exists(f"onnx/{model_name}.onnx"):
        if not os.path.exists("onnx"):
            os.mkdir("onnx")
        print(f"[{local_rank}-start-convert-onnx]", return_time())
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
        print(f"[{local_rank}-start-done]", return_time())

    return model, encoded_tensors, labels, f"onnx/{model_name}.onnx"


class GPT2Classifier(torch.nn.Module):
    def __init__(self, model, input_dim: int, num_classes: int):
        super(GPT2Classifier, self).__init__()
        # input_dim: 768 or 1280
        self.gpt2model = model
        self.fc1 = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        gpt_out = self.gpt2model(x)
        linear_output = self.fc1(gpt_out.last_hidden_state)
        return linear_output.sum(axis=2)


class CustomDataset(Dataset):
    def __init__(self, encoded_tensors, labels, device, local_rank=-1):
        if local_rank == -1:
            self.encoded_tensors = [e.to(device) for e in encoded_tensors]
            self.labels = labels.to(torch.float32).to(device)
        else:
            self.encoded_tensors = [
                e.to(device)
                for i, e in enumerate(encoded_tensors)
                if i % WORLD_SIZE == local_rank
            ]
            self.labels = labels.to(torch.float32).to(device)[local_rank::WORLD_SIZE]
        self.transform = None
        self.target_transform = None
        self.local_rank = local_rank

    def __len__(self):
        return len(self.encoded_tensors)

    def __getitem__(self, idx):
        return self.encoded_tensors[idx], self.labels[idx]


class CustomLoss(torch.nn.L1Loss):
    def __init__(self, cst, cst2):
        super(CustomLoss, self).__init__()
        self.cst = cst
        self.cst2 = cst2

    def forward(self, outputs, labels):
        res = super(CustomLoss, self).forward(outputs * self.cst, labels)
        return res * self.cst2


def train(
    model,
    model_name,
    epochs,
    encoded_tensors,
    labels,
    device,
    scenario=None,
    train_batch_size=10,
):
    if scenario is not None and scenario.lower().startswith("ort-"):
        scenario = scenario[4:]
        use_ort = True
    else:
        use_ort = scenario.lower() == "ort"
        if use_ort:
            scenario = "torch"
    local_rank = 0
    print(f"[{local_rank}-train-scenario] {scenario} model_name={model_name!r}")
    print(f"[{local_rank}-train-device]", device, len(encoded_tensors))
    print(f"[{local_rank}-train-model-gpu]", return_time())
    # print(f"[{local_rank}-train-rank]", torch.distributed.get_rank())
    if model_name == "gpt2":
        input_dim = 768
    elif model_name == "gpt2-large":
        input_dim = 1280
    else:
        raise ValueError(f"Unexpected model name {model_name!r}.")
    model = GPT2Classifier(model, input_dim, 5)
    print(f"[{local_rank}-train-dataset]", return_time())
    ds = CustomDataset(
        encoded_tensors, labels.reshape((-1, 1, 5)).to(torch.float32), device, -1
    )
    my_dataloader = DataLoader(ds)
    print(
        f"[{local_rank}-train-done]",
        return_time(),
        type(encoded_tensors[0]),
        labels.shape,
    )
    criterion = CustomLoss(
        torch.Tensor([1e-4]).to(device), torch.Tensor([1]).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    logger = logging.getLogger("DeepSpeed")
    logger.setLevel(logging.WARNING)

    print(f"[{local_rank}-train-done]", return_time())
    times = []

    if use_ort:
        from onnxruntime.training.ortmodule import ORTModule

        print(f"[{local_rank}-train-ORTModule]", return_time())
        model = ORTModule(model)
        print(f"[{local_rank}-train-done]", return_time())

    if scenario.lower() in {
        "ds0",
        "ds1",
        "ds2",
        "ds3",
        "ds0-16",
        "ds1-16",
        "ds2-16",
        "ds3-16",
        "ds0-b16",
        "ds1-b16",
        "ds2-b16",
        "ds3-b16",
    }:
        import deepspeed

        # deepspeed.init_distributed()
        ds_config = get_deepspeed_config(scenario, train_batch_size)
        train_batch_size = ds_config["train_batch_size"]
        pprint.pprint(ds_config)
        dschf = HfDeepSpeedConfig(ds_config)
        model, optimizer, _, __ = deepspeed.initialize(
            config_params=ds_config,
            model=model,
            model_parameters=model.parameters(),
            # optimizer=optimizer,
        )
        assert optimizer is not None
        object_step = model
        f_backward = lambda model, loss: model.backward(loss)
        print(
            f"[{local_rank}-train-deepspeed] is_gradient_accumulation_boundary={model.is_gradient_accumulation_boundary()}"
        )
        print(
            f"[{local_rank}-train-deepspeed] custom_loss_scaler={getattr(optimizer, 'custom_loss_scaler', None)}"
        )
    elif scenario.lower() == "torch":
        model = model.to(device)
        object_step = optimizer
        f_backward = lambda model, loss: loss.backward()
    else:
        raise ValueError(f"Unexpected value for scenario={scenario!r}.")

    print(f"[{local_rank}-train-type] {type(my_dataloader)}")
    print(f"[{local_rank}-train-type] {type(model)}")
    print(f"[{local_rank}-train-type] {type(optimizer)}")
    print(f"[{local_rank}-train-type] {type(criterion)}")
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
                    print(f"[{local_rank}-train] optimizer.{k}: {len(v)}:{type(v)}")
                else:
                    print(f"[{local_rank}-train] optimizer.{k}: {type(v)}")

    for epoch_num in tqdm(range(epochs + 1)):
        total_loss_train = 0
        begin = time.perf_counter()

        object_step.zero_grad()

        for i, (x, y) in enumerate(my_dataloader):

            # optimizer.zero_grad()
            output = model(x["input_ids"])

            batch_loss = criterion(output, y.to(output.dtype))
            total_loss_train += batch_loss.to(float).item()

            f_backward(
                model, batch_loss
            )  # batch_loss.backward() or model.backward(batch_loss) for deepspeed

            if i % train_batch_size == train_batch_size - 1:
                object_step.step()  # optimizer.step() or model.step() for deepspeed
                object_step.zero_grad()

            if False and getattr(optimizer, "overflow", False):  # only for Zero
                lo = batch_loss.detach().float()
                raise RuntimeError(
                    f"Overflow after step(), offload={optimizer.cpu_offload}, "
                    f"len(optimizer.bit16_groups)={len(optimizer.bit16_groups)}, "
                    f"len(optimizer.averaged_gradients)={len(optimizer.averaged_gradients)}, "
                    f"partition_gradients={optimizer.partition_gradients}, "
                    f"loss={lo!r}, total_loss_train={total_loss_train!r}, i={i}"
                )

        end = time.perf_counter() - begin
        if epoch_num > 0:
            times.append(end)
        print(f"epoch_num={epoch_num}, total_loss_train={total_loss_train}, time={end}")
    return times


def main(epochs=10, n_obs=100, scenario="ds2", train_batch_size=10, model_name="gpt2"):
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
    * `'ds1'`: pytorch + deepspeed, no zero
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
    local_rank = -1
    print(
        f"[{local_rank}-train] get_device_capability()={torch.cuda.get_device_capability()}"
    )
    print(f"[{local_rank}-train] get_arch_list()={torch.cuda.get_arch_list()}")
    print(
        f"[{local_rank}-train] get_device_properties(...)={torch.cuda.get_device_properties(torch.device('cuda'))}"
    )
    print(
        f"[{local_rank}-train] epochs={epochs}, n_obs={n_obs}, scenario={scenario!r}, model_name={model_name!r}"
    )
    model, encoded_tensors, labels, model_name = startup(model_name)

    epochs = 10
    print()
    print(f"[{local_rank}-train]", return_time())
    device = torch.device("cuda:%d" % max(local_rank, 0))
    times = train(
        model,
        os.path.splitext(os.path.split(model_name)[-1])[0],
        epochs,
        encoded_tensors[:n_obs],
        labels,
        device,
        scenario=scenario,
        train_batch_size=train_batch_size,
    )
    print("[times]", sum(times) / len(times), times)
    print("[done]", return_time())


def _main_deepspeed(config_name, model_name, cmd_args):
    logger = logging.getLogger("DeepSpeed")
    logger.setLevel(logging.WARNING)

    if model_name == "gpt2":
        input_dim = 768
    elif model_name == "gpt2-large":
        input_dim = 1280
    else:
        raise ValueError(f"Unexpected model name {model_name!r}.")

    ds_config = get_deepspeed_config(config_name)
    dschf = HfDeepSpeedConfig(ds_config)
    train_batch_size = ds_config["train_batch_size"]
    if train_batch_size <= 1:
        raise ValueError(
            f"train_batch_size={train_batch_size} must be > 1\n{pprint.pformat(ds_config)}"
        )

    torch.cuda.set_device(cmd_args.local_rank)
    deepspeed.init_distributed()

    model, encoded_tensors, labels, model_name = startup(
        model_name, cmd_args.local_rank
    )
    model = GPT2Classifier(model, input_dim, 5)
    device = torch.device("cuda:%d" % cmd_args.local_rank)
    ds = CustomDataset(
        encoded_tensors,
        labels.reshape((-1, 1, 5)).to(torch.float32),
        device,
        cmd_args.local_rank,
    )
    my_dataloader = DataLoader(ds)
    criterion = CustomLoss(
        torch.Tensor([1e-4]).to(device), torch.Tensor([1]).to(device)
    )
    model, optimizer, _, __ = deepspeed.initialize(
        args=cmd_args,
        model=model,
        model_parameters=model.parameters(),
        config_params=ds_config,
        # optimizer=optimizer,
    )
    print(f"[{cmd_args.local_rank}-trainds] {type(model)}")
    print(f"[{cmd_args.local_rank}-trainds] {type(optimizer)}")
    times = []
    # model.train()

    for epoch_num in tqdm(range(10 + 1)):
        total_loss_train = 0
        begin = time.perf_counter()

        model.zero_grad()

        for i, (x, y) in enumerate(my_dataloader):

            # optimizer.zero_grad()
            output = model(x["input_ids"])

            batch_loss = criterion(output, y.to(output.dtype))
            total_loss_train += batch_loss.to(float).item()

            model.backward(batch_loss)

            if i % train_batch_size == train_batch_size - 1:
                model.step()
                model.zero_grad()

        end = time.perf_counter() - begin
        if epoch_num > 0:
            times.append(end)
        print(
            f"local_rank={cmd_args.local_rank}, epoch_num={epoch_num}, total_loss_train={total_loss_train}, time={end}"
        )
    return times


if __name__ == "__main__":
    # nvitop -m
    if any(map(lambda x: x.startswith("--local_rank"), sys.argv)):
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
        _main_deepspeed("ds3", "gpt2", cmd_args)
    elif any(map(lambda x: x.startswith("--scenario"), sys.argv)):
        print(f"[WORLD_SIZE={WORLD_SIZE}]")
        import fire

        fire.Fire(main)
    else:
        print(f"[WORLD_SIZE={WORLD_SIZE}-default]")
        main()
