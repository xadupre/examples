import argparse
import logging
import os
import pickle
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


def get_deepspeed_config(scenario_full, train_batch_size=1):
    scenario = scenario_full.split("-")[0]
    if scenario == "ds0":
        ds_config = {
            "zero_optimization": {
                "allgather_bucket_size": 5e8,
                "contiguous_gradients": True,
                "stage": 0,
                "overlap_comm": True,
            },
        }
    elif scenario == "ds1":
        ds_config = {
            "zero_optimization": {
                "allgather_bucket_size": 5e8,
                "contiguous_gradients": True,
                "stage": 1,
                "overlap_comm": True,
            },
        }
    elif scenario == "ds2":
        ds_config = {
            "zero_optimization": {
                "allgather_bucket_size": 5e8,
                "allgather_partitions": True,
                "contiguous_gradients": True,
                # "offload_optimizer": { "device": "cpu", },
                "overlap_comm": True,
                "stage": 2,
            },
        }
    elif scenario == "ds3":
        ds_config = {
            "zero_optimization": {
                "allgather_bucket_size": 5e8,
                "allgather_partitions": True,
                "contiguous_gradients": True,
                "offload_param": {"device": "cpu", "pin_memory": True},
                "overlap_comm": True,
                "reduce_bucket_size": 5e8,
                "reduce_scatter": True,
                "stage": 3,
            },
        }
    else:
        raise ValueError(f"Unknwon scenario {scenario!r}.")

    optimizer_config = {
        "type": "Adam",
        "params": {
            "betas": [0.8, 0.999],
            "eps": 1e-8,
            "lr": 0.0001,
            "weight_decay": 3e-7,
        },
    }

    ds_config.update(
        {
            "gradient_accumulation_steps": 1,
            # "gradient_clipping": 1.0,
            "optimizer": optimizer_config,
            "train_micro_batch_size_per_gpu": 1,
            "train_batch_size": WORLD_SIZE,
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
                    "loss_scale_window": 1000,
                    "hysteresis": 2,
                    "min_loss_scale": 1
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


def startup(local_rank=-1):
    print(f"[{local_rank}-start]")
    # print(f"[{local_rank}-start-load-datasets]", return_time())
    # data = load_dataset("wikitext", "wikitext-2-raw-v1")
    print(f"[{local_rank}-start-load-tokenizer]", return_time())
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print(f"[{local_rank}-start-load-model]", return_time())
    model = GPT2Model.from_pretrained("gpt2")
    print(f"[{local_rank}-start-run-tokenizer]", return_time())

    print(f"[{local_rank}-start-done]", return_time())
    name = "encoded_tensors.pkl"
    if not os.path.exists(name):
        print(f"[{local_rank}-start-loading-data]", return_time())
        df = pandas.read_csv("data/train.csv")
        print(f"[{local_rank}-start-done]", df.shape, df.columns, return_time())
        labels = torch.from_numpy(pandas.get_dummies(df.category).values)
        model_input = df.text
        encoded_tensors = []
        for t in tqdm(model_input):
            tens = torch.tensor([tokenizer.encode(t, add_special_tokens=True)])
            if tens.shape[-1] > 1024:
                tens = tens[:, :1024]
            encoded_tensors.append(tens)
        print(f"[{local_rank}-start-pickle]", len(encoded_tensors), return_time())
        with open(name, "wb") as f:
            pickle.dump([encoded_tensors, labels], f)
    else:
        print(f"[{local_rank}-start-unpickle]", return_time())
        with open(name, "rb") as f:
            [encoded_tensors, labels] = pickle.load(f)
    print(f"[{local_rank}-start-done]", len(encoded_tensors), return_time())

    if not os.path.exists("gpt2.onnx"):
        print(f"[{local_rank}-start-convert-onnx]", return_time())
        torch.onnx.export(
            model,
            encoded_tensors[0],
            "gpt2.onnx",
            verbose=False,
            input_names=["X"],
            output_names=["Y"],
            opset_version=15,
            dynamic_axes={"X": [0, 1]},
        )
        print(f"[{local_rank}-start-done]", return_time())

    return tokenizer, model, encoded_tensors, labels, "gpt2.onnx"


class GPT2Classifier(torch.nn.Module):
    def __init__(self, model, num_classes: int):
        super(GPT2Classifier, self).__init__()
        self.gpt2model = model
        self.fc1 = torch.nn.Linear(768, num_classes)

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


def train(model, epochs, encoded_tensors, labels, device, scenario=None):
    local_rank = 0
    print(f"[{local_rank}-train-device]", device, len(encoded_tensors))
    print(f"[{local_rank}-train-model-gpu]", return_time())
    # print(f"[{local_rank}-train-rank]", torch.distributed.get_rank())
    model = GPT2Classifier(model, 5)
    model = model.to(device)
    print(f"[{local_rank}-train-dataset]", return_time())
    ds = CustomDataset(
        encoded_tensors, labels.reshape((-1, 1, 5)).to(torch.float32), device, -1
    )
    my_dataloader = DataLoader(ds)
    print(
        f"[{local_rank}-train-done]",
        return_time(),
        encoded_tensors[0].shape,
        labels.shape,
    )
    criterion = CustomLoss(
        torch.Tensor([1e-4]).to(device), torch.Tensor([1]).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_batch_size = 1

    logger = logging.getLogger("DeepSpeed")
    logger.setLevel(logging.WARNING)

    print(f"[{local_rank}-train-done]", return_time())
    times = []

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
    elif scenario.lower() == "ort":
        from onnxruntime.training.ortmodule import ORTModule

        print(f"[{local_rank}-train-ORTModule]", return_time())
        model = ORTModule(model)
        print(f"[{local_rank}-train-done]", return_time())
        object_step = optimizer
        f_backward = lambda model, loss: loss.backward()
    elif scenario.lower() == "torch":
        object_step = optimizer
        f_backward = lambda model, loss: loss.backward()
    else:
        raise ValueError(f"Unexpected value for scenario={scenario!r}.")

    print(f"[{local_rank}-train-type] {type(my_dataloader)}")
    print(f"[{local_rank}-train-type] {type(model)}")
    print(f"[{local_rank}-train-type] {type(optimizer)}")
    print(f"[{local_rank}-train-type] {type(criterion)}")
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
        model.train()

        for i, (x, y) in enumerate(my_dataloader):

            model.zero_grad()
            # optimizer.zero_grad()
            output = model(x)

            batch_loss = criterion(output, y.to(output.dtype))
            total_loss_train += batch_loss.to(float).item()

            f_backward(
                model, batch_loss
            )  # batch_loss.backward() or model.backward(batch_loss) for deepspeed

            object_step.step()  # optimizer.step() or model.step() for deepspeed
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


def main(epochs=10, n_obs=100, scenario="ds1"):
    """
    Trains a dummy model based on GPT-2. The model has no real meaning
    but to measure the training of a GPT-2 model.

    :param epochs: number of training iterations (+ 1),
        first one is not included in the average time
    :param n_obs: number of observations in the training set
    :param scenario: training to measure. See below.

    About scenario:

    * `torch`: pytorch training
    * `'ort'`: pytorch + ORTModule
    * `'ds1'`: pytorch + deepspeed, no zero
    * `'ds1'`: pytorch + deepspeed stage zero 1
    * `'ds2'`: pytorch + deepspeed stage zero 2
    * `'ds3'`: pytorch + deepspeed stage zero 3

    Measures:

    * epochs=10, n_obs=100, scenario=torch, average=6.512568323302548,
      details=[6.761976952009718, 6.499132192999241, 6.414225077009178, 6.695275236997986, 6.624779493009555, 6.6113296980038285, 6.447838849009713, 6.350545073000831, 6.364595259990892, 6.355985400994541]
    * epochs=10, n_obs=100, scenario='prt', average=5.557973394899454,
      details=[5.551999910996528, 5.556793335999828, 5.55558459898748, 5.562688290010556, 5.553183950003586, 5.542683186009526, 5.560743605994503, 5.565259896990028, 5.562582682003267, 5.5682144919992425]
    * epochs=10, n_obs=100, scenario='ds0', average=5.75481926240027,
      details=[5.735581899993122, 5.819239084026776, 5.7707890839956235, 5.75322976699681, 5.737118029996054, 5.7314762890164275, 5.737127778003924, 5.7447630599781405, 5.746277269005077, 5.772590362990741]
    * epochs=10, n_obs=100, scenario='ds1', average=8.96984206599991,
      after disabling check_overflow https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/stage_1_and_2.py#L1753
      details=[8.972679587999664, 8.954322099999445, 8.997046710000177, 8.964849541999683, 8.973226385000089, 8.94784984800026, 8.957941382999707, 8.954076443000304, 8.988149599999815, 8.988279060999957]

    Building instructions:

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
    print(f"[{local_rank}-train] epochs={epochs}, n_obs={n_obs}, scenario={scenario!r}")
    tokenizer, model, encoded_tensors, labels, model_name = startup()

    epochs = 10
    print()
    print(f"[{local_rank}-train]", return_time())
    device = torch.device("cuda:%d" % max(local_rank, 0))
    times = train(
        model, epochs, encoded_tensors[:n_obs], labels, device, scenario=scenario
    )
    print("[times]", sum(times) / len(times), times)
    print("[done]", return_time())


def _main(cmd_args):
    logger = logging.getLogger("DeepSpeed")
    logger.setLevel(logging.WARNING)

    ds_config = get_deepspeed_config("ds1")
    dschf = HfDeepSpeedConfig(ds_config)
    tokenizer, model, encoded_tensors, labels, model_name = startup(cmd_args.local_rank)
    model = GPT2Classifier(model, 5)
    device = torch.device("cuda:%d" % cmd_args.local_rank)
    model = model.to(device)
    ds = CustomDataset(
        encoded_tensors,
        labels.reshape((-1, 1, 5)).to(torch.float32),
        device,
        cmd_args.local_rank,
    )
    my_dataloader = DataLoader(ds)
    criterion = CustomLoss(torch.Tensor([1e-4]).to(device))
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

    for epoch_num in tqdm(range(10 + 1)):
        total_loss_train = 0
        begin = time.perf_counter()
        model.train()

        for i, (x, y) in enumerate(my_dataloader):

            model.zero_grad()
            # optimizer.zero_grad()
            output = model(x)

            batch_loss = criterion(output, y.to(output.dtype))
            total_loss_train += batch_loss.to(float).item()

            model.backward(batch_loss)
            model.step()

            if getattr(optimizer, "overflow", False):  # only for Zero
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
        print(
            f"local_rank={cmd_args.local_rank}, epoch_num={epoch_num}, total_loss_train={total_loss_train}, time={end}"
        )
    return times


if __name__ == "__main__":
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
        print(f"[WORLD_SIZE={WORLD_SIZE}-LOCAL_RANK={cmd_args.local_rank}]")
        _main(cmd_args)
    elif any(map(lambda x: x.startswith("--scenario"), sys.argv)):
        print(f"[WORLD_SIZE={WORLD_SIZE}]")
        import fire

        fire.Fire(main)
    else:
        print(f"[WORLD_SIZE={WORLD_SIZE}-default]")
        main()
