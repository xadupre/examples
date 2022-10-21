import os
import time
import pickle
import numpy as np
import pandas
from tqdm import tqdm
from datasets import list_datasets, load_dataset
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import GPT2Tokenizer, GPT2Model

_origin = time.perf_counter()


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


def startup():
    print("[start]")
    # print("[start-load-datasets]", return_time())
    # data = load_dataset("wikitext", "wikitext-2-raw-v1")
    print("[start-load-tokenizer]", return_time())
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print("[start-load-model]", return_time())
    model = GPT2Model.from_pretrained("gpt2")
    print("[start-run-tokenizer]", return_time())

    print("[start-done]", return_time())
    name = "encoded_tensors.pkl"
    if not os.path.exists(name):
        print("[start-loading-data]", return_time())
        df = pandas.read_csv("data/train.csv")
        print("[start-done]", df.shape, df.columns, return_time())
        labels = torch.from_numpy(pandas.get_dummies(df.category).values)
        model_input = df.text
        encoded_tensors = []
        for t in tqdm(model_input):
            tens = torch.tensor([tokenizer.encode(t, add_special_tokens=True)])
            if tens.shape[-1] > 1024:
                tens = tens[:, :1024]
            encoded_tensors.append(tens)
        print("[start-pickle]", len(encoded_tensors), return_time())
        with open(name, "wb") as f:
            pickle.dump([encoded_tensors, labels], f)
    else:
        print("[start-unpickle]", return_time())
        with open(name, "rb") as f:
            [encoded_tensors, labels] = pickle.load(f)
    print("[start-done]", len(encoded_tensors), return_time())

    if not os.path.exists("gpt2.onnx"):
        print("[start-convert-onnx]", return_time())
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
        print("[start-done]", return_time())

    return tokenizer, model, encoded_tensors, labels, "gpt2.onnx"


class GPT2Classifier(torch.nn.Module):
    def __init__(self, model, num_classes: int):
        super(GPT2Classifier, self).__init__()
        self.gpt2model = model
        self.fc1 = torch.nn.Linear(768, num_classes)

    def forward(self, x):
        gpt_out = self.gpt2model(x)
        linear_output = self.fc1(gpt_out.last_hidden_state)
        return linear_output.sum(axis=1)


def train(model, epochs, encoded_tensors, labels, device, scenario=None):
    print("[train-device]", device, len(encoded_tensors))
    print("[train-model-gpu]", return_time())
    model = GPT2Classifier(model, 5)
    model = model.to(device)
    print("[train-tensors-gpu]", return_time())
    encoded_tensors = [t.to(device) for t in encoded_tensors]
    labels = labels.to(torch.float32).to(device)
    print("[train-done]", return_time())
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    print("[train-done]", return_time())
    times = []

    if scenario in {"ds0", "ds1", "ds2", "ds3"}:
        import deepspeed

        if scenario == "ds1":
            config = {
                "zero_optimization": {
                    "stage": 1,
                },
                "reduce_bucket_size": 5e8,
                "train_batch_size": 1,
            }
        elif scenario == "ds2":
            config = {
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {
                        "device": "cpu",
                    },
                    "contiguous_gradients": True,
                },
                "reduce_bucket_size": 5e8,
                "train_batch_size": 1,
            }
        else:
            stop

        model, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
            config=config,
            model=model,
            optimizer=optimizer,
        )
    elif scenario == "ORT":
        from onnxruntime.training.ortmodule import ORTModule

        print("[train-ORTModule]", return_time())
        model = ORTModule(model)
        print("[train-done]", return_time())
    elif scenario is not None:
        raise ValueError(f"Unexpected value for scenario={scenario!r}.")

    for epoch_num in range(epochs + 1):
        total_loss_train = 0
        begin = time.perf_counter()

        for i, x in tqdm(enumerate(encoded_tensors)):

            model.zero_grad()
            output = model(x)

            label = labels[i : i + 1]
            batch_loss = criterion(output, label)
            total_loss_train += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

        end = time.perf_counter() - begin
        if epoch_num > 0:
            times.append(end)
        print(f"epoch_num={epoch_num}, total_loss_train={total_loss_train}, time={end}")
    return times


def main(epochs=10, n_obs=100, scenario=None):
    """
    Trains a dummy model based on GPT-2. The model has no real meaning
    but to measure the training of a GPT-2 model.

    :param epochs: number of training iterations (+ 1),
        first one is not included in the average time
    :param n_obs: number of observations in the training set
    :param scenario: training to measure. See below.

    About scenario:

    * `None`: pytorch training
    * `'ORT'`: pytorch + ORTModule
    * `'ds1'`: pytorch + deepstage stage 1
    * `'ds2'`: pytorch + deepstage stage 2
    * `'ds3'`: pytorch + deepstage stage 3

    Measures:

    * epochs=10, n_obs=100, scenario=None, average=6.512568323302548,
      details=[6.761976952009718, 6.499132192999241, 6.414225077009178, 6.695275236997986, 6.624779493009555, 6.6113296980038285, 6.447838849009713, 6.350545073000831, 6.364595259990892, 6.355985400994541]
    * epochs=10, n_obs=100, scenario='ORT', average=5.557973394899454,
      details=[5.551999910996528, 5.556793335999828, 5.55558459898748, 5.562688290010556, 5.553183950003586, 5.542683186009526, 5.560743605994503, 5.565259896990028, 5.562582682003267, 5.5682144919992425]
    * epochs=10, n_obs=100, scenario='ds1', failure in `optimizer.step()`,
      (in stage_1_and_2.py", line 1870, self.averaged_gradients[i], key 0 does not exist)

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
    print(f"[TRAIN] epochs={epochs}, n_obs={n_obs}, scenario={scenario!r}")
    tokenizer, model, encoded_tensors, labels, model_name = startup()

    epochs = 10
    print()
    print("[train]", return_time())
    device = torch.device("cuda:0")
    times = train(
        model, epochs, encoded_tensors[:n_obs], labels, device, scenario=scenario
    )
    print("[times]", sum(times) / len(times), times)
    print("[done]", return_time())


if __name__ == "__main__":
    import fire

    fire.Fire(dict(train=main))

