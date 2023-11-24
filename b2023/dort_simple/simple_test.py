import os
import platform
import multiprocessing
import numpy as np
import pandas
import onnx
from onnxrewriter.optimizer import optimize
from onnx_extended.ext_test_case import measure_time
import torch
from torch import nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 5)
        self.conv2 = nn.Conv2d(128, 16, 5)
        self.fc1 = nn.Linear(13456, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def export_utils(prefix, model, *args):
    # dynamic
    name = f"{prefix}_script.onnx"
    print(f"torchscript {name!r}")
    torch.onnx.export(model, *args, name, input_names=["input"])
    print("size:", len(onnx.load(name).SerializeToString()))

    name = f"{prefix}_dynamo.onnx"
    print(f"torch dynamo {name!r}")
    export_output = torch.onnx.dynamo_export(model, *args)
    export_output.save(name)
    print("size:", len(onnx.load(name).SerializeToString()))

    model_onnx = onnx.load(name)
    name = f"{prefix}_dynamo_rewritten.onnx"
    print(f"optimizer {name!r}")
    optimized_model = optimize(model_onnx)
    with open(name, "wb") as f:
        f.write(optimized_model.SerializeToString())
    print("size:", len(onnx.load(name).SerializeToString()))
    print("done")


def save_optimized(model_name):
    from onnxruntime import InferenceSession, SessionOptions

    for aot in ["0", "1"]:
        sess = SessionOptions()
        path = f"{model_name}.cpu.aot{aot}.onnx"
        sess.optimized_model_filepath = path
        sess.add_session_config_entry("session.disable_aot_function_inlining", aot)
        InferenceSession(model_name, sess, providers=["CPUExecutionProvider"])

        sess = SessionOptions()
        path = f"{model_name}.gpu.aot{aot}.onnx"
        sess.optimized_model_filepath = path
        sess.add_session_config_entry("session.disable_aot_function_inlining", aot)
        InferenceSession(
            model_name,
            sess,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )


def system_info():
    obs = {}
    obs["processor"] = platform.processor()
    obs["cores"] = multiprocessing.cpu_count()
    obs["cuda"] = 1 if torch.cuda.is_available() else 0
    obs["cuda_count"] = torch.cuda.device_count()
    obs["cuda_name"] = torch.cuda.get_device_name()
    obs["cuda_capa"] = torch.cuda.get_device_capability()
    return obs


def benchmark():
    from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel

    shape = [1, 1, 128, 128]
    data = []
    for name in os.listdir("."):
        root = os.path.split(name)[-1]
        _, ext = os.path.splitext(root)
        if ext != ".onnx":
            continue
        for ps in [
            ["CPUExecutionProvider"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ]:
            opts = SessionOptions()
            opts.add_session_config_entry("session.disable_aot_function_inlining", "1")
            opts.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL

            obs = system_info()
            obs["name"] = name
            obs["providers"] = ",".join(ps)
            obs["optimization"] = (
                "CPU" if ".cpu." in name else ("GPU" if ".gpu." in name else "")
            )
            obs["AOT"] = 1 if "aot1" not in name else 0
            obs["rewriter"] = 1 if "rewritten" in name else 0
            obs["export"] = "dynamo" if "dynamo" in name else "script"

            try:
                sess = InferenceSession(name, opts, providers=ps)
            except Exception as e:
                print(f"ERROR-load: {name} {e}")
                obs.update({"error": e, "step": "run"})
                data.append(obs)
                continue

            input_name = sess.get_inputs()[0].name
            feeds = {input_name: np.random.rand(*shape).astype(np.float32)}
            try:
                for i in range(0, 10):
                    sess.run(None, feeds)
            except Exception as e:
                print(f"ERROR-run: {name} {e}")
                obs.update({"error": e, "step": "load"})
                data.append(obs)
                continue
            obs.update(measure_time(lambda: sess.run(None, feeds)))

            print(f"{obs['average']} {name} {ps}")
            data.append(obs)

    df = pandas.DataFrame(data)
    df.to_csv("benchmark.csv", index=False)
    df.to_excel("benchmark.xlsx", index=False)


if not os.path.exists("simple_script.onnx"):
    export_utils("simple", MyModel(), torch.rand((1, 1, 128, 128), dtype=torch.float32))
    save_optimized("simple_script.onnx")
    save_optimized("simple_dynamo.onnx")
    save_optimized("simple_dynamo_rewritten.onnx")
benchmark()
