import os
import platform
import psutil
from onnx_extended.tools.run_onnx import bench_virtual
from onnx_extended.ext_test_case import get_parsed_args

args = get_parsed_args(
    "run_bench",
    **dict(
        test_name=(
            "test_ort_version-F100-T500-D10-B1000",
            "folder containing the benchmark to run",
        ),
    ),
)

name = args.test_name
folder = os.path.abspath(f"{name}/rf")
if not os.path.exists(folder):
    raise FileNotFoundError(f"Unable to find {folder!r}.")
virtual_env = os.path.abspath("venv")

runtimes = ["onnxruntime"]
modules = [
    {"onnx-extended": "0.2.3", "onnx": "1.14.1", "onnxruntime": "1.16.1"},
    {"onnx-extended": "0.2.3", "onnx": "1.14.1", "onnxruntime": "1.16.0"},
    {"onnx-extended": "0.2.3", "onnx": "1.14.1", "onnxruntime": "1.15.1"},
    {"onnx-extended": "0.2.3", "onnx": "1.14.1", "onnxruntime": "1.14.1"},
    {"onnx-extended": "0.2.3", "onnx": "1.14.1", "onnxruntime": "1.13.1"},
    {"onnx-extended": "0.2.3", "onnx": "1.14.1", "onnxruntime": "1.12.1"},
]

print("--------------------------")
print(platform.machine(), platform.version(), platform.platform())
print(platform.processor())
print(f"RAM: {psutil.virtual_memory().total / (1024.0 **3):1.3f} GB")
print("Physical cores:", psutil.cpu_count(logical=False))
print("Total cores:", psutil.cpu_count(logical=True))
print("--------------------------")
print(name)
for t in range(3):
    print("--------------------------")
    df = bench_virtual(
        folder,
        virtual_env,
        verbose=1,
        modules=modules,
        runtimes=runtimes,
        warmup=5,
        repeat=10,
        save_as_dataframe=f"result-{name}.t{t}.csv",
        filter_fct=lambda rt, modules: True,
    )

    columns = ["runtime", "b_avg_time", "runtime", "v_onnxruntime"]
    df[columns].to_csv(f"summary-{name}.t{t}.csv")
    print(df[columns])
