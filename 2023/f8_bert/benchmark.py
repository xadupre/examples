"""
Inspired from
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/
python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_CPU.ipynb.
"""
import os
import pickle
import time
import onnxruntime
import pandas


def benchmark(
    session: onnxruntime.InferenceSession,
    dataset,
    total_samples: int,
    max_seq_length: int,
):
    latency = []
    for i in range(total_samples):
        data = dataset[i]
        ort_inputs = {
            "input_ids": data[0].cpu().reshape(1, max_seq_length).numpy(),
            "input_mask": data[1].cpu().reshape(1, max_seq_length).numpy(),
            "segment_ids": data[2].cpu().reshape(1, max_seq_length).numpy(),
        }
        start = time.perf_counter()
        session.run(None, ort_inputs)
        latency.append(time.perf_counter() - start)
    return latency


if __name__ == "__main__":
    max_seq_length = 128
    total_samples = 100
    doc_stride = 128
    max_query_length = 64
    data = []

    dataset_file = "dataset.pkl"

    onnx_files = list(
        sorted(
            [
                name
                for name in os.listdir(".")
                if os.path.splitext(name)[-1] == ".onnx" and "-squad" in name
            ]
        )
    )
    dataset_file = "dataset.pkl"

    print(f"restoring dataset {dataset_file!r}")
    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)

    # original model
    for ni in range(2):
        print("---------------------------------------")
        for im, model_file in enumerate(onnx_files):
            print()
            print(f"creating inference {im+1}/{len(onnx_files)}: {model_file!r}")
            try:
                session = onnxruntime.InferenceSession(
                    model_file,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
            except onnxruntime.capi.onnxruntime_pybind11_state.Fail as e:
                print(f"FAIL: {e}")
                data.append(
                    dict(
                        j=ni,
                        onnx=model_file,
                        error=str(e),
                        event="load",
                        size=os.stat(model_file).st_size,
                    )
                )
                continue

            print("warmup")
            try:
                benchmark(
                    session,
                    dataset,
                    total_samples=total_samples,
                    max_seq_length=max_seq_length,
                )
            except onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException as e:
                print(f"FAIL: {e}")
                data.append(
                    dict(
                        j=ni,
                        onnx=model_file,
                        error=str(e),
                        event="run",
                        size=os.stat(model_file).st_size,
                    )
                )
                continue

            print(f"starting benchmark {model_file!r}")
            for i in range(2):
                latency = benchmark(
                    session,
                    dataset,
                    total_samples=total_samples,
                    max_seq_length=max_seq_length,
                )
                lat = sum(latency) * 1000 / len(latency)
                print(f"try {i+1}: ort inference time = {lat:1.2f} ms")
                data.append(
                    dict(
                        i=i,
                        j=ni,
                        onnx=model_file,
                        latency=lat,
                        event="run",
                        size=os.stat(model_file).st_size,
                    )
                )

    df = pandas.DataFrame(data)
    print(df)
    df.to_csv("report.csv", index=False)
    df.to_excel("report.xlsx", index=False)
    with open("report.csv") as f:
        print(f.read())
