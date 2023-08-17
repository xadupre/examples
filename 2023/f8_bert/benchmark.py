"""
Inspired from
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/
python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_CPU.ipynb.
"""
import pickle
import time
import onnxruntime


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

    dataset_file = "dataset.pkl"
    onnx_file = "bert-base-cased-squad.onnx"
    onnx_quant_file = "bert-base-cased-squad-int8.onnx"
    onnx_quant_file_f8 = "bert-base-cased-squad-fp8-local.onnx"
    onnx_quant_file_fp16 = "bert-base-cased-squad-fp16.onnx"
    dataset_file = "dataset.pkl"

    print(f"restoring dataset {dataset_file!r}")
    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)

    # original model
    for model_file in [
        onnx_file,
        onnx_quant_file,
        onnx_quant_file_f8,
        onnx_quant_file_fp16,
    ]:
        print()
        print(f"creating inference {model_file!r}")
        try:
            session = onnxruntime.InferenceSession(
                model_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
        except onnxruntime.capi.onnxruntime_pybind11_state.Fail as e:
            print(f"FAIL: {e}")
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
