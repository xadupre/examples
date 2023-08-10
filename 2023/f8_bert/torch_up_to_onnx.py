"""
Inspired from
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/
python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_CPU.ipynb.
"""
import os
import pickle
import time
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import torch
from transformers import BertConfig, BertForQuestionAnswering, BertTokenizer
from transformers.data.processors.squad import SquadV1Processor
from transformers import squad_convert_examples_to_features


def transform(cache_dir, max_query_length, doc_stride, total_samples):
    config_class, model_class, tokenizer_class = (
        BertConfig,
        BertForQuestionAnswering,
        BertTokenizer,
    )

    config = config_class.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer = tokenizer_class.from_pretrained(
        model_name_or_path, do_lower_case=True, cache_dir=cache_dir
    )
    model = model_class.from_pretrained(
        model_name_or_path, from_tf=False, config=config, cache_dir=cache_dir
    )

    processor = SquadV1Processor()
    examples = processor.get_dev_examples(None, filename=predict_file)

    features, dataset = squad_convert_examples_to_features(
        examples=examples[:total_samples],
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False,
        return_dataset="pt",
    )
    return model, features, dataset


def export_to_onnx(export_model_path, model, dataset, max_seq_length):
    device = torch.device("cpu")

    data = dataset[0]
    inputs = {
        "input_ids": data[0].to(device).reshape(1, max_seq_length),
        "attention_mask": data[1].to(device).reshape(1, max_seq_length),
        "token_type_ids": data[2].to(device).reshape(1, max_seq_length),
    }

    model.eval()
    model.to(device)

    with torch.no_grad():
        symbolic_names = {0: "batch_size", 1: "max_seq_len"}
        torch.onnx.export(
            model,
            args=tuple(inputs.values()),
            f=export_model_path,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input_ids", "input_mask", "segment_ids"],
            output_names=["start", "end"],
            dynamic_axes={
                "input_ids": symbolic_names,
                "input_mask": symbolic_names,
                "segment_ids": symbolic_names,
                "start": symbolic_names,
                "end": symbolic_names,
            },
        )


def benchmark(
    session: onnxruntime.InferenceSession,
    dataset,
    total_samples: int,
    max_seq_length: int,
):
    print("creating ineference")
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
    model_name_or_path = "bert-base-cased"
    cache_dir = "./cache"
    predict_file = "dev-v1.1.json"

    model_file = "model.pt"
    onnx_file = "bert-base-cased-squad.onnx"
    onnx_quant_file = "bert-base-cased-squad-int8.onnx"
    dataset_file = "dataset.pkl"

    if not os.path.exists(dataset_file) or not os.path.exists(model_file):
        print(f"loading dataset {dataset_file!r} and {model_file!r}")
        model, features, dataset = transform(
            cache_dir, max_query_length, doc_stride, total_samples
        )
        with open(dataset_file, "wb") as f:
            pickle.dump(dataset, f)
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
    else:
        print(f"restoring dataset {dataset_file!r} and {model_file!r}")
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)
        with open(model_file, "rb") as f:
            model = pickle.load(f)

    if not os.path.exists(onnx_file):
        print(f"export model {onnx_file!r}")
        export_to_onnx(onnx_file, model, dataset, max_seq_length)
    else:
        print(f"model already exported {onnx_file!r}")

    if not os.path.exists(onnx_quant_file):
        print(f"quantize model {onnx_quant_file!r}")
        quantized_model = quantize_dynamic(onnx_file, onnx_quant_file)
    else:
        print(f"model already quantized {onnx_quant_file!r}")

    print(f"creating inference {onnx_file!r}")
    session = onnxruntime.InferenceSession(
        onnx_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    print(f"starting benchmark {onnx_file!r}")
    latency = benchmark(
        session,
        dataset,
        total_samples=total_samples,
        max_seq_length=max_seq_length,
    )
    print(
        "OnnxRuntime cpu Inference time = {} ms".format(
            format(sum(latency) * 1000 / len(latency), ".2f")
        )
    )

    print(f"creating inference {onnx_quant_file!r}")
    session = onnxruntime.InferenceSession(
        onnx_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    print(f"starting benchmark {onnx_quant_file!r}")
    latency = benchmark(
        session,
        dataset,
        total_samples=total_samples,
        max_seq_length=max_seq_length,
    )
    print(
        "OnnxRuntime cpu Inference time = {} ms".format(
            format(sum(latency) * 1000 / len(latency), ".2f")
        )
    )
