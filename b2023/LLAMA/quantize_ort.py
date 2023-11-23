"""
Inspired from
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/
python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_CPU.ipynb.
"""
import collections
import logging
import os
import pickle
import time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization import (
    QuantFormat,
    QuantType,
    quantize_static,
    CalibrationDataReader,
)


class CalibrationInputReader(CalibrationDataReader):
    def __init__(self, data_folder: str):
        self.batch_id = 0
        self.input_folder = Path(data_folder)

        if not self.input_folder.is_dir():
            raise RuntimeError(
                f"Can't find input data directory: {str(self.input_folder)}"
            )
        data_file = self.input_folder / f"batch_{self.batch_id}.npz"
        if not data_file.exists():
            raise RuntimeError(f"No data files found under '{self.input_folder}'")

    def get_next(self):
        self.input_dict = None
        data_file = self.input_folder / f"batch_{self.batch_id}.npz"
        if not data_file.exists():
            return None
        self.batch_id += 1

        self.input_dict = {}
        npy_file = np.load(data_file)
        for name in npy_file.files:
            self.input_dict[name] = npy_file[name]

        return self.input_dict

    def rewind(self):
        self.batch_id = 0


if __name__ == "__main__":
    max_seq_length = 128
    total_samples = 100
    doc_stride = 128
    max_query_length = 64
    model_name_or_path = "bert-base-cased"
    cache_dir = "./cache"
    predict_file = "dev-v1.1.json"

    onnx_file = "models/llama_16_block_list_1.onnx"
    onnx_quant_file = "models/llama_16_block_list_1-dyn-{qtype}.onnx"
    onnx_quant_qdq_file = "models/llama_16_block_list_1-qdq-{qtype}.onnx"
    onnx_quant_qo_file = "models/llama_16_block_list_1-qo-{qtype}.onnx"
    dataset_file = "dataset.pkl"

    # dynamically quantize
    for qtype in [QuantType.QFLOAT8E4M3FN, QuantType.QInt8, QuantType.QUInt8]:
        qfile = onnx_quant_file.format(qtype=qtype.name.lower())
        if not os.path.exists(qfile):
            logging.basicConfig(level=logging.ERROR)
            print(f"quantize (dynamic) model {qfile!r}")
            quantize_dynamic(onnx_file, qfile, weight_type=qtype, op_types_to_quantize=["MatMul"])
            print("done.")
        else:
            model = onnx.load_model(qfile)
            qop = collections.Counter([n.op_type for n in model.graph.node])
            if "QuantizeLinear" not in qop:
                # ???
                logging.basicConfig(level=logging.DEBUG)
                print(f"quantize again (dynamic) model {qfile!r}")
                quantize_dynamic(onnx_file, qfile, weight_type=qtype, op_types_to_quantize=["MatMul"])
                print("done.")
        model = onnx.load_model(qfile)
        qop = collections.Counter([n.op_type for n in model.graph.node])
        print(f"quantized (dynamic) model {qfile!r}: {qop}")

    # first benchmark
    print(f"creating inference {onnx_file!r}")
    session = onnxruntime.InferenceSession(
        onnx_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    # static quantize qdq
    for qtype in [QuantType.QFLOAT8E4M3FN, QuantType.QInt8, QuantType.QUInt8]:
        qfile = onnx_quant_qdq_file.format(qtype=qtype.name.lower())
        if not os.path.exists(qfile):
            print(f"quantize (static qdq) model {qfile!r}")
            input_reader = CalibrationInputReader("data")
            quantize_static(
                onnx_file,
                qfile,
                input_reader,
                quant_format=QuantFormat.QDQ,
                per_channel=False,
                weight_type=qtype,
                activation_type=qtype,
            )
            print("done.")

        # static quantize qo
        qfile = onnx_quant_qo_file.format(qtype=qtype.name.lower())
        if not os.path.exists(qfile):
            print(f"quantize (static qo) model {qfile!r}")
            input_reader = CalibrationInputReader("data")
            quantize_static(
                onnx_file,
                qfile,
                input_reader,
                quant_format=QuantFormat.QOperator,
                per_channel=False,
                weight_type=qtype,
                activation_type=qtype,
            )
            print("done.")
