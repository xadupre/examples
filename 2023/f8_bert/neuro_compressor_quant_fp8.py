if False:
    # Simple matmul to test.
    import os
    import numpy as np
    from onnx import TensorProto
    from onnx.helper import (
        make_model,
        make_graph,
        make_node,
        make_tensor_value_info,
        make_opsetid,
    )
    from onnx.numpy_helper import from_array
    from onnxruntime import InferenceSession
    from neural_compressor import quantization, PostTrainingQuantConfig

    if not os.path.exists("_quantized"):
        os.mkdir("_quantized")

    model = "matmul.onnx"
    if not os.path.exists(model):
        print("Create a simple matmul")

        onx = make_model(
            make_graph(
                [make_node("MatMul", ["X", "W"], ["Y"])],
                "matmul",
                [make_tensor_value_info("X", TensorProto.FLOAT16, [1024, 1024])],
                [make_tensor_value_info("Y", TensorProto.FLOAT16, [1024, 1024])],
                [from_array(np.random.randn(1024, 1024).astype(np.float16), "W")],
            ),
            opset_imports=[make_opsetid("", 18)],
        )
        with open(model, "wb") as f:
            f.write(onx.SerializeToString())

    sess = InferenceSession(model, providers=["CPUExecutionProvider"])
    feeds = {"X": np.random.randn(1024, 1024).astype(np.float16)}
    got = sess.run(None, feeds)
    assert got[0].shape == feeds["X"].shape

    class Dataloader:
        def __init__(self, total_samples=10):
            self.batch_size = total_samples

        def __len__(self):
            return self.batch_size

        def __iter__(self):
            for i in range(self.batch_size):
                yield np.random.randn(1024, 1024).astype(np.float16), np.random.randn(
                    1024, 1024
                ).astype(np.float16)

    configs = [
        (
            "static_qdq",
            PostTrainingQuantConfig(
                approach="static",
                quant_format="QDQ",
                precision="fp8_e4m3",
                backend="default",
            ),
        ),
        (
            "dynamic_qdq",
            PostTrainingQuantConfig(
                approach="dynamic",
                quant_format="QDQ",
                precision="fp8_e4m3",
                backend="default",
            ),
        ),
    ]

    dataloader = Dataloader(10)

    for name, config in configs:
        print("------------------------------------------------------")
        print(config)
        q_model = quantization.fit(model, config, calib_dataloader=dataloader)
        if q_model is not None:
            print(type(q_model))
            q_model.save(f"./_quantized/conf_{name}.onnx")

if True:
    # BERT
    import os
    import pickle
    from neural_compressor import quantization, PostTrainingQuantConfig
    from onnxruntime import InferenceSession

    if not os.path.exists("_quantized"):
        os.mkdir("_quantized")

    class Dataloader:
        def __init__(self, model, dataset_file, total_samples, max_seq_length):
            print(f"restoring dataset {dataset_file!r}")
            with open(dataset_file, "rb") as f:
                self.dataset = pickle.load(f)
            self.batch_size = total_samples
            self.max_seq_length = max_seq_length
            self.sess = InferenceSession(
                model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

        def __len__(self):
            return self.batch_size

        def __iter__(self):
            for i in range(self.batch_size):
                data = self.dataset[i]
                ort_inputs = {
                    "input_ids": data[0].cpu().reshape(1, self.max_seq_length).numpy(),
                    "input_mask": data[1].cpu().reshape(1, self.max_seq_length).numpy(),
                    "segment_ids": data[2]
                    .cpu()
                    .reshape(1, self.max_seq_length)
                    .numpy(),
                }
                expected = self.sess.run(None, ort_inputs)
                yield ort_inputs, expected

    configs = [
        (
            "static_qdq",
            PostTrainingQuantConfig(
                approach="static",
                quant_format="QDQ",
                precision="fp8_e4m3",
                backend="default",
            ),
        ),
        (
            "dynamic_qdq",
            PostTrainingQuantConfig(
                approach="dynamic",
                quant_format="QDQ",
                precision="fp8_e4m3",
                backend="default",
            ),
        ),
    ]

    model = "bert-base-cased-squad-fp16.onnx"
    dataloader = Dataloader(model, "dataset.pkl", 11, max_seq_length=128)

    for name, config in configs:
        print("------------------------------------------------------")
        print(config)
        q_model = quantization.fit(model, config, calib_dataloader=dataloader)
        print(type(q_model))
        q_model.save(f"./_quantized/bert-{name}-e4m3.onnx")


"""
if False:
    # old API
    from neural_compressor.experimental import Quantization

    dataset = quantizer.dataset("dummy", shape=(1, 224, 224, 3))
    quantizer.calib_dataloader = common.DataLoader(dataset)

    quant_conf = PostTrainingQuantConfig(
        precision="fp8_e4m3",
        calibration_sampling_size=[300],
        batchnorm_calibration_sampling_size=[3000],
    )
    q_model = quantization.fit(
        model, quant_conf, eval_func=eval_func, calib_dataloader=self.cv_dataloader
    )
"""
