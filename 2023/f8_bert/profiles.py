"""
Inspired from
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/
python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_CPU.ipynb.
"""
import os
import pickle
import shutil
import onnxruntime
import logging
import matplotlib.pyplot as plt
from onnx_extended.tools.js_profile import (
    js_profile_to_dataframe,
    plot_ort_profile,
    _preprocess_graph1,
    _preprocess_graph2,
)


for name in [
    "matplotlib.font_manager",
    "PIL.PngImagePlugin",
    "matplotlib",
    "matplotlib.pyplot",
]:
    log = logging.getLogger(name)
    log.setLevel(logging.ERROR)


def profiles(
    session: onnxruntime.InferenceSession,
    dataset,
    total_samples: int,
    max_seq_length: int,
):
    for i in range(total_samples):
        data = dataset[i]
        ort_inputs = {
            "input_ids": data[0].cpu().reshape(1, max_seq_length).numpy(),
            "input_mask": data[1].cpu().reshape(1, max_seq_length).numpy(),
            "segment_ids": data[2].cpu().reshape(1, max_seq_length).numpy(),
        }
        session.run(None, ort_inputs)


if __name__ == "__main__":
    if not os.path.exists("stat"):
        os.mkdir("stat")
    if not os.path.exists("optimized"):
        os.mkdir("optimized")
    max_seq_length = 128
    total_samples = 11
    doc_stride = 128
    max_query_length = 64
    data = []
    N = 4

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
    print("---------------------------------------")
    for im, model_file in enumerate(onnx_files):
        print()
        print(f"creating inference {im+1}/{len(onnx_files)}: {model_file!r}")
        options = onnxruntime.SessionOptions()
        options.enable_profiling = True
        options.optimized_model_filepath = f"optimized/optimized-{model_file}"
        if "-ext" in model_file:
            from onnx_extended.ortops.tutorial.cuda import get_ort_ext_libs

            options.register_custom_ops_library(get_ort_ext_libs()[0])
        try:
            session = onnxruntime.InferenceSession(
                model_file,
                options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except onnxruntime.capi.onnxruntime_pybind11_state.Fail as e:
            print(f"FAIL: {e}")
            continue

        print(f"starting profiling {model_file!r}")
        profiles(
            session,
            dataset,
            total_samples=total_samples,
            max_seq_length=max_seq_length,
        )
        prof = session.end_profiling()

        noext = "stat/" + os.path.splitext(model_file)[0]
        profname = noext + ".prof"
        shutil.copy(prof, profname)

        df = js_profile_to_dataframe(profname, first_it_out=True)
        cols = list(df.reset_index(drop=False).columns)
        assert "it==0" in str(cols), f"Unexpected columns {cols}"
        print(
            "profiling data per operator, shape=",
            df.shape,
            df.shape[0] // (total_samples * 5 * 50),
        )
        df.to_csv(noext + ".raw.op.csv")
        fig, ax = plt.subplots(1, 2, figsize=(10, 25))
        plot_ort_profile(df, ax[0], ax[1], title=profname)
        fig.tight_layout()
        df2 = _preprocess_graph1(df)[-1]
        cols = list(df2.reset_index(drop=False).columns)
        assert "it==0" in str(cols), f"Unexpected columns {cols}"
        print("profiling data per operator, processed shape=", df2.shape)
        df2.to_csv(noext + ".op.csv")
        fig.savefig(noext + ".op.png")

        df = js_profile_to_dataframe(
            profname, first_it_out=True, agg=True, with_shape=True
        )
        cols = list(df.reset_index(drop=False).columns)
        assert "it==0" in str(cols), f"Unexpected columns {cols}"
        assert "shape" in str(cols), f"Unexpected columns {cols}"
        print("profiling data per node, shape=", df.shape)
        df.to_csv(noext + ".raw.node.csv")
        fig, ax = plt.subplots(1, 1, figsize=(10, 200))
        plot_ort_profile(df, ax, title=profname)
        fig.tight_layout()
        df3 = _preprocess_graph2(df)
        cols = list(df3.reset_index(drop=False).columns)
        assert "it==0" in str(cols), f"Unexpected columns {cols}"
        assert "shape" in str(cols), f"Unexpected columns {cols}"
        print("profiling data per node, processed shape=", df3.shape)
        df3.to_csv(noext + ".node.csv")
        fig.savefig(noext + ".node.png")
