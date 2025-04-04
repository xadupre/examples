import argparse
import numpy as np
import onnx
import onnxruntime as ort
import onnxscript
import os
import requests
import shutil
import soundfile
import subprocess
import sys
import torch

from onnx import helper, numpy_helper, TensorProto
from onnxruntime_genai.models.builder import create_model
from onnxruntime.transformers.dynamo_onnx_helper import DynamoOnnxHelper
from onnxscript import ir
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM


def build_vision(args):
    # Many images:
    prompt = f"{user_prompt}<|image_1|>\n<|image_2|>\n<|image_3|>\n<|image_4|>\nWhat is shown in these four images?{prompt_suffix}{assistant_prompt}"
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image_1 = Image.open(requests.get(url, stream=True).raw)
    url = "https://img.freepik.com/free-photo/painting-mountain-lake-with-mountain-background_188544-9126.jpg?w=2000"
    image_2 = Image.open(requests.get(url, stream=True).raw)
    url = "https://th.bing.com/th/id/OIP.gCvQ1vmPVJmrq1nnzM3ZHQHaEo?rs=1&pid=ImgDetMain"
    image_3 = Image.open(requests.get(url, stream=True).raw)
    url = "https://wallpaper.dog/large/10809054.jpg"
    image_4 = Image.open(requests.get(url, stream=True).raw)
    images = [image_1, image_2, image_3, image_4]
    inputs = processor(prompt, images=images, return_tensors="pt").to(
        args.execution_provider.replace("dml", "cuda")
    )
    inputs["input_image_embeds"] = inputs["input_image_embeds"].to(args.precision)
    inputs["image_attention_mask"] = inputs["image_attention_mask"].to(args.precision)

    # TorchScript export
    dummy_inputs = (
        inputs["input_image_embeds"],  # image_embeds: torch.FloatTensor
        inputs["image_attention_mask"],  # image_attention_mask: torch.FloatTensor
        inputs["image_sizes"],  # image_sizes: torch.LongTensor
    )
    dynamic_axes = {
        "pixel_values": {0: "num_images", 1: "max_num_crops", 3: "height", 4: "width"},
        "image_attention_mask": {0: "num_images", 1: "max_num_crops"},
        "image_sizes": {0: "num_images"},
        "image_features": {0: "num_image_tokens"},
    }
    filename = "phi-4-mm-vision.onnx"

    temp_folder_1 = os.path.join(args.output, "vision_init_export")
    os.makedirs(temp_folder_1, exist_ok=True)

    fpath_1 = os.path.join(temp_folder_1, filename)
    torch.onnx.export(
        model.model.embed_tokens_extend.image_embed,
        args=dummy_inputs,
        f=fpath_1,
        export_params=True,
        input_names=["pixel_values", "image_attention_mask", "image_sizes"],
        output_names=["image_features"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )

    onnx.checker.check_model(fpath_1)
    onnx.shape_inference.infer_shapes_path(fpath_1)
    onnx_model = onnx.load_model(fpath_1, load_external_data=True)

    temp_folder_2 = os.path.join(args.output, "vision_after_export")
    os.makedirs(temp_folder_2, exist_ok=True)

    fpath_2 = os.path.join(temp_folder_2, filename)
    onnx.save_model(
        onnx_model,
        fpath_2,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{filename}.data",
        size_threshold=0,
        convert_attribute=False,
    )
    shutil.rmtree(temp_folder_1)

    # ORT transformer optimizer
    temp_folder_3 = os.path.join(args.output, "vision_after_opt")
    fpath_3 = os.path.join(temp_folder_3, filename)
    subprocess.run(
        [
            f"{sys.executable}",
            "-m",
            "onnxruntime.transformers.optimizer",
            "--input",
            fpath_2,
            "--output",
            fpath_3,
            "--model_type",
            "clip",
            "--num_heads",
            str(16),
            "--hidden_size",
            str(1152),
            "--use_external_data_format",
            "--opt_level",
            str(0),
            "--disable_shape_inference",
        ]
    )
    shutil.rmtree(temp_folder_2)

    # ORT 4-bits quantizer
    fpath_4 = os.path.join(args.output, filename)
    cmd = [
        f"{sys.executable}",
        "-m",
        "onnxruntime.quantization.matmul_4bits_quantizer",
        "--input_model",
        fpath_3,
        "--output_model",
        fpath_4,
        "--block_size",
        str(32),
    ]
    if args.precision == torch.float32:
        cmd.extend(["--accuracy_level", str(4)])
    subprocess.run(cmd)
    shutil.rmtree(temp_folder_3)


def build_speech(args):
    # Speech file:
    prompt = f"{user_prompt}<|audio_1|>\n<|audio_2|>\nWhat are the stories that these audios come from?{prompt_suffix}{assistant_prompt}"
    audio1 = soundfile.read(
        os.path.join(
            args.input, "examples", "what_is_the_traffic_sign_in_the_image.wav"
        )
    )
    audio2 = soundfile.read(
        os.path.join(args.input, "examples", "what_is_shown_in_this_image.wav")
    )
    inputs = processor(prompt, audios=[audio1, audio2], return_tensors="pt").to(
        args.execution_provider.replace("dml", "cuda")
    )
    inputs["input_audio_embeds"] = inputs["input_audio_embeds"].to(args.precision)

    # TorchScript export
    dummy_inputs = (
        inputs["input_audio_embeds"],  # audio_embeds: torch.FloatTensor
        inputs["audio_attention_mask"],  # audio_attention_mask: torch.BoolTensor
        inputs["audio_embed_sizes"],  # audio_sizes: torch.LongTensor
        inputs["input_mode"],  # audio_projection_mode: int
    )
    dynamic_axes = {
        "audio_embeds": {0: "num_audios", 1: "num_frames", 2: "feature_size"},
        "audio_attention_mask": {0: "num_audios", 1: "num_frames"},
        "audio_sizes": {0: "num_audios"},
        "audio_features": {0: "num_audio_tokens"},
    }
    filename = "phi-4-mm-speech.onnx"

    temp_folder_1 = os.path.join(args.output, "speech_init_export")
    os.makedirs(temp_folder_1, exist_ok=True)

    fpath_1 = os.path.join(temp_folder_1, filename)
    torch._dynamo.config.capture_scalar_outputs = True
    ep = torch.export.export(
        model.model.embed_tokens_extend.audio_embed,
        args=dummy_inputs,
        strict=False,
        dynamic_shapes=[
            {
                0: torch.export.Dim.AUTO,
                1: torch.export.Dim.AUTO,
                2: torch.export.Dim.AUTO,
            },
            {0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO},
            {0: torch.export.Dim.AUTO},
            {0: torch.export.Dim.AUTO},
        ],
    )
    onnx_program = torch.onnx.export(
        ep,
        (),
        input_names=[
            "audio_embeds",
            "audio_attention_mask",
            "audio_sizes",
            "audio_projection_mode",
        ],
        output_names=["audio_features"],
    )
    onnx_program.optimize()
    onnx_program.save(fpath_1, external_data=True)

    onnx.checker.check_model(fpath_1)
    onnx.shape_inference.infer_shapes_path(fpath_1)
    onnx_model = onnx.load_model(fpath_1, load_external_data=True)

    temp_folder_2 = os.path.join(args.output, "speech_after_export")
    os.makedirs(temp_folder_2, exist_ok=True)

    fpath_2 = os.path.join(temp_folder_2, filename)
    onnx.save_model(
        onnx_model,
        fpath_2,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{filename}.data",
        size_threshold=0,
        convert_attribute=False,
    )
    shutil.rmtree(temp_folder_1)

    # ONNX/ORT rewriter
    temp_folder_3 = os.path.join(args.output, "speech_after_rewrite")
    os.makedirs(temp_folder_3, exist_ok=True)

    onnx_model = ir.load(fpath_2)
    DynamoOnnxHelper.fold_transpose_initializers(onnx_model)
    onnxscript.rewriter.rewrite(onnx_model)
    onnxscript.optimizer.optimize(
        onnx_model,
        onnx_shape_inference=False,
        input_size_limit=4 * 2048 * 2048,
        output_size_limit=4 * 2048 * 2048,
    )

    fpath_3 = os.path.join(temp_folder_3, filename)
    ir.save(onnx_model, fpath_3, external_data=f"{filename}.data")
    shutil.rmtree(temp_folder_2)

    onnx_model = onnx.load_model(fpath_3, load_external_data=True)
    # Fix labels of dynamic axes since they can't be specified during Dynamo export currently
    onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "num_audios"
    onnx_model.graph.input[0].type.tensor_type.shape.dim[1].dim_param = "num_frames"
    onnx_model.graph.input[1].type.tensor_type.shape.dim[0].dim_param = "num_audios"
    onnx_model.graph.input[1].type.tensor_type.shape.dim[1].dim_param = "num_frames"
    onnx_model.graph.input[2].type.tensor_type.shape.dim[0].dim_param = "num_audios"
    onnx_model.graph.output[0].type.tensor_type.shape.dim[
        0
    ].dim_param = "num_audio_tokens"

    onnx_model = DynamoOnnxHelper(onnx_model)
    onnx_model.convert_constants_to_initializers()
    onnx_model.clear_metadata()

    os.remove(fpath_3)
    os.remove(fpath_3 + ".data")
    onnx_model.model.save_model_to_file(
        fpath_3,
        use_external_data_format=True,
        all_tensors_to_one_file=True,
        convert_attribute=True,
    )  # convert_attribute = True needed because of ONNX/ORT rewriter

    # ORT transformer optimizer
    temp_folder_4 = os.path.join(args.output, "speech_after_opt")
    fpath_4 = os.path.join(temp_folder_4, filename)
    subprocess.run(
        [
            f"{sys.executable}",
            "-m",
            "onnxruntime.transformers.optimizer",
            "--input",
            fpath_3,
            "--output",
            fpath_4,
            "--model_type",
            "conformer",
            "--num_heads",
            str(16),
            "--hidden_size",
            str(1024),
            "--use_external_data_format",
            "--opt_level",
            str(0),
            "--disable_shape_inference",
            "--convert_attribute",
        ]
    )
    shutil.rmtree(temp_folder_3)

    # ORT 4-bits quantizer
    fpath_5 = os.path.join(args.output, filename)
    cmd = [
        f"{sys.executable}",
        "-m",
        "onnxruntime.quantization.matmul_4bits_quantizer",
        "--input_model",
        fpath_4,
        "--output_model",
        fpath_5,
        "--block_size",
        str(32),
    ]
    if args.precision == torch.float32:
        cmd.extend(["--accuracy_level", str(4)])
    subprocess.run(cmd)
    shutil.rmtree(temp_folder_4)


def build_embedding(args):
    # TorchScript export
    batch_size, sequence_length, num_image_tokens, num_audio_tokens = 2, 8, 2, 2
    inputs = {
        "input_ids": torch.randint(
            low=0,
            high=config.vocab_size,
            size=(batch_size, sequence_length),
            device=args.execution_provider.replace("dml", "cuda"),
            dtype=torch.int64,
        ),
        "image_features": torch.randn(
            num_image_tokens,
            config.hidden_size,
            device=args.execution_provider.replace("dml", "cuda"),
            dtype=args.precision,
        ),
        "audio_features": torch.randn(
            num_audio_tokens,
            config.hidden_size,
            device=args.execution_provider.replace("dml", "cuda"),
            dtype=args.precision,
        ),
    }
    inputs["input_ids"][0][0] = -1
    inputs["input_ids"][0][1] = -1
    inputs["input_ids"][0][2] = -10000
    inputs["input_ids"][0][3] = -10000
    dummy_inputs = (
        inputs["input_ids"],  # input_ids: torch.LongTensor
        inputs["image_features"],  # image_features: Optional[torch.FloatTensor] = None,
        inputs["audio_features"],  # audio_features: Optional[torch.FloatTensor] = None,
    )
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "image_features": {0: "num_image_tokens"},
        "audio_features": {0: "num_audio_tokens"},
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
    }
    filename = "phi-4-mm-embedding.onnx"

    temp_folder_1 = os.path.join(args.output, "embedding_init_export")
    os.makedirs(temp_folder_1, exist_ok=True)

    fpath_1 = os.path.join(temp_folder_1, filename)
    torch.onnx.export(
        model.model.combined_embed,
        args=dummy_inputs,
        f=fpath_1,
        export_params=True,
        input_names=["input_ids", "image_features", "audio_features"],
        output_names=["inputs_embeds"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )

    onnx.checker.check_model(fpath_1)
    onnx.shape_inference.infer_shapes_path(fpath_1)
    onnx_model = onnx.load_model(fpath_1, load_external_data=True)

    fpath_2 = os.path.join(args.output, filename)
    onnx.save_model(
        onnx_model,
        fpath_2,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{filename}.data",
        size_threshold=0,
        convert_attribute=False,
    )
    shutil.rmtree(temp_folder_1)


def build_text(args):
    # Create ONNX model
    model_name = None
    precision = "int4"
    extra_options = {
        "exclude_embeds": "true",
        "filename": "phi-4-mm-text.onnx",
    }
    if args.precision == torch.float32:
        extra_options["int4_accuracy_level"] = 4
    create_model(
        model_name,
        args.input,
        args.output,
        precision,
        args.execution_provider,
        args.cache_dir,
        **extra_options,
    )


def build_adapters(args):
    # setattr(args, 'use_ortvalue', True)
    # build_float_adapters(args)

    setattr(args, "use_ortvalue", False)
    build_quantized_adapters(args)


def extract_adapters_from_torch(args):
    # Extract LoRAs from PyTorch model
    hidden_size = config.hidden_size
    num_kv_heads = config.num_key_value_heads
    num_attn_heads = config.num_attention_heads
    head_size = hidden_size // num_attn_heads

    q_size = num_attn_heads * head_size
    kv_size = num_kv_heads * head_size
    intermediate_size = config.intermediate_size

    vision_scaling = config.vision_lora["lora_alpha"] / config.vision_lora["r"]
    speech_scaling = config.speech_lora["lora_alpha"] / config.speech_lora["r"]

    vision_adapters = {}
    speech_adapters = {}
    for key, val in model.state_dict().items():
        # Map name in graph as key
        new_dict = {}
        key = (
            key.replace("self_attn", "attn")
            .replace("lora_A", "lora_A.MatMul")
            .replace("lora_B", "lora_B.MatMul")
        )

        if "lora_A" in key:
            # LoRA_A is shared across projections
            if "qkv_proj" in key:
                new_dict[key.replace("qkv_proj", "q_proj")] = val
                new_dict[key.replace("qkv_proj", "k_proj")] = val
                new_dict[key.replace("qkv_proj", "v_proj")] = val
            elif "gate_up_proj" in key:
                new_dict[key.replace("gate_up_proj", "gate_proj")] = val
                new_dict[key.replace("gate_up_proj", "up_proj")] = val
            else:
                new_dict[key] = val

        elif "lora_B" in key:
            # LoRA_B is split across projections
            if "qkv_proj" in key:
                new_dict[key.replace("qkv_proj", "q_proj")] = val[:q_size, :]
                new_dict[key.replace("qkv_proj", "k_proj")] = val[
                    q_size : q_size + kv_size, :
                ]
                new_dict[key.replace("qkv_proj", "v_proj")] = val[q_size + kv_size :, :]
            elif "gate_up_proj" in key:
                new_dict[key.replace("gate_up_proj", "gate_proj")] = val[
                    :intermediate_size, :
                ]
                new_dict[key.replace("gate_up_proj", "up_proj")] = val[
                    intermediate_size:, :
                ]
            else:
                new_dict[key] = val

        else:
            continue

        for new_key, new_val in new_dict.items():
            new_key = new_key.replace(".vision", "").replace(".speech", "")
            if "vision" in key:
                np_data = new_val.detach().cpu().to(args.precision).numpy().transpose()
                if "lora_B" in key:
                    np_data *= vision_scaling
                vision_adapters[new_key] = (
                    ort.OrtValue.ortvalue_from_numpy(np_data)
                    if args.use_ortvalue
                    else np_data
                )
            elif "speech" in key:
                np_data = new_val.detach().cpu().to(args.precision).numpy().transpose()
                if "lora_B" in key:
                    np_data *= speech_scaling
                speech_adapters[new_key] = (
                    ort.OrtValue.ortvalue_from_numpy(np_data)
                    if args.use_ortvalue
                    else np_data
                )
            else:
                raise ValueError(f"Unknown LoRA key found: {key}")

    return vision_adapters, speech_adapters


def build_onnx_adapters(vision_adapters, speech_adapters):
    # Convert vision LoRAs
    adapter_format = ort.AdapterFormat()
    adapter_format.set_adapter_version(1)
    adapter_format.set_model_version(1)
    adapter_format.set_parameters(vision_adapters)
    adapter_format.export_adapter(
        os.path.join(args.output, "phi-4-mm-vision.onnx_adapter")
    )

    # Convert speech LoRAs
    adapter_format = ort.AdapterFormat()
    adapter_format.set_adapter_version(1)
    adapter_format.set_model_version(1)
    adapter_format.set_parameters(speech_adapters)
    adapter_format.export_adapter(
        os.path.join(args.output, "phi-4-mm-speech.onnx_adapter")
    )

    # Convert LoRA weights in ONNX model to inputs
    filename = "phi-4-mm-text.onnx"
    fpath = os.path.join(args.output, filename)
    onnx_model = onnx.load_model(fpath)

    to_proto = {
        "tensor(int8)": TensorProto.INT8,
        "tensor(uint8)": TensorProto.UINT8,
        "tensor(float16)": TensorProto.FLOAT16,
        "tensor(float)": TensorProto.FLOAT,
    }
    for key, val in vision_adapters.items():
        # Handle different sized feature dimensions between adapters by using dynamic axes
        shape = val.shape()
        if "lora_A.MatMul.weight_Q4" in key:
            shape[0] = "out_features"
        elif "lora_B.MatMul.weight_Q4" in key:
            shape[1] = "(in_features + block_size - 1) // block_size"
        elif (
            "lora_A.MatMul.weight_scales" in key or "lora_B.MatMul.weight_scales" in key
        ):
            shape[0] = "in_features * out_features / block_size"
        elif "lora_A.MatMul.weight" in key:
            shape[1] = "out_features"
        elif "lora_B.MatMul.weight" in key:
            shape[0] = "in_features"

        new_input = helper.make_tensor_value_info(key, to_proto[val.data_type()], shape)
        onnx_model.graph.input.extend([new_input])
        for initializer in onnx_model.graph.initializer:
            if initializer.name == key:
                # Add 0-filled static initializer for when LoRA isn't used
                # since size of inner dims in LoRA path don't matter
                zero_initializer = helper.make_tensor(
                    name=initializer.name,
                    data_type=initializer.data_type,
                    dims=val.shape(),
                    vals=np.zeros(
                        val.shape(),
                        dtype=helper.tensor_dtype_to_np_dtype(initializer.data_type),
                    ).flatten(),
                )
                onnx_model.graph.initializer.remove(initializer)
                onnx_model.graph.initializer.append(zero_initializer)
                break

    os.remove(fpath)
    os.remove(fpath + ".data")
    onnx.save_model(
        onnx_model,
        fpath,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{filename}.data",
        size_threshold=0,
        convert_attribute=False,
    )


def build_float_adapters(args):
    vision_adapters, speech_adapters = extract_adapters_from_torch(args)
    build_onnx_adapters(vision_adapters, speech_adapters)


def build_adapter_only_onnx_model(args, adapters, filename, fpath):
    inputs, outputs, initializers, value_infos, nodes = [], [], [], [], []
    dtype = (
        TensorProto.FLOAT16 if args.precision == torch.float16 else TensorProto.FLOAT
    )
    for key, val in adapters.items():
        # Create input and output
        inputs.append(
            helper.make_tensor_value_info(
                f"input.{key}", dtype, ["batch_size", "sequence_length", val.shape[0]]
            )
        )
        outputs.append(
            helper.make_tensor_value_info(
                f"output.{key}", dtype, ["batch_size", "sequence_length", val.shape[1]]
            )
        )

        # Create initializer data
        tensor = numpy_helper.from_array(val)
        tensor.name = key
        initializers.append(tensor)

        # Create MatMul node
        matmul_node = helper.make_node(
            "MatMul",
            inputs=[inputs[-1].name, tensor.name],
            outputs=[outputs[-1].name],
            name=f"node.{key}",
        )
        nodes.append(matmul_node)

    model = helper.make_model(
        opset_imports=[helper.make_operatorsetid("", 14)],
        ir_version=7,
        producer_name="onnxruntime-genai",
        producer_version="0.0.0",
        graph=helper.make_graph(
            name="main_graph",
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
            value_info=value_infos,
            nodes=nodes,
        ),
    )
    onnx.save_model(
        model,
        fpath,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{filename}.data",
        size_threshold=0,
        convert_attribute=False,
    )


def extract_adapters_from_onnx(args, fpath):
    adapters = {}
    model = onnx.load_model(fpath)
    for initializer in model.graph.initializer:
        val = numpy_helper.to_array(initializer)
        adapters[initializer.name] = ort.OrtValue.ortvalue_from_numpy(val)
    return adapters


def build_quantized_adapters(args):
    # 1. Extract LoRAs from PyTorch model
    vision_adapters, speech_adapters = extract_adapters_from_torch(args)

    # 2. Put LoRAs into separate ONNX models
    filename = "phi-4-mm-lora-vision.onnx"
    fpath_1 = os.path.join(args.output, filename)
    vision_model = build_adapter_only_onnx_model(
        args, vision_adapters, filename, fpath_1
    )

    filename = "phi-4-mm-lora-speech.onnx"
    fpath_2 = os.path.join(args.output, filename)
    speech_model = build_adapter_only_onnx_model(
        args, speech_adapters, filename, fpath_2
    )

    # 3. Quantize ONNX models to int4
    filename = "phi-4-mm-qlora-vision.onnx"
    fpath_3 = os.path.join(args.output, filename)
    cmd = [
        f"{sys.executable}",
        "-m",
        "onnxruntime.quantization.matmul_4bits_quantizer",
        "--input_model",
        fpath_1,
        "--output_model",
        fpath_3,
        "--block_size",
        str(32),
    ]
    if args.precision == torch.float32:
        cmd.extend(["--accuracy_level", str(4)])
    subprocess.run(cmd)

    filename = "phi-4-mm-qlora-speech.onnx"
    fpath_4 = os.path.join(args.output, filename)
    cmd = [
        f"{sys.executable}",
        "-m",
        "onnxruntime.quantization.matmul_4bits_quantizer",
        "--input_model",
        fpath_2,
        "--output_model",
        fpath_4,
        "--block_size",
        str(32),
    ]
    if args.precision == torch.float32:
        cmd.extend(["--accuracy_level", str(4)])
    subprocess.run(cmd)

    os.remove(fpath_1)
    os.remove(fpath_1 + ".data")
    os.remove(fpath_2)
    os.remove(fpath_2 + ".data")

    # 4. Extract quantized LoRAs from ONNX models
    vision_adapters = extract_adapters_from_onnx(args, fpath_3)
    speech_adapters = extract_adapters_from_onnx(args, fpath_4)

    # 5. Store quantized LoRAs in adapter files
    build_onnx_adapters(vision_adapters, speech_adapters)

    os.remove(fpath_3)
    os.remove(fpath_3 + ".data")
    os.remove(fpath_4)
    os.remove(fpath_4 + ".data")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to folder on disk containing the Hugging Face config, model, tokenizer, etc.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to folder to store ONNX model and additional files (e.g. GenAI config, external data files, etc.)",
    )

    parser.add_argument(
        "-p",
        "--precision",
        required=True,
        choices=["fp16", "fp32"],
        help="Precision to export PyTorch components with",
    )

    parser.add_argument(
        "-e",
        "--execution_provider",
        required=True,
        choices=["cpu", "cuda", "dml"],
        help="Execution provider for Phi-4 multimodal components",
    )

    parser.add_argument(
        "-c",
        "--cache_dir",
        required=False,
        default=os.path.join(".", "cache_dir"),
        help="Cache directory for Hugging Face files and temporary ONNX external data files",
    )

    args = parser.parse_args()
    args.precision = torch.float16 if args.precision == "fp16" else torch.float32
    return args


if __name__ == "__main__":
    user_prompt = "<|user|>\n"
    assistant_prompt = "<|assistant|>\n"
    prompt_suffix = "<|end|>\n"

    args = get_args()
    config = AutoConfig.from_pretrained(args.input, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.input, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.input, trust_remote_code=True, torch_dtype=args.precision
    ).to(args.execution_provider.replace("dml", "cuda"))

    # Build model components
    build_vision(args)
    build_speech(args)
    build_embedding(args)
    build_text(args)
    build_adapters(args)
