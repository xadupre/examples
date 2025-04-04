import json
import os
import pprint
import soundfile
import torch
import transformers
from onnx_diagnostic.helpers import string_type
from phi.configuration_phi4mm import Phi4MMConfig
from phi.modeling_phi4mm import Phi4MMForCausalLM
from phi.processing_phi4mm import InputMode


def export_speech(
    model,
    execution_provider="cuda",
    filename="phi-4-mm-speech.onnx",
    output="speech_init_export",
    precision="float32",
):
    sound1 = os.path.join(
        os.path.dirname(__file__), "examples", "what_is_shown_in_this_image.wav"
    )
    sound2 = os.path.join(
        os.path.dirname(__file__),
        "examples",
        "what_is_the_traffic_sign_in_the_image.wav",
    )

    user_prompt = "<|user|>\n"
    assistant_prompt = "<|assistant|>\n"
    prompt_suffix = "<|end|>\n"

    prompt = f"{user_prompt}<|audio_1|>\n<|audio_2|>\nWhat are the stories that these audios come from?{prompt_suffix}{assistant_prompt}"

    print("-- loading audio")
    audio1 = soundfile.read(sound1)
    audio2 = soundfile.read(sound2)
    print("-- loading processor")
    processor = transformers.AutoProcessor.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
    )
    print("-- creating inputs")
    inputs = processor(prompt, audios=[audio1, audio2], return_tensors="pt").to(
        execution_provider
    )
    inputs["input_audio_embeds"] = inputs["input_audio_embeds"].to(
        getattr(torch, precision)
    )

    # TorchScript export
    dummy_inputs = (
        inputs["input_audio_embeds"],  # audio_embeds: torch.FloatTensor
        inputs["audio_attention_mask"],  # audio_attention_mask: torch.BoolTensor
        inputs["audio_embed_sizes"],  # audio_sizes: torch.LongTensor
        inputs["input_mode"],  # audio_projection_mode: int
    )
    print(f"-- runs the model with {string_type(dummy_inputs, with_shape=True)}")
    expected = model.model.embed_tokens_extend.audio_embed(*dummy_inputs, wte=model.model.embed_tokens)

    dynamic_axes = {
        "audio_embeds": {0: "num_audios", 1: "num_frames", 2: "feature_size"},
        "audio_attention_mask": {0: "num_audios", 1: "num_frames"},
        "audio_sizes": {0: "num_audios"},
        "audio_features": {0: "num_audio_tokens"},
    }

    torch._dynamo.config.capture_scalar_outputs = True

    print(f"-- exporting with {string_type(dummy_inputs, with_shape=True)}")
    dynamic_shapes = tuple(dynamic_axes.values())
    pprint.pprint(dynamic_shapes)
    export_ds = (
        {
            0: torch.export.Dim.DYNAMIC,
            1: torch.export.Dim.DYNAMIC,
            2: torch.export.Dim.DYNAMIC,
        },
        {0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},
        {0: torch.export.Dim.DYNAMIC},
        {0: torch.export.Dim.DYNAMIC},
    )
    ep = torch.export.export(
        model.model.embed_tokens_extend.audio_embed,
        args=dummy_inputs,
        kwargs=dict(wte=model.model.embed_tokens),
        strict=False,
        dynamic_shapes=export_ds,
    )

    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, filename + ".ep"), "w") as f:
        f.write(str(ep))

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
    fpath_1 = os.path.join(output, filename)
    onnx_program.save(fpath_1, external_data=True)


if __name__ == "__main__":
    config_filename = os.path.join(os.path.dirname(__file__), "phi", "config.json")
    with open(config_filename) as f:
        config = json.load(f)

    config["num_hidden_layers"] = 2
    config["_attn_implementation"] = "eager"
    conf = Phi4MMConfig(**config)
    model = Phi4MMForCausalLM(conf)
    model.eval()
    export_speech(model)
