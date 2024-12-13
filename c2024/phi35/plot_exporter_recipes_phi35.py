import copy
from typing import Any, Dict
import torch
import transformers
from onnx_export_errors import bypass_export_some_errors

true_model = True
experimental = False


def get_phi35(
    batch_size: int = 2, true_model: bool = False, **kwargs
) -> Dict[str, Any]:
    """
    Gets a non initialized model with its inputs

    :param batch_size: batch size
    :param true_model: convert the true model
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :return: dictionary

    See `Phi-3.5-mini-instruct/config.json
    <https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/config.json>`_.
    """
    if not true_model:
        config = {
            "_name_or_path": "Phi-3.5-mini-instruct",
            "architectures": ["Phi3ForCausalLM"],
            "attention_dropout": 0.0,
            "auto_map": {
                "AutoConfig": "configuration_phi3.Phi3Config",
                "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM",
            },
            "bos_token_id": 1,
            "embd_pdrop": 0.0,
            "eos_token_id": 32000,
            "hidden_act": "silu",
            "hidden_size": 3072,
            "initializer_range": 0.02,
            "intermediate_size": 8192,
            "max_position_embeddings": 131072,
            "model_type": "phi3",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32,
            "original_max_position_embeddings": 4096,
            "pad_token_id": 32000,
            "resid_pdrop": 0.0,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "long_factor": [
                    1.0800000429153442,
                    1.1100000143051147,
                    1.1399999856948853,
                    1.340000033378601,
                    1.5899999141693115,
                    1.600000023841858,
                    1.6200000047683716,
                    2.620000123977661,
                    3.2300000190734863,
                    3.2300000190734863,
                    4.789999961853027,
                    7.400000095367432,
                    7.700000286102295,
                    9.09000015258789,
                    12.199999809265137,
                    17.670000076293945,
                    24.46000099182129,
                    28.57000160217285,
                    30.420001983642578,
                    30.840002059936523,
                    32.590003967285156,
                    32.93000411987305,
                    42.320003509521484,
                    44.96000289916992,
                    50.340003967285156,
                    50.45000457763672,
                    57.55000305175781,
                    57.93000411987305,
                    58.21000289916992,
                    60.1400032043457,
                    62.61000442504883,
                    62.62000274658203,
                    62.71000289916992,
                    63.1400032043457,
                    63.1400032043457,
                    63.77000427246094,
                    63.93000411987305,
                    63.96000289916992,
                    63.970001220703125,
                    64.02999877929688,
                    64.06999969482422,
                    64.08000183105469,
                    64.12000274658203,
                    64.41000366210938,
                    64.4800033569336,
                    64.51000213623047,
                    64.52999877929688,
                    64.83999633789062,
                ],
                "short_factor": [
                    1.0,
                    1.0199999809265137,
                    1.0299999713897705,
                    1.0299999713897705,
                    1.0499999523162842,
                    1.0499999523162842,
                    1.0499999523162842,
                    1.0499999523162842,
                    1.0499999523162842,
                    1.0699999332427979,
                    1.0999999046325684,
                    1.1099998950958252,
                    1.1599998474121094,
                    1.1599998474121094,
                    1.1699998378753662,
                    1.2899998426437378,
                    1.339999794960022,
                    1.679999828338623,
                    1.7899998426437378,
                    1.8199998140335083,
                    1.8499997854232788,
                    1.8799997568130493,
                    1.9099997282028198,
                    1.9399996995925903,
                    1.9899996519088745,
                    2.0199997425079346,
                    2.0199997425079346,
                    2.0199997425079346,
                    2.0199997425079346,
                    2.0199997425079346,
                    2.0199997425079346,
                    2.0299997329711914,
                    2.0299997329711914,
                    2.0299997329711914,
                    2.0299997329711914,
                    2.0299997329711914,
                    2.0299997329711914,
                    2.0299997329711914,
                    2.0299997329711914,
                    2.0299997329711914,
                    2.0799996852874756,
                    2.0899996757507324,
                    2.189999580383301,
                    2.2199995517730713,
                    2.5899994373321533,
                    2.729999542236328,
                    2.749999523162842,
                    2.8399994373321533,
                ],
                "type": "longrope",
            },
            "rope_theta": 10000.0,
            "sliding_window": 262144,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "attention_bias": False,
            "vocab_size": 32064,
        }
        config.update(**kwargs)
        conf = transformers.Phi3Config(**config)
        model = transformers.Phi3ForCausalLM(conf)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct",
            device_map="cpu",
            torch_dtype=kwargs.get("torch_dtype", "bfloat16"),
            trust_remote_code=True,
        )

    model.eval()

    batch = torch.export.Dim("batch")
    seq_length = torch.export.Dim("seq_length")
    shapes = {}

    dtype = getattr(torch, kwargs.get("torch_dtype", "bfloat16"))

    # You can get the cache dimension by running the model without it.
    n_layers = 32 if true_model else config["num_hidden_layers"]
    cache = transformers.cache_utils.DynamicCache(n_layers)
    for i in range(n_layers):
        cache.update(
            torch.randn(batch_size, 32, 30, 96).to(dtype),
            torch.randn(batch_size, 32, 30, 96).to(dtype),
            i,
        )
    cache2 = transformers.cache_utils.DynamicCache(n_layers)
    for i in range(n_layers):
        cache2.update(
            torch.randn(batch_size + 1, 32, 31, 96).to(dtype),
            torch.randn(batch_size + 1, 32, 31, 96).to(dtype),
            i,
        )

    inputs = dict(
        input_ids=torch.randint(0, 32063, (batch_size, 3)).to(torch.int64),
        attention_mask=torch.ones((batch_size, 33)).to(torch.int64),
        past_key_values=cache,
    )
    inputs2 = dict(
        input_ids=torch.randint(0, 32063, (batch_size + 1, 4)).to(torch.int64),
        attention_mask=torch.ones((batch_size + 1, 35)).to(torch.int64),
        past_key_values=cache2,
    )
    n = len(cache.key_cache)
    cache_length = torch.export.Dim("cache_length")
    shapes.update(
        {
            "input_ids": {0: batch, 1: seq_length},
            "attention_mask": {
                0: batch,
                1: torch.export.Dim.DYNAMIC,  # cache_length + seq_length
            },
            "past_key_values": [
                [{0: batch, 2: cache_length} for _ in range(n)],  # 0: batch,
                [{0: batch, 2: cache_length} for _ in range(n)],  # 0: batch,
            ],
        }
    )

    return dict(inputs=inputs, model=model, dynamic_shapes=shapes, inputs2=inputs2)


if true_model:
    data = None
else:
    print(f"load model true_model={true_model}...")
    data = get_phi35(num_hidden_layers=2)
    print("done.")

    model = data["model"]
    inputs = data["inputs"]
    dynamic_shapes = data["dynamic_shapes"]

    ###################################
    # Let's check it is working.
    # We need to copy the input before calling the model
    # because it modified the inputs and they are not properly
    # set up when the export starts.
    if not true_model:
        # too long
        model(**copy.deepcopy(inputs))

###################################
# Let's export.
name = (
    f"plot_exporter_recipes_phi35{'-true' if true_model else ''}"
    f"{'-experimental' if experimental else ''}.onnx"
)

print(f"start conversion experimental={experimental}...")
if experimental:
    from experimental_experiment.torch_interpreter import to_onnx, ExportOptions

    with bypass_export_some_errors(
        patch_transformers=True, replace_dynamic_cache=True, verbose=1
    ) as modificator:
        if true_model:
            # This section must be under this section to avoid creation true DynamicCache
            print(f"load model true_model={true_model}...")
            # bfloat16 not fully supported by this exporter
            data = get_phi35(true_model=True, torch_dtype="float16")
            print("done")
            model = data["model"]
            inputs = data["inputs"]
            dynamic_shapes = data["dynamic_shapes"]

        inputs = modificator(inputs)
        large_onx = to_onnx(
            model,
            (),
            inputs,
            dynamic_shapes=dynamic_shapes,
            export_options=ExportOptions(strict=False),
            large_model=True,
            verbose=1,
        )
        print("save the model")
        large_onx.save(name, all_tensors_to_one_file=True)
else:
    with bypass_export_some_errors(
        patch_transformers=True, replace_dynamic_cache=True, verbose=1
    ) as modificator:
        if true_model:
            # This section must be under this section to avoid creation true DynamicCache
            print(f"load model true_model={true_model}...")
            data = get_phi35(true_model=True)
            print("done")
            model = data["model"]
            inputs = data["inputs"]
            dynamic_shapes = data["dynamic_shapes"]

        inputs = modificator(inputs)
        ep = torch.onnx.export(
            model, (), kwargs=inputs, dynamic_shapes=dynamic_shapes, dynamo=True
        )
        ep.optimize()
        print("save the model")
        ep.save(name)
print("done.")
