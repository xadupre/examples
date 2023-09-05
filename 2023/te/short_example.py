import os
import pprint
import warnings
import torch
from torch.onnx import export as torch_export


def create_fp8_recipe():
    from transformer_engine.common import recipe

    return recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)


def te_export(
    model: torch.nn.Module,
    inp: torch.Tensor,
    fname: str,
    use_fp8: bool = True,
    opset: int = 15,
    input_names: list = ["input"],
    output_names: list = ["output"],
    ONNX_FILES_DIR=".",
):
    """
    Export to ONNX
    Taken from the unit test of TransformerEngine.
    """
    import transformer_engine.pytorch as te

    fp8_recipe = create_fp8_recipe()

    with (
        torch.inference_mode(),
        te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe),
        warnings.catch_warnings(),
    ):
        warnings.filterwarnings(
            action="ignore", category=torch.jit.TracerWarning, module=r".*"
        )

        model.cuda().eval()
        os.makedirs(ONNX_FILES_DIR, exist_ok=True)
        fname = os.path.join(ONNX_FILES_DIR, fname)
        torch.onnx.export(
            model,
            inp if isinstance(inp, list) or isinstance(inp, tuple) else (inp,),
            fname,
            verbose=False,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
        )


class BasicLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, layernorm_eps: int = 1e-5):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x):
        x1 = self.linear1(x)
        return torch.nn.functional.gelu(x1, approximate="tanh")


dtype = torch.float16
hidden_size = 128
sequence_length = 64
x = torch.rand(sequence_length, hidden_size).cuda().to(dtype=dtype)
y = torch.rand(sequence_length, 1).cuda().to(dtype=dtype)

layer = BasicLayer(hidden_size)
layer.to(dtype=dtype).cuda()


timing_iters = 10


def train_torch(layer, x, y, timing_iters=10):
    for _ in range(timing_iters):
        output = layer(x)
        loss = ((output - y) ** 2).sum(axis=0)
        layer.zero_grad()
        loss.backward()


print("training BasicLayer")
train_torch(layer, x, y)
print("done.")

print("TransformerEngine")
import transformer_engine.pytorch as te


class TeBasicLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, layernorm_eps: int = 1e-5):
        super().__init__()
        self.linear1 = te.Linear(hidden_size, 1, bias=True)

    def forward(self, x):
        x1 = self.linear1(x)
        return torch.nn.functional.gelu(x1, approximate="tanh")


layer_te = TeBasicLayer(hidden_size)
layer_te.to(dtype=dtype).cuda()


def train_te(layer, x, y, timing_iters=10, fp8=False):
    fp8_autocast_kwargs = {"enabled": fp8}
    for _ in range(timing_iters):
        with te.fp8_autocast(**fp8_autocast_kwargs):
            output = layer(x)
            loss = ((output - y) ** 2).sum(axis=0)
        layer.zero_grad()
        loss.backward()


print("training TeBasicLayer")
train_te(layer_te, x, y)
print("done.")

print("#################################################")
print("to_onnx - torch")
# state_dict = layer.state_dict(keep_vars=True)
state_dict = layer.state_dict()
for k, v in state_dict.items():
    print("TO", k, v.shape, v.dtype)

torch_export(
    layer, x, "basic.torch.onnx", verbose=False, input_names=["X"], output_names=["Y"]
)

print("#################################################")
state_dict = layer_te.state_dict(keep_vars=True)
state_dict = layer_te.state_dict()
for k, v in state_dict.items():
    print("TE:", k, type(v), v.dtype)
print("to_onnx - torch - TransformerEngine")
# This does not work when get_extra_state is overriden.
# In TransformerEngine, it returns None when fp8 is disabled.
# But even though, it says only tensors can be traced in
# torch._C._create_graph_by_tracing.
te_export(
    layer_te, x, "basic.te.onnx", input_names=["X"], output_names=["Y"], use_fp8=False
)
print("to_onnx - torch - TransformerEngine - f8")
try:
    te_export(
        layer_te,
        x,
        "basic.te.fp8.onnx",
        input_names=["X"],
        output_names=["Y"],
        use_fp8=True,
    )
except AssertionError as e:
    print(e)
print("#################################################")


if False:
    from pyquickhelper.pycode.profiling import profile
    from pyquickhelper.pycode.profiling import profile2graph

    stat, text = profile(lambda: train_torch(layer, x, y))
    gr = profile2graph(stat)
    print(gr[0].to_text(fct_width=80))

    stat, text = profile(lambda: train_te(layer_te, x, y))
    gr = profile2graph(stat)
    print(gr[0].to_text(fct_width=80))


print("Same with scaling")


class BasicLayerSoft(torch.nn.Module):
    def __init__(self, hidden_size: int, layernorm_eps: int = 1e-5):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_size, 1, bias=True)
        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.linear1(x)
        y = self.soft(x1)
        return y


layer_norm = BasicLayerSoft(hidden_size)
layer_norm.to(dtype=dtype).cuda()
layer_norm(x).shape

print("training BasicLayerSoft")
train_torch(layer_norm, x, y)
print("done.")


class TeBasicLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, layernorm_eps: int = 1e-5):
        super().__init__()
        self.linear1 = te.Linear(hidden_size, 1, bias=True)
        self.norm = te.LayerNormLinear(1, 1)

    def forward(self, x):
        x1 = self.linear1(x)
        y = self.norm(x1)
        return y


print("TransformerEngine: TeBasicLayerNorm")
te_layer_norm = TeBasicLayerNorm(hidden_size)
te_layer_norm.to(dtype=dtype).cuda()
te_layer_norm(x).shape


print("training TeBasicLayerNorm")
train_te(te_layer_norm, x, y)
print("done.")


print("#################################################")
print("to_onnx - torch")
from torch.onnx import export

torch_export(
    layer_norm,
    x,
    "norm.torch.onnx",
    verbose=False,
    input_names=["X"],
    output_names=["Y"],
)

print("#################################################")
print("to_onnx - torch - TransformerEngine")

# Once get_extra_state is removed. It fails here in _trace_and_get_graph_from_model (C++ issue).
# Gemm is used in the normalization. It goes through one of the cpp extension.
# A custom gemm function is called with a preallocated buffer: gemm(input, output) -> None.
# This call is made in a torch function and then encapsulated in a torch module.
# My guess it breaks the tracing somehow.
# https://github.com/microsoft/onnxruntime-extensions/blob/main/tutorials/pytorch_custom_ops_tutorial.ipynb

te_export(
    te_layer_norm,
    x,
    "norm.te.onnx",
    input_names=["X"],
    output_names=["Y"],
    use_fp8=False,
)
te_export(
    te_layer_norm,
    x,
    "norm.te.fp8.onnx",
    input_names=["X"],
    output_names=["Y"],
    use_fp8=False,
)


if False:
    stat, text = profile(lambda: train_te(te_layer_norm, x, y))
    gr = profile2graph(stat)
    print(gr[0].to_text(fct_width=80))

    global_in = True

    def tracefunc(frame, event, arg, indent=[0]):
        global global_in
        if event == "call":
            if frame.f_code.co_name == "tracefunc":
                global_in = False
            if global_in:
                filename = frame.f_code.co_filename
                filename = filename.split("site-packages")[-1]
                filename = filename.split("python3.10")[-1]
                line = frame.f_code.co_firstlineno
                print(
                    "-" * indent[0] + f"> {frame.f_code.co_name!r} - {filename}:{line}"
                )
                indent[0] += 2
        elif event == "return":
            indent[0] -= 2
            if global_in:
                print("-" * indent[0] + f"< {frame.f_code.co_name!r}")
            if frame.f_code.co_name == "tracefunc":
                global_in = True
        return tracefunc

    import sys

    f = sys.getprofile()
    sys.setprofile(tracefunc)
    train_te(te_layer_norm, x, y, 2)
    sys.setprofile(f)
    f = sys.getprofile()
    sys.setprofile(tracefunc)
    train_te(te_layer_norm, x, y, 2, fp8=True)
    sys.setprofile(f)
    sys.setprofile(f)
    train_te(te_layer_norm, x, y, 2, fp8=True)
