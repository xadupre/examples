import os
import onnx
from onnx_extended.tools import load_model, save_model, load_external
from onnx_extended.tools.onnx_manipulations import select_model_inputs_outputs

root = os.path.dirname(__file__)

llama = (
    "/home/xadupre/github/Llama-2-Onnx/7B_FT_float16/ONNX/LlamaV2_7B_FT_float16.onnx"
)

print("load model")
onx = load_model(llama, external=False)

print("extract model")
outputs = ["/transformer/block_list.1/attention/Gather_output_0"]
new_onx = select_model_inputs_outputs(onx, outputs)

print("load external data")
load_external(new_onx, os.path.dirname(llama))

print("save model")
model = os.path.join(root, "models")
if not os.path.exists(model):
    os.mkdir(model)
name = os.path.join(model, "llama_16_block_list_1.onnx")
save_model(new_onx, name, external=False)

print("done.")
