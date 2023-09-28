# Benchmark float 8

## Preparation steps

The benchmarks are implemented with
[onnx-extended](https://github.com/sdpython/onnx-extended).

```bash
git clone https://github.com/sdpython/onnx-extended.git 
cd onnx-extended 
python -m pip install -r requirements-dev.txt 
python setup.py build_ext --inplace --cuda-version=11.8 \
       --ort-verion=<path to the built onnxruntime or empty to get the published version>
export PYTHONPATH=. 
```

## Benchmark 1: cublasLtMatMul

Measure performance of cublasLtMatMul for different types and dimensions
on two square matrices.

```bash
python _doc/examples/plot_bench_gemm_f8.py 
```

## Benchmark: bert-squad

See [PyTorch_Bert-Squad_OnnxRuntime_CPU.ipynb](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_CPU.ipynb)

**download the data**

```bash
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

**convert the model into onnx and dynamically quantize it (static, dynamic, int8, uint8)**

```bash
python torch_up_to_onnx.py
```

**quantize to float 8 e4m3fn**

The quantization is a custom one. It only converts a *MatMul* into
a sequence *Transpose + DynamicQuantizeLinear + GemmFloat8*.

```bash
python -m onnx_extended quantize -i bert-base-cased-squad.onnx -o bert-base-cased-squad-dyn-fp8.onnx -v -v -k fp8 -q -x /qa_outputs/MatMul
python -m onnx_extended quantize -i bert-base-cased-squad.onnx -o bert-base-cased-squad-dyn-fp8-ext.onnx -s onnx-extended -v -v -k fp8 -q -x /qa_outputs/MatMul
```

**quantize to float 16**

The quantization is a custom one. It casts all float constant tensor to float 16.
It does the same for inputs and outputs as well.

```bash
python -m onnx_extended quantize -i bert-base-cased-squad.onnx -o bert-base-cased-squad-fp16.onnx -v -v -k fp16 -q
```

**benchmark**

```bash
python benchmark.py
```

**profiling**

```bash
python profiles.py
```

## Results Example

```
restoring dataset 'dataset.pkl'
creating inference 'bert-base-cased-squad.onnx'
starting benchmark 'bert-base-cased-squad.onnx'
try 1: ort inference time = 187.06 ms
try 2: ort inference time = 187.69 ms
creating inference 'bert-base-cased-squad-int8.onnx'
starting benchmark 'bert-base-cased-squad-int8.onnx'
try 1: ort inference time = 268.20 ms
try 2: ort inference time = 265.09 ms
```
