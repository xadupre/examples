# Benchmark float 8

## Preparation steps

The benchmarks are implemented with
[onnx-extended](https://github.com/xadupre/onnx-extended).

```bash
git clone https://github.com/xadupre/onnx-extended.git 
cd onnx-extended 
python -m pip install -r requirements-dev.txt 
python setup.py build_ext --inplace --cuda-version=11.8 
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

**convert the model into onnx and dynamically quantize it**

```bash
python torch_up_to_onnx.py
```

**quantize**

The quantization is a custom one. It only converts a *MatMul* into
a sequence *Transpose + DynamicQuantizeLinear + GemmFloat8*.

```bash
python3 -m onnx_extended quantize -i bert-base-cased-squad.onnx -o bert-base-cased-squad-fp8.onnx -v -v -k fp8 -q
```

**benchmark**

```bash
python benchmark.py
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
