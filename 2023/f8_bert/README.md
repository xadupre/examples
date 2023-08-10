# Benchmark float 8

## Preparation steps

The benchamrks are implemented with
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

**download the data**

```bash
wget https://github.com/onnx/models/raw/main/text/machine_comprehension/bert-squad/model/bertsquad-12.onnx
wget https://github.com/onnx/models/raw/main/text/machine_comprehension/bert-squad/model/bertsquad-12-int8.onnx
```

**quantize**

The quantization is a custom one. It only converts a *MatMul* into
a sequence *Transpose + DynamicQuantizeLinear + GemmFloat8*.

```bash
python3 -m onnx_extended quantize -i bertsquad-12.onnx -o bertsquad-12-fp8.onnx -v -v -k fp8 -q
```

**benchmark**

```bash
mkdir tmp

