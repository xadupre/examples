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

**download the model**

From page [bert-squad](https://github.com/onnx/models/tree/main/text/machine_comprehension/bert-squad)

```bash
wget https://github.com/onnx/models/raw/main/text/machine_comprehension/bert-squad/model/bertsquad-12.onnx
wget https://github.com/onnx/models/raw/main/text/machine_comprehension/bert-squad/model/bertsquad-12-int8.onnx
```

**download the data**

See [google-research/bert](https://github.com/google-research/bert/)

```bash
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-128_A-2.zip
unzip uncased_L-2_H-128_A-2.zip -d data
```

**quantize**

The quantization is a custom one. It only converts a *MatMul* into
a sequence *Transpose + DynamicQuantizeLinear + GemmFloat8*.

```bash
python3 -m onnx_extended quantize -i bertsquad-12.onnx -o bertsquad-12-fp8.onnx -v -v -k fp8 -q
```

**preparation**

```bash
mkdir tmp
```

**benchmark**

```bash
python run_onnx_squad.py \
    --model ./bertsquad-12.onnx \
    --vocab_file ./data/vocab.txt \
    --predict_file ./tmp/dev-v1.1.json \
    --bert_config_file ./data/bert_config.json \
    --output ./tmp/
```
