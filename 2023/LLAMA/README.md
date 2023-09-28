# Quantize LLAMA with FP8

## A small model: first layer

It creates a small models with the same inputs and an intermediate outputs.

```python
python select.py
```

## Quantization

The model quantizes most of the matrix multiplication as they operator
between an input and a fixed tensor of weights.

```bash
bash quantize.sh
```
