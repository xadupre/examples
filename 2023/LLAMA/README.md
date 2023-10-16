# Quantize LLAMA with FP8

## A small model: first layer

It creates a small models with the same inputs and an intermediate outputs.

```python
python select.py
```

## Quantization with functions (dynamic)

The model quantizes most of the matrix multiplications as they are an operator
between an input and a fixed tensor of weights.

```bash
bash quantize_with_functions.sh
```

## Quantization with QDQ


```bash
bash ...
```
