# Export Phi model

Example [plot_exporter_recipes_phi35.py](plot_exporter_recipes_phi35.py)
exports model [Phi3.5 mini instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
into ONNX. `torch.export.export` fails on that model so some preprocessing were necessary
to patch ``torch`` and ``transformers`` to fix serialization issues and 
shape inferences issues introduced by DynamicCache.

The model without weights can be found in [results](results/).
The patches are implemented in [onnx_export_errors.py](onnx_export_errors.py)
and in the folder [patches](patches/).

The same code works for Phi-2 as well.
