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

Function ``bypass_export_some_errors`` makes some changes to avoid some of the error coming from torch.export.export.
After it is done, it reverts every change back to what it was.
See [onnx_export_errors.py](onnx_export_errors.py) to see what it does.

```python
    with bypass_export_some_errors(
        patch_transformers=True, replace_dynamic_cache=True, verbose=1
    ) as modificator:
        inputs = modificator(inputs)
        ep = torch.onnx.export(
            model, (), kwargs=inputs, dynamic_shapes=dynamic_shapes, dynamo=True
        )
        ep.optimize()
        print("save the model")
        ep.save(name)
```

