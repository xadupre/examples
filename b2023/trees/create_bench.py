import os
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from skl2onnx import to_onnx
from onnx_extended.tools.run_onnx import save_for_benchmark_or_test
from onnx_extended.ext_test_case import get_parsed_args

# The dimension of the problem.

args = get_parsed_args(
    "create_bench",
    **dict(
        batch_size=(1000, "batch size"),
        n_features=(100, "number of features"),
        n_trees=(500, "number of trees"),
        max_depth=(10, "max detph"),
    ),
)

batch_size = args.batch_size
n_features = args.n_features
n_trees = args.n_trees
max_depth = args.max_depth


# Let's create model.
X, y = make_regression(
    batch_size + 2**max_depth * 2, n_features=n_features, n_targets=1
)
X, y = X.astype(np.float32), y.astype(np.float32)

print(
    f"train RandomForestRegressor n_trees={n_trees} "
    f"n_features={n_features} batch_size={batch_size} "
    f"max_depth={max_depth}"
)
model = RandomForestRegressor(n_trees, max_depth=max_depth, n_jobs=-1, verbose=1)
model.fit(X[:-batch_size], y[:-batch_size])

# target_opset is used to select opset an old version of onnxruntime can process.
print("conversion to onnx")
onx = to_onnx(model, X[:1], target_opset=17)

print(f"size: {len(onx.SerializeToString())}")

# Let's save the model and the inputs on disk.
folder = f"test_ort_version-F{n_features}-T{n_trees}-D{max_depth}-B{batch_size}"
if not os.path.exists(folder):
    os.mkdir(folder)

print("create the benchmark")
inputs = [X[:batch_size]]
save_for_benchmark_or_test(folder, "rf", onx, inputs)

print("end")
# Let's see what was saved.
for r, d, f in os.walk(folder):
    for name in f:
        full_name = os.path.join(r, name)
        print(f"{os.stat(full_name).st_size / 2 ** 10:1.1f} Kb: {full_name}")
