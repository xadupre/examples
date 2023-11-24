import os
from itertools import product
import numpy
from tqdm import tqdm
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from skl2onnx import to_onnx

nfs = [10]
n_trees = [100, 500]
max_depths = [5, 10, 15]

for nf, n_tree, max_depth in tqdm(product(nfs, n_trees, max_depths)):
    filename = f"rf_nf{nf}_T{n_tree}_d{max_depth}.onnx"
    if not os.path.exists(filename):
        print(f"creating {filename}")
        X, y = make_regression(max_depth * 1000, nf)
        X = X.astype(numpy.float32)
        rf = RandomForestRegressor(n_estimators=n_tree, max_depth=max_depth)
        print("  training...")
        rf.fit(X, y)
        print("  converting...")
        onx = to_onnx(rf, X[:1])
        raw = onx.SerializeToString()
        with open(filename, "wb") as f:
            f.write(raw)
        print("  done, size=", len(raw))
