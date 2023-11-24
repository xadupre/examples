
export PYTHONPATH=$PYTHONPATH:./tbuild/treebeard/src/python
# export PATH=./tbuild/treebeard/build/lib:$PATH
# cp ./tbuild/treebeard/build/lib/*.so ./tbuild/treebeard/src/python
# ls ./tbuild/treebeard/src/python
export PATH=$PATH:/home/xadupre/example/treebeard_expe/tbuild/llvm-project/build/bin/
python3 build_so_from_onnx.py --onnx rf_nf10_T100_d5.onnx --out_dir tree_so --llvm_dir /home/xadupre/example/treebeard_expe/tbuild/llvm-project/build/bin/