import os
import sys
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Build SO from ONNX model")

parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model")
parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
parser.add_argument(
    "--llvm_dir", type=str, required=True, help="Parent directory of LLVM binaries"
)

args = parser.parse_args()

filepath = os.path.abspath(__file__)
treebeard_repo_dir = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(filepath)))
)
# sys.path.append(os.path.join(treebeard_repo_dir, 'src', 'python'))

import treebeard

print("BEGIN")

# Setup treebeard options
BATCH_SIZE = 1024
TREE_TILE_SIZE = 8
compiler_options = treebeard.CompilerOptions(BATCH_SIZE, TREE_TILE_SIZE)

compiler_options.SetNumberOfCores(16)
compiler_options.SetMakeAllLeavesSameDepth(
    1
)  # make all leaves same depth. Enables unrolling tree walks of same depth.
compiler_options.SetReorderTreesByDepth(
    True
)  # reorder trees by depth. Enables grouping of trees by depth
compiler_options.SetPipelineWidth(
    8
)  # set pipeline width. Enables jamming of unrolled loops. Should be less than batch size.

print("CONTEXT")

onnx_model_path = os.path.abspath(args.onnx)
print(f"onnx_model_path={onnx_model_path!r}")
tbContext = treebeard.TreebeardContext(onnx_model_path, "", compiler_options)
tbContext.SetRepresentationType("sparse")
tbContext.SetInputFiletype("onnx_file")

print("DONE")

model_file_name = os.path.abspath(os.path.basename(onnx_model_path))
if not os.path.exists(model_file_name):
    raise FileNotFoundError(f"Unable to find {model_file_name!r}")
llvm_file_path = os.path.abspath(os.path.join(args.out_dir, model_file_name + ".ll"))

print("DUMP")
print(f"model_file_name={model_file_name!r}")
print(f"llvm_file_path={llvm_file_path!r}")

if tbContext.DumpLLVMIR(llvm_file_path) is False:
    raise RuntimeError("Failed to dump LLVM IR")


asm_file_path = os.path.join(args.out_dir, model_file_name + ".s")
so_file_path = os.path.join(args.out_dir, model_file_name + ".so")
print(f"asm_file_path={asm_file_path!r}")
print(f"so_file_path={so_file_path!r}")

# Run LLC
subprocess.run(
    [
        os.path.join(args.llvm_dir, "bin", "llc"),
        llvm_file_path,
        "-O3",
        "-march=x86-64",
        "-mcpu=native",
        "--relocation-model=pic",
        "-o",
        asm_file_path,
    ]
)

# Run CLANG
subprocess.run(
    [
        os.path.join(args.llvm_dir, "bin", "clang"),
        "-shared",
        asm_file_path,
        "-fopenmp=libomp",
        "-o",
        so_file_path,
    ]
)

# Delete ll, asm files
os.remove(llvm_file_path)
os.remove(asm_file_path)
