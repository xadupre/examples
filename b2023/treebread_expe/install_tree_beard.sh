if [[ ! -e "tbuild" ]]; then
    mkdir tbuild
fi
pushd tbuild
echo "--------------------------------------------"
python3 --version
cmake --version
echo "--------------------------------------------"
# apt-get clang clang-tools libomp-dev build-essential
# apt-get install lld ninja-build


if [[ ! -e "ninja-linux.zip" ]]; then
    wget https://github.com/ninja-build/ninja/releases/download/v1.11.1/ninja-linux.zip
    unzip ninja-linux.zip -d .
fi

export OLD_PATH=$PATH
MY_VARIABLE=$(pwd -P)
export PATH=$PATH:$MY_VARIABLE

echo "CURRENT DIRECTORY: $(pwd -P)"


if [[ ! -e "llvm-project/build/bin" ]]; then
    echo "--------------------------------------------"
    # https://apt.llvm.org/
    bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
    echo "--------------------------------------------"
    git clone https://github.com/llvm/llvm-project.git
    pushd llvm-project
    if [[ ! -e "build" ]]; then
        mkdir build
    fi
    git checkout release/16.x
    pushd build
    cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS="llvm;clang;lld;mlir;openmp" \
        -DLLVM_BUILD_EXAMPLES=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DLLVM_ENABLE_LLD=ON
    cmake .
    cmake --build . --target check-mlir
    popd
    popd
    echo "--------------------------------------------"
fi

echo "CURRENT DIRECTORY: $(pwd -P)"

export PROTOBUF_BUILD=$MY_VARIABLE/onnx/.setuptools-cmake-build/_deps/protobuf-build
export PROTOBUF_SOURCE=$MY_VARIABLE/onnx/.setuptools-cmake-build/_deps/protobuf-src
export ABSL_SOURCE=$MY_VARIABLE/onnx/.setuptools-cmake-build/_deps/abseil-src
export ABSL_LIB=$MY_VARIABLE/onnx/.setuptools-cmake-build/_deps/protobuf-build/third_party/abseil-cpp
if [[ ! -e $PROTOBUF_BUILD ]]; then
    git clone https://github.com/onnx/onnx.git
    pushd onnx
    python3 -m pip install -e . -v
    popd
fi

echo "CURRENT DIRECTORY: $(pwd -P)"


if [[ ! -e "treebeard/build/bin/treebeard" ]]; then
    export PATH=$MY_VARIABLE/onnx/.setuptools-cmake-build/_deps/protobuf-build:$PATH
    echo "--------------------------------------------"
    echo "protoc --version=$(protoc --version)"
    echo "--------------------------------------------"
    echo "Cloning Treebeard git repo ..."
    #pushd llvm-project/mlir/examples/
    # fork from https://github.com/asprasad/treebeard
    git clone https://github.com/xadupre/treebeard.git
    pushd treebeard
    git checkout build || exit 1
    popd
    if [[ ! -e "treebeard/build" ]]; then
        mkdir treebeard/build
    fi

    # echo "---------------------------------------"
    # cp $MY_VARIABLE/onnx/onnx/onnx.proto $MY_VARIABLE/treebeard/src/json || exit 1
    # protoc --proto_path=$MY_VARIABLE/treebeard/src/json/ --cpp_out=$MY_VARIABLE/treebeard/src/json/ onnx.proto || exit 1


    pushd treebeard/build

    export CMAKE_MODULE_PATH=$CMAKE_MODULE_PATH:$PROTOBUF_PATH

    # issue here: treebeard/src/json/onnx.pb.h:10:10: fatal error: 'google/protobuf/port_def.inc' file not found

    cmake -G Ninja .. \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        -DMLIR_DIR=$MY_VARIABLE/llvm-project/build/lib/cmake/mlir \
        -DLLVM_BUILD_DIRECTORY=$MY_VARIABLE/llvm-project/build/ \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_CXX_FLAGS="-std=c++17" \
        -DLLVM_ENABLE_LLD=ON

        #-DABSL_INCLUDE_DIR=$ABSL_SOURCE \
        #-DABSL_LIBRARIES_DIR=$ABSL_LIB \
        #-DProtobuf_INCLUDE_DIR=$PROTOBUF_SOURCE/src \
        #-DProtobuf_INCLUDE_DIRS=$PROTOBUF_SOURCE/src \
        #-DProtobuf_LIBRARIES=$PROTOBUF_BUILD/libprotobuf.a \

    cmake --build .
    echo "--------------------------------------------"
    popd
    #popd
fi

popd

echo "copy ./tbuild/treebeard/build/lib/*.so to ./tbuild/treebeard/src/python"
export PATH=$OLD_PATH
cp ./tbuild/treebeard/build/lib/*.so ./tbuild/treebeard/src/python
ls -l ./tbuild/treebeard/src/python

