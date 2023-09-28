# quantize the model into fp8
python -m onnx_extended quantize -i models/llama_16_block_list_1.onnx \
                                 -o models/llama_16_block_list_1.fp8.onnx \
                                 -k fp8 -v \
                                 -x /transformer/block_list.0/attention/MatMul,/transformer/block_list.0/attention/MatMul_1

python -m onnx_extended quantize -i models/llama_16_block_list_1.onnx \
                                 -o models/llama_16_block_list_1.fp8.opt.onnx \
                                 -k fp8 -v \
                                 -p optimize \
                                 -x /transformer/block_list.0/attention/MatMul,/transformer/block_list.0/attention/MatMul_1
