python3 perf.py --scenario=torch
python3 perf.py --scenario=torch-ort
python3 perf.py --scenario=ds0
python3 perf.py --scenario=ds0-ort
python3 perf.py --scenario=ds0-f16
python3 perf.py --scenario=ds0-f16-ort
python3 perf.py --scenario=ds1
python3 perf.py --scenario=ds1-f16
python3 perf.py --scenario=ds1-f16-offload
python3 perf.py --scenario=ds1-f16-ort
python3 perf.py --scenario=ds1-f16-offload-ort
python3 perf.py --scenario=ds2
python3 perf.py --scenario=ds2-f16
python3 perf.py --scenario=ds2-f16-offload
python3 perf.py --scenario=ds2-f16-offload-ort
python3 perf.py --scenario=ds3
python3 perf.py --scenario=ds3-f16
python3 perf.py --scenario=ds3-f16-offload
python3 perf.py --scenario=ds2-f16-ort
python3 perf.py --scenario=ds3-f16-ort
python3 perf.py --scenario=ds3-f16-offload-ort

python3 perf.py --scenario=ds1-ort
python3 perf.py --scenario=ds2-ort

# python3 perf.py --scenario=ds0-offload
# python3 perf.py --scenario=ds3-ort
python3 perf.py --scenario=ds1-offload
python3 perf.py --scenario=ds2-offload
python3 perf.py --scenario=ds1-offload-ort
python3 perf.py --scenario=ds2-offload-ort
python3 perf.py --scenario=ds3-offload-ort
python3 perf.py --scenario=ds3-offload


deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds0.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds1.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds2.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds3.json

deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds0-f16.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds1-f16.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds2-f16.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds3-f16.json

deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds0-offload.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds1-offload.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds2-offload.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds3-offload.json

deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds0-f16-offload.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds1-f16-offload.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds2-f16-offload.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds3-f16-offload.json

deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds0-ort.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds1-ort.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds2-ort.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds3-ort.json

deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds0-f16-ort.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds1-f16-ort.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds2-f16-ort.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds3-f16-ort.json

deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds0-offload-ort.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds1-offload-ort.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds2-offload-ort.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds3-offload-ort.json

deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds0-f16-offload-ort.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds1-f16-offload-ort.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds2-f16-offload-ort.json
deepspeed --num_gpu=4  distri.py --deepspeed --deepspeed_config config/last-config-ds3-f16-offload-ort.json

