import numpy as np
import nvitop
import torch.distributed as dist


def get_stats(local_rank, model):
    kinds = {}
    try:
        rk = dist.get_rank()
    except RuntimeError:
        rk = -1
    if model is not None:
        params = list(model.parameters())
        for p in params:
            key = str(p.dtype), str(p.device)
            if key not in kinds:
                kinds[key] = 0
            kinds[key] += np.prod(tuple(p.shape))
        for mod in model.modules():
            pref = mod.__class__.__name__
            params = list(mod.parameters())
            for p in params:
                size = np.prod(tuple(p.shape))
                if size == 0:
                    continue
                key = pref, str(p.dtype), str(p.device)
                if key not in kinds:
                    kinds[key] = 0
                kinds[key] += size

    kinds["get_rank"] = rk
    kinds["memory_cpu_percent"] = nvitop.host.memory_percent()
    cuda = nvitop.PhysicalDevice(local_rank)
    kinds["gpu_used_gb"] = cuda.memory_used() / 2**30
    kinds["local_rank"] = local_rank
    return kinds


print(get_stats(0, None))
