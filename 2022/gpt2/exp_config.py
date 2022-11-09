def get_deepspeed_config(WORLD_SIZE, scenario_full, train_batch_size=None):
    # batch size
    if train_batch_size is None:
        train_batch_size = 24
        gradient_acc_step = 4
    else:
        gradient_acc_step = max(1, train_batch_size // 8)
    micro_batch_per_gpu = train_batch_size // (gradient_acc_step * WORLD_SIZE)
    if micro_batch_per_gpu * gradient_acc_step * WORLD_SIZE != train_batch_size:
        micro_batch_per_gpu += 1
        train_batch_size = micro_batch_per_gpu * gradient_acc_step * WORLD_SIZE

    spl = scenario_full.split("-")
    scenario = spl[0]
    offload = "offload" in spl

    if scenario == "ds0":
        ds_config = {
            "zero_optimization": {
                "allgather_bucket_size": 5e8,
                "contiguous_gradients": True,
                "stage": 0,
                "overlap_comm": True,
            },
        }
    elif scenario == "ds1":
        ds_config = {
            "zero_optimization": {
                "allgather_bucket_size": 2e8,
                "allgather_partitions": True,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_bucket_size": 2e8,
                "reduce_scatter": True,
                "stage": 1,
            },
            "zero_allow_untested_optimizer": True,
        }
    elif scenario == "ds2":
        ds_config = {
            "zero_optimization": {
                "allgather_bucket_size": 2e8,
                "allgather_partitions": True,
                "contiguous_gradients": True,
                "offload_optimizer": {"device": "cpu"},
                "reduce_bucket_size": 2e8,
                "reduce_scatter": True,
                "overlap_comm": True,
                "stage": 2,
            },
            "zero_allow_untested_optimizer": True,
        }
    elif scenario == "ds3":
        ds_config = {
            "zero_optimization": {
                "allgather_bucket_size": 2e8,
                "allgather_partitions": True,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_bucket_size": 2e8,
                "reduce_scatter": True,
                "stage": 3,
            },
            "zero_allow_untested_optimizer": True,
        }
    else:
        raise ValueError(f"Unknwon scenario {scenario!r}.")

    if offload:
        ds_config["zero_optimization"].update(
            {
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "offload_param": {"device": "cpu", "pin_memory": True},
            }
        )

    optimizer_config = {
        "type": "AdamW",
        "params": {
            "betas": [0.8, 0.999],
            "eps": 1e-8,
            "lr": 0.0001,
            "weight_decay": 3e-7,
        },
    }

    scheduler_config = {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
        },
    }

    ds_config.update(
        {
            # "gradient_clipping": 1.0,
            "micro_batch_per_gpu": micro_batch_per_gpu,
            "optimizer": optimizer_config,
            # "scheduler": scheduler_config,
            "train_batch_size": train_batch_size,
            "wall_clock_breakdown": False,
        },
    )
    if "f16" in spl:
        ds_config.update(
            {
                "bf16": {"enabled": False},
                "fp16": {
                    "enabled": True,
                    "loss_scale": 0,
                    "min_loss_scale": 0,
                },
            }
        )
    elif "b16" in spl:
        ds_config.update(
            {
                "bf16": {"enabled": True},
                "fp16": {"enabled": False},
            }
        )
    else:
        ds_config.update(
            {
                "bf16": {"enabled": False},
                "fp16": {"enabled": False},
            }
        )

    return ds_config
