#!/usr/bin/env python
# coding: utf-8

# # Getting Started
#
# ## Overview
#
# Transformer Engine (TE) is a library for accelerating Transformer models on NVIDIA GPUs, providing better performance with lower memory utilization in both training and inference. It provides support for 8-bit floating point (FP8) precision on Hopper GPUs, implements a collection of highly optimized building blocks for popular Transformer architectures, and exposes an automatic-mixed-precision-like API that can be used seamlessy with your PyTorch code. It also includes a framework-agnostic C++ API that can be integrated with other deep learning libraries to enable FP8 support for Transformers.
#
# ## Let's build a Transformer layer!
#
# <div class="alert alert-info">
#
# <b>Summary</b>
#
# We build a basic Transformer layer using regular PyTorch modules. This will be our baseline for later comparisons with Transformer Engine.
#
# </div>
#
# Let's start with creating a GPT encoder layer using plain PyTorch. Figure 1 shows the overall structure.
#
# <figure align="center">
# <img src="transformer_layer.png" width="20%">
# <figcaption> Figure 1: Structure of a GPT encoder layer.</figcaption>
# </figure>
#
# We construct the components as follows:
#
# - `LayerNorm`: `torch.nn.LayerNorm`
# - `QKV Projection`: `torch.nn.Linear` (conceptually three `Linear` layers for Q, K, and V separately, but we fuse into a single `Linear` layer that is three times larger)
# - `DotProductAttention`: `DotProductAttention` from [quickstart_utils.py](quickstart_utils.py)
# - `Projection`: `torch.nn.Linear`
# - `Dropout`: `torch.nn.Dropout`
# - `MLP`: `BasicMLP` from [quickstart_utils.py](quickstart_utils.py)
#
# Over the course of this tutorial we will use a few modules and helper functions defined in [quickstart_utils.py](quickstart_utils.py). Putting it all together:

# In[1]:


import torch
import quickstart_utils as utils


class BasicTransformerLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        layernorm_eps: int = 1e-5,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = hidden_size // num_attention_heads
        self.ln1 = torch.nn.LayerNorm(hidden_size, eps=layernorm_eps)
        self.qkv_projection = torch.nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.attention = utils.DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=self.kv_channels,
            attention_dropout=attention_dropout,
        )
        self.projection = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = torch.nn.Dropout(hidden_dropout)
        self.ln2 = torch.nn.LayerNorm(hidden_size, eps=layernorm_eps)
        self.mlp = utils.BasicMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
        )

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        res = x
        x = self.ln1(x)

        # Fused QKV projection
        qkv = self.qkv_projection(x)
        qkv = qkv.view(
            qkv.size(0), qkv.size(1), self.num_attention_heads, 3 * self.kv_channels
        )
        q, k, v = torch.split(qkv, qkv.size(3) // 3, dim=3)

        x = self.attention(q, k, v, attention_mask)
        x = self.projection(x)
        x = self.dropout(x)
        x = res + x
        res = x
        x = self.ln2(x)
        x = self.mlp(x)

        return x + res


# That's it! We now have a simple Transformer layer. We can test it:

# In[2]:


# Layer configuration
hidden_size = 128
sequence_length = 64
batch_size = 4
ffn_hidden_size = 512
num_attention_heads = 32
dtype = torch.float16

# Synthetic data
x = torch.rand(sequence_length, batch_size, hidden_size).cuda().to(dtype=dtype)
dy = torch.rand(sequence_length, batch_size, hidden_size).cuda().to(dtype=dtype)


# In[3]:


basic_transformer = BasicTransformerLayer(
    hidden_size,
    ffn_hidden_size,
    num_attention_heads,
)
basic_transformer.to(dtype=dtype).cuda()


# In[4]:


torch.manual_seed(1234)
y = basic_transformer(x, attention_mask=None)


# In[5]:


fprof = lambda: utils.speedometer_simple(
    basic_transformer,
    x,
    dy,
    forward_kwargs={"attention_mask": None},
)
fprof()


# In[6]:


from torch.onnx import export

export(
    basic_transformer,
    x,
    "basic.onnx",
    verbose=False,
    input_names=["X"],
    output_names=["Y"],
)


# In[7]:


from pyquickhelper.pycode.profiling import profile
from pyquickhelper.pycode.profiling import profile2graph

stat, text = profile(fprof)

gr = profile2graph(stat)
print(gr[0].to_text(fct_width=80))


# ## Meet Transformer Engine
#
# <div class="alert alert-info">
#
# <b>Summary</b>
#
# We modify the example Transformer layer to include the simplest TE modules: `Linear` and `LayerNorm`.
#
# </div>
#
# Now that we have a basic Transformer layer, let's use Transformer Engine to speed up the training.

# In[8]:


import transformer_engine.pytorch as te


# TE provides a set of PyTorch modules that can be used to build Transformer layers. The simplest of the provided modules are the `Linear` and `LayerNorm` layers, which we can use instead of `torch.nn.Linear` and `torch.nn.LayerNorm`. Let's modify `BasicTransformerLayer`:

# In[14]:


class BasicTEMLP(torch.nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.linear1 = te.Linear(hidden_size, ffn_hidden_size, bias=True)
        self.linear2 = te.Linear(ffn_hidden_size, hidden_size, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        x = self.linear2(x)
        return x


class BasicTETransformerLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        layernorm_eps: int = 1e-5,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = hidden_size // num_attention_heads
        self.ln1 = te.LayerNorm(hidden_size, eps=layernorm_eps)
        self.qkv_projection = te.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.attention = utils.DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=self.kv_channels,
            attention_dropout=attention_dropout,
        )
        self.projection = te.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = torch.nn.Dropout(hidden_dropout)
        self.ln2 = te.LayerNorm(hidden_size, eps=layernorm_eps)
        self.mlp = BasicTEMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        res = x
        x = self.ln1(x)

        # Fused QKV projection
        qkv = self.qkv_projection(x)
        qkv = qkv.view(
            qkv.size(0), qkv.size(1), self.num_attention_heads, 3 * self.kv_channels
        )
        q, k, v = torch.split(qkv, qkv.size(3) // 3, dim=3)

        x = self.attention(q, k, v, attention_mask)
        x = self.projection(x)
        x = self.dropout(x)
        x = res + x
        res = x
        x = self.ln2(x)
        x = self.mlp(x)

        return x + res


# In[15]:


basic_te_transformer = BasicTETransformerLayer(
    hidden_size,
    ffn_hidden_size,
    num_attention_heads,
)
basic_te_transformer.to(dtype=dtype).cuda()
utils.share_parameters_with_basic_te_model(basic_te_transformer, basic_transformer)


# In[16]:


torch.manual_seed(1234)
y = basic_te_transformer(x, attention_mask=None)


# In[19]:


fprof_te = lambda: utils.speedometer_simple(
    basic_te_transformer,
    x,
    dy,
    forward_kwargs={"attention_mask": None},
)
fprof_te()


# In[20]:


stat, text = profile(fprof_te)

gr = profile2graph(stat)
print(gr[0].to_text(fct_width=80))


# In[13]:


from torch.onnx import export

export(
    basic_te_transformer,
    x,
    "te.onnx",
    verbose=True,
    input_names=["X"],
    output_names=["Y"],
)


# ## Fused TE Modules
#
# <div class="alert alert-info">
#
# <b>Summary</b>
#
# We optimize the example Transformer layer with TE modules for fused operations.
#
# </div>
#
# The `Linear` layer is enough to build any Transformer model and it enables usage of Transformer Engine even for very custom Transformers. However, having more knowledge about the model allows for additional optimizations like kernel fusion, increasing the achievable speedup.
#
# Transformer Engine therefore provides coarser modules that span multiple layers:
#
# * `LayerNormLinear`
# * `LayerNormMLP`
# * `TransformerLayer`
#
# Building a third iteration of our Transformer layer with `LayerNormLinear` and `LayerNormMLP`:

# In[ ]:


class FusedTETransformerLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        layernorm_eps: int = 1e-5,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = hidden_size // num_attention_heads
        self.ln_qkv = te.LayerNormLinear(
            hidden_size, 3 * hidden_size, eps=layernorm_eps, bias=True
        )
        self.attention = utils.DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=self.kv_channels,
            attention_dropout=attention_dropout,
        )
        self.projection = te.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = torch.nn.Dropout(hidden_dropout)
        self.ln_mlp = te.LayerNormMLP(
            hidden_size, ffn_hidden_size, eps=layernorm_eps, bias=True
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        res = x
        qkv = self.ln_qkv(x)

        # Split qkv into query, key and value
        qkv = qkv.view(
            qkv.size(0), qkv.size(1), self.num_attention_heads, 3 * self.kv_channels
        )
        q, k, v = torch.split(qkv, qkv.size(3) // 3, dim=3)

        x = self.attention(q, k, v, attention_mask)
        x = self.projection(x)
        x = self.dropout(x)
        x = res + x
        res = x
        x = self.ln_mlp(x)

        return x + res


# In[ ]:


fused_te_transformer = FusedTETransformerLayer(
    hidden_size, ffn_hidden_size, num_attention_heads
)
fused_te_transformer.to(dtype=dtype).cuda()
utils.share_parameters_with_fused_te_model(fused_te_transformer, basic_transformer)


# In[ ]:


torch.manual_seed(1234)
y = fused_te_transformer(x, attention_mask=None)


# In[ ]:


utils.speedometer(
    fused_te_transformer,
    x,
    dy,
    forward_kwargs={"attention_mask": None},
)


# Finally, the `TransformerLayer` module is convenient for creating standard Transformer architectures and it provides the highest degree of performance optimization:

# In[ ]:


te_transformer = te.TransformerLayer(hidden_size, ffn_hidden_size, num_attention_heads)
te_transformer.to(dtype=dtype).cuda()
utils.share_parameters_with_transformerlayer_te_model(te_transformer, basic_transformer)


# In[ ]:


torch.manual_seed(1234)
y = te_transformer(x, attention_mask=None)


# In[ ]:


utils.speedometer(
    te_transformer,
    x,
    dy,
    forward_kwargs={"attention_mask": None},
)


# ## Enabling FP8
#
# <div class="alert alert-info">
#
# <b>Summary</b>
#
# We configure a TE module to perform compute in FP8.
#
# </div>
#
# Enabling FP8 support is very simple in Transformer Engine. We just need to wrap the modules within an [fp8_autocast](../api/pytorch.rst#transformer_engine.pytorch.fp8_autocast) context manager. Note that fp8_autocast should only be used to wrap the forward pass and must exit before starting a backward pass. See the [FP8 tutorial](fp8_primer.ipynb) for a detailed explanation of FP8 recipes and the supported options.

# In[ ]:


from transformer_engine.common.recipe import Format, DelayedScaling

te_transformer = te.TransformerLayer(hidden_size, ffn_hidden_size, num_attention_heads)
te_transformer.to(dtype=dtype).cuda()
utils.share_parameters_with_transformerlayer_te_model(te_transformer, basic_transformer)

fp8_format = Format.HYBRID
fp8_recipe = DelayedScaling(
    fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max"
)
torch.manual_seed(1234)
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    y = te_transformer(x, attention_mask=None)


# In[ ]:


utils.speedometer(
    te_transformer,
    x,
    dy,
    forward_kwargs={"attention_mask": None},
    fp8_autocast_kwargs={"enabled": True, "fp8_recipe": fp8_recipe},
)
