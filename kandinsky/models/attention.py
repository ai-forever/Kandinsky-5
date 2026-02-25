import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

try:
    from flash_attn import flash_attn_func as flash_attention_2
    print("FlashAttention 2 is found")
except ImportError:
    flash_attention_2 = None

try:
    from flash_attn_interface import flash_attn_func as flash_attention_3
    print("FlashAttention 3 is found")
except ImportError:
    flash_attention_3 = None

try:
    import sageattention
    print(f"Sage Attention is found")
except ImportError:
    sageattention = None

@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def sdpa(q, k, v, attn_mask=None):
    query = q.transpose(1, 2).contiguous()
    key = k.transpose(1, 2).contiguous()
    value = v.transpose(1, 2).contiguous()
    out = (
        F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask
        )
        .transpose(1, 2)
        .contiguous()
    )
    return out

@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def sage_attn(q, k, v):
    out = (
        sageattention.sageattn(
            q, k, v,
            tensor_layout="NHD",
            is_causal=False
        )
    )
    return out

class SelfAttentionEngine():
    def __init__(self, engine="auto"):
        assert engine in ["auto", "flash_attention_2", "flash_attention_3", "sage", "sdpa"]
        self.attention_fn = None

        if engine == "flash_attention_2":
            if flash_attention_2 is None:
                raise RuntimeError("flash_attention_2 engine selected, but it can't be imported.")
            self.attention_fn = flash_attention_2

        if engine == "flash_attention_3":
            if flash_attention_3 is None:
                raise RuntimeError("flash_attention_3 engine selected, but it can't be imported.")
            self.attention_fn = flash_attention_3

        if engine == "sage":
            if sageattention is None:
                raise RuntimeError("sage engine selected, but it can't be imported.")
            self.attention_fn = sage_attn

        if engine == "sdpa":
            self.attention_fn = sdpa
        
        if engine == "auto":
            self.attention_fn = sdpa
            if not sageattention is None:
                self.attention_fn = sage_attn
            if not flash_attention_2 is None:
                self.attention_fn = flash_attention_2
            if not flash_attention_3 is None:
                self.attention_fn = flash_attention_3
    
    def get_attention(self):
        return self.attention_fn

