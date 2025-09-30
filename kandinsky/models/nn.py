import math

import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention

from .utils import get_freqs, nablaT_v2

if torch.cuda.get_device_capability()[0] >= 9:        
    try:
        from flash_attn import flash_attn_func as FA
    except:
        FA = None

    try:
        from flash_attn_interface import flash_attn_func as FA
    except:
        FA = FA
else:
    try:
        from flash_attn import flash_attn_func as FA
    except:
        FA = None

@torch.compile()
@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_scale_shift_norm(norm, x, scale, shift):
    return (norm(x) * (scale + 1.0) + shift).to(torch.bfloat16)

@torch.compile()
@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_gate_sum(x, out, gate):
    return (x + gate * out).to(torch.bfloat16)

@torch.compile()
@torch.autocast(device_type="cuda", enabled=False)
def apply_rotary(x, rope):
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
    x_out = (rope * x_).sum(dim=-1)
    return x_out.reshape(*x.shape).to(torch.bfloat16)


class TimeEmbeddings(nn.Module):
    def __init__(self, model_dim, time_dim, max_period=10000.0):
        super().__init__()
        assert model_dim % 2 == 0
        self.model_dim = model_dim
        self.max_period = max_period
        self.register_buffer(
            "freqs", get_freqs(model_dim // 2, max_period), persistent=False
        )
        self.in_layer = nn.Linear(model_dim, time_dim, bias=True)
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, time_dim, bias=True)

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, time):
        args = torch.outer(time, self.freqs.to(device=time.device))
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        time_embed = self.out_layer(self.activation(self.in_layer(time_embed)))
        return time_embed


class TextEmbeddings(nn.Module):
    def __init__(self, text_dim, model_dim):
        super().__init__()
        self.in_layer = nn.Linear(text_dim, model_dim, bias=True)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, text_embed):
        text_embed = self.in_layer(text_embed)
        return self.norm(text_embed).type_as(text_embed)


class VisualEmbeddings(nn.Module):
    def __init__(self, visual_dim, model_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)

    def forward(self, x):
        duration, height, width, dim = x.shape
        x = (
            x.view(
                duration // self.patch_size[0],
                self.patch_size[0],
                height // self.patch_size[1],
                self.patch_size[1],
                width // self.patch_size[2],
                self.patch_size[2],
                dim,
            )
            .permute(0, 2, 4, 1, 3, 5, 6)
            .flatten(3, 6)
        )
        return self.in_layer(x)


class RoPE1D(nn.Module):
    def __init__(self, dim, max_pos=1024, max_period=10000.0):
        super().__init__()
        self.max_period = max_period
        self.dim = dim
        self.max_pos = max_pos
        freq = get_freqs(dim // 2, max_period)
        pos = torch.arange(max_pos, dtype=freq.dtype)
        self.register_buffer(f"args", torch.outer(pos, freq), persistent=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, pos):
        args = self.args[pos]
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class RoPE3D(nn.Module):
    def __init__(self, axes_dims, max_pos=(128, 128, 128), max_period=10000.0):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period

        for i, (axes_dim, ax_max_pos) in enumerate(zip(axes_dims, max_pos)):
            freq = get_freqs(axes_dim // 2, max_period)
            pos = torch.arange(ax_max_pos, dtype=freq.dtype)
            self.register_buffer(f"args_{i}", torch.outer(pos, freq), persistent=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, shape, pos, scale_factor=(1.0, 1.0, 1.0)):
        duration, height, width = shape
        args_t = self.args_0[pos[0]] / scale_factor[0]
        args_h = self.args_1[pos[1]] / scale_factor[1]
        args_w = self.args_2[pos[2]] / scale_factor[2]

        args = torch.cat(
            [
                args_t.view(duration, 1, 1, -1).repeat(1, height, width, 1),
                args_h.view(1, height, 1, -1).repeat(duration, 1, width, 1),
                args_w.view(1, 1, width, -1).repeat(duration, height, 1, 1),
            ],
            dim=-1,
        )
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class Modulation(nn.Module):
    def __init__(self, time_dim, model_dim, num_params):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()

    @torch.compile()
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x):
        return self.out_layer(self.activation(x))

class MultiheadSelfAttentionEnc(nn.Module):
    def __init__(self, num_channels, head_dim):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)

    @torch.compile()
    def get_qkv(self, x):
        query = self.to_query(x)
        key = self.to_key(x)
        value = self.to_value(x)

        shape = query.shape[:-1]
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*shape, self.num_heads, -1)
        value = value.reshape(*shape, self.num_heads, -1)

        return query, key, value

    @torch.compile()
    def norm_qk(self, q, k):
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    @torch.compile()
    def scaled_dot_product_attention(self, query, key, value):
        out = FA(q=query.unsqueeze(0), k=key.unsqueeze(0), v=value.unsqueeze(0))[0].flatten(-2, -1)
        return out

    @torch.compile()
    def out_l(self, x):
        return self.out_layer(x)

    def forward(self, x, rope):
        query, key, value = self.get_qkv(x)
        query, key = self.norm_qk(query, key)
        query = apply_rotary(query, rope).type_as(query)
        key = apply_rotary(key, rope).type_as(key)

        out = self.scaled_dot_product_attention(query, key, value)

        out = self.out_l(out)
        return out

class MultiheadSelfAttentionDec(nn.Module):
    def __init__(self, num_channels, head_dim):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)

    @torch.compile()
    def get_qkv(self, x):
        query = self.to_query(x)
        key = self.to_key(x)
        value = self.to_value(x)

        shape = query.shape[:-1]
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*shape, self.num_heads, -1)
        value = value.reshape(*shape, self.num_heads, -1)

        return query, key, value

    @torch.compile()
    def norm_qk(self, q, k):
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    @torch.compile()
    def attention(self, query, key, value):
        out = FA(q=query.unsqueeze(0), k=key.unsqueeze(0), v=value.unsqueeze(0))[0].flatten(-2, -1)
        return out

    @torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
    def nabla(self, query, key, value, sparse_params=None):
        query = query.unsqueeze(0).transpose(1, 2).contiguous()
        key = key.unsqueeze(0).transpose(1, 2).contiguous()
        value = value.unsqueeze(0).transpose(1, 2).contiguous()
        block_mask = nablaT_v2(
            query,
            key,
            sparse_params["sta_mask"],
            thr=sparse_params["P"],
        )
        out = (
            flex_attention(
                query,
                key,
                value,
                block_mask=block_mask
            )
            .transpose(1, 2)
            .squeeze(0)
            .contiguous()
        )
        out = out.flatten(-2, -1)
        return out

    @torch.compile()
    def out_l(self, x):
        return self.out_layer(x)

    def forward(self, x, rope, sparse_params=None):
        query, key, value = self.get_qkv(x)
        query, key = self.norm_qk(query, key)
        query = apply_rotary(query, rope).type_as(query)
        key = apply_rotary(key, rope).type_as(key)

        if sparse_params is not None:
            out = self.nabla(query, key, value, sparse_params=sparse_params)
        else:
            out = self.attention(query, key, value)

        out = self.out_l(out)
        return out


class MultiheadCrossAttention(nn.Module):
    def __init__(self, num_channels, head_dim):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)

    @torch.compile()
    def get_qkv(self, x, cond):
        query = self.to_query(x)
        key = self.to_key(cond)
        value = self.to_value(cond)

        shape, cond_shape = query.shape[:-1], key.shape[:-1]
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*cond_shape, self.num_heads, -1)
        value = value.reshape(*cond_shape, self.num_heads, -1)

        return query, key, value

    @torch.compile()
    def norm_qk(self, q, k):
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    @torch.compile()
    def attention(self, query, key, value):
        out = FA(q=query.unsqueeze(0), k=key.unsqueeze(0), v=value.unsqueeze(0))[0].flatten(-2, -1)
        return out

    @torch.compile()
    def out_l(self, x):
        return self.out_layer(x)

    def forward(self, x, cond):
        query, key, value = self.get_qkv(x, cond)
        query, key = self.norm_qk(query, key)

        out = self.attention(query, key, value)
        out = self.out_l(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=False)

    @torch.compile()
    def forward(self, x):
        return self.out_layer(self.activation(self.in_layer(x)))


class OutLayer(nn.Module):
    def __init__(self, model_dim, time_dim, visual_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(
            model_dim, math.prod(patch_size) * visual_dim, bias=True
        )

    def forward(self, visual_embed, text_embed, time_embed):
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        visual_embed = apply_scale_shift_norm(
            self.norm,
            visual_embed,
            scale[:, None, None],
            shift[:, None, None],
        ).type_as(visual_embed)
        x = self.out_layer(visual_embed)

        duration, height, width, _ = x.shape
        x = (
            x.view(
                duration,
                height,
                width,
                -1,
                self.patch_size[0],
                self.patch_size[1],
                self.patch_size[2],
            )
            .permute(0, 4, 1, 5, 2, 6, 3)
            .flatten(0, 1)
            .flatten(1, 2)
            .flatten(2, 3)
        )
        return x
