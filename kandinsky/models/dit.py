import torch
from torch import nn
# from flash_attn_interface import flash_attn_varlen_func
from flash_attn import flash_attn_varlen_func

from .nn import (
    TimeEmbeddings,
    TextEmbeddings,
    VisualEmbeddings,
    RoPE1D,
    RoPE3D,
    Modulation,
    MultiheadSelfAttention,
    MultiheadCrossAttention,
    FeedForward,
    OutLayer,
    apply_scale_shift_norm,
    apply_gate_sum,
)
from .utils import fractal_flatten, fractal_unflatten


class TransformerEncoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim):
        super().__init__()
        self.text_modulation = Modulation(time_dim, model_dim, 6)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim)
        self.out_layer_self = nn.Linear(model_dim, model_dim, bias=True)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    @torch.compiler.disable
    def scaled_dot_product_attention(
        self, query, key, value, cu_seqlens, cond_cu_seqlens, max_seqlen, cond_max_seqlen):
        out = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cond_cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=cond_max_seqlen
        )
        return out

    def forward(self, x, time_embed, rope, cu_seqlens, max_seqlen, time_embed_idx):
        def _pre_attention(time_embed, x, time_embed_idx):
            self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)

            shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
            out = apply_scale_shift_norm(self.self_attention_norm, x, scale, shift, time_embed_idx)

            query, key, value = self.self_attention(out, rope)
            return query, key, value, gate, ff_params

        query, key, value, gate, ff_params = _pre_attention(time_embed, x, time_embed_idx)

        out = self.scaled_dot_product_attention(
            query, key, value, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)

        def _post_attention(x, out, gate, time_embed_idx, ff_params):
            out = out.flatten(-2, -1)
            out = self.out_layer_self(out)

            x = apply_gate_sum(x, out, gate, time_embed_idx)

            shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
            out = apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift, time_embed_idx)
            out = self.feed_forward(out)
            x = apply_gate_sum(x, out, gate, time_embed_idx)
            return x

        x = _post_attention(x, out, gate, time_embed_idx, ff_params)
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim):
        super().__init__()
        self.visual_modulation = Modulation(time_dim, model_dim, 9)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim)
        self.out_layer_self = nn.Linear(model_dim, model_dim, bias=True)

        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = MultiheadCrossAttention(model_dim, head_dim)
        self.out_layer_cross = nn.Linear(model_dim, model_dim, bias=True)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    @torch.compiler.disable
    def scaled_dot_product_attention(
        self, query, key, value, cu_seqlens, cond_cu_seqlens, max_seqlen, cond_max_seqlen):
        out = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cond_cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=cond_max_seqlen
        )
        return out

    def forward(
        self, visual_embed, text_embed, time_embed, rope, visual_cu_seqlens, text_cu_seqlens, max_seqlen, 
        cond_max_seqlen, time_embed_idx, block_mask, torch_mask, sparse_params):
        def _pre_attention(time_embed, visual_embed, time_embed_idx):
            self_attn_params, cross_attn_params, ff_params = torch.chunk(
                self.visual_modulation(time_embed), 3, dim=-1)

            shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
            visual_out = apply_scale_shift_norm(
                self.self_attention_norm, visual_embed, scale, shift, time_embed_idx)

            query, key, value = self.self_attention(visual_out, rope)
            return query, key, value, gate, cross_attn_params, ff_params

        query, key, value, gate, cross_attn_params, ff_params = _pre_attention(
            time_embed, visual_embed, time_embed_idx)

        visual_out = self.scaled_dot_product_attention(
            query, key, value, visual_cu_seqlens, visual_cu_seqlens, max_seqlen, max_seqlen)

        def _post_self_attention(visual_out, visual_embed, gate, time_embed_idx, cross_attn_params):
            visual_out = visual_out.flatten(-2, -1)
            visual_out = self.out_layer_self(visual_out)
            visual_embed = apply_gate_sum(visual_embed, visual_out, gate, time_embed_idx)

            shift, scale, gate = torch.chunk(cross_attn_params, 3, dim=-1)
            visual_out = apply_scale_shift_norm(
                self.cross_attention_norm, visual_embed, scale, shift, time_embed_idx)
            query, key, value = self.cross_attention(visual_out, text_embed)
            return query, key, value, gate, visual_embed

        query, key, value, gate, visual_embed = _post_self_attention(
            visual_out, visual_embed, gate, time_embed_idx, cross_attn_params)

        visual_out = self.scaled_dot_product_attention(
            query, key, value, visual_cu_seqlens, text_cu_seqlens, max_seqlen, cond_max_seqlen)

        def _post_cross_attention(visual_embed, visual_out, gate, time_embed_idx, ff_params):
            visual_out = visual_out.flatten(-2, -1)
            visual_out = self.out_layer_cross(visual_out)

            visual_embed = apply_gate_sum(visual_embed, visual_out, gate, time_embed_idx)

            shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
            visual_out = apply_scale_shift_norm(
                self.feed_forward_norm, visual_embed, scale, shift, time_embed_idx)
            visual_out = self.feed_forward(visual_out)
            visual_embed = apply_gate_sum(visual_embed, visual_out, gate, time_embed_idx)
            return visual_embed

        visual_embed = _post_cross_attention(visual_embed, visual_out, gate, time_embed_idx, ff_params)
        return visual_embed


class DiffusionTransformer3D(nn.Module):
    def __init__(
        self,
        in_visual_dim=4,
        in_text_dim=3584,
        in_text_dim2=768,
        time_dim=512,
        out_visual_dim=4,
        patch_size=(1, 2, 2),
        model_dim=2048,
        ff_dim=5120,
        num_text_blocks=2,
        num_visual_blocks=32,
        axes_dims=(16, 24, 24),
        visual_cond=False,
    ):
        super().__init__()
        head_dim = sum(axes_dims)
        self.in_visual_dim = in_visual_dim
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.visual_cond = visual_cond

        visual_embed_dim = 2 * in_visual_dim + 1 if visual_cond else in_visual_dim
        self.time_embeddings = TimeEmbeddings(model_dim, time_dim)
        self.text_embeddings = TextEmbeddings(in_text_dim, model_dim)
        self.pooled_text_embeddings = TextEmbeddings(in_text_dim2, time_dim)
        self.visual_embeddings = VisualEmbeddings(
            visual_embed_dim, model_dim, patch_size
        )

        self.text_rope_embeddings = RoPE1D(head_dim)
        self.text_transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(model_dim, time_dim, ff_dim, head_dim)
                for _ in range(num_text_blocks)
            ]
        )

        self.visual_rope_embeddings = RoPE3D(axes_dims)
        self.visual_transformer_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim)
                for _ in range(num_visual_blocks)
            ]
        )

        self.out_layer = OutLayer(model_dim, time_dim, out_visual_dim, patch_size)

    @torch.compile
    def forward(
        self,
        x,
        text_embed,
        pooled_text_embed,
        time,
        visual_cu_seqlens,
        text_cu_seqlens,
        visual_rope_pos,
        text_rope_pos,
        scale_factor=(1.0, 1.0, 1.0),
        block_mask=None,
        torch_mask=None,
        sparse_params=None
    ):
        if block_mask is None and sparse_params is not None:
            block_mask = sparse_params["block_mask"]
        if torch_mask is None and sparse_params is not None:
            torch_mask = sparse_params["torch_mask"]

        def before_text_transformer_blocks(
            text_embed, time, pooled_text_embed, x, visual_cu_seqlens, text_rope_pos, text_cu_seqlens
        ):
            text_embed = self.text_embeddings(text_embed)
            time_embed, time_embed_idx = self.time_embeddings(time)
            time_embed = time_embed + self.pooled_text_embeddings(pooled_text_embed)
            visual_embed, visual_cu_seqlens = self.visual_embeddings(x, visual_cu_seqlens)

            text_rope = self.text_rope_embeddings(text_rope_pos)
            text_diff = torch.diff(text_cu_seqlens)
            text_max_seqlen = text_diff.max()
            text_time_embed_idx = time_embed_idx.repeat_interleave(text_diff, dim=0)
            return (text_embed, time_embed, text_rope, text_max_seqlen, text_time_embed_idx, 
            visual_embed, time_embed_idx, visual_cu_seqlens)

        (text_embed, time_embed, text_rope, text_max_seqlen, text_time_embed_idx, visual_embed,
        time_embed_idx, visual_cu_seqlens) = before_text_transformer_blocks(
            text_embed, time, pooled_text_embed, x, visual_cu_seqlens, text_rope_pos, text_cu_seqlens)

        for text_transformer_block in self.text_transformer_blocks:
            text_embed = text_transformer_block(
                text_embed, time_embed, text_rope, text_cu_seqlens, text_max_seqlen, text_time_embed_idx
            )

        def before_visual_transformer_blocks(
            visual_embed, visual_rope_pos, scale_factor, sparse_params, time_embed_idx, visual_cu_seqlens
        ):
            visual_shape = visual_embed.shape[:-1]
            visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor)
            to_fractal = sparse_params["to_fractal"] if sparse_params is not None else False
            visual_embed, visual_rope, visual_cu_seqlens = fractal_flatten(
                visual_embed, visual_rope, visual_cu_seqlens, visual_shape, block_mask=to_fractal)
            visual_diff = torch.diff(visual_cu_seqlens)
            visual_max_seqlen = visual_diff.max()
            visual_time_embed_idx = time_embed_idx.repeat_interleave(visual_diff, dim=0)
            return (visual_embed, visual_shape, to_fractal, visual_rope, 
            visual_max_seqlen, visual_time_embed_idx, visual_cu_seqlens)

        (visual_embed, visual_shape, to_fractal, visual_rope, visual_max_seqlen, visual_time_embed_idx, 
        visual_cu_seqlens) = before_visual_transformer_blocks(
            visual_embed, visual_rope_pos, scale_factor, sparse_params, time_embed_idx, visual_cu_seqlens)

        for visual_transformer_block in self.visual_transformer_blocks:
            visual_embed = visual_transformer_block(
                visual_embed, text_embed, time_embed, visual_rope, visual_cu_seqlens, text_cu_seqlens,
                visual_max_seqlen, text_max_seqlen, visual_time_embed_idx,
                block_mask, torch_mask, sparse_params
            )   

        def after_blocks(
            visual_embed, visual_cu_seqlens, visual_shape, to_fractal, text_embed, time_embed_idx, time_embed
        ):
            visual_embed, visual_cu_seqlens = fractal_unflatten(
                visual_embed, visual_cu_seqlens, visual_shape, block_mask=to_fractal)
            
            visual_diff = torch.diff(visual_cu_seqlens)
            visual_time_embed_idx = time_embed_idx.repeat_interleave(visual_diff, dim=0)
            x = self.out_layer(visual_embed, text_embed, time_embed, visual_cu_seqlens, visual_time_embed_idx)
            return x
        
        x = after_blocks(
            visual_embed, visual_cu_seqlens, visual_shape, to_fractal, text_embed, time_embed_idx, time_embed)
        return x


def get_dit(conf):
    dit = DiffusionTransformer3D(**conf)
    return dit
