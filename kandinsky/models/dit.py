import torch
from torch import nn

from .nn import (
    TimeEmbeddings, TextEmbeddings, VisualEmbeddings, 
    RoPE1D, RoPE3D, Modulation,
    MultiheadSelfAttention, MultiheadCrossAttention, 
    FeedForward, OutLayer, 
    apply_scale_shift_norm, apply_gate_sum
)
from .utils import fractal_flatten, fractal_unflatten

class TransformerEncoderBlock(nn.Module):

    def __init__(self, model_dim, time_dim, ff_dim, head_dim):
        super().__init__()
        self.text_modulation = Modulation(time_dim, model_dim, 6)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(self, x, time_embed, rope, cu_seqlens, time_embed_idx):
        self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)        
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = apply_scale_shift_norm(
            self.self_attention_norm, x, scale, shift, time_embed_idx
            ).type_as(x)
        out = self.self_attention(out, rope, cu_seqlens)
        x = apply_gate_sum(x, out, gate, time_embed_idx).type_as(x)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = apply_scale_shift_norm(
            self.feed_forward_norm, x, scale, shift, time_embed_idx
            ).type_as(x)
        out = self.feed_forward(out)
        x = apply_gate_sum(x, out, gate, time_embed_idx).type_as(x)     
        return x


class TransformerDecoderBlock(nn.Module):

    def __init__(self, model_dim, time_dim, ff_dim, head_dim):
        super().__init__()
        self.visual_modulation = Modulation(time_dim, model_dim, 9)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim)

        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = MultiheadCrossAttention(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(
            self, visual_embed, text_embed, time_embed, 
            rope, visual_cu_seqlens, text_cu_seqlens, 
            time_embed_idx, block_mask, torch_mask, sparse_params
            ):
        self_attn_params, cross_attn_params, ff_params = torch.chunk(
            self.visual_modulation(time_embed), 3, dim=-1
            )
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(
            self.self_attention_norm, visual_embed, 
            scale, shift, time_embed_idx
            ).type_as(visual_embed)
        visual_out = self.self_attention(
            visual_out, rope, visual_cu_seqlens, block_mask, torch_mask, sparse_params
            )
        visual_embed = apply_gate_sum(
            visual_embed, visual_out, gate, time_embed_idx
            ).type_as(visual_embed)

        shift, scale, gate = torch.chunk(cross_attn_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(
            self.cross_attention_norm, visual_embed, scale, shift, time_embed_idx
            ).type_as(visual_embed)
        visual_out = self.cross_attention(
            visual_out, text_embed, visual_cu_seqlens, text_cu_seqlens
            )
        visual_embed = apply_gate_sum(
            visual_embed, visual_out, gate, time_embed_idx
            ).type_as(visual_embed)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(
            self.feed_forward_norm, visual_embed, scale, shift, time_embed_idx
            ).type_as(visual_embed)
        visual_out = self.feed_forward(visual_out)
        visual_embed = apply_gate_sum(
            visual_embed, visual_out, gate, time_embed_idx
            ).type_as(visual_embed)

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
        visual_cond=False
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
        self.visual_embeddings = VisualEmbeddings(visual_embed_dim, model_dim, patch_size)

        self.text_rope_embeddings = RoPE1D(head_dim)
        self.text_transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(model_dim, time_dim, ff_dim, head_dim)
            for _ in range(num_text_blocks)
        ])

        self.visual_rope_embeddings = RoPE3D(axes_dims)
        self.visual_transformer_blocks = nn.ModuleList([
            TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim)
            for _ in range(num_visual_blocks)
        ])

        self.out_layer = OutLayer(model_dim, time_dim, out_visual_dim, patch_size)

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
        text_embed = self.text_embeddings(text_embed)
        time_embed, time_embed_idx = self.time_embeddings(time)
        time_embed = time_embed + self.pooled_text_embeddings(pooled_text_embed)
        visual_embed, visual_cu_seqlens = self.visual_embeddings(x, visual_cu_seqlens)

        text_rope = self.text_rope_embeddings(text_rope_pos)
        text_time_embed_idx = time_embed_idx.repeat_interleave(torch.diff(text_cu_seqlens), dim=0)
        for text_transformer_block in self.text_transformer_blocks:
            text_embed = text_transformer_block(
                text_embed, time_embed, text_rope, text_cu_seqlens, text_time_embed_idx
            )

        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor)
        to_fractal = sparse_params["to_fractal"] if sparse_params is not None else False
        visual_embed, visual_rope, visual_cu_seqlens = fractal_flatten(
            visual_embed, visual_rope, visual_cu_seqlens, visual_shape, block_mask=to_fractal
            )
        visual_time_embed_idx = time_embed_idx.repeat_interleave(
            torch.diff(visual_cu_seqlens), dim=0
            )

        for visual_transformer_block in self.visual_transformer_blocks:
            visual_embed = visual_transformer_block(
                visual_embed, text_embed, time_embed,
                visual_rope, visual_cu_seqlens, text_cu_seqlens,
                visual_time_embed_idx,
                block_mask, torch_mask, sparse_params
            )
        visual_embed, visual_cu_seqlens = fractal_unflatten(
            visual_embed, visual_cu_seqlens, visual_shape, block_mask=to_fractal
            )
        visual_time_embed_idx = time_embed_idx.repeat_interleave(
            torch.diff(visual_cu_seqlens), dim=0
            )
        x = self.out_layer(
            visual_embed, text_embed, time_embed, visual_cu_seqlens, visual_time_embed_idx
            )        
        return x


def get_dit(conf):
    dit = DiffusionTransformer3D(**conf)
    return dit
