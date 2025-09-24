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