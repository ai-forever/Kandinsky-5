def magcache_calibration(
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

    ori_visual_embed = visual_embed
    for visual_transformer_block in self.visual_transformer_blocks:
        visual_embed = visual_transformer_block(
            visual_embed, text_embed, time_embed, visual_rope, visual_cu_seqlens, text_cu_seqlens,
            visual_max_seqlen, text_max_seqlen, visual_time_embed_idx,
            block_mask, torch_mask, sparse_params
        )   
    residual_visual_embed = visual_embed - ori_visual_embed
    self.residual_cache[self.cnt%2] = residual_visual_embed 

    if self.cnt >= 2:
        norm_ratio = ((residual_visual_embed.norm(dim=-1)/self.residual_cache[self.cnt%2].norm(dim=-1)).mean()).item()
        norm_std = (residual_visual_embed.norm(dim=-1)/self.residual_cache[self.cnt%2].norm(dim=-1)).std().item()
        cos_dis = (1-torch.nn.functional.cosine_similarity(residual_visual_embed, self.residual_cache[self.cnt%2], dim=-1, eps=1e-8)).mean().item()
        self.norm_ratio.append(round(norm_ratio, 5))
        self.norm_std.append(round(norm_std, 5))
        self.cos_dis.append(round(cos_dis, 5))
        print(f"time: {self.cnt}, norm_ratio: {norm_ratio}, norm_std: {norm_std}, cos_dis: {cos_dis}")

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

    self.cnt += 1

    if self.cnt >= self.num_steps: # clear the history of current video and prepare for generating the next video.
        self.cnt = 0
        print("norm ratio")
        print(self.norm_ratio)
        print("norm std")
        print(self.norm_std)
        print("cos_dis")
        print(self.cos_dis)
        save_json("wan2_1_mag_ratio", self.norm_ratio)
        save_json("wan2_1_mag_std", self.norm_std)
        save_json("wan2_1_cos_dis", self.cos_dis)
    return x