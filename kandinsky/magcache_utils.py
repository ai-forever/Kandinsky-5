import numpy as np
import json

import torch

from .models.utils import fractal_flatten, fractal_unflatten


def nearest_interp(src_array, target_length):
    src_length = len(src_array)
    if target_length == 1:
        return np.array([src_array[-1]])

    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src_array[mapped_indices]


def save_json(filename, obj_list):
    with open(filename+".json", "w") as f:
        json.dump(obj_list, f)


def set_magcache_params(dit, mag_ratios):
    print(f'using Magcache')
    dit.__class__.forward = magcache_forward
    dit.cnt = 0
    dit.num_steps = 50 * 2
    dit.magcache_thresh = 0.12
    dit.K = 2
    dit.accumulated_err = [0.0, 0.0]
    dit.accumulated_steps = [0, 0]
    dit.accumulated_ratio = [1.0, 1.0]
    dit.retention_ratio = 0.2
    dit.residual_cache = [None, None]
    dit.mag_ratios = np.array([1.0]*2 + mag_ratios)

    if len(dit.mag_ratios) != 50 * 2:
        print(f'interpolate MAG RATIOS: curr len {len(dit.mag_ratios)}')
        mag_ratio_con = nearest_interp(dit.mag_ratios[0::2], 50)
        mag_ratio_ucon = nearest_interp(dit.mag_ratios[1::2], 50)
        interpolated_mag_ratios = np.concatenate(
            [mag_ratio_con.reshape(-1, 1), mag_ratio_ucon.reshape(-1, 1)], axis=1).reshape(-1)
        dit.mag_ratios = interpolated_mag_ratios

    # calibration
    # dit.__class__.forward = magcache_calibration
    # dit.cnt = 0
    # dit.num_steps = 50 * 2
    # dit.norm_ratio = [] # mean of magnitude ratio
    # dit.norm_std = [] # std of magnitude ratio
    # dit.cos_dis = [] # cosine distance of residual features
    # dit.residual_cache = [None, None]


@torch.compile(mode="max-autotune-no-cudagraphs")
def magcache_forward(
    self,
    x,
    text_embed,
    pooled_text_embed,
    time,
    visual_rope_pos,
    text_rope_pos,
    scale_factor=(1.0, 1.0, 1.0),
    sparse_params=None
):
    text_embed, time_embed, text_rope, visual_embed = self.before_text_transformer_blocks(
        text_embed, time, pooled_text_embed, x, text_rope_pos)

    for text_transformer_block in self.text_transformer_blocks:
        text_embed = text_transformer_block(text_embed, time_embed, text_rope)

    visual_embed, visual_shape, to_fractal, visual_rope = self.before_visual_transformer_blocks(
        visual_embed, visual_rope_pos, scale_factor, sparse_params)

    skip_forward = False
    ori_visual_embed = visual_embed

    if self.cnt>=int(self.num_steps*self.retention_ratio):
        cur_mag_ratio = self.mag_ratios[self.cnt] # conditional and unconditional in one list
        self.accumulated_ratio[self.cnt%2] = self.accumulated_ratio[self.cnt%2]*cur_mag_ratio 
        self.accumulated_steps[self.cnt%2] += 1 # skip steps plus 1
        cur_skip_err = np.abs(1-self.accumulated_ratio[self.cnt%2]) # skip error of current steps
        self.accumulated_err[self.cnt%2] += cur_skip_err # accumulated error of multiple steps
        
        if self.accumulated_err[self.cnt%2]<self.magcache_thresh and self.accumulated_steps[self.cnt%2]<=self.K:
            skip_forward = True
            residual_visual_embed = self.residual_cache[self.cnt%2]
        else:
            self.accumulated_err[self.cnt%2] = 0
            self.accumulated_steps[self.cnt%2] = 0
            self.accumulated_ratio[self.cnt%2] = 1.0

    if skip_forward: # skip this step with cached residual
        visual_embed =  visual_embed + residual_visual_embed
    else:
        # Original Forward 
        for visual_transformer_block in self.visual_transformer_blocks:
            visual_embed = visual_transformer_block(visual_embed, text_embed, time_embed,
                                                    visual_rope, sparse_params) 
        residual_visual_embed = visual_embed - ori_visual_embed

    self.residual_cache[self.cnt%2] = residual_visual_embed 

    x = self.after_blocks(visual_embed, visual_shape, to_fractal, text_embed, time_embed)

    self.cnt += 1

    if self.cnt >= self.num_steps: 
        self.cnt = 0
        self.accumulated_ratio = [1.0, 1.0]
        self.accumulated_err = [0.0, 0.0]
        self.accumulated_steps = [0, 0]
    return x


def magcache_calibration(
    self,
    x,
    text_embed,
    pooled_text_embed,
    time,
    visual_rope_pos,
    text_rope_pos,
    scale_factor=(1.0, 1.0, 1.0),
    sparse_params=None
):
    text_embed, time_embed, text_rope, visual_embed = self.before_text_transformer_blocks(
        text_embed, time, pooled_text_embed, x, text_rope_pos)

    for text_transformer_block in self.text_transformer_blocks:
        text_embed = text_transformer_block(text_embed, time_embed, text_rope)

    visual_embed, visual_shape, to_fractal, visual_rope = self.before_visual_transformer_blocks(
        visual_embed, visual_rope_pos, scale_factor, sparse_params)

    ori_visual_embed = visual_embed
    for visual_transformer_block in self.visual_transformer_blocks:
        visual_embed = visual_transformer_block(visual_embed, text_embed, time_embed,
                                                visual_rope, sparse_params)  
    residual_visual_embed = visual_embed - ori_visual_embed

    if self.cnt >= 2:
        norm_ratio = ((residual_visual_embed.norm(dim=-1)/self.residual_cache[self.cnt%2].norm(dim=-1)).mean()).item()
        norm_std = (residual_visual_embed.norm(dim=-1)/self.residual_cache[self.cnt%2].norm(dim=-1)).std().item()
        cos_dis = (1-torch.nn.functional.cosine_similarity(residual_visual_embed, self.residual_cache[self.cnt%2], dim=-1, eps=1e-8)).mean().item()
        self.norm_ratio.append(round(norm_ratio, 5))
        self.norm_std.append(round(norm_std, 5))
        self.cos_dis.append(round(cos_dis, 5))
        print(f"time: {self.cnt}, norm_ratio: {norm_ratio}, norm_std: {norm_std}, cos_dis: {cos_dis}")

    self.residual_cache[self.cnt%2] = residual_visual_embed 

    x = self.after_blocks(visual_embed, visual_shape, to_fractal, text_embed, time_embed)

    self.cnt += 1

    if self.cnt >= self.num_steps: # clear the history of current video and prepare for generating the next video.
        self.cnt = 0
        print("norm ratio")
        print(self.norm_ratio)
        print("norm std")
        print(self.norm_std)
        print("cos_dis")
        print(self.cos_dis)
        save_json("kandinsky5_mag_ratio", self.norm_ratio)
        save_json("kandinsky5_mag_std", self.norm_std)
        save_json("kandinsky5_cos_dis", self.cos_dis)
    return x
