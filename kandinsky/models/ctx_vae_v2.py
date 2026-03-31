import math
import os
from typing import Dict, Optional, Tuple, Union, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution

from .utils_cp_internals import (
    initialize_context_parallel,
    get_context_parallel_rank,
    get_context_parallel_world_size
)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True

import torch._inductor.config
torch._inductor.config.max_autotune_gemm_backends = "ATEN,TRITON,CUTLASS"  # "ATEN,TRITON,CUTLASS,CPP"
torch._inductor.config.max_autotune = True
torch._inductor.config.memory_planning = True
torch._inductor.config.force_same_precision = True


USE_TILING_DEC = True
USE_TILING_ENC = True
CHANNELS = 192
FRAMES = 49
MEMORY_FORMAT = torch.contiguous_format  # [torch.contiguous_format, torch.channels_last_3d]


def cast_tuple(t: Union[int, Tuple[int, int, int]], length: int = 3) -> Tuple[int, int, int]:
    return t if isinstance(t, tuple) else ((t,) * length)


def get_context(x: torch.Tensor) -> Dict:
    return {"dtype": x.dtype, "device": x.device, "memory_format": MEMORY_FORMAT}


@torch.compile(dynamic=True, fullgraph=False)
def prepare_causal_attention_mask(
    f: int, s: int, dtype: torch.dtype, device: torch.device, b: int
) -> torch.Tensor:
    return (
        torch.ones((1, f, 1, f, 1), dtype=dtype, device=device)
        .tril_()
        .log_()
        .expand(b, f, s, f, s)
        .reshape(b, f * s, f * s)
        .contiguous()
    )


class SimpleConv3d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv3d(*args, **kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.conv(hidden_states)


class HunyuanVideoCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
        pad_mode: str = "replicate",
    ) -> None:
        super().__init__()

        tks, hks, wks = cast_tuple(kernel_size, 3)
        self.time_causal_padding = (wks // 2, wks // 2, hks // 2, hks // 2, tks - 1, 0)
        self.pad_mode = pad_mode
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )

    @torch.compile(dynamic=True, fullgraph=False)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(hidden_states)


class HunyuanVideoUpsampleCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
        upsample_factor: Tuple[float, float, float] = (2, 2, 2),
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        self.upsample_factor = upsample_factor
        self.conv = HunyuanVideoCausalConv3d(
            in_channels, out_channels, kernel_size, stride, bias=bias
        )

    def upsample(self, hidden_states: torch.Tensor, is_2d: bool) -> torch.Tensor:
        N, C, T, H, W = hidden_states.shape
        ust, ush, usw = list(map(int, self.upsample_factor))
        if is_2d:
            ust = 1
        repeated = (
            hidden_states
            .view(N, C, T, 1, H, 1, W, 1)
            .expand(-1, -1, -1, ust, -1, ush, -1, usw)
            .reshape(N, C, T * ust, H * ush, W * usw)
        )

        return repeated

    @torch.compile(dynamic=True, fullgraph=False)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        N, C, T, H, W = hidden_states.shape
        if T == 1 or self.upsample_factor[0] == 1:
            repeated = self.upsample(hidden_states, is_2d=True)
        else:
            ust, ush, usw = list(map(int, self.upsample_factor))
            repeated = torch.empty((N, C, (T - 1) * ust + 1, H * ush, W * usw), **get_context(hidden_states))
            repeated[:, :, :1] = self.upsample(hidden_states[:, :, :1], is_2d=True)
            repeated[:, :, 1:] = self.upsample(hidden_states[:, :, 1:], is_2d=False)

        hidden_states = self.conv(repeated)
        return hidden_states


class HunyuanVideoDownsampleCausal3D(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        padding: int = 1,
        kernel_size: int = 3,
        bias: bool = True,
        stride: int = 2,
    ) -> None:
        super().__init__()
        out_channels = out_channels or channels

        self.conv = HunyuanVideoCausalConv3d(
            channels, out_channels, kernel_size, stride, padding, bias=bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        return hidden_states


class HunyuanVideoResnetBlockCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps, affine=True)
        self.conv1 = HunyuanVideoCausalConv3d(in_channels, out_channels, 3, 1, 0)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps, affine=True)
        self.conv2 = HunyuanVideoCausalConv3d(out_channels, out_channels, 3, 1, 0)

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = SimpleConv3d(
                in_channels, out_channels, 1, 1, 0
            )

    @torch.compile(dynamic=True, fullgraph=False)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        dtp = hidden_states.dtype

        hidden_states = self.norm1(hidden_states)
        hidden_states = F.silu(hidden_states, inplace=True).to(dtp)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = F.silu(hidden_states, inplace=True).to(dtp)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        hidden_states = hidden_states + residual
        return hidden_states


class HunyuanVideoMidBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_attention: bool = True,
        attention_head_dim: int = 1,
    ) -> None:
        super().__init__()
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )
        self.add_attention = add_attention

        # There is always at least one resnet
        resnets = [
            HunyuanVideoResnetBlockCausal3D(
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                non_linearity=resnet_act_fn,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                HunyuanVideoResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    @torch.compile(dynamic=True, fullgraph=False)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                batch_size, _, num_frames, height, width = hidden_states.shape
                hidden_states = hidden_states.permute(0, 2, 3, 4, 1).flatten(1, 3)
                mask = prepare_causal_attention_mask(
                    num_frames,
                    height * width,
                    hidden_states.dtype,
                    hidden_states.device,
                    batch_size,
                )
                hidden_states = attn(hidden_states, attention_mask=mask)
                hidden_states = hidden_states.unflatten(
                    1, (num_frames, height, width)
                ).permute(0, 4, 1, 2, 3)

            hidden_states = resnet(hidden_states)

        return hidden_states


class HunyuanVideoDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_stride: int = 2,
        downsample_padding: int = 1,
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                HunyuanVideoResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    HunyuanVideoDownsampleCausal3D(
                        out_channels,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        stride=downsample_stride,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class HunyuanVideoUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_upsample: bool = True,
        upsample_scale_factor: Tuple[int, int, int] = (2, 2, 2),
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                HunyuanVideoResnetBlockCausal3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    HunyuanVideoUpsampleCausal3D(
                        out_channels,
                        out_channels=out_channels,
                        upsample_factor=upsample_scale_factor,
                    )
                ]
            )
        else:
            self.upsamplers = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class HunyuanVideoEncoder3D(nn.Module):
    r"""
    Causal encoder for 3D video-like data introduced
    in [Hunyuan Video](https://huggingface.co/papers/2412.03603).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = (
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
        temporal_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
    ) -> None:
        super().__init__()

        self.conv_in = HunyuanVideoCausalConv3d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1
        )
        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            if down_block_type != "HunyuanVideoDownBlock3D":
                raise ValueError(f"Unsupported down_block_type: {down_block_type}")

            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_downsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_downsample_layers = int(np.log2(temporal_compression_ratio))

            if temporal_compression_ratio == 4:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(
                    i >= (len(block_out_channels) - 1 - num_time_downsample_layers)
                    and not is_final_block
                )
            elif temporal_compression_ratio == 8:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(i < num_time_downsample_layers)
            else:
                raise ValueError(
                    f"Unsupported time_compression_ratio: {temporal_compression_ratio}"
                )

            downsample_stride_HW = (2, 2) if add_spatial_downsample else (1, 1)
            downsample_stride_T = (2,) if add_time_downsample else (1,)
            downsample_stride = tuple(downsample_stride_T + downsample_stride_HW)

            down_block = HunyuanVideoDownBlock3D(
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=bool(add_spatial_downsample or add_time_downsample),
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                downsample_stride=downsample_stride,
                downsample_padding=0,
            )

            self.down_blocks.append(down_block)

        self.mid_block = HunyuanVideoMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = HunyuanVideoCausalConv3d(
            block_out_channels[-1], conv_out_channels, kernel_size=3
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)

        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)

        hidden_states = self.mid_block(hidden_states)

        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


def split_input_over_time_dim(x: torch.Tensor, temporal_compression: int = 4):
    cp_world_size = get_context_parallel_world_size()
    # Bypass the function if context parallel is 1
    if cp_world_size == 1:
        return x, [(x.shape[2] - 1) * temporal_compression + 1]

    cp_rank = get_context_parallel_rank()
    input_size = (x.shape[2] - 1)
    if input_size == 60:
        indices_1 = list(range(0, 41, 8)) + [46, 52]
        indices_2 = list(range(11, 44, 8)) + [49, 55, 61]
    elif input_size == 30:
        indices_1 = list(range(0, 25, 4)) + [26]
        indices_2 = list(range(7, 28, 4)) + [29, 31]
    else:
        raise NotImplementedError("Only 121(31) and 241(61) frames are supported now")

    output = x[:, :, indices_1[cp_rank]:indices_2[cp_rank]].contiguous()

    indices = [
        ((indices_2[j] - indices_1[j] - 1) * temporal_compression + 1 * (j == 0))
        for j in range(cp_world_size)
    ]

    return output, indices


class HunyuanVideoDecoder3D(nn.Module):
    r"""
    Causal decoder for 3D video-like data introduced
    in [Hunyuan Video](https://huggingface.co/papers/2412.03603).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = (
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention=True,
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = HunyuanVideoCausalConv3d(
            in_channels, block_out_channels[-1], kernel_size=3, stride=1
        )
        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = HunyuanVideoMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            if up_block_type != "HunyuanVideoUpBlock3D":
                raise ValueError(f"Unsupported up_block_type: {up_block_type}")

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_upsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_upsample_layers = int(np.log2(time_compression_ratio))

            if time_compression_ratio == 4:
                add_spatial_upsample = bool(i < num_spatial_upsample_layers)
                add_time_upsample = bool(
                    i >= len(block_out_channels) - 1 - num_time_upsample_layers
                    and not is_final_block
                )
            else:
                raise ValueError(
                    f"Unsupported time_compression_ratio: {time_compression_ratio}"
                )

            upsample_scale_factor_HW = (2, 2) if add_spatial_upsample else (1, 1)
            upsample_scale_factor_T = (2,) if add_time_upsample else (1,)
            upsample_scale_factor = tuple(
                upsample_scale_factor_T + upsample_scale_factor_HW
            )

            up_block = HunyuanVideoUpBlock3D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=bool(add_spatial_upsample or add_time_upsample),
                upsample_scale_factor=upsample_scale_factor,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_out = HunyuanVideoCausalConv3d(
            block_out_channels[0], out_channels, kernel_size=3
        )

    @torch.compile(dynamic=True, fullgraph=False)
    def piece_0a(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.last_upsample(hidden_states)
        hidden_states = self.up_blocks[-1].resnets[0](hidden_states)
        return hidden_states

    # @torch.compile(dynamic=True, fullgraph=False)
    def piece_0(self, hidden_states: torch.Tensor, has_split: bool = False) -> torch.Tensor:
        if has_split:
            N, C, T, H, W = hidden_states.shape
            output = torch.empty((N, C // 2, (2 * T - 1), 2 * H, 2 * W), **get_context(hidden_states))
            W2 = W // 2
            output[:, :, :, :, :W] = self.piece_0a(hidden_states[:, :, :, :, :W2 + 2])[:, :, :, :, :W]
            output[:, :, :, :, W:] = self.piece_0a(hidden_states[:, :, :, :, W2 - 2:])[:, :, :, :, 4:W + 4]
        else:
            output = self.piece_0a(hidden_states)
        return output

    @torch.compile(dynamic=True, fullgraph=False)
    def piece_1(self, hidden_states: torch.Tensor) -> torch.Tensor:

        dtp = hidden_states.dtype
        resnets = self.up_blocks[-1].resnets
        hidden_states = resnets[1](hidden_states)
        hidden_states = resnets[2](hidden_states)

        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = F.silu(hidden_states, inplace=True).to(dtp)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states

    # @torch.compile(dynamic=True, fullgraph=False)
    def piece_2(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(memory_format=MEMORY_FORMAT)
        hidden_states = self.conv_in(hidden_states)
        hidden_states = self.mid_block(hidden_states)
        for up_block in self.up_blocks[:-1]:
            hidden_states = up_block(hidden_states)

        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.piece_2(hidden_states)
        hidden_states = self.piece_0(hidden_states, has_split=True)
        hidden_states = self.piece_1(hidden_states)
        return hidden_states

    def update(self):

        self.last_upsample = self.up_blocks[2].upsamplers[0]
        self.up_blocks[2].upsamplers = None

class AutoencoderKLHunyuanVideo(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding videos into latents
    and decoding latent representations into videos.
    Introduced in [HunyuanVideo](https://huggingface.co/papers/2412.03603).

    This model inherits from [`ModelMixin`]. Check the superclass
    documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        down_block_types: Tuple[str, ...] = (
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
        ),
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        scaling_factor: float = 0.476986,
        spatial_compression_ratio: int = 8,
        temporal_compression_ratio: int = 4,
        mid_block_add_attention: bool = True,
    ) -> None:
        super().__init__()

        self.time_compression_ratio = temporal_compression_ratio

        self.encoder = HunyuanVideoEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
            temporal_compression_ratio=temporal_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
        )
        self.encoder = nn.utils.convert_conv3d_weight_memory_format(self.encoder, MEMORY_FORMAT)

        self.decoder = HunyuanVideoDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            time_compression_ratio=temporal_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.quant_conv = nn.Conv3d(
            2 * latent_channels, 2 * latent_channels, kernel_size=1
        )
        self.post_quant_conv = nn.Conv3d(
            latent_channels, latent_channels, kernel_size=1
        )

        self.spatial_compression_ratio = spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio
        self.latent_channels = latent_channels
        self.out_channels = out_channels

        self.use_slicing = False

        self.use_tiling_enc = True
        self.use_tiling_dec = True

        self.use_framewise_encoding = True
        self.use_framewise_decoding = True

        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_num_frames = 16

        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192
        self.tile_sample_stride_num_frames = 12

        self.tile_latent_min_height = self.tile_sample_min_height // spatial_compression_ratio
        self.tile_latent_min_width  = self.tile_sample_min_width  // spatial_compression_ratio
        self.tile_latent_stride_height = self.tile_sample_stride_height // spatial_compression_ratio
        self.tile_latent_stride_width  = self.tile_sample_stride_width  // spatial_compression_ratio
        self.tile_latent_min_num_frames = self.tile_sample_min_num_frames // temporal_compression_ratio
        self.tile_latent_stride_num_frames = self.tile_sample_stride_num_frames // temporal_compression_ratio

        self.tile_size = None
        self.is_parallel = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        _, _, num_frames, height, width = x.shape

        if self.use_framewise_decoding and num_frames > (self.tile_sample_min_num_frames + 1):
            return self._temporal_tiled_encode(x)

        if self.use_tiling_enc and (
            width > self.tile_sample_min_width or height > self.tile_sample_min_height
        ):
            return self.tiled_encode(x)

        x = self.encoder(x)
        enc = self.quant_conv(x)
        return enc

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, opt_tiling: bool = True, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        r"""
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`]
                instead of a plain tuple.

        Returns:
                The latent representations of the encoded videos. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned,
                otherwise a plain `tuple` is returned.
        """
        if opt_tiling:
            tile_size, tile_stride = self.get_enc_optimal_tiling(x.shape)
        else:
            b, _, f, h, w = x.shape
            tile_size, tile_stride = (b, f, h, w), (f, h, w)
        if tile_size != self.tile_size:
            self.tile_size = tile_size
            self.apply_tiling(tile_size, tile_stride)

        h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        _, _, num_frames, height, width = z.shape

        if self.use_framewise_decoding and num_frames > (self.tile_latent_min_num_frames + 1):
            return self._temporal_tiled_decode(z, return_dict=return_dict)

        if self.use_tiling_dec and (
            width > self.tile_latent_min_width or height > self.tile_latent_min_height
        ):
            return self.tiled_decode(z, return_dict=return_dict)

        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self, z: torch.Tensor, return_dict: bool = True, verbose: bool = False,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        r"""
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned,
                otherwise a plain `tuple` is returned.
        """
        # z = z.to(memory_format=torch.channels_last_3d)
        tile_size, tile_stride = self.get_dec_optimal_tiling(z.shape, verbose=verbose)
        if tile_size != self.tile_size:
            self.tile_size = tile_size
            self.apply_tiling(tile_size, tile_stride)

        decoded = self._decode(z).sample
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def prepare_for_blending(self, y: torch.Tensor, cp_rank: int, cp_world_size: int, blend_extent: int) -> torch.Tensor:
        if cp_rank > 0:
            y = y[:, :, 1:]

        range_right = torch.arange(blend_extent, dtype=y.dtype, device=y.device).view(1, 1, -1, 1, 1) / blend_extent
        range_left = 1 - range_right

        if cp_rank > 0:
            y[:, :, :blend_extent, :, :] *= range_right

        if cp_rank < (cp_world_size - 1):
            y[:, :, -blend_extent:, :, :] *= range_left

        return y

    def blend_tiles(self, y: torch.Tensor, indices: List[int], num_frames: int) -> torch.Tensor:
        ## blending via add
        cp_rank = get_context_parallel_rank()
        cp_world_size = get_context_parallel_world_size()
        blend_num_frames = self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames

        y = self.prepare_for_blending(y, cp_rank, cp_world_size, blend_num_frames)
        N, C, _, H, W = y.shape

        tensor_list = [
            torch.empty((N, C, indices[j], H, W), **get_context(y)) for j in range(cp_world_size)
        ]
        torch.distributed.all_gather(tensor_list, y.contiguous())

        output = torch.empty((N, C, num_frames, H, W), **get_context(y))
        output.fill_(0)
        begin = 0
        for i, tile in enumerate(tensor_list):
            end = begin + indices[i]
            output[:, :, begin:end] += tile
            begin = end - blend_num_frames
        return output

    @apply_forward_hook
    def decode_parallel(
        self, z: torch.Tensor, return_dict: bool = True, verbose: bool = False
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:

        cp_rank = get_context_parallel_rank()
        verbose = verbose and cp_rank == 0
        num_frames = self.temporal_compression_ratio * (z.shape[2] - 1) + 1

        x, indices = split_input_over_time_dim(z, self.temporal_compression_ratio)
        y = self.decode(x, verbose=verbose).sample
        decoded = self.blend_tiles(y, indices, num_frames)

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode a batch of images using a tiled encoder.

        Args:
            x (`torch.Tensor`): Input batch of videos.

        Returns:
            `torch.Tensor`:
                The latent representation of the encoded videos.
        """
        N, _, frames, height, width = x.shape
        latent_frames = (frames - 1) // self.temporal_compression_ratio + 1
        latent_height = height // self.spatial_compression_ratio
        latent_width  = width  // self.spatial_compression_ratio
        context = {"dtype": x.dtype, "device": x.device}

        output = torch.empty((N, 2 * self.latent_channels, latent_frames, latent_height, latent_width), **get_context(x))
        output.fill_(0)

        blend_sample_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_sample_width  = self.tile_sample_min_width - self.tile_sample_stride_width
        blend_height = blend_sample_height // self.spatial_compression_ratio
        blend_width  = blend_sample_width  // self.spatial_compression_ratio

        range_right_height = torch.arange(blend_height, **context).view(1, 1, 1, -1, 1) / blend_height
        range_right_width  = torch.arange(blend_width,  **context).view(1, 1, 1, 1, -1) / blend_width 
        range_left_height, range_left_width = (1 - range_right_height), (1 - range_right_width)

        height_indices = list(range(0, height - blend_sample_height, self.tile_sample_stride_height))
        width_indices  = list(range(0, width  - blend_sample_width,  self.tile_sample_stride_width))
        
        i_begin = 0
        for i in height_indices:
            j_begin = 0
            for j in width_indices:
                tile = x[ :, :, :, i:i + self.tile_sample_min_height, j:j + self.tile_sample_min_width]
                tile = self.encoder(tile).clone()
                tile = self.quant_conv(tile)

                if i > height_indices[0]:
                    tile[:, :, :, :blend_height, :]  *= range_right_height
                if i < height_indices[-1]:
                    tile[:, :, :, -blend_height:, :] *= range_left_height

                if j > width_indices[0]:
                    tile[:, :, :, :, :blend_width]  *= range_right_width
                if j < width_indices[-1]:
                    tile[:, :, :, :, -blend_width:] *= range_left_width

                tile_height, tile_width = tile.shape[3:]
                i_end, j_end = (i_begin + tile_height), (j_begin + tile_width)
                output[:, :, :, i_begin:i_end, j_begin:j_end] += tile
                j_begin = j_end - blend_width
            i_begin = i_end - blend_height

        return output

    def tiled_decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned,
                otherwise a plain `tuple` is returned.
        """

        N, _, frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width  = width  * self.spatial_compression_ratio
        sample_frames = (frames - 1) * self.temporal_compression_ratio + 1
        context = {"dtype": z.dtype, "device": z.device}

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width  = self.tile_sample_min_width  - self.tile_sample_stride_width
        blend_latent_height = blend_height // self.spatial_compression_ratio
        blend_latent_width  = blend_width  // self.spatial_compression_ratio

        output = torch.empty((N, self.out_channels, sample_frames, sample_height, sample_width), **get_context(z))
        output.fill_(0)

        range_right_height = torch.arange(blend_height, **context).view(1, 1, 1, -1, 1) / blend_height
        range_right_width  = torch.arange(blend_width,  **context).view(1, 1, 1, 1, -1) / blend_width 
        range_left_height, range_left_width = (1 - range_right_height), (1 - range_right_width)

        height_indices = list(range(0, height - blend_latent_height, self.tile_latent_stride_height))
        width_indices  = list(range(0, width  - blend_latent_width,  self.tile_latent_stride_width))
        
        i_begin = 0
        for i in height_indices:
            j_begin = 0
            for j in width_indices:
                tile = z[:, :, :, i:i + self.tile_latent_min_height, j:j + self.tile_latent_min_width]
                tile = self.post_quant_conv(tile)
                tile = self.decoder(tile)

                if i > height_indices[0]:
                    tile[:, :, :, :blend_height, :]  *= range_right_height
                if i < height_indices[-1]:
                    tile[:, :, :, -blend_height:, :] *= range_left_height

                if j > width_indices[0]:
                    tile[:, :, :, :, :blend_width]  *= range_right_width
                if j < width_indices[-1]:
                    tile[:, :, :, :, -blend_width:] *= range_left_width

                tile_height, tile_width = tile.shape[3:]
                i_end, j_end = (i_begin + tile_height), (j_begin + tile_width)
                output[:, :, :, i_begin:i_end, j_begin:j_end] += tile
                j_begin = j_end - blend_width
            i_begin = i_end - blend_height

        if not return_dict:
            return (output,)
        return DecoderOutput(sample=output)

    def _temporal_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        N, _, num_frames, height, width = x.shape
        latent_num_frames = (num_frames - 1) // self.temporal_compression_ratio + 1
        latent_heigth = height // self.spatial_compression_ratio
        latent_width  = width  // self.spatial_compression_ratio
        context = {"dtype": x.dtype, "device": x.device}

        blend_sample_frames = self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
        blend_num_frames = blend_sample_frames // self.temporal_compression_ratio

        output = torch.empty((N, 2 * self.latent_channels, latent_num_frames, latent_heigth, latent_width), **get_context(x))
        output.fill_(0)

        range_right = torch.arange(blend_num_frames, **context).view(1, 1, -1, 1, 1) / blend_num_frames
        range_left = 1 - range_right

        frames_indices = list(range(0, num_frames - blend_sample_frames, self.tile_sample_stride_num_frames))

        i_begin = 0
        for i in frames_indices:
            tile = x[:, :, i : i + self.tile_sample_min_num_frames + 1, :, :]
            if self.use_tiling_enc and (height > self.tile_sample_min_height or width > self.tile_sample_min_width):
                tile = self.tiled_encode(tile)
            else:
                tile = self.encoder(tile).clone()
                tile = self.quant_conv(tile)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            
            if i > frames_indices[0]:
                tile[:, :, :blend_num_frames, :, :]  *= range_right
            if i < frames_indices[-1]:
                tile[:, :, -blend_num_frames:, :, :] *= range_left

            i_end = i_begin + tile.shape[2]
            output[:, :, i_begin:i_end, :, :] += tile
            i_begin = i_end - blend_num_frames

        return output

    def _temporal_tiled_decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        N, _, num_frames, latent_heigth, latent_width = z.shape
        num_sample_frames = (num_frames - 1) * self.temporal_compression_ratio + 1
        sample_heigth = latent_heigth * self.spatial_compression_ratio
        sample_width = latent_width * self.spatial_compression_ratio
        context = {"dtype": z.dtype, "device": z.device}

        blend_num_frames = self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
        blend_latent_fames = blend_num_frames // self.temporal_compression_ratio

        output = torch.empty((N, self.out_channels, num_sample_frames, sample_heigth, sample_width), **get_context(z))
        output.fill_(0)

        range_right = torch.arange(blend_num_frames, **context).view(1, 1, -1, 1, 1) / blend_num_frames
        range_left = 1 - range_right

        frames_indices = list(range(0, num_frames - blend_latent_fames, self.tile_latent_stride_num_frames))
        
        i_begin = 0
        for i in frames_indices:
            tile = z[:, :, i : i + self.tile_latent_min_num_frames + 1, :, :]
            _, tile_height, tile_width = tile.shape[2:]

            if self.use_tiling_dec and (
                tile_height > self.tile_latent_min_height or tile_width > self.tile_latent_min_width
            ):
                tile = self.tiled_decode(tile, return_dict=True).sample
            else:
                tile = self.post_quant_conv(tile)
                tile = self.decoder(tile)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            
            if i > frames_indices[0]:
                tile[:, :, :blend_num_frames, :, :]  *= range_right
            if i < frames_indices[-1]:
                tile[:, :, -blend_num_frames:, :, :] *= range_left

            i_end = i_begin + tile.shape[2]
            output[:, :, i_begin:i_end, :, :] += tile
            i_begin = i_end - blend_num_frames

        if not return_dict:
            return (output,)
        return DecoderOutput(sample=output)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, return_dict=return_dict)
        return dec

    def apply_tiling(
        self, tile: Tuple[int, int, int, int], stride: Tuple[int, int, int]
    ):
        """Applies tiling."""
        _, ft, ht, wt = tile
        fs, hs, ws = stride

        self.use_tiling_enc = USE_TILING_ENC
        self.use_tiling_dec = USE_TILING_DEC
        self.tile_sample_min_num_frames = ft - 1
        self.tile_sample_stride_num_frames = fs
        self.tile_sample_min_height = ht
        self.tile_sample_min_width = wt
        self.tile_sample_stride_height = hs
        self.tile_sample_stride_width = ws
        self.tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        self.tile_latent_min_width  = self.tile_sample_min_width  // self.spatial_compression_ratio
        self.tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        self.tile_latent_stride_width  = self.tile_sample_stride_width  // self.spatial_compression_ratio
        self.tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
        self.tile_latent_stride_num_frames = self.tile_sample_stride_num_frames // self.temporal_compression_ratio

    def get_enc_optimal_tiling(
        self, shape: List[int], frames: int = FRAMES, channels: int = CHANNELS, verbose: bool = False
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int]]:
        """Returns optimal tiling for given shape."""
        h, w = shape[3:]

        frames = min((frames, shape[2]))
        free_mem = torch.cuda.mem_get_info()[0]
        max_area = free_mem / channels / frames / 8
        num_vals = channels * frames * (h + 32) * (w + 32) 

        if h * w < max_area and num_vals < 2**31:
            return (1, frames, h, w), (frames - 9, h, w)

        def factorize(n, k):
            return math.ceil(math.sqrt(n / k)), math.ceil(math.sqrt(n * k))

        k = max(h / w, w / h)
        N = max(math.ceil(h * w / max_area), math.ceil(num_vals / 2**31))
        a, b = factorize(N, k)
        wn, hn = (a, b) if h >= w else (b, a)

        if wn > 1:
            wt = math.ceil(w / wn / 8) * 8 + 16
            ws = wt - 32
        else:
            wt = ws = w
        if hn > 1:
            ht = math.ceil(h / hn / 8) * 8 + 16
            hs = ht - 32
        else:
            ht = hs = h
        if verbose:
            print(f"{wn=}, {hn=}, {frames=}, {channels=}, {ht=}, {wt=}, {hs=}, {ws=}")
        return (1, frames, ht, wt), (frames - 9, hs, ws)

    def get_dec_optimal_tiling(
        self, shape: List[int], verbose: bool = False
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int]]:
        """Returns optimal tiling for given shape."""
        b, _, f, h, w = shape
        frames = (41 if f == 61 else 25) if self.is_parallel else 49
        enc_inp_shape = [b, 3, 4 * (f - 1) + 1, 8 * h, 8 * w]
        return self.get_enc_optimal_tiling(enc_inp_shape, frames=frames, verbose=verbose)

    def set_parallel(self, is_parallel: Optional[bool] = None):
        if is_parallel is not None:
            self.is_parallel = is_parallel
        elif hasattr(self, "is_parallel"):
            is_parallel = self.is_parallel
        else:
            raise ValueError(f"is_parallel and self.is_parallel are not given")

        if is_parallel:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_context_parallel(world_size)


def build_vae(conf):
    if conf.name == "hunyuan":
        vae = AutoencoderKLHunyuanVideo.from_pretrained(
            conf.checkpoint_path, subfolder="vae", torch_dtype=torch.float16
        )
        vae.decoder.update()
        # vae.decoder = nn.utils.convert_conv3d_weight_memory_format(vae.decoder, torch.channels_last_3d)
        vae.is_parallel = conf.get("is_parallel", False)
        # vae.set_parallel(conf.get("is_parallel", False))
        return vae
    elif conf.name == "flux":
        from diffusers.models import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(conf.checkpoint_path, subfolder="flux/vae", torch_dtype=torch.bfloat16)
        return vae
    else:
        assert False, f"unknown vae name {conf.name}"
