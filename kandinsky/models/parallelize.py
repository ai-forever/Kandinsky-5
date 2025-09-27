from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)


def parallelize_dit(model, tp_mesh):
    if tp_mesh.size() > 1:
        plan = {
            "in_layer": ColwiseParallel(),
            "out_layer": RowwiseParallel(
                output_layouts=Replicate(),
            ),
        }
        parallelize_module(model.time_embeddings, tp_mesh, plan)

        plan = {
            "in_layer": ColwiseParallel(
                output_layouts=Replicate(),
            )
        }
        parallelize_module(model.text_embeddings, tp_mesh, plan)
        parallelize_module(model.pooled_text_embeddings, tp_mesh, plan)
        parallelize_module(model.visual_embeddings, tp_mesh, plan)

        for visual_transformer_block in model.visual_transformer_blocks:
            plan = {
                "visual_modulation": PrepareModuleInput(
                    input_layouts=(None),
                    desired_input_layouts=(Replicate()),
                ),
                "visual_modulation.out_layer": ColwiseParallel(
                    output_layouts=Replicate(),
                ),
                "self_attention_norm": SequenceParallel(
                    sequence_dim=0, use_local_output=True
                ),
                "self_attention.to_query": ColwiseParallel(
                    input_layouts=Replicate(),
                ),
                "self_attention.to_key": ColwiseParallel(
                    input_layouts=Replicate(),
                ),
                "self_attention.to_value": ColwiseParallel(
                    input_layouts=Replicate(),
                ),
                "self_attention.query_norm": SequenceParallel(
                    sequence_dim=0, use_local_output=True
                ),
                "self_attention.key_norm": SequenceParallel(
                    sequence_dim=0, use_local_output=True
                ),
                "self_attention.out_layer": RowwiseParallel(
                    output_layouts=Replicate(),
                ),
                "cross_attention_norm": SequenceParallel(
                    sequence_dim=0, use_local_output=True
                ),
                "cross_attention.to_query": ColwiseParallel(
                    input_layouts=Replicate(),
                ),
                "cross_attention.to_key": ColwiseParallel(
                    input_layouts=Replicate(),
                ),
                "cross_attention.to_value": ColwiseParallel(
                    input_layouts=Replicate(),
                ),
                "cross_attention.query_norm": SequenceParallel(
                    sequence_dim=0, use_local_output=True
                ),
                "cross_attention.key_norm": SequenceParallel(
                    sequence_dim=0, use_local_output=True
                ),
                "cross_attention.out_layer": RowwiseParallel(
                    output_layouts=Replicate(),
                ),
                "feed_forward_norm": SequenceParallel(
                    sequence_dim=0, use_local_output=True
                ),
                "feed_forward.in_layer": ColwiseParallel(),
                "feed_forward.out_layer": RowwiseParallel(),
            }
            self_attn = visual_transformer_block.self_attention
            self_attn.num_heads = self_attn.num_heads // tp_mesh.size()

            cross_attn = visual_transformer_block.cross_attention
            cross_attn.num_heads = cross_attn.num_heads // tp_mesh.size()

            parallelize_module(visual_transformer_block, tp_mesh, plan)

        plan = {
            "out_layer": ColwiseParallel(
                output_layouts=Replicate(),
            ),
        }
        parallelize_module(model.out_layer, tp_mesh, plan)

    return model
