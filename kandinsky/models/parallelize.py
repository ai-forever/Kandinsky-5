# Deferred: torch.distributed._tensor / tensor.parallel require the C
# extension built with USE_DISTRIBUTED=1, which most non-CUDA (e.g. MPS)
# and many single-GPU CUDA builds don't enable. Both functions below are
# only ever exercised when tp_mesh.size() > 1 (real multi-GPU tensor
# parallelism), so importing lazily lets single-device inference run on
# builds that lack torch.distributed support, mirroring the existing
# lazy `from torch.distributed.fsdp import ...` in get_distributed_pipeline.


def parallelize_dit(model, tp_mesh):
    if tp_mesh.size() > 1:
        from torch.distributed._tensor import Replicate, Shard
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            PrepareModuleInput,
            PrepareModuleOutput,
            RowwiseParallel,
            SequenceParallel,
            parallelize_module,
        )

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
  
  
def get_module_by_name(module, access_string):
    names = access_string.split('.')
    cur_m = module
    for n in names:
        cur_m = getattr(cur_m, n, None)
        if cur_m is None:
            break
    return cur_m
  
  
def update_plan_for_lora(root_module, plan):
    # Only reached from parallelize_seq's tp_mesh.size() > 1 branch - see the
    # module-level comment above parallelize_dit for why this is a local import.
    from torch.distributed.tensor.parallel import PrepareModuleInput, PrepareModuleOutput

    new_plan = {}

    for key, val in plan.items():
        m = get_module_by_name(root_module, key)

        if "lora" in str(m.__class__).lower() and hasattr(m, 'base_layer'):
            new_plan[key+".base_layer"] = val

            if isinstance(val, PrepareModuleOutput):
                new_plan[key+".lora_B.default"] = val

            if isinstance(val, PrepareModuleInput):
                new_plan[key+".lora_A.default"] = val

        else:
            new_plan[key] = val

    plan = new_plan
    return new_plan


def parallelize_seq(model, tp_mesh, mode='t2v'):
    if tp_mesh.size() > 1:
        from torch.distributed._tensor import Replicate, Shard
        from torch.distributed.tensor.parallel import (
            PrepareModuleInput,
            PrepareModuleOutput,
            SequenceParallel,
            parallelize_module,
        )

        if mode != 'i2v':
            plan_in = {
                "out_layer": PrepareModuleInput(
                        input_layouts=(Replicate(), None, None),
                        desired_input_layouts=(Shard(1), None, None),
                        use_local_output=True
                    ),
                }
            
            if hasattr(model, 'base_model'):
                parallelize_module(model.base_model.model, tp_mesh, plan_in)
            else:
                parallelize_module(model, tp_mesh, plan_in)

            plan_out = {
                "visual_embeddings": PrepareModuleOutput(
                    output_layouts=(Shard(1)),
                    desired_output_layouts=(Replicate()),
                    )
            }
            
            if hasattr(model, 'base_model'):
                parallelize_module(model.base_model.model, tp_mesh, plan_out)
            else:
                parallelize_module(model, tp_mesh, plan_out)

        for i, block in enumerate(model.visual_transformer_blocks):
            plan = update_plan_for_lora(
                block,
                {
                "self_attention_norm": SequenceParallel(sequence_dim=1, use_local_output=True),
                "self_attention.to_query": PrepareModuleOutput(
                    output_layouts=(Shard(1)), desired_output_layouts=(Shard(-1))
                ),
                "self_attention.to_key": PrepareModuleOutput(
                    output_layouts=(Shard(1)), desired_output_layouts=(Shard(-1))
                ),
                "self_attention.to_value": PrepareModuleOutput(
                    output_layouts=(Shard(1)), desired_output_layouts=(Shard(-1))
                ),
                "self_attention.out_layer": PrepareModuleInput(
                    input_layouts=(Shard(-1)),
                    desired_input_layouts=(Shard(0)),
                    use_local_output=True,
                ),
                "cross_attention_norm": SequenceParallel(sequence_dim=1, use_local_output=True),
                "cross_attention.to_query": PrepareModuleOutput(
                    output_layouts=(Shard(1)), desired_output_layouts=(Shard(-1))
                ),
                "cross_attention.to_key": PrepareModuleOutput(
                    output_layouts=(Replicate()), desired_output_layouts=(Shard(-1))
                ),
                "cross_attention.to_value": PrepareModuleOutput(
                    output_layouts=(Replicate()), desired_output_layouts=(Shard(-1))
                ),
                "cross_attention.out_layer": PrepareModuleInput(
                    input_layouts=(Shard(-1)),
                    desired_input_layouts=(Shard(0)),
                    use_local_output=True,
                ),
                "feed_forward_norm": SequenceParallel(sequence_dim=1, use_local_output=True),
                }
            )

            self_attn = block.self_attention
            self_attn.num_heads = self_attn.num_heads // tp_mesh.size()
            cross_attn = block.cross_attention
            cross_attn.num_heads = cross_attn.num_heads // tp_mesh.size()
            parallelize_module(block, tp_mesh, plan)
            #shard input of first block and idx for all blocks
            if i == 0:
                parallelize_module(
                    block,
                    tp_mesh,
                    PrepareModuleInput(
                        input_layouts=(Replicate(),None,None,None,None, None),
                        desired_input_layouts=(Shard(1),None,None,None,None, None),
                        use_local_output=True,
                    ),
                )

            if i == len(model.visual_transformer_blocks)-1:
                parallelize_module(
                    block,
                    tp_mesh,
                    PrepareModuleOutput(
                        output_layouts=(Shard(1)),
                        desired_output_layouts=(Replicate())
                    ),
                )        

    return model
