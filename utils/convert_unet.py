UNET_MAP_ATTENTIONS = {
    "proj_in.weight",
    "proj_in.bias",
    "proj_out.weight",
    "proj_out.bias",
    "norm.weight",
    "norm.bias",
}

TRANSFORMER_BLOCKS = {
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "norm3.weight",
    "norm3.bias",
    "attn1.to_q.weight",
    "attn1.to_k.weight",
    "attn1.to_v.weight",
    "attn1.to_out.0.weight",
    "attn1.to_out.0.bias",
    "attn2.to_q.weight",
    "attn2.to_k.weight",
    "attn2.to_v.weight",
    "attn2.to_out.0.weight",
    "attn2.to_out.0.bias",
    "ff.net.0.proj.weight",
    "ff.net.0.proj.bias",
    "ff.net.2.weight",
    "ff.net.2.bias",
}

UNET_MAP_RESNET = {
    "in_layers.2.weight": "conv1.weight",
    "in_layers.2.bias": "conv1.bias",
    "emb_layers.1.weight": "time_emb_proj.weight",
    "emb_layers.1.bias": "time_emb_proj.bias",
    "out_layers.3.weight": "conv2.weight",
    "out_layers.3.bias": "conv2.bias",
    "skip_connection.weight": "conv_shortcut.weight",
    "skip_connection.bias": "conv_shortcut.bias",
    "in_layers.0.weight": "norm1.weight",
    "in_layers.0.bias": "norm1.bias",
    "out_layers.0.weight": "norm2.weight",
    "out_layers.0.bias": "norm2.bias",
}

UNET_MAP_BASIC = {
    ("label_emb.0.0.weight", "class_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "class_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "class_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "class_embedding.linear_2.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias")
}
TEMPORAL_TRANSFORMER_BLOCKS = {
    "norm_in.weight",
    "norm_in.bias",
    "ff_in.net.0.proj.weight",
    "ff_in.net.0.proj.bias",
    "ff_in.net.2.weight",
    "ff_in.net.2.bias",
}
TEMPORAL_TRANSFORMER_BLOCKS.update(TRANSFORMER_BLOCKS)


TEMPORAL_UNET_MAP_ATTENTIONS = {
    "time_mixer.mix_factor",
}
TEMPORAL_UNET_MAP_ATTENTIONS.update(UNET_MAP_ATTENTIONS)


TEMPORAL_TRANSFORMER_MAP = {
    "time_pos_embed.0.weight": "time_pos_embed.linear_1.weight",
    "time_pos_embed.0.bias": "time_pos_embed.linear_1.bias",
    "time_pos_embed.2.weight": "time_pos_embed.linear_2.weight",
    "time_pos_embed.2.bias": "time_pos_embed.linear_2.bias",
}


TEMPORAL_RESNET = {
     "time_mixer.mix_factor",
}


unet_config = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False, 'adm_in_channels': None,
            'in_channels': 8, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0],
            'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': False, 'context_dim': 768, 'num_heads': 8,
            'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            'use_temporal_attention': False, 'use_temporal_resblock': False}

def convert_iclight_unet(state_dict):
    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    transformer_depth = unet_config["transformer_depth"][:]
    transformer_depth_output = unet_config["transformer_depth_output"][:]
    num_blocks = len(channel_mult)

    transformers_mid = unet_config.get("transformer_depth_middle", None)

    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in TEMPORAL_RESNET:
                diffusers_unet_map["down_blocks.{}.resnets.{}.{}".format(x, i, b)] = "input_blocks.{}.0.{}".format(n, b)
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["down_blocks.{}.resnets.{}.spatial_res_block.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.{}".format(n, b)
                diffusers_unet_map["down_blocks.{}.resnets.{}.temporal_res_block.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.time_stack.{}".format(n, b)
                diffusers_unet_map["down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.{}".format(n, b)
            num_transformers = transformer_depth.pop(0)
            if num_transformers > 0:
                for b in TEMPORAL_UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["down_blocks.{}.attentions.{}.{}".format(x, i, b)] = "input_blocks.{}.1.{}".format(n, b)
                for b in TEMPORAL_TRANSFORMER_MAP:
                    diffusers_unet_map["down_blocks.{}.attentions.{}.{}".format(x, i, TEMPORAL_TRANSFORMER_MAP[b])] = "input_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
                    for b in TEMPORAL_TRANSFORMER_BLOCKS:
                        diffusers_unet_map["down_blocks.{}.attentions.{}.temporal_transformer_blocks.{}.{}".format(x, i, t, b)] = "input_blocks.{}.1.time_stack.{}.{}".format(n, t, b)
            n += 1
        for k in ["weight", "bias"]:
            diffusers_unet_map["down_blocks.{}.downsamplers.0.conv.{}".format(x, k)] = "input_blocks.{}.0.op.{}".format(n, k)

    i = 0
    for b in TEMPORAL_UNET_MAP_ATTENTIONS:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, b)] = "middle_block.1.{}".format(b)
    for b in TEMPORAL_TRANSFORMER_MAP:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, TEMPORAL_TRANSFORMER_MAP[b])] = "middle_block.1.{}".format(b)
    for t in range(transformers_mid):
        for b in TRANSFORMER_BLOCKS:
            diffusers_unet_map["mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b)] = "middle_block.1.transformer_blocks.{}.{}".format(t, b)
        for b in TEMPORAL_TRANSFORMER_BLOCKS:
            diffusers_unet_map["mid_block.attentions.{}.temporal_transformer_blocks.{}.{}".format(i, t, b)] = "middle_block.1.time_stack.{}.{}".format(t, b)

    for i, n in enumerate([0, 2]):
        for b in TEMPORAL_RESNET:
            diffusers_unet_map["mid_block.resnets.{}.{}".format(i, b)] = "middle_block.{}.{}".format(n, b)
        for b in UNET_MAP_RESNET:
            diffusers_unet_map["mid_block.resnets.{}.spatial_res_block.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.{}".format(n, b)
            diffusers_unet_map["mid_block.resnets.{}.temporal_res_block.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.time_stack.{}".format(n, b)
            diffusers_unet_map["mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.{}".format(n, b)

    num_res_blocks = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks[x] + 1) * x
        l = num_res_blocks[x] + 1
        for i in range(l):
            for b in TEMPORAL_RESNET:
                diffusers_unet_map["up_blocks.{}.resnets.{}.{}".format(x, i, b)] = "output_blocks.{}.0.{}".format(n, b)
            c = 0
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.{}".format(n, b)
                diffusers_unet_map["up_blocks.{}.resnets.{}.spatial_res_block.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.{}".format(n, b)
                diffusers_unet_map["up_blocks.{}.resnets.{}.temporal_res_block.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.time_stack.{}".format(n, b)
            for b in TEMPORAL_RESNET:
                diffusers_unet_map["up_blocks.{}.resnets.{}".format(i, b)] = "output_blocks.{}.{}".format(n, b)
            c += 1
            num_transformers = transformer_depth_output.pop()
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["up_blocks.{}.attentions.{}.{}".format(x, i, b)] = "output_blocks.{}.1.{}".format(n, b)
                for b in TEMPORAL_TRANSFORMER_MAP:
                    diffusers_unet_map["up_blocks.{}.attentions.{}.{}".format(x, i, TEMPORAL_TRANSFORMER_MAP[b])] = "output_blocks.{}.1.{}".format(n, b)
                for b in TEMPORAL_UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["up_blocks.{}.attentions.{}.{}".format(x, i, b)] = "output_blocks.{}.1.{}".format(n, b)
                    
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "output_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
                    for b in TEMPORAL_TRANSFORMER_BLOCKS:
                        diffusers_unet_map["up_blocks.{}.attentions.{}.temporal_transformer_blocks.{}.{}".format(x, i, t, b)] = "output_blocks.{}.1.time_stack.{}.{}".format(n, t, b)
            if i == l - 1:
                for k in ["weight", "bias"]:
                    diffusers_unet_map["up_blocks.{}.upsamplers.0.conv.{}".format(x, k)] = "output_blocks.{}.{}.conv.{}".format(n, c, k)
            n += 1

    for k in UNET_MAP_BASIC:
        diffusers_unet_map[k[1]] = k[0]

    unet_state_dict = state_dict
    diffusers_keys = diffusers_unet_map

    new_sd = {}
    for k in diffusers_keys:
        if k in unet_state_dict:
            new_sd[diffusers_keys[k]] = unet_state_dict.pop(k)

    leftover_keys = unet_state_dict.keys()
    if len(leftover_keys) > 0:
        spatial_leftover_keys = []
        temporal_leftover_keys = []
        other_leftover_keys = []
        for key in leftover_keys:
            if "spatial" in key:
                spatial_leftover_keys.append(key)
            elif "temporal" in key:
                temporal_leftover_keys.append(key)
            else:
                other_leftover_keys.append(key)
        print("spatial_leftover_keys:")
        for key in spatial_leftover_keys:
            print(key)
        print("temporal_leftover_keys:")
        for key in temporal_leftover_keys:
            print(key)
        print("other_leftover_keys:")
        for key in other_leftover_keys:
            print(key)

    new_sd = {"diffusion_model." + k: v for k, v in new_sd.items()}
    return new_sd