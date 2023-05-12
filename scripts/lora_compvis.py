# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import copy
import math
import re
from typing import NamedTuple
import torch


class LoRAInfo(NamedTuple):
    lora_name: str
    module_name: str
    module: torch.nn.Module
    multiplier: float
    dim: int
    alpha: float


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(self, lora_name, org_module: torch.nn.Module, multiplier=1.0, lora_dim=4, alpha=1):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            # self.lora_dim = min(self.lora_dim, in_dim, out_dim)
            # if self.lora_dim != lora_dim:
            #   print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_forward = org_module.forward
        self.org_module = org_module  # remove in applying
        self.mask_dic = None
        self.mask = None
        self.mask_area = -1

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def set_mask_dic(self, mask_dic):
        # called before every generation

        # check this module is related to h,w (not context and time emb)
        if "attn2_to_k" in self.lora_name or "attn2_to_v" in self.lora_name or "emb_layers" in self.lora_name:
            # print(f"LoRA for context or time emb: {self.lora_name}")
            self.mask_dic = None
        else:
            self.mask_dic = mask_dic

        self.mask = None

    def forward(self, x):
        """
        may be cascaded.
        """
        if self.mask_dic is None:
            return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale

        # regional LoRA

        # calculate lora and get size
        lx = self.lora_up(self.lora_down(x))

        if len(lx.size()) == 4:  # b,c,h,w
            area = lx.size()[2] * lx.size()[3]
        else:
            area = lx.size()[1]  # b,seq,dim

        if self.mask is None or self.mask_area != area:
            # get mask
            # print(self.lora_name, x.size(), lx.size(), area)
            mask = self.mask_dic[area]
            if len(lx.size()) == 3:
                mask = torch.reshape(mask, (1, -1, 1))
            self.mask = mask
            self.mask_area = area

        return self.org_forward(x) + lx * self.multiplier * self.scale * self.mask


def create_network_and_apply_compvis(du_state_dict, multiplier_tenc, multiplier_unet, text_encoder, unet, **kwargs):
    # get device and dtype from unet
    for module in unet.modules():
        if module.__class__.__name__ == "Linear":
            param: torch.nn.Parameter = module.weight
            # device = param.device
            dtype = param.dtype
            break

    # get dims (rank) and alpha from state dict
    modules_dim = {}
    modules_alpha = {}
    for key, value in du_state_dict.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = float(value.detach().to(torch.float).cpu().numpy())
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim

    # support old LoRA without alpha
    for key in modules_dim.keys():
        if key not in modules_alpha:
            modules_alpha[key] = modules_dim[key]

    print(
        f"dimension: {set(modules_dim.values())}, alpha: {set(modules_alpha.values())}, multiplier_unet: {multiplier_unet}, multiplier_tenc: {multiplier_tenc}"
    )

    # if network_dim is None:
    #   print(f"The selected model is not LoRA or not trained by `sd-scripts`?")
    #   network_dim = 4
    #   network_alpha = 1

    # create, apply and load weights
    network = LoRANetworkCompvis(text_encoder, unet, multiplier_tenc, multiplier_unet, modules_dim, modules_alpha)
    state_dict = network.apply_lora_modules(du_state_dict)  # some weights are applied to text encoder
    network.to(dtype)  # with this, if error comes from next line, the model will be used
    info = network.load_state_dict(state_dict, strict=False)

    # remove redundant warnings
    if len(info.missing_keys) > 4:
        missing_keys = []
        alpha_count = 0
        for key in info.missing_keys:
            if "alpha" not in key:
                missing_keys.append(key)
            else:
                if alpha_count == 0:
                    missing_keys.append(key)
                alpha_count += 1
        if alpha_count > 1:
            missing_keys.append(
                f"... and {alpha_count-1} alphas. The model doesn't have alpha, use dim (rannk) as alpha. You can ignore this message."
            )

        info = torch.nn.modules.module._IncompatibleKeys(missing_keys, info.unexpected_keys)

    return network, info


class LoRANetworkCompvis(torch.nn.Module):
    # UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
    # TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    UNET_TARGET_REPLACE_MODULE = ["SpatialTransformer", "ResBlock", "Downsample", "Upsample"]  # , "Attention"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["ResidualAttentionBlock", "CLIPAttention", "CLIPMLP"]

    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    @classmethod
    def convert_diffusers_name_to_compvis(cls, v2, du_name):
        """
        convert diffusers's LoRA name to CompVis
        """
        cv_name = None
        if "lora_unet_" in du_name:
            m = re.search(r"_down_blocks_(\d+)_attentions_(\d+)_(.+)", du_name)
            if m:
                du_block_index = int(m.group(1))
                du_attn_index = int(m.group(2))
                du_suffix = m.group(3)

                cv_index = 1 + du_block_index * 3 + du_attn_index  # 1,2, 4,5, 7,8
                cv_name = f"lora_unet_input_blocks_{cv_index}_1_{du_suffix}"
                return cv_name

            m = re.search(r"_mid_block_attentions_(\d+)_(.+)", du_name)
            if m:
                du_suffix = m.group(2)
                cv_name = f"lora_unet_middle_block_1_{du_suffix}"
                return cv_name

            m = re.search(r"_up_blocks_(\d+)_attentions_(\d+)_(.+)", du_name)
            if m:
                du_block_index = int(m.group(1))
                du_attn_index = int(m.group(2))
                du_suffix = m.group(3)

                cv_index = du_block_index * 3 + du_attn_index  # 3,4,5, 6,7,8, 9,10,11
                cv_name = f"lora_unet_output_blocks_{cv_index}_1_{du_suffix}"
                return cv_name

            m = re.search(r"_down_blocks_(\d+)_resnets_(\d+)_(.+)", du_name)
            if m:
                du_block_index = int(m.group(1))
                du_res_index = int(m.group(2))
                du_suffix = m.group(3)
                cv_suffix = {
                    "conv1": "in_layers_2",
                    "conv2": "out_layers_3",
                    "time_emb_proj": "emb_layers_1",
                    "conv_shortcut": "skip_connection",
                }[du_suffix]

                cv_index = 1 + du_block_index * 3 + du_res_index  # 1,2, 4,5, 7,8
                cv_name = f"lora_unet_input_blocks_{cv_index}_0_{cv_suffix}"
                return cv_name

            m = re.search(r"_down_blocks_(\d+)_downsamplers_0_conv", du_name)
            if m:
                block_index = int(m.group(1))
                cv_index = 3 + block_index * 3
                cv_name = f"lora_unet_input_blocks_{cv_index}_0_op"
                return cv_name

            m = re.search(r"_mid_block_resnets_(\d+)_(.+)", du_name)
            if m:
                index = int(m.group(1))
                du_suffix = m.group(2)
                cv_suffix = {
                    "conv1": "in_layers_2",
                    "conv2": "out_layers_3",
                    "time_emb_proj": "emb_layers_1",
                    "conv_shortcut": "skip_connection",
                }[du_suffix]
                cv_name = f"lora_unet_middle_block_{index*2}_{cv_suffix}"
                return cv_name

            m = re.search(r"_up_blocks_(\d+)_resnets_(\d+)_(.+)", du_name)
            if m:
                du_block_index = int(m.group(1))
                du_res_index = int(m.group(2))
                du_suffix = m.group(3)
                cv_suffix = {
                    "conv1": "in_layers_2",
                    "conv2": "out_layers_3",
                    "time_emb_proj": "emb_layers_1",
                    "conv_shortcut": "skip_connection",
                }[du_suffix]

                cv_index = du_block_index * 3 + du_res_index  # 1,2, 4,5, 7,8
                cv_name = f"lora_unet_output_blocks_{cv_index}_0_{cv_suffix}"
                return cv_name

            m = re.search(r"_up_blocks_(\d+)_upsamplers_0_conv", du_name)
            if m:
                block_index = int(m.group(1))
                cv_index = block_index * 3 + 2
                cv_name = f"lora_unet_output_blocks_{cv_index}_{bool(block_index)+1}_conv"
                return cv_name

        elif "lora_te_" in du_name:
            m = re.search(r"_model_encoder_layers_(\d+)_(.+)", du_name)
            if m:
                du_block_index = int(m.group(1))
                du_suffix = m.group(2)

                cv_index = du_block_index
                if v2:
                    if "mlp_fc1" in du_suffix:
                        cv_name = (
                            f"lora_te_wrapped_model_transformer_resblocks_{cv_index}_{du_suffix.replace('mlp_fc1', 'mlp_c_fc')}"
                        )
                    elif "mlp_fc2" in du_suffix:
                        cv_name = (
                            f"lora_te_wrapped_model_transformer_resblocks_{cv_index}_{du_suffix.replace('mlp_fc2', 'mlp_c_proj')}"
                        )
                    elif "self_attn":
                        # handled later
                        cv_name = f"lora_te_wrapped_model_transformer_resblocks_{cv_index}_{du_suffix.replace('self_attn', 'attn')}"
                else:
                    cv_name = f"lora_te_wrapped_transformer_text_model_encoder_layers_{cv_index}_{du_suffix}"

        assert cv_name is not None, f"conversion failed: {du_name}. the model may not be trained by `sd-scripts`."
        return cv_name

    @classmethod
    def convert_state_dict_name_to_compvis(cls, v2, state_dict):
        """
        convert keys in state dict to load it by load_state_dict
        """
        new_sd = {}
        for key, value in state_dict.items():
            tokens = key.split(".")
            compvis_name = LoRANetworkCompvis.convert_diffusers_name_to_compvis(v2, tokens[0])
            new_key = compvis_name + "." + ".".join(tokens[1:])

            new_sd[new_key] = value

        return new_sd

    def __init__(self, text_encoder, unet, multiplier_tenc=1.0, multiplier_unet=1.0, modules_dim=None, modules_alpha=None) -> None:
        super().__init__()
        self.multiplier_unet = multiplier_unet
        self.multiplier_tenc = multiplier_tenc
        self.latest_mask_info = None

        # check v1 or v2
        self.v2 = False
        for _, module in text_encoder.named_modules():
            for _, child_module in module.named_modules():
                if child_module.__class__.__name__ == "MultiheadAttention":
                    self.v2 = True
                    break
            if self.v2:
                break

        # convert lora name to CompVis and get dim and alpha
        comp_vis_loras_dim_alpha = {}
        for du_lora_name in modules_dim.keys():
            dim = modules_dim[du_lora_name]
            alpha = modules_alpha[du_lora_name]
            comp_vis_lora_name = LoRANetworkCompvis.convert_diffusers_name_to_compvis(self.v2, du_lora_name)
            comp_vis_loras_dim_alpha[comp_vis_lora_name] = (dim, alpha)

        # create module instances
        def create_modules(prefix, root_module: torch.nn.Module, target_replace_modules, multiplier):
            loras = []
            replaced_modules = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        # enumerate all Linear and Conv2d
                        if child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "Conv2d":
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")
                            if "_resblocks_23_" in lora_name:  # ignore last block in StabilityAi Text Encoder
                                break
                            if lora_name not in comp_vis_loras_dim_alpha:
                                continue

                            dim, alpha = comp_vis_loras_dim_alpha[lora_name]
                            lora = LoRAModule(lora_name, child_module, multiplier, dim, alpha)
                            loras.append(lora)

                            replaced_modules.append(child_module)
                        elif child_module.__class__.__name__ == "MultiheadAttention":
                            # make four modules: not replacing forward method but merge weights later
                            for suffix in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                                module_name = prefix + "." + name + "." + child_name  # ~.attn
                                module_name = module_name.replace(".", "_")
                                if "_resblocks_23_" in module_name:  # ignore last block in StabilityAi Text Encoder
                                    break

                                lora_name = module_name + "_" + suffix
                                if lora_name not in comp_vis_loras_dim_alpha:
                                    continue
                                dim, alpha = comp_vis_loras_dim_alpha[lora_name]
                                lora_info = LoRAInfo(lora_name, module_name, child_module, multiplier, dim, alpha)
                                loras.append(lora_info)

                                replaced_modules.append(child_module)
            return loras, replaced_modules

        self.text_encoder_loras, te_rep_modules = create_modules(
            LoRANetworkCompvis.LORA_PREFIX_TEXT_ENCODER,
            text_encoder,
            LoRANetworkCompvis.TEXT_ENCODER_TARGET_REPLACE_MODULE,
            self.multiplier_tenc,
        )
        print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        self.unet_loras, unet_rep_modules = create_modules(
            LoRANetworkCompvis.LORA_PREFIX_UNET, unet, LoRANetworkCompvis.UNET_TARGET_REPLACE_MODULE, self.multiplier_unet
        )
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        # make backup of original forward/weights, if multiple modules are applied, do in 1st module only
        backed_up = False  # messaging purpose only
        for rep_module in te_rep_modules + unet_rep_modules:
            if (
                rep_module.__class__.__name__ == "MultiheadAttention"
            ):  # multiple MHA modules are in list, prevent to backed up forward
                if not hasattr(rep_module, "_lora_org_weights"):
                    # avoid updating of original weights. state_dict is reference to original weights
                    rep_module._lora_org_weights = copy.deepcopy(rep_module.state_dict())
                    backed_up = True
            elif not hasattr(rep_module, "_lora_org_forward"):
                rep_module._lora_org_forward = rep_module.forward
                backed_up = True
        if backed_up:
            print("original forward/weights is backed up.")

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def restore(self, text_encoder, unet):
        # restore forward/weights from property for all modules
        restored = False  # messaging purpose only
        modules = []
        modules.extend(text_encoder.modules())
        modules.extend(unet.modules())
        for module in modules:
            if hasattr(module, "_lora_org_forward"):
                module.forward = module._lora_org_forward
                del module._lora_org_forward
                restored = True
            if hasattr(
                module, "_lora_org_weights"
            ):  # module doesn't have forward and weights at same time currently, but supports it for future changing
                module.load_state_dict(module._lora_org_weights)
                del module._lora_org_weights
                restored = True

        if restored:
            print("original forward/weights is restored.")

    def apply_lora_modules(self, du_state_dict):
        # conversion 1st step: convert names in state_dict
        state_dict = LoRANetworkCompvis.convert_state_dict_name_to_compvis(self.v2, du_state_dict)

        # check state_dict has text_encoder or unet
        weights_has_text_encoder = weights_has_unet = False
        for key in state_dict.keys():
            if key.startswith(LoRANetworkCompvis.LORA_PREFIX_TEXT_ENCODER):
                weights_has_text_encoder = True
            elif key.startswith(LoRANetworkCompvis.LORA_PREFIX_UNET):
                weights_has_unet = True
            if weights_has_text_encoder and weights_has_unet:
                break

        apply_text_encoder = weights_has_text_encoder
        apply_unet = weights_has_unet

        if apply_text_encoder:
            print("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        # add modules to network: this makes state_dict can be got from LoRANetwork
        mha_loras = {}
        for lora in self.text_encoder_loras + self.unet_loras:
            if type(lora) == LoRAModule:
                lora.apply_to()  # ensure remove reference to original Linear: reference makes key of state_dict
                self.add_module(lora.lora_name, lora)
            else:
                # SD2.x MultiheadAttention merge weights to MHA weights
                lora_info: LoRAInfo = lora
                if lora_info.module_name not in mha_loras:
                    mha_loras[lora_info.module_name] = {}

                lora_dic = mha_loras[lora_info.module_name]
                lora_dic[lora_info.lora_name] = lora_info
                if len(lora_dic) == 4:
                    # calculate and apply
                    module = lora_info.module
                    module_name = lora_info.module_name
                    w_q_dw = state_dict.get(module_name + "_q_proj.lora_down.weight")
                    if w_q_dw is not None:  # corresponding LoRA module exists
                        w_q_up = state_dict[module_name + "_q_proj.lora_up.weight"]
                        w_k_dw = state_dict[module_name + "_k_proj.lora_down.weight"]
                        w_k_up = state_dict[module_name + "_k_proj.lora_up.weight"]
                        w_v_dw = state_dict[module_name + "_v_proj.lora_down.weight"]
                        w_v_up = state_dict[module_name + "_v_proj.lora_up.weight"]
                        w_out_dw = state_dict[module_name + "_out_proj.lora_down.weight"]
                        w_out_up = state_dict[module_name + "_out_proj.lora_up.weight"]
                        q_lora_info = lora_dic[module_name + "_q_proj"]
                        k_lora_info = lora_dic[module_name + "_k_proj"]
                        v_lora_info = lora_dic[module_name + "_v_proj"]
                        out_lora_info = lora_dic[module_name + "_out_proj"]

                        sd = module.state_dict()
                        qkv_weight = sd["in_proj_weight"]
                        out_weight = sd["out_proj.weight"]
                        dev = qkv_weight.device

                        def merge_weights(l_info, weight, up_weight, down_weight):
                            # calculate in float
                            scale = l_info.alpha / l_info.dim
                            dtype = weight.dtype
                            weight = (
                                weight.float()
                                + l_info.multiplier
                                * (up_weight.to(dev, dtype=torch.float) @ down_weight.to(dev, dtype=torch.float))
                                * scale
                            )
                            weight = weight.to(dtype)
                            return weight

                        q_weight, k_weight, v_weight = torch.chunk(qkv_weight, 3)
                        if q_weight.size()[1] == w_q_up.size()[0]:
                            q_weight = merge_weights(q_lora_info, q_weight, w_q_up, w_q_dw)
                            k_weight = merge_weights(k_lora_info, k_weight, w_k_up, w_k_dw)
                            v_weight = merge_weights(v_lora_info, v_weight, w_v_up, w_v_dw)
                            qkv_weight = torch.cat([q_weight, k_weight, v_weight])

                            out_weight = merge_weights(out_lora_info, out_weight, w_out_up, w_out_dw)

                            sd["in_proj_weight"] = qkv_weight.to(dev)
                            sd["out_proj.weight"] = out_weight.to(dev)

                            lora_info.module.load_state_dict(sd)
                        else:
                            # different dim, version mismatch
                            print(f"shape of weight is different: {module_name}. SD version may be different")

                        for t in ["q", "k", "v", "out"]:
                            del state_dict[f"{module_name}_{t}_proj.lora_down.weight"]
                            del state_dict[f"{module_name}_{t}_proj.lora_up.weight"]
                            alpha_key = f"{module_name}_{t}_proj.alpha"
                            if alpha_key in state_dict:
                                del state_dict[alpha_key]
                    else:
                        # corresponding weight not exists: version mismatch
                        pass

        # conversion 2nd step: convert weight's shape (and handle wrapped)
        state_dict = self.convert_state_dict_shape_to_compvis(state_dict)

        return state_dict

    def convert_state_dict_shape_to_compvis(self, state_dict):
        # shape conversion
        current_sd = self.state_dict()  # to get target shape
        wrapped = False
        count = 0
        for key in list(state_dict.keys()):
            if key not in current_sd:
                continue  # might be error or another version
            if "wrapped" in key:
                wrapped = True

            value: torch.Tensor = state_dict[key]
            if value.size() != current_sd[key].size():
                # print(f"convert weights shape: {key}, from: {value.size()}, {len(value.size())}")
                count += 1
                if len(value.size()) == 4:
                    value = value.squeeze(3).squeeze(2)
                else:
                    value = value.unsqueeze(2).unsqueeze(3)
                state_dict[key] = value
            if tuple(value.size()) != tuple(current_sd[key].size()):
                print(
                    f"weight's shape is different: {key} expected {current_sd[key].size()} found {value.size()}. SD version may be different"
                )
                del state_dict[key]
        print(f"shapes for {count} weights are converted.")

        # convert wrapped
        if not wrapped:
            print("remove 'wrapped' from keys")
            for key in list(state_dict.keys()):
                if "_wrapped_" in key:
                    new_key = key.replace("_wrapped_", "_")
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

        return state_dict

    def set_mask(self, mask, height=None, width=None, hr_height=None, hr_width=None):
        if mask is None:
            # clear latest mask
            # print("clear mask")
            self.latest_mask_info = None
            for lora in self.unet_loras:
                lora.set_mask_dic(None)
            return

        # check mask image and h/w are same
        if (
            self.latest_mask_info is not None
            and torch.equal(mask, self.latest_mask_info[0])
            and (height, width, hr_height, hr_width) == self.latest_mask_info[1:]
        ):
            # print("mask not changed")
            return

        self.latest_mask_info = (mask, height, width, hr_height, hr_width)

        org_dtype = mask.dtype
        if mask.dtype == torch.bfloat16:
            mask = mask.to(torch.float)

        mask_dic = {}
        mask = mask.unsqueeze(0).unsqueeze(1)  # b(1),c(1),h,w

        def resize_add(mh, mw):
            # print(mh, mw, mh * mw)
            m = torch.nn.functional.interpolate(mask, (mh, mw), mode="bilinear")  # doesn't work in bf16
            m = m.to(org_dtype)
            mask_dic[mh * mw] = m

        for h, w in [(height, width), (hr_height, hr_width)]:
            if not h or not w:
                continue

            h = h // 8
            w = w // 8
            for i in range(4):
                resize_add(h, w)
                if h % 2 == 1 or w % 2 == 1:  # add extra shape if h/w is not divisible by 2
                    resize_add(h + h % 2, w + w % 2)
                h = (h + 1) // 2
                w = (w + 1) // 2

        for lora in self.unet_loras:
            lora.set_mask_dic(mask_dic)
        return
