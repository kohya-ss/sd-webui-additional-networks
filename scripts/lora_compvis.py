# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import copy
import math
import re
from typing import Dict, NamedTuple
import torch


class LoRAInfo(NamedTuple):
    lora_name: str
    module_name: str
    org_module: torch.nn.Module
    multiplier: float
    dim: int
    alpha: float
    first_lora_in_text_encoder: bool


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

        # Attention Couple
        self.sub_prompt_index: int = None  # None for not coupled/mask
        self.network: LoRANetworkCompvis = None
        self.is_unet = True
        self.mask_dic = None  # None for not regional
        self.mask = None
        self.is_head = False  # head LoRA calls original module (Linear/Conv2d)

        self.current_step = -1  # for Text Encoder LoRAs
        self.first_lora_in_text_encoder = False

    def apply_to(self):
        self.is_head = self.org_module.forward.__qualname__.split(".")[0] != "LoRAModule"
        # if not self.is_unet:
        #     print(
        #         self.lora_name,
        #         self.org_module.forward == self.org_forward,
        #         self.org_module.__class__.__name__,
        #         self.org_module.forward,
        #         self.org_module.forward.__name__,
        #         self.org_module.forward.__module__,
        #         self.org_module.forward.__qualname__,
        #         self.org_forward.__qualname__,
        #     )
        #     # input()

        self.org_forward = self.org_module.forward  # may be already replaced
        self.org_module.forward = self.forward
        del self.org_module  # remove including in state_dict

    def set_mask(self, network, is_unet, sub_prompt_index, mask_dic):
        # called before every generation
        self.network = network
        self.sub_prompt_index = sub_prompt_index
        self.is_unet = is_unet
        self.mask_dic = mask_dic
        self.mask = None  # reset mask
        self.current_step = -1

        # if time in U-Net, mask and couple cannot be applied
        if is_unet and "emb_layers" in self.lora_name:
            self.sub_prompt_index = None
            self.mask_dic = None

        # if not reginal, apply specific subprompt only: Text Encoder or to_k/v in U-Net
        if not is_unet or ("attn2_to_k" in self.lora_name or "attn2_to_v" in self.lora_name):
            self.mask_dic = None

    def forward(self, x, is_tail=True):
        """
        may be cascaded.
        """
        if self.network is None or not self.network.mask_enabled:  # mask/couple unsupported
            return self.org_forward(x) + self.lora_up(self.lora_down(x)) * (self.multiplier * self.scale)

        # reset or increment sub prompt index for Text Encoder
        if self.first_lora_in_text_encoder:
            # print("first LoRA in TE", self.lora_name, self.current_step, self.network.current_step)
            if self.current_step != self.network.current_step:
                self.network.text_encoder_sub_prompt_index = 0
                self.current_step = self.network.current_step
            else:
                self.network.text_encoder_sub_prompt_index += 1
            # print("first LoRA in TE", self.lora_name, self.network.text_encoder_sub_prompt_index)

        # special processing for attn2 to_out: combine to_q/k/v
        if self.is_unet and "attn2_to_out" in self.lora_name:
            return self.forward_to_out(x, is_tail)

        # special processing for to_k/v: handle context
        if self.is_unet and ("attn2_to_k" in self.lora_name or "attn2_to_v" in self.lora_name):
            return self.forward_to_k_v(x, is_tail)

        # if not regional, calculate without mask
        # print(self.sub_prompt_index, self.lora_name, self.mask_dic is None)
        if self.mask_dic is None:
            return self.forward_not_regional(x, is_tail)

        # calculate lora
        if is_tail:
            lx = None
        else:
            x, lx = x

        lx1 = self.lora_up(self.lora_down(x))
        self.update_mask(lx1)
        lx1 *= (self.multiplier * self.scale) * self.mask

        if self.is_head:
            x = self.org_forward(x)
        else:
            x, lx = self.org_forward((x, lx), False)

        if lx is None:
            lx = torch.zeros_like(x)
        lx += lx1

        if is_tail:
            x = x + lx
            # special postprocessing for to_q: repeat outputs for attention
            if self.is_unet and "attn2_to_q" in self.lora_name:
                x = self.postp_to_q(x)
            return x

        return x, lx

    def update_mask(self, lx):
        # select mask for this size
        if self.mask is None:
            if len(lx.size()) == 4:  # b,c,h,w
                area = lx.size()[2] * lx.size()[3]
            else:
                area = lx.size()[1]  # b,seq,dim

            # get mask
            # print("get mask", self.lora_name, lx.size(), area)
            mask = self.mask_dic[area]
            if len(lx.size()) == 3:
                mask = torch.reshape(mask, (1, -1, 1))
            self.mask = mask

    def forward_not_regional(self, x, is_tail):
        # if not regional, apply specific subprompt: Text Encoder or  time emb/to_k/v in U-Net

        # Text Encoder is called without batching, so get sub prompt index from network
        if not self.is_unet:
            if self.sub_prompt_index is None or self.network.text_encoder_sub_prompt_index == self.sub_prompt_index:  # matched?
                lx = self.lora_up(self.lora_down(x)) * (self.multiplier * self.scale)
            else:
                lx = None
            # print("text encoder", self.sub_prompt_index, self.network.text_encoder_sub_prompt_index, self.is_head)

            if self.is_head:
                x = self.org_forward(x)
            else:
                x = self.org_forward(x, False)

            if lx is not None:
                x += lx
            return x

        # U-Net
        if is_tail:
            lx = None  # shape cannot be calculated here
        else:
            x, lx = x

        # apply whole X or specific subprompt?
        if self.sub_prompt_index is None:
            lx1 = self.lora_up(self.lora_down(x)) * (self.multiplier * self.scale)
            if lx is None:
                lx = torch.zeros_like(lx1)
            lx += lx1
        else:
            i1 = self.sub_prompt_index
            i2 = self.network.num_sub_prompts * self.network.batch_size
            i3 = self.network.num_sub_prompts
            lx1 = self.lora_up(self.lora_down(x[i1:i2:i3]))

            if lx is None:
                lx = torch.zeros((x.size()[0], *lx1.size()[1:]), dtype=lx1.dtype, device=lx1.device)
            lx[i1:i2:i3] = lx1

        # call previous LoRA
        if self.is_head:
            x = self.org_forward(x)
        else:
            x, lx = self.org_forward((x, lx), False)

        # print("not regional x and lx", self.sub_prompt_index, x.size(), lx.size(), self.lora_name)

        if is_tail:
            # write back to x
            x += lx
            return x

        return x, lx

    def forward_to_k_v(self, x, is_tail):
        if is_tail:
            # doesn't use x
            # print("forward_to_k_v called", self.lora_name, x.size(), self.network.cond_uncond.size(), is_tail, self.is_head)
            cond_uncond = self.network.cond_uncond
            lx = None
        else:
            cond_uncond, lx = x

        # calc LoRA
        if self.sub_prompt_index is None:
            lx1 = self.lora_up(self.lora_down(cond_uncond)) * (self.multiplier * self.scale)
            if lx is None:
                lx = torch.zeros_like(lx1)
            lx += lx1
        else:
            i1 = self.sub_prompt_index
            i2 = self.network.num_sub_prompts * self.network.batch_size
            i3 = self.network.num_sub_prompts
            cond = cond_uncond[i1:i2:i3]
            # print(self.sub_prompt_index, cond_uncond.size(), cond.size())
            lx1 = self.lora_up(self.lora_down(cond)) * (self.multiplier * self.scale)
            if lx is None:
                lx = torch.zeros((cond_uncond.size()[0], *lx1.size()[1:]), dtype=lx1.dtype, device=lx1.device)
            lx[i1:i2:i3] += lx1

        # call previous LoRA
        if self.is_head:
            x = self.org_forward(cond_uncond)
        else:
            x, cond_uncond, lx = self.org_forward((cond_uncond, lx), False)

        # print("forward_to_k_v", x.size(), cond_uncond.size(), lx.size(), self.is_head, is_tail)

        if is_tail:
            # print("forward_to_k_v finished", x.size())
            return x + lx

        return x, cond_uncond, lx

    def postp_to_q(self, x):
        # if to_q, repeat x to same number of the cond and uncond (num_sub_prompts+1)
        # x: b1c, b2c, ... , b1uc, b2uc
        query = []
        for i in range(self.network.batch_size):
            for _ in range(self.network.num_sub_prompts):  # repeat cond and uncond
                query.append(x[i])
        for i in range(self.network.batch_size):
            query.append(x[self.network.batch_size + i])
        x = torch.stack(query)
        # print("to_q postp", x.size())

        # now x has subprompts and uncond, same as U-Net modules:
        # x: b1s1, b1s2, b1s3, b2s1, b2s2, b2s3, ..., b1u1, b2u1, ...

        return x

    def forward_to_out(self, x, is_tail):
        # if this LoRA is tail, create sum of LoRAs
        if is_tail:
            # print("to_out called", x.size())
            lx = None
            masks = [None] * self.network.num_sub_prompts
        else:
            x, lx, masks = x

        # apply this LoRA
        if self.sub_prompt_index is None:  # all prompt
            lx1 = self.lora_up(self.lora_down(x)) * (self.multiplier * self.scale)
            if lx is None:
                lx = torch.zeros_like(lx1)
            lx += lx1
        else:  # specific prompt
            i1 = self.sub_prompt_index
            i2 = self.network.num_sub_prompts * self.network.batch_size
            i3 = self.network.num_sub_prompts
            lx1 = x[i1:i2:i3]

            self.update_mask(lx1)
            masks[self.network.sub_prompt_index] = self.mask

            lx1 = self.lora_up(self.lora_down(lx1)) * (self.multiplier * self.scale) * self.mask
            if lx is None:
                lx = torch.zeros((x.size()[0], *lx1.size()[1:]), dtype=lx1.dtype, device=lx1.device)
            lx[i1:i2:i3] += lx1

        # call previous LoRA
        if self.is_head:
            x = self.org_forward(x)
        else:
            x, lx, masks = self.org_forward((x, lx, masks), False)

        # combine separated x with weighting
        if is_tail:
            out = []

            # how to make x...?

            # # average sub prompts
            # for i in range(0, self.network.num_sub_prompts * self.network.batch_size, self.network.num_sub_prompts):
            #     x_cond = x[i : i + self.network.num_sub_prompts]
            #     x_cond = torch.mean(x_cond, dim=0)
            #     x_cond = x_cond + torch.sum(lx[i : i + self.network.num_sub_prompts], dim=0)
            #     out.append(x_cond)

            # mask weighted sum
            mask = torch.cat(masks)
            mask_sum = torch.sum(mask, dim=0) + 1e-4
            for i in range(0, self.network.num_sub_prompts * self.network.batch_size, self.network.num_sub_prompts):
                x_cond = x[i : i + self.network.num_sub_prompts]
                lx1 = lx[i : i + self.network.num_sub_prompts]

                x_cond = x_cond * mask
                x_cond = torch.sum(x_cond, dim=0)
                x_cond = x_cond / mask_sum

                x_cond = x_cond + torch.sum(lx1, dim=0)
                out.append(x_cond)

            for i in range(self.network.batch_size):
                x_uncond = x[self.network.num_sub_prompts * self.network.batch_size + i]
                x_uncond = x_uncond + lx[self.network.num_sub_prompts * self.network.batch_size + i]
                out.append(x_uncond)

            x = torch.stack(out)
            # print("to_out finished", x.size())
            return x

        return x, lx, masks


def create_network(compatible_mode, du_state_dict, multiplier_tenc, multiplier_unet, text_encoder, unet, device, dtype, **kwargs):
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

    network = LoRANetworkCompvis(text_encoder, unet, multiplier_tenc, multiplier_unet, modules_dim, modules_alpha)
    network.application_info = (compatible_mode, du_state_dict, device, dtype)
    network.weights_mergeable = True  # always True currently
    return network


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

    def __init__(
        self,
        text_encoder,
        unet,
        multiplier_tenc=1.0,
        multiplier_unet=1.0,
        modules_dim=None,
        modules_alpha=None,
        compatible_mode=False,
    ) -> None:
        super().__init__()
        self.multiplier_unet = multiplier_unet
        self.multiplier_tenc = multiplier_tenc
        self.mask_enabled = False
        self.sub_prompt_index = None
        self.latest_mask_info = None
        self.compatible_mode = compatible_mode  # True for not replace MHA
        self.support_mask = True

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
                                lora_info = LoRAInfo(lora_name, module_name, child_module, multiplier, dim, alpha, False)
                                loras.append(lora_info)
            return loras

        self.text_encoder_loras = create_modules(
            LoRANetworkCompvis.LORA_PREFIX_TEXT_ENCODER,
            text_encoder,
            LoRANetworkCompvis.TEXT_ENCODER_TARGET_REPLACE_MODULE,
            self.multiplier_tenc,
        )
        print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        self.unet_loras = create_modules(
            LoRANetworkCompvis.LORA_PREFIX_UNET, unet, LoRANetworkCompvis.UNET_TARGET_REPLACE_MODULE, self.multiplier_unet
        )
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

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

    def apply_to_compvis(self, merge_weights, **kwargs):
        compatible_mode, du_state_dict, device, dtype = self.application_info
        del self.application_info

        # make backup of original forward/weights, if multiple modules are applied, do in 1st module only
        backed_up = False  # messaging purpose only
        if not merge_weights:
            # replace forwards (also replace weights of MHA in compatible mode)
            for lora in self.text_encoder_loras + self.unet_loras:
                rep_module = lora.org_module
                if compatible_mode and rep_module.__class__.__name__ == "MultiheadAttention":
                    if not hasattr(rep_module, "_lora_org_weights"):
                        # avoid updating of original weights. state_dict is reference to original weights
                        rep_module._lora_org_weights = copy.deepcopy(rep_module.state_dict())
                        backed_up = True
                if not hasattr(rep_module, "_lora_org_forward"):
                    rep_module._lora_org_forward = rep_module.forward
                    backed_up = True
        else:
            # replace weights
            for lora in self.text_encoder_loras + self.unet_loras:
                rep_module = lora.org_module
                if not hasattr(rep_module, "_lora_org_weights"):
                    # avoid updating of original weights. state_dict is reference to original weights
                    rep_module._lora_org_weights = copy.deepcopy(rep_module.state_dict())
                    backed_up = True

        if backed_up:
            print("original forward/weights is backed up.")

        # apply and convert state dict
        state_dict = self.apply_lora_modules(du_state_dict, device, dtype, merge_weights, compatible_mode)
        if state_dict is not None:
            info = self.load_state_dict(state_dict, strict=False)

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
        else:
            # weights are already merged
            info = "<All weights are successfully merged.>"

        self.to(device, dtype=dtype)
        return info

    def apply_lora_modules(self, du_state_dict, device, dtype, merge_weights, compatible_mode):
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

        if compatible_mode:
            print("compatible mode")
            return self.apply_lora_modules_compatible(state_dict)  # returns state_dict

        # if SD2.x and not compatible mode, replace MHA
        if self.v2 and apply_text_encoder:
            self.replace_text_encoder_MHA(state_dict, device)
        if apply_text_encoder:
            self.text_encoder_loras[0].first_lora_in_text_encoder = True  # no LoRAInfo here

        # if not merging, apply
        if not merge_weights:
            # add modules to network: this makes state_dict can be got from LoRANetwork
            for lora in self.text_encoder_loras + self.unet_loras:
                if type(lora) == LoRAModule:
                    lora.apply_to()  # ensure remove reference to original Linear: reference makes key of state_dict
                    self.add_module(lora.lora_name, lora)

            # conversion 2nd step: convert weight's shape (and handle wrapped)
            state_dict = self.convert_state_dict_shape_to_compvis(state_dict)
            return state_dict

        # merge weights
        print("merging weights.")
        for lora in self.text_encoder_loras + self.unet_loras:
            assert type(lora) == LoRAModule, f"MHA not replaced?: {lora.lora_name}"

            up_weight = state_dict.get(lora.lora_name + ".lora_up.weight", None)
            down_weight = state_dict.get(lora.lora_name + ".lora_down.weight", None)
            if up_weight is None or down_weight is None:
                print(f"missing weight: {lora.lora_name}")
                continue
            up_weight = up_weight.to(device)
            down_weight = down_weight.to(device)

            # calculate in float
            sd = lora.org_module.state_dict()
            weight = sd["weight"]
            weight = weight.to(torch.float32)
            # print(lora.lora_name, weight.size(), up_weight.size(), down_weight.size())

            if len(weight.size()) == 2:
                # linear
                if len(down_weight.size()) == 4:
                    down_weight = down_weight.squeeze(3).squeeze(2)
                    up_weight = up_weight.squeeze(3).squeeze(2)
                weight = weight + (up_weight @ down_weight) * lora.scale
            elif down_weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                weight = (
                    weight
                    + (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * lora.scale
                )
            else:
                # conv2d 3x3
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                weight = weight + conved * lora.scale

            weight = weight.to(device, dtype=dtype)

            sd["weight"] = weight
            lora.org_module.load_state_dict(sd)

        return None

    def apply_lora_modules_compatible(self, state_dict):
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
                    module = lora_info.org_module
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

                            lora_info.org_module.load_state_dict(sd)
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

    def replace_text_encoder_MHA(self, state_dict, device):
        print("replacing MultiheadAttention")
        mha_loras = {}
        lora_replacement = []

        for i, lora in enumerate(self.text_encoder_loras):
            if type(lora) == LoRAModule:
                continue

            # SD2.x MultiheadAttention: Make new MHA class
            lora_info: LoRAInfo = lora
            if lora_info.module_name not in mha_loras:
                mha_loras[lora_info.module_name] = {}

            lora_dic = mha_loras[lora_info.module_name]
            lora_dic[lora_info.lora_name] = (i, lora_info)
            if len(lora_dic) < 4:
                continue

            # 4 lora infos all together
            module: torch.nn.MultiheadAttention = lora_info.org_module
            module_name = lora_info.module_name
            w_q_dw = state_dict.get(module_name + "_q_proj.lora_down.weight")
            if w_q_dw is None:
                # corresponding weight not exists: version mismatch
                continue

            lora_idx_info_to_q = lora_dic[module_name + "_q_proj"]
            lora_idx_info_to_k = lora_dic[module_name + "_k_proj"]
            lora_idx_info_to_v = lora_dic[module_name + "_v_proj"]
            lora_idx_info_to_out = lora_dic[module_name + "_out_proj"]

            # check MHA is already replaced
            if module.forward.__qualname__.split(".")[0] == "MultiheadAttention":
                # print(f"replace MultiheadAttention for {lora_info.lora_name}")

                # corresponding LoRA module exists, create dummy MHA
                mha_attn_rep = MultiheadAttentionReplace(module.num_heads, module.embed_dim)

                # load weights and convert to q/k/v/out
                sd = module.state_dict()
                qkv_weight = sd["in_proj_weight"]
                qkv_bias = sd["in_proj_bias"]
                out_weight = sd["out_proj.weight"]
                out_bias = sd["out_proj.bias"]

                q_weight, k_weight, v_weight = torch.chunk(qkv_weight, 3)
                q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3)

                rep_sd = {
                    "to_q.weight": q_weight,
                    "to_q.bias": q_bias,
                    "to_k.weight": k_weight,
                    "to_k.bias": k_bias,
                    "to_v.weight": v_weight,
                    "to_v.bias": v_bias,
                    "to_out.weight": out_weight,
                    "to_out.bias": out_bias,
                }
                info = mha_attn_rep.load_state_dict(rep_sd)
                mha_attn_rep.to(device)
                module.forward = mha_attn_rep.forward
            else:
                # print(f"MultiheadAttention already replaced for {lora_info.lora_name}")
                mha_attn_rep = module.forward(None, None, None)  # Noneを引数に呼ぶとモジュールが返ってくるという無茶な実装

            for (index, lora_info), replaced_linear in zip(
                [lora_idx_info_to_q, lora_idx_info_to_k, lora_idx_info_to_v, lora_idx_info_to_out],
                [mha_attn_rep.to_q, mha_attn_rep.to_k, mha_attn_rep.to_v, mha_attn_rep.to_out],
            ):
                # print(f"create new LoRA {lora_info.lora_name}, {replaced_linear.__class__.__name__}")
                lora = LoRAModule(lora_info.lora_name, replaced_linear, lora_info.multiplier, lora_info.dim, lora_info.alpha)
                # lora.network = self # DO NOT set here
                lora.sub_prompt_index = self.sub_prompt_index
                lora.is_unet = False
                lora.mask_dic = None
                lora.current_step = -1

                lora_replacement.append((index, lora))

        for index, lora in lora_replacement:
            self.text_encoder_loras[index] = lora

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

    def set_mask(self, index, mask, height=None, width=None):
        if index is None:
            # clear latest mask
            self.mask_enabled = False
            self.sub_prompt_index = None
            self.latest_mask_info = None
            for lora in self.text_encoder_loras:
                if type(lora) == LoRAModule:
                    lora.set_mask(self, False, None, None)
            for lora in self.unet_loras:
                lora.set_mask(self, True, None, None)
            return

        # check mask image and h/w are same
        self.mask_enabled = True
        self.current_step = 0  # doesn't same to `step` in denoising
        self.text_encoder_sub_prompt_index = -1

        if self.sub_prompt_index is None and index is None:  # index not changed
            return
        if self.sub_prompt_index == index and (height, width) == self.latest_mask_info[1:]:
            if mask is None and self.latest_mask_info[0] is None:
                return
            if mask is not None and self.latest_mask_info[0] is not None and torch.equal(mask, self.latest_mask_info[0]):
                return

        self.sub_prompt_index = index
        self.latest_mask_info = (mask, height, width)

        org_dtype = mask.dtype
        if mask.dtype == torch.bfloat16:
            mask = mask.to(torch.float)

        # create masks
        mask_dic = {}
        mask = mask.unsqueeze(0).unsqueeze(1)  # b(1),c(1),h,w

        def resize_add(mh, mw):
            # print(mh, mw, mh * mw)
            m = torch.nn.functional.interpolate(mask, (mh, mw), mode="bilinear")  # doesn't work in bf16
            m = m.to(dtype=org_dtype)
            mask_dic[mh * mw] = m

        h = height // 8
        w = width // 8
        for _ in range(4):
            resize_add(h, w)
            if h % 2 == 1 or w % 2 == 1:  # add extra shape if h/w is not divisible by 2
                resize_add(h + h % 2, w + w % 2)
            h = (h + 1) // 2
            w = (w + 1) // 2

        for lora in self.text_encoder_loras:
            lora.set_mask(self, False, self.sub_prompt_index, mask_dic)
        for lora in self.unet_loras:
            lora.set_mask(self, True, self.sub_prompt_index, mask_dic)

    def new_step_started(self, batch_size, num_sub_prompts):
        self.batch_size: int = batch_size
        self.num_sub_prompts: int = num_sub_prompts
        self.current_step += 1
        self.text_encoder_sub_prompt_index = -1

    def set_cond_uncond(self, cond_uncond):
        self.cond_uncond = cond_uncond


class MultiheadAttentionReplace(torch.nn.Module):
    def __init__(self, heads, dim) -> None:
        super(MultiheadAttentionReplace, self).__init__()
        self.to_q = torch.nn.Linear(dim, dim)
        self.to_k = torch.nn.Linear(dim, dim)
        self.to_v = torch.nn.Linear(dim, dim)
        self.to_out = torch.nn.Linear(dim, dim)
        self.num_heads = heads
        self.embed_dim = dim
        self.head_dim = dim // heads

    def forward(self, q, k, v, need_weights=False, attn_mask=None):
        if q is None:
            return self

        # print("CAR called", self.__class__.__name__, q.size(), q.dtype, "none" if attn_mask is None else attn_mask.size())
        # in default, batch_first = False
        batch_size = q.size()[1]

        # Linear transformations for query, key, and value
        query = self.to_q(q) * (self.head_dim**-0.5)
        key = self.to_k(k)
        value = self.to_v(v)
        del q, k, v

        # Splitting heads
        query = query.view(-1, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        key = key.view(-1, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        value = value.view(-1, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # Attention scores
        key = key.transpose(-2, -1)
        scores = torch.matmul(query, key)
        del query, key
        if attn_mask is not None:
            max_neg_value = -torch.finfo(torch.float16).max  # attn_mask.dtype causes error
            scores = scores.masked_fill(attn_mask != 0, max_neg_value)  # -5e4)  # 9)

        # Attention weights
        weights = torch.softmax(scores, dim=-1)
        del scores

        # Attention output
        attn_output = torch.matmul(weights, value)
        del weights, value
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(-1, batch_size, self.embed_dim)

        # Final linear transformation
        attn_output = self.to_out(attn_output)

        return (attn_output, 0)
