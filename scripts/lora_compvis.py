# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import copy
import math
import re
from typing import Dict, List, NamedTuple
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
    # this module doesn't call original module's forward
    def __init__(self, lora_name, org_module, multiplier=1.0, lora_dim=4, alpha=1, is_mha=False, module_name=None):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.is_mha = is_mha
        self.module_name = module_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

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

    def set_weights(self, up_weight, down_weight):
        if self.lora_up.weight.size() != up_weight.size():
            if len(up_weight.size()) == 4:
                up_weight = up_weight.squeeze(3).squeeze(2)
            else:
                up_weight = up_weight.unsqueeze(2).unsqueeze(3)
        self.lora_up.weight.data = up_weight

        if self.lora_down.weight.size() != down_weight.size():
            if len(down_weight.size()) == 4:
                down_weight = down_weight.squeeze(3).squeeze(2)
            else:
                down_weight = down_weight.unsqueeze(2).unsqueeze(3)
        self.lora_down.weight.data = down_weight

    def forward(self, x):
        return self.lora_up(self.lora_down(x)) * (self.multiplier * self.scale)


class LoRAChainContainer:
    """
    Container for LoRAChain. Holds global information such as mask, batch size, current step, etc.
    """

    def __init__(self, will_be_merged=False):
        self.loras_chains: Dict[str, LoRAChain] = {}
        self.mask_dic: List[Dict[int, torch.Tensor]] = None
        self.mask_enabled = False
        self.will_be_merged = will_be_merged

    def add_chain(self, first_in_text_encoder, lora_name: str, is_text_encoder: bool, module: torch.nn.Module):
        lora_chain = LoRAChain(self, first_in_text_encoder, lora_name, module, is_text_encoder, self.will_be_merged)
        self.loras_chains[lora_name] = lora_chain

    def get_module(self, lora_name: str):
        return self.loras_chains[lora_name].module

    def get_lora_chain(self, lora_name: str):
        return self.loras_chains[lora_name]

    def prepare_generation(self, num_sub_prompts: int):
        self.batch_size = 0  # not set yet
        self.num_sub_prompts = num_sub_prompts  # split by " AND "
        self.text_encoder_sub_prompt_index = -1  # increment in first lora in text encoder

    def new_step_started(self, batch_size, num_sub_prompts):
        self.batch_size = batch_size
        self.num_sub_prompts = num_sub_prompts
        self.text_encoder_sub_prompt_index = -1

    def set_cond_uncond(self, cond_uncond):
        self.cond_uncond = cond_uncond

    def clear_mask(self):
        self.mask_dic = None
        self.mask_enabled = False

    def set_mask(self, index: int, mask: torch.Tensor, height: int, width: int):
        self.mask_enabled = True

        if self.mask_dic is None:
            self.mask_dic = []
        while len(self.mask_dic) <= index:
            self.mask_dic.append(None)
        self.mask_dic[index] = {}

        curr_mask_dic = self.mask_dic[index]

        org_dtype = mask.dtype
        if mask.dtype == torch.bfloat16:
            mask = mask.to(torch.float)

        # create masks
        mask = mask.unsqueeze(0).unsqueeze(1)  # b(1),c(1),h,w

        def resize_add(mh, mw):
            # print(mh, mw, mh * mw)
            m = torch.nn.functional.interpolate(mask, (mh, mw), mode="bilinear")  # doesn't work in bf16
            m = m.to(dtype=org_dtype)
            curr_mask_dic[(mh, mw)] = m
            curr_mask_dic[mh * mw] = torch.reshape(m, (1, -1, 1))

        h = height // 8
        w = width // 8
        for _ in range(4):
            resize_add(h, w)
            if h % 2 == 1 or w % 2 == 1:  # add extra shape if h/w is not divisible by 2
                resize_add(h + h % 2, w + w % 2)
            h = (h + 1) // 2
            w = (w + 1) // 2

    def get_mask(self, index, size):
        if self.mask_dic is None or self.mask_dic[index] is None:
            return None
        if len(size) == 4:
            size = size[2:4]
        else:
            size = size[1]
        return self.mask_dic[index].get(size, None)


class LoRAChain:
    """
    LoRAChain is a chain of LoRAModules for the single Linear/Conv2d module.
    Call each LoRAModule in the chain in order with applying mask (regional) or for each sub-prompt (text encoder or not regional).
    """

    def __init__(
        self,
        container: LoRAChainContainer,
        first_in_text_encoder: bool,
        lora_name: str,
        module: torch.nn.Module,
        is_text_encoder,
        will_be_merged=False,
    ):
        self.container = container
        self.first_lora_in_text_encoder = first_in_text_encoder  # increment sub prompt index if true
        self.lora_name = lora_name
        self.module = module
        self.is_text_encoder = is_text_encoder
        self.loras: List[LoRAModule] = []

        if not will_be_merged:
            self.org_forward = module.forward
            module.forward = self.forward

        # check regional or not by lora_name
        if self.is_text_encoder:
            self.regional = False
            self.use_sub_prompt = True
        elif "attn2_to_k" in lora_name or "attn2_to_v" in lora_name:
            self.regional = False
            self.use_sub_prompt = True
        elif "time_emb" in lora_name:
            self.regional = False
            self.use_sub_prompt = False
        else:
            self.regional = True
            self.use_sub_prompt = False

    def add_lora(self, index, lora: LoRAModule):
        while len(self.loras) <= index:
            self.loras.append(None)
        self.loras[index] = lora

    def default_forward_with_lora(self, x):
        # call loras without mask
        ox = self.org_forward(x)
        for lora in self.loras:
            if lora is not None:
                ox += lora(x)
        return ox

    def forward(self, x):
        if not self.container.mask_enabled:
            return self.default_forward_with_lora(x)

        # increment sub_prompt_index when text encoder is called
        if self.first_lora_in_text_encoder:
            # print(
            #     "first lora in text encoder", self.container.text_encoder_sub_prompt_index, self.container.num_sub_prompts, x.size()
            # )
            self.container.text_encoder_sub_prompt_index += 1

        if not self.regional and not self.use_sub_prompt:
            return self.default_forward_with_lora(x)
        if self.regional:
            return self.regional_forward(x)
        if self.is_text_encoder:
            return self.text_encoder_forward(x)
        return self.sub_prompt_forward(x)

    def text_encoder_forward(self, x):
        # x is for single lora

        # if no subprompt, apply LoRA to all (cond and uncond)
        if self.container.num_sub_prompts <= 1:
            return self.default_forward_with_lora(x)

        # if uncond in text_encoder, do not apply LoRA
        if x.size()[0] == self.container.batch_size:
            return self.org_forward(x)

        # if no LoRA for this subprompt, do not apply LoRA
        if self.container.text_encoder_sub_prompt_index >= len(self.loras):
            return self.org_forward(x)
        if self.container.text_encoder_sub_prompt_index < 0:
            # in case of no LoRA for text encoder, sub_prompt_index is not incremented
            return self.org_forward(x)

        lora = self.loras[self.container.text_encoder_sub_prompt_index]
        if lora is None:
            return self.org_forward(x)

        return self.org_forward(x) + lora(x)

    def sub_prompt_forward(self, x):
        assert not self.is_text_encoder and (
            "attn2_to_k" in self.lora_name or "attn2_to_v" in self.lora_name,
            "sub_prompt_forward is called for attn2_to_k/v",
        )

        # doesn't use x
        cond_uncond = self.container.cond_uncond
        # print(
        #     "attn2_to_k_v_forward called",
        #     self.lora_name,
        #     cond_uncond.size(),
        #     self.container.num_sub_prompts,
        #     self.container.batch_size,
        # )

        ox = self.org_forward(cond_uncond)

        lx = torch.zeros_like(ox)
        for i, lora in enumerate(self.loras):
            if lora is not None:
                emb_idx = i
                cond = cond_uncond[emb_idx :: self.container.num_sub_prompts]
                lx1 = lora(cond)
                lx[emb_idx :: self.container.num_sub_prompts] = lx1

        return ox + lx

    def regional_forward(self, x):
        if "attn2_to_out" in self.lora_name:
            return self.to_out_forward(x)

        ox = self.org_forward(x)

        for i, lora in enumerate(self.loras):
            if lora is not None:
                lx = lora(x)

                mask = self.container.get_mask(i, lx.size())
                if mask is not None:
                    lx = lx * mask

                ox = ox + lx

        x = ox
        if "attn2_to_q" in self.lora_name:
            x = self.postp_to_q(x)
        return x

    def postp_to_q(self, x):
        # if to_q, repeat x to same number of the cond and uncond (num_sub_prompts+1)
        # x: b1c, b2c, ... , b1uc, b2uc
        query = []
        for i in range(self.container.batch_size):
            for _ in range(self.container.num_sub_prompts):
                query.append(x[i])

        for i in range(self.container.batch_size):
            query.append(x[self.container.batch_size + i])

        x = torch.stack(query)
        # print("to_q postp", x.size())

        # now x has subprompts and uncond, same as U-Net modules:
        # x: b1s1, b1s2, b1s3, b2s1, b2s2, b2s3, ..., b1u1, b2u1, ...

        return x

    def to_out_forward(self, x):
        ox = self.org_forward(x)
        lx = torch.zeros_like(ox)

        # apply this LoRA
        masks = []
        for i, lora in enumerate(self.loras):
            mask = None
            if lora is not None:
                if i < self.container.num_sub_prompts:
                    # apply to  single subprompt
                    emb_idx = i
                    lx1 = lora(x[emb_idx :: self.container.num_sub_prompts])

                    mask = self.container.get_mask(i, lx1.size())
                    if mask is not None:
                        lx1 = lx1 * mask
                    else:
                        mask = torch.ones((1, *lx1.size()[1:-1], 1), dtype=lx1.dtype, device=lx1.device)
                    masks.append(mask)

                    lx[emb_idx :: self.container.num_sub_prompts] = lx1
                else:
                    # apply to all subprompts
                    lx1 = lora(x)
                    lx = lx + lx1
        x = ox

        # mask weighted sum
        mask = torch.cat(masks, dim=0)  # (num_sub_prompts, ...)

        # print("to_out", mask.size(), x.size(), lx.size())

        # if num of masks is more than num_sub_prompts, modify mask
        # for example, if num of masks is 3, and num_sub_prompts is 2, then third mask is applied to all subprompts.
        # it means that third mask is added to first and second mask.
        # because 1st and 2nd mask will be applied to each subprompt.
        num_extra_masks = len(masks) - self.container.num_sub_prompts
        if num_extra_masks > 0:
            # if there are 5 masks, and num_sub_prompts is 3, then weight1 is 1/5.
            weight1 = 1 / len(masks)

            mask[: self.container.num_sub_prompts] *= weight1 * self.container.num_sub_prompts  # 1/5 * 3 = 3/5

            for i in range(num_extra_masks):
                mask[: self.container.num_sub_prompts] += mask[self.container.num_sub_prompts + i] * weight1  # 1/5

            # total weight is 1.0. however I'm not sure if it is correct (;'∀')

            mask = mask[: self.container.num_sub_prompts]

        # if num of masks is less than num_sub_prompts, it means num of loras is less than num_sub_prompts
        # in this case, extra subprompts are not applied to any lora.
        # however, x_cond should be calculated from all subprompts, so we need to add extra masks.
        if len(masks) < self.container.num_sub_prompts:
            mask = torch.cat(
                [
                    mask,
                    torch.ones(
                        (self.container.num_sub_prompts - len(masks), *mask.size()[1:]), dtype=mask.dtype, device=mask.device
                    ),
                ],
                dim=0,
            )

        mask_sum = torch.sum(mask, dim=0) + 1e-4

        out = []
        for i in range(0, self.container.num_sub_prompts * self.container.batch_size, self.container.num_sub_prompts):
            emb_idx = i
            num_loras = len(self.loras)  # == num of masks, may be less than num_sub_prompts

            x_cond = x[emb_idx : emb_idx + self.container.num_sub_prompts]
            lx1 = lx[emb_idx : emb_idx + num_loras]

            # print("x_cond", x_cond.size(), "lx1", lx1.size(), "mask", mask.size())
            x_cond = x_cond * mask / mask_sum
            x_cond = torch.sum(x_cond, dim=0)

            x_cond = x_cond + torch.sum(lx1, dim=0)
            out.append(x_cond)

        for i in range(self.container.batch_size):
            x_uncond = x[self.container.num_sub_prompts * self.container.batch_size + i]
            # x_uncond = x_uncond + lx[self.container.num_sub_prompts * self.container.batch_size + i]
            out.append(x_uncond)

        x = torch.stack(out)
        # print("to_out finished", x.size())
        return x


def create_network(du_state_dict, multiplier_tenc, multiplier_unet, text_encoder, unet, device, dtype, **kwargs):
    network = LoRANetworkCompvis(text_encoder, unet, multiplier_tenc, multiplier_unet)
    network.application_info = (du_state_dict, device, dtype)
    network.weights_mergeable = True  # always True if no mask currently
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
                        cv_name = f"lora_te_model_transformer_resblocks_{cv_index}_{du_suffix.replace('mlp_fc1', 'mlp_c_fc')}"
                    elif "mlp_fc2" in du_suffix:
                        cv_name = f"lora_te_model_transformer_resblocks_{cv_index}_{du_suffix.replace('mlp_fc2', 'mlp_c_proj')}"
                    elif "self_attn":
                        # handled later
                        cv_name = f"lora_te_model_transformer_resblocks_{cv_index}_{du_suffix.replace('self_attn', 'attn')}"
                else:
                    cv_name = f"lora_te_transformer_text_model_encoder_layers_{cv_index}_{du_suffix}"

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
    ) -> None:
        super().__init__()
        self.multiplier_unet = multiplier_unet
        self.multiplier_tenc = multiplier_tenc
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
            if hasattr(module, "_lora_org_weights"):
                module.load_state_dict(module._lora_org_weights)
                del module._lora_org_weights
                restored = True

        if restored:
            print("original forward/weights is restored.")

    def apply_to_compvis(self, index, text_encoder, unet, merge_weights, shared, **kwargs):
        du_state_dict, device, dtype = self.application_info
        del self.application_info  # state_dict is large, so delete it

        state_dict = LoRANetworkCompvis.convert_state_dict_name_to_compvis(self.v2, du_state_dict)

        # get dims (rank) and alpha from state dict
        modules_dim = {}
        modules_alpha = {}
        for key, value in state_dict.items():
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
            f"dimension: {set(modules_dim.values())}, alpha: {set(modules_alpha.values())}, multiplier_unet: {self.multiplier_unet}, multiplier_tenc: {self.multiplier_tenc}"
        )

        # make backup of original forward/weights, if multiple modules are applied, do in 1st module only
        if "lora_prepared" not in shared:
            shared["lora_prepared"] = True
            lora_chains = self.prepare_applying(text_encoder, unet, merge_weights, device)
            shared["lora_chains"] = lora_chains
        lora_chains: LoRAChainContainer = shared["lora_chains"]

        if merge_weights:
            self.merge_weights(state_dict, modules_dim, modules_alpha, lora_chains, device, dtype)
            self.lora_chains = None
            return "weights are mereged."

        print("create LoRA modules")
        self.lora_chains = lora_chains
        for key in list(state_dict.keys()):
            if not key.endswith(".lora_up.weight"):
                continue

            lora_name = key.replace(".lora_up.weight", "")

            up_weight = state_dict[key]
            down_weight = state_dict.get(lora_name + ".lora_down.weight", None)

            dim, alpha = modules_dim[lora_name], modules_alpha[lora_name]

            if lora_name.startswith(LoRANetworkCompvis.LORA_PREFIX_TEXT_ENCODER):
                multiplier = self.multiplier_tenc
            else:
                multiplier = self.multiplier_unet

            lora_chain = lora_chains.get_lora_chain(lora_name)

            lora = LoRAModule(lora_name, lora_chain.module, multiplier, dim, alpha)
            lora.set_weights(up_weight, down_weight)
            lora.to(device, dtype=dtype)

            lora_chain.add_lora(index, lora)

        return f"LoRA modules are applied to {len(modules_dim)} modules."

    def merge_weights(self, state_dict, modules_dim, modules_alpha, lora_chains: LoRAChainContainer, device, dtype):
        print("merge weights")
        for key in list(state_dict.keys()):
            if "lora_up.weight" not in key:
                continue
            lora_name = key.replace(".lora_up.weight", "")

            multiplier = self.multiplier_unet if lora_name.startswith(LoRANetworkCompvis.LORA_PREFIX_UNET) else self.multiplier_tenc

            up_weight = state_dict[key]
            down_weight = state_dict.get(lora_name + ".lora_down.weight", None)
            up_weight = up_weight.to(device)
            down_weight = down_weight.to(device)

            dim, alpha = modules_dim[lora_name], modules_alpha[lora_name]
            scale = alpha / dim * multiplier

            # calculate in float
            sd = lora_chains.get_module(lora_name).state_dict()
            weight = sd["weight"]
            weight = weight.to(torch.float32)

            if len(weight.size()) == 2:
                # linear
                if len(down_weight.size()) == 4:
                    down_weight = down_weight.squeeze(3).squeeze(2)
                    up_weight = up_weight.squeeze(3).squeeze(2)
                weight = weight + (up_weight @ down_weight) * scale
            elif down_weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                weight = (
                    weight + (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * scale
                )
            else:
                # conv2d 3x3
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                weight = weight + conved * scale

            weight = weight.to(device, dtype=dtype)

            sd["weight"] = weight
            lora_chains.get_module(lora_name).load_state_dict(sd)

    # backup weights / forwards, replace MultiheadAttention for LoRA, add undecorator
    def prepare_applying(self, text_encoder, unet, merge_weights, device) -> List[LoRAChain]:
        mhas = []
        target_modules = []
        for name, module in text_encoder.named_modules():
            if module.__class__.__name__ in LoRANetworkCompvis.TEXT_ENCODER_TARGET_REPLACE_MODULE:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "Conv2d":
                        target_modules.append((True, child_module, name + "." + child_name))
                    if child_module.__class__.__name__ == "MultiheadAttention":
                        mhas.append((True, child_module, name + "." + child_name))
        for name, module in unet.named_modules():
            if module.__class__.__name__ in LoRANetworkCompvis.UNET_TARGET_REPLACE_MODULE:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "Conv2d":
                        target_modules.append((False, child_module, name + "." + child_name))

        backed_up_count = 0
        if merge_weights:
            # backup weights
            for _, rep_module, _ in target_modules:  # mhas are not backed up, because their weights are copied to to_q/k/v/out
                if not hasattr(rep_module, "_lora_org_weights"):
                    # avoid updating of original weights. state_dict is reference to original weights
                    rep_module._lora_org_weights = copy.deepcopy(rep_module.state_dict())
                    backed_up_count += 1

        # backup forward
        for _, rep_module, _ in mhas + ([] if merge_weights else target_modules):
            if not hasattr(rep_module, "_lora_org_forward"):
                rep_module._lora_org_forward = rep_module.forward
                backed_up_count += 1

        print(f"{backed_up_count} weights/forwards are backed up.")

        print("replacing MultiheadAttention")
        for _, mha_module, name in mhas:
            # print(f"replace MultiheadAttention for {name}")

            mha_attn_rep = MultiheadAttentionReplace(mha_module.num_heads, mha_module.embed_dim)

            # load weights and convert to q/k/v/out
            sd = mha_module.state_dict()
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
            mha_attn_rep.load_state_dict(rep_sd)
            mha_attn_rep.to(device)
            mha_module.forward = mha_attn_rep.forward

            target_modules.append((True, mha_attn_rep.to_q, name + ".q_proj"))
            target_modules.append((True, mha_attn_rep.to_k, name + ".k_proj"))
            target_modules.append((True, mha_attn_rep.to_v, name + ".v_proj"))
            target_modules.append((True, mha_attn_rep.to_out, name + ".out_proj"))

        lora_chains = LoRAChainContainer(merge_weights)
        first_in_text_encoder = True
        for is_text_encoder, module, name in target_modules:
            prefix = LoRANetworkCompvis.LORA_PREFIX_TEXT_ENCODER if is_text_encoder else LoRANetworkCompvis.LORA_PREFIX_UNET

            name = name.replace("wrapped.", "")  # wrapped is added by Web UI
            lora_name = prefix + "_" + name.replace(".", "_")
            lora_chains.add_chain(first_in_text_encoder, lora_name, is_text_encoder, module)
            first_in_text_encoder = False

        return lora_chains

    def set_mask(self, index, mask, height=None, width=None, **kwargs):
        if mask is None:
            # clear latest mask
            if self.lora_chains is None:
                return
            self.lora_chains.clear_mask()
            return

        self.lora_chains.set_mask(index, mask, height, width)

    def prepare_generation(self, index, num_subprompts, shared, **kwargs):
        if "lora_proc_prepare_generation" not in shared:
            shared["lora_proc_prepare_generation"] = True
            if self.lora_chains is None:
                return
            self.lora_chains.prepare_generation(num_subprompts)

    def new_step_started(self, batch_size, num_sub_prompts, shared):
        if "lora_proc_new_step" not in shared:
            shared["lora_proc_new_step"] = True
            if self.lora_chains is None:
                return
            self.lora_chains.new_step_started(batch_size, num_sub_prompts)

    def set_cond_uncond(self, cond_uncond, shared):
        if "lora_proc_set_cond_uncond" not in shared:
            shared["lora_proc_set_cond_uncond"] = True
            if self.lora_chains is None:
                return
            self.lora_chains.set_cond_uncond(cond_uncond)


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
