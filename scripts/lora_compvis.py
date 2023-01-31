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
    """ if alpha == 0 or None, alpha is rank (no scaling). """
    super().__init__()
    self.lora_name = lora_name
    self.lora_dim = lora_dim

    if org_module.__class__.__name__ == 'Conv2d':
      in_dim = org_module.in_channels
      out_dim = org_module.out_channels
      self.lora_down = torch.nn.Conv2d(in_dim, lora_dim, (1, 1), bias=False)
      self.lora_up = torch.nn.Conv2d(lora_dim, out_dim, (1, 1), bias=False)
    else:
      in_dim = org_module.in_features
      out_dim = org_module.out_features
      self.lora_down = torch.nn.Linear(in_dim, lora_dim, bias=False)
      self.lora_up = torch.nn.Linear(lora_dim, out_dim, bias=False)

    if type(alpha) == torch.Tensor:
      alpha = alpha.detach().float().numpy()                              # without casting, bf16 causes error
    alpha = lora_dim if alpha is None or alpha == 0 else alpha
    self.scale = alpha / self.lora_dim
    self.register_buffer('alpha', torch.tensor(alpha))                    # 定数として扱える

    # same as microsoft's
    torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
    torch.nn.init.zeros_(self.lora_up.weight)

    self.multiplier = multiplier
    self.org_forward = org_module.forward
    self.org_module = org_module                  # remove in applying

  def apply_to(self):
    self.org_forward = self.org_module.forward
    self.org_module.forward = self.forward
    del self.org_module

  def forward(self, x):
    """
    may be cascaded.
    """
    return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale


def create_network_and_apply_compvis(du_state_dict, multiplier_tenc, multiplier_unet, text_encoder, unet, **kwargs):
  # get device and dtype from unet
  for module in unet.modules():
    if module.__class__.__name__ == "Linear":
      param: torch.nn.Parameter = module.weight
      # device = param.device
      dtype = param.dtype
      break

  # get dims (rank) and alpha from state dict
  # currently it is assumed all LoRA have same alpha. alpha may be different in future.
  network_alpha = None
  network_dim = None
  for key, value in du_state_dict.items():
    if network_alpha is None and 'alpha' in key:
      network_alpha = value
    if network_dim is None and 'lora_down' in key and len(value.size()) == 2:
      network_dim = value.size()[0]
    if network_alpha is not None and network_dim is not None:
      break
  if network_alpha is None:
    network_alpha = network_dim

  print(f"dimension: {network_dim}, alpha: {network_alpha}, multiplier_unet: {multiplier_unet}, multiplier_tenc: {multiplier_tenc}")
  if network_dim is None:
    print(f"The selected model is not LoRA or not trained by `sd-scripts`?")
    network_dim = 4
    network_alpha = 1

  # create, apply and load weights
  network = LoRANetworkCompvis(text_encoder, unet, multiplier_tenc=multiplier_tenc,
                               multiplier_unet=multiplier_unet, lora_dim=network_dim, alpha=network_alpha)
  state_dict = network.apply_lora_modules(du_state_dict)              # some weights are applied to text encoder
  network.to(dtype)                                              # with this, if error comes from next line, the model will be used
  info = network.load_state_dict(state_dict, strict=False)

  # remove redundant warnings
  if len(info.missing_keys) > 4:
    missing_keys = []
    alpha_count = 0
    for key in info.missing_keys:
      if 'alpha' not in key:
        missing_keys.append(key)
      else:
        if alpha_count == 0:
          missing_keys.append(key)
        alpha_count += 1
    if alpha_count > 1:
      missing_keys.append(
          f"... and {alpha_count-1} alphas. The model doesn't have alpha, use dim (rannk) as alpha. You can ignore this message.")

    info = torch.nn.modules.module._IncompatibleKeys(missing_keys, info.unexpected_keys)

  return network, info


class LoRANetworkCompvis(torch.nn.Module):
  # UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
  # TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
  UNET_TARGET_REPLACE_MODULE = ["SpatialTransformer"]  # , "Attention"]
  TEXT_ENCODER_TARGET_REPLACE_MODULE = ["ResidualAttentionBlock", "CLIPAttention", "CLIPMLP"]

  LORA_PREFIX_UNET = 'lora_unet'
  LORA_PREFIX_TEXT_ENCODER = 'lora_te'

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

        cv_index = 1 + du_block_index * 3 + du_attn_index      # 1,2, 4,5, 7,8
        cv_name = f"lora_unet_input_blocks_{cv_index}_1_{du_suffix}"
      else:
        m = re.search(r"_mid_block_attentions_(\d+)_(.+)", du_name)
        if m:
          du_suffix = m.group(2)
          cv_name = f"lora_unet_middle_block_1_{du_suffix}"
        else:
          m = re.search(r"_up_blocks_(\d+)_attentions_(\d+)_(.+)", du_name)
          if m:
            du_block_index = int(m.group(1))
            du_attn_index = int(m.group(2))
            du_suffix = m.group(3)

            cv_index = du_block_index * 3 + du_attn_index      # 3,4,5, 6,7,8, 9,10,11
            cv_name = f"lora_unet_output_blocks_{cv_index}_1_{du_suffix}"
    elif "lora_te_" in du_name:
      m = re.search(r"_model_encoder_layers_(\d+)_(.+)", du_name)
      if m:
        du_block_index = int(m.group(1))
        du_suffix = m.group(2)

        cv_index = du_block_index
        if v2:
          if 'mlp_fc1' in du_suffix:
            cv_name = f"lora_te_wrapped_model_transformer_resblocks_{cv_index}_{du_suffix.replace('mlp_fc1', 'mlp_c_fc')}"
          elif 'mlp_fc2' in du_suffix:
            cv_name = f"lora_te_wrapped_model_transformer_resblocks_{cv_index}_{du_suffix.replace('mlp_fc2', 'mlp_c_proj')}"
          elif 'self_attn':
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
      tokens = key.split('.')
      compvis_name = LoRANetworkCompvis.convert_diffusers_name_to_compvis(v2, tokens[0])
      new_key = compvis_name + '.' + '.'.join(tokens[1:])

      new_sd[new_key] = value

    return new_sd

  def __init__(self, text_encoder, unet, multiplier_tenc=1.0, multiplier_unet=1.0, lora_dim=4, alpha=1) -> None:
    super().__init__()
    self.multiplier_unet = multiplier_unet
    self.multiplier_tenc = multiplier_tenc
    self.lora_dim = lora_dim
    self.alpha = alpha

    # create module instances
    self.v2 = False

    def create_modules(prefix, root_module: torch.nn.Module, target_replace_modules, multiplier):
      loras = []
      replaced_modules = []
      for name, module in root_module.named_modules():
        if module.__class__.__name__ in target_replace_modules:
          for child_name, child_module in module.named_modules():
            if child_module.__class__.__name__ == "Linear" or (child_module.__class__.__name__ == "Conv2d" and child_module.kernel_size == (1, 1)):
              lora_name = prefix + '.' + name + '.' + child_name
              lora_name = lora_name.replace('.', '_')
              if '_resblocks_23_' in lora_name:                           # ignore last block in StabilityAi Text Encoder
                break
              lora = LoRAModule(lora_name, child_module, multiplier, self.lora_dim, self.alpha)
              loras.append(lora)

              replaced_modules.append(child_module)
            elif child_module.__class__.__name__ == "MultiheadAttention":
              # make four modules: not replacing forward method but merge weights
              self.v2 = True
              for suffix in ['q', 'k', 'v', 'out']:
                module_name = prefix + '.' + name + '.' + child_name          # ~.attn
                module_name = module_name.replace('.', '_')
                if '_resblocks_23_' in module_name:                           # ignore last block in StabilityAi Text Encoder
                  break
                lora_name = module_name + '_' + suffix
                lora_info = LoRAInfo(lora_name, module_name, child_module, multiplier, self.lora_dim, self.alpha)
                loras.append(lora_info)

                replaced_modules.append(child_module)
      return loras, replaced_modules

    self.text_encoder_loras, te_rep_modules = create_modules(LoRANetworkCompvis.LORA_PREFIX_TEXT_ENCODER,
                                                             text_encoder, LoRANetworkCompvis.TEXT_ENCODER_TARGET_REPLACE_MODULE, self.multiplier_tenc)
    print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

    self.unet_loras, unet_rep_modules = create_modules(
        LoRANetworkCompvis.LORA_PREFIX_UNET, unet, LoRANetworkCompvis.UNET_TARGET_REPLACE_MODULE, self.multiplier_unet)
    print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

    # make backup of original forward/weights, if multiple modules are applied, do in 1st module only
    backed_up = False                     # messaging purpose only
    for rep_module in te_rep_modules + unet_rep_modules:
      if rep_module.__class__.__name__ == "MultiheadAttention":      # multiple MHA modules are in list, prevent to backed up forward
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
    restored = False                        # messaging purpose only
    modules = []
    modules.extend(text_encoder.modules())
    modules.extend(unet.modules())
    for module in modules:
      if hasattr(module, "_lora_org_forward"):
        module.forward = module._lora_org_forward
        del module._lora_org_forward
        restored = True
      if hasattr(module, "_lora_org_weights"):        # module doesn't have forward and weights at same time currently, but supports it for future changing
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
        lora.apply_to()                           # ensure remove reference to original Linear: reference makes key of state_dict
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
          w_q_dw = state_dict.get(lora_info.module_name + '_q_proj.lora_down.weight')
          if w_q_dw is not None:                       # corresponding LoRa module exists
            w_q_up = state_dict[lora_info.module_name + '_q_proj.lora_up.weight']
            w_k_dw = state_dict[lora_info.module_name + '_k_proj.lora_down.weight']
            w_k_up = state_dict[lora_info.module_name + '_k_proj.lora_up.weight']
            w_v_dw = state_dict[lora_info.module_name + '_v_proj.lora_down.weight']
            w_v_up = state_dict[lora_info.module_name + '_v_proj.lora_up.weight']
            w_out_dw = state_dict[lora_info.module_name + '_out_proj.lora_down.weight']
            w_out_up = state_dict[lora_info.module_name + '_out_proj.lora_up.weight']

            sd = lora_info.module.state_dict()
            qkv_weight = sd['in_proj_weight']
            out_weight = sd['out_proj.weight']
            dev = qkv_weight.device

            def merge_weights(weight, up_weight, down_weight):
              # calculate in float
              scale = lora_info.alpha / lora_info.dim
              dtype = weight.dtype
              weight = weight.float() + lora_info.multiplier * (up_weight.to(dev, dtype=torch.float) @ down_weight.to(dev, dtype=torch.float)) * scale
              weight = weight.to(dtype)
              return weight

            q_weight, k_weight, v_weight = torch.chunk(qkv_weight, 3)
            if q_weight.size()[1] == w_q_up.size()[0]:
              q_weight = merge_weights(q_weight, w_q_up, w_q_dw)
              k_weight = merge_weights(k_weight, w_k_up, w_k_dw)
              v_weight = merge_weights(v_weight, w_v_up, w_v_dw)
              qkv_weight = torch.cat([q_weight, k_weight, v_weight])

              out_weight = merge_weights(out_weight, w_out_up, w_out_dw)

              sd['in_proj_weight'] = qkv_weight.to(dev)
              sd['out_proj.weight'] = out_weight.to(dev)

              lora_info.module.load_state_dict(sd)
            else:
              # different dim, version mismatch
              print(f"shape of weight is different: {lora_info.module_name}. SD version may be different")

            for t in ["q", "k", "v", "out"]:
              del state_dict[f"{lora_info.module_name}_{t}_proj.lora_down.weight"]
              del state_dict[f"{lora_info.module_name}_{t}_proj.lora_up.weight"]
              alpha_key = f"{lora_info.module_name}_{t}_proj.alpha"
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
    current_sd = self.state_dict()        # to get target shape
    wrapped = False
    count = 0
    for key in list(state_dict.keys()):
      if key not in current_sd:
        continue                          # might be error or another version
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
            f"weight's shape is different: {key} expected {current_sd[key].size()} found {value.size()}. SD version may be different")
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
