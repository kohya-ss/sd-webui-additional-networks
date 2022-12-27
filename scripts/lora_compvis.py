# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import math
import os
import re
import torch


class LoRAModule(torch.nn.Module):
  """
  replaces forward method of the original Linear, instead of replacing the original Linear module.
  """

  def __init__(self, lora_name, org_module: torch.nn.Module, multiplier=1.0, lora_dim=4):
    super().__init__()
    self.lora_name = lora_name

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
    return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier


def create_network_and_apply_compvis(du_state_dict, multiplier, text_encoder, unet, **kwargs):
  # check dims
  size = du_state_dict[list(du_state_dict.keys())[0]].size()         # in conv2d size is like [320,4,1,1]
  network_dim = min([s for s in size if s > 1])
  print(f"dimension: {network_dim}, multiplier: {multiplier}")

  network = LoRANetworkCompvis(text_encoder, unet, multiplier=multiplier, lora_dim=network_dim)
  state_dict = network.apply_lora_modules(du_state_dict)
  info = network.load_state_dict(state_dict)

  # get device and dtype from unet
  for module in unet.modules():
    if module.__class__.__name__ == "Linear":
      param: torch.nn.Parameter = module.weight
      device = param.device
      dtype = param.dtype
      break
  network.to(device, dtype=dtype)

  return network, info


class LoRANetworkCompvis(torch.nn.Module):
  # UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
  # TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
  UNET_TARGET_REPLACE_MODULE = ["SpatialTransformer"]  # , "Attention"]
  TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
  LORA_PREFIX_UNET = 'lora_unet'
  LORA_PREFIX_TEXT_ENCODER = 'lora_te'

  @classmethod
  def convert_diffusers_name_to_compvis(cls, du_name):
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
        cv_name = f"lora_te_wrapped_transformer_text_model_encoder_layers_{cv_index}_{du_suffix}"
    assert cv_name is not None, f"conversion failed: {du_name}"
    return cv_name

  @classmethod
  def convert_state_dict_name_to_compvis(cls, state_dict):
    new_sd = {}
    for key, value in state_dict.items():
      tokens = key.split('.')
      compvis_name = LoRANetworkCompvis.convert_diffusers_name_to_compvis(tokens[0])
      new_key = compvis_name + '.' + '.'.join(tokens[1:])

      new_sd[new_key] = value

    return new_sd

  def __init__(self, text_encoder, unet, multiplier=1.0, lora_dim=4) -> None:
    super().__init__()
    self.multiplier = multiplier
    self.lora_dim = lora_dim

    # create module instances
    def create_modules(prefix, root_module: torch.nn.Module, target_replace_modules) -> list[LoRAModule]:
      loras = []
      replaced_modules = []
      for name, module in root_module.named_modules():
        if module.__class__.__name__ in target_replace_modules:
          for child_name, child_module in module.named_modules():
            if child_module.__class__.__name__ == "Linear" or (child_module.__class__.__name__ == "Conv2d" and child_module.kernel_size == (1, 1)):
              lora_name = prefix + '.' + name + '.' + child_name
              lora_name = lora_name.replace('.', '_')
              lora = LoRAModule(lora_name, child_module, self.multiplier, self.lora_dim)
              loras.append(lora)

              replaced_modules.append(child_module)
      return loras, replaced_modules

    self.text_encoder_loras, te_rep_modules = create_modules(LoRANetworkCompvis.LORA_PREFIX_TEXT_ENCODER,
                                                             text_encoder, LoRANetworkCompvis.TEXT_ENCODER_TARGET_REPLACE_MODULE)
    print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

    self.unet_loras, unet_rep_modules = create_modules(
        LoRANetworkCompvis.LORA_PREFIX_UNET, unet, LoRANetworkCompvis.UNET_TARGET_REPLACE_MODULE)
    print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

    # make backup of original forward
    backed_up = False
    for rep_module in te_rep_modules + unet_rep_modules:
      if not hasattr(rep_module, "_lora_org_forward"):              # 1st model only
        rep_module._lora_org_forward = rep_module.forward
        backed_up = True
    if backed_up:
      print("original forward is backed up.")

    # assertion
    names = set()
    for lora in self.text_encoder_loras + self.unet_loras:
      assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
      names.add(lora.lora_name)

  def restore(self, text_encoder, unet):
    # restore forward from property for all modules
    restored = False
    modules = []
    modules.extend(text_encoder.modules())
    modules.extend(unet.modules())
    for module in modules:
      if hasattr(module, "_lora_org_forward"):
        module.forward = module._lora_org_forward
        del module._lora_org_forward
        restored = True

    if restored:
      print("original forward is restored.")

  def apply_lora_modules(self, du_state_dict):
    # conversion 1st step: convert names in state_dict
    state_dict = LoRANetworkCompvis.convert_state_dict_name_to_compvis(du_state_dict)

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

    # add modules to network: this makes state_dict can be got
    for lora in self.text_encoder_loras + self.unet_loras:
      lora.apply_to()                           # ensure remove reference to original Linear: reference makes key of state_dict
      self.add_module(lora.lora_name, lora)

    # conversion 2nd step: convert shape (and handle wrapped)
    state_dict = self.convert_state_dict_shape_to_compvis(state_dict)

    return state_dict

  def convert_state_dict_shape_to_compvis(self, state_dict):
    # shape conversion
    current_sd = self.state_dict()
    wrapped = False
    for key in state_dict.keys():
      if key not in current_sd:
        continue                        # might be error or another version
      if "wrapped" in key:
        wrapped = True

      value: torch.Tensor = state_dict[key]
      if value.size() != current_sd[key].size():
        print(f"convert weights shape: {key}")
        if len(value.size()) == 4:
          value = value.squeeze(3).squeeze(2)
        else:
          value = value.unsqueeze(2).unsqueeze(3)
        state_dict[key] == value

    # convert wrapped
    if not wrapped:
      print("remove 'wrapped' from keys")
      for key in list(state_dict.keys()):
        if "_wrapped_" in key:
          new_key = key.replace("_wrapped_")
          state_dict[new_key] = state_dict[key]
          del state_dict[key]

    return state_dict

