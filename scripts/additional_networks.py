import os
import glob
from collections import OrderedDict

import torch

import modules.scripts as scripts
from modules import shared, script_callbacks
import gradio as gr

from modules.processing import Processed, process_images
from modules import sd_models
import modules.ui

from scripts import lora_compvis


MAX_MODEL_COUNT = 5
LORA_MODEL_EXTS = ["pt", "ckpt", "safetensors"]
lora_models = {}
lora_models_dir = os.path.join(scripts.basedir(), "models/LoRA")
os.makedirs(lora_models_dir, exist_ok=True)


def update_lora_models():
  global lora_models
  res = {}
  paths = [lora_models_dir]
  extra_lora_path = shared.opts.data.get("additional_networks_extra_lora_path", None)
  if extra_lora_path and os.path.exists(extra_lora_path):
    paths.append(extra_lora_path)
  for path in paths:
    for ext in LORA_MODEL_EXTS:
      for filename in sorted(glob.iglob(os.path.join(path, f"**/*.{ext}"), recursive=True)):
        name = os.path.splitext(os.path.basename(filename))[0]
        # Prevent a hypothetical "None.pt" from being listed.
        if name != "None":
          res[name + f"({sd_models.model_hash(filename)})"] = filename
  lora_models = OrderedDict(**{"None": None}, **res)


update_lora_models()


class Script(scripts.Script):
  def __init__(self) -> None:
    super().__init__()
    self.latest_params = [(None, None, None)] * MAX_MODEL_COUNT
    self.latest_networks = []
    self.latest_model_hash = ""

  def title(self):
    return "Additional networks for generating"

  def show(self, is_img2img):
    return scripts.AlwaysVisible

  def ui(self, is_img2img):
    ctrls = []
    model_dropdowns = []
    self.infotext_fields = []
    with gr.Group():
      with gr.Accordion('Additional Networks', open=False):
        enabled = gr.Checkbox(label='Enable', value=False)
        ctrls.append(enabled)
        self.infotext_fields.append((enabled, "AddNet Enabled"))

        for i in range(MAX_MODEL_COUNT):
          with gr.Row():
            module = gr.Dropdown(["LoRA"], label=f"Network module {i+1}", value="LoRA")
            model = gr.Dropdown(sorted(lora_models.keys()),
                                          label=f"Model {i+1}",
                                          value="None")

            weight = gr.Slider(label=f"Weight {i+1}", value=1.0, minimum=-1.0, maximum=2.0, step=.05)
          ctrls.extend((module, model, weight))
          model_dropdowns.append(model)

          self.infotext_fields.extend([
              (module, f"AddNet Module {i+1}"),
              (model, f"AddNet Model {i+1}"),
              (weight, f"AddNet Weight {i+1}"),
          ])

        def refresh_all_models(*dropdowns):
          update_lora_models()
          updates = []
          for dd in dropdowns:
            if dd in lora_models:
              selected = dd
            else:
              selected = "None"
            update = gr.Dropdown.update(value=selected, choices=sorted(lora_models.keys()))
            updates.append(update)
          return updates

        refresh_models = gr.Button(value='Refresh models')
        refresh_models.click(refresh_all_models, inputs=model_dropdowns, outputs=model_dropdowns)
        ctrls.append(refresh_models)

    return ctrls

  def set_infotext_fields(self, p, params):
    for i, t in enumerate(params):
      module, model, weight = t
      if model is None or model == "None" or len(model) == 0 or weight == 0:
        continue
      p.extra_generation_params.update({
        "AddNet Enabled": True,
        f"AddNet Module {i+1}": module,
        f"AddNet Model {i+1}": model,
        f"AddNet Weight {i+1}": weight,
       })

  def process(self, p, *args):
    unet = p.sd_model.model.diffusion_model
    text_encoder = p.sd_model.cond_stage_model

    def restore_networks():
      if len(self.latest_networks) > 0:
        print("restoring last networks")
        for network, _ in self.latest_networks[::-1]:
          network.restore(text_encoder, unet)
        self.latest_networks.clear()

    if not args[0]:
      restore_networks()
      return

    params = []
    for i, ctrl in enumerate(args[1:]):
      if i % 3 == 0:
        param = [ctrl]
      else:
        param.append(ctrl)
        if i % 3 == 2:
          params.append(param)

    models_changed = (len(self.latest_networks) == 0)                   # no latest network (cleared by check-off)
    models_changed = models_changed or self.latest_model_hash != p.sd_model.sd_model_hash
    if not models_changed:
      for (l_module, l_model, l_weight), (module, model, weight) in zip(self.latest_params, params):
        if l_module != module or l_model != model or l_weight != weight:
          models_changed = True
          break

    if models_changed:
      restore_networks()
      self.latest_params = params
      self.latest_model_hash = p.sd_model.sd_model_hash

      for module, model, weight in self.latest_params:
        if model is None or model == "None" or len(model) == 0:
          continue
        if weight == 0:
          print(f"ignore because weight is 0: {model}")
          continue

        model_path = lora_models.get(model, None)
        if model_path is None:
          raise RuntimeError(f"model not found: {model}")

        if model_path.startswith("\"") and model_path.endswith("\""):             # trim '"' at start/end
          model_path = model_path[1:-1]
        if not os.path.exists(model_path):
          print(f"file not found: {model_path}")
          continue

        print(f"{module} weight: {weight}, model: {model}")
        if module == "LoRA":
          if os.path.splitext(model_path)[1] == '.safetensors':
            from safetensors.torch import load_file
            du_state_dict = load_file(model_path)
          else:
            du_state_dict = torch.load(model_path, map_location='cpu')

          network, info = lora_compvis.create_network_and_apply_compvis(du_state_dict, weight, text_encoder, unet)
          network.to(p.sd_model.device, dtype=p.sd_model.dtype)         # in medvram, device is different for u-net and sd_model, so use sd_model's

          print(f"LoRA model {model} loaded: {info}")
          self.latest_networks.append((network, model))
      if len(self.latest_networks) > 0:
        print("setting (or sd model) changed. new networks created.")

    self.set_infotext_fields(p, self.latest_params)


def on_ui_settings():
    section = ('additional_networks', "Additional Networks")
    shared.opts.add_option("additional_networks_extra_lora_path", shared.OptionInfo("", "Extra path to scan for LoRA models (e.g. training output directory)", section=section))


script_callbacks.on_ui_settings(on_ui_settings)
