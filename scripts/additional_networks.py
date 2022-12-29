import os

import torch

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images

from scripts import lora_compvis


class Script(scripts.Script):
  def __init__(self) -> None:
    super().__init__()
    self.latest_params = [(None, None, None)] * 5
    self.latest_networks = []

  def title(self):
    return "Additional networks for generating"

  def show(self, is_img2img):
    return scripts.AlwaysVisible

  def ui(self, is_img2img):
    ctrls = []
    with gr.Group():
      with gr.Accordion('Additional Networks', open=False):
        enabled = gr.Checkbox(label='Enable', value=False)
        ctrls.append(enabled)

        for i in range(5):
          with gr.Row():
            module = gr.Dropdown(["LoRA"], label=f"Network module {i+1}", value="LoRA")
            model = gr.Textbox(label=f"Model {i+1}")
            weight = gr.Slider(label=f"Weight {i+1}", value=1, minimum=-1.0, maximum=2.0, step=.05)
          ctrls.extend((module, model, weight))

    return ctrls

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
    if not models_changed:
      for (l_module, l_model, l_weight), (module, model, weight) in zip(self.latest_params, params):
        if l_module != module or l_model != model or l_weight != weight:
          models_changed = True
          break
    
    if models_changed:
      print("models are changed")
      restore_networks()
      self.latest_params = params

      print("creating new networks")
      for module, model, weight in self.latest_params:
        if model is None or len(model) == 0:
          continue
        if weight <= 0:
          print(f"ignore because weight is 0: {model}")
          continue
        if not os.path.exists(model):
          return Processed(p, [], info=f"file not found: {model}")

        print(f"{module} weight: {weight}, model: {model}")
        if module == "LoRA":
          if os.path.splitext(model)[1] == '.safetensors':
            from safetensors.torch import load_file
            du_state_dict = load_file(model)
          else:
            du_state_dict = torch.load(model, map_location='cpu')

          network, info = lora_compvis.create_network_and_apply_compvis(du_state_dict, weight, text_encoder, unet)
          print(f"model loaded: {info}")
          self.latest_networks.append((network, model))
