import os
import glob
import zipfile
import json
import stat
import sys
import inspect
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
LORA_MODEL_EXTS = [".pt", ".ckpt", ".safetensors"]
lora_models = {}      # "My_Lora(abcd1234)" -> C:/path/to/model.safetensors
lora_model_names = {} # "my_lora" -> "My_Lora(abcd1234)"
lora_models_dir = os.path.join(scripts.basedir(), "models/LoRA")
os.makedirs(lora_models_dir, exist_ok=True)


def traverse_all_files(curr_path, model_list):
  f_list = [(os.path.join(curr_path, entry.name), entry.stat()) for entry in os.scandir(curr_path)]
  for f_info in f_list:
    fname, fstat = f_info
    if os.path.splitext(fname)[1] in LORA_MODEL_EXTS:
      model_list.append(f_info)
    elif stat.S_ISDIR(fstat.st_mode):
      model_list = traverse_all_files(fname, model_list)
  return model_list


def get_all_models(sort_by, filter_by, path):
  res = OrderedDict()
  fileinfos = traverse_all_files(path, [])
  filter_by = filter_by.strip(" ")
  if len(filter_by) != 0:
    fileinfos = [x for x in fileinfos if filter_by.lower() in os.path.basename(x[0]).lower()]
  if sort_by == "name":
    fileinfos = sorted(fileinfos, key=lambda x: os.path.basename(x[0]))
  elif sort_by == "date":
    fileinfos = sorted(fileinfos, key=lambda x: -x[1].st_mtime)
  elif sort_by == "path name":
    fileinfos = sorted(fileinfos)

  for finfo in fileinfos:
    filename = finfo[0]
    name = os.path.splitext(os.path.basename(filename))[0]
    # Prevent a hypothetical "None.pt" from being listed.
    if name != "None":
      res[name + f"({sd_models.model_hash(filename)})"] = filename

  return res


def find_closest_lora_model_name(search: str):
    if not search:
        return None
    if search in lora_models:
        return search
    search = search.lower()
    if search in lora_model_names:
        return lora_model_names.get(search)
    applicable = [name for name in lora_model_names.keys() if search in name.lower()]
    if not applicable:
        return None
    applicable = sorted(applicable, key=lambda name: len(name))
    return lora_model_names[applicable[0]]


def update_lora_models():
  global lora_models, lora_model_names
  res = OrderedDict()
  paths = [lora_models_dir]
  extra_lora_path = shared.opts.data.get("additional_networks_extra_lora_path", None)
  if extra_lora_path and os.path.exists(extra_lora_path):
    paths.append(extra_lora_path)
  for path in paths:
    sort_by = shared.opts.data.get("additional_networks_sort_models_by", "name")
    filter_by = shared.opts.data.get("additional_networks_model_name_filter", "")
    found = get_all_models(sort_by, filter_by, path)
    res = {**found, **res}
  lora_models = OrderedDict(**{"None": None}, **res)
  lora_model_names = {}
  for name_and_hash, filename in lora_models.items():
      if filename == None:
          continue
      name = os.path.splitext(os.path.basename(filename))[0].lower()
      lora_model_names[name] = name_and_hash


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
    # NOTE: Changing the contents of `ctrls` means the XY Grid support may need
    # to be updated, see end of file
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
            model = gr.Dropdown(list(lora_models.keys()),
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
            update = gr.Dropdown.update(value=selected, choices=list(lora_models.keys()))
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


def read_lora_metadata(model_path, module):
  if model_path.startswith("\"") and model_path.endswith("\""):             # trim '"' at start/end
    model_path = model_path[1:-1]
  if not os.path.exists(model_path):
    return None

  metadata = None
  if module == "LoRA":
    if os.path.splitext(model_path)[1] == '.safetensors':
      from safetensors.torch import safe_open
      with safe_open(model_path, framework="pt") as f:
        metadata = f.metadata()

  return metadata


def on_ui_tabs():
  with gr.Blocks(analytics_enabled=False) as additional_networks_interface:
    with gr.Row().style(equal_height=False):
      with gr.Column(variant='panel'):
        with gr.Row():
          module = gr.Dropdown(["LoRA"], label=f"Network module (used throughout this tab)", value="LoRA", interactive=True)
          model = gr.Dropdown(list(lora_models.keys()), label=f"Model", value="None", interactive=True)
          modules.ui.create_refresh_button(model, update_lora_models, lambda: {"choices": list(lora_models.keys())}, "refresh_lora_models")

        with gr.Row():
            with gr.Column():
              gr.HTML(value="Get comma-separated list of models (for XY Grid)")
              model_dir = gr.Textbox("", label=f"Model directory", placeholder="Optional, uses selected model's directory if blank")
              model_sort_by = gr.Radio(label="Sort models by", choices=["name", "date", "path name"], value="name", type="value")
              get_list_button = gr.Button("Get List")
            with gr.Column():
              model_list = gr.Textbox(value="", label="Model list", placeholder="Model list will be output here")

      with gr.Column():
        metadata_view = gr.JSON(value="test", label="Network metadata")

    def update_metadata(module, model):
      if model == "None":
        return {}

      model_path = lora_models.get(model, None)
      if model_path is None:
        return f"file not found: {model_path}"

      metadata = read_lora_metadata(model_path, module)

      if metadata is None:
        return "No metadata found."
      else:
        return metadata

    model.change(update_metadata, inputs=[module, model], outputs=[metadata_view])

    def output_model_list(module, model, model_dir, sort_by):
        if model_dir == "":
            # Get list of models with same folder as this one
            model_path = lora_models.get(model, None)
            if model_path is None:
                return f"file not found: {model_path}"
            model_dir = os.path.dirname(model_path)

        if not os.path.isdir(model_dir):
            return f"directory not found: {model_dir}"

        found = get_all_models(sort_by, "", model_dir)
        return ", ".join(found.keys())

    get_list_button.click(output_model_list, inputs=[module, model, model_dir, model_sort_by], outputs=[model_list])

  return [(additional_networks_interface, "Additional Networks", "additional_networks")]


def update_script_args(p, value, arg_idx):
    for s in scripts.scripts_txt2img.alwayson_scripts:
        if isinstance(s, Script):
            args = list(p.script_args)
            # print(f"Changed arg {arg_idx} from {args[s.args_from + arg_idx - 1]} to {value}")
            args[s.args_from + arg_idx] = value
            p.script_args = tuple(args)
            break


def confirm_models(p, xs):
    for x in xs:
        if x in ["", "None"]:
            continue
        if not find_closest_lora_model_name(x):
            raise RuntimeError(f"Unknown LoRA model: {x}")


def apply_module(p, x, xs, i):
    update_script_args(p, True, 0)      # set Enabled to True
    update_script_args(p, x, 1 + 3 * i) # enabled, ({module}, model, weight), ...


def apply_model(p, x, xs, i):
    name = find_closest_lora_model_name(x)
    update_script_args(p, True, 0)
    update_script_args(p, name, 2 + 3 * i) # enabled, (module, {model}, weight), ...


def apply_weight(p, x, xs, i):
    update_script_args(p, True, 0)
    update_script_args(p, x, 3 + 3 * i) # enabled, (module, model, {weight), ...


LORA_METADATA_NAMES = {
    "ss_learning_rate": "Learning rate",
    "ss_text_encoder_lr": "Text encoder LR",
    "ss_unet_lr": "UNet LR",
    "ss_num_train_images": "# of training images",
    "ss_num_reg_images": "# of reg images",
    "ss_num_batches_per_epoch": "Batches per epoch",
    "ss_num_epochs": "Total epochs",
    "ss_batch_size_per_device": "Batch size/device",
    "ss_total_batch_size": "Total batch size",
    "ss_gradient_accumulation_steps": "Gradient accum. steps",
    "ss_max_train_steps": "Max train steps",
    "ss_lr_warmup_steps": "LR warmup steps",
    "ss_lr_scheduler": "LR scheduler",
    "ss_network_module": "Network module",
    "ss_network_dim": "Network dim",
    "ss_mixed_precision": "Mixed precision", 
    "ss_full_fp16": "Full FP16",
    "ss_v2": "V2",
    "ss_resolution": "Resolution",
    "ss_clip_skip": "Clip skip",
    "ss_max_token_length": "Max token length",
    "ss_color_aug": "Color aug",
    "ss_flip_aug": "Flip aug",
    "ss_random_crop": "Random crop",
    "ss_shuffle_caption": "Shuffle caption",
    "ss_cache_latents": "Cache latents",
    "ss_enable_bucket": "Enable bucket",
    "ss_min_bucket_reso": "Min bucket reso.",
    "ss_max_bucket_reso": "Max bucket reso.",
    "ss_seed": "Seed", 
    "ss_sd_model_name": "SD model name",
    "ss_vae_name": "VAE name"
}


def format_lora_model(p, opt, x):
    model = find_closest_lora_model_name(x)
    if model is None or model.lower() in ["", "none"]:
        return "None"

    value = xy_grid.format_value(p, opt, model)

    model_path = lora_models.get(model)
    metadata = read_lora_metadata(model_path, "LoRA")
    if not metadata:
        return value

    metadata_names = shared.opts.data.get("additional_networks_xy_grid_model_metadata", "").split(",")
    if not metadata_names:
        return value

    for name in metadata_names:
        name = name.strip()
        if name in metadata:
            formatted_name = LORA_METADATA_NAMES.get(name, name)
            value += f"\n{formatted_name}: {metadata[name]}, "

    return value.strip(" ").strip(",")


for script_class, path, basedir, script_module in scripts.scripts_data:
    if os.path.basename(path) == "xy_grid.py":
        xy_grid = script_module
        for i in range(MAX_MODEL_COUNT):
           model = xy_grid.AxisOption(f"AddNet Model {i+1}", str, lambda p, x, xs, i=i: apply_model(p, x, xs, i), format_lora_model, confirm_models)
           weight = xy_grid.AxisOption(f"AddNet Weight {i+1}", float, lambda p, x, xs, i=i: apply_weight(p, x, xs, i), xy_grid.format_value_add_label, None)
           xy_grid.axis_options.extend([model, weight])


def on_ui_settings():
    section = ('additional_networks', "Additional Networks")
    shared.opts.add_option("additional_networks_extra_lora_path", shared.OptionInfo("", "Extra path to scan for LoRA models (e.g. training output directory)", section=section))
    shared.opts.add_option("additional_networks_sort_models_by", shared.OptionInfo("name", "Sort LoRA models by", gr.Radio, {"choices": ["name", "date", "path name"]}, section=section))
    shared.opts.add_option("additional_networks_model_name_filter", shared.OptionInfo("", "LoRA model name filter", section=section))
    shared.opts.add_option("additional_networks_xy_grid_model_metadata", shared.OptionInfo("", "Metadata to show in XY-Grid label for Model axes, comma-separated (example: \"ss_learning_rate, ss_num_epochs\")", section=section))


def on_infotext_pasted(infotext, params):
    for i in range(MAX_MODEL_COUNT):
        if f"AddNet Module {i+1}" not in params:
            params[f"AddNet Module {i+1}"] = "LoRA"
        if f"AddNet Model {i+1}" not in params:
            params[f"AddNet Model {i+1}"] = "None"
        if f"AddNet Weight {i+1}" not in params:
            params[f"AddNet Weight {i+1}"] = "0"


script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_infotext_pasted(on_infotext_pasted)
