import os
import glob
import zipfile
import json
import stat
import sys
import inspect
import re
import tqdm
from collections import OrderedDict
from multiprocessing.pool import ThreadPool as Pool

import torch

import modules.scripts as scripts
from modules import shared, script_callbacks
import gradio as gr

from modules.processing import Processed, process_images
from modules import shared, sd_models, hashes
import modules.ui

from scripts import lora_compvis, safetensors_hack


LORA_TRAIN_METADATA_NAMES = {
    "ss_session_id": "Session ID",
    "ss_training_started_at": "Training started at",
    "ss_output_name": "Output name",
    "ss_learning_rate": "Learning rate",
    "ss_text_encoder_lr": "Text encoder LR",
    "ss_unet_lr": "UNet LR",
    "ss_num_train_images": "# of training images",
    "ss_num_reg_images": "# of reg images",
    "ss_num_batches_per_epoch": "Batches per epoch",
    "ss_num_epochs": "Total epochs",
    "ss_epoch": "Epoch",
    "ss_batch_size_per_device": "Batch size/device",
    "ss_total_batch_size": "Total batch size",
    "ss_gradient_checkpointing": "Gradient checkpointing",
    "ss_gradient_accumulation_steps": "Gradient accum. steps",
    "ss_max_train_steps": "Max train steps",
    "ss_lr_warmup_steps": "LR warmup steps",
    "ss_lr_scheduler": "LR scheduler",
    "ss_network_module": "Network module",
    "ss_network_dim": "Network dim",
    "ss_network_alpha": "Network alpha",
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
    "ss_keep_tokens": "Keep tokens",
    "ss_dataset_dirs": "Dataset dirs.",
    "ss_reg_dataset_dirs": "Reg dataset dirs.",
    "ss_sd_model_name": "SD model name",
    "ss_vae_name": "VAE name",
    "ss_training_comment": "Comment"
}


MAX_MODEL_COUNT = shared.cmd_opts.addnet_max_model_count or 5
LORA_MODEL_EXTS = [".pt", ".ckpt", ".safetensors"]
re_legacy_hash = re.compile("\(([0-9a-f]{8})\)$") # matches 8-character hashes, new hash has 12 characters
lora_models = {}       # "My_Lora(abcdef123456)" -> "C:/path/to/model.safetensors"
lora_model_names = {}  # "my_lora" -> "My_Lora(My_Lora(abcdef123456)"
legacy_model_names = {}
lora_models_dir = os.path.join(scripts.basedir(), "models/lora")
axis_params = [{}] * MAX_MODEL_COUNT
os.makedirs(lora_models_dir, exist_ok=True)


def get_model_list(module, model, model_dir, sort_by):
    if model_dir == "":
        # Get list of models with same folder as this one
        model_path = lora_models.get(model, None)
        if model_path is None:
            return []
        model_dir = os.path.dirname(model_path)

    if not os.path.isdir(model_dir):
        return []

    found, _ = get_all_models([model_dir], sort_by, "")
    return found.keys()


def update_axis_params(i, module, model):
  axis_params[i] = {"module": module, "model": model}


def get_axis_model_choices(i):
  module = axis_params[i].get("module", "None")
  model = axis_params[i].get("model", "None")

  if module == "LoRA":
    if model != "None":
      sort_by = shared.opts.data.get("additional_networks_sort_models_by", "name")
      return get_model_list(module, model, "", sort_by)

  return [f"select `Model {i+1}` in `Additional Networks`. models in same folder for selected one will be shown here."]


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
            formatted_name = LORA_TRAIN_METADATA_NAMES.get(name, name)
            value += f"\n{formatted_name}: {metadata[name]}, "

    return value.strip(" ").strip(",")


for scriptDataTuple in scripts.scripts_data:
    if os.path.basename(scriptDataTuple.path) == "xy_grid.py" or os.path.basename(scriptDataTuple.path) == "xyz_grid.py":
        xy_grid = scriptDataTuple.module
        for i in range(MAX_MODEL_COUNT):
           model = xy_grid.AxisOption(f"AddNet Model {i+1}", str, lambda p, x, xs, i=i: apply_model(p, x, xs, i), format_lora_model, confirm_models, cost=0.5, choices=lambda i=i: get_axis_model_choices(i))
           weight = xy_grid.AxisOption(f"AddNet Weight {i+1}", float, lambda p, x, xs, i=i: apply_weight(p, x, xs, i), xy_grid.format_value_add_label, None, cost=0)
           xy_grid.axis_options.extend([model, weight])


def traverse_all_files(curr_path, model_list):
  f_list = [(os.path.join(curr_path, entry.name), entry.stat()) for entry in os.scandir(curr_path)]
  for f_info in f_list:
    fname, fstat = f_info
    if os.path.splitext(fname)[1] in LORA_MODEL_EXTS:
      model_list.append(f_info)
    elif stat.S_ISDIR(fstat.st_mode):
      model_list = traverse_all_files(fname, model_list)
  return model_list


def get_model_hash(metadata, filename):
  if metadata is None:
    return hashes.calculate_sha256(filename)

  if "sshs_model_hash" in metadata:
    return metadata["sshs_model_hash"]

  return safetensors_hack.hash_file(filename)


def get_legacy_hash(metadata, filename):
  if metadata is None:
    return sd_models.model_hash(filename)

  if "sshs_legacy_hash" in metadata:
    return metadata["sshs_legacy_hash"]

  return safetensors_hack.legacy_hash_file(filename)


import filelock
cache_filename = os.path.join(scripts.basedir(), "hashes.json")
cache_data = None

def cache(subsection):
    global cache_data

    if cache_data is None:
        with filelock.FileLock(cache_filename+".lock"):
            if not os.path.isfile(cache_filename):
                cache_data = {}
            else:
                with open(cache_filename, "r", encoding="utf8") as file:
                    cache_data = json.load(file)

    s = cache_data.get(subsection, {})
    cache_data[subsection] = s

    return s


def dump_cache():
    with filelock.FileLock(cache_filename+".lock"):
        with open(cache_filename, "w", encoding="utf8") as file:
            json.dump(cache_data, file, indent=4)



def is_safetensors(filename):
    return os.path.splitext(filename)[1] == ".safetensors"


def get_model_rating(filename):
  if not is_safetensors(filename):
    return 0

  metadata = safetensors_hack.read_metadata(filename)
  return int(metadata.get("ssmd_rating", "0"))


def has_user_metadata(filename):
  if not is_safetensors(filename):
    return False

  metadata = safetensors_hack.read_metadata(filename)
  return any(k.startswith("ssmd_") for k in metadata.keys())


def hash_model_file(finfo):
  filename = finfo[0]
  stat = finfo[1]
  name = os.path.splitext(os.path.basename(filename))[0]

  # Prevent a hypothetical "None.pt" from being listed.
  if name != "None":
    metadata = None

    cached = cache("hashes").get(filename, None)
    if cached is None or stat.st_mtime != cached["mtime"]:
      if metadata is None and is_safetensors(filename):
        metadata = safetensors_hack.read_metadata(filename)
      model_hash = get_model_hash(metadata, filename)
      legacy_hash = get_legacy_hash(metadata, filename)
    else:
      model_hash = cached["model"]
      legacy_hash = cached["legacy"]

  return {"model": model_hash, "legacy": legacy_hash, "fileinfo": finfo}


def get_all_models(paths, sort_by, filter_by):
  fileinfos = []
  for path in paths:
    if os.path.isdir(path):
      fileinfos += traverse_all_files(path, [])

  print("[AddNet] Updating model hashes...")
  data = []
  thread_count = max(1, int(shared.opts.data.get("additional_networks_hash_thread_count", 1)))
  p = Pool(processes=thread_count)
  with tqdm.tqdm(total=len(fileinfos)) as pbar:
      for res in p.imap_unordered(hash_model_file, fileinfos):
          pbar.update()
          data.append(res)
  p.close()

  cache_hashes = cache("hashes")

  res = OrderedDict()
  res_legacy = OrderedDict()
  filter_by = filter_by.strip(" ")
  if len(filter_by) != 0:
    data = [x for x in data if filter_by.lower() in os.path.basename(x["fileinfo"][0]).lower()]
  if sort_by == "name":
    data = sorted(data, key=lambda x: os.path.basename(x["fileinfo"][0]))
  elif sort_by == "date":
    data = sorted(data, key=lambda x: -x["fileinfo"][1].st_mtime)
  elif sort_by == "path name":
    data = sorted(data, key=lambda x: x["fileinfo"][0])
  elif sort_by == "rating":
    data = sorted(data, key=lambda x: get_model_rating(x["fileinfo"][0]), reverse=True)
  elif sort_by == "has user metadata":
    data = sorted(data, key=lambda x: os.path.basename(x["fileinfo"][0]) if has_user_metadata(x["fileinfo"][0]) else "", reverse=True)

  for result in data:
    finfo = result["fileinfo"]
    filename = finfo[0]
    stat = finfo[1]
    model_hash = result["model"]
    legacy_hash = result["legacy"]

    name = os.path.splitext(os.path.basename(filename))[0]

    # Prevent a hypothetical "None.pt" from being listed.
    if name != "None":
      full_name = name + f"({model_hash[0:12]})"
      res[full_name] = filename
      res_legacy[legacy_hash] = full_name
      cache_hashes[filename] = {"model": model_hash, "legacy": legacy_hash, "mtime": stat.st_mtime}

  return res, res_legacy


def find_closest_lora_model_name(search: str):
    if not search or search == "None":
        return None

    # Match name and hash, case-sensitive
    # "MyModel-epoch00002(abcdef123456)"
    if search in lora_models:
        return search

    # Match full name, case-insensitive
    # "mymodel-epoch00002"
    search = search.lower()
    if search in lora_model_names:
        return lora_model_names.get(search)

    # Match legacy hash (8 characters)
    # "MyModel(abcd1234)"
    result = re_legacy_hash.search(search)
    if result is not None:
        model_hash = result.group(1)
        if model_hash in legacy_model_names:
            new_model_name = legacy_model_names[model_hash]
            return new_model_name

    # Use any model with the search term as the prefix, case-insensitive, sorted
    # by name length
    # "mymodel"
    applicable = [name for name in lora_model_names.keys() if search in name.lower()]
    if not applicable:
        return None
    applicable = sorted(applicable, key=lambda name: len(name))
    return lora_model_names[applicable[0]]


def update_lora_models():
  global lora_models, lora_model_names, legacy_model_names
  paths = [lora_models_dir]
  extra_lora_path = shared.opts.data.get("additional_networks_extra_lora_path", None)
  if extra_lora_path and os.path.exists(extra_lora_path):
    paths.append(extra_lora_path)

  sort_by = shared.opts.data.get("additional_networks_sort_models_by", "name")
  filter_by = shared.opts.data.get("additional_networks_model_name_filter", "")
  res, res_legacy = get_all_models(paths, sort_by, filter_by)

  lora_models = OrderedDict(**{"None": None}, **res)
  lora_model_names = {}

  for name_and_hash, filename in lora_models.items():
      if filename == None:
          continue
      name = os.path.splitext(os.path.basename(filename))[0].lower()
      lora_model_names[name] = name_and_hash

  legacy_model_names = res_legacy
  dump_cache()


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

            module.change(lambda module, model, i=i: update_axis_params(i, module, model), inputs=[module, model], outputs=[])
            model.change(lambda module, model, i=i: update_axis_params(i, module, model), inputs=[module, model], outputs=[])

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
          # in medvram, device is different for u-net and sd_model, so use sd_model's
          network.to(p.sd_model.device, dtype=p.sd_model.dtype)

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
      metadata = safetensors_hack.read_metadata(model_path)

  return metadata


def on_ui_tabs():
  with gr.Blocks(analytics_enabled=False) as additional_networks_interface:
    with gr.Row().style(equal_height=False):
      with gr.Column(variant='panel'):
        with gr.Row():
          module = gr.Dropdown(["LoRA"], label=f"Network module (used throughout this tab)", value="LoRA", interactive=True)
          model = gr.Dropdown(list(lora_models.keys()), label=f"Model", value="None", interactive=True)
          modules.ui.create_refresh_button(model, update_lora_models, lambda: {
                                           "choices": list(lora_models.keys())}, "refresh_lora_models")

      with gr.Column():
        metadata_view = gr.JSON(value="{}", label="Network metadata")

    def update_metadata(module, model):
      if model == "None":
        return {}

      model_path = lora_models.get(model, None)
      if model_path is None:
        return f"file not found: {model_path}"

      metadata = read_lora_metadata(model_path, module)

      if metadata is None:
        return '{"info":"No metadata found."}'
      else:
        return metadata

    model.change(update_metadata, inputs=[module, model], outputs=[metadata_view])

  return [(additional_networks_interface, "Additional Networks", "additional_networks")]


def on_ui_settings():
  section = ('additional_networks', "Additional Networks")
  shared.opts.add_option("additional_networks_extra_lora_path", shared.OptionInfo(
      "", "Extra path to scan for LoRA models (e.g. training output directory)", section=section))
  shared.opts.add_option("additional_networks_sort_models_by", shared.OptionInfo(
      "name", "Sort LoRA models by", gr.Radio, {"choices": ["name", "date", "path name", "rating", "has user metadata"]}, section=section))
  shared.opts.add_option("additional_networks_model_name_filter", shared.OptionInfo("", "LoRA model name filter", section=section))
  shared.opts.add_option("additional_networks_xy_grid_model_metadata", shared.OptionInfo(
      "", "Metadata to show in XY-Grid label for Model axes, comma-separated (example: \"ss_learning_rate, ss_num_epochs\")", section=section))
  shared.opts.add_option("additional_networks_hash_thread_count", shared.OptionInfo(1, "# of threads to use for hash calculation (increase if using an SSD)", section=section))


def on_infotext_pasted(infotext, params):
  if "AddNet Enabled" not in params:
    params["AddNet Enabled"] = "False"

  for i in range(MAX_MODEL_COUNT):
    if f"AddNet Module {i+1}" not in params:
      params[f"AddNet Module {i+1}"] = "LoRA"
    if f"AddNet Model {i+1}" not in params:
      params[f"AddNet Model {i+1}"] = "None"
    if f"AddNet Weight {i+1}" not in params:
      params[f"AddNet Weight {i+1}"] = "0"

    # Convert potential legacy name/hash to new format
    params[f"AddNet Model {i+1}"] = str(find_closest_lora_model_name(params[f"AddNet Model {i+1}"]))



script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_infotext_pasted(on_infotext_pasted)
