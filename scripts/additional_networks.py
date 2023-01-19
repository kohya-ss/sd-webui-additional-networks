import os
import glob
import zipfile
import json
import stat
import sys
import io
import inspect
import base64
import shutil
import re
import platform
import subprocess as sp
from collections import OrderedDict
from multiprocessing.pool import ThreadPool as Pool
import tqdm
from PIL import PngImagePlugin, Image

import torch

import modules.scripts as scripts
from modules import shared, script_callbacks
import gradio as gr

from modules.processing import Processed, process_images
from modules import sd_models, hashes
import modules.ui
from modules.ui_components import ToolButton
import modules.extras
import modules.generation_parameters_copypaste as parameters_copypaste

from scripts import lora_compvis, safetensors_hack


folder_symbol = '\U0001f4c2'  # 📂
keycap_symbols = [
  '\u0031\ufe0f\u20e3',  # 1️⃣
  '\u0032\ufe0f\u20e3',  # 2️⃣
  '\u0033\ufe0f\u20e3',  # 3️⃣
  '\u0034\ufe0f\u20e3',  # 4️⃣
  '\u0035\ufe0f\u20e3',  # 5️⃣
  '\u0036\ufe0f\u20e3',  # 6️⃣
  '\u0037\ufe0f\u20e3',  # 7️⃣
  '\u0038\ufe0f\u20e3',  # 8️
  '\u0039\ufe0f\u20e3',  # 9️
  '\u1f51f'              # 🔟
]


# Metadata pertaining to LoRA training
LORA_TRAIN_METADATA_NAMES = {
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
    "ss_vae_name": "VAE name",
    "ss_training_started_at": "Training started at",
    "ss_output_name": "Output name",
    "ss_session_id": "Session ID",
    "ss_dataset_dirs": "Dataset dirs",
    "ss_reg_dataset_dirs": "Reg. dataset dirs",
    "ss_keep_tokens": "Keep tokens",
}


MAX_MODEL_COUNT = 5
LORA_MODEL_EXTS = [".pt", ".ckpt", ".safetensors"]
re_legacy_hash = re.compile("\(([0-9a-f]{8})\)$")
lora_models = {}             # "My_Lora(abcd1234)" -> "C:/path/to/model.safetensors"
lora_model_names = {}        # "my_lora" -> "My_Lora(abcd1234)"
legacy_model_names = {}
lora_models_dir = os.path.join(scripts.basedir(), "models/lora")
addnet_paste_params = {"txt2img": [], "img2img": []}
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


def get_model_hash(metadata, filename):
  if not metadata:
    return hashes.calculate_sha256(filename)

  if "sshs_model_hash" in metadata:
    return metadata["sshs_model_hash"]

  return safetensors_hack.hash_file(filename)


def get_legacy_hash(metadata, filename):
  if not metadata:
    return sd_models.model_hash(filename)

  if "sshs_legacy_hash" in metadata:
    return metadata["sshs_legacy_hash"]

  return sd_models.model_hash(filename)


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
    fileinfos += traverse_all_files(path, [])

  print("[AddNet] Updating model hashes...")
  data = []
  thread_count = shared.opts.data.get("additional_networks_hash_thread_count", 1)
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
    data = sorted(data)
  elif sort_by == "rating":
    data = sorted(data, key=lambda x: get_model_rating(x["fileinfo"][0]), reverse=True)

  for result in data:
    finfo = result["fileinfo"]
    filename = finfo[0]
    stat = finfo[1]
    model_hash = result["model"]
    legacy_hash = result["legacy"]

    name = os.path.splitext(os.path.basename(filename))[0]

    # Prevent a hypothetical "None.pt" from being listed.
    if name != "None":
      full_name = name + f"({model_hash[0:10]})"
      res[full_name] = filename
      res_legacy[legacy_hash] = full_name
      cache_hashes[filename] = {"model": model_hash, "legacy": legacy_hash, "mtime": stat.st_mtime}

  return res, res_legacy


def find_closest_lora_model_name(search: str):
    if not search or search == "None":
        return None

    # Match name and hash, case-sensitive
    # "MyModel-epoch00002(abcde12345)"
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
    global addnet_paste_params
    # NOTE: Changing the contents of `ctrls` means the XY Grid support may need
    # to be updated, see end of file
    ctrls = []
    model_dropdowns = []

    tabname = "txt2img"
    if is_img2img:
      tabname = "img2img"

    paste_params = addnet_paste_params[tabname]
    paste_params.clear()

    self.infotext_fields = []
    with gr.Group():
      with gr.Accordion('Additional Networks', open=False, elem_id=f"additional_networks_{tabname}"):
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
            paste_params.append({"module": module, "model": model})
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
      metadata = safetensors_hack.read_metadata(model_path)

  return metadata


def write_lora_metadata(model_path, module, updates):
  if model_path.startswith("\"") and model_path.endswith("\""):             # trim '"' at start/end
    model_path = model_path[1:-1]
  if not os.path.exists(model_path):
    return None

  from safetensors.torch import save_file

  back_up = shared.opts.data.get("additional_networks_back_up_model_when_saving", True)
  if back_up:
    backup_path = model_path + ".backup"
    if not os.path.exists(backup_path):
      print(f"[AddNet] Backing up current model to {backup_path}")
      shutil.copyfile(model_path, backup_path)

  metadata = None
  tensors = {}
  if module == "LoRA":
    if os.path.splitext(model_path)[1] == '.safetensors':
      tensors, metadata = safetensors_hack.load_file(model_path, "cpu")

      for k, v in updates.items():
        metadata[k] = str(v)

      save_file(tensors, model_path, metadata)
      print(f"[AddNet] Model saved: {model_path}")


def decode_base64_to_pil(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(encoding)))


def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:

        # Copy any text-only metadata
        use_metadata = False
        metadata = PngImagePlugin.PngInfo()
        for key, value in image.info.items():
            if isinstance(key, str) and isinstance(value, str):
                metadata.add_text(key, value)
                use_metadata = True

        image.save(
            output_bytes, "PNG", pnginfo=(metadata if use_metadata else None)
        )
        bytes_data = output_bytes.getvalue()
    return base64.b64encode(bytes_data)


def on_ui_tabs():
  can_edit = False

  def open_folder(f):
    if not os.path.exists(f):
      print(f'Folder "{f}" does not exist. After you create an image, the folder will be created.')
      return
    elif not os.path.isdir(f):
      print(f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
""", file=sys.stderr)
      return

    if not shared.cmd_opts.hide_ui_dir_config:
      path = os.path.normpath(f)
      if platform.system() == "Windows":
        os.startfile(path)
      elif platform.system() == "Darwin":
        sp.Popen(["open", path])
      elif "microsoft-standard-WSL2" in platform.uname().release:
        sp.Popen(["wsl-open", path])
      else:
        sp.Popen(["xdg-open", path])

  with gr.Blocks(analytics_enabled=False) as additional_networks_interface:
    with gr.Row().style(equal_height=False):
      with gr.Column(variant='panel'):
        with gr.Row():
          module = gr.Dropdown(["LoRA"], label=f"Network module", value="LoRA", interactive=True)
          model = gr.Dropdown(list(lora_models.keys()), label=f"Model", value="None", interactive=True)
          modules.ui.create_refresh_button(model, update_lora_models, lambda: {"choices": list(lora_models.keys())}, "refresh_lora_models")

        with gr.Row():
          model_hash = gr.Textbox("", label="Model hash", interactive=False)
          legacy_hash = gr.Textbox("", label="Legacy hash", interactive=False)
        with gr.Row():
          model_path = gr.Textbox("", label="Model path", interactive=False)
          open_folder_button = ToolButton(value=folder_symbol, elem_id="hidden_element" if shared.cmd_opts.hide_ui_dir_config else "open_folder_additional_networks")
        for tabname in ["txt2img", "img2img"]:
          with gr.Row():
            with gr.Box():
              with gr.Row():
                gr.HTML(f"Send to {tabname}:")
                for i in range(MAX_MODEL_COUNT):
                  send_to_button = gr.Button(value=keycap_symbols[i], elem_id=f"additional_networks_send_to_{tabname}_{i}")
                  send_to_button.click(fn=lambda *x: x, inputs=[module, model], outputs=[addnet_paste_params[tabname][i]["module"], addnet_paste_params[tabname][i]["model"]])
                  send_to_button.click(fn=None,_js=f"addnet_switch_to_{tabname}", inputs=None, outputs=None)

        with gr.Row():
          with gr.Column():
            gr.HTML(value="Copy metadata to other models in directory")
            copy_metadata_dir = gr.Textbox("", label="Containing directory", placeholder="All models in this directory will receive the selected model's metadata")
            copy_same_session = gr.Checkbox(True, label="Only copy to models with same session ID")
            copy_metadata_button = gr.Button("Copy Metadata", variant="primary")

        with gr.Row():
          with gr.Column():
            gr.HTML(value="Get comma-separated list of models (for XY Grid)")
            model_dir = gr.Textbox("", label=f"Model directory", placeholder="Optional, uses selected model's directory if blank")
            model_sort_by = gr.Radio(label="Sort models by", choices=["name", "date", "path name", "rating"], value="name", type="value")
            get_list_button = gr.Button("Get List")
          with gr.Column():
            model_list = gr.Textbox(value="", label="Model list", placeholder="Model list will be output here")

      with gr.Column():
        with gr.Row():
          display_name = gr.Textbox(value="", label="Name", placeholder="Display name for this model", interactive=can_edit)
          author = gr.Textbox(value="", label="Author", placeholder="Author of this model", interactive=can_edit)
        with gr.Row():
          keywords = gr.Textbox(value="", label="Keywords", placeholder="Activation keywords, comma-separated", interactive=can_edit)
        with gr.Row():
          description = gr.Textbox(value="", label="Description", placeholder="Model description/readme/notes/instructions", lines=15, interactive=can_edit)
        with gr.Row():
          rating = gr.Slider(minimum=0, maximum=10, step=1, label="Rating", value=0)
          tags = gr.Textbox(value="", label="Tags", placeholder="Comma-separated list of tags (\"artist, style, landscape, 2d, 3d...\")", lines=2, interactive=can_edit)
        with gr.Row():
          editing_enabled = gr.Checkbox(label="Editing Enabled", value=can_edit)
          with gr.Row():
            save_metadata_button = gr.Button("Save Metadata", variant="primary", interactive=can_edit)
        with gr.Row():
          save_output = gr.HTML("")
      with gr.Column():
        with gr.Row():
          cover_image = gr.Image(label="Cover image", elem_id="additional_networks_cover_image", source="upload", interactive=can_edit, type="pil", image_mode="RGBA").style(height=480)
        with gr.Row():
          try:
              send_to_buttons = parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "extras"])
          except:
              pass
        with gr.Row():
          metadata_view = gr.JSON(value="{}", label="Training parameters")
        with gr.Row(visible=False):
          info1 = gr.Textbox()
          info2 = gr.Textbox()
          img_file_info = gr.Textbox(label="Generate Info", interactive=False, lines=6)

    open_folder_button.click(fn=lambda p: open_folder(os.path.dirname(p)), inputs=[model_path], outputs=[])

    def copy_metadata_to_all(module, model, copy_dir, same_session_only):
      if model == "None":
        return "No model loaded."

      model_path = lora_models.get(model, None)
      if model_path is None:
        return f"Model path not found: {model}"

      model_path = os.path.realpath(model_path)

      if os.path.splitext(model_path)[1] != ".safetensors":
        return "Model is not in .safetensors format."

      if not os.path.isdir(copy_dir):
        return "Please provide a directory containing models in .safetensors format."

      metadata = read_lora_metadata(model_path, module)
      count = 0
      for entry in os.scandir(copy_dir):
        if entry.is_file():
          path = os.path.realpath(os.path.join(copy_dir, entry.name))
          if path != model_path and is_safetensors(path):
            if same_session_only:
              other_metadata = safetensors_hack.read_metadata(path)
              session_id = metadata.get("ss_session_id", None)
              other_session_id = other_metadata.get("ss_session_id", None)
              if session_id is None or other_session_id is None or session_id != other_session_id:
                continue

            updates = {
              "ssmd_cover_images": "[]",
              "ssmd_display_name": "",
              "ssmd_keywords": "",
              "ssmd_author": "",
              "ssmd_description": "",
              "ssmd_rating": "0",
              "ssmd_tags": "",
            }

            for k, v in metadata.items():
              if k.startswith("ssmd_"):
                updates[k] = v

            del updates["ssmd_rating"]

            write_lora_metadata(path, module, updates)
            count += 1

      print(f"[AddNet] Updated {count} models in directory {copy_dir}.")
      return f"Updated {count} models in directory {copy_dir}."

    copy_metadata_button.click(fn=copy_metadata_to_all, inputs=[module, model, copy_metadata_dir, copy_same_session], outputs=[save_output])

    def update_editing(enabled):
      updates = [gr.Textbox.update(interactive=enabled)] * 5
      updates.append(gr.Image.update(interactive=enabled))
      return updates
    editing_enabled.change(fn=update_editing, inputs=[editing_enabled], outputs=[display_name, author, keywords, description, tags, cover_image])

    cover_image.change(fn=modules.extras.run_pnginfo, inputs=[cover_image], outputs=[info1, img_file_info, info2])

    try:
        parameters_copypaste.bind_buttons(send_to_buttons, cover_image, img_file_info)
    except:
        pass

    def refresh_metadata(module, model):
      if model == "None":
        return {"info": "No model loaded."}, None, "", "", "", "", 0, "", "", "", ""

      model_path = lora_models.get(model, None)
      if model_path is None:
        return {"info": f"Model path not found: {model}"}, None, "", "", "", "", 0, "", "", "", ""

      if os.path.splitext(model_path)[1] != ".safetensors":
        return {"info": "Model is not in .safetensors format."}, None, "", "", "", "", 0, "", "", "", ""

      metadata = read_lora_metadata(model_path, module)

      if metadata is None:
        training_params = {}
        metadata = {}
      else:
        training_params = {k: v for k, v in metadata.items() if k.startswith("ss_")}

      cover_images = json.loads(metadata.get("ssmd_cover_images", "[]"))
      cover_image = None
      if len(cover_images) > 0:
        cover_image = decode_base64_to_pil(cover_images[0])
      display_name = metadata.get("ssmd_display_name", "")
      author = metadata.get("ssmd_author", "")
      keywords = metadata.get("ssmd_keywords", "")
      description = metadata.get("ssmd_description", "")
      rating = int(metadata.get("ssmd_rating", "0"))
      tags = metadata.get("ssmd_tags", "")
      model_hash = metadata.get("sshs_model_hash", cache("hashes").get(model_path, {}).get("model", ""))
      legacy_hash = metadata.get("sshs_legacy_hash", cache("hashes").get(model_path, {}).get("legacy", ""))

      return training_params, cover_image, display_name, author, keywords, description, rating, tags, model_hash, legacy_hash, model_path

    model.change(refresh_metadata, inputs=[module, model], outputs=[metadata_view, cover_image, display_name, author, keywords, description, rating, tags, model_hash, legacy_hash, model_path])
    model.change(lambda: "", inputs=[], outputs=[copy_metadata_dir])

    def save_metadata(module, model, cover_image, display_name, author, keywords, description, rating, tags):
      if model == "None":
        return "No model selected.", "", ""

      model_path = lora_models.get(model, None)
      if model_path is None:
        return f"file not found: {model_path}", "", ""

      if os.path.splitext(model_path)[1] != ".safetensors":
        return "Model is not in .safetensors format", "", ""

      metadata = safetensors_hack.read_metadata(model_path)
      model_hash = safetensors_hack.hash_file(model_path)
      legacy_hash = get_legacy_hash(metadata, model_path)

      # TODO: Support multiple images
      # Blocked on gradio not having a gallery upload option
      # https://github.com/gradio-app/gradio/issues/1379
      cover_images = []
      if cover_image is not None:
        cover_images.append(encode_pil_to_base64(cover_image).decode("ascii"))

      # NOTE: User-specified metadata should NOT be prefixed with "ss_". This is
      # to maintain backwards compatibility with the old hashing method. "ss_"
      # should be used for training parameters that will never be manually
      # updated on the model.
      updates = {
        "ssmd_cover_images": json.dumps(cover_images),
        "ssmd_display_name": display_name,
        "ssmd_keywords": keywords,
        "ssmd_author": author,
        "ssmd_description": description,
        "ssmd_rating": rating,
        "ssmd_tags": tags,
        "sshs_model_hash": model_hash,
        "sshs_legacy_hash": legacy_hash
      }

      write_lora_metadata(model_path, module, updates)
      return "Model saved.", model_hash, legacy_hash

    save_metadata_button.click(save_metadata, inputs=[module, model, cover_image, display_name, author, keywords, description, rating, tags], outputs=[save_output, model_hash, legacy_hash])

    def output_model_list(module, model, model_dir, sort_by):
        if model_dir == "":
            # Get list of models with same folder as this one
            model_path = lora_models.get(model, None)
            if model_path is None:
                return f"file not found: {model_path}"
            model_dir = os.path.dirname(model_path)

        if not os.path.isdir(model_dir):
            return f"directory not found: {model_dir}"

        found, found_legacy = get_all_models([model_dir], sort_by, "")
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
    if os.path.basename(scriptDataTuple.path) == "xy_grid.py":
        xy_grid = scriptDataTuple.module
        for i in range(MAX_MODEL_COUNT):
           model = xy_grid.AxisOption(f"AddNet Model {i+1}", str, lambda p, x, xs, i=i: apply_model(p, x, xs, i), format_lora_model, confirm_models, 0.5)
           weight = xy_grid.AxisOption(f"AddNet Weight {i+1}", float, lambda p, x, xs, i=i: apply_weight(p, x, xs, i), xy_grid.format_value_add_label, None, 0)
           xy_grid.axis_options.extend([model, weight])


def on_ui_settings():
    section = ('additional_networks', "Additional Networks")
    shared.opts.add_option("additional_networks_extra_lora_path", shared.OptionInfo("", "Extra path to scan for LoRA models (e.g. training output directory)", section=section))
    shared.opts.add_option("additional_networks_sort_models_by", shared.OptionInfo("name", "Sort LoRA models by", gr.Radio, {"choices": ["name", "date", "path name", "rating"]}, section=section))
    shared.opts.add_option("additional_networks_model_name_filter", shared.OptionInfo("", "LoRA model name filter", section=section))
    shared.opts.add_option("additional_networks_xy_grid_model_metadata", shared.OptionInfo("", "Metadata to show in XY-Grid label for Model axes, comma-separated (example: \"ss_learning_rate, ss_num_epochs\")", section=section))
    shared.opts.add_option("additional_networks_back_up_model_when_saving", shared.OptionInfo(True, "Make a backup copy of the model being edited when saving its metadata.", section=section))
    shared.opts.add_option("additional_networks_hash_thread_count", shared.OptionInfo(1, "# of threads to use for hash calculation (increase if using an SSD)", section=section))


def on_infotext_pasted(infotext, params):
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
