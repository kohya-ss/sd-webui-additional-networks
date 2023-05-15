import os
import os.path
from modules import shared
import modules.scripts as scripts
from scripts import model_util, util
from scripts.model_util import MAX_MODEL_COUNT


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
    "ss_training_comment": "Comment",
}


xy_grid = None  # XY Grid module
script_class = None  # additional_networks scripts.Script class
axis_params = [{}] * MAX_MODEL_COUNT


def update_axis_params(i, module, model):
    axis_params[i] = {"module": module, "model": model}


def get_axis_model_choices(i):
    module = axis_params[i].get("module", "None")
    model = axis_params[i].get("model", "None")

    if module == "LoRA":
        if model != "None":
            sort_by = shared.opts.data.get("additional_networks_sort_models_by", "name")
            return ["None"] + model_util.get_model_list(module, model, "", sort_by)

    return [f"select `Model {i+1}` in `Additional Networks`. models in same folder for selected one will be shown here."]


def update_script_args(p, value, arg_idx):
    global script_class
    for s in scripts.scripts_txt2img.alwayson_scripts:
        if isinstance(s, script_class):
            args = list(p.script_args)
            # print(f"Changed arg {arg_idx} from {args[s.args_from + arg_idx - 1]} to {value}")
            args[s.args_from + arg_idx] = value
            p.script_args = tuple(args)
            break


def confirm_models(p, xs):
    for x in xs:
        if x in ["", "None"]:
            continue
        if not model_util.find_closest_lora_model_name(x):
            raise RuntimeError(f"Unknown LoRA model: {x}")


def apply_module(p, x, xs, i):
    update_script_args(p, True, 0)  # set Enabled to True
    update_script_args(p, x, 2 + 4 * i)  # enabled, separate_weights, ({module}, model, weight_unet, weight_tenc), ...


def apply_model(p, x, xs, i):
    name = model_util.find_closest_lora_model_name(x)
    update_script_args(p, True, 0)
    update_script_args(p, name, 3 + 4 * i)  # enabled, separate_weights, (module, {model}, weight_unet, weight_tenc), ...


def apply_weight(p, x, xs, i):
    update_script_args(p, True, 0)
    update_script_args(p, x, 4 + 4 * i)  # enabled, separate_weights, (module, model, {weight_unet, weight_tenc}), ...
    update_script_args(p, x, 5 + 4 * i)


def apply_weight_unet(p, x, xs, i):
    update_script_args(p, True, 0)
    update_script_args(p, x, 4 + 4 * i)  # enabled, separate_weights, (module, model, {weight_unet}, weight_tenc), ...


def apply_weight_tenc(p, x, xs, i):
    update_script_args(p, True, 0)
    update_script_args(p, x, 5 + 4 * i)  # enabled, separate_weights, (module, model, weight_unet, {weight_tenc}), ...


def format_lora_model(p, opt, x):
    global xy_grid
    model = model_util.find_closest_lora_model_name(x)
    if model is None or model.lower() in ["", "none"]:
        return "None"

    value = xy_grid.format_value(p, opt, model)

    model_path = model_util.lora_models.get(model)
    metadata = model_util.read_model_metadata(model_path, "LoRA")
    if not metadata:
        return value

    metadata_names = util.split_path_list(shared.opts.data.get("additional_networks_xy_grid_model_metadata", ""))
    if not metadata_names:
        return value

    for name in metadata_names:
        name = name.strip()
        if name in metadata:
            formatted_name = LORA_TRAIN_METADATA_NAMES.get(name, name)
            value += f"\n{formatted_name}: {metadata[name]}, "

    return value.strip(" ").strip(",")


def initialize(script):
    global xy_grid, script_class
    xy_grid = None
    script_class = script
    for scriptDataTuple in scripts.scripts_data:
        if os.path.basename(scriptDataTuple.path) == "xy_grid.py" or os.path.basename(scriptDataTuple.path) == "xyz_grid.py":
            xy_grid = scriptDataTuple.module
            for i in range(MAX_MODEL_COUNT):
                model = xy_grid.AxisOption(
                    f"AddNet Model {i+1}",
                    str,
                    lambda p, x, xs, i=i: apply_model(p, x, xs, i),
                    format_lora_model,
                    confirm_models,
                    cost=0.5,
                    choices=lambda i=i: get_axis_model_choices(i),
                )
                weight = xy_grid.AxisOption(
                    f"AddNet Weight {i+1}",
                    float,
                    lambda p, x, xs, i=i: apply_weight(p, x, xs, i),
                    xy_grid.format_value_add_label,
                    None,
                    cost=0.5,
                )
                weight_unet = xy_grid.AxisOption(
                    f"AddNet UNet Weight {i+1}",
                    float,
                    lambda p, x, xs, i=i: apply_weight_unet(p, x, xs, i),
                    xy_grid.format_value_add_label,
                    None,
                    cost=0.5,
                )
                weight_tenc = xy_grid.AxisOption(
                    f"AddNet TEnc Weight {i+1}",
                    float,
                    lambda p, x, xs, i=i: apply_weight_tenc(p, x, xs, i),
                    xy_grid.format_value_add_label,
                    None,
                    cost=0.5,
                )
                xy_grid.axis_options.extend([model, weight, weight_unet, weight_tenc])
