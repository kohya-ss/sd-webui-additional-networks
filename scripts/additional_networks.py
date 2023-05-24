import inspect
import os

import torch
import numpy as np

import modules.scripts as scripts
from modules import shared, script_callbacks
import gradio as gr

import modules.ui
from modules.ui_components import ToolButton, FormRow

from scripts import addnet_xyz_grid_support, lora_compvis, model_util, metadata_editor
from scripts.model_util import lora_models, MAX_MODEL_COUNT
from modules.script_callbacks import CFGDenoiserParams, cfg_denoiser_callback
from modules.script_callbacks import CFGDenoisedParams, cfg_denoised_callback

memo_symbol = "\U0001F4DD"  # 📝
addnet_paste_params = {"txt2img": [], "img2img": []}

# TODO load values from settings or script files in the folder
network_modules = ["LoRA"]
module_to_python_module_mappings = {"LoRA": "lora_compvis"}


class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.latest_params = [(None, None, None, None)] * MAX_MODEL_COUNT
        self.latest_networks = []
        self.latest_model_hash = ""
        self.has_mask = False

    def title(self):
        return "Additional networks for generating"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        global addnet_paste_params
        # NOTE: Changing the contents of `ctrls` means the XY Grid support may need
        # to be updated, see xyz_grid_support.py
        ctrls = []
        weight_sliders = []
        model_dropdowns = []

        tabname = "txt2img"
        if is_img2img:
            tabname = "img2img"

        paste_params = addnet_paste_params[tabname]
        paste_params.clear()

        self.infotext_fields = []
        self.paste_field_names = []

        with gr.Group():
            with gr.Accordion("Additional Networks", open=False):
                with gr.Row():
                    enabled = gr.Checkbox(label="Enable", value=False)
                    ctrls.append(enabled)
                    self.infotext_fields.append((enabled, "AddNet Enabled"))
                    separate_weights = gr.Checkbox(label="Separate UNet/Text Encoder weights", value=False)
                    ctrls.append(separate_weights)
                    self.infotext_fields.append((separate_weights, "AddNet Separate Weights"))

                for i in range(MAX_MODEL_COUNT):
                    with FormRow(variant="compact"):
                        module = gr.Dropdown(["LoRA"], label=f"Network module {i+1}", value="LoRA")
                        model = gr.Dropdown(list(lora_models.keys()), label=f"Model {i+1}", value="None")
                        with gr.Row(visible=False):
                            model_path = gr.Textbox(value="None", interactive=False, visible=False)
                        model.change(
                            lambda module, model, i=i: model_util.lora_models.get(model, "None"),
                            inputs=[module, model],
                            outputs=[model_path],
                        )

                        # Sending from the script UI to the metadata editor has to bypass
                        # gradio since this button will exit the gr.Blocks context by the
                        # time the metadata editor tab is created, so event handlers can't
                        # be registered on it by then.
                        model_info = ToolButton(value=memo_symbol, elem_id=f"additional_networks_send_to_metadata_editor_{i}")
                        model_info.click(fn=None, _js="addnet_send_to_metadata_editor", inputs=[module, model_path], outputs=[])

                        module.change(
                            lambda module, model, i=i: addnet_xyz_grid_support.update_axis_params(i, module, model),
                            inputs=[module, model],
                            outputs=[],
                        )
                        model.change(
                            lambda module, model, i=i: addnet_xyz_grid_support.update_axis_params(i, module, model),
                            inputs=[module, model],
                            outputs=[],
                        )

                        # perhaps there is no user to train Text Encoder only, Weight A is U-Net
                        # The name of label will be changed in future (Weight A and B), but UNet and TEnc for now for easy understanding
                        with gr.Column() as col:
                            weight = gr.Slider(label=f"Weight {i+1}", value=1.0, minimum=-1.0, maximum=2.0, step=0.05, visible=True)
                            weight_unet = gr.Slider(
                                label=f"UNet Weight {i+1}", value=1.0, minimum=-1.0, maximum=2.0, step=0.05, visible=False
                            )
                            weight_tenc = gr.Slider(
                                label=f"TEnc Weight {i+1}", value=1.0, minimum=-1.0, maximum=2.0, step=0.05, visible=False
                            )

                        weight.change(lambda w: (w, w), inputs=[weight], outputs=[weight_unet, weight_tenc])
                        weight.release(lambda w: (w, w), inputs=[weight], outputs=[weight_unet, weight_tenc])
                        paste_params.append({"module": module, "model": model})

                    ctrls.extend((module, model, weight_unet, weight_tenc))
                    weight_sliders.extend((weight, weight_unet, weight_tenc))
                    model_dropdowns.append(model)

                    self.infotext_fields.extend(
                        [
                            (module, f"AddNet Module {i+1}"),
                            (model, f"AddNet Model {i+1}"),
                            (weight, f"AddNet Weight {i+1}"),
                            (weight_unet, f"AddNet Weight A {i+1}"),
                            (weight_tenc, f"AddNet Weight B {i+1}"),
                        ]
                    )

                for _, field_name in self.infotext_fields:
                    self.paste_field_names.append(field_name)

                def update_weight_sliders(separate, *sliders):
                    updates = []
                    for w, w_unet, w_tenc in zip(*(iter(sliders),) * 3):
                        if not separate:
                            w_unet = w
                            w_tenc = w
                        updates.append(gr.Slider.update(visible=not separate))  # Combined
                        updates.append(gr.Slider.update(visible=separate, value=w_unet))  # UNet
                        updates.append(gr.Slider.update(visible=separate, value=w_tenc))  # TEnc
                    return updates

                separate_weights.change(update_weight_sliders, inputs=[separate_weights] + weight_sliders, outputs=weight_sliders)

                def refresh_all_models(*dropdowns):
                    model_util.update_models()
                    updates = []
                    for dd in dropdowns:
                        if dd in lora_models:
                            selected = dd
                        else:
                            selected = "None"
                        update = gr.Dropdown.update(value=selected, choices=list(lora_models.keys()))
                        updates.append(update)
                    return updates

                # mask for regions
                with gr.Accordion("Extra args", open=False):
                    with gr.Row():
                        mask_image = gr.Image(label="mask image:")
                        ctrls.append(mask_image)

                refresh_models = gr.Button(value="Refresh models")
                refresh_models.click(refresh_all_models, inputs=model_dropdowns, outputs=model_dropdowns)
                ctrls.append(refresh_models)

        return ctrls

    def set_infotext_fields(self, p, params):
        for i, t in enumerate(params):
            module, model, weight_unet, weight_tenc = t
            if model is None or model == "None" or len(model) == 0 or (weight_unet == 0 and weight_tenc == 0):
                continue
            p.extra_generation_params.update(
                {
                    "AddNet Enabled": True,
                    f"AddNet Module {i+1}": module,
                    f"AddNet Model {i+1}": model,
                    f"AddNet Weight A {i+1}": weight_unet,
                    f"AddNet Weight B {i+1}": weight_tenc,
                }
            )

    def restore_networks(self, sd_model):
        unet = sd_model.model.diffusion_model
        text_encoder = sd_model.cond_stage_model

        if len(self.latest_networks) > 0:
            print("restoring last networks")
            for network, _ in self.latest_networks[::-1]:
                network.restore(text_encoder, unet)
            self.latest_networks.clear()

    def process_batch(self, p, *args, **kwargs):
        unet = p.sd_model.model.diffusion_model
        text_encoder = p.sd_model.cond_stage_model

        if not args[0]:
            self.restore_networks(p.sd_model)
            return

        params = []
        for i, ctrl in enumerate(args[2:]):
            if i % 4 == 0:
                param = [ctrl]
            else:
                param.append(ctrl)
                if i % 4 == 3:
                    params.append(param)

        models_changed = len(self.latest_networks) == 0  # no latest network (cleared by check-off)
        models_changed = models_changed or self.latest_model_hash != p.sd_model.sd_model_hash
        if not models_changed:
            for (l_module, l_model, l_weight_unet, l_weight_tenc), (module, model, weight_unet, weight_tenc) in zip(
                self.latest_params, params
            ):
                if l_module != module or l_model != model or l_weight_unet != weight_unet or l_weight_tenc != weight_tenc:
                    models_changed = True
                    break

        mask_image = args[-2]
        mask_exists = mask_image is not None
        models_changed = models_changed or (mask_exists != self.has_mask)

        if models_changed:
            self.restore_networks(p.sd_model)
            self.latest_params = params
            self.latest_model_hash = p.sd_model.sd_model_hash

            merge_weights = shared.opts.data.get("additional_networks_merge_weights", False)
            merge_weights = merge_weights and not mask_exists  # if mask exists, do not merge

            for module, model, weight_unet, weight_tenc in self.latest_params:
                if model is None or model == "None" or len(model) == 0:
                    continue
                if weight_unet == 0 and weight_tenc == 0:
                    print(f"ignore because weight is 0: {model}")
                    continue

                model_path = lora_models.get(model, None)
                if model_path is None:
                    raise RuntimeError(f"model not found: {model}")

                if model_path.startswith('"') and model_path.endswith('"'):  # trim '"' at start/end
                    model_path = model_path[1:-1]
                if not os.path.exists(model_path):
                    print(f"file not found: {model_path}")
                    continue

                # py_module = importlib.import_module("scripts." + module_to_python_module_mappings[module])
                print(f"{module} weight_unet: {weight_unet}, weight_tenc: {weight_tenc}, model: {model}")

                if os.path.splitext(model_path)[1] == ".safetensors":
                    from safetensors.torch import load_file

                    du_state_dict = load_file(model_path)
                else:
                    du_state_dict = torch.load(model_path, map_location="cpu")

                # TODO check if model is LoRA or not (support drop-in extra networks?)
                # like this:
                # py_module = None
                # for m in py_modules:
                #     if m.is_supported(du_state_dict):
                #         py_module = m
                #         break
                # if py_module is None:
                #     print(f"model {model} not supported")
                #     continue

                network = lora_compvis.create_network(
                    du_state_dict,
                    weight_tenc,
                    weight_unet,
                    text_encoder,
                    unet,
                    p.sd_model.device,
                    p.sd_model.dtype,
                )
                merge_weights = merge_weights and network.weights_mergeable

                self.latest_networks.append((network, model))

            if len(self.latest_networks) > 0:
                networks_shared = {}
                for i, (network, model) in enumerate(self.latest_networks):
                    if hasattr(network, "apply_to_compvis"):
                        info = network.apply_to_compvis(i, text_encoder, unet, merge_weights, networks_shared)
                        print(f"model {model} loaded: {info}")
                for i, (network, model) in enumerate(self.latest_networks):
                    if hasattr(network, "postprocess_apply"):
                        network.postprocess_apply(i, networks_shared)
                del networks_shared

                print("setting (or sd model) changed. new networks created.")

        # apply mask: currently only top 3 networks are supported
        networks_shared = {}
        self.has_mask = False
        if len(self.latest_networks) > 0 and mask_exists:
            print(f"use mask image to control LoRA regions.")
            self.has_mask = True
            mask_image = mask_image.astype(np.float32) / 255.0

            for i, (network, model) in enumerate(self.latest_networks):
                if not hasattr(network, "set_mask") or not network.support_mask:
                    print(f"This model does not support regional mask: {model}")
                    continue

                mask = None
                if i < 3:
                    img_ch = mask_image[:, :, i]  # R,G,B
                    if img_ch.max() > 0:
                        mask = torch.tensor(img_ch, dtype=p.sd_model.dtype, device=p.sd_model.device)
                if mask is None:
                    # if mask is None, the network is applied to whole image
                    mask = torch.ones((p.height // 8, p.width // 8), dtype=p.sd_model.dtype, device=p.sd_model.device)

                network.set_mask(i, mask, height=p.height, width=p.width, shared=networks_shared)
                print(f"apply mask and to subprompt. channel/subprompt: {i}, model: {model}")
            for i, (network, _) in enumerate(self.latest_networks):
                if hasattr(network, "postprocess_set_mask"):
                    network.postprocess_set_mask(i, networks_shared)

            if not shared.batch_cond_uncond:
                print("this overrides `batch_cond_uncond` to True. To avoid OOM, set batch size to 1.")
        else:
            for i, (network, _) in enumerate(self.latest_networks):
                if hasattr(network, "set_mask"):
                    network.set_mask(i, None, shared=networks_shared)

        # count num of subprompts: does prompts have always same number of subprompts?
        num_subprompts = 1  # default
        prompts = kwargs.get("prompts", None)
        if prompts is not None:
            from modules import prompt_parser

            subprompts = prompt_parser.re_AND.split(prompts[0])
            num_subprompts = len(subprompts)

        for i, (network, _) in enumerate(self.latest_networks):
            if hasattr(network, "prepare_generation"):
                network.prepare_generation(i, num_subprompts, shared=networks_shared)
        del networks_shared

        self.set_infotext_fields(p, self.latest_params)

        if not hasattr(self, "callbacks_added"):
            script_callbacks.on_cfg_denoiser(self.denoiser_callback)
            script_callbacks.on_cfg_denoised(self.denoised_callback)
            self.callbacks_added = True

    def denoiser_callback(self, params: CFGDenoiserParams):
        if not self.latest_networks:
            return
        if not hasattr(params, "text_uncond"):
            print("Web UI may not be the latest version. Attention Couple and Regional LoRA is not working.")
            return
        if params.text_uncond is None:  # no uncond?
            return

        # x, image_cond, sigma, sampling_step, total_sampling_steps, text_cond, text_uncond
        # x: sum(c+1),4,h,w       text_cond: sum(c),77*n,dim     text_uncond: b,77*n,dim
        # print(params.x.size())
        # print(params.image_cond.size())
        # print(params.text_cond.size())
        # print(params.text_uncond.size())
        """
        x
            batch #1
                subprompt #1
                subprompt #2
                subprompt #3
                uncond
            batch #2
                subprompt #1
                subprompt #2
                subprompt #3
                uncond
        
        batch #1
            subprompt #1
            subprompt #2
            subprompt #3
        batch #2
            subprompt #1
            subprompt #2
            subprompt #3
        uncond
            batch #1
            batch #2
        """
        # print("cfg_denoiser_callback", params.x.size(), params.text_cond.size(), params.text_uncond.size())
        batch_size = params.text_uncond.size()[0]
        num_sub_prompts = params.text_cond.size()[0] // batch_size
        self.batch_size = batch_size
        self.num_sub_prompts = num_sub_prompts
        # print(batch_size, num_sub_prompts)

        # set batch size and num of sub prompts to LoRA Networks (this is required only in first step)
        network_shared = {}
        for network, _ in self.latest_networks:
            network.new_step_started(batch_size, num_sub_prompts, network_shared)
        if not self.has_mask:
            return  # no mask, run as usual with LoRA

        self.org_batch_cond_uncond = shared.batch_cond_uncond
        shared.batch_cond_uncond = True  # force batch cond/uncond

        # remove extra x and sigma: attention couple requires only two x per image, cond + uncond
        nx = []
        nsigma = []
        for i in range(0, params.x.size()[0] - batch_size, num_sub_prompts):
            nx.append(params.x[i])
            nsigma.append(params.sigma[i])
        for i in range(num_sub_prompts * batch_size, params.x.size()[0]):
            nx.append(params.x[i])
            nsigma.append(params.sigma[i])
        params.x = torch.stack(nx)
        params.sigma = torch.stack(nsigma)

        # pad cond and uncond to make them have same length: this limitation came from sd_samplers_kdiffusion.py, CrossAttention and ControlNet
        # cond and uncond must have same length for single batch
        # cond and uncond may change every batch
        cond = params.text_cond
        uncond = params.text_uncond
        cond_len = cond.size()[1]
        uncond_len = uncond.size()[1]

        if params.sampling_step == 0 and cond_len != uncond_len:
            print(f"lengths of cond and uncond are mismatch. shorter one is padded: {cond_len}/{uncond_len}")
        if cond_len < uncond_len:
            cond = torch.cat(
                [cond, torch.zeros((cond.size()[0], uncond_len - cond_len, cond.size()[2]), dtype=cond.dtype, device=cond.device)],
                dim=1,
            )
        elif cond_len > uncond_len:
            uncond = torch.cat(
                [
                    uncond,
                    torch.zeros(
                        (uncond.size()[0], cond_len - uncond_len, uncond.size()[2]), dtype=uncond.dtype, device=uncond.device
                    ),
                ],
                dim=1,
            )

        # set cond and uncond for each network. network doesn't use given context
        cond_uncond = torch.cat([cond, uncond])
        for i, (network, _) in enumerate(self.latest_networks):
            network.set_cond_uncond(cond_uncond, network_shared)
        del network_shared

        # for ControlNet: use last cond, and repeat batch_size times
        cond = cond[-1].unsqueeze(0).repeat(batch_size, 1, 1)

        params.text_cond = cond
        params.text_uncond = uncond

        # print("cfg_denoiser_callback end", params.x.size(), params.text_cond.size(), params.text_uncond.size())

    def denoised_callback(self, params: CFGDenoisedParams):
        if not self.latest_networks or not self.has_mask:
            return

        # (x, sampling_step, total_sampling_steps)
        # print("cfg_denoised_callback")

        # modification to params doesn't affect caller... so remove conds from conds_list to align the number of conds with params.x with inspection(!!!)
        
        # get conds_list from parent scope
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe)
        parframe = calframe[2]
        par_locals = parframe[0].f_locals
        conds_list = par_locals.get("conds_list", None)

        if conds_list is not None:
            # set all member to (i,1). length of conds_list is batch_size
            for i in range(len(conds_list)):
                conds_list[i] = [(i, 1.0)]
        else:
            print("No 'conds_list' in the parent scope. Web UI might be different version.")

        # print(batch_size, num_sub_prompts, params.x.size(), params.sampling_step, params.total_sampling_steps)

        if hasattr(self, "org_batch_cond_uncond"):
            shared.batch_cond_uncond = self.org_batch_cond_uncond
            del self.org_batch_cond_uncond


def on_script_unloaded():
    if shared.sd_model:
        for s in scripts.scripts_txt2img.alwayson_scripts:
            if isinstance(s, Script):
                s.restore_networks(shared.sd_model)
                break


def on_ui_tabs():
    global addnet_paste_params
    with gr.Blocks(analytics_enabled=False) as additional_networks_interface:
        metadata_editor.setup_ui(addnet_paste_params)

    return [(additional_networks_interface, "Additional Networks", "additional_networks")]


def on_ui_settings():
    section = ("additional_networks", "Additional Networks")
    shared.opts.add_option(
        "additional_networks_merge_weights", shared.OptionInfo(False, "Merge weights in advance", section=section)
    )
    shared.opts.add_option(
        "additional_networks_extra_lora_path",
        shared.OptionInfo(
            "",
            """Extra paths to scan for LoRA models, comma-separated. Paths containing commas must be enclosed in double quotes. In the path, " (one quote) must be replaced by "" (two quotes).""",
            section=section,
        ),
    )
    shared.opts.add_option(
        "additional_networks_sort_models_by",
        shared.OptionInfo(
            "name",
            "Sort LoRA models by",
            gr.Radio,
            {"choices": ["name", "date", "path name", "rating", "has user metadata"]},
            section=section,
        ),
    )
    shared.opts.add_option(
        "additional_networks_reverse_sort_order", shared.OptionInfo(False, "Reverse model sort order", section=section)
    )
    shared.opts.add_option(
        "additional_networks_model_name_filter", shared.OptionInfo("", "LoRA model name filter", section=section)
    )
    shared.opts.add_option(
        "additional_networks_xy_grid_model_metadata",
        shared.OptionInfo(
            "",
            'Metadata to show in XY-Grid label for Model axes, comma-separated (example: "ss_learning_rate, ss_num_epochs")',
            section=section,
        ),
    )
    shared.opts.add_option(
        "additional_networks_hash_thread_count",
        shared.OptionInfo(1, "# of threads to use for hash calculation (increase if using an SSD)", section=section),
    )
    shared.opts.add_option(
        "additional_networks_back_up_model_when_saving",
        shared.OptionInfo(True, "Make a backup copy of the model being edited when saving its metadata.", section=section),
    )
    shared.opts.add_option(
        "additional_networks_show_only_safetensors",
        shared.OptionInfo(False, "Only show .safetensors format models", section=section),
    )
    shared.opts.add_option(
        "additional_networks_show_only_models_with_metadata",
        shared.OptionInfo(
            "disabled",
            "Only show models that have/don't have user-added metadata",
            gr.Radio,
            {"choices": ["disabled", "has metadata", "missing metadata"]},
            section=section,
        ),
    )
    shared.opts.add_option(
        "additional_networks_max_top_tags", shared.OptionInfo(20, "Max number of top tags to show", section=section)
    )
    shared.opts.add_option(
        "additional_networks_max_dataset_folders", shared.OptionInfo(20, "Max number of dataset folders to show", section=section)
    )
    # shared.opts.add_option(
    #     "additional_networks_additional_modules", shared.OptionInfo("", "Python modules to load", section=section)
    # )


def on_infotext_pasted(infotext, params):
    if "AddNet Enabled" not in params:
        params["AddNet Enabled"] = "False"

    # TODO changing "AddNet Separate Weights" does not seem to work
    if "AddNet Separate Weights" not in params:
        params["AddNet Separate Weights"] = "False"

    for i in range(MAX_MODEL_COUNT):
        # Convert combined weight into new format
        if f"AddNet Weight {i+1}" in params:
            params[f"AddNet Weight A {i+1}"] = params[f"AddNet Weight {i+1}"]
            params[f"AddNet Weight B {i+1}"] = params[f"AddNet Weight {i+1}"]

        if f"AddNet Module {i+1}" not in params:
            params[f"AddNet Module {i+1}"] = "LoRA"
        if f"AddNet Model {i+1}" not in params:
            params[f"AddNet Model {i+1}"] = "None"
        if f"AddNet Weight A {i+1}" not in params:
            params[f"AddNet Weight A {i+1}"] = "0"
        if f"AddNet Weight B {i+1}" not in params:
            params[f"AddNet Weight B {i+1}"] = "0"

        params[f"AddNet Weight {i+1}"] = params[f"AddNet Weight A {i+1}"]

        if params[f"AddNet Weight A {i+1}"] != params[f"AddNet Weight B {i+1}"]:
            params["AddNet Separate Weights"] = "True"

        # Convert potential legacy name/hash to new format
        params[f"AddNet Model {i+1}"] = str(model_util.find_closest_lora_model_name(params[f"AddNet Model {i+1}"]))

        addnet_xyz_grid_support.update_axis_params(i, params[f"AddNet Module {i+1}"], params[f"AddNet Model {i+1}"])


addnet_xyz_grid_support.initialize(Script)


script_callbacks.on_script_unloaded(on_script_unloaded)
script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_infotext_pasted(on_infotext_pasted)
