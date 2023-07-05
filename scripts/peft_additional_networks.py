import os

import torch
import numpy as np

import modules.scripts as scripts
from modules import shared, script_callbacks
import gradio as gr

import modules.ui
from modules.ui_components import ToolButton, FormRow

from scripts import addnet_xyz_grid_support, peft_lora, lora_compvis, model_util, metadata_editor
from scripts.model_util import lora_models, MAX_MODEL_COUNT

from peft import PeftModel


memo_symbol = "\U0001F4DD"  # 📝
addnet_paste_params = {"txt2img": [], "img2img": []}


class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.latest_params = [(None, None, None, None, None)] * MAX_MODEL_COUNT
        self.latest_networks = []
        self.latest_model_hash = ""

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
            with gr.Accordion("PEFT LoRA Networks", open=False):
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
                        with gr.Row(visible=True):
                            adapter_name = gr.Textbox(value=f"lora_{i}", visible=True)
                        model.change(
                            lambda module, model, i=i: model_util.lora_models.get(model, "None"),
                            inputs=[module, model],
                            outputs=[model_path],
                        )

                        # Sending from the script UI to the metadata editor has to bypass
                        # gradio since this button will exit the gr.Blocks context by the
                        # time the metadata editor tab is created, so event handlers can't
                        # be registered on it by then.
                        model_info = ToolButton(
                            value=memo_symbol, elem_id=f"additional_networks_send_to_metadata_editor_{i}"
                        )
                        model_info.click(
                            fn=None, _js="addnet_send_to_metadata_editor", inputs=[module, model_path], outputs=[]
                        )

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
                            weight = gr.Slider(
                                label=f"Weight {i+1}", value=1.0, minimum=-1.0, maximum=2.0, step=0.05, visible=True
                            )
                            weight_unet = gr.Slider(
                                label=f"UNet Weight {i+1}",
                                value=1.0,
                                minimum=-1.0,
                                maximum=2.0,
                                step=0.05,
                                visible=False,
                            )
                            weight_tenc = gr.Slider(
                                label=f"TEnc Weight {i+1}",
                                value=1.0,
                                minimum=-1.0,
                                maximum=2.0,
                                step=0.05,
                                visible=False,
                            )

                        weight.change(lambda w: (w, w), inputs=[weight], outputs=[weight_unet, weight_tenc])
                        weight.release(lambda w: (w, w), inputs=[weight], outputs=[weight_unet, weight_tenc])
                        paste_params.append({"module": module, "model": model})

                    ctrls.extend((module, model, weight_unet, weight_tenc, adapter_name))
                    weight_sliders.extend((weight, weight_unet, weight_tenc, adapter_name))
                    model_dropdowns.append(model)

                    self.infotext_fields.extend(
                        [
                            (module, f"AddNet Module {i+1}"),
                            (model, f"AddNet Model {i+1}"),
                            (weight, f"AddNet Weight {i+1}"),
                            (weight_unet, f"AddNet Weight A {i+1}"),
                            (weight_tenc, f"AddNet Weight B {i+1}"),
                            (adapter_name, f"AddNet Model Name {i+1}"),
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

                separate_weights.change(
                    update_weight_sliders, inputs=[separate_weights] + weight_sliders, outputs=weight_sliders
                )

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
            module, model, weight_unet, weight_tenc, adapter_name = t
            if model is None or model == "None" or len(model) == 0 or (weight_unet == 0 and weight_tenc == 0):
                continue
            p.extra_generation_params.update(
                {
                    "AddNet Enabled": True,
                    f"AddNet Module {i+1}": module,
                    f"AddNet Model {i+1}": model,
                    f"AddNet Weight A {i+1}": weight_unet,
                    f"AddNet Weight B {i+1}": weight_tenc,
                    f"AddNet Model Name {i+1}": adapter_name,
                }
            )

    def restore_networks(self, sd_model):
        unet = sd_model.model.diffusion_model
        text_encoder = sd_model.cond_stage_model

        if isinstance(unet, PeftModel):
            unet.unload(merge=False)
        if isinstance(text_encoder, PeftModel):
            text_encoder.unload(merge=False)

    def process_batch(self, p, *args, **kwargs):
        unet = p.sd_model.model.diffusion_model
        text_encoder = p.sd_model.cond_stage_model

        if not args[0]:
            self.restore_networks(p.sd_model)
            return

        params = []
        for i, ctrl in enumerate(args[2:]):
            if i % 5 == 0:
                param = [ctrl]
            else:
                param.append(ctrl)
                if i % 5 == 4:
                    params.append(param)
        self.latest_params = params

        unet_weights = []
        te_weights = []
        adapters = []

        for module, model, weight_unet, weight_tenc, adapter_name in self.latest_params:
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

            print(f"{module} weight_unet: {weight_unet}, weight_tenc: {weight_tenc}, model: {model}")
            unet_weights.append(weight_unet)
            te_weights.append(weight_tenc)
            adapters.append(adapter_name)
            if isinstance(unet, PeftModel):
                print(list(unet.peft_config.keys()))
            unet, text_encoder = peft_lora.load_lora_model(unet, text_encoder, model_path, adapter_name)

        if len(unet_weights) > 0:
            weighted_adapter_name = "_".join(
                [f"{a}:{str(w).replace('.', '')}" for w, a in zip(unet_weights, adapters)]
            )
            if weighted_adapter_name in unet.peft_config:
                return
            adapters_to_remove = [
                adapter_name for adapter_name in list(unet.peft_config.keys()) if adapter_name not in adapters
            ]
            print(adapters_to_remove)
            print(adapters)
            print(list(unet.peft_config.keys()))
            print([adapter_name in unet.peft_config.keys()])
            for remove_adapter in adapters_to_remove:
                peft_lora.delete_lora_adapter(unet, text_encoder, remove_adapter)
            peft_lora.create_weighted_lora_adapter(
                unet, text_encoder, adapters, unet_weights, te_weights, weighted_adapter_name
            )
            print(list(unet.peft_config.keys()))
        print(f"{unet=}")
        print(f"{text_encoder=}")

        self.set_infotext_fields(p, self.latest_params)


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
        shared.OptionInfo(
            True, "Make a backup copy of the model being edited when saving its metadata.", section=section
        ),
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
        "additional_networks_max_dataset_folders",
        shared.OptionInfo(20, "Max number of dataset folders to show", section=section),
    )


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
