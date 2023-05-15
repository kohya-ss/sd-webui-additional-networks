import os
import json
import sys
import io
import base64
import platform
import subprocess as sp
from PIL import PngImagePlugin, Image

from modules import shared
import gradio as gr

import modules.ui
from modules.ui_components import ToolButton
import modules.extras
import modules.generation_parameters_copypaste as parameters_copypaste

from scripts import safetensors_hack, model_util
from scripts.model_util import MAX_MODEL_COUNT


folder_symbol = "\U0001f4c2"  # 📂
keycap_symbols = [
    "\u0031\ufe0f\u20e3",  # 1️⃣
    "\u0032\ufe0f\u20e3",  # 2️⃣
    "\u0033\ufe0f\u20e3",  # 3️⃣
    "\u0034\ufe0f\u20e3",  # 4️⃣
    "\u0035\ufe0f\u20e3",  # 5️⃣
    "\u0036\ufe0f\u20e3",  # 6️⃣
    "\u0037\ufe0f\u20e3",  # 7️⃣
    "\u0038\ufe0f\u20e3",  # 8️
    "\u0039\ufe0f\u20e3",  # 9️
    "\u1f51f",  # 🔟
]


def write_webui_model_preview_image(model_path, image):
    basename, ext = os.path.splitext(model_path)
    preview_path = f"{basename}.png"

    # Copy any text-only metadata
    use_metadata = False
    metadata = PngImagePlugin.PngInfo()
    for key, value in image.info.items():
        if isinstance(key, str) and isinstance(value, str):
            metadata.add_text(key, value)
            use_metadata = True

    image.save(preview_path, "PNG", pnginfo=(metadata if use_metadata else None))


def delete_webui_model_preview_image(model_path):
    basename, ext = os.path.splitext(model_path)
    preview_paths = [f"{basename}.preview.png", f"{basename}.png"]

    for preview_path in preview_paths:
        if os.path.isfile(preview_path):
            os.unlink(preview_path)


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

        image.save(output_bytes, "PNG", pnginfo=(metadata if use_metadata else None))
        bytes_data = output_bytes.getvalue()
    return base64.b64encode(bytes_data)


def open_folder(f):
    if not os.path.exists(f):
        print(f'Folder "{f}" does not exist. After you create an image, the folder will be created.')
        return
    elif not os.path.isdir(f):
        print(
            f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
""",
            file=sys.stderr,
        )
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


def copy_metadata_to_all(module, model_path, copy_dir, same_session_only, missing_meta_only, cover_image):
    """
    Given a model with metadata, copies that metadata to all models in copy_dir.

    :str module: Module name ("LoRA")
    :str model: Model key in lora_models ("MyModel(123456abcdef)")
    :str copy_dir: Directory to copy to
    :bool same_session_only: Only copy to modules with the same ss_session_id
    :bool missing_meta_only: Only copy to modules that are missing user metadata
    :Optional[Image] cover_image: Cover image to embed in the file as base64
    :returns: gr.HTML.update()
    """
    if model_path == "None":
        return "No model selected."

    if not os.path.isfile(model_path):
        return f"Model path not found: {model_path}"

    model_path = os.path.realpath(model_path)

    if os.path.splitext(model_path)[1] != ".safetensors":
        return "Model is not in .safetensors format."

    if not os.path.isdir(copy_dir):
        return "Please provide a directory containing models in .safetensors format."

    print(f"[MetadataEditor] Copying metadata to models in {copy_dir}.")
    metadata = model_util.read_model_metadata(model_path, module)
    count = 0
    for entry in os.scandir(copy_dir):
        if entry.is_file():
            path = os.path.realpath(os.path.join(copy_dir, entry.name))
            if path != model_path and model_util.is_safetensors(path):
                if same_session_only:
                    other_metadata = safetensors_hack.read_metadata(path)
                    if missing_meta_only and other_metadata.get("ssmd_display_name", "").strip():
                        print(f"[MetadataEditor] Skipping {path} as it already has metadata")
                        continue

                    session_id = metadata.get("ss_session_id", None)
                    other_session_id = other_metadata.get("ss_session_id", None)
                    if session_id is None or other_session_id is None or session_id != other_session_id:
                        continue

                updates = {
                    "ssmd_cover_images": "[]",
                    "ssmd_display_name": "",
                    "ssmd_version": "",
                    "ssmd_keywords": "",
                    "ssmd_author": "",
                    "ssmd_source": "",
                    "ssmd_description": "",
                    "ssmd_rating": "0",
                    "ssmd_tags": "",
                }

                for k, v in metadata.items():
                    if k.startswith("ssmd_") and k != "ssmd_cover_images":
                        updates[k] = v

                model_util.write_model_metadata(path, module, updates)
                count += 1

    print(f"[MetadataEditor] Updated {count} models in directory {copy_dir}.")
    return f"Updated {count} models in directory {copy_dir}."


def load_cover_image(model_path, metadata):
    """
    Loads a cover image either from embedded metadata or an image file with
    .preview.png/.png format
    """
    cover_images = json.loads(metadata.get("ssmd_cover_images", "[]"))
    cover_image = None
    if len(cover_images) > 0:
        print("[MetadataEditor] Loading embedded cover image.")
        cover_image = decode_base64_to_pil(cover_images[0])
    else:
        basename, ext = os.path.splitext(model_path)

        preview_paths = [f"{basename}.preview.png", f"{basename}.png"]

        for preview_path in preview_paths:
            if os.path.isfile(preview_path):
                print(f"[MetadataEditor] Loading webui preview image: {preview_path}")
                cover_image = Image.open(preview_path)

    return cover_image


# Dummy value since gr.Dataframe cannot handle an empty list
# https://github.com/gradio-app/gradio/issues/3182
unknown_folders = ["(Unknown)", 0, 0, 0]


def refresh_metadata(module, model_path):
    """
    Reads metadata from the model on disk and updates all Gradio components
    """
    if model_path == "None":
        return {}, None, "", "", "", "", "", 0, "", "", "", "", "", {}, [unknown_folders]

    if not os.path.isfile(model_path):
        return (
            {"info": f"Model path not found: {model_path}"},
            None,
            "",
            "",
            "",
            "",
            "",
            0,
            "",
            "",
            "",
            "",
            "",
            {},
            [unknown_folders],
        )

    if os.path.splitext(model_path)[1] != ".safetensors":
        return (
            {"info": "Model is not in .safetensors format."},
            None,
            "",
            "",
            "",
            "",
            "",
            0,
            "",
            "",
            "",
            "",
            "",
            {},
            [unknown_folders],
        )

    metadata = model_util.read_model_metadata(model_path, module)

    if metadata is None:
        training_params = {}
        metadata = {}
    else:
        training_params = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    cover_image = load_cover_image(model_path, metadata)

    display_name = metadata.get("ssmd_display_name", "")
    author = metadata.get("ssmd_author", "")
    # version = metadata.get("ssmd_version", "")
    source = metadata.get("ssmd_source", "")
    keywords = metadata.get("ssmd_keywords", "")
    description = metadata.get("ssmd_description", "")
    rating = int(metadata.get("ssmd_rating", "0"))
    tags = metadata.get("ssmd_tags", "")
    model_hash = metadata.get("sshs_model_hash", model_util.cache("hashes").get(model_path, {}).get("model", ""))
    legacy_hash = metadata.get("sshs_legacy_hash", model_util.cache("hashes").get(model_path, {}).get("legacy", ""))

    top_tags = {}
    if "ss_tag_frequency" in training_params:
        tag_frequency = json.loads(training_params.pop("ss_tag_frequency"))
        count_max = 0
        for dir, frequencies in tag_frequency.items():
            for tag, count in frequencies.items():
                tag = tag.strip()
                existing = top_tags.get(tag, 0)
                top_tags[tag] = count + existing
        if len(top_tags) > 0:
            top_tags = dict(sorted(top_tags.items(), key=lambda x: x[1], reverse=True))

            count_max = max(top_tags.values())
            top_tags = {k: float(v / count_max) for k, v in top_tags.items()}

    dataset_folders = []
    if "ss_dataset_dirs" in training_params:
        dataset_dirs = json.loads(training_params.pop("ss_dataset_dirs"))
        for dir, counts in dataset_dirs.items():
            img_count = int(counts["img_count"])
            n_repeats = int(counts["n_repeats"])
            dataset_folders.append([dir, img_count, n_repeats, img_count * n_repeats])
    if dataset_folders:
        dataset_folders.append(
            ["(Total)", sum(r[1] for r in dataset_folders), sum(r[2] for r in dataset_folders), sum(r[3] for r in dataset_folders)]
        )
    else:
        dataset_folders.append(unknown_folders)

    return (
        training_params,
        cover_image,
        display_name,
        author,
        source,
        keywords,
        description,
        rating,
        tags,
        model_hash,
        legacy_hash,
        model_path,
        os.path.dirname(model_path),
        top_tags,
        dataset_folders,
    )


def save_metadata(module, model_path, cover_image, display_name, author, source, keywords, description, rating, tags):
    """
    Writes metadata from the Gradio components to the model file
    """
    if model_path == "None":
        return "No model selected.", "", ""

    if not os.path.isfile(model_path):
        return f"file not found: {model_path}", "", ""

    if os.path.splitext(model_path)[1] != ".safetensors":
        return "Model is not in .safetensors format", "", ""

    metadata = safetensors_hack.read_metadata(model_path)
    model_hash = safetensors_hack.hash_file(model_path)
    legacy_hash = model_util.get_legacy_hash(metadata, model_path)

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
        "ssmd_author": author,
        # "ssmd_version": version,
        "ssmd_source": source,
        "ssmd_keywords": keywords,
        "ssmd_description": description,
        "ssmd_rating": rating,
        "ssmd_tags": tags,
        "sshs_model_hash": model_hash,
        "sshs_legacy_hash": legacy_hash,
    }

    model_util.write_model_metadata(model_path, module, updates)
    if cover_image is None:
        delete_webui_model_preview_image(model_path)
    else:
        write_webui_model_preview_image(model_path, cover_image)

    model_name = os.path.basename(model_path)
    return f"Model saved: {model_name}", model_hash, legacy_hash


model_name_filter = ""


def get_filtered_model_paths(s):
    # newer Gradio seems to show None in the list?
    # if not s:
    #     return ["None"] + list(model_util.lora_models.values())
    # return ["None"] + [v for v in model_util.lora_models.values() if v and s in v.lower()]
    if not s:
        l =  list(model_util.lora_models.values())
    else:
        l =  [v for v in model_util.lora_models.values() if v and s in v.lower()]
    l = [v for v in l if v]     # remove None
    l = ["None"] + l
    return l

def get_filtered_model_paths_global():
    global model_name_filter
    return get_filtered_model_paths(model_name_filter)


def setup_ui(addnet_paste_params):
    """
    :dict addnet_paste_params: Dictionary of txt2img/img2img controls for each model weight slider,
                               for sending module and model to them from the metadata editor
    """
    can_edit = False

    with gr.Row().style(equal_height=False):
        # Lefthand column
        with gr.Column(variant="panel"):
            # Module and model selector
            with gr.Row():
                model_filter = gr.Textbox("", label="Model path filter", placeholder="Filter models by path name")

                def update_model_filter(s):
                    global model_name_filter
                    model_name_filter = s.strip().lower()

                model_filter.change(update_model_filter, inputs=[model_filter], outputs=[])
            with gr.Row():
                module = gr.Dropdown(
                    ["LoRA"],
                    label="Network module",
                    value="LoRA",
                    interactive=True,
                    elem_id="additional_networks_metadata_editor_module",
                )
                model = gr.Dropdown(
                    get_filtered_model_paths_global(),
                    label="Model",
                    value="None",
                    interactive=True,
                    elem_id="additional_networks_metadata_editor_model",
                )
                modules.ui.create_refresh_button(
                    model, model_util.update_models, lambda: {"choices": get_filtered_model_paths_global()}, "refresh_lora_models"
                )

                def submit_model_filter(s):
                    global model_name_filter
                    model_name_filter = s
                    paths = get_filtered_model_paths(s)
                    return gr.Dropdown.update(choices=paths, value="None")

                model_filter.submit(submit_model_filter, inputs=[model_filter], outputs=[model])

            # Model hashes and path
            with gr.Row():
                model_hash = gr.Textbox("", label="Model hash", interactive=False)
                legacy_hash = gr.Textbox("", label="Legacy hash", interactive=False)
            with gr.Row():
                model_path = gr.Textbox("", label="Model path", interactive=False)
                open_folder_button = ToolButton(
                    value=folder_symbol,
                    elem_id="hidden_element" if shared.cmd_opts.hide_ui_dir_config else "open_folder_metadata_editor",
                )

            # Send to txt2img/img2img buttons
            for tabname in ["txt2img", "img2img"]:
                with gr.Row():
                    with gr.Box():
                        with gr.Row():
                            gr.HTML(f"Send to {tabname}:")
                            for i in range(MAX_MODEL_COUNT):
                                send_to_button = ToolButton(
                                    value=keycap_symbols[i], elem_id=f"additional_networks_send_to_{tabname}_{i}"
                                )
                                send_to_button.click(
                                    fn=lambda modu, mod: (modu, model_util.find_closest_lora_model_name(mod) or "None"),
                                    inputs=[module, model],
                                    outputs=[addnet_paste_params[tabname][i]["module"], addnet_paste_params[tabname][i]["model"]],
                                )
                                send_to_button.click(fn=None, _js=f"addnet_switch_to_{tabname}", inputs=None, outputs=None)

            # "Copy metadata to other models" panel
            with gr.Row():
                with gr.Column():
                    gr.HTML(value="Copy metadata to other models in directory")
                    copy_metadata_dir = gr.Textbox(
                        "",
                        label="Containing directory",
                        placeholder="All models in this directory will receive the selected model's metadata",
                    )
                    with gr.Row():
                        copy_same_session = gr.Checkbox(True, label="Only copy to models with same session ID")
                        copy_no_metadata = gr.Checkbox(True, label="Only copy to models with no metadata")
                    copy_metadata_button = gr.Button("Copy Metadata", variant="primary")

        # Center column, metadata viewer/editor
        with gr.Column():
            with gr.Row():
                display_name = gr.Textbox(value="", label="Name", placeholder="Display name for this model", interactive=can_edit)
                author = gr.Textbox(value="", label="Author", placeholder="Author of this model", interactive=can_edit)
            with gr.Row():
                keywords = gr.Textbox(
                    value="", label="Keywords", placeholder="Activation keywords, comma-separated", interactive=can_edit
                )
            with gr.Row():
                description = gr.Textbox(
                    value="",
                    label="Description",
                    placeholder="Model description/readme/notes/instructions",
                    lines=15,
                    interactive=can_edit,
                )
            with gr.Row():
                source = gr.Textbox(
                    value="", label="Source", placeholder="Source URL where this model could be found", interactive=can_edit
                )
            with gr.Row():
                rating = gr.Slider(minimum=0, maximum=10, step=1, label="Rating", value=0, interactive=can_edit)
                tags = gr.Textbox(
                    value="",
                    label="Tags",
                    placeholder='Comma-separated list of tags ("artist, style, character, 2d, 3d...")',
                    lines=2,
                    interactive=can_edit,
                )
            with gr.Row():
                editing_enabled = gr.Checkbox(label="Editing Enabled", value=can_edit)
                with gr.Row():
                    save_metadata_button = gr.Button("Save Metadata", variant="primary", interactive=can_edit)
            with gr.Row():
                save_output = gr.HTML("")

        # Righthand column, cover image and training parameters view
        with gr.Column():
            # Cover image
            with gr.Row():
                cover_image = gr.Image(
                    label="Cover image",
                    elem_id="additional_networks_cover_image",
                    source="upload",
                    interactive=can_edit,
                    type="pil",
                    image_mode="RGBA",
                ).style(height=480)

            # Image parameters
            with gr.Accordion("Image Parameters", open=False):
                with gr.Row():
                    info2 = gr.HTML()
            with gr.Row():
                try:
                    send_to_buttons = parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "extras"])
                except:
                    pass

            # Training info, below cover image
            with gr.Accordion("Training info", open=False):
                # Top tags used
                with gr.Row():
                    max_top_tags = int(shared.opts.data.get("additional_networks_max_top_tags", 20))
                    most_frequent_tags = gr.Label(value={}, label="Most frequent tags in captions", num_top_classes=max_top_tags)

                # Dataset folders
                with gr.Row():
                    max_dataset_folders = int(shared.opts.data.get("additional_networks_max_dataset_folders", 20))
                    dataset_folders = gr.Dataframe(
                        headers=["Name", "Image Count", "Repeats", "Total Images"],
                        datatype=["str", "number", "number", "number"],
                        label="Dataset folder structure",
                        max_rows=max_dataset_folders,
                        col_count=(4, "fixed"),
                    )

                # Training Parameters
                with gr.Row():
                    metadata_view = gr.JSON(value={}, label="Training parameters")

            # Hidden/internal
            with gr.Row(visible=False):
                info1 = gr.HTML()
                img_file_info = gr.Textbox(label="Generate Info", interactive=False, lines=6)

    open_folder_button.click(fn=lambda p: open_folder(os.path.dirname(p)), inputs=[model_path], outputs=[])
    copy_metadata_button.click(
        fn=copy_metadata_to_all,
        inputs=[module, model, copy_metadata_dir, copy_same_session, copy_no_metadata, cover_image],
        outputs=[save_output],
    )

    def update_editing(enabled):
        """
        Enable/disable components based on "Editing Enabled" status
        """
        updates = [gr.Textbox.update(interactive=enabled)] * 6
        updates.append(gr.Image.update(interactive=enabled))
        updates.append(gr.Slider.update(interactive=enabled))
        updates.append(gr.Button.update(interactive=enabled))
        return updates

    editing_enabled.change(
        fn=update_editing,
        inputs=[editing_enabled],
        outputs=[display_name, author, source, keywords, description, tags, cover_image, rating, save_metadata_button],
    )

    cover_image.change(fn=modules.extras.run_pnginfo, inputs=[cover_image], outputs=[info1, img_file_info, info2])

    try:
        parameters_copypaste.bind_buttons(send_to_buttons, cover_image, img_file_info)
    except:
        pass

    model.change(
        refresh_metadata,
        inputs=[module, model],
        outputs=[
            metadata_view,
            cover_image,
            display_name,
            author,
            source,
            keywords,
            description,
            rating,
            tags,
            model_hash,
            legacy_hash,
            model_path,
            copy_metadata_dir,
            most_frequent_tags,
            dataset_folders,
        ],
    )
    save_metadata_button.click(
        save_metadata,
        inputs=[module, model, cover_image, display_name, author, source, keywords, description, rating, tags],
        outputs=[save_output, model_hash, legacy_hash],
    )
