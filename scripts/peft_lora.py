import os
from peft import PeftModel, LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
import safetensors
from safetensors.torch import load_file


def load_lora_model(unet, text_encoder, lora_path, adapter_name):
    if isinstance(unet, PeftModel) and adapter_name in unet.peft_config:
        return
    convert_text_encoder = False
    with safetensors.safe_open(lora_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        r = int(metadata["r"])
        lora_alpha = float(metadata["lora_alpha"])
        unet_target_modules = metadata["unet_target_modules"].split(",")
        if "text_encoder_target_modules" in metadata:
            convert_text_encoder = True
            text_encoder_target_modules = metadata["text_encoder_target_modules"].split(",")
    unet_peft_config = LoraConfig(
        inference_mode=True,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=unet_target_modules,
    )
    if convert_text_encoder:
        text_encoder_peft_config = LoraConfig(
            inference_mode=True,
            r=r,
            lora_alpha=lora_alpha,
            target_modules=text_encoder_target_modules,
        )
    state_dict = load_file(lora_path)

    if isinstance(unet, PeftModel):
        unet.add_adapter(adapter_name, unet_peft_config)
        if convert_text_encoder:
            text_encoder.add_adapter(adapter_name, text_encoder_peft_config)
            set_peft_model_state_dict(text_encoder, state_dict, adapter_name=adapter_name)
    else:
        if not hasattr(unet, "config"):
            setattr(unet, "config", {})
        unet = get_peft_model(unet, unet_peft_config, adapter_name)
        print(get_peft_model_state_dict(unet, adapter_name=adapter_name).keys())
        if convert_text_encoder:
            if not hasattr(text_encoder, "config"):
                setattr(text_encoder, "config", {})
            text_encoder = get_peft_model(text_encoder, text_encoder_peft_config, adapter_name)
            print(get_peft_model_state_dict(text_encoder, adapter_name=adapter_name).keys())

    set_peft_model_state_dict(unet, state_dict, adapter_name=adapter_name)
    if convert_text_encoder:
        set_peft_model_state_dict(text_encoder, state_dict, adapter_name=adapter_name)
    return unet, text_encoder


def create_weighted_lora_adapter(unet, text_encoder, adapters, unet_weights, te_weights, adapter_name):
    unet.add_weighted_adapter(adapters, unet_weights, adapter_name)
    if isinstance(text_encoder, PeftModel):
        text_encoder.add_weighted_adapter(adapters, te_weights, adapter_name)


def delete_lora_adapter(unet, text_encoder, adapter_name):
    unet.delete_adapter(adapter_name)
    if isinstance(text_encoder, PeftModel):
        text_encoder.delete_adapter(adapter_name)
