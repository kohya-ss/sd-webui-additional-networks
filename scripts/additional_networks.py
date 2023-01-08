import os

import torch
import json

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images

from scripts import lora_compvis

try:
  from tkinter import filedialog, Tk
  tkinter_found = True
except ImportError:
  tkinter_found = False


class Script(scripts.Script):
  def __init__(self) -> None:
    super().__init__()
    self.latest_params = [(None, None, None)] * 5
    self.latest_networks = []
    self.latest_model_hash = ""

  def title(self):
    return "Additional networks for generating"

  def show(self, is_img2img):
    return scripts.AlwaysVisible
  
  def get_dir_and_file(self, file_path):
    dir_path, file_name = os.path.split(file_path)
    return (dir_path, file_name)
  
  def get_any_file_path(self, file_path='', defaultextension='.json', extension_name='Config files'):
    if not tkinter_found:
      return "tkinter not found"

    current_file_path = file_path
    # print(f'current file path: {current_file_path}')

    initial_dir, initial_file = self.get_dir_and_file(file_path)

    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    file_path = filedialog.askopenfilename(
      filetypes=((f'{extension_name}', f'{defaultextension}'), ('All files', '*')),
      defaultextension=defaultextension,
      initialdir=initial_dir,
      initialfile=initial_file,
    )
    root.destroy()

    if file_path == '':
        file_path = current_file_path

    return file_path
  
  def get_saveasfilename_path(self, file_path='', defaultextension='*.json', extension_name='Config files'):
    if not tkinter_found:
      return "tkinter not found"
    
    current_file_path = file_path
    # print(f'current file path: {current_file_path}')
    
    initial_dir, initial_file = self.get_dir_and_file(file_path)

    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    save_file_path = filedialog.asksaveasfilename(
        filetypes=((f'{extension_name}', f'{defaultextension}'), ('All files', '*')),
        defaultextension=defaultextension,
        initialdir=initial_dir,
        initialfile=initial_file,
    )
    root.destroy()

    if save_file_path == '':
        file_path = current_file_path
    else:
        # print(save_file_path)
        file_path = save_file_path

    return file_path
  
  def save_lora_config(self, config_file_path, module0, module1, module2, module3, module4, model0, model1, model2, model3, model4, weight0, weight1, weight2, weight3, weight4):
    
    # Return the values of the variables as a dictionary
    variables = {
        'module0': module0, 
        'module1': module1, 
        'module2': module2, 
        'module3': module3, 
        'module4': module4, 
        'model0': model0, 
        'model1': model1, 
        'model2': model2, 
        'model3': model3, 
        'model4': model4, 
        'weight0': weight0, 
        'weight1': weight1, 
        'weight2': weight2, 
        'weight3': weight3, 
        'weight4': weight4,
    }

    file_path = self.get_saveasfilename_path(file_path=config_file_path)
    
    if not file_path == '':
      print(f"Save LoRA config {file_path}...")
      # Save the data to the selected file
      with open(file_path, 'w') as file:
          json.dump(variables, file)
      return file_path
    else:
      print("Can't save config, no file path provided...")
        
  def load_lora_config(self, config_file_path, module0, module1, module2, module3, module4, model0, model1, model2, model3, model4, weight0, weight1, weight2, weight3, weight4):
    file_path = self.get_any_file_path(file_path=config_file_path)
    
    with open(file_path, 'r') as f:
      my_data = json.load(f)
    
    print(f"Load LoRA config {file_path}...")
    # Return the values of the variables as a dictionary
    return (
      file_path,
      my_data.get('module0', module0), 
      my_data.get('module1', module1), 
      my_data.get('module2', module2), 
      my_data.get('module3', module3), 
      my_data.get('module4', module4), 
      my_data.get('model0', model0), 
      my_data.get('model1', model1), 
      my_data.get('model2', model2), 
      my_data.get('model3', model3), 
      my_data.get('model4', model4), 
      my_data.get('weight0', weight0), 
      my_data.get('weight1', weight1), 
      my_data.get('weight2', weight2), 
      my_data.get('weight3', weight3), 
      my_data.get('weight4', weight4),
    )

  def ui(self, is_img2img):
    ctrls = []
    lora_ext = gr.Textbox(value='*.pt *.safetensors', visible=False)
    lora_ext_name = gr.Textbox(value='LoRA model types', visible=False)
    config_file_path = gr.Textbox(value='', visible=False)
    with gr.Group():
      with gr.Accordion('Additional Networks', open=False):
        with gr.Row():
          enabled = gr.Checkbox(label='Enable', value=False)
          load = gr.Button("Load config")
          save = gr.Button("Save config")
          ctrls.append(enabled)

        # for i in range(5):
        i=1
        with gr.Row():
          module0 = gr.Dropdown(["LoRA"], label=f"Network module {i+1}", value="LoRA",
              interactive=True,)
          model0 = gr.Textbox(label=f"Model {i+1}",
              interactive=True,)
          
          model_file0 = gr.Button(
              '📂', elem_id='open_folder'
          )
          model_file0.click(self.get_any_file_path, inputs=[model0,lora_ext,lora_ext_name], outputs=model0)
          weight0 = gr.Slider(label=f"Weight {i+1}", value=1, minimum=-1.0, maximum=2.0, step=.05,
              interactive=True,)
          ctrls.extend((module0, model0, weight0))
        i += 1
        with gr.Row():
          module1 = gr.Dropdown(["LoRA"], label=f"Network module {i+1}", value="LoRA",
              interactive=True,)
          model1 = gr.Textbox(label=f"Model {i+1}",
              interactive=True,)
          
          model_file1 = gr.Button(
              '📂', elem_id='open_folder'
          )
          model_file1.click(self.get_any_file_path, inputs=[model1,lora_ext,lora_ext_name], outputs=model1)
          weight1 = gr.Slider(label=f"Weight {i+1}", value=1, minimum=-1.0, maximum=2.0, step=.05,
              interactive=True,)
          ctrls.extend((module1, model1, weight1))
        i += 1
        with gr.Row():
          module2 = gr.Dropdown(["LoRA"], label=f"Network module {i+1}", value="LoRA",
              interactive=True,)
          model2 = gr.Textbox(label=f"Model {i+1}",
              interactive=True,)
          
          model_file2 = gr.Button(
              '📂', elem_id='open_folder'
          )
          model_file2.click(self.get_any_file_path, inputs=[model2,lora_ext,lora_ext_name], outputs=model2)
          weight2 = gr.Slider(label=f"Weight {i+1}", value=1, minimum=-1.0, maximum=2.0, step=.05,
              interactive=True,)
          ctrls.extend((module2, model2, weight2))
        i += 1
        with gr.Row():
          module3 = gr.Dropdown(["LoRA"], label=f"Network module {i+1}", value="LoRA",
              interactive=True,)
          model3 = gr.Textbox(label=f"Model {i+1}",
              interactive=True,)
          
          model_file3 = gr.Button(
              '📂', elem_id='open_folder'
          )
          model_file3.click(self.get_any_file_path, inputs=[model3,lora_ext,lora_ext_name], outputs=model3)
          weight3 = gr.Slider(label=f"Weight {i+1}", value=1, minimum=-1.0, maximum=2.0, step=.05,
              interactive=True,)
          ctrls.extend((module3, model3, weight3))
        i += 1
        with gr.Row():
          module4 = gr.Dropdown(["LoRA"], label=f"Network module {i+1}", value="LoRA",
              interactive=True,)
          model4 = gr.Textbox(label=f"Model {i+1}",
              interactive=True,)
          
          model_file4 = gr.Button(
              '📂', elem_id='open_folder'
          )
          model_file4.click(self.get_any_file_path, inputs=[model4,lora_ext,lora_ext_name], outputs=model4)
          weight4 = gr.Slider(label=f"Weight {i+1}", value=1, minimum=-1.0, maximum=2.0, step=.05,
              interactive=True,)
          ctrls.extend((module4, model4, weight4))
      var_list = [
        config_file_path,
        module0, 
        module1, 
        module2, 
        module3, 
        module4, 
        model0, 
        model1, 
        model2, 
        model3, 
        model4, 
        weight0, 
        weight1, 
        weight2, 
        weight3,
        weight4,
      ]
      save.click(self.save_lora_config, 
                 inputs=var_list,
                 outputs=config_file_path
                 )
      load.click(self.load_lora_config,
                 inputs=var_list,
                 outputs=var_list
                 )
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
        if model is None or len(model) == 0:
          continue
        if weight == 0:
          print(f"ignore because weight is 0: {model}")
          continue

        if model.startswith("\"") and model.endswith("\""):             # trim '"' at start/end
          model = model[1:-1]
        if not os.path.exists(model):
          print(f"file not found: {model}")
          continue

        print(f"{module} weight: {weight}, model: {model}")
        if module == "LoRA":
          if os.path.splitext(model)[1] == '.safetensors':
            from safetensors.torch import load_file
            du_state_dict = load_file(model)
          else:
            du_state_dict = torch.load(model, map_location='cpu')

          network, info = lora_compvis.create_network_and_apply_compvis(du_state_dict, weight, text_encoder, unet)
          print(f"LoRA model {model} loaded: {info}")
          self.latest_networks.append((network, model))
      if len(self.latest_networks) > 0:
        print("setting (or sd model) changed. new networks created.")
