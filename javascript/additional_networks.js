function addnet_switch_to_txt2img(){
  switch_to_txt2img();
  setTimeout(function() { gradioApp().getElementById("additional_networks_txt2img").scrollIntoView(); }, 100);
  return args_to_array(arguments);
}

function addnet_switch_to_img2img(){
  switch_to_img2img();
  setTimeout(function() { gradioApp().getElementById("additional_networks_img2img").scrollIntoView(); }, 100);
  return args_to_array(arguments);
}

function addnet_switch_to_metadata_editor(){
  Array.from(gradioApp().querySelector('#tabs').querySelectorAll('button')).filter(e => e.textContent.trim() === "Additional Networks")[0].click();
  return args_to_array(arguments);
}

function addnet_send_to_metadata_editor() {
  var module = arguments[0];
  var model_path = arguments[1];

  if (model_path == "None") {
    return args_to_array(arguments);
  }

  console.log(arguments);
  console.log(model_path);
  var select = gradioApp().querySelector("#additional_networks_metadata_editor_model > label > select");

  var opt = [...select.options].filter(o => o.text == model_path)[0];
  if (opt == null) {
    return;
  }

  addnet_switch_to_metadata_editor();
  select.selectedIndex = opt.index;
  select.dispatchEvent(new Event("change", { bubbles: true }));

  return args_to_array(arguments);
}
