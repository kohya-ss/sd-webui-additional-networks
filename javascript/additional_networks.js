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

function addnet_switch_to_addnet(){
  Array.from(gradioApp().querySelector('#tabs').querySelectorAll('button')).filter(e => e.textContent.trim() === "Additional Networks")[0].click()
  return args_to_array(arguments);
}
