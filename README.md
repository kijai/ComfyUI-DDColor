# ComfyUI-DDColor

Node to use DDColor (https://github.com/piddnad/DDColor) in ComfyUI

![image](https://github.com/kijai/ComfyUI-DDColor/assets/40791699/2a1357a7-5780-4efe-84e9-c7ceb43cef07)

The models are selected from the dropdown, and automatically downloaded to the `/checkpoints` folder using hugginface_hub, alternatively download them manually from here: 

https://huggingface.co/piddnad/DDColor-models/tree/main

As settings go, I don't understand yet what the model input size should be, seems to vary per model/image, working values to try: 256, 512, 1024.
