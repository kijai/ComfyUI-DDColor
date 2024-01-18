# ComfyUI-DDColor

Node to use DDColor (https://github.com/piddnad/DDColor) in ComfyUI

![image](https://github.com/kijai/ComfyUI-DDColor/assets/40791699/6c1bd9d1-8b8a-4c03-9768-806adf8b1920)


The models are selected from the dropdown, and automatically downloaded to the `/checkpoints` folder using hugginface_hub, alternatively download them manually from here: 

https://huggingface.co/piddnad/DDColor-models/tree/main

Info about the different models: https://github.com/piddnad/DDColor/blob/master/MODEL_ZOO.md

As settings go, I don't understand yet what the model input size should be, seems to vary per model/image, working values to try: 256, 512, 1024.
