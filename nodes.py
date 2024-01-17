import os
import torch
import cv2
import numpy as np
from .ddcolor.ddcolor_arch import DDColor
import torch.nn.functional as F
import comfy.model_management
from huggingface_hub import snapshot_download

script_directory = os.path.dirname(os.path.abspath(__file__))

class DDColor_Colorize:
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE", ),
            "model_input_size": ("INT", {"default": 512,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "checkpoint": (
            [   
                "ddcolor_paper_tiny.pth",
                "ddcolor_paper.pth",
                "ddcolor_modelscope.pth",
                "ddcolor_artistic.pth",
            ], {
               "default": "ddcolor_paper.pth"
            }),
            
            
            },
            
            
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("colorized_image",)
    FUNCTION = "process"

    CATEGORY = "DDColor"
    @torch.no_grad()
    def process(self, image, model_input_size, checkpoint):
        self.device = comfy.model_management.get_torch_device()
        batch_size = image.shape[0]
        self.input_size = model_input_size
        self.checkpoint = checkpoint
        self.checkpoint_folder = os.path.join(script_directory, f"checkpoints")
        self.checkpoint_path = os.path.join(script_directory, f"checkpoints/{checkpoint}")

        if not os.path.isfile(self.checkpoint_path):
            try:
                snapshot_download(repo_id="piddnad/DDColor-models", allow_patterns=[self.checkpoint], local_dir=self.checkpoint_folder, local_dir_use_symlinks=False)
            except:
                raise FileNotFoundError("Checkpoint load failed.")
        if not hasattr(self, "model") or not hasattr(self, "ddcolor_model") or self.model is None or self.checkpoint != self.ddcolor_model:

            self.ddcolor_model = self.checkpoint
            if self.ddcolor_model == "ddcolor_paper_tiny.pth":
                encoder="convnext-t"
            else:
                encoder="convnext-l"
            self.model = DDColor(
                encoder_name=encoder,
                decoder_name="MultiScaleColorDecoder",
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm="Spectral",
                do_normalize=False,
                num_queries=100,
                num_scales=3,
                dec_layers=9,
            ).to(self.device)
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=torch.device("cpu"))["params"], strict=False)
            self.model.eval()

        out=[]
        for i in range(batch_size):
            self.height, self.width = image.shape[1:3]
            img = image[i].numpy().astype(np.float32)
    
            orig_l = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, :1] # (h, w, 1)

            # resize rgb image -> lab -> get grey -> rgb
            img = cv2.resize(img, (self.input_size, self.input_size))
            img_l = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, :1]
            img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
            img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

            tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
            output_ab = self.model(tensor_gray_rgb).cpu()  # (1, 2, self.height, self.width)

            # resize ab -> concat original l -> rgb
            output_ab_resize = F.interpolate(output_ab, size=(self.height, self.width))[0].float().numpy().transpose(1, 2, 0)
            output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)

            output_rgb = cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB)
            output_img = torch.from_numpy(output_rgb).float() # CHW format and add batch dimension
            out.append(output_img)

        batch_out = torch.stack(out, dim=0)
        return(batch_out,)
    
NODE_CLASS_MAPPINGS = {
    "DDColor_Colorize": DDColor_Colorize,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DDColor_Colorize": "DDColor_Colorize",
}