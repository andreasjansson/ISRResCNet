from pathlib import Path
import os.path as osp
import glob
import cv2
import numpy as np
import torch
from models.ISRResCNet import ResFBNet, ISRResCNet
from modules.problems import Super_Resolution
from utils import timer
from collections import OrderedDict

import cog

model_path = "trained_nets/isrrescnet_x4.pth"  # path of trained ISRResCNet model
test_img_folder = "LR/*"  # testset LR images path
upscale_factor = 4  # upscaling factor
num_iter_steps = 10  # number of iterative steps


class Model(cog.Model):
    def setup(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = ResFBNet(depth=5)
        self.model = ISRResCNet(self.model, max_iter=10, sigma_max=2, sigma_min=1)
        states = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(states["model_state_dict"])
        self.model.eval()
        self.model = self.model.to(self.device)

    @cog.input("image", type=Path, help="Input image to be upscaled")
    def run(self, image):
        img_lr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        img_LR = torch.from_numpy(
            np.transpose(img_lr[:, :, [2, 1, 0]], (2, 0, 1))
        ).float()
        img_LR = img_LR.unsqueeze(0)
        img_LR = img_LR.to(self.device)

        # initially upscale the LR image
        p = Super_Resolution(img_LR, scale=upscale_factor, mode="bicubic")
        if self.device.type == "cuda":
            p.cuda_()  # run on GPU
        else:
            p.cpu_()  # run on CPU

        with torch.no_grad():
            outputs = self.model.forward_all_iter(
                p, init=True, noise_estimation=True, max_iter=num_iter_steps
            )
            if isinstance(outputs, list):
                output_SR = outputs[-1]
            else:
                output_SR = outputs

        output_sr = output_SR.data.squeeze().float().cpu().clamp_(0, 255).numpy()
        output_sr = np.transpose(output_sr[[2, 1, 0], :, :], (1, 2, 0))

        out_path = cog.make_temp_path("out.png")
        cv2.imwrite(str(out_path), output_sr)

        del img_LR, img_lr
        del output_SR, output_sr
        torch.cuda.empty_cache()

        return out_path
