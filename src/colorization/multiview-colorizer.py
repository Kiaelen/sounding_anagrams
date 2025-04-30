import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm 
import os

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import yaml

from diffusers import DiffusionPipeline

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from visual_anagrams.views import get_views

from src.colorization.views import LView_Composit, ABView_Composit, ColorABView, ColorLView
from src.colorization.samplers import sample_stage_1, sample_stage_2



class FactorizedColorization(nn.Module):
    '''Colorization diffusion model by Factorized Diffusion
    '''
    def __init__(
        self, 
        views,
        inverse_color=False,
        **kwargs
    ):
        super().__init__()

        # Make DeepFloyd IF stage I
        self.stage_1 = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-I-M-v1.0",
            variant="fp16",
            torch_dtype=torch.float16
        )
        self.stage_1.enable_model_cpu_offload()

        # Make DeepFloyd IF stage II
        self.stage_2 = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-II-M-v1.0",
            text_encoder=None,
            variant="fp16",
            torch_dtype=torch.float16,
        )
        self.stage_2.enable_model_cpu_offload()
        
        # if inverse the gray scale
        self.inverse_color = inverse_color

        # get views
        self.views = []
        for view in views:
            self.views += [LView_Composit(view), ABView_Composit(view)]

        self.num_inference_steps = kwargs.get("num_inference_steps", 30)
        self.guidance_scale = kwargs.get("guidance_scale", 10.0)
        self.start_diffusion_step = kwargs.get("start_diffusion_step", 0)
        self.noise_level = kwargs.get("noise_level", 50)

    
    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # Get prompt embeddings (need two, because code is designed for 
        # two components: L and ab)
        prompts = []
        for p in prompt:
            prompts += [p] * 2
        prompt_embeds = [self.stage_1.encode_prompt(p) for p in prompts]
        prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
        prompt_embeds = torch.cat(prompt_embeds)
        negative_prompt_embeds = torch.cat(negative_prompt_embeds)  # These are just null embeds
        return prompt_embeds, negative_prompt_embeds
    
    def forward(
        self,
        gray_im, 
        prompt, 
        num_inference_steps=None, 
        guidance_scale=None, 
        start_diffusion_step=None, 
        noise_level=None, 
        generator=None
    ):  
        
        # 1. overwrite the hyparams if provided
        num_inference_steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        guidance_scale = self.guidance_scale if guidance_scale is None else guidance_scale
        start_diffusion_step = self.start_diffusion_step if start_diffusion_step is None else start_diffusion_step
        noise_level = self.noise_level if noise_level is None else noise_level

        # 2. prepare the text embeddings
        prompt_embeds, negative_prompt_embeds = self.get_text_embeds(prompt)

        # import pdb; pdb.set_trace()

        # 3. prepare grayscale image
        _, height, width = gray_im.shape
        if self.inverse_color:
            gray_im = 1.0 - gray_im
        
        gray_im = gray_im * 2.0 - 1 # normalize the pixel value

        # 4. Sample 64x64 image
        image = sample_stage_1(
            self.stage_1, 
            prompt_embeds,
            negative_prompt_embeds,
            self.views,
            height=height // 4,
            width=width // 4,
            fixed_im=gray_im,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            start_diffusion_step=start_diffusion_step
        )

        # 5. Sample 256x256 image, by upsampling 64x64 image
        image = sample_stage_2(
            self.stage_2,
            image,
            prompt_embeds,
            negative_prompt_embeds, 
            self.views,
            height=height,
            width=width,
            fixed_im=gray_im,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_level=noise_level,
            generator=generator
        )

        # 6. return the final image
        image = image / 2 + 0.5 
        return image

if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser()
    # parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--sample_dir", required=True, type=str)
    # parser.add_argument("--prompt", required=True, type=str, help='Prompts to use for colorization')
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--noise_level", type=int, default=50, help='Noise level for stage 2')
    parser.add_argument("--start_diffusion_step", type=int, default=7, help='What step to start the diffusion process')


    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    sample_dir = args.sample_dir
    cfg_path = os.path.join(sample_dir, "config.yaml")
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    # Get prompts
    cfg = cfg["trainer"]
    img_prompts = [prompt.split(',')[0] for prompt in cfg["image_prompt"]]
    # img_prompts = [cfg['image_prompt'][0].split(',')[0]]
    
    print(img_prompts)
    
    # Get views
    view_names = cfg["views"]
    # view_names = ["identity"]
    views = get_views(view_names)
    
    # Load gray image
    img_path = os.path.join(sample_dir, "image", "identity.png")
    gray_im = Image.open(img_path)
    gray_im = TF.to_tensor(gray_im).to(device)

    save_dir = os.path.join(sample_dir, "colored_image")
    os.makedirs(save_dir, exist_ok=True)

    # create diffusion colorization instance 
    colorizer = FactorizedColorization(
        views=views,
        inverse_color=False,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        start_diffusion_step=args.start_diffusion_step,
        noise_level=args.noise_level,
    ).to(device)
    
    # # Sample illusions
    for i in tqdm(range(args.num_samples), desc="Sampling images"):
        generator = torch.manual_seed(args.seed + i)
        image = colorizer(gray_im, img_prompts, generator=generator)
        
        for view_name, view in zip(view_names, views):
            img = view.view(image[0])
            save_image(img, f'{save_dir}/{i}.{view_name}.png', padding=0)
