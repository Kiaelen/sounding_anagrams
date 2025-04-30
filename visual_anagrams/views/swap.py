from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange

from .permutations import get_inv_perm
from .view_base import BaseView


class SWAP(BaseView):
    def __init__(self, ):
        pass

    def view(self, im):
        h, w = im.shape[-2:]
        patches = rearrange(im, 
                            'c (h p1) (w p2) -> (h w) c p1 p2', 
                            p1=h, 
                            p2=w//2)
        shift_patches = patches[[1, 0]]
        im_rearr = rearrange(shift_patches, 
                            '(h w) c p1 p2 -> c (h p1) (w p2)', 
                            h=1, 
                            w=2, 
                            p1=h, 
                            p2=w//2)
        return im_rearr

    def inverse_view(self, noise):
        h, w = noise.shape[-2:]
        patches = rearrange(noise, 
                            'c (h p1) (w p2) -> (h w) c p1 p2', 
                            p1=h, 
                            p2=w//2)
        shift_patches = patches[[1, 0]]
        im_rearr = rearrange(shift_patches, 
                            '(h w) c p1 p2 -> c (h p1) (w p2)', 
                            h=1, 
                            w=2, 
                            p1=h, 
                            p2=w//2)
        return im_rearr
    
    def make_frame(self, im, t):
        im_size = im.size[0]
        frame_size = int(im_size * 1.5)
        theta = t * -90

        frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
        centered_loc = (frame_size - im_size) // 2
        frame.paste(im, (centered_loc, centered_loc))
        frame = frame.rotate(theta, 
                             resample=Image.Resampling.BILINEAR, 
                             expand=False, 
                             fillcolor=(255,255,255))

        return frame