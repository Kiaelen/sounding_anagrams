import os
from typing import Optional
from tqdm import tqdm
import warnings
import soundfile as sf
import numpy as np
import shutil
import glob
import copy
import matplotlib.pyplot as plt

import hydra
from omegaconf import OmegaConf, DictConfig, open_dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import logging
from lightning import seed_everything

from visual_anagrams.views import get_views
from visual_anagrams.samplers import sample_stage_1, sample_stage_2
from visual_anagrams.utils import add_args, save_illusion, save_metadata

from torchvision.utils import save_image
import torchvision.transforms as transforms

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.rich_utils import print_config_tree
from src.utils.animation_with_text import create_animation_with_text, create_single_image_animation_with_text
from src.utils.re_ranking import select_top_k_ranking, select_top_k_clip_ranking, select_top_k_clap_ranking
from src.utils.pylogger import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)


def save_audio(audio, save_path):
    sf.write(save_path, audio, samplerate=16000)


def encode_prompt(prompt, diffusion_guidance, device, negative_prompt='', time_repeat=1):
    '''Encode text prompts into embeddings 
    '''
    prompts = [prompt] * time_repeat
    negative_prompts = [negative_prompt] * time_repeat

    # Prompts -> text embeds
    cond_embeds = diffusion_guidance.get_text_embeds(prompts, device) # [B, 77, 768]
    uncond_embeds = diffusion_guidance.get_text_embeds(negative_prompts, device) # [B, 77, 768]
    text_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0) # [2 * B, 77, 768]
    return text_embeds

def estimate_noise(diffusion, latents, t, text_embeddings, guidance_scale): 
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    # predict the noise residual
    noise_pred = diffusion.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

    # perform guidance
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    return noise_pred


@hydra.main(version_base="1.3", config_path="configs/main", config_name="main.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main function for training
    """
    views = get_views(cfg.trainer.views)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    log.info(f"Instantiating Image Diffusion model <{cfg.image_diffusion_guidance._target_}>")
    image_diffusion_guidance = hydra.utils.instantiate(cfg.image_diffusion_guidance).to(device)

    log.info(f"Instantiating Audio Diffusion guidance model <{cfg.audio_diffusion_guidance._target_}>")
    audio_diffusion_guidance = hydra.utils.instantiate(cfg.audio_diffusion_guidance).to(device)

    # create the shared noise scheduler 
    log.info(f"Instantiating joint diffusion scheduler <{cfg.diffusion_scheduler._target_}>")
    scheduler = hydra.utils.instantiate(cfg.diffusion_scheduler)

    # create transformation
    log.info(f"Instantiating latent transformation <{cfg.latent_transformation._target_}>")
    latent_transformation = hydra.utils.instantiate(cfg.latent_transformation).to(device)

    # create audio evaluator
    if cfg.audio_evaluator: 
        log.info(f"Instantiating audio evaluator <{cfg.audio_evaluator._target_}>")
        audio_evaluator = hydra.utils.instantiate(cfg.audio_evaluator).to(device)
    else:
        audio_evaluator = None

    if cfg.visual_evaluator:
        log.info(f"Instantiating visual evaluator <{cfg.visual_evaluator._target_}>")
        visual_evaluator = hydra.utils.instantiate(cfg.visual_evaluator).to(device)
    else:
        visual_evaluator = None

    log.info(f"Starting sampling!")
    clip_scores = []
    clap_scores = []
    
    for idx in tqdm(range(cfg.trainer.num_samples), desc='Sampling'):     
        clip_score, clap_score = denoise(cfg, image_diffusion_guidance, audio_diffusion_guidance, scheduler, latent_transformation, visual_evaluator, audio_evaluator, views, idx, device)
        clip_scores.append(clip_score)
        clap_scores.append(clap_score)

    # re-ranking by metrics
    enable_rank = cfg.trainer.get("enable_rank", False)
    if enable_rank:
        log.info(f"Starting re-ranking and selection!")
        select_top_k_ranking(cfg, clip_scores, clap_scores)
    
    enable_clip_rank = cfg.trainer.get("enable_clip_rank", False)
    if enable_clip_rank:
        log.info(f"Starting re-ranking and selection by CLIP score!")
        select_top_k_clip_ranking(cfg, clip_scores)

    enable_clap_rank = cfg.trainer.get("enable_clap_rank", False)
    if enable_clap_rank:
        log.info(f"Starting re-ranking and selection by CLAP score!")
        select_top_k_clap_ranking(cfg, clap_scores)
        
    log.info(f"Finished!")


@torch.no_grad()
def denoise(cfg, image_diffusion, audio_diffusion, scheduler, latent_transformation, visual_evaluator, audio_evaluator, views, idx, device):
    image_guidance_scale, audio_guidance_scale = cfg.trainer.image_guidance_scale, cfg.trainer.audio_guidance_scale
    height, width = cfg.trainer.img_height, cfg.trainer.img_width
    image_start_step = cfg.trainer.get("image_start_step", 0)
    audio_start_step = cfg.trainer.get("audio_start_step", 0)
    audio_weight = cfg.trainer.get("audio_weight", 0.5)
    use_colormap = cfg.trainer.get("use_colormap", False)

    cutoff_latent = cfg.trainer.get("cutoff_latent", False)
    crop_image = cfg.trainer.get("crop_image", False)
    anagram_image_weight = cfg.trainer.get("anagram_image_weight", [0.5, 0.5])
    anagram_audio_weight = cfg.trainer.get("anagram_audio_weight", [0.5, 0.5])

    num_views = len(views)

    # obtain the text embeddings for each modality's diffusion process
    image_text_embeds = [encode_prompt(prompt, image_diffusion, device, negative_prompt=cfg.trainer.image_neg_prompt, time_repeat=1) for prompt in cfg.trainer.image_prompt]
    audio_text_embeds = [encode_prompt(prompt, audio_diffusion, device, negative_prompt=cfg.trainer.image_neg_prompt, time_repeat=1) for prompt in cfg.trainer.audio_prompt]

    scheduler.set_timesteps(cfg.trainer.num_inference_steps)

    # init random latents
    generator = torch.manual_seed(cfg.seed + idx)
    latents = torch.randn((image_text_embeds[0].shape[0] // 2, image_diffusion.unet.config.in_channels, height // 8, width // 8), dtype=image_diffusion.precision_t, generator=generator).to(device)

    for i, t in enumerate(scheduler.timesteps):
        # multiview image noise
        if i >= image_start_step: 
            viewed_noisy_images = []
            for embeds, view_fn, weight in zip(image_text_embeds, views, anagram_image_weight):
                latent = view_fn.view(latents[0])[None, ...]
                latent = estimate_noise(image_diffusion, latent, t, embeds, image_guidance_scale)[0]
                latent = view_fn.inverse_view(latent) * weight
                viewed_noisy_images.append(latent)
            image_noise = torch.stack(viewed_noisy_images)
        else: 
            image_noise = None
                
        # multiview audio noise
        if i >= audio_start_step:
            viewed_noisy_audio = []
            for embeds, view_fn, weight in zip(audio_text_embeds, views, anagram_audio_weight):
                latent = view_fn.view(latents[0])[None, ...]
                transform_latent = latent_transformation(latent, inverse=False)
                transform_latent = estimate_noise(audio_diffusion, transform_latent, t, embeds, audio_guidance_scale)
                latent = latent_transformation(transform_latent, inverse=True)[0]
                latent = view_fn.inverse_view(latent) * weight
                viewed_noisy_audio.append(latent)
            audio_noise = torch.stack(viewed_noisy_audio)
        else: 
            audio_noise = None
        
        if image_noise is not None and audio_noise is not None:
            noise_pred = (1.0 - audio_weight) * image_noise + audio_weight * audio_noise
            noise_pred = noise_pred.view(num_views, -1, image_diffusion.unet.config.in_channels, height // 8, width // 8)
            noise_pred = noise_pred.sum(0)
        elif image_noise is not None and audio_noise is None:
            noise_pred = image_noise.view(num_views, -1, image_diffusion.unet.config.in_channels, height // 8, width // 8)
            noise_pred = noise_pred.sum(0)
        elif image_noise is None and audio_noise is not None:
            noise_pred = audio_noise.view(num_views, -1, image_diffusion.unet.config.in_channels, height // 8, width // 8)
            noise_pred = noise_pred.sum(0)
        else: 
            log.info("No estimated noise! Exit.")
            raise NotImplementedError
        
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents)['prev_sample']

    if cutoff_latent and not crop_image:
        latents = latents[..., :-4] # we cut off 4 latents so that we can directly remove the black region
    
    # Img latents -> Viewed imgs
    img = image_diffusion.decode_latents(latents) # [1, 3, H, W]
    viewed_image = {}
    for view_name, view in zip(cfg.trainer.views, views):
        if not "pass" in view_name:
            viewed_image[view_name] = view.view(img[0])
        else:
            if "high" in view_name:
                viewed_image[view_name] = img[0]
            else:
                low_pass = get_views(["low_pass"], [10])[0]
                viewed_image[view_name] = low_pass.inverse_view(img[0].clone())
                viewed_image[view_name] = transforms.Resize((height // 8, width // 8))(viewed_image[view_name])
    
    audio_latents = latent_transformation(latents, inverse=False)
    spec = audio_diffusion.decode_latents(audio_latents).squeeze(0) # [3, 256, 1024]
    viewed_audio = {}
    for view_name, view in zip(cfg.trainer.views, views):
        if not "pass" in view_name:
            viewed_spec = view.view(spec)
        else:
            if "high" in view_name:
                viewed_spec = spec
            else:
                low_pass = get_views(["low_pass"], [10])[0]
                viewed_spec = low_pass.inverse_view(spec.clone())
        audio = audio_diffusion.spec_to_audio(viewed_spec)
        audio = np.ravel(audio)
        viewed_audio[view_name] = audio

    # if crop_image and not cutoff_latent:
    #     pixel = 32
    #     audio_length = int(pixel / width * audio.shape[0])
    #     img = img[..., :-pixel]
    #     spec = spec[..., :-pixel] 
    #     audio = audio[:-audio_length]   

    # evaluate with CLIP
    clip_score = 0
    if visual_evaluator is not None:
        for i, img in enumerate(viewed_image.values()):
            clip_score += visual_evaluator(img[None, ...], cfg.trainer.image_prompt[i])
    else:
        clip_score = None

    # evaluate with CLAP
    clap_score = 0
    if audio_evaluator is not None:
        for i, audio in enumerate(viewed_audio.values()):
            clap_score += audio_evaluator(cfg.trainer.audio_prompt[i], audio)
    else:
        clap_score = None

    sample_dir = os.path.join(cfg.output_dir, 'results', f'example_{str(idx+1).zfill(3)}')
    os.makedirs(sample_dir, exist_ok=True)

    # save config with example-specific information 
    cfg_save_path = os.path.join(sample_dir, 'config.yaml')
    current_cfg = copy.deepcopy(cfg)
    current_cfg.seed = cfg.seed + idx
    with open_dict(current_cfg):
        current_cfg.clip_score = clip_score
        current_cfg.clap_score = clap_score
    OmegaConf.save(current_cfg, cfg_save_path)
    
    # save image
    image_dir = os.path.join(sample_dir, 'image')
    os.makedirs(image_dir, exist_ok=True)
    for view_name in cfg.trainer.views:
        img_save_path = os.path.join(image_dir, f'{view_name}.png')
        save_image(viewed_image[view_name], img_save_path)

    # save audio
    audio_dir = os.path.join(sample_dir, 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    for view_name in cfg.trainer.views:
        audio_save_path = os.path.join(audio_dir, f'{view_name}.wav')
        save_audio(viewed_audio[view_name], audio_save_path)

    # save spec
    spec_save_path = os.path.join(sample_dir, f'spec.png')
    save_image(spec.mean(dim=0, keepdim=True), spec_save_path)

    # save spec with colormap (renormalize the spectrogram range)
    if use_colormap:
        spec_save_path = os.path.join(sample_dir, f'spec_colormap.png')
        spec_colormap = spec.mean(dim=0).cpu().numpy()
        plt.imsave(spec_save_path, spec_colormap, cmap='gray')

    # save video 
    # video_output_path = os.path.join(sample_dir, f'video.mp4')
    # if img.shape[-2:] == spec.shape[-2:]:
    #     create_single_image_animation_with_text(spec_save_path, audio_save_path, video_output_path, cfg.trainer.image_prompt, cfg.trainer.audio_prompt)
    # else:
    #     create_animation_with_text(img_save_path, spec_save_path, audio_save_path, video_output_path, cfg.trainer.image_prompt, cfg.trainer.audio_prompt)
    
    return clip_score, clap_score


if __name__ == "__main__":
    main()
