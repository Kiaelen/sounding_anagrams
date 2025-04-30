# Sounding Anagrams

**Anagram + Sound** in a single image!

Adapted from [image-that-sound](https://github.com/IFICL/images-that-sound) and [visual anagrams](https://github.com/dangeng/visual_anagrams).
<hr>

## Method

We use the combined score 

$$\displaystyle \epsilon^t_{\text{combined}}(z_t)=W\epsilon^t_{\text{audio}}(z_t)+(1-W)\epsilon_{\text{image}}^t(z_t),$$

where

$$\epsilon^t_{\text{audio}}(z_t)=\sum_{v \in \text{views}}w^a_{v}v^{-1}(\epsilon_{\theta_{a}}(v(z_t),t,y^a_v)),$$
$$\epsilon^t_{\text{image}}(z_t)=\sum_{v \in \text{views}}w^i_{v}v^{-1}(\epsilon_{\theta_{i}}(v(z_t),t,y^i_v)),$$

s.t. $w^a_v=w^i_v=1, \forall v$ for gaussian-blur hybrids, and $\sum_v w^a_v=\sum w^i_v=1$ for other anagram types.

Notations:
- $z_t$ -> latent at timestep $t$
- $W$ -> audio weight of the image
- $w^i_v$ and $w^a_v$ -> anagram image/audio weight. Used to control visual/auditory emphasis on different views
- $y_v^i$ and $y_v^a$ -> image/audio prompts for different views
- $\epsilon_{\theta_s}$ -> audio denoisor
- $\epsilon_{\theta_i}$ -> image denoisor

This method works for all invertible views by the linearity of the denoising process.

## Results

Download **audio** results in [assets](https://github.com/Kiaelen/Sounding-Anagrams/tree/main/assets).

### Gaussian Filter Example

  **image_prompt:**
  - a castle with bell towers, grayscale
  - an antique model car, grayscale

**audio_prompt:** bird chirping

<div align="center">
  <img width="100%" src="assets\gaussian\car-castle-bird\view0.img.png">
  <img width="30%" src="assets\gaussian\car-castle-bird\view1.img.png">
</div>

### Patch Permute Example

  **image_prompt:**
  - painting of trees, lithograph style, grayscale
  - painting of castles, lithograph style, grayscale

**audio_prompt:** bell ringing

<div align="center">
  <img width="100%" src="assets\permute\tree-castle-bell\view0.img.png">
  <img width="100%" src="assets\permute\tree-castle-bell\view1.img.png">
</div>

### Patch Rotate Example

  **image_prompt:**
  - painting of a garden, lithograph style, grayscale
  - paiting of a helicopter, lithograph style, grayscale ~~(the result is actually a jet plane)~~

**audio_prompt:** bird chirping

<div align="center">
  <img width="100%" src="assets\rotate\trees-helicopter-bird\view0.img.png">
  <img width="100%" src="assets\rotate\trees-helicopter-bird\view1.img.png">
</div>

## Main paramters to tune in [config](https://github.com/Kiaelen/Sounding-Anagrams/blob/main/configs/main_denoise/main.yaml).trainer

<code> views </code> This is the anagram type you want to create. See /visual_anagrams/\_\_init\_\_.py for type list.

<code> anagram_balance_weight </code>

<code> image_prompt </code>

<code> audio_prompt </code>

<code> audio_weight </code>
<hr>

## Limitations

Good results extremely hard to obtain. Have to painstakingly tune hyperparameters for half an hour on average to obtain a relatively good sounding anagram.

VAE Decoder cannot preserve boundaries properly as can be seen in the patch permute examples.

Black strip identifid in generated images as Auffusion denoising results.

### Why not colorized

Colorization as post-processing tend to yield bad results, as shown in experiments, especially for the task of sounding-anagram generation that involves intricate semantics.

## How to run

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Cj_SOwWNtn4z0XPShqs0Qu-Ib4lMhRis?usp=sharing)
