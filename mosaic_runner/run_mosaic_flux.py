import torch
from PIL import Image
import random 
import numpy as np
import yaml
import os
import pdb
import ast
from tqdm import tqdm
import re
import gc
import json
import copy
import math
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from itertools import combinations
import inspect

import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import FluxPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps



# -----------------------------
# Seed fix
# -----------------------------
def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# Encode Prompts
# -----------------------------
@torch.no_grad()
def encode_prompt(pipe: FluxPipeline, prompt: str, device: str):
    out = pipe.encode_prompt(prompt=prompt, prompt_2=None, device=device)
    return out[0], out[1], out[2]

# -----------------------------
# random noise를 만드는 함수
# -----------------------------
@torch.no_grad()
def prepare_random_latent_flux(
    pipe,
    batch_size: int,
    height: int,
    width: int,
    generator: torch.Generator,
    dtype=torch.float16,
    device="cuda",
):
    """
    Create random initial latent for Flux text-to-image generation.
    """
    num_channels_latents = pipe.transformer.config.in_channels // 4

    latents, latent_image_ids = pipe.prepare_latents(
        batch_size=batch_size,
        num_channels_latents=num_channels_latents,
        height=height,
        width=width,
        dtype=dtype,
        device=device,
        generator=generator,
        latents=None,   # 🔥 핵심: None → random noise
    )

    return latents, latent_image_ids

# for flux
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# -----------------------------
# timestep 준비
# -----------------------------
def prepare_flux_timesteps(
    pipe,
    scheduler,
    z0,
    T_steps: int,
):
    device = z0.device

    sigmas = np.linspace(1.0, 1.0 / T_steps, T_steps)

    image_seq_len = z0.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )

    timesteps, _ = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )

    return timesteps

# -----------------------
# Predict vector field
# -----------------------
@torch.no_grad()
def predict_v(
    pipe: FluxPipeline,
    latents: torch.Tensor,
    t: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    guidance_scale: float
    ):
    device = latents.device

    # pipeline: timestep = t.expand(B).to(latents.dtype)
    timestep = t.expand(latents.shape[0]).to(latents.dtype)

    # pipeline: guidance handling
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
    return noise_pred

# -----------------------------
# Noisy Latent 계산
# -----------------------------
@torch.no_grad()
def make_noisy_latent_flux(z0, sigma):
    """
    z0: initial noise latent (B, T, D)
    sigma: scalar tensor (noise level at timestep t)
    """
    dtype = z0.dtype
    eps = torch.randn_like(z0, dtype=dtype)
    zt = (1.0 - sigma).to(dtype) * z0 + sigma.to(dtype) * eps
    return zt

@torch.no_grad()
def make_noisy_latents_per_timestep(
    z0: torch.Tensor,
    scheduler,
    timesteps,
):
    """
    Returns:
        zt_list: List[Tensor]  (len=T_steps), each (B, 1024, 64)
    """
    zt_list = []

    for t in timesteps:
        scheduler._init_step_index(t)
        sigma = scheduler.sigmas[scheduler.step_index]

        eps = torch.randn_like(z0)   # ✅ timestep당 1번만 샘플
        zt = (1.0 - sigma) * z0 + sigma * eps

        zt_list.append(zt)

    return zt_list



# -----------------------------
# Inference 함수
# -----------------------------
def make_2x2_grid(img_list):
    """
    img_list: [img00, img01, img10, img11] (PIL Images)
    """
    w, h = img_list[0].size
    grid = Image.new("RGB", (2 * w, 2 * h))

    grid.paste(img_list[0], (0, 0))       # base
    grid.paste(img_list[1], (w, 0))       # A
    grid.paste(img_list[2], (0, h))       # B
    grid.paste(img_list[3], (w, h))       # composed

    return grid


# -----------------------------
# Heatmap 함수
# -----------------------------
def vf_l2_diff_heatmap(v1, v2, H=32, W=32):
    diff = v1[0] - v2[0]          # (1024, 64)
    diff_norm = diff.norm(dim=-1) # (1024,)
    return diff_norm.view(H, W)


def vf_cosine_heatmap(v1, v2, H=32, W=32, eps=1e-8):
    a = v1[0]
    b = v2[0]
    cos = F.cosine_similarity(a, b, dim=-1, eps=eps)
    return cos.view(H, W)


def save_pairwise_vf_heatmaps(
    vector_fields: list,
    names: list,
    save_path: str = "results/vector_field_heatmap",
    H: int = 32,
    W: int = 32,
    mode: str = "l2",  # "l2" or "cosine"
    step_idx: int = 0,
):
    """
    Args:
        vector_fields: list of torch.Tensor
            each tensor shape: (1, 1024, 64)
        names: list of str, same length as vector_fields
        save_path: directory to save image
        mode: "l2" or "cosine"
        step_idx: diffusion step index
    """
    assert mode in ["l2", "cosine"]
    assert len(vector_fields) == len(names)
    assert len(vector_fields) >= 2

    os.makedirs(save_path, exist_ok=True)

    # 모든 pair 생성
    pairs = list(combinations(range(len(vector_fields)), 2))
    num_pairs = len(pairs)

    cols = len(vector_fields)
    rows = math.ceil(num_pairs / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 4 * rows),
        squeeze=False,
    )

    axes = axes.flatten()

    for ax, (i, j) in zip(axes, pairs):
        v1 = vector_fields[i]
        v2 = vector_fields[j]
        title = f"{names[i]} vs {names[j]}"

        if mode == "l2":
            diff = v1[0] - v2[0]
            hm = diff.norm(dim=-1).view(H, W)
            cmap = "inferno"
            label = "L2 diff"
            vmin, vmax = 0.0, 30.0   # ✅ 고정
        else:
            a = v1[0]
            b = v2[0]
            hm = F.cosine_similarity(a, b, dim=-1).view(H, W)
            cmap = "coolwarm"
            label = "cosine sim"
            vmin, vmax = -1.0, 1.0  # ✅ 고정

        im = ax.imshow(hm.detach().cpu(), 
                       cmap=cmap, 
                    #    vmin=vmin, 
                    #    vmax=vmax
                       )
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(label)

    # pair보다 subplot이 더 많은 경우, 남은 축은 제거
    for ax in axes[len(pairs):]:
        ax.axis("off")

    plt.tight_layout()
    save_file = os.path.join(save_path, f"{mode}_step{step_idx:03d}.png")
    plt.savefig(save_file, dpi=300)
    plt.close()

    print(f"[Saved] {save_file}")


def save_mask_image(mask_1024, save_path):
    """
    mask_1024: torch.Tensor, shape (1024,) or (1,1024)
               values in {0,1} or [0,1]
    """
    if mask_1024.dim() == 3:
        mask_1024 = mask_1024[0, :, 0]
    elif mask_1024.dim() == 2:
        mask_1024 = mask_1024[0]

    # (1024,) -> (32,32)
    mask_2d = mask_1024.view(32, 32)

    # to cpu numpy
    mask_np = mask_2d.detach().cpu().float().numpy()

    # [0,1] -> [0,255]
    mask_np = (mask_np * 255).astype("uint8")

    img = Image.fromarray(mask_np)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)


def save_vf_l2_heatmap(diff_1024, save_path: str, title: str = ""):
    """
    diff_1024: torch.Tensor, shape (1024,)
               raw L2 distance between base and one LoRA vector field.
    """
    heatmap = diff_1024.view(32, 32).detach().cpu().float().numpy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(heatmap, cmap="inferno")
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def normalize_quantile(
    x: torch.Tensor,
    q_low: float = 0.05,
    q_high: float = 0.95,
    eps: float = 1e-6,
):
    """
    x: (N,) or arbitrary shape
    returns: same shape, in [0,1]
    """
    x_f = x.float() 

    low = torch.quantile(x_f, q_low)
    high = torch.quantile(x_f, q_high)

    return ((x_f - low) / (high - low + eps)).clamp(0.0, 1.0).to(x.dtype)

# -----------------------------
# s-curve function
# -----------------------------
def s_curve_score(d, q_low=0.2, q_high=0.8, alpha=12.0):
    d_q = d.float()
    lo = torch.quantile(d_q, q_low)
    hi = torch.quantile(d_q, q_high)

    x = torch.clamp((d - lo) / (hi - lo + 1e-8), 0, 1)
    return torch.sigmoid(alpha * (x - 0.5))

def calculate_ema_beta(timestep, t1=825, t2=999, gamma=0.1):
    beta = torch.sigmoid(gamma * (timestep - t1)) - torch.sigmoid(gamma * (timestep - t2))
    # beta_soft = 0.85 * beta + 0.05
    return beta


# -----------------------------
# lora load & unload
# -----------------------------
def set_adapters_safe(pipe, names, weights=None):
    sig = inspect.signature(pipe.set_adapters)
    if weights is None:
        return pipe.set_adapters(names)
    if "adapter_weights" in sig.parameters:
        return pipe.set_adapters(names, adapter_weights=weights)
    if "weights" in sig.parameters:
        return pipe.set_adapters(names, weights=weights)
    return pipe.set_adapters(names, weights)

def onehot(K, i, val=1.0):
    w = [0.0]*K
    w[i] = float(val)
    return w

# -----------------------
# run_mosaic
# -----------------------
@torch.no_grad()
def run_mosaic(
    pipe,
    lora_paths: List[str],
    prompts: List[str],          # ✅ str -> List[str]
    height: int = 512,
    width: int = 512,
    T_steps: int = 28,
    guidance_scale: float = 5.5,
    max_sequence_length: int = 512,

    mask_type: str = "binary",
    scaling: bool = True,
    sigmoid: bool = False,
    adaptive: bool = False,
    do_ema: bool = True,

    mask_ema_start_step: int = 1,
    mask_ema_end_step: int = 28,
    ema_alpha: float = 0.9,

    mask_apply_start_step: int = 2,
    mask_apply_end_step: int = 28,

    l2_threshold: float = 10.0,
    q_low: float = 0.1,
    q_high: float = 0.9,
    continuous_mask_threshold: Optional[float] = None,

    seed: int = 42,
    device_base: str = "cuda:0",
    save_intermediate: bool = True,
    save_vf_heatmaps: bool = False,
    save_dir: str = "./results/loraflow",
    sample_ids: Optional[List[int]] = None,
):
    assert isinstance(prompts, (list, tuple)) and len(prompts) > 0
    B = len(prompts)
    K = len(lora_paths)
    if sample_ids is None:
        sample_ids = list(range(B))
    assert len(sample_ids) == B
    if continuous_mask_threshold is not None and not (0.0 <= continuous_mask_threshold <= 1.0):
        raise ValueError(
            f"continuous_mask_threshold must be in [0, 1], got {continuous_mask_threshold}"
        )

    adapter_names = [f"lora_{i}" for i in range(K)]
    for name, lp in zip(adapter_names, lora_paths):
        pipe.load_lora_weights(lp, adapter_name=name)

    os.makedirs(save_dir, exist_ok=True)

    # ✅ encode_prompt에 list 그대로 넣기
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompts,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=device_base,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )

    # ✅ latents batch_size = B
    generator = torch.Generator(device=device_base).manual_seed(seed)
    num_channels_latents = pipe.transformer.config.in_channels // 4
    z0, latent_image_ids = pipe.prepare_latents(
        B,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device_base,
        generator,
        None,
    )

    sigmas = np.linspace(1.0, 1.0 / T_steps, T_steps)
    image_seq_len = z0.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    timesteps, _ = retrieve_timesteps(pipe.scheduler, T_steps, device_base, sigmas=sigmas, mu=mu)

    sched_comp = copy.deepcopy(pipe.scheduler)
    latents_comp = z0.clone().half()

    prev_A = None   # binary: (K,B,1024)
    prev_d = None   # cont:   (K,B,1024)

    for step_idx, t in enumerate(timesteps):
        print(f"[Inference] step {step_idx}/{T_steps-1}")
        t = t.to(device_base, dtype=torch.float32)

        _do_ema = (mask_ema_start_step <= step_idx <= mask_ema_end_step) if do_ema else False
        do_apply = (mask_apply_start_step <= step_idx <= mask_apply_end_step)

        # =====================================================
        # ✅ base vf: LoRA OFF
        # =====================================================
        set_adapters_safe(pipe, adapter_names, [0.0] * K)
        sched_comp._init_step_index(t)
        v_base_c = predict_v(
            pipe,
            latents=latents_comp,
            t=t,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            latent_image_ids=latent_image_ids,
            guidance_scale=guidance_scale,
        )  # (B,1024,64)

        if do_apply:
            # lora vfs
            v_loras = []
            for i in range(K):
                set_adapters_safe(pipe, adapter_names, onehot(K, i, 1.0))
                v_k = predict_v(
                    pipe,
                    latents=latents_comp,
                    t=t,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    latent_image_ids=latent_image_ids,
                    guidance_scale=guidance_scale,
                )
                v_loras.append(v_k)

            # (선택) 다음 계산 전에 다시 all-off로 정리
            set_adapters_safe(pipe, adapter_names, [0.0] * K)

            # ✅ d: (K,B,1024)
            d = torch.stack([(v_base_c - v_k).norm(dim=-1) for v_k in v_loras], dim=0)

            if save_vf_heatmaps:
                for b in range(B):
                    gid = sample_ids[b]
                    step_dir = os.path.join(
                        save_dir,
                        f"{gid:04d}",
                        "vector_field_heatmaps",
                        f"step_{step_idx:03d}",
                    )
                    for k in range(K):
                        save_vf_l2_heatmap(
                            d[k, b],
                            os.path.join(step_dir, f"base_vs_k{k:02d}_l2.png"),
                            title=f"base vs k{k:02d} L2",
                        )

            if mask_type == "binary":
                # winners: (B,1024), best: (B,1024)
                winners = torch.argmax(d, dim=0)
                best = torch.max(d, dim=0).values

                mask_raw = torch.zeros_like(d, dtype=torch.float32)           # (K,B,1024)
                mask_raw.scatter_(0, winners.unsqueeze(0), 1.0)               # one-hot
                mask_raw = mask_raw * (best > l2_threshold).float().unsqueeze(0)

                if _do_ema and prev_A is not None:
                    mask = ema_alpha * prev_A + (1 - ema_alpha) * mask_raw
                else:
                    mask = mask_raw
                prev_A = mask

                mask_base = (1.0 - mask.sum(dim=0)).clamp(min=0.0)            # (B,1024)

                mask_base_ = mask_base.view(B, 1024, 1).to(v_base_c.dtype)     # (B,1024,1)
                mask_ = mask.permute(0,1,2).unsqueeze(-1).to(v_base_c.dtype)   # (K,B,1024,1)

            else:
                if _do_ema and prev_d is not None:
                    if adaptive:
                        beta = calculate_ema_beta(t, t1=850, t2=1025.0, gamma=0.05)
                        _ema_alpha = 1 - beta
                    else:
                        _ema_alpha = ema_alpha
                    d = _ema_alpha * prev_d + (1 - _ema_alpha) * d
                prev_d = d

                # normalize per (K,B) each token vector
                if sigmoid:
                    m = torch.stack([s_curve_score(d[k], q_low, q_high, 12.0) for k in range(K)], dim=0)  # (K,B,1024)
                else:
                    m = torch.stack([normalize_quantile(d[k], q_low, q_high) for k in range(K)], dim=0)   # (K,B,1024)

                if scaling:
                    # scale by relative magnitude among LoRAs: r_k = d_k / sum_j d_j
                    denom = d.sum(dim=0) + 1e-6  # (1024,)
                    r = d / denom.unsqueeze(0)   # (K,1024)
                    m = r * m
                    m_base = (1.0 - m.sum(dim=0)).clamp(min=0.0, max=1.0)
                else:
                    s = m.sum(dim=0)
                    m_base = (1.0 - s).clamp(min=0.0, max=1.0)

                    Z = m_base + s + 1e-9

                    m = m/Z.unsqueeze(0)
                    m_base = m_base / Z 

                if continuous_mask_threshold is not None:
                    m = (m >= continuous_mask_threshold).to(m.dtype)
                    m_base = (m.sum(dim=0) == 0).to(m.dtype)

                mask_base_ = m_base.view(B, 1024, 1).to(v_base_c.dtype)
                mask_ = m.unsqueeze(-1).to(v_base_c.dtype)        # (K,B,1024,1)
        else:
            v_loras = []
            mask_base_ = torch.ones((B, 1024, 1), device=v_base_c.device, dtype=v_base_c.dtype)
            mask_ = torch.zeros((K, B, 1024, 1), device=v_base_c.device, dtype=v_base_c.dtype)
        
        # =====================================================
        # ✅ save masks per step (B x (1+K))
        # =====================================================
        if save_intermediate:
            # mask_base_1d: (B,1024), mask_k_1d: (K,B,1024)

            for b in range(B):
                gid = sample_ids[b]  # ✅ global index (0000, 0001, ...)

                # ✅ 0000/intermediate_masks/step_XXX/
                step_dir = os.path.join(
                    save_dir,
                    f"{gid:04d}",
                    "intermediate_masks",
                    f"step_{step_idx:03d}",
                )
                os.makedirs(step_dir, exist_ok=True)

                # base mask 저장 (파일명 단순화)
                save_mask_image(
                    mask_base_[b].squeeze(-1),
                    os.path.join(step_dir, "base.png")
                )

                # lora masks 저장
                for k in range(K):
                    save_mask_image(
                        mask_[k, b].squeeze(-1),
                        os.path.join(step_dir, f"k{k:02d}.png")
                    )
        # apply
        if do_apply:
            v_comp = mask_base_ * v_base_c
            for k, v_k in enumerate(v_loras):
                v_comp = v_comp + mask_[k] * v_k
        else:
            v_comp = v_base_c

        latents_comp = sched_comp.step(v_comp, t, latents_comp).prev_sample

        # # ✅ intermediate 저장: batch 각각 저장
        # if save_intermediate:
        #     lat_img = pipe._unpack_latents(latents_comp, height, width, pipe.vae_scale_factor).to(pipe.vae.dtype)
        #     img = pipe.vae.decode(lat_img / pipe.vae.config.scaling_factor).sample
        #     imgs = pipe.image_processor.postprocess(img)  # list of PIL, len=B

        #     os.makedirs(f"{save_dir}/intermediate_images", exist_ok=True)
        #     for b, im in enumerate(imgs):
        #         im.save(f"{save_dir}/intermediate_images/step_{step_idx:03d}_b{b:02d}.png")

    # final decode (batch)
    lat = pipe._unpack_latents(latents_comp, height, width, pipe.vae_scale_factor).to(pipe.vae.dtype)
    img = pipe.vae.decode(lat / pipe.vae.config.scaling_factor).sample
    imgs = pipe.image_processor.postprocess(img)

    # (선택) 끝나고 LoRA 내리기
    pipe.unload_lora_weights()

    return imgs

# -----------------------------
# Lora마다의 Heatmap 시각화
# -----------------------------
@torch.no_grad()
def run_vf_heatmap_multi(
    model_id: str,
    Lora_A: str,
    Lora_B: str,
    prompt: str,
    batch_size: int = 1,
    height: int = 512,
    width: int = 512,
    T_steps: int = 28,
    guidance_scale: float = 5.5,
    # heatmap 설정
    save_dir: str = "results/vf_heatmaps",
    modes=("l2", "cosine"),
    # 어떤 step에서 저장할지
    save_steps: str = "all",   # "all" or "early" or "custom"
    custom_step_indices=None,  # e.g., [0,1,2,3,5,10]
    seed: int=42,
    device_base: str = "cuda:0",
    dtype=torch.float16,
):
    """
    base / loraA / loraB vector field를 동일 latents에서 예측하고,
    pairwise heatmap(L2 / cosine)을 step별로 저장.

    NOTE:
      - Single pipeline version (VRAM-safe):
        Base model stays on device_base.
        LoRA A/B are attached/detached only when needed.
      - device_A/device_B 인자는 시그니처 호환을 위해 남겨둠 (실제로는 사용 안 함)
    """

    os.makedirs(save_dir, exist_ok=True)

    device_base = torch.device(device_base)

    # -----------------------------
    # Load SINGLE pipeline (base GPU 고정)
    # -----------------------------
    pipe_base = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device_base)
    pipe_base.set_progress_bar_config(disable=True)

    # -----------------------------
    # Encode prompt (base에서 1번만)
    # -----------------------------
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe_base.encode_prompt(
        prompt=prompt, prompt_2=None, device=device_base
    )

    # -----------------------------
    # Initial latent (base에서 생성)
    # -----------------------------
    generator = torch.Generator(device=device_base).manual_seed(seed)
    z0, latent_image_ids = prepare_random_latent_flux(
        pipe_base,
        batch_size=batch_size,
        height=height,
        width=width,
        generator=generator,
        dtype=dtype,
        device=device_base,
    )

    # -----------------------------
    # Timesteps (base scheduler 기반)
    # -----------------------------
    timesteps = prepare_flux_timesteps(pipe_base, pipe_base.scheduler, z0, T_steps)

    # -----------------------------
    # 어떤 step에서 저장할지 결정
    # -----------------------------
    if save_steps == "all":
        step_indices = list(range(len(timesteps)))
    elif save_steps == "early":
        step_indices = list(range(min(8, len(timesteps))))
    elif save_steps == "custom":
        assert custom_step_indices is not None and len(custom_step_indices) > 0
        step_indices = list(custom_step_indices)
    else:
        raise ValueError("save_steps must be in {'all','early','custom'}")

    step_indices = [i for i in step_indices if 0 <= i < len(timesteps)]

    # -----------------------------
    # Probe latents
    # -----------------------------
    sched_probe = copy.deepcopy(pipe_base.scheduler)
    latents_probe = z0.clone()

    # guidance = torch.full((z0.shape[0],), guidance_scale, device=device_base)

    # -----------------------------
    # Loop
    # -----------------------------
    for step_idx, t in enumerate(timesteps):
        t_base = t.to(device_base) if torch.is_tensor(t) else torch.tensor(t, device=device_base)

        sched_probe._init_step_index(t_base)

        # ---- v_base (NO LoRA)
        v_base = predict_v(
            pipe_base,
            latents=latents_probe,
            t=t_base,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            latent_image_ids=latent_image_ids,
            guidance_scale=guidance_scale,
        )

        # ---- v_A (attach LoRA A -> predict -> detach)
        pipe_base.load_lora_weights(Lora_A)
        v_A = predict_v(
            pipe_base,
            latents=latents_probe,
            t=t_base,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            latent_image_ids=latent_image_ids,
            guidance_scale=guidance_scale,
        )
        pipe_base.unload_lora_weights()
        torch.cuda.empty_cache()

        # ---- v_B (attach LoRA B -> predict -> detach)
        pipe_base.load_lora_weights(Lora_B)
        v_B = predict_v(
            pipe_base,
            latents=latents_probe,
            t=t_base,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            latent_image_ids=latent_image_ids,
            guidance_scale=guidance_scale,
        )
        pipe_base.unload_lora_weights()
        torch.cuda.empty_cache()

        # ---- heatmap 저장 (원하면 step 일부만)
        if step_idx in step_indices:
            if "l2" in modes:
                save_pairwise_vf_heatmaps(
                    vector_fields=[v_base, v_A, v_B],
                    names=["base", "A", "B"],
                    save_path=save_dir,
                    step_idx=step_idx,
                    mode="l2",
                )
            if "cosine" in modes:
                save_pairwise_vf_heatmaps(
                    vector_fields=[v_base, v_A, v_B],
                    names=["base", "A", "B"],
                    save_path=save_dir,
                    step_idx=step_idx,
                    mode="cosine",
                )

    
        latents_probe = sched_probe.step(v_base, t_base, latents_probe).prev_sample
        lat_img = pipe_base._unpack_latents(
                latents_probe, height, width, pipe_base.vae_scale_factor
            ).to(pipe_base.vae.dtype)

        img = pipe_base.vae.decode(
            lat_img / pipe_base.vae.config.scaling_factor
        ).sample

        img = pipe_base.image_processor.postprocess(img)[0]
        os.makedirs(f"{save_dir}/intermediate_images", exist_ok=True)
        img.save(f"{save_dir}/intermediate_images/step_{step_idx:03d}.png")

    print(f"[Done] saved heatmaps to: {save_dir}")

###############################
# Lora path mapping
###############################
def build_concept_to_lora_map(
    lora_root: str,
    weight_filename: str = "pytorch_lora_weights.safetensors",
):
    """
    lora_root 아래를 훑어서
      concept_name -> weight_path
    매핑을 만든다.

    기대 구조 예:
      lora_root/
        checkpoint-Mario-200_original/pytorch_lora_weights.safetensors
        checkpoint-SpongeBob-200/pytorch_lora_weights.safetensors
        ...
    """
    concept2path = {}

    for root, dirs, files in os.walk(lora_root):
        if weight_filename in files:
            wpath = os.path.join(root, weight_filename)
            folder = os.path.basename(root)

            # 폴더명에서 concept 추출
            # checkpoint-Mario-200_original -> Mario
            concept = None
            if folder.startswith("checkpoint-"):
                rest = folder[len("checkpoint-"):]
                # Mario-200_original -> Mario
                concept = rest.split("-")[0].strip()

            if concept is None or concept == "":
                continue

            # 중복이면 가장 최근/짧은 경로 등 정책 선택 가능
            concept2path[concept] = wpath

    return concept2path

def parse_concept_key(key: str) -> List[str]:
    # "Mario + SpongeBob + Luigi" 같은 형태 지원
    parts = [p.strip() for p in key.split("+")]
    return [p for p in parts if p]



# -----------------------------
# Set up
# -----------------------------
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="FLUX FlowBlending (2-LoRA) with EMA masks")

    # required-ish
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--lora_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    # which keys to run
    parser.add_argument("--run_all_keys", action="store_true")
    parser.add_argument("--keys", nargs="+", default=None, help='e.g. "SpongeBob SquarePants + Mario"')
    parser.add_argument("--skip_missing_lora", action="store_true")

    # general generation
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--T_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--max_sequence_length", type=int, default=512)

    # mask type + params
    parser.add_argument("--mask_type", type=str, default="continuous", choices=["binary", "continuous"])
    parser.add_argument("--scaling", action="store_true", help="(kept for compatibility; continuous uses scaling in code)")
    parser.add_argument("--sigmoid", action="store_true")
    parser.add_argument("--adaptive", action="store_true")

    # EMA control (mask smoothing을 언제/어떻게 할지)
    parser.add_argument("--do_ema", action="store_true")
    parser.add_argument("--ema_alpha", type=float, default=0.9)
    parser.add_argument("--mask_ema_start_step", type=int, default=1)
    parser.add_argument("--mask_ema_end_step", type=int, default=28)

    # mask apply range (실제로 블렌딩을 언제부터 적용할지)
    parser.add_argument("--mask_apply_start_step", type=int, default=1)
    parser.add_argument("--mask_apply_end_step", type=int, default=28)

    # thresholds
    parser.add_argument("--l2_threshold", type=float, default=10.0)
    parser.add_argument("--q_low", type=float, default=0.1)
    parser.add_argument("--q_high", type=float, default=0.9)
    parser.add_argument(
        "--continuous_mask_threshold",
        type=float,
        default=None,
        help="If set with --mask_type continuous, binarize the continuous LoRA mask with m >= threshold.",
    )

    # outputs
    parser.add_argument("--save_intermediate", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # 1) load prompts json
    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2) build concept -> lora weight map
    concept2lora = build_concept_to_lora_map(args.lora_root)

    print(f"[LoRA] found {len(concept2lora)} concepts in: {args.lora_root}")
    for i, (k, v) in enumerate(concept2lora.items()):
        if i >= 10:
            break
        print(f"  - {k}: {v}")

    # 3) decide which keys to run
    all_keys = list(data.keys())
    if args.run_all_keys:
        keys_to_run = all_keys
    elif args.keys is not None and len(args.keys) > 0:
        keys_to_run = args.keys
    else:
        keys_to_run = all_keys

    # 4) run for each concept_key
    # =====================================================
    # 1) Load base model
    # =====================================================
    device_base = torch.device(args.device)
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device_base)
    pipe.set_progress_bar_config(disable=True)
    overall_pbar = tqdm(keys_to_run, desc="All concepts", unit="concept")
    for concept_key in overall_pbar:
        if concept_key not in data:
            print(f"[Skip] key not in json: {concept_key}")
            continue

        concepts = parse_concept_key(concept_key)  # e.g. ["Mario", "SpongeBob"]
        if len(concepts) < 2:
            print(f"[Skip] need >=2 concepts, got {concepts}")
            continue

        # (A) collect LoRA paths (MULTI)
        lora_paths = []
        missing = []

        for c in concepts:
            if c in concept2lora:
                lora_paths.append(concept2lora[c])
            else:
                missing.append(c)

        if len(missing) > 0:
            msg = f"[Missing LoRA] key={concept_key} missing={missing}"
            if args.skip_missing_lora:
                print(msg + " -> skip this key")
                continue
            else:
                raise KeyError(msg)

        print(f"\n[Run] key={concept_key} | loras={concepts}")
        overall_pbar.set_postfix_str(" + ".join(concepts))

        # (B) prompts
        prompt_list = data[concept_key]

        def chunk_list(lst, n):
            for start in range(0, len(lst), n):
                yield start, lst[start:start+n]
        
        n = max(1, len(prompt_list) // args.chunk_size)

        total_batches = math.ceil(len(prompt_list) / n)
        concept_desc = " + ".join(concepts)

        # (C) run prompts
        for start_idx, batch in tqdm(
            chunk_list(prompt_list, n=n),
            total=total_batches,
            desc=concept_desc,
            unit="batch",
            leave=False,
        ):
            out_dir = os.path.join(
                args.save_dir,
                f"seed_{args.seed}",
                " + ".join(concepts),
            )
            os.makedirs(out_dir, exist_ok=True)
            
            prompts_batch = [item["prompt"] for item in batch]
            
            prompts_batch = [item["prompt"] for item in batch]

            # 배치 결과 파일 경로들
            out_paths = []
            for b in range(len(prompts_batch)):
                global_idx = start_idx + b
                out_paths.append(os.path.join(out_dir, f"{global_idx:04d}", f"result_comp_{global_idx:04d}.png"))

            # 전부 존재하면 이 배치 스킵
            if all(os.path.exists(p) for p in out_paths):
                print(f"[Skip batch] already exists: {start_idx}~{start_idx+len(prompts_batch)-1}")
                continue

            sample_ids = [start_idx + b for b in range(len(prompts_batch))]

            imgs = run_mosaic(
                pipe=pipe,
                lora_paths=lora_paths,              # ✅ 여기 핵심
                prompts=prompts_batch,
                T_steps=args.T_steps,
                guidance_scale=args.guidance_scale,

                mask_type=args.mask_type,
                scaling=args.scaling,
                sigmoid=args.sigmoid,
                adaptive=args.adaptive,

                do_ema=args.do_ema,
                mask_ema_start_step=args.mask_ema_start_step,
                mask_ema_end_step=args.mask_ema_end_step,
                ema_alpha=args.ema_alpha,

                mask_apply_start_step=args.mask_apply_start_step,
                mask_apply_end_step=args.mask_apply_end_step,

                l2_threshold=args.l2_threshold,
                q_low=args.q_low,
                q_high=args.q_high,
                continuous_mask_threshold=args.continuous_mask_threshold,

                seed=args.seed,
                device_base=device_base,
                save_intermediate=args.save_intermediate,
                save_dir=out_dir, 
                sample_ids=sample_ids
            )
            
            for b, im in enumerate(imgs):
                gid = sample_ids[b]
                out_img_dir = os.path.join(out_dir, f"{gid:04d}")
                os.makedirs(out_img_dir, exist_ok=True)
                im.save(os.path.join(out_img_dir, f"result_comp_{gid:04d}.png"))


if __name__ == "__main__":
    import time

    start_time = time.time()

    main()

    elapsed = time.time() - start_time

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    print(f"\n⏱ Total execution time: {hours}h {minutes}m {seconds:.2f}s")
