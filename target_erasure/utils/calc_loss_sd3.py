# coding: UTF-8
"""
    @func: loss + bi-level + InfoNCE  (SD3 / SD3.5 version)
    - Keep Flux calc_loss.py structure as much as possible.
"""

import random
import torch
import torch.nn.functional as F

from .esd_utils_sd3 import latent_sample_sd3, predict_noise_sd3
from .infoNCE import calculate_steer_loss


# ============================================================
# Small helpers (minimal + Flux-like)
# ============================================================

def _unpack_text_embeds(ret):
    """
    compute_text_embeddings(...) may return:
      (emb, pooled) or (emb, pooled, ids)
    SD3 uses only (emb, pooled).
    """
    if isinstance(ret, (list, tuple)) and len(ret) >= 2:
        emb, pooled = ret[0], ret[1]
        return emb, pooled
    raise ValueError("compute_text_embeddings must return (emb, pooled) or (emb, pooled, ids)")


def normalize_remove_indices(remove_indices, device, dtype=torch.long) -> torch.Tensor:
    if remove_indices is None:
        return torch.empty(0, device=device, dtype=dtype)

    if isinstance(remove_indices, torch.Tensor):
        return remove_indices.to(device=device, dtype=dtype).view(-1)

    if isinstance(remove_indices, (list, tuple)):
        # common: [[...]] from dataloader
        if len(remove_indices) == 1 and isinstance(remove_indices[0], (list, tuple, torch.Tensor)):
            remove_indices = remove_indices[0]
        if isinstance(remove_indices, torch.Tensor):
            return remove_indices.to(device=device, dtype=dtype).view(-1)
        return torch.tensor(list(remove_indices), device=device, dtype=dtype).view(-1)

    return torch.tensor(remove_indices, device=device, dtype=dtype).view(-1)


def _normalize_timesteps_for_sd3(timesteps: torch.Tensor, device: torch.device) -> torch.Tensor:
    t = timesteps.to(device=device)
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        t = t.float()
    return t.to(dtype=torch.float32)


def _infer_L_img_from_latents(transformer, latents: torch.Tensor) -> int:
    patch = int(getattr(transformer.config, "patch_size", 2))
    H = int(latents.shape[-2])
    W = int(latents.shape[-1])
    return (H // patch) * (W // patch)


def _pick_attn_tensor(attn_maps_list_or_tensor):
    """
    Your SD3 transformer returns (pred, attn_maps_list).
    In latent_sample_sd3(return_attn=True), we store per-step "attn_maps_list".
    This helper takes:
      - list[Tensor] (per-layer) -> pick last layer
      - Tensor -> use it directly
    """
    if isinstance(attn_maps_list_or_tensor, (list, tuple)):
        if len(attn_maps_list_or_tensor) == 0:
            raise RuntimeError("attn_maps_list is empty.")
        return attn_maps_list_or_tensor[-1]
    if torch.is_tensor(attn_maps_list_or_tensor):
        return attn_maps_list_or_tensor
    raise TypeError(f"Unknown attn container type: {type(attn_maps_list_or_tensor)}")


def _extract_cross_img_to_txt(attn_map: torch.Tensor, L_img: int) -> torch.Tensor:
    """
    attn_map: (B,H,L_total,L_total) or (B,H,Q,K).
    joint order assumed: [image_tokens, text_tokens] for both Q and K.
    return: (B,H,L_img,L_txt)
    """
    if attn_map.dim() != 4:
        raise ValueError(f"attn_map must be 4D (B,H,*,*), got {tuple(attn_map.shape)}")

    Lq = attn_map.shape[-2]
    Lk = attn_map.shape[-1]

    # already cross (B,H,L_img,L_txt)
    if Lq == L_img and Lk != L_img:
        return attn_map

    if Lq < L_img:
        raise ValueError(f"Cannot extract cross: Lq={Lq} < L_img={L_img}")

    L_txt = Lk - L_img
    if L_txt <= 0:
        raise ValueError(f"Cannot extract cross: inferred L_txt={L_txt} (Lk={Lk}, L_img={L_img})")

    return attn_map[..., :L_img, L_img:]


def _attn_deactivate_loss(attn_img_txt: torch.Tensor, remove_indices: torch.Tensor) -> torch.Tensor:
    """
    Flux-style: build mask on text dim and penalize.
    attn_img_txt: (B,H,L_img,L_txt)
    remove_indices: (n_remove,)
    """
    if remove_indices.numel() == 0:
        return attn_img_txt.new_zeros(())

    L_txt = attn_img_txt.shape[-1]
    idx = remove_indices[remove_indices < L_txt]
    if idx.numel() == 0:
        return attn_img_txt.new_zeros(())

    mask = torch.zeros_like(attn_img_txt)
    mask[..., idx] = 1.0
    return torch.norm(mask * attn_img_txt, dim=(0, 1)).sum()


def _set_train_timesteps_if_needed(noise_scheduler, device, n_train: int = 1000):
    """
    Flux calc_loss assumes 1000-step DDPM indexing.
    For SD3 schedulers, we try to create a similar setup.
    """
    if hasattr(noise_scheduler, "set_train_timesteps"):
        noise_scheduler.set_train_timesteps(n_train, device=device)
    elif hasattr(noise_scheduler, "set_timesteps"):
        noise_scheduler.set_timesteps(n_train, device=device)
    else:
        # if scheduler doesn't support setting, we just assume .timesteps exists
        pass


def _sample_t_enc_ddpm_like_flux(noise_scheduler, device, ddim_steps: int = 28, n_train: int = 1000) -> torch.Tensor:
    """
    Keep Flux logic:
      t_enc ~ U[0, ddim_steps)
      map to ddpm-bin in [0, n_train)
      pick one index in that bin
      then convert to scheduler timestep value
    """
    _set_train_timesteps_if_needed(noise_scheduler, device, n_train=n_train)

    # Flux: t_enc is in [0, ddim_steps)
    t_enc = torch.randint(ddim_steps, (1,), device=device)

    og_low = round((int(t_enc) / ddim_steps) * n_train)
    og_high = round((int(t_enc + 1) / ddim_steps) * n_train)
    if og_high <= og_low:
        og_high = min(og_low + 1, n_train)

    # choose an index in [og_low, og_high)
    step_idx = torch.randint(og_low, og_high, (1,), device=device).item()

    # map index -> actual scheduler timestep value if available
    if hasattr(noise_scheduler, "timesteps") and noise_scheduler.timesteps is not None:
        # timesteps typically length n_train, often descending ints
        t_val = noise_scheduler.timesteps[step_idx].to(device)
    else:
        # fallback: use index itself (works for some schedulers)
        t_val = torch.tensor([step_idx], device=device).view(())

    return t_val  # scalar tensor


# ============================================================
# 1) calculate_loss (Flux-style) for SD3
# ============================================================

def calculate_loss(
    args,
    batch,
    compute_text_embeddings,
    text_encoders,
    tokenizers,
    transformer,
    noise_scheduler,
    prompts,
    vae,
    criteria,
    negative_guidance,
    weight_dtype,
    start_guidance=3,
    ddim_steps=28,
    lamb1=1,
    lamb2=1,
    lamb3=0.2,
    opt_name="ESD",
    joint_attention_kwargs=None,
):
    """
    Flux-style:
      - prepare latents from pixel_values
      - sample t_enc_ddpm
      - (ESD) do latent_sample -> e0, ep -> en -> loss_esd
      - (ESD+) additionally:
          * attention deactivation loss using remove_indices
          * flow-matching loss (LoRA) target = noise - model_input
    """
    device = transformer.device

    # ---- 0) VAE latent prep (same idea as Flux)
    pixel_values = batch["pixel_values"].to(dtype=vae.dtype, device=device)
    model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
    model_input = model_input.to(dtype=weight_dtype)

    if opt_name == "ESD":
        print("OPT Name", opt_name)

    # ---- 1) embeddings (Flux: emb_0/emb_p)
    emb_0, pooled_0 = _unpack_text_embeds(compute_text_embeddings("", text_encoders, tokenizers))
    emb_p, pooled_p = _unpack_text_embeds(compute_text_embeddings(prompts, text_encoders, tokenizers))

    # ---- 2) timestep sample (Flux-like binning)
    # prefer scheduler's train timesteps if known, else 1000
    n_train = int(getattr(getattr(noise_scheduler, "config", None), "num_train_timesteps", 1000))
    t_enc_ddpm = _sample_t_enc_ddpm_like_flux(noise_scheduler, device=device, ddim_steps=ddim_steps, n_train=n_train)
    t_enc_ddpm_b = t_enc_ddpm.expand(model_input.shape[0])  # (B,)

    # ---- 3) (ESD) sample z then compute e0/ep, then en
    # SD3: no txt_ids/img_ids, no pack/unpack, no guidance embeds here.
    with torch.no_grad():
        # generate z from prompt (like Flux latent_sample)
        # NOTE: latent_sample_sd3 returns final latents (B,C,H/8,W/8); we keep it as "z"
        z = latent_sample_sd3(
            transformer=transformer,
            scheduler=noise_scheduler,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=512,
            width=512,
            prompt_embeds=emb_p.to(device),
            pooled_prompt_embeds=pooled_p.to(device),
            guidance_scale=None,
            num_inference_steps=int(ddim_steps),
            latents=None,
            return_attn=False,
            dtype=weight_dtype,
        )

        # e_0 & e_p at t_enc_ddpm on z
        e_0 = predict_noise_sd3(
            transformer, z, emb_0, pooled_0, t_enc_ddpm_b, dtype=weight_dtype
        )
        e_p = predict_noise_sd3(
            transformer, z, emb_p, pooled_p, t_enc_ddpm_b, dtype=weight_dtype
        )

    # student pred (LoRA-applied transformer) at same point
    e_n = predict_noise_sd3(
        transformer, z, emb_p, pooled_p, t_enc_ddpm_b, dtype=weight_dtype
    )

    e_0.requires_grad = False
    e_p.requires_grad = False

    total_loss = []

    loss_esd = criteria(
        e_n,
        e_0 - (negative_guidance * (e_p - e_0))
    )
    total_loss.append(lamb1 * loss_esd)

    # ---- 4) (ESD+) attention deactivation + flow matching (Flux-style)
    if opt_name == "ESD+":
        # Add noise (Flux does this on model_input, not on z)
        noise = torch.randn_like(model_input)
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, t_enc_ddpm_b)

        # forward w/ attentions (your SD3 transformer modification)
        model_dtype = next(transformer.parameters()).dtype
        ja = dict(joint_attention_kwargs) if joint_attention_kwargs is not None else {}
        ja["output_attentions"] = True

        pred, attn_maps_list = transformer(
            hidden_states=noisy_model_input.to(device=device, dtype=model_dtype),
            timestep=_normalize_timesteps_for_sd3(t_enc_ddpm_b, device=device),
            encoder_hidden_states=emb_p.to(device=device, dtype=model_dtype),
            pooled_projections=pooled_p.to(device=device, dtype=model_dtype),
            joint_attention_kwargs=ja if ja else None,
            return_dict=False,
        )

        # pick last layer and extract img->txt cross
        attn_map = _pick_attn_tensor(attn_maps_list)
        L_img = _infer_L_img_from_latents(transformer, noisy_model_input)
        attn_img_txt = _extract_cross_img_to_txt(attn_map, L_img=L_img)

        # remove_indices
        remove_indices = normalize_remove_indices(batch.get("remove_indices", None), device=device)

        # attention loss (Flux style)
        loss_attn = _attn_deactivate_loss(attn_img_txt, remove_indices)
        total_loss.append(lamb2 * loss_attn)

        # flow matching loss (Flux: target = noise - model_input)
        target = noise - model_input
        loss_lora = torch.mean(
            ((pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )[0]
        total_loss.append(lamb3 * loss_lora)

    return total_loss


# ============================================================
# 2) upper_ca_loss (Flux-style) for SD3
# ============================================================

def calculate_upper_ca_loss(
    args,
    batch,
    compute_text_embeddings,
    text_encoders,
    tokenizers,
    transformer,
    noise_scheduler,
    prompts,  # keep signature
    vae,
    weight_dtype,
    ca_prompt_p,
    ca_prompt_0,
    start_guidance=3,
    ddim_steps=28,
    lamb1=1,
    lamb2=1,
    joint_attention_kwargs=None,
):
    """
    Flux calculate_upper_ca_loss 형태 유지:
      - same noisy_model_input and timestep
      - pred(prompt_p) vs pred(prompt_0)
      - attention deactivation loss (from prompt_p forward)
    """
    device = transformer.device

    pixel_values = batch["pixel_values"].to(dtype=vae.dtype, device=device)
    model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
    model_input = model_input.to(dtype=weight_dtype)

    emb_0, pooled_0 = _unpack_text_embeds(compute_text_embeddings(ca_prompt_0, text_encoders, tokenizers))
    emb_p, pooled_p = _unpack_text_embeds(compute_text_embeddings(ca_prompt_p, text_encoders, tokenizers))

    n_train = int(getattr(getattr(noise_scheduler, "config", None), "num_train_timesteps", 1000))
    t_enc_ddpm = _sample_t_enc_ddpm_like_flux(noise_scheduler, device=device, ddim_steps=ddim_steps, n_train=n_train)
    t_b = t_enc_ddpm.expand(model_input.shape[0])

    noise = torch.randn_like(model_input)
    noisy_model_input = noise_scheduler.add_noise(model_input, noise, t_b)

    model_dtype = next(transformer.parameters()).dtype
    ja = dict(joint_attention_kwargs) if joint_attention_kwargs is not None else {}
    ja["output_attentions"] = True

    pred_p, attn_maps_list = transformer(
        hidden_states=noisy_model_input.to(device=device, dtype=model_dtype),
        timestep=_normalize_timesteps_for_sd3(t_b, device=device),
        encoder_hidden_states=emb_p.to(device=device, dtype=model_dtype),
        pooled_projections=pooled_p.to(device=device, dtype=model_dtype),
        joint_attention_kwargs=ja if ja else None,
        return_dict=False,
    )

    with torch.no_grad():
        pred_0 = predict_noise_sd3(
            transformer, noisy_model_input, emb_0, pooled_0, t_b, dtype=weight_dtype
        )

    total_loss = []

    loss_ca = lamb1 * torch.mean(
        ((pred_p.float() - pred_0.float()) ** 2).reshape(pred_0.shape[0], -1),
        1,
    )[0]
    total_loss.append(loss_ca)

    # attention deactivation
    remove_indices = normalize_remove_indices(batch.get("remove_indices", None), device=device)
    attn_map = _pick_attn_tensor(attn_maps_list)
    L_img = _infer_L_img_from_latents(transformer, noisy_model_input)
    attn_img_txt = _extract_cross_img_to_txt(attn_map, L_img=L_img)

    loss_attn = _attn_deactivate_loss(attn_img_txt, remove_indices)
    total_loss.append(lamb2 * loss_attn)

    return total_loss, t_enc_ddpm


# ============================================================
# 3) upper_loss (Flux-style) for SD3
# ============================================================

def calculate_upper_loss(
    args,
    batch,
    compute_text_embeddings,
    text_encoders,
    tokenizers,
    transformer,
    noise_scheduler,
    prompts,
    vae,
    criteria,
    negative_guidance,
    weight_dtype,
    neg_prompts,
    start_guidance=3,
    ddim_steps=28,
    lamb1=1,
    lamb2=1,
    joint_attention_kwargs=None,
):
    """
    Flux calculate_upper_loss 형태 유지:
      - ESD loss (via latent_sample + predict_noise)
      - attention deactivation (via noisy_model_input forward)
    """
    device = transformer.device

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    pixel_values = batch["pixel_values"].to(dtype=vae.dtype, device=device)
    # torch.Size([1, 3, 512, 512])
    model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    # torch.Size([1, 16, 64, 64])

    emb_0, pooled_0 = _unpack_text_embeds(compute_text_embeddings(neg_prompts, text_encoders, tokenizers))
    # torch.Size([1, 333, 4096]), torch.Size([1, 2048])
    emb_p, pooled_p = _unpack_text_embeds(compute_text_embeddings(prompts, text_encoders, tokenizers))

    # (ESD) ddim_steps
    n_train = int(getattr(getattr(noise_scheduler, "config", None), "num_train_timesteps", 1000))
    t_enc_ddpm = _sample_t_enc_ddpm_like_flux(noise_scheduler, device=device, ddim_steps=ddim_steps, n_train=n_train)
    t_b = t_enc_ddpm.expand(model_input.shape[0])

    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))

    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    # ---- ESD part (Flux-style: sample z, eval e0/ep, then en)
    with torch.no_grad():
        z = latent_sample_sd3(
            transformer=transformer,
            scheduler=noise_scheduler,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=512,
            width=512,
            prompt_embeds=emb_p.to(device),
            pooled_prompt_embeds=pooled_p.to(device),
            guidance_scale=start_guidance,
            neg_prompt_embeds=emb_0.to(device),
            neg_pooled_prompt_embeds=pooled_0.to(device),
            timesteps=int(ddim_steps),
            latents=None,
            return_attn=False,
            dtype=weight_dtype,
        )
        # torch.Size([1, 16, 64, 64])
        e_0 = predict_noise_sd3(transformer, z, emb_0, pooled_0, t_b, dtype=weight_dtype, guidance_scale=None)
        e_p = predict_noise_sd3(transformer, z, emb_p, pooled_p, t_b, dtype=weight_dtype, guidance_scale=start_guidance, neg_prompt_embeds=emb_0, neg_pooled_prompt_embeds=pooled_0)

    e_n = predict_noise_sd3(transformer, z, emb_p, pooled_p, t_b, dtype=weight_dtype, guidance_scale=start_guidance, neg_prompt_embeds=emb_0, neg_pooled_prompt_embeds=pooled_0)

    e_0.requires_grad = False
    e_p.requires_grad = False

    total_loss = []
    loss_esd = criteria(e_n.to(transformer.device), e_0.to(transformer.device) - (negative_guidance * (e_p.to(transformer.device) - e_0.to(transformer.device))))
    total_loss.append(lamb1 * loss_esd)

    # ---- attention deactivation (Flux-style on model_input)
    noise = torch.randn_like(model_input)
    noisy_model_input = noise_scheduler.add_noise(model_input, noise, t_b)

    model_dtype = next(transformer.parameters()).dtype
    ja = dict(joint_attention_kwargs) if joint_attention_kwargs is not None else {}
    ja["output_attentions"] = True

    _, attn_maps_list = transformer(
        hidden_states=noisy_model_input.to(device=device, dtype=model_dtype),
        timestep=_normalize_timesteps_for_sd3(t_b, device=device),
        encoder_hidden_states=emb_p.to(device=device, dtype=model_dtype),
        pooled_projections=pooled_p.to(device=device, dtype=model_dtype),
        joint_attention_kwargs=ja if ja else None,
        return_dict=False,
    )
    # torch.Size([1, 24, 1357, 1357]) * 24

    remove_indices = normalize_remove_indices(batch.get("remove_indices", None), device=device)
    attn_map = _pick_attn_tensor(attn_maps_list) # torch.Size([1, 24, 1357, 1357])
    L_img = _infer_L_img_from_latents(transformer, noisy_model_input)
    attn_img_txt = _extract_cross_img_to_txt(attn_map, L_img=L_img)

    loss_attn = _attn_deactivate_loss(attn_img_txt, remove_indices)
    total_loss.append(lamb2 * loss_attn)

    return total_loss, t_enc_ddpm


# ============================================================
# 4) lower_loss (Flux-style) for SD3
# ============================================================

def calculate_lower_loss(
    args,
    batch,
    compute_text_embeddings,
    text_encoders,
    tokenizers,
    transformer,
    noise_scheduler,
    prompts,
    neg_prompts,
    vae,
    weight_dtype,
    t_enc_ddpm,
    start_guidance=3,
    ddim_steps=28,
    K=3,
    ir_concept_lst=[],
    joint_attention_kwargs=None,
):
    """
    Flux calculate_lower_loss 형태 유지:
      - flow matching loss: MSE(pred, target=noise-model_input)
      - InfoNCE: use step-wise attn maps from latent_sample_sd3(return_attn=True)
        with same start_code for center/neg/pos
    """
    device = transformer.device

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))

    pixel_values = batch["pixel_values"].to(dtype=vae.dtype, device=device)
    model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)

    # normalize t_enc_ddpm to (B,)
    if isinstance(t_enc_ddpm, torch.Tensor):
        if t_enc_ddpm.numel() == 1:
            t_b = t_enc_ddpm.to(device).expand(model_input.shape[0])
        else:
            t_b = t_enc_ddpm.to(device)
    else:
        t_b = torch.tensor([t_enc_ddpm], device=device).expand(model_input.shape[0])

    noise = torch.randn_like(model_input)
    noisy_model_input = noise_scheduler.add_noise(model_input, noise, t_b)
    
    emb_0, pooled_0 = _unpack_text_embeds(compute_text_embeddings(neg_prompts, text_encoders, tokenizers))
    emb_p, pooled_p = _unpack_text_embeds(compute_text_embeddings(prompts, text_encoders, tokenizers))

    # ---- flow matching pred (pred only)
    pred, attn_maps = predict_noise_sd3(
        transformer, 
        noisy_model_input, 
        emb_p, 
        pooled_p, 
        t_b, 
        dtype=weight_dtype, 
        guidance_scale=3.5,
        neg_prompt_embeds=emb_0,
        neg_pooled_prompt_embeds=pooled_0,
        return_attn=True
        )
    # torch.Size([1, 16, 64, 64])

    target = noise - model_input
    loss_lora = torch.mean(
        ((pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        1,
    )[0]

    total_loss = [loss_lora]

    # ---- InfoNCE (Flux-style)
    if len(ir_concept_lst) != K:
        raise Exception("Please check ir_concept_lst length == K")

    # choose random step among ddim_steps
    attn_map_rand_idx = random.randint(0, int(ddim_steps) - 1)

    # same start latent code for fair comparison (Flux does this)
    start_code = torch.randn_like(model_input)

    # negative sample (synonym)
    emb_neg, pooled_neg = _unpack_text_embeds(compute_text_embeddings(batch["synonym_words"], text_encoders, tokenizers))
    _, attn_steps_neg = latent_sample_sd3(
        transformer=transformer,
        scheduler=noise_scheduler,
        batch_size=model_input.shape[0],
        num_channels_latents=model_input.shape[1],
        height=512,
        width=512,
        prompt_embeds=emb_neg.to(device),
        pooled_prompt_embeds=pooled_neg.to(device),
        guidance_scale=None,
        timesteps=int(ddim_steps),
        latents=start_code,
        return_attn=True,
        dtype=weight_dtype,
    )
    attn_item_neg = attn_steps_neg[attn_map_rand_idx]
    attn_map_neg_full = _pick_attn_tensor(attn_item_neg)

    # positives (irrelevant concepts)
    pos_maps_full = []
    for idx in range(K):
        emb_pos, pooled_pos = _unpack_text_embeds(compute_text_embeddings(ir_concept_lst[idx], text_encoders, tokenizers))
        _, attn_steps_pos = latent_sample_sd3(
            transformer=transformer,
            scheduler=noise_scheduler,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=512,
            width=512,
            prompt_embeds=emb_pos.to(device),
            pooled_prompt_embeds=pooled_pos.to(device),
            guidance_scale=start_guidance,
            neg_prompt_embeds=emb_neg,
            neg_pooled_prompt_embeds=pooled_neg,
            timesteps=int(ddim_steps),
            latents=start_code,
            return_attn=True,
            dtype=weight_dtype,
        )
        attn_item_pos = attn_steps_pos[attn_map_rand_idx]
        pos_maps_full.append(_pick_attn_tensor(attn_item_pos))

    # center: get attn at same step for current prompt
    # _, attn_steps_center = latent_sample_sd3(
    #     transformer=transformer,
    #     scheduler=noise_scheduler,
    #     batch_size=model_input.shape[0],
    #     num_channels_latents=model_input.shape[1],
    #     height=512,
    #     width=512,
    #     prompt_embeds=emb_p.to(device),
    #     pooled_prompt_embeds=pooled_p.to(device),
    #     guidance_scale=None,
    #     num_inference_steps=int(ddim_steps),
    #     latents=start_code,
    #     return_attn=True,
    #     dtype=weight_dtype,
    # )
    
    attn_map_center_full = _pick_attn_tensor(attn_maps)

    # ---- Convert full joint attn -> cross img->txt, then select remove_indices like Flux
    remove_indices = normalize_remove_indices(batch.get("remove_indices", None), device=device)

    # infer L_img from start_code (latent space)
    L_img = _infer_L_img_from_latents(transformer, start_code)

    attn_neg = _extract_cross_img_to_txt(attn_map_neg_full, L_img=L_img)       # (B,H,L_img,L_txt)
    attn_center = _extract_cross_img_to_txt(attn_map_center_full, L_img=L_img)
    attn_pos_list = [_extract_cross_img_to_txt(a, L_img=L_img) for a in pos_maps_full]

    # Flux: info = attn[..., remove_indices][:,0].permute(0,2,1)
    # Here: we mimic with cross (B,H,L_img,L_txt) -> select remove along L_txt
    def _to_info(attn_img_txt: torch.Tensor) -> torch.Tensor:
        # take head 0 -> (B, L_img, L_txt)
        a = attn_img_txt[:, 0]
        L_txt = a.shape[-1]
        idx = remove_indices[remove_indices < L_txt]
        if idx.numel() == 0:
            # (B,1,L_img) dummy
            return a.new_zeros(a.shape[0], 1, a.shape[1])
        a = a.index_select(dim=-1, index=idx)      # (B, L_img, n_remove)
        return a.permute(0, 2, 1).contiguous()     # (B, n_remove, L_img)
    
    info_neg = _to_info(attn_neg)
    info_center = _to_info(attn_center)
    info_pos_lst = [_to_info(a) for a in attn_pos_list]

    loss_contrastive = calculate_steer_loss(
        info_center,
        info_neg,
        info_pos_lst,
        temperature=0.07,
    )
    total_loss.append(loss_contrastive)

    return total_loss