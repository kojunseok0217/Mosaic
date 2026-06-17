# coding: UTF-8
"""
    @func: loss + bi-level + InfoNCE
"""

import random
import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from .esd_utils import latent_sample, predict_noise
from .infoNCE import calculate_steer_loss


def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(dtype=dtype) # [1000]
    schedule_timesteps = noise_scheduler.timesteps
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def calculate_loss(args, batch, compute_text_embeddings, text_encoders, tokenizers, transformer, noise_scheduler, prompts, vae, criteria, negative_guidance, weight_dtype, start_guidance=3, ddim_steps=28, lamb1=1, lamb2=1, lamb3=0.2, opt_name="ESD"):
    
    
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    # Convert images to latent space
    if args.cache_latents:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype).cuda()
        model_input = vae.encode(pixel_values).latent_dist.sample()

    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    if opt_name == "ESD":
        print("OPT Name", opt_name)
    # (ESD) get conditional embedding for the prompt
    emb_0, pooled_emb_0, text_ids_0 = compute_text_embeddings(
                "", text_encoders, tokenizers
            )
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
                prompts, text_encoders, tokenizers
            )

    # (ESD) ddim_steps
    t_enc = torch.randint(ddim_steps, (1,), device=transformer.device)
    # time step from 1000 to 0 (0 being good)
    og_num = round((int(t_enc)/ddim_steps)*1000)
    og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=transformer.device)

    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))

    # (ESD) start_guidance = 3
    start_guidance = 3
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    with torch.no_grad():
        # generate an image with the concept from ESD model
        z, latent_image_ids = latent_sample(transformer,
                                            noise_scheduler,
                                            1,
                                            model_input.shape[1], 
                                            512,
                                            512,
                                            emb_p.to(transformer.device),
                                            pooled_emb_p.to(transformer.device),
                                            text_ids_p.to(transformer.device),
                                            start_guidance, 
                                            int(ddim_steps),
                                            vae_scale_factor)
        # e_0 & e_p
        e_0 = predict_noise(transformer, z, emb_0, pooled_emb_0, text_ids_0, latent_image_ids, guidance=start_guidance, timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True)
        e_p = predict_noise(transformer, z, emb_p, pooled_emb_p, text_ids_p, latent_image_ids, guidance=start_guidance, timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True)

    # get conditional score from ESD model
    e_n = predict_noise(transformer, z, emb_p, pooled_emb_p, text_ids_p, latent_image_ids, guidance=start_guidance, timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True)
    e_0.requires_grad = False
    e_p.requires_grad = False
    
    total_loss = []
    
    loss_esd = criteria(e_n.to(transformer.device), e_0.to(transformer.device) - (negative_guidance*(e_p.to(transformer.device) - e_0.to(transformer.device))))
    
    total_loss.append(loss_esd)
    
    if opt_name == "ESD+":
        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                model_input.shape[0],
                model_input.shape[2] // 2,
                model_input.shape[3] // 2,
                transformer.device,
                weight_dtype,
            )
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        # u = compute_density_for_timestep_sampling(
        #     weighting_scheme=args.weighting_scheme,
        #     batch_size=bsz,
        #     logit_mean=args.logit_mean,
        #     logit_std=args.logit_std,
        #     mode_scale=args.mode_scale,
        # )
        # indices = (u * noise_scheduler.config.num_train_timesteps).long()
        # timesteps = noise_scheduler.timesteps[indices].to(device=transformer.device)
        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        noisy_model_input = noise_scheduler.add_noise(model_input,
                                                      noise,
                                                      t_enc_ddpm)
        
        packed_noisy_model_input = FluxPipeline._pack_latents(
                noisy_model_input,
                batch_size=model_input.shape[0],
                num_channels_latents=model_input.shape[1],
                height=model_input.shape[2],
                width=model_input.shape[3],
            )
        
        if transformer.config.guidance_embeds:
            guidance = torch.tensor([args.guidance_scale], device=transformer.device)
            guidance = guidance.expand(model_input.shape[0])
        else:
            guidance = None
        
        
        remove_indices = batch['remove_indices'][0]
        
        model_pred, attn_maps, attn_maps_single = transformer(
            hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
            timestep=t_enc_ddpm / 1000,
            guidance=guidance.to(dtype=weight_dtype, device=transformer.device),
            pooled_projections=pooled_emb_p.to(dtype=weight_dtype, device=transformer.device),
            encoder_hidden_states=emb_p.to(dtype=weight_dtype, device=transformer.device),
            txt_ids=text_ids_p.to(dtype=weight_dtype, device=transformer.device),
            img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
            return_dict=False,
        )[0:2]
        
        attn_map_mask = torch.ones_like(attn_maps).to(transformer.device)
        attn_map_mask[..., remove_indices] = 0
        attn_map_mask = 1 - attn_map_mask
                
        model_pred = FluxPipeline._unpack_latents(
            model_pred,
            height=model_input.shape[2] * vae_scale_factor,
            width=model_input.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )

        # flow matching loss
        target = noise - model_input
        
        # Compute regular loss.
        loss_attn = lamb2 * sum(torch.norm(attn_map_mask*attn_maps, dim=(0, 1))).sum()
        loss_lora = lamb3 * torch.mean(
            ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )[0]
        
        total_loss.append(loss_attn)
        total_loss.append(loss_lora)
            
    return total_loss


def calculate_upper_ca_loss(args, batch, compute_text_embeddings, text_encoders, tokenizers, transformer, noise_scheduler, prompts, vae, criteria, negative_guidance, weight_dtype, ca_prompt_p, ca_prompt_0, start_guidance=3, ddim_steps=28, lamb1=1, lamb2=1):
    """
        @date: 2024.12.14 
        @name: bi-level loss
        @func: replace esd with ca
    """
    
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    # Convert images to latent space
    if args.cache_latents:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype).cuda()
        model_input = vae.encode(pixel_values).latent_dist.sample()

    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    emb_0, pooled_emb_0, text_ids_0 = compute_text_embeddings(
                    ca_prompt_0, text_encoders, tokenizers
                )
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
                ca_prompt_p, text_encoders, tokenizers
        )

    t_enc = torch.randint(ddim_steps, (1,), device=transformer.device)
    og_num = round((int(t_enc)/ddim_steps)*1000)
    og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=transformer.device)

    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))

    start_guidance = 3
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
        model_input.shape[0],
        model_input.shape[2] // 2,
        model_input.shape[3] // 2,
        transformer.device,
        weight_dtype,
    )
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]

    noisy_model_input = noise_scheduler.add_noise(model_input,
                                                  noise,
                                                  t_enc_ddpm)

    packed_noisy_model_input = FluxPipeline._pack_latents(
            noisy_model_input,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3],
        )

    
    if transformer.config.guidance_embeds:
        guidance = torch.tensor([args.guidance_scale], device=transformer.device)
        guidance = guidance.expand(model_input.shape[0])
    else:
        guidance = None
    
    model_pred, attn_maps, attn_maps_single = transformer(
        hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
        timestep=t_enc_ddpm / 1000,
        guidance=guidance.to(dtype=weight_dtype, device=transformer.device),
        pooled_projections=pooled_emb_p.to(dtype=weight_dtype, device=transformer.device),
        encoder_hidden_states=emb_p.to(dtype=weight_dtype, device=transformer.device),
        txt_ids=text_ids_p.to(dtype=weight_dtype, device=transformer.device),
        img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
        return_dict=False,
    )[0:2]

    model_pred = FluxPipeline._unpack_latents(
        model_pred,
        height=model_input.shape[2] * vae_scale_factor,
        width=model_input.shape[3] * vae_scale_factor,
        vae_scale_factor=vae_scale_factor,
    )
    
    total_loss = []

    remove_indices = batch['remove_indices'][0]
    
    with torch.no_grad():
        model_pred_sg = transformer(
            hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
            timestep=t_enc_ddpm / 1000,
            guidance=guidance.to(dtype=weight_dtype, device=transformer.device),
            pooled_projections=pooled_emb_0.to(dtype=weight_dtype, device=transformer.device),
            encoder_hidden_states=emb_0.to(dtype=weight_dtype, device=transformer.device),
            txt_ids=text_ids_p.to(dtype=weight_dtype, device=transformer.device),
            img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
            return_dict=False,
        )[0]

        model_pred_sg = FluxPipeline._unpack_latents(
            model_pred_sg,
            height=model_input.shape[2] * vae_scale_factor,
            width=model_input.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )

    loss_ca = lamb1 * torch.mean(
        ((model_pred.float() - model_pred_sg.float()) ** 2).reshape(model_pred_sg.shape[0], -1),
        1,
    )[0]
    
    total_loss.append(loss_ca)

    attn_map_mask = torch.ones_like(attn_maps).to(transformer.device)
    attn_map_mask[..., remove_indices] = 0
    attn_map_mask = 1 - attn_map_mask

    # Compute regular loss.
    loss_attn = sum(torch.norm(attn_map_mask*attn_maps, dim=(0, 1))).sum()

    total_loss.append(loss_attn)
            
    return total_loss, t_enc_ddpm



def calculate_upper_loss(args, batch, compute_text_embeddings, text_encoders, tokenizers, transformer, noise_scheduler, prompts, vae, criteria, negative_guidance, weight_dtype, neg_prompts, start_guidance=3, ddim_steps=28, lamb1=1, lamb2=1):
    """
        @date: 2024.11.30 
        @name: bi-level loss
        @func: a) upper_loss: make sure samples from D_un(unlearning) is removed.
               b) lower_loss: make sure samples from D_ir(irrelevant) is perserved.
    """
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    # Convert images to latent space
    if args.cache_latents:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype, device = transformer.device)
        # torch.Size([1, 3, 512, 512])
        model_input = vae.encode(pixel_values).latent_dist.sample()

    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    # torch.Size([1, 16, 64, 64])
    
    # (ESD) get conditional embedding for the prompt
    emb_0, pooled_emb_0, text_ids_0 = compute_text_embeddings(
                neg_prompts, text_encoders, tokenizers
            )
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
                prompts, text_encoders, tokenizers
            )
    # torch.Size([1, 256, 4096]), torch.Size([1, 768]), torch.Size([1, 256, 3])

    # (ESD) ddim_steps
    t_enc = torch.randint(ddim_steps, (1,), device=transformer.device)
    # time step from 1000 to 0 (0 being good) 1로 시작
    og_num = round((int(t_enc)/ddim_steps)*1000) # 36
    og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=transformer.device)

    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))

    # (ESD) start_guidance = 3
    start_guidance = 3
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    with torch.no_grad():
        # generate an image with the concept from ESD model
        z, latent_image_ids = latent_sample(transformer,
                                            noise_scheduler,
                                            1,
                                            model_input.shape[1], 
                                            512,
                                            512,
                                            emb_p.to(transformer.device),
                                            pooled_emb_p.to(transformer.device),
                                            text_ids_p.to(transformer.device),
                                            start_guidance, 
                                            int(ddim_steps),
                                            vae_scale_factor)
        # torch.Size([1, 1024, 64])
        
        # Disable LoRA to get original model predictions (for Eq. 2)
        if hasattr(transformer, 'disable_adapter_layers'):
            transformer.disable_adapter_layers()
        
        # e_0 & e_p from original model (vθ₀)
        e_0 = predict_noise(transformer, z, emb_0, pooled_emb_0, text_ids_0, latent_image_ids, guidance=start_guidance, timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True)
        e_p = predict_noise(transformer, z, emb_p, pooled_emb_p, text_ids_p, latent_image_ids, guidance=start_guidance, timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True)
        
        # Re-enable LoRA
        if hasattr(transformer, 'enable_adapter_layers'):
            transformer.enable_adapter_layers()
        
        # torch.Size([1, 16, 64, 64])
    
    # get conditional score from LoRA model (vθ₀+∆θ)
    e_n = predict_noise(transformer, z, emb_p, pooled_emb_p, text_ids_p, latent_image_ids, guidance=start_guidance, timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True)
    e_0.requires_grad = False
    e_p.requires_grad = False
    
    total_loss = []
    
    loss_esd = criteria(e_n.to(transformer.device), e_0.to(transformer.device) - (negative_guidance*(e_p.to(transformer.device) - e_0.to(transformer.device))))
    
    total_loss.append(loss_esd)
    
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            model_input.shape[0],
            model_input.shape[2] // 2,
            model_input.shape[3] // 2,
            transformer.device,
            weight_dtype,
        )
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]

    noisy_model_input = noise_scheduler.add_noise(model_input,
                                                  noise,
                                                  t_enc_ddpm)

    packed_noisy_model_input = FluxPipeline._pack_latents(
            noisy_model_input,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3],
        )

    if transformer.config.guidance_embeds:
        guidance = torch.tensor([args.guidance_scale], device=transformer.device)
        guidance = guidance.expand(model_input.shape[0])
    else:
        guidance = None
    remove_indices = batch['remove_indices'][0]
    
    model_pred, attn_maps, attn_maps_single = transformer(
        hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
        timestep=t_enc_ddpm / 1000,
        guidance=guidance.to(dtype=weight_dtype, device=transformer.device),
        pooled_projections=pooled_emb_p.to(dtype=weight_dtype, device=transformer.device),
        encoder_hidden_states=emb_p.to(dtype=weight_dtype, device=transformer.device),
        txt_ids=text_ids_p.to(dtype=weight_dtype, device=transformer.device),
        img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
        return_dict=False,
    )
    # torch.Size([1, 24, 1280, 1280]) * 1

    attn_map_mask = torch.ones_like(attn_maps).to(transformer.device)
    attn_map_mask[..., remove_indices] = 0
    attn_map_mask = 1 - attn_map_mask

    # Compute regular loss.
    loss_attn = sum(torch.norm(attn_map_mask*attn_maps, dim=(0, 1))).sum()

    total_loss.append(loss_attn)
            
    return total_loss, t_enc_ddpm


def calculate_upper_loss_with_single_attn(
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
    lamb1=1.0,   # ESD loss weight
    lamb2=1.0,   # attn_maps loss weight
    lamb3=1.0,   # attn_maps_single loss weight
):
    """
    @func:
        upper loss + single attention suppression loss
        a) upper_loss: make sure samples from D_un(unlearning) is removed.
        b) attention loss on attn_maps
        c) attention loss on attn_maps_single
    """

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels

    # Convert images to latent space
    if args.cache_latents:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype, device=transformer.device)
        model_input = vae.encode(pixel_values).latent_dist.sample()

    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)

    # text embeddings
    emb_0, pooled_emb_0, text_ids_0 = compute_text_embeddings(
        neg_prompts, text_encoders, tokenizers
    )
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
        prompts, text_encoders, tokenizers
    )

    # sample timestep
    t_enc = torch.randint(ddim_steps, (1,), device=transformer.device)
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=transformer.device)

    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))

    start_guidance_tensor = torch.tensor([start_guidance], device=transformer.device)
    start_guidance_tensor = start_guidance_tensor.expand(model_input.shape[0])

    with torch.no_grad():
        # generate latent with concept from ESD model
        z, latent_image_ids = latent_sample(
            transformer,
            noise_scheduler,
            1,
            model_input.shape[1],
            512,
            512,
            emb_p.to(transformer.device),
            pooled_emb_p.to(transformer.device),
            text_ids_p.to(transformer.device),
            start_guidance_tensor,
            int(ddim_steps),
            vae_scale_factor,
        )

        e_0 = predict_noise(
            transformer, z, emb_0, pooled_emb_0, text_ids_0, latent_image_ids,
            guidance=start_guidance_tensor,
            timesteps=t_enc_ddpm.to(transformer.device),
            CPU_only=True,
        )
        e_p = predict_noise(
            transformer, z, emb_p, pooled_emb_p, text_ids_p, latent_image_ids,
            guidance=start_guidance_tensor,
            timesteps=t_enc_ddpm.to(transformer.device),
            CPU_only=True,
        )

    e_n = predict_noise(
        transformer, z, emb_p, pooled_emb_p, text_ids_p, latent_image_ids,
        guidance=start_guidance_tensor,
        timesteps=t_enc_ddpm.to(transformer.device),
        CPU_only=True,
    )

    e_0.requires_grad = False
    e_p.requires_grad = False

    # 1. ESD loss
    target_noise = e_0.to(transformer.device) - (
        negative_guidance * (e_p.to(transformer.device) - e_0.to(transformer.device))
    )
    loss_esd = criteria(e_n.to(transformer.device), target_noise)

    # prepare noisy latent input
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
        model_input.shape[0],
        model_input.shape[2] // 2,
        model_input.shape[3] // 2,
        transformer.device,
        weight_dtype,
    )

    noise = torch.randn_like(model_input)
    noisy_model_input = noise_scheduler.add_noise(model_input, noise, t_enc_ddpm)

    packed_noisy_model_input = FluxPipeline._pack_latents(
        noisy_model_input,
        batch_size=model_input.shape[0],
        num_channels_latents=model_input.shape[1],
        height=model_input.shape[2],
        width=model_input.shape[3],
    )

    if transformer.config.guidance_embeds:
        guidance = torch.tensor([args.guidance_scale], device=transformer.device)
        guidance = guidance.expand(model_input.shape[0])
    else:
        guidance = None

    remove_indices = batch["remove_indices"][0]

    model_pred, attn_maps, attn_maps_single = transformer(
        hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
        timestep=t_enc_ddpm / 1000,
        guidance=guidance.to(dtype=weight_dtype, device=transformer.device) if guidance is not None else None,
        pooled_projections=pooled_emb_p.to(dtype=weight_dtype, device=transformer.device),
        encoder_hidden_states=emb_p.to(dtype=weight_dtype, device=transformer.device),
        txt_ids=text_ids_p.to(dtype=weight_dtype, device=transformer.device),
        img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
        return_dict=False,
    )

    def compute_attention_suppression_loss(attn_tensor, remove_indices):
        attn_map_mask = torch.ones_like(attn_tensor, device=attn_tensor.device)
        attn_map_mask[..., remove_indices] = 0
        attn_map_mask = 1 - attn_map_mask

        loss = sum(torch.norm(attn_map_mask * attn_tensor, dim=(0, 1))).sum()
        return loss

    # 2. attention loss on attn_maps
    loss_attn = compute_attention_suppression_loss(attn_maps, remove_indices)

    # 3. attention loss on attn_maps_single
    loss_attn_single = compute_attention_suppression_loss(attn_maps_single, remove_indices)

    return loss_esd, loss_attn, loss_attn_single, t_enc_ddpm


def calculate_upper_loss_sup_esd(
    args,
    batch,
    compute_text_embeddings,
    text_encoders,
    tokenizers,
    transformer,             # student
    transformer_base,        # teacher
    noise_scheduler,
    prompts,                 # original prompts (contain target concept)
    super_concept,             # superclass prompts
    vae,
    criteria,
    negative_guidance,
    weight_dtype,
    start_guidance=3,
    ddim_steps=28,
    lamb_sup=1.0,      # weight for superclass
    lamb_attn=1.0      # weight for attention removal
):
    """
    NEW UPPER LOSS:
        L = λ_sup * L_sup + λ_attn * L_attn

    - No ESD loss
    - No RSC loss
    """

    ############################################
    # 0. Latent 준비
    ############################################
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    # Convert images to latent space
    if args.cache_latents:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype, device = transformer.device)
        model_input = vae.encode(pixel_values).latent_dist.sample()

    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)

    pixel_values = batch["pixel_values"].to(
        dtype=vae.dtype, device=transformer.device
    )
    latents = vae.encode(pixel_values).latent_dist.sample()

    # scale
    shift = vae.config.shift_factor
    scale = vae.config.scaling_factor
    latents = (latents - shift) * scale
    latents = latents.to(dtype=weight_dtype)

    ############################################
    # 1. Get three embeddings (neg, target, superclass)
    ############################################
    if super_concept is not None:
        # 6-1. super_prompt 생성
        # 단일 단어 치환 (더 robust하게 하려면 tokenizer 기반 word-level 치환 가능)
        super_prompts = []
        for p in prompts:
            # 예: "a nude girl on beach" → "a girl on beach"
            tokens = p.split()
            replaced = [super_concept if t == batch["synonym_words"][0] else t for t in tokens]
            super_prompts.append(" ".join(replaced))

    emb_0, pooled_0, ids_0 = compute_text_embeddings(
        args.prompt_b, text_encoders, tokenizers
    )
    emb_p, pooled_p, ids_p = compute_text_embeddings(
        prompts, text_encoders, tokenizers
    )
    emb_sup, pooled_sup, ids_sup = compute_text_embeddings(
        super_prompts, text_encoders, tokenizers
    )

    ############################################
    # 2. Random timestep
    ############################################
    t_enc = torch.randint(ddim_steps, (1,), device=transformer.device)
    og_low = round((int(t_enc) / ddim_steps) * 1000)
    og_high = round((int(t_enc + 1) / ddim_steps) * 1000)
    t_ddpm = torch.randint(og_low, og_high, (1,), device=transformer.device)

    ############################################
    # 3. Sample z via teacher
    ############################################
    vae_scale = 2 ** len(vae.config.block_out_channels)

    start_guid = torch.tensor([start_guidance], device=transformer.device)
    start_guid = start_guid.expand(latents.shape[0])

    with torch.no_grad():
        z, latent_image_ids = latent_sample(
            transformer_base,
            noise_scheduler,
            1,
            latents.shape[1],
            512, 512,
            emb_p,
            pooled_p,
            ids_p,
            start_guid,
            int(ddim_steps),
            vae_scale
        )

        # unconditional teacher
        e_0 = predict_noise(
            transformer_base, z,
            emb_0, pooled_0, ids_0,
            latent_image_ids,
            guidance=start_guid,
            timesteps=t_ddpm,
            CPU_only=True
        )

        # superclass raw teacher
        e_sup_raw = predict_noise(
            transformer_base, z,
            emb_sup, pooled_sup, ids_sup,
            latent_image_ids,
            guidance=start_guid,
            timesteps=t_ddpm,
            CPU_only=True
        )

    ############################################
    # 4. Superclass CFG target
    ############################################
    e_sup_cfg = e_0 - negative_guidance * (e_sup_raw - e_0)

    ############################################
    # 5. Student prediction (LoRA)
    ############################################
    e_n = predict_noise(
        transformer, z,
        emb_p, pooled_p, ids_p,
        latent_image_ids,
        guidance=start_guid,
        timesteps=t_ddpm,
        CPU_only=True
    )

    ############################################
    # 6. Superclass loss
    ############################################
    L_sup = criteria(e_n, e_sup_cfg.to(e_n.device))

    ############################################
    # 7. Attention loss
    ############################################
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            model_input.shape[0],
            model_input.shape[2] // 2,
            model_input.shape[3] // 2,
            transformer.device,
            weight_dtype,
        )
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]

    noisy_model_input = noise_scheduler.add_noise(model_input,
                                                  noise,
                                                  t_ddpm)

    packed_noisy_model_input = FluxPipeline._pack_latents(
            noisy_model_input,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3],
        )

    if transformer.config.guidance_embeds:
        guidance = torch.tensor([args.guidance_scale], device=transformer.device)
        guidance = guidance.expand(model_input.shape[0])
    else:
        guidance = None

    remove_indices = batch['remove_indices'][0]

    model_pred, attn_maps, attn_maps_single = transformer(
        hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
        timestep=t_ddpm / 1000,
        guidance=guidance.to(dtype=weight_dtype, device=transformer.device),
        pooled_projections=pooled_p.to(dtype=weight_dtype, device=transformer.device),
        encoder_hidden_states=emb_p.to(dtype=weight_dtype, device=transformer.device),
        txt_ids=ids_p.to(dtype=weight_dtype, device=transformer.device),
        img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
        return_dict=False,
    )[0:2]

    attn_map_mask = torch.ones_like(attn_maps).to(transformer.device)
    attn_map_mask[..., remove_indices] = 0
    attn_map_mask = 1 - attn_map_mask

    # Compute regular loss.
    L_attn = sum(torch.norm(attn_map_mask*attn_maps, dim=(0, 1))).sum()

    ############################################
    # 8. Final loss
    ############################################
    total_loss = lamb_sup * L_sup + lamb_attn * L_attn

    return (L_sup, L_attn, total_loss), t_ddpm


def calculate_upper_loss_new(
    args,
    batch,
    compute_text_embeddings,
    text_encoders,
    tokenizers,
    transformer,               # ★ LoRA 적용 모델 (student)
    transformer_base,          # ★ LoRA 적용 안 된 base 모델 (teacher)
    noise_scheduler,
    prompts,                   # 원래 prompt (지울 concept 포함)
    vae,
    criteria,
    negative_guidance,
    weight_dtype,
    neg_prompts,
    super_concept=None,        # ★ Moe_superclass_qwen에서 얻은 superclass 단어
    lamb_sup=1.0,              # ★ L_sup weight
    start_guidance=3,
    ddim_steps=28,
    lamb1=1,
    lamb2=1
):
    """
    Modified version:
    - Adds superclass-based replacement loss L_sup
    - super_concept: str (e.g., 'girl')
    """

    # ===========================================
    # 0. latent 준비 (기존 코드 그대로)
    # ===========================================
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels

    pixel_values = batch["pixel_values"].to(dtype=vae.dtype, device=transformer.device)
    model_input = vae.encode(pixel_values).latent_dist.sample()

    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)

    # ===========================================
    # 1. ESD text embeddings
    # ===========================================
    emb_0, pooled_emb_0, text_ids_0 = compute_text_embeddings(
        neg_prompts, text_encoders, tokenizers
    )
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
        prompts, text_encoders, tokenizers
    )

    # ===========================================
    # 2. timestep 샘플링
    # ===========================================
    t_enc = torch.randint(ddim_steps, (1,), device=transformer.device)
    og_num = round((int(t_enc)/ddim_steps)*1000)
    og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=transformer.device)

    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))
    start_guidance = torch.tensor([start_guidance], device=transformer.device).expand(model_input.shape[0])

    # ===========================================
    # 3. sample latent z (기존 코드)
    # ===========================================
    with torch.no_grad():
        z, latent_image_ids = latent_sample(
            transformer,
            noise_scheduler,
            1,
            model_input.shape[1],
            512,
            512,
            emb_p.to(transformer.device),
            pooled_emb_p.to(transformer.device),
            text_ids_p.to(transformer.device),
            start_guidance,
            int(ddim_steps),
            vae_scale_factor
        )

        e_0 = predict_noise(transformer, z, emb_0, pooled_emb_0, text_ids_0, latent_image_ids,
                            guidance=start_guidance, timesteps=t_enc_ddpm, CPU_only=True)
        e_p = predict_noise(transformer, z, emb_p, pooled_emb_p, text_ids_p, latent_image_ids,
                            guidance=start_guidance, timesteps=t_enc_ddpm, CPU_only=True)

    # ===========================================
    # 4. student velocity (LoRA 모델)
    # ===========================================
    e_n = predict_noise(
        transformer,
        z,
        emb_p,
        pooled_emb_p,
        text_ids_p,
        latent_image_ids,
        guidance=start_guidance,
        timesteps=t_enc_ddpm,
        CPU_only=True
    )

    total_loss = []

    # ===========================================
    # 5. 기존 upper ESD loss
    # ===========================================
    loss_esd = criteria(
        e_n.to(transformer.device),
        e_0.to(transformer.device) - (negative_guidance * (e_p.to(transformer.device) - e_0.to(transformer.device)))
    )
    total_loss.append(loss_esd)

    # ===========================================
    # 6. ★★★ L_sup 추가 ★★★
    # ===========================================

    if super_concept is not None:
        # 6-1. super_prompt 생성
        # 단일 단어 치환 (더 robust하게 하려면 tokenizer 기반 word-level 치환 가능)
        super_prompts = []
        for p in prompts:
            # 예: "a nude girl on beach" → "a girl on beach"
            tokens = p.split()
            replaced = [super_concept if t == batch["synonym_words"][0] else t for t in tokens]
            super_prompts.append(" ".join(replaced))

        # 6-2. superclass prompt embedding
        sup_emb, sup_pooled, sup_ids = compute_text_embeddings(
            super_prompts, text_encoders, tokenizers
        )

        # 6-3. base velocity (teacher)
        with torch.no_grad():
            e_sup = predict_noise(
                transformer_base,      # base 모델
                z,
                sup_emb,
                sup_pooled,
                sup_ids,
                latent_image_ids,
                guidance=start_guidance,
                timesteps=t_enc_ddpm,
                CPU_only=True
            )

        # 6-4. L_sup 계산
        loss_sup = torch.nn.functional.mse_loss(
            e_n.to(transformer.device),      # student
            e_sup.to(transformer.device)     # teacher
        )

        total_loss.append(lamb_sup * loss_sup)

    # ===========================================
    # 7. 기존 attention penalty
    # ===========================================
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
        model_input.shape[0],
        model_input.shape[2] // 2,
        model_input.shape[3] // 2,
        transformer.device,
        weight_dtype,
    )

    noise = torch.randn_like(model_input)
    noisy_model_input = noise_scheduler.add_noise(model_input, noise, t_enc_ddpm)

    packed_noisy = FluxPipeline._pack_latents(
        noisy_model_input,
        batch_size=model_input.shape[0],
        num_channels_latents=model_input.shape[1],
        height=model_input.shape[2],
        width=model_input.shape[3],
    )

    guidance = None
    if transformer.config.guidance_embeds:
        guidance = torch.tensor([args.guidance_scale], device=transformer.device).expand(model_input.shape[0])

    remove_indices = batch['remove_indices'][0]

    model_pred, attn_maps, attn_maps_single = transformer(
        hidden_states=packed_noisy.to(weight_dtype),
        timestep=t_enc_ddpm / 1000,
        guidance=guidance,
        pooled_projections=pooled_emb_p.to(weight_dtype),
        encoder_hidden_states=emb_p.to(weight_dtype),
        txt_ids=text_ids_p.to(weight_dtype),
        img_ids=latent_image_ids.to(weight_dtype),
        return_dict=False,
    )[0:2]

    attn_map_mask = torch.ones_like(attn_maps).to(transformer.device)
    attn_map_mask[..., remove_indices] = 0
    attn_map_mask = 1 - attn_map_mask

    loss_attn = sum(torch.norm(attn_map_mask * attn_maps, dim=(0, 1))).sum()
    total_loss.append(loss_attn)

    return total_loss, t_enc_ddpm



def calculate_lower_loss(args, batch, compute_text_embeddings, text_encoders, tokenizers, transformer, noise_scheduler, prompts, vae, criteria, negative_guidance, weight_dtype, t_enc_ddpm, start_guidance=3, ddim_steps=28, K=3, ir_concept_lst=[]):
    """
        @date: 2024.11.30 
        @name: bi-level loss
        @func: a) upper_loss: make sure samples from D_un(unlearning) is removed.
               · ESD 
               · attn map deactivation
               b) lower_loss: make sure samples from D_ir(irrelevant) is perserved.
               · lora loss (low med high timesteps)
               · InfoNCE loss
    """
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))
    
    # Convert images to latent space
    if args.cache_latents:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype, device = transformer.device)
        model_input = vae.encode(pixel_values).latent_dist.sample()

    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                model_input.shape[0],
                model_input.shape[2] // 2,
                model_input.shape[3] // 2,
                transformer.device,
                weight_dtype,
            )
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]
    
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
            prompts, text_encoders, tokenizers
        )
        
    noisy_model_input = noise_scheduler.add_noise(model_input,
                                                  noise,
                                                  t_enc_ddpm)

    packed_noisy_model_input = FluxPipeline._pack_latents(
            noisy_model_input,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3],
        )
        
    if transformer.config.guidance_embeds:
        guidance = torch.tensor([args.guidance_scale], device=transformer.device)
        guidance = guidance.expand(model_input.shape[0])
    else:
        guidance = None
        
    model_pred = transformer(
        hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
        timestep=t_enc_ddpm / 1000,
        guidance=guidance.to(dtype=weight_dtype, device=transformer.device),
        pooled_projections=pooled_emb_p.to(dtype=weight_dtype, device=transformer.device),
        encoder_hidden_states=emb_p.to(dtype=weight_dtype, device=transformer.device),
        txt_ids=text_ids_p.to(dtype=weight_dtype, device=transformer.device),
        img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
        return_dict=False,
        output_attentions=False,
    )[0]
    # torch.Size([1, 1024, 64]), torch.Size([1, 24, 1280, 1280])
        
    model_pred = FluxPipeline._unpack_latents(
        model_pred,
        height=model_input.shape[2] * vae_scale_factor,
        width=model_input.shape[3] * vae_scale_factor,
        vae_scale_factor=vae_scale_factor,
    )
    # torch.Size([1, 16, 64, 64])

    # flow matching loss
    target = noise - model_input

    loss_lora = torch.mean(
        ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        1,
    )[0]
    
    total_loss = []
    total_loss.append(loss_lora)
    
    # one negtive sample (synonym) + K positive sample (irrelevant)
    start_code = torch.randn_like(model_input)
    start_guidance = 3
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    
    # negtive sample: emb_neg
    emb_neg, pooled_emb_neg, text_ids_neg = compute_text_embeddings(
            batch["synonym_words"], text_encoders, tokenizers
        )
    
    with torch.no_grad():
        _, _, attn_map_lst_neg, attn_map_lst_single_neg = latent_sample(transformer,
                                               noise_scheduler,
                                               1,
                                               model_input.shape[1], 
                                               512,
                                               512,
                                               emb_neg.to(transformer.device),
                                               pooled_emb_neg.to(transformer.device),
                                               text_ids_neg.to(transformer.device),
                                               start_guidance, 
                                               int(ddim_steps),
                                               vae_scale_factor,
                                               latents=start_code,
                                               return_attn=True)
    
    # irrelevant sample: emb_pos
    if len(ir_concept_lst) != K:
        raise Exception("请检查ir_concept_lst")
    
    # random_attn_map
    # exp1: 20, 27
    # exp2: 14, 27
    # exp3: 0, 27
    attn_map_rand_idx = random.randint(0, int(ddim_steps)-1)
    
    pos_lst = []
    for idx in range(K): 
        
        emb_pos, pooled_emb_pos, text_ids_pos = compute_text_embeddings(
                ir_concept_lst[idx], text_encoders, tokenizers
            )
        _, _, attn_map_lst_pos_sub, attn_map_single_lst_pos_sub = latent_sample(transformer,
                                                   noise_scheduler,
                                                   1,
                                                   model_input.shape[1], 
                                                   512,
                                                   512,
                                                   emb_pos.to(transformer.device),
                                                   pooled_emb_pos.to(transformer.device),
                                                   text_ids_pos.to(transformer.device),
                                                   start_guidance, 
                                                   int(ddim_steps),
                                                   vae_scale_factor,
                                                   latents=start_code,
                                                   return_attn=True)

        tmp_attn_pos = attn_map_lst_pos_sub[attn_map_rand_idx]
        pos_lst.append(tmp_attn_pos)
        
    attn_map_neg = attn_map_lst_neg[attn_map_rand_idx]
    attn_map_pos = pos_lst
    
    info_neg = attn_map_neg[..., batch['remove_indices'][0]][:, 0, ...].permute(0, 2, 1)
    info_pos_lst = []
    
    for idx in range(K):
        info_pos = pos_lst[idx][..., batch['remove_indices'][0]][:, 0, ...].permute(0, 2, 1)
        info_pos_lst.append(info_pos)
    
    info_center = attn_maps[..., batch['remove_indices'][0]][:, 0, ...].permute(0, 2, 1)
    
    loss_contrastive = calculate_steer_loss(info_center,
                                            info_neg,
                                            info_pos_lst,
                                            temperature=0.07)
    
    total_loss.append(loss_contrastive)
    return total_loss

def calculate_lower_loss_with_single_attn(
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
    t_enc_ddpm,
    start_guidance=3,
    ddim_steps=28,
    K=3,
    ir_concept_lst=[],
):
    """
        lower_loss:
        - shared LoRA reconstruction loss
        - dual-stream contrastive loss
        - single-stream contrastive loss
    """
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))

    # Convert images to latent space
    if args.cache_latents:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype, device=transformer.device)
        model_input = vae.encode(pixel_values).latent_dist.sample()

    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)

    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
        model_input.shape[0],
        model_input.shape[2] // 2,
        model_input.shape[3] // 2,
        transformer.device,
        weight_dtype,
    )

    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]

    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
        prompts, text_encoders, tokenizers
    )

    noisy_model_input = noise_scheduler.add_noise(
        model_input,
        noise,
        t_enc_ddpm
    )

    packed_noisy_model_input = FluxPipeline._pack_latents(
        noisy_model_input,
        batch_size=model_input.shape[0],
        num_channels_latents=model_input.shape[1],
        height=model_input.shape[2],
        width=model_input.shape[3],
    )

    if transformer.config.guidance_embeds:
        guidance = torch.tensor([args.guidance_scale], device=transformer.device)
        guidance = guidance.expand(model_input.shape[0])
    else:
        guidance = None

    # 반드시 dual / single attention 둘 다 받도록 확인
    out = transformer(
        hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
        timestep=t_enc_ddpm / 1000,
        guidance=guidance.to(dtype=weight_dtype, device=transformer.device) if guidance is not None else None,
        pooled_projections=pooled_emb_p.to(dtype=weight_dtype, device=transformer.device),
        encoder_hidden_states=emb_p.to(dtype=weight_dtype, device=transformer.device),
        txt_ids=text_ids_p.to(dtype=weight_dtype, device=transformer.device),
        img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
        return_dict=False,
    )

    # 네 transformer 반환 형식에 맞게 꼭 확인
    # 기대: model_pred, attn_maps, attn_maps_single, ...
    model_pred = out[0]
    attn_maps = out[1]          # dual-stream attn
    attn_maps_single = out[2]   # single-stream attn

    model_pred = FluxPipeline._unpack_latents(
        model_pred,
        height=model_input.shape[2] * vae_scale_factor,
        width=model_input.shape[3] * vae_scale_factor,
        vae_scale_factor=vae_scale_factor,
    )

    # -----------------------------
    # 1) shared LoRA loss
    # -----------------------------
    target = noise - model_input

    loss_lora = torch.mean(
        ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        1,
    )[0]

    # -----------------------------
    # 2) contrastive prep
    # -----------------------------
    start_code = torch.randn_like(model_input)
    start_guidance_tensor = torch.tensor([start_guidance], device=transformer.device)
    start_guidance_tensor = start_guidance_tensor.expand(model_input.shape[0])

    # negative sample
    emb_neg, pooled_emb_neg, text_ids_neg = compute_text_embeddings(
        batch["synonym_words"], text_encoders, tokenizers
    )

    with torch.no_grad():
        _, _, attn_map_lst_neg, attn_map_lst_single_neg = latent_sample(
            transformer,
            noise_scheduler,
            1,
            model_input.shape[1],
            512,
            512,
            emb_neg.to(transformer.device),
            pooled_emb_neg.to(transformer.device),
            text_ids_neg.to(transformer.device),
            start_guidance_tensor,
            int(ddim_steps),
            vae_scale_factor,
            latents=start_code,
            return_attn=True,
        )

    if len(ir_concept_lst) != K:
        raise ValueError("ir_concept_lst length must be equal to K")

    attn_map_rand_idx = random.randint(0, int(ddim_steps) - 1)

    pos_lst_dual = []
    pos_lst_single = []

    for idx in range(K):
        emb_pos, pooled_emb_pos, text_ids_pos = compute_text_embeddings(
            ir_concept_lst[idx], text_encoders, tokenizers
        )

        with torch.no_grad():
            _, _, attn_map_lst_pos_sub, attn_map_single_lst_pos_sub = latent_sample(
                transformer,
                noise_scheduler,
                1,
                model_input.shape[1],
                512,
                512,
                emb_pos.to(transformer.device),
                pooled_emb_pos.to(transformer.device),
                text_ids_pos.to(transformer.device),
                start_guidance_tensor,
                int(ddim_steps),
                vae_scale_factor,
                latents=start_code,
                return_attn=True,
            )

        pos_lst_dual.append(attn_map_lst_pos_sub[attn_map_rand_idx])
        pos_lst_single.append(attn_map_single_lst_pos_sub[attn_map_rand_idx])

    # -----------------------------
    # 3) dual contrastive loss
    # -----------------------------
    attn_map_neg_dual = attn_map_lst_neg[attn_map_rand_idx]
    info_neg_dual = attn_map_neg_dual[..., batch["remove_indices"][0]][:, 0, ...].permute(0, 2, 1)

    info_pos_dual_lst = []
    for idx in range(K):
        info_pos_dual = pos_lst_dual[idx][..., batch["remove_indices"][0]][:, 0, ...].permute(0, 2, 1)
        info_pos_dual_lst.append(info_pos_dual)

    info_center_dual = attn_maps[..., batch["remove_indices"][0]][:, 0, ...].permute(0, 2, 1)

    loss_contrastive_dual = calculate_steer_loss(
        info_center_dual,
        info_neg_dual,
        info_pos_dual_lst,
        temperature=0.07,
    )

    # -----------------------------
    # 4) single contrastive loss
    # -----------------------------
    attn_map_neg_single = attn_map_lst_single_neg[attn_map_rand_idx]
    info_neg_single = attn_map_neg_single[..., batch["remove_indices"][0]][:, 0, ...].permute(0, 2, 1)

    info_pos_single_lst = []
    for idx in range(K):
        info_pos_single = pos_lst_single[idx][..., batch["remove_indices"][0]][:, 0, ...].permute(0, 2, 1)
        info_pos_single_lst.append(info_pos_single)

    info_center_single = attn_maps_single[..., batch["remove_indices"][0]][:, 0, ...].permute(0, 2, 1)

    loss_contrastive_single = calculate_steer_loss(
        info_center_single,
        info_neg_single,
        info_pos_single_lst,
        temperature=0.07,
    )

    return loss_lora, loss_contrastive_dual, loss_contrastive_single

def calculate_lower_loss_z_erase(args, batch, compute_text_embeddings, text_encoders, tokenizers, transformer, noise_scheduler, prompts, vae, criteria, weight_dtype, t_enc_ddpm, start_guidance=3, ddim_steps=28, K=3, ir_concept_lst=[]):
    """
        @date: 2025.04.13
        @name: Z-Erase Loss (Eq. 8)
        @func: Combined loss function for concept erasure with preservation
        
        Combines three objectives:
        1. Attention suppression: Reduce attention to target concept
        2. Preservation objective (Eq. 8): Minimize shift in generation
        
        Lpr = E_{x_t,t} ||v_{θ+∆θ}(x_t,∅,t) − v_θ(x_t,∅,t)||² 
            + E_{x_t,t,c_pr ∼ D_pr} ||v_{θ+∆θ}(x_t,c_pr,t) − v_θ(x_t,c_pr,t)||²
    """
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))
    
    # Convert images to latent space
    if args.cache_latents:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype, device = transformer.device)
        model_input = vae.encode(pixel_values).latent_dist.sample()

    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                model_input.shape[0],
                model_input.shape[2] // 2,
                model_input.shape[3] // 2,
                transformer.device,
                weight_dtype,
            )
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]
    
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
            prompts, text_encoders, tokenizers
        )
        
    noisy_model_input = noise_scheduler.add_noise(model_input,
                                                  noise,
                                                  t_enc_ddpm)

    packed_noisy_model_input = FluxPipeline._pack_latents(
            noisy_model_input,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3],
        )
        
    if transformer.config.guidance_embeds:
        guidance = torch.tensor([args.guidance_scale], device=transformer.device)
        guidance = guidance.expand(model_input.shape[0])
    else:
        guidance = None
        
    model_pred, attn_maps = transformer(
        hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
        timestep=t_enc_ddpm / 1000,
        guidance=guidance.to(dtype=weight_dtype, device=transformer.device),
        pooled_projections=pooled_emb_p.to(dtype=weight_dtype, device=transformer.device),
        encoder_hidden_states=emb_p.to(dtype=weight_dtype, device=transformer.device),
        txt_ids=text_ids_p.to(dtype=weight_dtype, device=transformer.device),
        img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
        return_dict=False,
    )[0:2]
        
    model_pred = FluxPipeline._unpack_latents(
        model_pred,
        height=model_input.shape[2] * vae_scale_factor,
        width=model_input.shape[3] * vae_scale_factor,
        vae_scale_factor=vae_scale_factor,
    )

    # =========================================
    # Preservation Loss (Eq. 8)
    # =========================================
    # Eq. 8: Lpr = E_{x_t,t} ||v_{θ+∆θ}(x_t,∅,t) − v_θ(x_t,∅,t)||² 
    #            + E_{x_t,t,c_pr ∼ D_pr} ||v_{θ+∆θ}(x_t,c_pr,t) − v_θ(x_t,c_pr,t)||²
    
    # Get empty prompt embeddings (for unconditional)
    emb_empty, pooled_emb_empty, text_ids_empty = compute_text_embeddings(
            "", text_encoders, tokenizers
        )
    
    if transformer.config.guidance_embeds:
        guidance = torch.tensor([args.guidance_scale], device=transformer.device)
        guidance = guidance.expand(model_input.shape[0])
    else:
        guidance = None
    
    # Part 1: Unconditional prediction shift
    # LoRA applied
    model_pred_empty = transformer(
        hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
        timestep=t_enc_ddpm / 1000,
        guidance=guidance.to(dtype=weight_dtype, device=transformer.device) if guidance is not None else None,
        pooled_projections=pooled_emb_empty.to(dtype=weight_dtype, device=transformer.device),
        encoder_hidden_states=emb_empty.to(dtype=weight_dtype, device=transformer.device),
        txt_ids=text_ids_empty.to(dtype=weight_dtype, device=transformer.device),
        img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
        return_dict=False,
        output_attentions=False,
    )[0]

    model_pred_empty_unpacked = FluxPipeline._unpack_latents(
        model_pred_empty,
        height=model_input.shape[2] * vae_scale_factor,
        width=model_input.shape[3] * vae_scale_factor,
        vae_scale_factor=vae_scale_factor,
    )
    
    # Reference (original model without LoRA)
    with torch.no_grad():
        # Disable LoRA adapter to get original model output
        if hasattr(transformer, 'disable_adapter_layers'):
            transformer.disable_adapter_layers()
        
        model_pred_empty_ref = transformer(
            hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
            timestep=t_enc_ddpm / 1000,
            guidance=guidance.to(dtype=weight_dtype, device=transformer.device) if guidance is not None else None,
            pooled_projections=pooled_emb_empty.to(dtype=weight_dtype, device=transformer.device),
            encoder_hidden_states=emb_empty.to(dtype=weight_dtype, device=transformer.device),
            txt_ids=text_ids_empty.to(dtype=weight_dtype, device=transformer.device),
            img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
            return_dict=False,
            output_attentions=False,
        )[0]

        model_pred_empty_ref_unpacked = FluxPipeline._unpack_latents(
            model_pred_empty_ref,
            height=model_input.shape[2] * vae_scale_factor,
            width=model_input.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )
        
        # Re-enable LoRA adapter
        if hasattr(transformer, 'enable_adapter_layers'):
            transformer.enable_adapter_layers()
    
    loss_empty = torch.mean(
        ((model_pred_empty_unpacked.float() - model_pred_empty_ref_unpacked.float()) ** 2).reshape(model_input.shape[0], -1),
        1,
    )[0]
    
    loss_preserve_total = loss_empty
    
    # Part 2: Preservation concepts prediction shift
    if len(ir_concept_lst) != K:
        raise Exception("请检查ir_concept_lst")
    
    for idx in range(K):
        emb_pr, pooled_emb_pr, text_ids_pr = compute_text_embeddings(
                ir_concept_lst[idx], text_encoders, tokenizers
            )
        
        # LoRA applied
        model_pred_pr = transformer(
            hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
            timestep=t_enc_ddpm / 1000,
            guidance=guidance.to(dtype=weight_dtype, device=transformer.device) if guidance is not None else None,
            pooled_projections=pooled_emb_pr.to(dtype=weight_dtype, device=transformer.device),
            encoder_hidden_states=emb_pr.to(dtype=weight_dtype, device=transformer.device),
            txt_ids=text_ids_pr.to(dtype=weight_dtype, device=transformer.device),
            img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
            return_dict=False,
            output_attentions=False,
        )[0]

        model_pred_pr_unpacked = FluxPipeline._unpack_latents(
            model_pred_pr,
            height=model_input.shape[2] * vae_scale_factor,
            width=model_input.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )
        
        # Reference (original model without LoRA)
        with torch.no_grad():
            # Disable LoRA adapter to get original model output
            if hasattr(transformer, 'disable_adapter_layers'):
                transformer.disable_adapter_layers()
            
            model_pred_pr_ref = transformer(
                hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
                timestep=t_enc_ddpm / 1000,
                guidance=guidance.to(dtype=weight_dtype, device=transformer.device) if guidance is not None else None,
                pooled_projections=pooled_emb_pr.to(dtype=weight_dtype, device=transformer.device),
                encoder_hidden_states=emb_pr.to(dtype=weight_dtype, device=transformer.device),
                txt_ids=text_ids_pr.to(dtype=weight_dtype, device=transformer.device),
                img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
                return_dict=False,
                output_attentions=False,
            )[0]

            model_pred_pr_ref_unpacked = FluxPipeline._unpack_latents(
                model_pred_pr_ref,
                height=model_input.shape[2] * vae_scale_factor,
                width=model_input.shape[3] * vae_scale_factor,
                vae_scale_factor=vae_scale_factor,
            )
            
            # Re-enable LoRA adapter
            if hasattr(transformer, 'enable_adapter_layers'):
                transformer.enable_adapter_layers()
        
        loss_pr = torch.mean(
            ((model_pred_pr_unpacked.float() - model_pred_pr_ref_unpacked.float()) ** 2).reshape(model_input.shape[0], -1),
            1,
        )[0]
        
        loss_preserve_total = loss_preserve_total + loss_pr
    
    # Normalize preservation loss
    if K > 0:
        loss_preserve = loss_preserve_total / (K + 1)
    else:
        loss_preserve = loss_preserve_total
    
    return loss_preserve


def calculate_lower_loss_sup(args, batch, compute_text_embeddings, text_encoders, tokenizers, transformer, noise_scheduler, prompts, vae, criteria, negative_guidance, weight_dtype, t_enc_ddpm, sup_concept, start_guidance=3, ddim_steps=28, K=3):
    """
        @date: 2024.11.30 
        @name: bi-level loss
        @func: a) upper_loss: make sure samples from D_un(unlearning) is removed.
               · ESD 
               · attn map deactivation
               b) lower_loss: make sure samples from D_ir(irrelevant) is perserved.
               · lora loss (low med high timesteps)
               · InfoNCE loss
    """
    
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))
    
    # Convert images to latent space
    if args.cache_latents:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype, device = transformer.device)
        model_input = vae.encode(pixel_values).latent_dist.sample()

    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                model_input.shape[0],
                model_input.shape[2] // 2,
                model_input.shape[3] // 2,
                transformer.device,
                weight_dtype,
            )
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]
    
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
            prompts, text_encoders, tokenizers
        )
        
    noisy_model_input = noise_scheduler.add_noise(model_input,
                                                  noise,
                                                  t_enc_ddpm)

    packed_noisy_model_input = FluxPipeline._pack_latents(
            noisy_model_input,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3],
        )
        
    if transformer.config.guidance_embeds:
        guidance = torch.tensor([args.guidance_scale], device=transformer.device)
        guidance = guidance.expand(model_input.shape[0])
    else:
        guidance = None
        
    model_pred, attn_maps, attn_maps_single = transformer(
        hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
        timestep=t_enc_ddpm / 1000,
        guidance=guidance.to(dtype=weight_dtype, device=transformer.device),
        pooled_projections=pooled_emb_p.to(dtype=weight_dtype, device=transformer.device),
        encoder_hidden_states=emb_p.to(dtype=weight_dtype, device=transformer.device),
        txt_ids=text_ids_p.to(dtype=weight_dtype, device=transformer.device),
        img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
        return_dict=False,
    )[0:2]
        
    model_pred = FluxPipeline._unpack_latents(
        model_pred,
        height=model_input.shape[2] * vae_scale_factor,
        width=model_input.shape[3] * vae_scale_factor,
        vae_scale_factor=vae_scale_factor,
    )

    # flow matching loss
    target = noise - model_input

    loss_lora = torch.mean(
        ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        1,
    )[0]
    
    total_loss = []
    total_loss.append(loss_lora)
    
    # one negtive sample (synonym) + K positive sample (irrelevant)
    start_code = torch.randn_like(model_input)
    start_guidance = 3
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    
    # negtive sample: emb_neg
    emb_neg, pooled_emb_neg, text_ids_neg = compute_text_embeddings(
            batch["synonym_words"], text_encoders, tokenizers
        )
    
    with torch.no_grad():
        _, _, attn_map_lst_neg, attn_map_single_lst_neg = latent_sample(transformer,
                                               noise_scheduler,
                                               1,
                                               model_input.shape[1], 
                                               512,
                                               512,
                                               emb_neg.to(transformer.device),
                                               pooled_emb_neg.to(transformer.device),
                                               text_ids_neg.to(transformer.device),
                                               start_guidance, 
                                               int(ddim_steps),
                                               vae_scale_factor,
                                               latents=start_code,
                                               return_attn=True)
    
    
    # random_attn_map
    # exp1: 20, 27
    # exp2: 14, 27
    # exp3: 0, 27
    attn_map_rand_idx = random.randint(0, int(ddim_steps)-1)
    
    pos_lst = []
        
    emb_pos, pooled_emb_pos, text_ids_pos = compute_text_embeddings(
            sup_concept, text_encoders, tokenizers
        )
    _, _, attn_map_lst_pos_sub, attn_map_single_lst_pos_sub = latent_sample(transformer,
                                                noise_scheduler,
                                                1,
                                                model_input.shape[1], 
                                                512,
                                                512,
                                                emb_pos.to(transformer.device),
                                                pooled_emb_pos.to(transformer.device),
                                                text_ids_pos.to(transformer.device),
                                                start_guidance, 
                                                int(ddim_steps),
                                                vae_scale_factor,
                                                latents=start_code,
                                                return_attn=True)

    tmp_attn_pos = attn_map_lst_pos_sub[attn_map_rand_idx]
        
    attn_map_neg = attn_map_lst_neg[attn_map_rand_idx]
    
    info_neg = attn_map_neg[..., batch['remove_indices'][0]][:, 0, ...].permute(0, 2, 1)
    info_pos_lst = []
    
    info_pos = tmp_attn_pos[..., batch['remove_indices'][0]][:, 0, ...].permute(0, 2, 1)
    info_pos_lst.append(info_pos)
    
    info_center = attn_maps[..., batch['remove_indices'][0]][:, 0, ...].permute(0, 2, 1)
    
    loss_contrastive = calculate_steer_loss(info_center,
                                            info_neg,
                                            info_pos_lst,
                                            temperature=0.07)
    
    total_loss.append(loss_contrastive)
    return total_loss
