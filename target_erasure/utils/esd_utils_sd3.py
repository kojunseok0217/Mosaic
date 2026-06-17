import torch
from diffusers.utils.torch_utils import randn_tensor
from typing import Any, List, Optional, Tuple, Union


from typing import Any, List, Optional, Tuple
import torch
from diffusers.utils.torch_utils import randn_tensor

@torch.no_grad()
def latent_sample_sd3(
    transformer,
    scheduler,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    timesteps: int,
    latents: Optional[torch.Tensor] = None,
    return_attn: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    # --- CFG 관련 추가 ---
    guidance_scale: Optional[float] = None,
    neg_prompt_embeds: Optional[torch.Tensor] = None,
    neg_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    # --- SD3 attention ---
    joint_attention_kwargs: Optional[dict] = None,
):
    device = transformer.device

    latent_h = int(height) // 8
    latent_w = int(width) // 8
    shape = (batch_size, num_channels_latents, latent_h, latent_w)

    if latents is None:
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        if hasattr(scheduler, "init_noise_sigma"):
            latents = latents * scheduler.init_noise_sigma
    else:
        latents = latents.to(device=device, dtype=dtype)

    # embeds
    model_dtype = next(transformer.parameters()).dtype
    prompt_embeds = prompt_embeds.to(device=device, dtype=model_dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=model_dtype)

    do_cfg = guidance_scale is not None and guidance_scale > 1.0
    if do_cfg:
        if neg_prompt_embeds is None or neg_pooled_prompt_embeds is None:
            raise ValueError("CFG requires neg_prompt_embeds and neg_pooled_prompt_embeds.")
        neg_prompt_embeds = neg_prompt_embeds.to(device=device, dtype=model_dtype)
        neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.to(device=device, dtype=model_dtype)

    scheduler.set_train_timesteps(timesteps, device=device)
    timesteps_list = scheduler.timesteps

    attn_map_lst: List[Any] = []

    # attention kwargs
    ja = dict(joint_attention_kwargs) if joint_attention_kwargs is not None else {}
    if return_attn:
        ja["output_attentions"] = True
    ja = ja if ja else None

    for t in timesteps_list:
        model_input = latents
        if hasattr(scheduler, "scale_model_input"):
            model_input = scheduler.scale_model_input(model_input, t)

        # timestep batch (float32 often safest for SD3)
        if torch.is_tensor(t):
            t_batch = t.to(device=device)
            if t_batch.ndim == 0:
                t_batch = t_batch[None]
        else:
            t_batch = torch.tensor([t], device=device)
        t_batch = t_batch.expand(model_input.shape[0]).to(torch.float32)

        if not do_cfg:
            out = transformer(
                hidden_states=model_input.to(device=device, dtype=model_dtype),
                timestep=t_batch,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                joint_attention_kwargs=ja,
                return_dict=False,
            )
            if return_attn:
                model_pred, attn_maps = out
                attn_map_lst.append(attn_maps)
            else:
                model_pred = out[0]
        else:
            # --- uncond forward ---
            out_u = transformer(
                hidden_states=model_input.to(device=device, dtype=model_dtype),
                timestep=t_batch,
                pooled_projections=neg_pooled_prompt_embeds,
                encoder_hidden_states=neg_prompt_embeds,
                joint_attention_kwargs=ja,
                return_dict=False,
            )
            # --- cond forward ---
            out_c = transformer(
                hidden_states=model_input.to(device=device, dtype=model_dtype),
                timestep=t_batch,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                joint_attention_kwargs=ja,
                return_dict=False,
            )

            if return_attn:
                pred_u, _attn_u = out_u
                pred_c, attn_c = out_c
                attn_map_lst.append(attn_c)  # 보통 cond쪽 attn만 씀
            else:
                pred_u = out_u[0]
                pred_c = out_c[0]

            # CFG combine
            model_pred = pred_u + guidance_scale * (pred_c - pred_u)

        latents = scheduler.step(model_pred, t, latents, return_dict=False)[0]

    return (latents, attn_map_lst) if return_attn else latents


def predict_noise_sd3(
    transformer,
    latent_code: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    timesteps: Union[torch.Tensor, float, int],
    dtype: torch.dtype = torch.bfloat16,
    # --- CFG ---
    guidance_scale: Optional[float] = None,
    neg_prompt_embeds: Optional[torch.Tensor] = None,
    neg_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    # --- attention ---
    joint_attention_kwargs: Optional[dict] = None,
    return_attn: bool = False,
):
    """
    SD3 apply_model equivalent.
    - If guidance_scale is None or <= 1: single conditional forward.
    - If guidance_scale > 1: do CFG by two forwards (uncond + cond) and combine outputs.

    Assumes transformer.forward(return_dict=False, joint_attention_kwargs={...})
    returns:
      - if output_attentions True: (pred, attn_maps_list)
      - else: (pred,) or (pred, something) depending on your customization.
    """
    device = transformer.device
    model_dtype = next(transformer.parameters()).dtype

    latent_code = latent_code.to(device=device, dtype=model_dtype)
    prompt_embeds = prompt_embeds.to(device=device, dtype=model_dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=model_dtype)

    # (B,) timestep (SD3 is typically float32)
    if torch.is_tensor(timesteps):
        t = timesteps.to(device=device)
        if t.ndim == 0:
            t = t[None]
    else:
        t = torch.tensor([timesteps], device=device)
    t = t.expand(latent_code.shape[0]).to(torch.float32)

    # attention kwargs
    ja = dict(joint_attention_kwargs) if joint_attention_kwargs is not None else {}
    if return_attn:
        ja["output_attentions"] = True
    ja = ja if ja else None

    do_cfg = guidance_scale is not None and guidance_scale > 1.0

    if not do_cfg:
        out = transformer(
            hidden_states=latent_code,
            timestep=t,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            joint_attention_kwargs=ja,
            return_dict=False,
        )
        if return_attn:
            pred, attn = out
            return pred, attn
        return out[0]

    # CFG needs uncond embeds
    if neg_prompt_embeds is None or neg_pooled_prompt_embeds is None:
        raise ValueError("CFG requires neg_prompt_embeds and neg_pooled_prompt_embeds.")

    neg_prompt_embeds = neg_prompt_embeds.to(device=device, dtype=model_dtype)
    neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.to(device=device, dtype=model_dtype)

    out_u = transformer(
        hidden_states=latent_code,
        timestep=t,
        pooled_projections=neg_pooled_prompt_embeds,
        encoder_hidden_states=neg_prompt_embeds,
        joint_attention_kwargs=ja,
        return_dict=False,
    )
    out_c = transformer(
        hidden_states=latent_code,
        timestep=t,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        joint_attention_kwargs=ja,
        return_dict=False,
    )

    if return_attn:
        pred_u, _attn_u = out_u
        pred_c, attn_c = out_c
    else:
        pred_u = out_u[0]
        pred_c = out_c[0]

    pred = pred_u + guidance_scale * (pred_c - pred_u)

    if return_attn:
        # 보통 cond 쪽 attn만 의미있어서 attn_c 반환
        return pred, attn_c
    return pred