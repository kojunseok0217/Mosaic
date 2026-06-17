import torch
import torch.nn.functional as F

def _encode_prompt_with_t5(
    text_encoder, tokenizer, max_sequence_length,
    prompt=None, num_images_per_prompt=1, device=None, text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,   # SD3 쪽은 True
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when tokenizer is None")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]  # (B, L, D_t5)
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder, tokenizer, prompt, device=None, text_input_ids=None, num_images_per_prompt=1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when tokenizer is None")

    out = text_encoder(text_input_ids.to(device), output_hidden_states=True)
    pooled = out[0]                    # (B, D_clip)  (SD3 스크립트는 이걸 pooled로 사용)
    clip_seq = out.hidden_states[-2]   # (B, 77, D_clip)

    clip_seq = clip_seq.to(dtype=text_encoder.dtype, device=device)
    _, seq_len, _ = clip_seq.shape

    clip_seq = clip_seq.repeat(1, num_images_per_prompt, 1)
    clip_seq = clip_seq.view(batch_size * num_images_per_prompt, seq_len, -1)

    pooled = pooled.to(dtype=text_encoder.dtype, device=device)
    pooled = pooled.repeat(1, num_images_per_prompt, 1)
    pooled = pooled.view(batch_size * num_images_per_prompt, -1)

    return clip_seq, pooled


def encode_prompt_sd3(
    text_encoders,     # [clip1, clip2, t5]
    tokenizers,        # [tok1, tok2, tok3]
    prompt,
    max_sequence_length,
    device=None,
    num_images_per_prompt=1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_seq_list = []
    clip_pooled_list = []

    for i, (tok, enc) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        clip_seq, pooled = _encode_prompt_with_clip(
            text_encoder=enc,
            tokenizer=tok,
            prompt=prompt,
            device=device if device is not None else enc.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_seq_list.append(clip_seq)
        clip_pooled_list.append(pooled)

    clip_seq = torch.cat(clip_seq_list, dim=-1)         # (B, 77, D1+D2)
    pooled_prompt_embeds = torch.cat(clip_pooled_list, dim=-1)  # (B, D1+D2)

    t5_seq = _encode_prompt_with_t5(
        text_encoder=text_encoders[-1],
        tokenizer=tokenizers[-1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
    )  # (B, L_t5, D_t5)

    # clip hidden dim을 t5 hidden dim에 맞추기 위해 pad
    clip_seq = F.pad(clip_seq, (0, t5_seq.shape[-1] - clip_seq.shape[-1]))

    # sequence concat: (B, 77 + L_t5, D_t5)
    prompt_embeds = torch.cat([clip_seq, t5_seq], dim=-2)

    return prompt_embeds, pooled_prompt_embeds