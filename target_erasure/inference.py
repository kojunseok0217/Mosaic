import os
import json
import torch
import random
import numpy as np
import argparse
from huggingface_hub import login
from diffusers import FluxPipeline
from safetensors.torch import load_file


def set_seed(seed: int = 42):
    """Reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Flux Image Generation Script")

    # ===== Required Arguments =====
    parser.add_argument("--concept", type=str, required=True,
                        help="Concept name (e.g., 'SpongeBob_HomerSimpson')")
    parser.add_argument("--prompt_json", type=str, required=True,
                        help="Path to JSON file containing prompts")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated images")

    # ===== Optional Arguments =====
    parser.add_argument("--use_lora", action="store_true",
                        help="Whether to use a LoRA weight file")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA safetensors file")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to pretrained Flux model directory")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace login token")
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ===== Set Environment & Seed =====
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    set_seed(args.seed)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # ===== Login to HuggingFace (optional) =====
    if args.hf_token:
        login(args.hf_token)

    # ===== Load Model =====
    pipe = FluxPipeline.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16)

    if args.use_lora:
        if args.lora_path is None:
            raise ValueError("--use_lora specified but --lora_path not provided.")
        print(f"🔹 Loading LoRA weights from {args.lora_path}")
        pipe.load_lora_weights(args.lora_path)

    pipe = pipe.to("cuda:0")

    # ===== Load Prompts from JSON =====
    with open(args.prompt_json, "r", encoding="utf-8") as f:
        prompts_dict = json.load(f)

    # If concept key exists, use it; else assume the file is a list
    if args.concept in prompts_dict:
        prompt_lst = prompts_dict[args.concept]
    elif isinstance(prompts_dict, list):
        prompt_lst = prompts_dict
    else:
        raise ValueError(f"Concept '{args.concept}' not found in {args.prompt_json}")

    # ===== Prepare Output Folder =====
    if args.use_lora:
        save_dir = os.path.join(args.output_dir, args.concept, "after")
    else:
        save_dir = os.path.join(args.output_dir, args.concept, "before")
    os.makedirs(save_dir, exist_ok=True)
    print(f"💾 Saving images to: {save_dir}")

    # ===== Generate Images =====
    for idx, prompt in enumerate(prompt_lst):
        print(f"[{idx+1}/{len(prompt_lst)}] Generating → {prompt}")
        image = pipe(
            prompt=prompt,
            generator=generator,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            max_sequence_length=256,
        ).images[0]
        image.save(os.path.join(save_dir, f"{idx:02d}.png"))

    print("✅ All images generated successfully.")


if __name__ == "__main__":
    main()
