# coding: UTF-8
"""
    @date:  2024.11.25  week48  Monday
    @func:  dataset generation.  
"""

import os
import json
import argparse
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from einops import rearrange
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import FluxPipeline
import torch
from matplotlib import pyplot as plt
import random
from huggingface_hub import login


def main(args):
    # 환경 설정
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 환경일 때

    generator = torch.Generator(device="cuda").manual_seed(seed)

    # 모델 로드
    model_path = (
        "black-forest-labs/FLUX.1-dev"
    )
    model = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to("cuda:0")

    json_path = args.json_path

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 저장 경로 생성
    base_dir = "/nas/home/junseok/workspace/multi_concept_erasure/EraseAnything/image"

    for category in data.keys():
        for keyword in data[category].keys():
            save_dir = os.path.join(base_dir, keyword)
            os.makedirs(save_dir, exist_ok=True)
            for idx, prompt in enumerate(data[category][keyword]):
                out = model(
                    prompt=prompt,
                    generator=generator,
                    guidance_scale=3.5,
                    height=512,
                    width=512,
                    num_inference_steps=28,
                    max_sequence_length=256,
                ).images[0]

                out.save(os.path.join(save_dir, f"{idx}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from prompts using FluxPipeline.")
    parser.add_argument("--json_path", type=str, required=True)
    args = parser.parse_args()

    main(args)


# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# from PIL import Image
# from torchvision import transforms
# from tqdm import tqdm
# from einops import rearrange
# import numpy as np
# from typing import Any, Callable, Dict, List, Optional, Union
# from diffusers import FluxPipeline
# import torch
# from matplotlib import pyplot as plt
# import pandas as pd
# import random

# model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
# model.to("cuda:0")

# keyword = "David Beckham"

# csv_path = "/home/juniboy97/workspace/Diffusion-Unlearning/HUB/prompts/target_image/David Beckham.csv"  # 여기에 실제 CSV 경로 넣기

# with open(csv_path, 'r', encoding='utf-8') as f:
#     all_prompts = [line.strip() for line in f if line.strip()]

# # keyword가 포함된 prompt만 필터링
# filtered_prompts = [p for p in all_prompts if keyword.lower() in p.lower()]
# prompt_lst = []

# prompt_lst.extend(filtered_prompts)

# # # 랜덤하게 20개 추출
# # prompt_lst = random.sample(all_prompts, 20)

# base_dir = "/home/juniboy97/workspace/Diffusion-Unlearning/EraseAnything/image"
# save_dir = os.path.join(base_dir, keyword)
# os.makedirs(save_dir, exist_ok=True)

# for idx in range(len(prompt_lst)):
#     out, out_attn_maps = model(
#         prompt=prompt_lst[idx],
#         guidance_scale=0.0,
#         height=512,
#         width=512,
#         num_inference_steps=28,
#         max_sequence_length=256,
#         num_images_per_prompt=1,
#         return_dict=False
#     )
    
#     # import pdb; pdb.set_trace()
#     for jdx, item in enumerate(out):
#         file_path = os.path.join(save_dir, f"{idx}.png")
#         item.save(file_path)
    
    
    # for jdx, attn_map in enumerate(out_attn_maps):
    #     for bbb in range(attn_map.size(-1)):
    #         fig, axs = plt.subplots(6, 4, figsize=(32, 32))
    #         for head in range(24):
    #             # import pdb; pdb.set_trace()
    #             attn_sub = attn_map[head, 256:, bbb]   #attn_map.mean(0)
    #             attn_sub = attn_sub.reshape(32, 32).float()
    #             attn_sub = torch.flip(attn_sub, [0])
    #             attn_sub_np = attn_sub.detach().cpu().numpy()
    #             row = head // 4
    #             col = head % 4
                
    #             im = axs[row, col].imshow(attn_sub_np, cmap='viridis', origin='lower')
    #             axs[row, col].axis('off')  # 关闭坐标轴
    #             axs[row, col].set_title(f'Head {head}')
    #             fig.colorbar(im, ax=axs[row, col], orientation='vertical')
            
    #         plt.tight_layout()
    #         plt.savefig("/home/juniboy97/workspace/Diffusion-Unlearning/EraseAnything/image/nude/combined_heads_idx{}.png".format(bbb))
    #         plt.close()
    
