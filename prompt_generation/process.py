import json
import os
import re
from itertools import combinations
from typing import List, Dict
from PIL import Image
import base64
import io
import numpy as np
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from diffusers import FluxPipeline

from huggingface_hub import login

def parse_removed_targets(removed_targets: str | List[str]) -> List[str]:
    """
    Normalize removed_targets into a list of strings.
    Accepts:
      - "Mario + SpongeBob"
      - ["Mario", "SpongeBob"]
    """
    if isinstance(removed_targets, list):
        return [t.strip() for t in removed_targets]

    if isinstance(removed_targets, str):
        return [t.strip() for t in removed_targets.split("+")]

    raise TypeError(f"Unsupported type for removed_targets: {type(removed_targets)}")


def extract_last_json(text: str) -> dict:
    """
    Extract the LAST valid JSON object from a text blob.
    Robust to system/user/assistant prefixes and extra text.
    """

    if not isinstance(text, str):
        raise TypeError("Input to extract_last_json must be a string")

    # 모든 JSON object 후보를 non-greedy로 찾음
    matches = re.findall(r'\{[\s\S]*?\}', text)

    if not matches:
        raise ValueError("No JSON object found in text")

    # 뒤에서부터 하나씩 시도 (마지막 JSON이 실제 답인 경우가 대부분)
    for json_str in reversed(matches):
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            continue

    raise ValueError("Found JSON-like blocks but none were valid JSON")



def build_element_extraction_prompt(
    original_prompt: str,
    removed_targets: List[str],
) -> str:
    
    return f"""
            You are an expert at analyzing image generation prompts.

            The following target concepts have been REMOVED and must be IGNORED:
            {removed_targets}

            Given the ORIGINAL PROMPT below, extract ONLY the remaining
            explicitly specified **external visual elements** that should still appear
            in the generated image.

            ALLOWED ELEMENT TYPES:
            - Backgrounds or environments (e.g., park, street, underwater environment)
            - Physical objects (e.g., table, car, building, trees)
            - Clothing or accessories (e.g., jacket, hat, glasses)
            - Physical attributes or appearance (e.g., beard, long hair, red dress)
            - Spatial or layout descriptors (e.g., indoors, outdoors, foreground, background)

            STRICTLY EXCLUDE:
            - Emotions or feelings (e.g., happy, nervous, angry)
            - Facial expressions (e.g., smiling, frowning)
            - Mental or psychological states
            - Personality traits or intentions
            - Abstract or non-physical concepts
            - Actions or verbs unless they define a **static visual state**
            (e.g., “standing” is allowed, “running” is NOT)

            RULES:
            - Ignore removed target concepts completely.
            - Extract ONLY explicitly stated elements (no inference or guessing).
            - Each element must correspond to something that could be
            directly **seen in a still image**.
            - Each element should be a short, concrete noun phrase.

            ORIGINAL PROMPT:
            "{original_prompt}"

            OUTPUT FORMAT (JSON ONLY):
            {{
            "explicit_elements": []
            }}
            """


def extract_elements_for_prompt_with_retry(
    text_model,
    tokenizer,
    prompt_text: str,
    removed_targets: List[str],
    max_new_tokens: int = 32768,
    max_retry: int = 3,
) -> List[str]:
    last_error = None

    # ✅ 여기서 단 한 번 정규화
    removed_target_list = parse_removed_targets(removed_targets)
    for attempt in range(1, max_retry + 1):
        try:
            user_prompt = build_element_extraction_prompt(
                original_prompt=prompt_text,
                removed_targets=removed_target_list,
            )

            messages = [
                {"role": "system", "content": "You are a precise and literal prompt analyzer."},
                {"role": "user", "content": user_prompt},
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(text_model.device)

            with torch.no_grad():
                outputs = text_model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            parsed = extract_last_json(text)

            elements = parsed.get("explicit_elements", [])

            # sanity check
            if not isinstance(elements, list):
                raise ValueError("explicit_elements is not a list")

            return elements

        except Exception as e:
            last_error = e
            print(f"    ⚠️ Extraction retry {attempt}/{max_retry} failed:", e)

    # -----------------------------
    # Hard failure
    # -----------------------------
    print("    🚨 Extraction failed after retries")
    return []


# =========================
# Main builder
# =========================
def build_prompt_element_json(
    prompt_json_path: str,
    output_json_path: str,
    text_model,
    tokenizer,
    removed_targets_map: Dict[str, str],
):
    """
    Build a JSON mapping from concept_key to a list of
    {prompt, nouns} entries.
    Only prompts with non-empty elements are kept.
    """

    with open(prompt_json_path, "r", encoding="utf-8") as f:
        prompt_data = json.load(f)

    results = {}   # 🔹 key 유지
    total = 0
    kept = 0

    for concept_key, prompts in prompt_data.items():
        print(f"✅start for {concept_key}")
        if not isinstance(prompts, list):
            continue

        removed_targets = removed_targets_map.get(concept_key, concept_key)

        concept_results = []

        for prompt_idx, prompt in enumerate(prompts):
            print(f"extract elements for {prompt}")
            total += 1

            elements = extract_elements_for_prompt_with_retry(
                text_model=text_model,
                tokenizer=tokenizer,
                prompt_text=prompt,
                removed_targets=removed_targets,
            )

            if not elements:
                continue  # ❌ skip prompts without explicit elements

            concept_results.append(
                {
                    "prompt": prompt,
                    "nouns": elements,
                    "index": prompt_idx
                }
            )
            kept += 1
        # 🔹 이 concept에서 하나라도 살아남았을 때만 추가
        if concept_results:
            results[concept_key] = concept_results
        print(f"✅ finish for {concept_key}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n[DONE] Prompt–element JSON created")
    print(f"  Total prompts processed: {total}")
    print(f"  Prompts kept (with elements): {kept}")
    print(f"  Concepts kept: {len(results)}")
    print(f"  Saved to: {output_json_path}")

# -----------------------------
# inference
# -----------------------------
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_json_path",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    # -----------------------------
    # FIXED SETTINGS (as-is)
    # -----------------------------
    model_id = "Qwen/Qwen3-4B-Instruct-2507"

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(hf_token)

    text_tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    text_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    removed_targets_map = {}  # 그대로 유지

    build_prompt_element_json(
        prompt_json_path=args.input_json_path,
        output_json_path=args.output_json_path,
        text_model=text_model,
        tokenizer=text_tokenizer,
        removed_targets_map=removed_targets_map,
    )

if __name__ == "__main__":
    main()
