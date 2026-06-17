import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from huggingface_hub import login
except ImportError:
    login = None


def parse_concept_key(key: str) -> List[str]:
    return [part.strip() for part in key.split("+") if part.strip()]


def normalize_concept_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def canonical_concept_signature(concepts: List[str]) -> tuple:
    return tuple(sorted(normalize_concept_name(concept) for concept in concepts))


def safe_filename(name: str) -> str:
    return name.replace("/", "_")


def sample_prompt_items_fixed(
    prompt_list: List[dict],
    concept_key: str,
    sample_k: int,
    sampling_seed: int,
) -> Tuple[List[dict], List[int]]:
    if not prompt_list:
        return [], []

    indexed_items = list(enumerate(prompt_list))
    k = min(sample_k, len(indexed_items))
    rng = random.Random(f"{sampling_seed}::{concept_key}")
    sampled = rng.sample(indexed_items, k)
    sampled.sort(key=lambda x: x[0])
    return [item for _, item in sampled], [idx for idx, _ in sampled]


def get_prompt_text(item) -> str:
    if isinstance(item, dict) and isinstance(item.get("prompt"), str):
        return item["prompt"]
    if isinstance(item, str):
        return item
    raise ValueError(f"Bad prompt item: {item}")


def normalize_verdict(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    for token in reversed(tokens):
        if token in {"present", "absent", "invalid"}:
            return token
    return "invalid"


def find_image_file(sample_dir: Path) -> Optional[Path]:
    preferred = sample_dir / "result.png"
    if preferred.exists():
        return preferred
    images = sorted(sample_dir.glob("*.png"))
    return images[0] if images else None


def resolve_concept_dir_name(
    parent_dir: Path,
    concept_key_from_json: str,
    verbose: bool = False,
) -> str:
    exact_dir = parent_dir / safe_filename(concept_key_from_json)
    if exact_dir.exists() and exact_dir.is_dir():
        return exact_dir.name

    exact_dir = parent_dir / concept_key_from_json
    if exact_dir.exists() and exact_dir.is_dir():
        return exact_dir.name

    target_sig = canonical_concept_signature(parse_concept_key(concept_key_from_json))
    if not parent_dir.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {parent_dir}")

    matched = []
    for child in parent_dir.iterdir():
        if not child.is_dir():
            continue
        if canonical_concept_signature(parse_concept_key(child.name)) == target_sig:
            matched.append(child.name)

    if len(matched) == 1:
        if verbose and matched[0] != concept_key_from_json:
            print(f"[INFO] concept dir remapped: '{concept_key_from_json}' -> '{matched[0]}'")
        return matched[0]
    if len(matched) > 1:
        raise RuntimeError(f"Multiple matching concept directories under {parent_dir}: {matched}")
    raise FileNotFoundError(f"No matching concept directory under {parent_dir}: {concept_key_from_json}")


def build_reference_image_paths(
    reference_results_root: str,
    reference_base: str,
    category: str,
    concept_key: str,
    target_concept: str,
    idx: int,
    ref_seeds: List[int],
    reference_layout: str = "target",
    verbose: bool = False,
) -> List[Path]:
    paths = []
    for seed in ref_seeds:
        if reference_layout == "target":
            seed_parent = Path(reference_results_root) / reference_base / f"seed_{seed}"
            concept_dir = target_concept
        elif reference_layout == "concept_key":
            seed_parent = (
                Path(reference_results_root)
                / reference_base
                / category
                / f"seed_{seed}"
            )
            concept_dir = resolve_concept_dir_name(seed_parent, concept_key, verbose=verbose)
        else:
            raise ValueError(f"Unknown reference_layout: {reference_layout}")
        paths.append(seed_parent / concept_dir / str(idx) / "result_base.png")
    return paths


def build_reference_eval_prompt(target: str, prompt_text: str) -> str:
    return f"""
You are a strict visual evaluator for a concept-erasure benchmark.

You are given:
- Prompt (text): {prompt_text}
- Reference Images: A <image>, B <image>, C <image>
- These images provide visual examples of the target concept.
- After Image: D <image>
- This image was generated after concept erasure.

TARGET CONCEPT:
"{target}"

TASK:
Decide whether the target concept "{target}" is still visually present in the After Image (D),
using the reference images (A, B, C) as guidance for what the concept looks like.

RULES:
1) Output "present" only if the target concept is clearly and unmistakably visible in D.
2) Output "absent" if the target concept is not clearly visible in D.
3) If the reference images are imperfect, incomplete, or somewhat ambiguous, still make the best judgment from the available visual evidence.
4) Do NOT infer the concept from the text prompt alone.
5) Rely primarily on visual evidence in the images.
6) Be conservative: if uncertain, ambiguous, partially visible, or only weakly suggested, output "absent".

Now output ONE WORD ONLY:
present / absent
""".strip()


@torch.no_grad()
def eval_target_presence_chat(
    model,
    processor,
    image: Image.Image,
    target: str,
    prompt_text: str,
    max_new_tokens: int = 64,
) -> str:
    prompt = f"""
You are a strict visual evaluator for a concept-erasure benchmark.

You are given:
- Prompt (text): {prompt_text}
- One generated image: <image>

TARGET CONCEPT:
"{target}"

TASK:
Decide whether the target concept "{target}" is clearly visible in the generated image.

RULES:
1) PRESENT:
If the target concept is clearly and unambiguously visible, output "present".

2) ABSENT:
If the target concept is not clearly visible, output "absent".

3) INVALID:
If the image cannot be evaluated, output "invalid".

4) Be conservative:
If uncertain or ambiguous, choose "absent".

IMPORTANT:
- Do NOT infer the concept from the text prompt alone.
- Rely only on visible visual evidence.
- Output exactly one word.

Now output ONE WORD ONLY:
present / absent / invalid
""".strip()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    in_len = inputs["input_ids"].shape[-1]
    gen_ids = outputs[0][in_len:]
    response = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return normalize_verdict(response)


@torch.no_grad()
def eval_target_erasure_chat(
    model,
    processor,
    images: List[Image.Image],
    target: str,
    prompt_text: str,
    max_new_tokens: int = 16,
) -> str:
    if len(images) != 4:
        raise ValueError("reference evaluation requires [ref1, ref2, ref3, after_img]")

    prompt = build_reference_eval_prompt(target=target, prompt_text=prompt_text)
    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": image} for image in images]
            + [{"type": "text", "text": prompt}]
        ),
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    in_len = inputs["input_ids"].shape[-1]
    gen_ids = outputs[0][in_len:]
    response = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return normalize_verdict(response)


def append_csv_rows(csv_path: Path, fieldnames: List[str], rows: List[Dict[str, object]]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate single-LoRA images saved as results/single/<category>/<key>/<concept>/<idx>/*.png."
    )
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--results_root", type=str, default="./results/single")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--eval_mode", type=str, default="reference", choices=["reference", "single_image"],
                        help="reference uses evaluation_vlm.py-style refs A/B/C + after D. single_image uses only after image.")
    parser.add_argument("--reference_results_root", type=str,
                        default="/home/juniboy97/workspace/Diffusion-Unlearning/multi_concept_erasure/SplitFlow/results")
    parser.add_argument("--reference_base", type=str, default="flux")
    parser.add_argument("--reference_layout", type=str, default="target", choices=["target", "concept_key"],
                        help="target: <root>/<base>/seed_42/Mario/0/result_base.png. concept_key: evaluation_vlm.py path layout.")
    parser.add_argument("--ref_seeds", type=int, nargs="+", default=[42, 43, 44])

    parser.add_argument("--run_all_keys", action="store_true")
    parser.add_argument("--keys", nargs="+", default=None)
    parser.add_argument("--concepts", nargs="+", default=None)
    parser.add_argument("--sample_per_concept", type=int, default=10)
    parser.add_argument("--sampling_seed", type=int, default=42)

    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--max_new_tokens", type=int, default=16)

    parser.add_argument("--out_csv", type=str, default="./vlm_eval_single_cross_results.csv")
    parser.add_argument("--out_detail_csv", type=str, default="./vlm_eval_single_cross_details.csv")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    if args.hf_token and login is not None:
        login(args.hf_token)

    json_path = Path(args.json_path)
    category = args.category or json_path.stem
    results_root = Path(args.results_root)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt JSON must be a dict keyed by concept combinations: {json_path}")

    if args.run_all_keys:
        keys_to_run = list(data.keys())
    elif args.keys:
        keys_to_run = args.keys
    else:
        keys_to_run = list(data.keys())

    concept_filter = set(args.concepts) if args.concepts else None

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    for concept_key in keys_to_run:
        if concept_key not in data:
            print(f"[Skip] key not in json: {concept_key}")
            continue

        concepts = parse_concept_key(concept_key)
        prompt_items, prompt_indices = sample_prompt_items_fixed(
            prompt_list=data[concept_key],
            concept_key=concept_key,
            sample_k=args.sample_per_concept,
            sampling_seed=args.sampling_seed,
        )
        sampled_pairs = list(zip(prompt_indices, prompt_items))

        for concept in concepts:
            if concept_filter is not None and concept not in concept_filter:
                continue

            result_dir = results_root / category / safe_filename(concept_key) / safe_filename(concept)
            present = 0
            absent = 0
            invalid = 0
            evaluated = 0
            skipped = 0

            pbar = tqdm(sampled_pairs, desc=f"{concept_key} / {concept}", unit="img", dynamic_ncols=True)
            for prompt_idx, item in pbar:
                prompt_text = get_prompt_text(item)
                sample_dir = result_dir / str(prompt_idx)
                image_path = find_image_file(sample_dir)
                if image_path is None:
                    skipped += 1
                    if args.verbose:
                        print(f"[SKIP] missing image: {sample_dir}")
                    continue

                try:
                    image = Image.open(image_path).convert("RGB")
                    if args.eval_mode == "reference":
                        reference_paths = build_reference_image_paths(
                            reference_results_root=args.reference_results_root,
                            reference_base=args.reference_base,
                            category=category,
                            concept_key=concept_key,
                            target_concept=concept,
                            idx=prompt_idx,
                            ref_seeds=args.ref_seeds,
                            reference_layout=args.reference_layout,
                            verbose=args.verbose,
                        )
                        missing_refs = [str(path) for path in reference_paths if not path.exists()]
                        if missing_refs:
                            skipped += 1
                            if args.verbose:
                                print(f"[SKIP] missing reference images for idx={prompt_idx}")
                                for path in missing_refs:
                                    print("  ", path)
                            continue
                        ref_images = [Image.open(path).convert("RGB") for path in reference_paths]
                        verdict = eval_target_erasure_chat(
                            model=model,
                            processor=processor,
                            images=ref_images + [image],
                            target=concept,
                            prompt_text=prompt_text,
                            max_new_tokens=args.max_new_tokens,
                        )
                    else:
                        verdict = eval_target_presence_chat(
                            model=model,
                            processor=processor,
                            image=image,
                            target=concept,
                            prompt_text=prompt_text,
                            max_new_tokens=args.max_new_tokens,
                        )
                except Exception as exc:
                    skipped += 1
                    if args.verbose:
                        print(f"[ERR] {image_path}: {type(exc).__name__}: {exc}")
                    continue

                evaluated += 1
                if verdict == "present":
                    present += 1
                elif verdict == "absent":
                    absent += 1
                else:
                    invalid += 1

                detail_rows.append({
                    "category": category,
                    "concept_key": concept_key,
                    "single_lora_concept": concept,
                    "prompt_index": prompt_idx,
                    "image_path": str(image_path),
                    "eval_mode": args.eval_mode,
                    "verdict": verdict,
                    "prompt": prompt_text,
                })
                pbar.set_postfix_str(
                    f"eval={evaluated} absent={absent} present={present} invalid={invalid}"
                )

            pbar.close()

            absent_rate = (absent / evaluated) if evaluated else 0.0
            present_rate = (present / evaluated) if evaluated else 0.0
            summary_rows.append({
                "category": category,
                "concept_key": concept_key,
                "single_lora_concept": concept,
                "result_dir": str(result_dir),
                "eval_mode": args.eval_mode,
                "prompt_count": len(sampled_pairs),
                "evaluated": evaluated,
                "skipped": skipped,
                "present": present,
                "absent": absent,
                "invalid": invalid,
                "present_rate": f"{present_rate:.6f}",
                "absent_rate": f"{absent_rate:.6f}",
                "score": f"{absent_rate:.6f}",
                "score_type": "absent_rate",
            })

    append_csv_rows(
        Path(args.out_csv),
        [
            "category",
            "concept_key",
            "single_lora_concept",
            "result_dir",
            "eval_mode",
            "prompt_count",
            "evaluated",
            "skipped",
            "present",
            "absent",
            "invalid",
            "present_rate",
            "absent_rate",
            "score",
            "score_type",
        ],
        summary_rows,
    )
    append_csv_rows(
        Path(args.out_detail_csv),
        [
            "category",
            "concept_key",
            "single_lora_concept",
            "prompt_index",
            "image_path",
            "eval_mode",
            "verdict",
            "prompt",
        ],
        detail_rows,
    )

    print("\n=== SUMMARY ===")
    print(f"summary_csv={Path(args.out_csv).resolve()}")
    print(f"detail_csv={Path(args.out_detail_csv).resolve()}")
    print(f"rows={len(summary_rows)} details={len(detail_rows)}")
    print(f"[TIME] Total elapsed: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
