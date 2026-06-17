import json
import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import argparse


# ======================================================
# Utils
# ======================================================
import json as _json


def extract_last_json(text: str) -> dict:
    s = text.strip()

    # 1) output 전체가 JSON인 경우
    if s.startswith("{") and s.endswith("}"):
        try:
            return _json.loads(s)
        except _json.JSONDecodeError:
            pass

    # 2) text 중간에 JSON이 섞인 경우 -> 마지막 완성 JSON object 추출
    last_obj = None
    start = None
    depth = 0

    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = s[start:i + 1]
                    try:
                        last_obj = _json.loads(candidate)
                    except _json.JSONDecodeError:
                        pass
                    start = None

    if last_obj is not None:
        return last_obj

    raise ValueError("Failed to parse JSON from VLM output")


def normalize_concept_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def parse_concept_key(concept_key: str) -> List[str]:
    """
    "A + B" 또는 "A + B + C" -> ["A", "B"] / ["A", "B", "C"]
    """
    return [x.strip() for x in concept_key.split(" + ") if x.strip()]


def canonical_concept_signature(concepts: List[str]) -> tuple:
    """
    concept 순서 무시 비교용 signature
    """
    return tuple(sorted(normalize_concept_name(c) for c in concepts))


def make_sample_uid(method: str, category: str, concept_key: str, index: int, seed: int) -> str:
    """
    concept 순서가 달라도 동일 sample이면 같은 uid가 되도록 canonicalize
    """
    concepts = parse_concept_key(concept_key)
    canonical_key = " + ".join(sorted(concepts, key=lambda x: normalize_concept_name(x)))
    return f"{method}|||{category}|||{canonical_key}|||{index}|||seed_{seed}"


def present_ratio_from_verdict(per_entity: dict) -> float:
    """
    per_entity: {"entity": "PRESENT"/"ABSENT"/"UNCERTAIN", ...}
    returns: #PRESENT / #total (0~1). If empty -> 0.0
    """
    if not isinstance(per_entity, dict) or len(per_entity) == 0:
        return 0.0
    total = len(per_entity)
    present = sum(1 for v in per_entity.values() if v == "PRESENT")
    return present / total


def append_jsonl(jsonl_path: str, record: dict):
    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_jsonl(jsonl_path: str) -> Dict[str, dict]:
    """
    이미 저장된 중간 결과를 uid 기준 dict로 반환
    jsonl이 없으면 빈 dict 반환
    """
    records = {}

    if not os.path.exists(jsonl_path):
        return records

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                uid = obj.get("uid")
                if uid is not None:
                    records[uid] = obj
            except Exception:
                continue

    return records


def save_summary_json(summary_path: str, summary: dict):
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def chunk_list(lst: List, batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


# ======================================================
# Directory resolver
# ======================================================
def resolve_concept_dir_name(
    parent_dir: Path,
    concept_key_from_json: str,
    verbose: bool = False,
) -> str:
    """
    JSON key의 concept 순서와 실제 폴더 순서가 달라도,
    parent_dir 아래에서 동일한 concept set을 가지는 폴더명을 찾아 반환.

    우선순위:
    1) exact match
    2) parent_dir 내 디렉토리들 중 concept set 동일한 것 탐색
    """
    exact_dir = parent_dir / concept_key_from_json
    if exact_dir.exists() and exact_dir.is_dir():
        return concept_key_from_json

    target_concepts = parse_concept_key(concept_key_from_json)
    target_sig = canonical_concept_signature(target_concepts)

    if not parent_dir.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {parent_dir}")

    matched = []
    for child in parent_dir.iterdir():
        if not child.is_dir():
            continue

        child_concepts = parse_concept_key(child.name)
        child_sig = canonical_concept_signature(child_concepts)

        if child_sig == target_sig:
            matched.append(child.name)

    if len(matched) == 1:
        if verbose and matched[0] != concept_key_from_json:
            print(f"[INFO] concept dir remapped: '{concept_key_from_json}' -> '{matched[0]}'")
        return matched[0]

    if len(matched) > 1:
        raise RuntimeError(
            f"Multiple matching concept directories found under {parent_dir} "
            f"for concept set {target_concepts}: {matched}"
        )

    raise FileNotFoundError(
        f"No matching concept directory found under {parent_dir} "
        f"for JSON concept key '{concept_key_from_json}'"
    )


# ======================================================
# Path builders
# ======================================================
def build_after_image_path(
    results_root: str,
    method_dirname: str,
    category: str,
    seed: int,
    concept_key: str,
    index: int,
    filename_template: str,
    verbose: bool = False,
) -> str:
    """
    after image 경로를 유연하게 찾는다.

    지원 형태 예시:
    1) 일반:
       /.../results/method/category/seed_42/concept_dir/0000/result_comp_0000.png

    2) flux:
       /.../results/flux/category/seed_44/concept_dir/0/result_base.png

    concept_key와 실제 폴더명이 순서만 다른 경우도 자동 처리
    """
    idx4 = f"{index:04d}"
    idx_plain = str(index)

    seed_parent = (
        Path(results_root)
        / method_dirname
        / category
        / f"seed_{seed}"
    )

    resolved_concept_dir = resolve_concept_dir_name(
        parent_dir=seed_parent,
        concept_key_from_json=concept_key,
        verbose=verbose,
    )

    concept_dir = seed_parent / resolved_concept_dir

    # 기본 filename 후보
    filename_candidates = []

    if "{index" in filename_template:
        filename_candidates.append(filename_template.format(index=index))
    else:
        filename_candidates.append(filename_template)

    # flux 예외 처리
    if method_dirname.lower() == "flux":
        if "result_base.png" not in filename_candidates:
            filename_candidates.append("result_base.png")

    # 디렉토리 index 후보
    index_dir_candidates = [idx4, idx_plain]

    # 후보 경로를 순서대로 시도
    candidate_paths = []
    for idx_dir in index_dir_candidates:
        for fname in filename_candidates:
            candidate_paths.append(concept_dir / idx_dir / fname)

    for p in candidate_paths:
        if p.exists():
            if verbose:
                print(f"[INFO] Resolved image path: {p}")
            return str(p)

    # 못 찾으면 디버깅용으로 첫 후보 반환 대신 에러 자세히 출력
    raise FileNotFoundError(
        f"No image found for concept='{concept_key}', index={index}, seed={seed}\n"
        f"Tried paths:\n" + "\n".join(str(p) for p in candidate_paths)
    )

# ======================================================
# Prompt builders
# ======================================================
def build_entity_presence_prompt(prompt_text: str, elements: List[str]) -> str:
    elements_str = "\n".join([f"- {e}" for e in elements])

    return f"""
You are a strict visual evaluator.

You are given:
1) An image generated from a text prompt
2) The original generation prompt
3) A list of visual entities extracted from the prompt

ORIGINAL PROMPT:
"{prompt_text}"

TASK:
For EACH listed entity, determine whether it is visible in the image.

Answer using EXACTLY one of the following labels:
- PRESENT: the entity is clearly and unambiguously visible
- ABSENT: the entity is clearly not visible
- UNCERTAIN: the image is ambiguous or does not provide enough evidence

IMPORTANT RULES:
- Judge ONLY what is visible in the image.
- Do NOT assume or infer missing details.
- If you are not confident, choose UNCERTAIN.
- Each entity must receive exactly one label.

ENTITIES TO EVALUATE:
{elements_str}

OUTPUT FORMAT (JSON ONLY):
{{
  "per_entity": {{
    "<entity>": "PRESENT" | "ABSENT" | "UNCERTAIN"
  }}
}}
"""


# ======================================================
# VLM evaluation
# ======================================================
@torch.no_grad()
def eval_entity_presence_vlm_batch(
    model,
    processor,
    images: List[Image.Image],
    prompt_texts: List[str],
    elements_list: List[List[str]],
    max_new_tokens: int = 512,
) -> List[Dict]:
    """
    여러 샘플을 batch로 넣어서 결과를 list[dict]로 반환
    """
    if not (len(images) == len(prompt_texts) == len(elements_list)):
        raise ValueError("Batch inputs must have the same length")

    messages = []
    for image, prompt_text, elements in zip(images, prompt_texts, elements_list):
        question = build_entity_presence_prompt(prompt_text=prompt_text, elements=elements)
        messages.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
        )

    text_inputs = [
        processor.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True,
        )
        for msg in messages
    ]

    inputs = processor(
        text=text_inputs,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]

    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    results = [extract_last_json(x) for x in decoded]
    return results


# ======================================================
# Aggregation
# ======================================================
def summarize_seed_from_jsonl(
    jsonl_path: str,
    method: str,
    category: str,
    eval_seed: int,
) -> Dict:
    ratios = []

    if not os.path.exists(jsonl_path):
        return {
            "method": method,
            "category": category,
            "seed": eval_seed,
            "evaluated": 0,
            "mean_present_ratio": 0.0,
        }

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception:
                continue

            if row.get("method") != method:
                continue
            if row.get("category") != category:
                continue
            if row.get("seed") != eval_seed:
                continue

            ratio = row.get("present_ratio")
            if ratio is None:
                continue

            ratios.append(float(ratio))

    mean_ratio = sum(ratios) / len(ratios) if len(ratios) > 0 else 0.0

    return {
        "method": method,
        "category": category,
        "seed": eval_seed,
        "evaluated": len(ratios),
        "mean_present_ratio": mean_ratio,
    }


def append_seed_summary_csv(csv_path: str, row: Dict):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "category", "seed", "mean_present_ratio", "evaluated"]
        )
        if not exists:
            writer.writeheader()
        writer.writerow({
            "method": row["method"],
            "category": row["category"],
            "seed": row["seed"],
            "mean_present_ratio": f"{row['mean_present_ratio']:.6f}",
            "evaluated": row["evaluated"],
        })


# ======================================================
# Main
# ======================================================
def main():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument(
        "--results_root",
        type=str,
        default="/home/juniboy97/workspace/Diffusion-Unlearning/multi_concept_erasure/SplitFlow/results",
    )
    parser.add_argument("--category", type=str, default="intra_2_character")
    parser.add_argument("--method_dirname", type=str, default="flowblending_multi_scale_ema")
    parser.add_argument(
        "--image_filename",
        type=str,
        default="result_comp_{index:04d}.png",
        help='예: "result_comp_{index:04d}.png"',
    )

    parser.add_argument(
        "--element_json",
        type=str,
        default="/home/juniboy97/workspace/Diffusion-Unlearning/multi_concept_erasure/SplitFlow/prompts/prompt_final/selective_alignment/intra_2_character.json",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="./log/intra_2_character_seedwise.csv",
        help="seed별 summary를 append 저장",
    )
    parser.add_argument(
        "--save_jsonl",
        type=str,
        default="./log/intra_2_character_results.jsonl",
        help="중간 결과 jsonl 저장",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default="./log/intra_2_character_summary.json",
        help="최종 summary json 저장",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="CSV/JSON에 기록할 method 이름 (미지정 시 method_dirname 사용)",
    )

    # vlm
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--max_new_tokens", type=int, default=512)

    # runtime
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    total_start_time = time.time()

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    # load json
    with open(args.element_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"Invalid/empty JSON: {args.element_json}")

    # load previous intermediate results
    existing_records = load_existing_jsonl(args.save_jsonl)
    if args.verbose:
        print(f"[INFO] Loaded {len(existing_records)} existing records from {args.save_jsonl}")

    # load vlm once
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_id)

    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    method_name = args.method if args.method is not None else args.method_dirname

    total = 0
    done = 0
    skipped = 0
    resumed = 0
    failed = 0

    # -----------------------------------
    # seed loop
    # -----------------------------------
    for eval_seed in args.eval_seeds:
        print(f"\n{'=' * 80}")
        print(f"[EVAL SEED] {eval_seed}")
        print(f"{'=' * 80}")

        seed_start_time = time.time()

        for concept_key, entries in data.items():
            if not isinstance(entries, list):
                if args.verbose:
                    print(f"[WARN] concept={concept_key}: entries not list. skip.")
                continue

            print(f"\n[Concept] {concept_key} | seed={eval_seed}")
            concept_start = time.time()

            valid_samples = []

            # -----------------------------------
            # 1) 샘플 수집 + 기존 중간결과 resume
            # -----------------------------------
            for index, entry in enumerate(entries):
                total += 1

                if not isinstance(entry, dict) or "prompt" not in entry or "nouns" not in entry:
                    skipped += 1
                    if args.verbose:
                        print(f"[WARN] concept={concept_key}: bad entry format. skip.")
                    continue

                prompt = entry["prompt"]
                elements = entry["nouns"]

                if not isinstance(elements, list) or not isinstance(index, int):
                    skipped += 1
                    if args.verbose:
                        print(f"[WARN] concept={concept_key}: invalid nouns/index. skip.")
                    continue

                uid = make_sample_uid(method_name, args.category, concept_key, index, eval_seed)
        
                # 이미 처리된 결과가 있으면 재사용
                if uid in existing_records:
                    done += 1
                    resumed += 1
                    continue
                
                try:
                    img_path = build_after_image_path(
                        results_root=args.results_root,
                        method_dirname=args.method_dirname,
                        category=args.category,
                        seed=eval_seed,
                        concept_key=concept_key,
                        index=index,
                        filename_template=args.image_filename,
                        verbose=args.verbose,
                    )
                except Exception as e:
                    skipped += 1
                    failed += 1
                    if args.verbose:
                        print(f"[ERR-PATH] {concept_key} | index={index} | seed={eval_seed} :: {type(e).__name__}: {e}")
                    continue

                if not os.path.exists(img_path):
                    skipped += 1
                    if args.verbose:
                        print(f"[WARN] Missing image: {img_path}")
                    continue

                valid_samples.append(
                    {
                        "uid": uid,
                        "method": method_name,
                        "category": args.category,
                        "seed": eval_seed,
                        "concept_key": concept_key,
                        "index": index,
                        "prompt": prompt,
                        "elements": elements,
                        "img_path": img_path,
                    }
                )

            # -----------------------------------
            # 2) batch inference
            # -----------------------------------
            batch_groups = list(chunk_list(valid_samples, args.batch_size))
            pbar = tqdm(batch_groups, desc=f"{concept_key} | seed={eval_seed}", leave=False)

            for batch_samples in pbar:
                batch_images = []
                batch_prompts = []
                batch_elements = []
                batch_meta = []

                # image load
                for sample in batch_samples:
                    try:
                        image = Image.open(sample["img_path"]).convert("RGB")
                        batch_images.append(image)
                        batch_prompts.append(sample["prompt"])
                        batch_elements.append(sample["elements"])
                        batch_meta.append(sample)
                    except Exception as e:
                        failed += 1
                        skipped += 1
                        print(f"[ERR-LOAD] {sample['concept_key']} | index={sample['index']} | seed={sample['seed']} :: {type(e).__name__}: {e}")

                if len(batch_images) == 0:
                    continue

                # batch vlm
                try:
                    verdicts = eval_entity_presence_vlm_batch(
                        model=model,
                        processor=processor,
                        images=batch_images,
                        prompt_texts=batch_prompts,
                        elements_list=batch_elements,
                        max_new_tokens=args.max_new_tokens,
                    )
                    
                    for sample, verdict in zip(batch_meta, verdicts):
                        per_entity = verdict.get("per_entity", {})
                        r = present_ratio_from_verdict(per_entity)
                        done += 1

                        record = {
                            "uid": sample["uid"],
                            "method": sample["method"],
                            "category": sample["category"],
                            "seed": sample["seed"],
                            "concept_key": sample["concept_key"],
                            "index": sample["index"],
                            "prompt": sample["prompt"],
                            "elements": sample["elements"],
                            "img_path": sample["img_path"],
                            "verdict": verdict,
                            "present_ratio": r,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        append_jsonl(args.save_jsonl, record)
                        existing_records[sample["uid"]] = record

                    pbar.set_postfix(done=done, skipped=skipped, resumed=resumed, failed=failed)

                except Exception as e:
                    print(f"[ERR-BATCH] {concept_key} | seed={eval_seed} :: {type(e).__name__}: {e}")
                    print("[INFO] Fallback to per-sample inference for this batch.")

                    # batch 실패 시 개별 fallback
                    for sample, image in zip(batch_meta, batch_images):
                        try:
                            verdict = eval_entity_presence_vlm_batch(
                                model=model,
                                processor=processor,
                                images=[image],
                                prompt_texts=[sample["prompt"]],
                                elements_list=[sample["elements"]],
                                max_new_tokens=args.max_new_tokens,
                            )[0]
                            
                            per_entity = verdict.get("per_entity", {})
                            r = present_ratio_from_verdict(per_entity)
                            done += 1

                            record = {
                                "uid": sample["uid"],
                                "method": sample["method"],
                                "category": sample["category"],
                                "seed": sample["seed"],
                                "concept_key": sample["concept_key"],
                                "index": sample["index"],
                                "prompt": sample["prompt"],
                                "elements": sample["elements"],
                                "img_path": sample["img_path"],
                                "verdict": verdict,
                                "present_ratio": r,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                            append_jsonl(args.save_jsonl, record)
                            existing_records[sample["uid"]] = record

                        except Exception as e2:
                            failed += 1
                            skipped += 1
                            print(f"[ERR-SINGLE] {sample['concept_key']} | index={sample['index']} | seed={sample['seed']} :: {type(e2).__name__}: {e2}")

            concept_time = time.time() - concept_start
            print(f"[DONE] {concept_key} | seed={eval_seed} took {concept_time:.2f} sec")

        seed_elapsed = time.time() - seed_start_time
        print(f"\n[SEED DONE] seed={eval_seed} took {seed_elapsed/60:.2f} minutes ({seed_elapsed:.2f} sec)")

    # -----------------------------------
    # 3) seed별 summary 계산 및 csv 저장
    # -----------------------------------
    seed_summaries = []
    for eval_seed in args.eval_seeds:
        summary_row = summarize_seed_from_jsonl(
            jsonl_path=args.save_jsonl,
            method=method_name,
            category=args.category,
            eval_seed=eval_seed,
        )
        seed_summaries.append(summary_row)
        append_seed_summary_csv(args.out_csv, summary_row)

    overall_ratios = []
    for row in seed_summaries:
        if row["evaluated"] > 0:
            overall_ratios.append((row["mean_present_ratio"], row["evaluated"]))

    total_eval_count = sum(row["evaluated"] for row in seed_summaries)
    weighted_mean = (
        sum(m * n for m, n in overall_ratios) / total_eval_count
        if total_eval_count > 0 else 0.0
    )

    total_time = time.time() - total_start_time

    summary = {
        "method": method_name,
        "results_root": args.results_root,
        "category": args.category,
        "method_dirname": args.method_dirname,
        "image_filename": args.image_filename,
        "element_json": args.element_json,
        "model_id": args.model_id,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "eval_seeds": args.eval_seeds,
        "max_new_tokens": args.max_new_tokens,
        "per_seed_summary": seed_summaries,
        "overall_weighted_mean_present_ratio": weighted_mean,
        "overall_evaluated": total_eval_count,
        "total": total,
        "done": done,
        "skipped": skipped,
        "resumed": resumed,
        "failed": failed,
        "elapsed_sec": total_time,
        "elapsed_min": total_time / 60.0,
        "jsonl_path": args.save_jsonl,
        "csv_path": args.out_csv,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_summary_json(args.summary_json, summary)

    print(f"\n{'=' * 80}")
    print("=== FINAL SUMMARY ===")
    print(f"method={method_name}")
    print(f"category={args.category}")
    print(f"eval_seeds={args.eval_seeds}")
    print(f"overall_weighted_mean_present_ratio={weighted_mean:.6f}")
    print(f"overall_evaluated={total_eval_count}")
    print(f"total={total}, done={done}, skipped={skipped}, resumed={resumed}, failed={failed}")
    print(f"intermediate_jsonl={os.path.abspath(args.save_jsonl)}")
    print(f"summary_csv={os.path.abspath(args.out_csv)}")
    print(f"summary_json={os.path.abspath(args.summary_json)}")
    print(f"=== TOTAL TIME: {total_time/60:.2f} minutes ({total_time:.2f} sec) ===")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()