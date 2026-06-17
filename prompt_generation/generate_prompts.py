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

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(hf_token)
# =========================
# Config
# =========================
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


# =========================
# Utilities
# =========================
def is_bad_background(prompt: str) -> bool:
    """Heuristic filter for background-less prompts"""
    bad_patterns = [
        r"white background",
        r"plain background",
        r"solid background",
        r"isolated",
        r"studio (shot|lighting)",
        r"cutout",
        r"on a blank background",
        r"no background",
    ]
    prompt_l = prompt.lower()
    return any(re.search(p, prompt_l) for p in bad_patterns)


def safe_json_load(s: str):
    s = s.strip()

    # 바깥 중괄호 개수 맞추기
    open_braces = s.count("{")
    close_braces = s.count("}")

    if close_braces < open_braces:
        s += "}" * (open_braces - close_braces)

    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        print("❌ JSON 파싱 실패")
        print(e)
        print("----- RAW STRING -----")
        print(s[-500:])  # 뒤쪽 일부만 출력
        raise

def extract_last_json(text: str) -> dict:
    """
    Extract the LAST valid JSON object from a text blob.
    Robust against system/user prefixes and extra text.
    """
    # DOTALL: 줄바꿈 포함
    matches = re.findall(r'\{[\s\S]*?\}', text)

    if not matches:
        raise ValueError("No JSON object found in LLM output")

    last_json_str = matches[-1]

    try:
        return safe_json_load(last_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON:\n{last_json_str}") from e

def is_placeholder_prompt_list(prompts: list[str]) -> bool:
    placeholder_tokens = [
        "prompt1",
        "prompt2",
        "exactly",
        "...",
    ]

    for p in prompts:
        if any(tok in p.lower() for tok in placeholder_tokens):
            return True
        if len(p.split()) < 6:  # 너무 짧으면 의심
            return True

    return False

def run_llm(user_msg: str) -> Dict:
    messages = [
        {"role": "system", "content": "You are a helpful assistant and an expert prompt writer."},
        {"role": "user", "content": user_msg},
    ]

    input_ids = text_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(text_model.device)

    with torch.no_grad():
        output = text_model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
        )

    text = text_tokenizer.decode(output[0], skip_special_tokens=True)
    return extract_last_json(text)


# =========================
# Prompt Template
# =========================
def build_user_msg(concept_key: List[str]) -> str:
    # return (f"""
    #         You are a skilled prompt writer creating natural, vivid image generation prompts
    #         for benchmarking multi-concept erasure in text-to-image models.
    #         Each prompt should describe a single coherent scene that is easy to visualize.
    #         Vary sentence structure and phrasing; avoid repeating templates.

    #         TASK:
    #         For the following concept combination: {concept_key},
    #         generate EXACTLY {NUM_PROMPTS_PER_CONCEPT} distinct image prompts that include ALL of the listed concepts
    #         together in the same image.
    #         Do NOT generate prompts for any other concepts or combinations.

    #         CONSTRAINTS:
    #         Each fictional character must be depicted as the original canonical character.
    #         Celebrities must be depicted as real human individuals.
    #         Do NOT reference games, movies, franchises, brands, roles, or companies.
    #         Do NOT use costumes, toys, statues, or abstract representations.

    #         The subjects MUST be clearly visible from the front.
    #         Do NOT obscure faces or bodies with extreme angles, back views, or heavy occlusion.
    #         The viewer should be able to clearly identify all concepts from a frontal view.

    #         The scene MUST include a clearly described background or environment.
    #         Do NOT generate prompts with plain, empty, minimal, white, or solid-color backgrounds.
    #         Do NOT describe isolated subjects or studio-style shots.
    #         The background must meaningfully contextualize the interaction.

    #         The background MUST be neutral with respect to the concepts.
    #         Do NOT include locations, objects, symbols, or environments that are canonically or semantically associated
    #         with any of the listed concepts.
    #         Do NOT use concept-specific places, items, or visual motifs as background cues.

    #         OUTPUT FORMAT:
    #         Return ONLY a valid JSON object with the following structure:

    #         {{
    #         "{concept_key}": [
    #             "prompt1",
    #             "prompt2",
    #             "... (exactly {NUM_PROMPTS_PER_CONCEPT} total)"
    #         ]
    #         }}

    #         EXAMPLES:

    #         Concepts: Mario + SpongeBob SquarePants
    #         Good prompt example:
    #         Mario and SpongeBob SquarePants are sitting side by side at a wooden table in a quiet countryside café,
    #         both facing forward toward the viewer with their faces clearly visible,
    #         while sunlight streams through large windows and other patrons sit and talk in the background.

    #         Concepts: Angelina Jolie + Brad Pitt
    #         Good prompt example:
    #         Angelina Jolie and Brad Pitt are standing in a busy farmers' market, both facing the viewer, as people browse fresh produce and handmade goods.

    #         Concepts: Mickey Mouse + Leonardo DiCaprio
    #         Good prompt example:
    #         Mickey Mouse and Leonardo DiCaprio are sitting on a bench in a park with a small playground nearby, 
    #         both facing forward, with children laughing and parents watching in the background.
    #         """
    #         )
    
    return (f"""You are a skilled prompt writer creating natural, vivid image generation prompts
            for benchmarking multi-concept erasure in text-to-image models.
            Each prompt should describe a single coherent scene that is easy to visualize.
            Vary sentence structure and phrasing; avoid repeating templates.

            TASK:
            For the following concept combination: {concept_key},
            generate EXACTLY {NUM_PROMPTS_PER_CONCEPT} distinct image prompts that include ALL of the listed concepts
            together in the same image.

            IMPORTANT:
            You MUST mix two prompt styles: SIMPLE and COMPLEX.
            The ratio between SIMPLE and COMPLEX prompts MUST be RANDOM.
            Do NOT output them in labeled groups — just return a single mixed list of {NUM_PROMPTS_PER_CONCEPT} prompts.

            Do NOT generate prompts for any other concepts or combinations.

            --------------------------------------------------
            SIMPLE PROMPT STYLE (lightweight scenes)
            --------------------------------------------------
            Simple prompts should be minimal and straightforward.

            Rules for SIMPLE-style prompts:
            - Describe ONLY that the listed concepts are together in the same scene.
            - Focus on a shared environment or background.
            - Do NOT describe detailed or dynamic actions.
            - At most ONE mild static posture is allowed (e.g., sitting, standing).
            - Avoid spatial relationships beyond basic co-presence.

            Example style (do NOT copy verbatim):
            "Mario and SpongeBob SquarePants are together in a public park, both facing forward,
            with trees and walking paths behind them."

            --------------------------------------------------
            COMPLEX PROMPT STYLE (rich scenes)
            --------------------------------------------------
            Complex prompts must include richer interactions, motion, or spatial diversity.

            Rules for COMPLEX-style prompts:
            - Include CLEAR and DISTINCT actions, positions, or motion.
            - The concepts MAY perform different actions simultaneously.
            - Describe spatial relationships (e.g., left/right, foreground/background, above/below).
            - Actions can be asymmetric or independent.
            - Movement, gestures, or physical interaction with the environment are encouraged.
            - Maintain a single coherent scene.

            DIVERSITY REQUIREMENT FOR COMPLEX PROMPTS:
            - Do NOT reuse the same sentence structure more than once.
            - Vary how actions are described (e.g., "is running", "is mid-jump", "leans forward", "gestures toward", "walks past", "stands near", etc.).
            - Vary how spatial relationships are described (e.g., foreground/background, left/right, near/far, above/below).
            - Avoid repeating the same opening phrase (e.g., do NOT start multiple prompts with "Mario is..." or "SpongeBob is...").

            Example styles (do NOT copy verbatim):
            "Mario is mid-jump near the center of a lively plaza while SpongeBob SquarePants runs past him
            on a stone walkway, both facing forward as pedestrians watch from behind."

            "SpongeBob SquarePants leans forward as if hurrying along a sidewalk,
            while Mario stands nearby pointing ahead, with storefronts and street signs behind them."

            "Mario walks slowly along a tree-lined path as SpongeBob SquarePants skips ahead,
            with sunlight filtering through the leaves and a small café visible in the background."

            --------------------------------------------------
            GLOBAL CONSTRAINTS (APPLY TO ALL PROMPTS)
            --------------------------------------------------
            Each fictional character must be depicted as the original canonical character.
            Celebrities must be depicted as real human individuals.

            Do NOT reference games, movies, franchises, brands, roles, or companies.
            Do NOT use costumes, toys, statues, or abstract representations.

            The subjects MUST be clearly visible from the front.
            Do NOT obscure faces or bodies with extreme angles, back views, or heavy occlusion.
            The viewer should be able to clearly identify all concepts from a frontal view.

            The scene MUST include a clearly described background or environment.
            Do NOT generate prompts with plain, empty, minimal, white, or solid-color backgrounds.
            Do NOT describe isolated subjects or studio-style shots.
            The background must meaningfully contextualize the interaction.

            The background MUST be neutral with respect to the concepts.
            Do NOT include locations, objects, symbols, or environments that are canonically or semantically associated
            with any of the listed concepts.
            Do NOT use concept-specific places, items, or visual motifs as background cues.

            --------------------------------------------------
            OUTPUT FORMAT:
            --------------------------------------------------
            Return ONLY a valid JSON object with the following structure:

            {{
            "{concept_key}": [
                "prompt 1",
                "prompt 2",
                "prompt 3",
                "... (exactly {NUM_PROMPTS_PER_CONCEPT} total)"
            ]
            }}

            Do NOT add explanations.
            Do NOT add extra fields.
            Return ONLY valid JSON.
            """)


# =========================
# Verify Prompts
# =========================
def encode_image_to_base64(img: Image.Image) -> str:
    """Encode PIL image to base64 string for LLM input."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_verification_prompt(concepts: List[str]) -> str:
    concept_text = ", ".join(concepts)

    return f"""
        You are a strict evaluator for a benchmark dataset used to assess
        multi-concept erasure in text-to-image generation models.

        You will be given:
        1) A list of target concepts
        2) One image generation prompt
        3) Three images generated independently from the same prompt using different random seeds

        ----------------------------------
        TARGET CONCEPTS
        ----------------------------------
        {concept_text}

        ----------------------------------
        CHECKLIST
        ----------------------------------

        1. Concept Preservation:
        Each target concept MUST be clearly, unambiguously, and correctly visible
        in ALL THREE images.
        Partial, ambiguous, stylized, or incorrect depictions do NOT count.

        2. Multi-Concept Coexistence:
        All target concepts must appear together in the SAME scene within each image.
        They must not appear separately or as background-only elements.

        3. Prompt–Image Faithfulness:
        The images must faithfully reflect the provided prompt in terms of
        scene, interaction, and environment.

        4. Robustness Across Seeds:
        The same target concepts and scene semantics must be preserved
        consistently across all three images.

        ----------------------------------
        DECISION RULE
        ----------------------------------
        - The prompt PASSES only if ALL checklist items PASS.
        - If ANY checklist item FAILS, the final decision must be FAIL.

        ----------------------------------
        OUTPUT FORMAT
        ----------------------------------
        Return ONLY a valid JSON object with the following structure:

        {{
        "concept_preservation": "PASS" or "FAIL",
        "multi_concept_coexistence": "PASS" or "FAIL",
        "prompt_image_faithfulness": "PASS" or "FAIL",
        "robustness_across_seeds": "PASS" or "FAIL",
        "final_decision": "PASS" or "FAIL"
        }}

        Do NOT include explanations or extra text.
        """


def verify_prompt_with_vl(
    processor,
    model,
    concepts: List[str],
    prompt_text: str,
    images: List[Image.Image],
    max_new_tokens: int = 512,
) -> Dict:
    """
    Verify whether a prompt reliably preserves target concepts in generated images
    using Qwen3-VL.
    """

    assert len(images) == 3, "Exactly 3 images are required."

    # -----------------------------
    # Build verification text
    # -----------------------------
    verification_text = (
        build_verification_prompt(concepts)
        + "\n\n----------------------------------\n"
        + "IMAGE GENERATION PROMPT\n"
        + "----------------------------------\n"
        + prompt_text
    )

    # -----------------------------
    # Build messages (Qwen3-VL format)
    # -----------------------------
    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": verification_text}]
            ),
        }
    ]

    # -----------------------------
    # Prepare inputs
    # -----------------------------
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # -----------------------------
    # Inference
    # -----------------------------
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )

    # Trim prompt tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    verdict = json.loads(output_text)

    return verdict


def verify_prompt_json_dataset(
    processor,
    model,
    flux_pipeline,              
    input_json_path: str,
    output_json_path: str,
    seeds: List[int] = [0, 1, 2],
    max_prompts_per_key: int | None = None,
    save_path: str = "./results/prompts",
):
    """
    Verify a full prompt JSON dataset using FLUX + LLM evaluator.

    Input JSON format:
    {
      "Concept A + Concept B": [
        "prompt 1",
        "prompt 2",
        "... (exactly 20 total)"
      ]
    }

    Output JSON format:
    {
      "Concept A + Concept B": [
        "verified prompt 1",
        "verified prompt 2",
        ...
      ]
    }
    """

    # -----------------------------
    # Load generated prompts
    # -----------------------------
    with open(input_json_path, "r", encoding="utf-8") as f:
        prompt_dict = json.load(f)

    final_verified: Dict[str, List[str]] = {}

    # -----------------------------
    # Iterate over concept keys
    # -----------------------------
    for concept_key, prompts in prompt_dict.items():
        print(f"\n[INFO] Verifying concept: {concept_key}")

        if not isinstance(prompts, list):
            print("  [WARN] prompts is not a list, skipping:", type(prompts))
            continue

        concepts = [c.strip() for c in concept_key.split("+")]
        verified_prompts: List[str] = []

        # -----------------------------
        # Optional cap for speed
        # -----------------------------
        if max_prompts_per_key is not None:
            prompts = prompts[:max_prompts_per_key]

        # -----------------------------
        # Iterate over prompts
        # -----------------------------
        for idx, prompt in enumerate(prompts):
            print(f"  → Prompt {idx+1}/{len(prompts)}")

            # -----------------------------
            # Generate images with FLUX
            # -----------------------------
            images: List[Image.Image] = []
            for seed in seeds:
                img = flux_pipeline(
                    prompt,
                    height=512,
                    width=512,
                    guidance_scale=3.5,
                    num_inference_steps=28,
                    generator=torch.Generator("cpu").manual_seed(seed)
                ).images[0]

                image_path = f"{save_path}/{concept_key}/{idx}"
                os.makedirs(image_path, exist_ok=True)
                img.save(os.path.join(image_path, f"{seed}.png"))

                images.append(img)

            # -----------------------------
            # Verify with LLM
            # -----------------------------
            try:
                verdict = verify_prompt_with_vl(
                    processor=processor,
                    model=model,
                    concepts=concepts,
                    prompt_text=prompt,
                    images=images,
                )
            except Exception as e:
                print("    [WARN] Verification failed (exception):", e)
                continue

            if verdict.get("final_decision") == "PASS":
                print("    ✅ PASS")
                verified_prompts.append(prompt)
            else:
                print("    ❌ FAIL:", verdict)

        if verified_prompts:
            final_verified[concept_key] = verified_prompts

    # -----------------------------
    # Save verified dataset
    # -----------------------------
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_verified, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Verified benchmark saved to: {output_json_path}")


# =========================
# Dataset Generation
# =========================
def generate_prompts(user_msg: str):
    MAX_RETRY = 5

    for attempt in range(1, MAX_RETRY + 1):
        print(f"\n🔁 [generate_prompts] Attempt {attempt}/{MAX_RETRY}")

        try:
            result = run_llm(user_msg)
        except Exception as e:
            print("❌ run_llm failed:", e)
            continue

        # -------------------------
        # 1) Top-level type check
        # -------------------------
        if not isinstance(result, dict) or len(result) == 0:
            print("❌ result is not a non-empty dict:", type(result))
            print("RAW:", repr(result)[:500])
            continue

        valid = True

        # -------------------------
        # 2) Iterate concept blocks
        # -------------------------
        for concept_key, prompts in result.items():
            print("  → Checking concept_key:", concept_key)
            print("    TYPE(prompts):", type(prompts))

            # 2-1) prompts must be list
            if not isinstance(prompts, list):
                print("❌ prompts is not list:", repr(prompts)[:200])
                valid = False
                break

            # 2-2) placeholder / short prompt check
            if is_placeholder_prompt_list(prompts):
                print("❌ placeholder detected in:", concept_key)
                valid = False
                break

            # 2-3) per-prompt sanity check
            for i, p in enumerate(prompts):
                if not isinstance(p, str):
                    print(f"❌ prompt[{i}] is not str:", type(p))
                    valid = False
                    break

                if len(p.split()) < 8:
                    print(f"❌ prompt[{i}] too short:", p)
                    valid = False
                    break

            if not valid:
                break

        # -------------------------
        # 3) Accept or retry
        # -------------------------
        if valid:
            print("✅ Prompt generation accepted")
            return result

        print("🔄 Invalid result → retrying...")

    # -------------------------
    # 4) Hard failure
    # -------------------------
    print("🚨 MAX_RETRY exceeded. Failed to generate valid prompts.")
    return None

def generate_dataset(concept_groups: List[List[str]], save_path: str):
    keys = [" + ".join(group) for group in concept_groups]

    clean_result: Dict[str, List[str]] = {}
    
    for key in keys:
        print(f"\n[INFO] Generating prompts for: {key}")

        user_msg = build_user_msg(key)

        result = generate_prompts(user_msg)
        if result is None:
            print("⚠️ LLM failed, skipping this concept")
            continue

        # -------------------------
        # result should be: { key: [prompt1, ...] }
        # -------------------------
        if key not in result:
            print("❌ Key mismatch in LLM output")
            print("  expected:", key)
            print("  got keys:", list(result.keys()))
            continue

        prompts = result[key]

        if not isinstance(prompts, list):
            print("❌ prompts is not list:", type(prompts))
            continue

        clean_result[key] = list(prompts)

        print(f"✅ {key} prompt generation finished")

    # -------------------------
    # Save generated dataset
    # -------------------------
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(clean_result, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Saved → {save_path}")

# =========================
# Prompt Augmentation (fill 부족분)
# =========================

def load_json_or_empty(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def dedup_preserve_order(prompts: List[str]) -> List[str]:
    seen = set()
    out = []
    for p in prompts:
        if isinstance(p, str) and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def find_insufficient_concepts_from_expected(
    expected_keys: List[str],
    verified_json_path: str,
    min_required: int = 100,
) -> List[str]:
    """
    verified_json_path(after.json)의 각 key prompt 개수가 min_required 미만인 key들을 반환.
    (키가 아예 없어서 0개인 경우도 포함)
    """
    data = load_json_or_empty(verified_json_path)
    insufficient = []
    for k in expected_keys:
        lst = data.get(k, [])
        if not isinstance(lst, list):
            lst = []
        if len(lst) < min_required:
            insufficient.append(k)
    return insufficient


def build_augmented_user_msg(
    concept_key: str,
    n_new: int,
    existing_prompts: List[str],
    nonce: int,
) -> str:
    """
    기존 instruction + augmentation rule + 최근 existing examples
    + nonce(랜덤 태그)를 넣어서 같은 seed여도 다른 출력을 유도
    """
    # 최근 30개만 예시로 제공 (너무 길면 성능/속도에 안좋음)
    recent = existing_prompts[-30:] if existing_prompts else []

    return f"""
{build_user_msg(concept_key)}

--------------------------------------------------
IMPORTANT (AUGMENTATION MODE)
--------------------------------------------------
You are generating ADDITIONAL prompts to expand an existing dataset.

Generate EXACTLY {n_new} NEW prompts for the SAME concept combination: {concept_key}.

STRICT RULES:
- Do NOT repeat any existing prompts exactly.
- Do NOT paraphrase or minimally modify existing prompts.
- Do NOT reuse the same scene layouts, backgrounds, or sentence structures.
- Each new prompt MUST be semantically distinct from the provided examples.

DIVERSITY TAG (do not mention this tag in the output):
nonce={nonce}

EXISTING PROMPTS (DO NOT COPY OR PARAPHRASE):
{json.dumps(recent, indent=2, ensure_ascii=False)}

OUTPUT FORMAT (JSON ONLY):
{{
  "{concept_key}": [
    "... (exactly {n_new} new prompts)"
  ]
}}
""".strip()


def run_llm_with_generator_seed(user_msg: str, seed: int) -> Dict:
    """
    text_model.generate에 generator seed를 넣어서
    '전역 seed를 고정해도' 호출마다 다른 출력을 뽑을 수 있게 함.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant and an expert prompt writer."},
        {"role": "user", "content": user_msg},
    ]

    input_ids = text_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(text_model.device)

    # generator device는 input_ids device type에 맞춤
    gen_device = "cuda" if input_ids.device.type == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)

    with torch.no_grad():
        output = text_model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            generator=generator,
        )

    text = text_tokenizer.decode(output[0], skip_special_tokens=True)
    return extract_last_json(text)


def verify_single_prompt_flux_qwenvl(
    processor,
    model,
    flux_pipeline,
    concept_key: str,
    prompt: str,
    seeds: List[int],
    save_dir: str,
    save_idx: int,
) -> bool:
    """
    prompt 하나에 대해:
    - FLUX로 seed 3개 이미지 생성
    - Qwen3-VL로 PASS/FAIL
    - PASS면 이미지 저장
    """
    concepts = [c.strip() for c in concept_key.split("+")]

    images: List[Image.Image] = []
    image_path = os.path.join(save_dir, concept_key, str(save_idx))
    os.makedirs(image_path, exist_ok=True)

    for s in seeds:
        # ✅ generator는 cuda 권장 (이미 pipe가 cuda면 cuda generator가 자연스럽고 재현도 좋음)
        gen = torch.Generator("cuda").manual_seed(int(s))

        img = flux_pipeline(
            prompt,
            height=512,
            width=512,
            guidance_scale=3.5,
            num_inference_steps=28,
            generator=gen,
        ).images[0]

        images.append(img)

    try:
        verdict = verify_prompt_with_vl(
            processor=processor,
            model=model,
            concepts=concepts,
            prompt_text=prompt,
            images=images,
        )
    except Exception as e:
        print("    [WARN] Verification exception:", e)
        return False

    if verdict.get("final_decision") == "PASS":
        # 저장
        for i, img in enumerate(images):
            img.save(os.path.join(image_path, f"{i}.png"))
        return True

    return False


def augment_verified_dataset_to_min(
    processor,
    model,
    flux_pipeline,
    expected_keys: List[str],
    verified_json_path: str,
    min_required: int = 100,
    seeds: List[int] = [42, 43, 41],
    batch_new_prompts: int = 20,
    max_rounds_per_key: int = 20,
    save_path: str = "./results/augmented_prompts",
):
    """
    after.json(verified_json_path)을 읽어서,
    key별 verified prompt 개수가 min_required 미만이면
    추가 생성 + 검증해서 채워 넣고, verified_json_path를 update 저장한다.

    - batch_new_prompts: 한 번 LLM에게 추가 생성 요청할 개수
    - max_rounds_per_key: key 하나당 재시도 라운드 상한
    """

    data = load_json_or_empty(verified_json_path)
    updated_any = False

    insufficient = find_insufficient_concepts_from_expected(
        expected_keys=expected_keys,
        verified_json_path=verified_json_path,
        min_required=min_required,
    )

    print(f"\n[Augment] insufficient keys: {len(insufficient)} (min_required={min_required})")

    for concept_key in insufficient:
        cur = data.get(concept_key, [])
        if not isinstance(cur, list):
            cur = []
        cur = dedup_preserve_order(cur)

        need = max(0, min_required - len(cur))
        if need == 0:
            data[concept_key] = cur
            continue

        print(f"\n[Augment] {concept_key}: current={len(cur)} need={need}")

        existing_set = set(cur)
        save_idx = len(cur)  # 저장 폴더 index 이어붙이기

        rounds = 0
        while len(cur) < min_required and rounds < max_rounds_per_key:
            rounds += 1

            # 이번 라운드에 생성할 개수 (너무 크게 주면 중복/quality 이슈)
            n_new = min(batch_new_prompts, min_required - len(cur))

            # nonce + seed를 매 라운드마다 바꾸면 같은 global seed여도 다른 출력 유도 가능
            nonce = random.randint(0, 10**9)
            llm_seed = random.randint(0, 10**9)

            user_msg = build_augmented_user_msg(
                concept_key=concept_key,
                n_new=n_new,
                existing_prompts=cur,
                nonce=nonce,
            )

            # LLM 생성
            try:
                result = run_llm_with_generator_seed(user_msg, seed=llm_seed)
            except Exception as e:
                print(f"  [WARN] LLM generation failed in round {rounds}: {e}")
                continue

            if concept_key not in result or not isinstance(result[concept_key], list):
                print(f"  [WARN] Key mismatch or invalid list in LLM output (round {rounds})")
                continue

            candidates = result[concept_key]
            # 기본 sanity & dedup
            candidates = [p for p in candidates if isinstance(p, str) and len(p.split()) >= 8]
            candidates = [p for p in candidates if p not in existing_set]
            candidates = dedup_preserve_order(candidates)

            if len(candidates) == 0:
                print(f"  [WARN] No usable candidates (round {rounds})")
                continue

            print(f"  [Round {rounds}] candidates={len(candidates)} verifying...")

            # 후보들 검증해서 PASS만 추가
            added_this_round = 0
            for p in candidates:
                if len(cur) >= min_required:
                    break

                ok = verify_single_prompt_flux_qwenvl(
                    processor=processor,
                    model=model,
                    flux_pipeline=flux_pipeline,
                    concept_key=concept_key,
                    prompt=p,
                    seeds=seeds,
                    save_dir=save_path,
                    save_idx=save_idx,
                )

                if ok:
                    cur.append(p)
                    existing_set.add(p)
                    save_idx += 1
                    added_this_round += 1
                    updated_any = True
                    print(f"    ✅ PASS (total now {len(cur)}/{min_required})")
                else:
                    print("    ❌ FAIL")

            data[concept_key] = cur

            # 라운드마다 중간 저장 (중요)
            save_json(verified_json_path, data)

            if added_this_round == 0:
                print(f"  [WARN] No prompts added in round {rounds} -> continue next round")

        print(f"[Augment DONE] {concept_key}: final={len(cur)}")

    if updated_any:
        save_json(verified_json_path, data)
        print(f"\n[DONE] Dataset augmented/updated → {verified_json_path}")
    else:
        print("\n[DONE] No new verified prompts added (already sufficient or all failed)")


# =========================
# Run All Benchmarks
# =========================
if __name__ == "__main__":
    # =========================
    # Config
    # =========================
    # MODEL_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
    # MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    TEXT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
    VL_MODEL_ID   = "Qwen/Qwen3-VL-8B-Instruct" 
    FLUX = "black-forest-labs/FLUX.1-dev"
    DEVICE = "cuda"
    DTYPE = torch.float16

    NUM_PROMPTS_PER_CONCEPT = 100
    MAX_NEW_TOKENS = 32768
    TEMPERATURE = 0.8

    OUTPUT_DIR = "./results/prompts"
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    # =========================
    # Concepts
    # =========================
    CHARACTERS = [
        "SpongeBob SquarePants", 
        "Mario", 
        "Snoopy", 
        "Stitch", 
        "Mickey Mouse", 
        "Buzz Lightyear", 
        "Homer Simpson", 
        "Pikachu", 
        "Sonic", 
        "Luigi"
    ]

    OBJECTS = [
        "French Bulldog", 
        "Louis Vuitton monogram backpack", 
        "Siberian Husky dog",
        "Sphynx cat", 
        "Baobab tree", 
        "Polar Bear", 
        "Wolf", 
        "Tank", 
        "Fox", 
        "Train"
    ]

    set_seed(42)
    # =========================
    # Load Models
    # =========================
    # text model (prompt generation)
    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID, trust_remote_code=True)
    text_model = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    # vision-language model (verification)
    vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
        VL_MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(VL_MODEL_ID)

    pipe = FluxPipeline.from_pretrained(FLUX, torch_dtype=torch.float16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    seeds = [42, 43, 41]

    # intra-category (character) 
    # intra_2 = list(combinations(CHARACTERS, 2))

    # generate_dataset(intra_2, f"{OUTPUT_DIR}/intra_2_before_character.json")
    # print("✅ prompt generation finished")

    # verify_prompt_json_dataset(
    #     processor=processor,
    #     model=vl_model,
    #     flux_pipeline=pipe,
    #     input_json_path=f"{OUTPUT_DIR}/intra_2_before_character.json",
    #     output_json_path=f"{OUTPUT_DIR}/intra_2_after_character.json",
    #     seeds=seeds
    # )
    # print("✅ prompt verification finished")

    # intra-category (Object)
    intra_2 = list(combinations(OBJECTS, 2))

    generate_dataset(intra_2, f"{OUTPUT_DIR}/intra_2_before_object.json")
    print("✅ prompt generation finished")

    verify_prompt_json_dataset(
        processor=processor,
        model=vl_model,
        flux_pipeline=pipe,
        input_json_path=f"{OUTPUT_DIR}/intra_2_before_object.json",
        output_json_path=f"{OUTPUT_DIR}/intra_2_after_object.json",
        seeds=seeds
    )
    print("✅ prompt verification finished")

    # ✅ 부족분 채우기 (after.json을 min_required까지 채움)
    expected_keys = list(load_json_or_empty(f"{OUTPUT_DIR}/intra_2_before_after.json").keys())
    augment_verified_dataset_to_min(
        processor=processor,
        model=vl_model,
        flux_pipeline=pipe,
        expected_keys=expected_keys,
        verified_json_path=f"{OUTPUT_DIR}/intra_2_after_after.json",
        min_required=100,
        seeds=seeds,
        batch_new_prompts=20,
        max_rounds_per_key=20,
        save_path=f"{OUTPUT_DIR}/augment_images/intra_2_object",
    )
    
    # cross-category (character object)
    cross_2_CO = [(c, h) for c in CHARACTERS for h in OBJECTS]

    generate_dataset(cross_2_CO, f"{OUTPUT_DIR}/cross_2_CO_before.json")
    print("✅ prompt generation finished")

    verify_prompt_json_dataset(
        processor=processor,
        model=vl_model,
        flux_pipeline=pipe,
        input_json_path=f"{OUTPUT_DIR}/cross_2_CO_before.json",
        output_json_path=f"{OUTPUT_DIR}/cross_2_CO_after.json",
        seeds=seeds
    )
    print("✅ prompt verification finished")

    # ✅ 부족분 채우기 (after.json을 min_required까지 채움)
    expected_keys = list(load_json_or_empty(f"{OUTPUT_DIR}/cross_2_CO_after.json").keys())
    augment_verified_dataset_to_min(
        processor=processor,
        model=vl_model,
        flux_pipeline=pipe,
        expected_keys=expected_keys,
        verified_json_path=f"{OUTPUT_DIR}/cross_2_CO_after.json",
        min_required=100,
        seeds=seeds,
        batch_new_prompts=20,
        max_rounds_per_key=20,
        save_path=f"{OUTPUT_DIR}/augment_images/cross_2_CO_after",
    )

    #####################################################################
    # intra_3p = list(combinations(CHARACTERS, 3))
    # generate_dataset(intra_3p, f"{OUTPUT_DIR}/intra_3_before_character.json")
    # print("✅ prompt generation finished")

    # verify_prompt_json_dataset(
    #     processor=processor,
    #     model=vl_model,
    #     flux_pipeline=pipe,
    #     input_json_path=f"{OUTPUT_DIR}/intra_3_before_character.json",
    #     output_json_path=f"{OUTPUT_DIR}/intra_3_after_character.json",
    #     seeds=seeds
    # )
    # print("✅ prompt verification finished")
    

    # intra_3p = list(combinations(OBJECTS, 3))
    # generate_dataset(intra_3p, f"{OUTPUT_DIR}/intra_3plus.json")
    # print("✅ prompt generation finished")

    # verify_prompt_json_dataset(
    #     processor=processor,
    #     model=vl_model,
    #     flux_pipeline=pipe,
    #     input_json_path=f"{OUTPUT_DIR}/intra_3_before_object.json",
    #     output_json_path=f"{OUTPUT_DIR}/intra_3_after_object.json",
    #     seeds=seeds
    # )
    # print("✅ prompt verification finished")

    # # ✅ 부족분 채우기 (after.json을 min_required까지 채움)
    # expected_keys = list(load_json_or_empty(f"{OUTPUT_DIR}/intra_3_after_object.json").keys())
    # augment_verified_dataset_to_min(
    #     processor=processor,
    #     model=vl_model,
    #     flux_pipeline=pipe,
    #     expected_keys=expected_keys,
    #     verified_json_path=f"{OUTPUT_DIR}/intra_3_after_object.json",
    #     min_required=10,
    #     seeds=seeds,
    #     batch_new_prompts=20,
    #     max_rounds_per_key=20,
    #     save_path=f"{OUTPUT_DIR}/augment_images/intra_3_after_object",
    # )


    # cross_3p = [(c1, c2, h) for c1, c2 in combinations(CHARACTERS, 2) for h in OBJECTS]
    # generate_dataset(cross_3p, f"{OUTPUT_DIR}/cross_3_CHOB_before.json")
    # print("✅ prompt generation finished")

    # verify_prompt_json_dataset(
    #     processor=processor,
    #     model=vl_model,
    #     flux_pipeline=pipe,
    #     input_json_path=f"{OUTPUT_DIR}/cross_3_CHOB_before.json",
    #     output_json_path=f"{OUTPUT_DIR}/cross_3_CHOB_after.json",
    #     seeds=seeds
    # )
    # print("✅ prompt verification finished")

    # # ✅ 부족분 채우기 (after.json을 min_required까지 채움)
    # expected_keys = list(load_json_or_empty(f"{OUTPUT_DIR}/cross_3_CHOB_after.json").keys())
    # augment_verified_dataset_to_min(
    #     processor=processor,
    #     model=vl_model,
    #     flux_pipeline=pipe,
    #     expected_keys=expected_keys,
    #     verified_json_path=f"{OUTPUT_DIR}/cross_3_CHOB_after.json",
    #     min_required=10,
    #     seeds=seeds,
    #     batch_new_prompts=20,
    #     max_rounds_per_key=20,
    #     save_path=f"{OUTPUT_DIR}/augment_images/cross_3_CHOB_after",
    # )



    # cross_3p = [(c1, c2, h) for c1, c2 in combinations(OBJECTS, 2) for h in CHARACTERS]
    # generate_dataset(cross_3p, f"{OUTPUT_DIR}/cross_3_OBCH_before.json")
    # print("✅ prompt generation finished")

    # verify_prompt_json_dataset(
    #     processor=processor,
    #     model=vl_model,
    #     flux_pipeline=pipe,
    #     input_json_path=f"{OUTPUT_DIR}/cross_3_OBCH_before.json",
    #     output_json_path=f"{OUTPUT_DIR}/cross_3_OBCH_after.json",
    #     seeds=seeds
    # )
    # print("✅ prompt verification finished")

    # # ✅ 부족분 채우기 (after.json을 min_required까지 채움)
    # expected_keys = list(load_json_or_empty(f"{OUTPUT_DIR}/cross_3_OBCH_after.json").keys())
    # augment_verified_dataset_to_min(
    #     processor=processor,
    #     model=vl_model,
    #     flux_pipeline=pipe,
    #     expected_keys=expected_keys,
    #     verified_json_path=f"{OUTPUT_DIR}/cross_3_OBCH_after.json",
    #     min_required=10,
    #     seeds=seeds,
    #     batch_new_prompts=20,
    #     max_rounds_per_key=20,
    #     save_path=f"{OUTPUT_DIR}/augment_images/cross_3_OBCH_after",
    # )
