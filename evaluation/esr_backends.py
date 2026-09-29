"""Gemma ESR adapter ported from SplitFlow/evaluation/evaluation_vlm_backends.py."""
import json
import os
from pathlib import Path


def validate_environment(backend):
    """Local import check only: never downloads or loads model weights."""
    import transformers
    from packaging.version import Version

    version = Version(transformers.__version__)
    if backend == "gemma":
        if version < Version("5.17.0") or not hasattr(transformers, "Gemma4UnifiedForConditionalGeneration"):
            raise RuntimeError(
                f"This Gemma 4 12B adapter needs Transformers >= 5.17.0 with gemma4_unified support; found {version}. "
                "Use a separate environment with requirements-esr-gemma.txt "
                "(keep the main Mosaic/Qwen environment separate)."
            )
        from transformers import AutoModelForMultimodalLM, AutoProcessor  # noqa: F401
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    print(f"[ENV] backend={backend}, transformers={version}")


def load_backend(backend, model_id, cache_dir=None):
    import torch
    from transformers import AutoProcessor

    validate_environment(backend)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run full Gemma inference on a GPU node.")
    from transformers import AutoModelForMultimodalLM

    cache_dir = cache_dir or os.environ.get("MODEL_CACHE_DIR")
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto", cache_dir=cache_dir,
    ).eval()
    return model, processor


def prepare_inputs(backend, processor, batch_samples):
    from evaluation_esr import build_eval_prompt

    messages_batch = []
    for sample in batch_samples:
        if len(sample["images"]) != 4:
            raise ValueError("Expected reference images A/B/C followed by after image D")
        content = [{"type": "image", "image": image} for image in sample["images"]]
        content.append({"type": "text", "text": build_eval_prompt(
            sample["target"], sample["prompt_text"],
        )})
        messages_batch.append([{"role": "user", "content": content}])

    return prepare_messages(processor, messages_batch)


def prepare_messages(processor, messages_batch):
    """Let Gemma expand image placeholders; disable thinking for one-word judgments."""
    return processor.apply_chat_template(
        messages_batch, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
        processor_kwargs={"padding": True},
    )


def infer_batch(backend, model, processor, batch_samples, max_new_tokens=16):
    """Return response texts; the shared evaluator applies its existing normalizer."""
    import torch

    inputs = prepare_inputs(backend, processor, batch_samples).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # Slice at padded input width, not attention-mask length, for every batch row.
    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True,
    )


def validate_resume(jsonl_path, eval_config):
    """Refuse to silently reuse judgments from another model or evaluation setup."""
    if not jsonl_path.exists():
        return
    seen = set()
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Incomplete/invalid JSONL at {jsonl_path}:{line_no}; repair it before resuming") from exc
            if row.get("eval_config") != eval_config:
                raise ValueError(
                    f"Evaluation configuration mismatch at {jsonl_path}:{line_no}. "
                    "Use separate --intermediate_jsonl and --out_csv paths for different models/settings."
                )
            uid = row.get("uid")
            concepts = row.get("concepts", [])
            if not uid or uid in seen or not concepts or any(
                row.get(f"verdict_{i}") not in {"present", "absent", "invalid"}
                for i in range(len(concepts))
            ):
                raise ValueError(f"Duplicate or incomplete evaluation row at {jsonl_path}:{line_no}")
            seen.add(uid)


def fractional_summary(jsonl_path, method, category, eval_seeds):
    scores = []
    invalid_targets = 0
    if Path(jsonl_path).exists():
        with Path(jsonl_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (row.get("method") != method or row.get("category") != category
                        or row.get("seed") not in eval_seeds):
                    continue
                verdicts = [row[f"verdict_{i}"] for i in range(len(row["concepts"]))]
                scores.append(verdicts.count("absent") / len(verdicts))
                invalid_targets += verdicts.count("invalid")
    return {
        "fractional_erasure_rate": sum(scores) / len(scores) if scores else 0.0,
        "invalid_targets": invalid_targets,
    }
