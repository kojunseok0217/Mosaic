#!/usr/bin/env python3
"""Evaluate generated images with LAION Aesthetics v2 (no reference).

See README.md for score definitions, result layouts and resume behavior.
Heavy dependencies are imported only for inference; --dry_run needs only Python.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import fnmatch
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
GENERAL = ["intra_2_character", "intra_3_character", "intra_2_object",
           "intra_3_object", "cross_2_CO", "cross_3_CHOB", "cross_3_OBCH"]
CATEGORIES = GENERAL + ["celeb", "nsfw", "style"]
METHOD_DIRS = {
    "eraseanything": ("EraseAnything", "EraseAnything_celeb", "EraseAnything_style_nsfw"),
    "mace": ("MACE", "MACE_celeb", "MACE_style_nsfw"),
    "mosaic": ("mosaic", "mosaic_celeb", "mosaic_style_nsfw"),
}
NEGATIVE_METHODS = ["negative_prompt_cfg3.5", "negative_guidance_eta1.0_uncond"]
SCORES = ("aesthetic",)
IMAGE_FIELDS = ["method", "source_dir", "category", "seed", "concept", "idx",
                "image_path", "size_bytes", "mtime_ns"]
DETAIL_FIELDS = IMAGE_FIELDS + ["width", "height", *SCORES, "status", "error"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=HERE.parent / "outputs")
    parser.add_argument("--out_dir", type=Path, default=HERE / "logs/aesthetic")
    parser.add_argument("--methods", nargs="+", type=str.lower,
                        choices=[*METHOD_DIRS, *NEGATIVE_METHODS], default=["mosaic"],
                        help="Baseline aliases or negative result directory names; negatives require --categories general")
    parser.add_argument("--categories", nargs="+", choices=["all", "general", *CATEGORIES],
                        default=["general"], help="Default: seven general categories")
    parser.add_argument("--seeds", nargs="+", default=["42"], help="42 (default), 42 43 44, or all")
    parser.add_argument("--mosaic_dir", default=METHOD_DIRS["mosaic"][0],
                        help="Mosaic directory for general categories, relative to results_root")
    parser.add_argument("--filename_pattern", default="result_comp_*.png")
    parser.add_argument("--selection", choices=["available", "common"], default="available",
                        help="All available images, or matching category/seed/concept/index across methods")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--scan_workers", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=0,
                        help="Deterministic first N per method/category/seed for smoke tests; 0 = all")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse successful scores for unchanged files; retry previous failures")
    parser.add_argument("--dry_run", "--dry-run", action="store_true",
                        help="Scan and print counts without importing models or writing files")
    args = parser.parse_args(argv)
    if args.seeds != ["all"] and not all(s.isdecimal() for s in args.seeds):
        parser.error("--seeds must contain nonnegative integers, or 'all' alone")
    if args.batch_size < 1 or args.scan_workers < 1 or args.max_images < 0:
        parser.error("batch_size and scan_workers must be positive; max_images must be nonnegative")
    if "/" in args.filename_pattern or "\\" in args.filename_pattern:
        parser.error("--filename_pattern must match a filename, not a directory")
    args.methods = list(dict.fromkeys(args.methods))
    args.categories = list(dict.fromkeys(c for name in args.categories
                          for c in (CATEGORIES if name == "all" else GENERAL if name == "general" else [name])))
    if any(m in NEGATIVE_METHODS for m in args.methods) and any(c not in GENERAL for c in args.categories):
        parser.error("Negative baselines support the seven general categories; pass --categories general "
                     "or explicit general category names")
    if args.selection == "common" and any(m in NEGATIVE_METHODS for m in args.methods) \
            and any(m in METHOD_DIRS for m in args.methods):
        parser.error("Common selection cannot mix negative baselines (explicit prompt indices) with "
                     "legacy baselines (positional indices); evaluate these families separately")
    if args.seeds != ["all"]:
        args.seeds = sorted({int(s) for s in args.seeds})
    args.results_root = args.results_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    return args


def source_dir(args, method, category):
    if method in NEGATIVE_METHODS:
        return args.results_root / method / category
    general, celeb, special = METHOD_DIRS[method]
    if category in GENERAL:
        return args.results_root / (args.mosaic_dir if method == "mosaic" else general) / category
    return args.results_root / (celeb if category == "celeb" else special) / f"{category}_after"


def subdirs(path):
    with os.scandir(path) as entries:
        return sorted((e for e in entries if e.is_dir()), key=lambda e: e.name)


def sample_key(row):
    return row["category"], row["seed"], row["concept"], row["idx"]


def scan_seed(seed_root, pattern, workers):
    """Scan exactly concept/index/image, including symlinks but never auxiliary subtrees."""
    def scan_concept(entry):
        rows, empty = [], 0
        concept = " + ".join(sorted(p.strip().casefold() for p in entry.name.split("+") if p.strip()))
        for index in subdirs(entry.path):
            if not index.name.isdecimal():
                continue
            with os.scandir(index.path) as files:
                matches = sorted((e for e in files if fnmatch.fnmatchcase(e.name, pattern) and e.is_file()),
                                 key=lambda e: e.name)
            if not matches:
                empty += 1
            for match in matches:
                stat = match.stat()
                rows.append(dict(concept=concept, idx=int(index.name), image_path=str(Path(match.path).resolve()),
                                 size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
        return rows, empty

    rows, empty, seen = [], 0, set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for found, count in pool.map(scan_concept, subdirs(seed_root)):
            empty += count
            for row in found:
                if row["image_path"] not in seen:
                    rows.append(row)
                    seen.add(row["image_path"])
    return rows, empty


def build_plan(args):
    rows, coverage = [], []
    for category in args.categories:
        roots = {method: source_dir(args, method, category) for method in args.methods}
        seeds = args.seeds
        if seeds == ["all"]:
            seeds = sorted({int(e.name[5:]) for root in roots.values() if root.is_dir()
                            for e in subdirs(root) if e.name.startswith("seed_") and e.name[5:].isdecimal()})
        for seed in seeds or [None]:
            groups = {}
            for method, root in roots.items():
                seed_root = root / f"seed_{seed}"
                exists = seed is not None and seed_root.is_dir()
                found, empty = scan_seed(seed_root, args.filename_pattern, args.scan_workers) if exists else ([], 0)
                for row in found:
                    row.update(method=method, source_dir=str(root), category=category, seed=seed)
                groups[method] = sorted(found, key=lambda r: (sample_key(r), r["image_path"]))
                ambiguous = sum(n > 1 for n in Counter(sample_key(r) for r in found).values())
                coverage.append(dict(method=method, category=category, seed=seed, source_dir=str(root),
                                     seed_exists=exists, n_available=len(found), empty_index_dirs=empty,
                                     n_ambiguous_keys=ambiguous, n_selected=0))
            if args.selection == "common":
                for method, found in groups.items():
                    duplicates = [key for key, n in Counter(sample_key(r) for r in found).items() if n > 1]
                    if duplicates:
                        raise ValueError(f"Ambiguous common samples in {method}: {duplicates[:3]}. "
                                         "Multiple distinct files share a normalized concept/index. "
                                         "Use --selection available to score every file, or disambiguate inputs.")
                common = set.intersection(*(set(sample_key(r) for r in found) for found in groups.values()))
                groups = {m: [r for r in found if sample_key(r) in common] for m, found in groups.items()}
            for item in coverage[-len(args.methods):]:
                found = groups[item["method"]]
                if args.max_images:
                    found = found[:args.max_images]
                item["n_selected"] = len(found)
                rows.extend(found)
                print(f"[SCAN] {item['method']} / {category} / seed={seed}: "
                      f"available={item['n_available']}, selected={len(found)}, "
                      f"empty_dirs={item['empty_index_dirs']}, ambiguous_keys={item['n_ambiguous_keys']}", flush=True)
    return rows, coverage


def write_csv(path, fields, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def cache_key(row):
    return row["method"], row["image_path"], row["size_bytes"], row["mtime_ns"]


def load_cache(path):
    """Recover a partially written final JSONL line after interruption."""
    cache = {}
    if not path.exists():
        return cache
    with path.open("r+b") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if handle.read(1):
                    raise ValueError(f"Invalid cache record inside {path} at byte {offset}")
                handle.seek(offset)
                handle.truncate()
                print("[RESUME] Removed interrupted final cache record", flush=True)
                break
            cache[cache_key(row)] = row
            if not line.endswith(b"\n"):
                handle.write(b"\n")
    return cache


def score_batch(items, tensors, models, torch, device):
    """Keep native image sizes; caller groups tensors by shape before batching."""
    batch = torch.stack(tensors).to(device)
    with torch.inference_mode():
        values = {name: model(batch).detach().float().reshape(-1).cpu().tolist()
                  for name, model in models.items()}
    if any(len(scores) != len(items) for scores in values.values()):
        raise RuntimeError("Metric returned a different number of scores than input images")
    if any(not math.isfinite(v) for scores in values.values() for v in scores):
        raise RuntimeError("Metric returned a non-finite score")
    return [dict(row, **{name: scores[i] for name, scores in values.items()}, status="ok", error="")
            for i, row in enumerate(items)]


def evaluate(rows, cache, progress_path, models, torch, device, batch_size):
    import numpy as np
    from PIL import Image, ImageOps

    from tqdm import tqdm

    results, pending = [], []
    for row in rows:
        previous = cache.get(cache_key(row))
        if previous and previous["status"] == "ok":
            results.append(dict(previous, **row))
        else:
            pending.append(row)
    print(f"[RESUME] cached={len(results)}, remaining={len(pending)}", flush=True)
    with progress_path.open("a", encoding="utf-8") as handle, tqdm(total=len(pending), desc="Aesthetic") as bar:
        def save(scored):
            for row in scored:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            results.extend(scored)
            bar.update(len(scored))

        # Only the next batch_size files are decoded at a time, bounding host memory.
        for start in range(0, len(pending), batch_size):
            by_shape = defaultdict(list)
            for row in pending[start:start + batch_size]:
                try:
                    stat = Path(row["image_path"]).stat()
                    if (stat.st_size, stat.st_mtime_ns) != (row["size_bytes"], row["mtime_ns"]):
                        raise OSError("Image changed after discovery; rerun with --resume")
                    with Image.open(row["image_path"]) as image:
                        rgb = ImageOps.exif_transpose(image).convert("RGB")
                        tensor = torch.from_numpy(np.array(rgb, dtype=np.uint8, copy=True))
                        tensor = tensor.permute(2, 0, 1).float().div_(255.0)
                        item = dict(row, width=rgb.width, height=rgb.height)
                    by_shape[tuple(tensor.shape)].append((item, tensor))
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    save([dict(row, width="", height="", aesthetic="",
                               status="error", error=f"{type(exc).__name__}: {exc}")])
            for group in by_shape.values():
                items, tensors = zip(*group)
                # Model/device errors abort visibly, preserving all completed batches for resume.
                save(score_batch(items, tensors, models, torch, device))
    return sorted(results, key=lambda r: (r["method"], sample_key(r), r["image_path"]))


def summaries(rows, coverage, selection, methods, group_fields):
    grouped, planned = defaultdict(list), defaultdict(int)
    for item in coverage:
        planned[tuple(item[k] for k in group_fields)] += item["n_selected"]
    for row in rows:
        grouped[tuple(row[k] for k in group_fields)].append(row)
    successful = defaultdict(set)
    if selection == "common":
        for row in rows:
            if row["status"] == "ok":
                successful[sample_key(row)].add(row["method"])
    output = []
    for key, n_selected in planned.items():
        found = grouped[key]
        ok = [r for r in found if r["status"] == "ok"]
        valid = [r for r in ok if selection != "common" or len(successful[sample_key(r)]) == len(methods)]
        item = dict(zip(group_fields, key))
        item.update(selection=selection, n_selected=n_selected, n_evaluated=len(valid),
                    n_failed=sum(r["status"] != "ok" for r in found), n_peer_failed=len(ok) - len(valid))
        for score in SCORES:
            values = [r[score] for r in valid]
            item[f"{score}_mean"] = statistics.fmean(values) if values else ""
            item[f"{score}_std"] = statistics.stdev(values) if len(values) > 1 else ""
        output.append(item)
    return output


def save_results(out_dir, rows, coverage, args):
    write_csv(out_dir / "per_image.csv", DETAIL_FIELDS, rows)
    for filename, fields in [("summary.csv", ["method"]),
                             ("category_summary.csv", ["method", "category"]),
                             ("seed_summary.csv", ["method", "category", "seed"])]:
        summary = summaries(rows, coverage, args.selection, args.methods, fields)
        columns = fields + ["selection", "n_selected", "n_evaluated", "n_failed", "n_peer_failed",
                            "aesthetic_mean", "aesthetic_std"]
        write_csv(out_dir / filename, columns, summary)
        if filename == "summary.csv":
            for item in summary:
                print(f"[SCORE] {item['method']}: n={item['n_evaluated']}/{item['n_selected']}, "
                      f"Aesthetic={item['aesthetic_mean']}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    if not args.results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {args.results_root}")
    rows, coverage = build_plan(args)
    print(f"[PLAN] {len(rows)} images; selection={args.selection}; "
          "aesthetic=laion_aes", flush=True)
    if args.dry_run:
        print("[DRY RUN] No models loaded and no files written.")
        return 0
    if not rows:
        raise ValueError("No images selected. Check roots, categories, seeds and filename_pattern.")

    try:
        import torch
        import pyiqa
    except ImportError as exc:
        raise RuntimeError("Install evaluation/requirements-aesthetic.txt in the evaluation environment.") from exc
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use a GPU node or pass --device cpu.")
    config = dict(schema_version=1, aesthetic_model="laion_aes",
                  preprocessing="exif_transpose_rgb_float32_0_1_native_size_pyiqa_defaults",
                  versions={name: importlib.metadata.version(name)
                            for name in ("pyiqa", "torch", "torchvision", "Pillow", "numpy")},
                  results_root=str(args.results_root), methods=args.methods, categories=args.categories,
                  seeds=args.seeds, mosaic_dir=args.mosaic_dir, filename_pattern=args.filename_pattern,
                  selection=args.selection, max_images=args.max_images)
    config_path = args.out_dir / "config.json"
    progress_path = args.out_dir / "progress.jsonl"
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        if not args.resume:
            raise FileExistsError(f"Output directory is not empty: {args.out_dir}. Use --resume or a new --out_dir.")
        if not config_path.exists() or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Resume configuration or package versions differ. Use a new --out_dir.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)
    write_json(args.out_dir / "manifest.json", dict(started_utc=datetime.now(timezone.utc).isoformat(),
               device=device, batch_size=args.batch_size, n_images=len(rows), images=rows))
    write_csv(args.out_dir / "coverage.csv", list(coverage[0]), coverage)
    cache = load_cache(progress_path) if args.resume else {}
    pending = [r for r in rows if cache.get(cache_key(r), {}).get("status") != "ok"]
    models = {}
    if pending:
        print(f"[MODELS] Loading laion_aes on {device}; first run downloads weights.", flush=True)
        models = {name: pyiqa.create_metric(metric, device=torch.device(device), as_loss=False).eval()
                  for name, metric in (("aesthetic", "laion_aes"),)}
    results = evaluate(rows, cache, progress_path, models, torch, device, args.batch_size)
    save_results(args.out_dir, results, coverage, args)
    errors = sum(r["status"] != "ok" for r in results)
    print(f"[DONE] {args.out_dir} (failed images={errors})", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
