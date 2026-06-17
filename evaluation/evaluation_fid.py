#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import time
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from T2IBenchmark.model_wrapper import T2IModelWrapper, ModelWrapperDataloader
from T2IBenchmark.feature_extractors import BaseFeatureExtractor, InceptionV3FE
from T2IBenchmark.loaders import BaseImageLoader, ImageDataset, get_images_from_folder, validate_image_paths
from T2IBenchmark.metrics import FIDStats, frechet_distance
from T2IBenchmark.utils import dprint, set_all_seeds


# -------------------------
# Core: dataset creation
# -------------------------
def create_dataset_from_input(obj: Union[str, List[str], BaseImageLoader, FIDStats]) -> Union[BaseImageLoader, FIDStats]:
    if isinstance(obj, str):
        if obj.endswith(".npz"):
            return FIDStats.from_npz(obj)
        else:
            image_paths = get_images_from_folder(obj)
            return ImageDataset(image_paths)
    elif isinstance(obj, list):
        validate_image_paths(obj)
        return ImageDataset(obj)
    elif isinstance(obj, BaseImageLoader):
        return obj
    elif isinstance(obj, FIDStats):
        return obj
    else:
        raise ValueError(f"Input {obj} has unknown type.")


def get_features_for_dataset(dataset, feature_extractor: BaseFeatureExtractor, verbose: bool = True) -> np.ndarray:
    features = []
    for x in tqdm(dataset, disable=not verbose):
        feats = feature_extractor.forward(x).cpu().numpy()
        features.append(feats)
    return np.concatenate(features, axis=0)


def calculate_fid(
    input1: Union[str, List[str], BaseImageLoader, FIDStats],
    input2: Union[str, List[str], BaseImageLoader, FIDStats],
    device: Union[str, torch.device] = "cuda",
    seed: Optional[int] = 42,
    batch_size: int = 128,
    dataloader_workers: int = 16,
    verbose: bool = True,
) -> Tuple[float, Tuple[dict, dict]]:
    if seed is not None:
        set_all_seeds(seed)

    input1 = create_dataset_from_input(input1)
    input2 = create_dataset_from_input(input2)

    inception_fe = InceptionV3FE(device)

    stats = []
    all_features = []

    for input_data in [input1, input2]:
        dprint(verbose, f"Processing: {input_data}")
        if isinstance(input_data, FIDStats):
            all_features.append([])
            stats.append(input_data)
        elif isinstance(input_data, ImageDataset):
            dataset = input_data
            dataset.preprocess_fn = inception_fe.get_preprocess_fn()
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=dataloader_workers,
            )
            feats = get_features_for_dataset(dataloader, inception_fe, verbose=verbose)
            all_features.append(feats)
            stats.append(FIDStats.from_features(feats))
        elif isinstance(input_data, T2IModelWrapper):
            dataloader = ModelWrapperDataloader(
                input_data, batch_size, preprocess_fn=inception_fe.get_preprocess_fn()
            )
            feats = get_features_for_dataset(dataloader, inception_fe, verbose=verbose)
            all_features.append(feats)
            stats.append(FIDStats.from_features(feats))
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

    fid = frechet_distance(stats[0], stats[1])
    dprint(verbose, f"FID is {fid}")
    return float(fid), (
        {"features": all_features[0], "stats": stats[0]},
        {"features": all_features[1], "stats": stats[1]},
    )


# -------------------------
# Path / key helpers
# -------------------------
def normalize_idx(idx_str: str) -> str:
    try:
        return str(int(idx_str))
    except ValueError:
        return idx_str


def normalize_concept(concept_str: str, sep: str = "+") -> str:
    parts = [x.strip() for x in concept_str.split(sep)]
    parts = [x for x in parts if x]
    parts = sorted(parts)
    return " + ".join(parts)


def append_method_fid(csv_path: str, method_name: str, fid: float):
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["method", "fid"])
        w.writerow([method_name, f"{fid:.6f}"])


def append_category_fid(csv_path: str, method_name: str, category: str, fid: float):
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["method", "category", "fid"])
        w.writerow([method_name, category, f"{fid:.6f}"])


def append_seed_fid(csv_path: str, method_name: str, seed_name: str, fid: float):
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["method", "seed", "fid"])
        w.writerow([method_name, seed_name, f"{fid:.6f}"])


def append_seed_summary(csv_path: str, method_name: str, mean_fid: float, std_fid: float, n_seeds: int):
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["method", "mean_fid", "std_fid", "n_seeds"])
        w.writerow([method_name, f"{mean_fid:.6f}", f"{std_fid:.6f}", n_seeds])


# -------------------------
# Collectors
# -------------------------
# Common key:
#   (category, seed, normalized_concept, normalized_idx)

def collect_flux_images(
    flux_root: Path,
    filename: str = "result_base.png",
    follow_symlinks: bool = False,
) -> Dict[Tuple[str, str, str, str], str]:
    if not flux_root.exists():
        raise FileNotFoundError(f"Flux directory not found: {flux_root}")

    out: Dict[Tuple[str, str, str, str], str] = {}

    for dirpath, _, filenames in os.walk(str(flux_root), followlinks=follow_symlinks):
        if filename not in filenames:
            continue

        p_dir = Path(dirpath)
        rel = p_dir.relative_to(flux_root)
        parts = rel.parts

        if len(parts) < 4:
            continue

        category, seed, concept, idx = parts[0], parts[1], parts[2], parts[3]
        concept = normalize_concept(concept)
        idx = normalize_idx(idx)
        key = (category, seed, concept, idx)

        out[key] = str((p_dir / filename).resolve())

    return out


def collect_method_images(
    method_root: Path,
    filename_pattern: str = "result_comp_*.png",
    follow_symlinks: bool = False,
) -> Dict[Tuple[str, str, str, str], str]:
    if not method_root.exists():
        raise FileNotFoundError(f"Method directory not found: {method_root}")

    out: Dict[Tuple[str, str, str, str], str] = {}

    for dirpath, _, _ in os.walk(str(method_root), followlinks=follow_symlinks):
        p_dir = Path(dirpath)
        rel = p_dir.relative_to(method_root)
        parts = rel.parts

        if len(parts) < 4:
            continue

        category, seed, concept, idx = parts[0], parts[1], parts[2], parts[3]
        concept = normalize_concept(concept)
        idx = normalize_idx(idx)

        matched_files = sorted(p_dir.glob(filename_pattern))
        if len(matched_files) == 0:
            continue

        img_path = matched_files[0].resolve()
        key = (category, seed, concept, idx)
        out[key] = str(img_path)

    return out


def collect_matched_pairs_by_seed(
    results_root: str,
    mode: str = "all",
    category: Optional[str] = None,
    method_dirname: str = "MACE",
    method_filename_pattern: str = "result_comp_*.png",
    flux_dirname: str = "flux",
    flux_filename: str = "result_base.png",
    follow_symlinks: bool = False,
    verbose: bool = True,
) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Returns:
      {
        "seed_42": ([method_paths...], [flux_paths...]),
        "seed_43": ([method_paths...], [flux_paths...]),
        ...
      }
    """
    root = Path(results_root)
    method_root = root / method_dirname
    flux_root = root / flux_dirname

    m = collect_method_images(
        method_root=method_root,
        filename_pattern=method_filename_pattern,
        follow_symlinks=follow_symlinks,
    )
    f = collect_flux_images(
        flux_root=flux_root,
        filename=flux_filename,
        follow_symlinks=follow_symlinks,
    )

    if mode == "category":
        if category is None:
            raise ValueError("--mode category requires --category")
        m = {k: v for k, v in m.items() if k[0] == category}
        f = {k: v for k, v in f.items() if k[0] == category}

    matched_keys = sorted(set(m.keys()) & set(f.keys()))
    only_m_keys = sorted(set(m.keys()) - set(f.keys()))
    only_f_keys = sorted(set(f.keys()) - set(m.keys()))

    grouped: Dict[str, Tuple[List[str], List[str]]] = {}
    for key in matched_keys:
        _, seed_name, _, _ = key
        if seed_name not in grouped:
            grouped[seed_name] = ([], [])
        grouped[seed_name][0].append(m[key])
        grouped[seed_name][1].append(f[key])

    if verbose:
        print(f"[METHOD] collected: {len(m)}")
        print(f"[FLUX]   collected: {len(f)}")
        print(f"[MATCH]  total matched pairs: {len(matched_keys)}")
        print(f"[SEEDS]  matched seed groups: {len(grouped)}")

        for seed_name in sorted(grouped.keys()):
            n = len(grouped[seed_name][0])
            print(f"  - {seed_name}: {n} pairs")

        if only_m_keys or only_f_keys:
            print(f"[WARN] unmatched: only_method={len(only_m_keys)}, only_flux={len(only_f_keys)}")

            preview_n = 10
            if only_m_keys:
                print("[WARN] examples only in method:")
                for k in only_m_keys[:preview_n]:
                    print("   ", k)
            if only_f_keys:
                print("[WARN] examples only in flux:")
                for k in only_f_keys[:preview_n]:
                    print("   ", k)

    return grouped


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        type=str,
        required=True,
        help="예: /nas/home/junseok/workspace/multi_concept_erasure/SplitFlow/results",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "category"],
        default="all",
        help="all: 전체 매칭, category: 특정 category만 매칭",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="mode=category 일 때 category 이름 (예: cross_2_CO)",
    )

    parser.add_argument("--method_dirname", type=str, default="MACE")
    parser.add_argument("--method_filename_pattern", type=str, default="result_comp_*.png")
    parser.add_argument("--flux_dirname", type=str, default="flux")
    parser.add_argument("--flux_filename", type=str, default="result_base.png")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dataloader_workers", type=int, default=16)
    parser.add_argument("--follow_symlinks", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--out_method_csv", type=str, default=None)
    parser.add_argument("--out_category_csv", type=str, default=None)
    parser.add_argument("--out_seed_csv", type=str, default=None)
    parser.add_argument("--out_seed_summary_csv", type=str, default=None)

    args = parser.parse_args()

    if args.mode == "category" and not args.category:
        raise ValueError("--mode category requires --category")

    grouped_pairs = collect_matched_pairs_by_seed(
        results_root=args.results_root,
        mode=args.mode,
        category=args.category,
        method_dirname=args.method_dirname,
        method_filename_pattern=args.method_filename_pattern,
        flux_dirname=args.flux_dirname,
        flux_filename=args.flux_filename,
        follow_symlinks=args.follow_symlinks,
        verbose=args.verbose,
    )

    if len(grouped_pairs) == 0:
        raise RuntimeError("No matched image pairs found. Check folder names / filenames.")

    compare_name = (
        f"{args.method_dirname}:{args.method_filename_pattern}"
        f"_vs_"
        f"{args.flux_dirname}:{args.flux_filename}"
    )

    seed_fids = []
    sorted_seed_names = sorted(grouped_pairs.keys())

    for seed_name in sorted_seed_names:
        paths_method, paths_flux = grouped_pairs[seed_name]

        if len(paths_method) == 0:
            print(f"[WARN] {seed_name}: no matched pairs, skipping.")
            continue

        if args.verbose:
            print(f"\n[SEED] {seed_name}")
            print(f"  method images: {len(paths_method)}")
            print(f"  flux images  : {len(paths_flux)}")

        fid, _ = calculate_fid(
            paths_method,
            paths_flux,
            device=args.device,
            seed=args.seed,
            batch_size=args.batch_size,
            dataloader_workers=args.dataloader_workers,
            verbose=args.verbose,
        )

        seed_fids.append(fid)
        print(f"[RESULT] FID({compare_name})[{seed_name}] = {fid:.6f}")

        if args.out_seed_csv:
            append_seed_fid(args.out_seed_csv, compare_name, seed_name, fid)

    if len(seed_fids) == 0:
        raise RuntimeError("All seed groups were empty. No FID computed.")

    mean_fid = float(np.mean(seed_fids))
    std_fid = float(np.std(seed_fids, ddof=0))

    print("\n[SUMMARY]")
    print(f"  seeds used : {len(seed_fids)}")
    print(f"  mean FID   : {mean_fid:.6f}")
    print(f"  std FID    : {std_fid:.6f}")

    if args.mode == "all":
        if args.out_method_csv:
            append_method_fid(args.out_method_csv, compare_name, mean_fid)
    else:
        if args.out_category_csv:
            append_category_fid(args.out_category_csv, compare_name, args.category, mean_fid)

    if args.out_seed_summary_csv:
        append_seed_summary(args.out_seed_summary_csv, compare_name, mean_fid, std_fid, len(seed_fids))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[TIME] Total elapsed: {time.time() - t0:.2f}s")