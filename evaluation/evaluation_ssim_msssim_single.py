#!/usr/bin/env python3

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for this script. Activate the evaluation environment that already has torch installed."
    ) from exc

from PIL import Image
from tqdm import tqdm

try:
    from pytorch_msssim import ms_ssim, ssim
except ImportError as exc:
    raise ImportError(
        "Install `pytorch-msssim` in the active environment before running this script: "
        "`pip install pytorch-msssim`"
    ) from exc


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SSIM and MS-SSIM between images in a reference directory and a hub directory."
    )
    parser.add_argument("--reference_dir", type=str, required=True)
    parser.add_argument("--hub_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--summary_csv_name",
        type=str,
        default="ssim_msssim_summary.csv",
        help="Filename for the summary CSV written inside output_dir.",
    )
    parser.add_argument(
        "--detail_csv_name",
        type=str,
        default="ssim_msssim_details.csv",
        help="Filename for the per-image detail CSV written inside output_dir.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for images under both directories.",
    )
    parser.add_argument(
        "--resize_to_reference",
        action="store_true",
        help="Resize each hub image to the matched reference image size before evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device. 'auto' picks CUDA when available.",
    )
    parser.add_argument("--window_size", type=int, default=11)
    parser.add_argument("--window_sigma", type=float, default=1.5)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def append_csv_rows(csv_path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def iter_image_paths(root: Path, recursive: bool) -> Iterable[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in iterator:
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def build_image_index(root: Path, recursive: bool) -> Dict[str, Path]:
    image_index: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}

    for path in iter_image_paths(root, recursive):
        relative_key = path.relative_to(root).with_suffix("").as_posix()
        if relative_key in image_index:
            duplicates.setdefault(relative_key, [image_index[relative_key]]).append(path)
            continue
        image_index[relative_key] = path

    if duplicates:
        duplicate_lines = []
        for key, paths in sorted(duplicates.items()):
            joined = ", ".join(str(p) for p in paths)
            duplicate_lines.append(f"{key}: {joined}")
        raise ValueError("Duplicate image keys detected:\n" + "\n".join(duplicate_lines))

    return image_index


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    channels = len(image.getbands())
    buffer = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    tensor = buffer.view(image.size[1], image.size[0], channels).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(dtype=torch.float32) / 255.0


def load_image_pair(
    reference_path: Path,
    hub_path: Path,
    resize_to_reference: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    reference_image = Image.open(reference_path).convert("RGB")
    hub_image = Image.open(hub_path).convert("RGB")

    if hub_image.size != reference_image.size:
        if not resize_to_reference:
            raise ValueError(
                f"Image size mismatch: reference={reference_image.size}, hub={hub_image.size}"
            )
        hub_image = hub_image.resize(reference_image.size, resample=Image.BICUBIC)

    reference_tensor = pil_to_tensor(reference_image).to(device=device, dtype=torch.float32)
    hub_tensor = pil_to_tensor(hub_image).to(device=device, dtype=torch.float32)
    return reference_tensor, hub_tensor


def compute_ssim_ms_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int,
    sigma: float,
) -> Tuple[float, float]:
    ssim_value = ssim(
        x,
        y,
        data_range=1.0,
        size_average=True,
        win_size=window_size,
        win_sigma=sigma,
        nonnegative_ssim=True,
    )
    ms_ssim_value = ms_ssim(
        x,
        y,
        data_range=1.0,
        size_average=True,
        win_size=window_size,
        win_sigma=sigma,
    )
    return float(ssim_value.item()), float(ms_ssim_value.item())


def main() -> None:
    args = parse_args()
    t0 = time.time()

    reference_dir = Path(args.reference_dir)
    hub_dir = Path(args.hub_dir)
    output_dir = Path(args.output_dir)

    if args.window_size <= 0 or args.window_size % 2 == 0:
        raise ValueError("--window_size must be a positive odd integer.")

    if not reference_dir.is_dir():
        raise FileNotFoundError(f"Reference image directory not found: {reference_dir}")
    if not hub_dir.is_dir():
        raise FileNotFoundError(f"Hub image directory not found: {hub_dir}")

    device = choose_device(args.device)

    reference_index = build_image_index(reference_dir, args.recursive)
    hub_index = build_image_index(hub_dir, args.recursive)

    matched_keys = sorted(set(reference_index) & set(hub_index))
    unmatched_reference = sorted(set(reference_index) - set(hub_index))
    unmatched_hub = sorted(set(hub_index) - set(reference_index))

    if not matched_keys:
        raise FileNotFoundError(
            "No matched images found between the two directories. "
            "Images are matched by relative path without extension."
        )

    if args.verbose:
        print(f"[INFO] reference images: {len(reference_index)}")
        print(f"[INFO] hub images: {len(hub_index)}")
        print(f"[INFO] matched images: {len(matched_keys)}")
        print(f"[INFO] unmatched reference images: {len(unmatched_reference)}")
        print(f"[INFO] unmatched hub images: {len(unmatched_hub)}")
        print(f"[INFO] device: {device}")

    detail_rows: List[Dict[str, object]] = []
    ssim_scores: List[float] = []
    ms_ssim_scores: List[float] = []
    skipped_count = 0

    progress_bar = tqdm(matched_keys, desc="SSIM/MS-SSIM", dynamic_ncols=True, unit="img")

    for key in progress_bar:
        reference_path = reference_index[key]
        hub_path = hub_index[key]

        try:
            reference_tensor, hub_tensor = load_image_pair(
                reference_path=reference_path,
                hub_path=hub_path,
                resize_to_reference=args.resize_to_reference,
                device=device,
            )
            ssim_score, ms_ssim_score = compute_ssim_ms_ssim(
                reference_tensor,
                hub_tensor,
                window_size=args.window_size,
                sigma=args.window_sigma,
            )
        except Exception as exc:
            skipped_count += 1
            if args.verbose:
                print(f"[SKIP] {key}: {type(exc).__name__}: {exc}")
            continue

        ssim_scores.append(ssim_score)
        ms_ssim_scores.append(ms_ssim_score)
        detail_rows.append(
            {
                "image_key": key,
                "reference_image_path": str(reference_path),
                "hub_image_path": str(hub_path),
                "ssim": f"{ssim_score:.6f}",
                "ms_ssim": f"{ms_ssim_score:.6f}",
            }
        )

        progress_bar.set_postfix_str(
            f"eval={len(ssim_scores)} ssim={mean(ssim_scores):.4f} ms_ssim={mean(ms_ssim_scores):.4f}"
        )

    progress_bar.close()

    evaluated_count = len(ssim_scores)
    if evaluated_count == 0:
        raise RuntimeError("No image pairs were successfully evaluated.")

    summary_row = {
        "reference_dir": str(reference_dir),
        "hub_dir": str(hub_dir),
        "output_dir": str(output_dir),
        "recursive": args.recursive,
        "resize_to_reference": args.resize_to_reference,
        "device": str(device),
        "reference_image_count": len(reference_index),
        "hub_image_count": len(hub_index),
        "matched_image_count": len(matched_keys),
        "evaluated_image_count": evaluated_count,
        "skipped_image_count": skipped_count,
        "unmatched_reference_count": len(unmatched_reference),
        "unmatched_hub_count": len(unmatched_hub),
        "mean_ssim": f"{mean(ssim_scores):.6f}",
        "mean_ms_ssim": f"{mean(ms_ssim_scores):.6f}",
    }

    summary_csv_path = output_dir / args.summary_csv_name
    detail_csv_path = output_dir / args.detail_csv_name

    append_csv_rows(
        summary_csv_path,
        [
            "reference_dir",
            "hub_dir",
            "output_dir",
            "recursive",
            "resize_to_reference",
            "device",
            "reference_image_count",
            "hub_image_count",
            "matched_image_count",
            "evaluated_image_count",
            "skipped_image_count",
            "unmatched_reference_count",
            "unmatched_hub_count",
            "mean_ssim",
            "mean_ms_ssim",
        ],
        [summary_row],
    )

    append_csv_rows(
        detail_csv_path,
        [
            "image_key",
            "reference_image_path",
            "hub_image_path",
            "ssim",
            "ms_ssim",
        ],
        detail_rows,
    )

    print("\n=== SUMMARY ===")
    print(f"reference_dir={reference_dir}")
    print(f"hub_dir={hub_dir}")
    print(f"output_dir={output_dir.resolve()}")
    print(f"evaluated_pairs={evaluated_count}")
    print(f"mean_ssim={summary_row['mean_ssim']}")
    print(f"mean_ms_ssim={summary_row['mean_ms_ssim']}")
    print(f"summary_csv={summary_csv_path.resolve()}")
    print(f"detail_csv={detail_csv_path.resolve()}")
    print(f"[TIME] Total elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
