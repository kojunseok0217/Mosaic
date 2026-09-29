"""CPU integration checks using deterministic stand-ins for the a learned metric."""

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluation_aesthetic as quality


class QualityEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / "results"
        self.results.mkdir()
        self.out = self.root / "output"
        self.shapes = []
        self.loads = []

    def image(self, method="EraseAnything", category="intra_2_character", seed=42,
              concept="Mario + Pikachu", idx="0000", color=51, size=(13, 9), filename=None):
        path = (self.results / method / category / f"seed_{seed}" / concept / idx
                / (filename or f"result_comp_{idx}.png"))
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (color, color, color)).save(path)
        return path

    def args(self, *extra):
        return ["--results_root", str(self.results), "--out_dir", str(self.out),
                "--scan_workers", "1", "--device", "cpu", "--methods", "eraseanything", "mace", "mosaic", *extra]

    def plan(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return quality.build_plan(quality.parse_args(self.args(*extra)))

    @staticmethod
    def read_csv(path):
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def run_main(self, *extra):
        def factory(metric, **kwargs):
            self.loads.append(metric)
            factor = 10

            class Metric(torch.nn.Module):
                def forward(inner, tensor):
                    self.shapes.append(tuple(tensor.shape))
                    return tensor.mean(dim=(1, 2, 3), keepdim=True) * factor

            return Metric()

        real_version = quality.importlib.metadata.version
        with patch.dict(sys.modules, pyiqa=SimpleNamespace(create_metric=factory)), \
             patch.object(quality.importlib.metadata, "version",
                          side_effect=lambda n: "test" if n == "pyiqa" else real_version(n)), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return quality.main(self.args(*extra))

    def test_discovery_routes_after_and_deduplicates_symlinks(self):
        image = self.image()
        (image.parent / "result_comp_alias.png").symlink_to(image)
        self.image(filename="result_base.png")
        self.image(method="EraseAnything_celeb", category="celeb_after")
        self.image(method="EraseAnything_celeb", category="celeb_before")
        self.image(method="EraseAnything_style_nsfw", category="style_after")
        rows, coverage = self.plan("--methods", "eraseanything", "--categories", "all")
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["category"] for r in rows}, {"intra_2_character", "celeb", "style"})
        self.assertEqual(len(coverage), 10)
        self.assertEqual(sum(c["n_available"] for c in coverage), 3)

    def test_common_normalizes_keys_and_preserves_missing_seeds(self):
        self.image()
        self.image(method="MACE", concept=" pikachu + MARIO ", idx="0")
        self.image(method="mosaic", concept="Pikachu+Mario", idx="00")
        self.image(idx="0001")
        self.image(seed=43)
        rows, coverage = self.plan("--categories", "intra_2_character", "--selection", "common", "--seeds", "all")
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({quality.sample_key(r) for r in rows}), 1)
        self.assertEqual({r["seed"] for r in rows}, {42})
        self.assertEqual(len(coverage), 6)
        self.assertEqual(sum(c["n_selected"] for c in coverage if c["seed"] == 43), 0)

    def test_ambiguous_common_key_is_not_silently_discarded(self):
        self.image()
        self.image(concept="Pikachu + Mario")
        rows, coverage = self.plan("--methods", "eraseanything", "--categories", "intra_2_character")
        self.assertEqual(len(rows), 2)
        self.assertEqual(coverage[0]["n_ambiguous_keys"], 1)
        with self.assertRaisesRegex(ValueError, "Ambiguous common samples"):
            self.plan("--methods", "eraseanything", "--categories", "intra_2_character", "--selection", "common")

    def test_dry_run_requires_no_models_and_writes_nothing(self):
        self.image()
        with patch.dict(sys.modules, pyiqa=None), contextlib.redirect_stdout(io.StringIO()):
            rc = quality.main(self.args("--dry_run", "--methods", "eraseanything"))
        self.assertEqual(rc, 0)
        self.assertFalse(self.out.exists())

    def test_negative_methods_discovery_common_and_inference(self):
        for method in quality.NEGATIVE_METHODS:
            self.image(method=method, idx="0017", color=51)
            self.image(method=method, category="cross_3_OBCH", idx="0005", color=102)
        # An unmatched image must be excluded from the common comparison.
        self.image(method=quality.NEGATIVE_METHODS[0], idx="0020")
        extra = ["--methods", *quality.NEGATIVE_METHODS, "--categories", "general", "--selection", "common"]
        rows, coverage = self.plan(*extra)
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["idx"] for r in rows}, {5, 17})
        self.assertEqual(len(coverage), 14)
        self.assertEqual(self.run_main(*extra), 0)
        summary = self.read_csv(self.out / "summary.csv")
        self.assertEqual({r["method"] for r in summary}, set(quality.NEGATIVE_METHODS))
        for row in summary:
            self.assertEqual(row["n_evaluated"], "2")
            self.assertAlmostEqual(float(row["aesthetic_mean"]), 3, places=4)
        self.loads.clear()
        self.assertEqual(self.run_main(*extra, "--resume"), 0)
        self.assertEqual(self.loads, [])

    def test_negative_methods_reject_unsupported_scope_and_mixed_common_indices(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                quality.parse_args(self.args("--methods", quality.NEGATIVE_METHODS[0], "--categories", "style"))
            with self.assertRaises(SystemExit):
                quality.parse_args(self.args("--methods", quality.NEGATIVE_METHODS[0], "mace",
                                            "--categories", "general", "--selection", "common"))

    def test_inference_native_shapes_weighted_mean_resume_and_changed_file(self):
        first = self.image(color=51)  # score 2
        self.image(idx="0001", color=102)  # score 4
        self.image(category="cross_2_CO", color=153, size=(7, 11))  # score 6
        extra = ["--methods", "eraseanything", "--categories", "intra_2_character", "cross_2_CO", "--batch_size", "3"]
        self.assertEqual(self.run_main(*extra), 0)
        self.assertEqual(self.loads, ["laion_aes"])
        self.assertIn((2, 3, 9, 13), self.shapes)
        self.assertIn((1, 3, 11, 7), self.shapes)
        summary = self.read_csv(self.out / "summary.csv")[0]
        self.assertAlmostEqual(float(summary["aesthetic_std"]), 2, places=4)
        self.assertAlmostEqual(float(summary["aesthetic_mean"]), 4, places=4)
        with self.assertRaises(FileExistsError):
            self.run_main(*extra)
        self.loads.clear()
        self.shapes.clear()
        self.assertEqual(self.run_main(*extra, "--resume"), 0)
        self.assertEqual(self.loads, [])
        self.assertEqual(self.shapes, [])
        # Resume must invalidate an image whose size or modification time changed.
        Image.new("RGB", (15, 12), (255, 255, 255)).save(first)
        self.assertEqual(self.run_main(*extra, "--resume"), 0)
        self.assertEqual(len(self.shapes), 1)  # only the changed image is rescored
        summary = self.read_csv(self.out / "summary.csv")[0]
        self.assertEqual(summary["n_evaluated"], "3")
        self.assertAlmostEqual(float(summary["aesthetic_mean"]), 20 / 3, places=4)
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.run_main(*extra, "--resume", "--filename_pattern", "*.png")

    def test_corrupt_image_common_denominators_and_retry(self):
        for method in quality.METHOD_DIRS:
            directory = quality.METHOD_DIRS[method][0]
            self.image(method=directory, color=51)
            self.image(method=directory, idx="0001", color=102)
        corrupt = self.results / "MACE/intra_2_character/seed_42/Mario + Pikachu/0001/result_comp_0001.png"
        corrupt.write_bytes(b"broken png")
        extra = ["--categories", "intra_2_character", "--selection", "common"]
        self.assertEqual(self.run_main(*extra), 1)
        summary = self.read_csv(self.out / "summary.csv")
        self.assertEqual([r["n_evaluated"] for r in summary], ["1"] * 3)
        self.assertEqual(sum(int(r["n_failed"]) for r in summary), 1)
        self.assertEqual(sum(int(r["n_peer_failed"]) for r in summary), 2)
        for row in summary:
            self.assertAlmostEqual(float(row["aesthetic_mean"]), 2, places=4)
        self.image(method="MACE", idx="0001", color=102)
        self.shapes.clear()
        self.assertEqual(self.run_main(*extra, "--resume"), 0)
        self.assertEqual(len(self.shapes), 1)
        self.assertEqual([r["n_evaluated"] for r in self.read_csv(self.out / "summary.csv")], ["2"] * 3)
        self.assertEqual(len(self.read_csv(self.out / "per_image.csv")), 6)

    def test_cache_recovers_interrupted_tail_but_rejects_internal_corruption(self):
        path = self.root / "progress.jsonl"
        record = dict(method="mace", image_path="/test.png", size_bytes=1, mtime_ns=2, status="ok")
        path.write_text(json.dumps(record) + '\n{"method":', encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(len(quality.load_cache(path)), 1)
        self.assertEqual(path.read_text(), json.dumps(record) + "\n")
        path.write_text(json.dumps(record))
        self.assertEqual(len(quality.load_cache(path)), 1)
        self.assertTrue(path.read_bytes().endswith(b"\n"))
        path.write_text('broken\n' + json.dumps(record) + '\n')
        with self.assertRaisesRegex(ValueError, "Invalid cache record inside"):
            quality.load_cache(path)

    def test_nonfinite_inference_aborts_without_fabricating_scores(self):
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            quality.score_batch([{}], [torch.zeros(3, 4, 4)],
                                {"aesthetic": lambda x: torch.tensor([float("nan")])}, torch, "cpu")


if __name__ == "__main__":
    unittest.main()
