"""CPU checks for shared scoring/resume and model-specific image batch handling."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import esr_backends as backends
import evaluation_esr as evaluator


class BackendEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prompts = self.root / "prompts.json"
        self.prompts.write_text(json.dumps({
            "Cat + Fox": [{"idx": 7, "prompt": "two targets"},
                          {"idx": 8, "prompt": "missing after image"}],
            "Cat + Fox + Tree": [{"index": 11, "prompt": "three targets"}],
            "Van Gogh + Train": ["plain string style prompt"],
        }))
        for pair, idx in [("Fox + Cat", 7), ("Tree + Fox + Cat", 11), ("Train + Van Gogh", 0)]:
            self.save_image(self.root / f"method/after/seed_42/{pair}/{idx:04d}/result_comp_{idx:04d}.png")
            for seed in [42, 43, 44]:
                self.save_image(self.root / f"flux/before/seed_{seed}/{pair}/{idx}/result_base.png")

    @staticmethod
    def save_image(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(path)

    def args(self, backend):
        return ["--results_root", str(self.root), "--prompts_json", str(self.prompts),
                "--method", "method", "--category", "after", "--base_category", "before",
                "--index_mode", "explicit", "--batch_size", "4", "--allow_partial",
                "--intermediate_jsonl", str(self.root / f"{backend}.jsonl"),
                "--out_csv", str(self.root / f"{backend}.csv")]

    @staticmethod
    def responses(**kwargs):
        result = []
        for sample in kwargs["batch_samples"]:
            if sample["target"] == "Tree":
                result.append("Cannot determine.")
            elif sample["target"] == "Cat" and sample["prompt_text"] == "two targets":
                result.append("present.")
            else:
                result.append("absent")
        return result

    def run_eval(self, backend, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            evaluator.main(backend=backend, argv=self.args(backend) + list(extra))

    def test_scoring_mixed_concepts_missing_image_and_resume(self):
        for backend in ("gemma",):
            with self.subTest(backend=backend), patch.object(backends, "load_backend", return_value=(Mock(), Mock())) as loader, \
                    patch.object(backends, "infer_batch", side_effect=self.responses):
                self.run_eval(backend)
                path = self.root / f"{backend}.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual([r["idx"] for r in rows], [7, 11, 0])
                self.assertEqual([r["all_absent"] for r in rows], [False, False, True])
                self.assertEqual([r["erasure_fraction"] for r in rows], [0.5, 2/3, 1.0])
                self.assertEqual(rows[1]["verdict_2"], "invalid")
                self.assertEqual(rows[1]["response_2"], "Cannot determine.")
                original = path.read_bytes()
                self.run_eval(backend)
                loader.assert_called_once()
                self.assertEqual(path.read_bytes(), original)
                with (self.root / f"{backend}.csv").open() as handle:
                    summaries = list(csv.DictReader(handle))
                self.assertEqual(len(summaries), 1)
                summary = summaries[0]
                self.assertEqual(summary["evaluated"], "3")
                self.assertEqual(summary["invalid_targets"], "1")
                self.assertAlmostEqual(float(summary["success_rate"]), 1/3, places=6)
                self.assertAlmostEqual(float(summary["fractional_erasure_rate"]), (0.5+2/3+1)/3, places=6)
                with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                    self.run_eval(backend, "--model_id", "different/model")

    def test_limited_run_resumes_remaining_samples(self):
        with patch.object(backends, "load_backend", return_value=(Mock(), Mock())), \
                patch.object(backends, "infer_batch", side_effect=self.responses):
            self.run_eval("gemma", "--max_samples", "1")
            path = self.root / "gemma.jsonl"
            self.assertEqual(len(path.read_text().splitlines()), 1)
            self.run_eval("gemma")
            self.assertEqual(len(path.read_text().splitlines()), 3)

    def test_separate_reference_root_and_resume_mismatch(self):
        reference_root = self.root / "reference_images"
        (self.root / "flux").rename(reference_root)
        with patch.object(backends, "load_backend", return_value=(Mock(), Mock())), \
                patch.object(backends, "infer_batch", side_effect=self.responses):
            self.run_eval("gemma", "--base_root", str(reference_root))
            rows = [json.loads(line) for line in (self.root / "gemma.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(Path(path).is_relative_to(reference_root)
                                for row in rows for path in row["before_paths"]))
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                self.run_eval("gemma", "--base_root", str(self.root / "different_references"))

    def test_existing_qwen_keeps_legacy_output_format(self):
        with patch("transformers.Qwen3VLForConditionalGeneration.from_pretrained", return_value=Mock()), \
                patch("transformers.AutoProcessor.from_pretrained", return_value=Mock()), \
                patch.object(evaluator, "eval_target_erasure_chat_batch", side_effect=self.responses):
            self.run_eval("qwen")
        rows = [json.loads(line) for line in (self.root / "qwen.jsonl").read_text().splitlines()]
        self.assertEqual([row["all_absent"] for row in rows], [False, False, True])
        self.assertNotIn("eval_config", rows[0])
        with (self.root / "qwen.csv").open() as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, ["method", "category", "seed", "success_rate", "evaluated", "success"])
            self.assertEqual(next(reader)["success"], "1")

    def test_inference_failure_is_not_recorded_as_erasure_success(self):
        with patch.object(backends, "load_backend", return_value=(Mock(), Mock())), \
                patch.object(backends, "infer_batch", side_effect=RuntimeError("CUDA out of memory")):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                self.run_eval("gemma")
        self.assertFalse((self.root / "gemma.jsonl").exists())

    def test_missing_images_require_explicit_partial_evaluation(self):
        argv = self.args("gemma")
        argv.remove("--allow_partial")
        with patch.object(backends, "load_backend", return_value=(Mock(), Mock())), \
                patch.object(backends, "infer_batch", side_effect=self.responses), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "skipped=1"):
                evaluator.main(backend="gemma", argv=argv)
        self.assertEqual(len((self.root / "gemma.jsonl").read_text().splitlines()), 3)

    def test_missing_concept_directory_is_skipped_in_partial_mode(self):
        prompts = json.loads(self.prompts.read_text())
        prompts["Missing + Concept"] = [{"prompt": "missing directory"}]
        self.prompts.write_text(json.dumps(prompts))
        with patch.object(backends, "load_backend", return_value=(Mock(), Mock())), \
                patch.object(backends, "infer_batch", side_effect=self.responses):
            self.run_eval("gemma")
        self.assertEqual(len((self.root / "gemma.jsonl").read_text().splitlines()), 3)

    def test_template_keeps_four_images_per_sample_and_disables_thinking(self):
        images = [[Image.new("RGB", (4, 4), (i*20, j*20, 0)) for j in range(4)] for i in range(2)]
        samples = [{"images": group, "target": "Fox", "prompt_text": "Fox and Cat"} for group in images]
        for backend in ("gemma",):
            processor = Mock()
            backends.prepare_inputs(backend, processor, samples)
            args, kwargs = processor.apply_chat_template.call_args
            messages = args[0]
            for i in range(2):
                content = messages[i][0]["content"]
                self.assertEqual([c["image"] for c in content[:4]], images[i])
                self.assertEqual(content[4]["text"], evaluator.build_eval_prompt("Fox", "Fox and Cat"))
            if backend == "gemma":
                self.assertIs(kwargs["enable_thinking"], False)
                self.assertTrue(kwargs["processor_kwargs"]["padding"])
        with self.assertRaisesRegex(ValueError, "reference images"):
            backends.prepare_inputs("gemma", Mock(), [dict(samples[0], images=images[0][:3])])

    def test_batch_decode_excludes_entire_padded_input(self):
        inputs = {"input_ids": torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]])}
        batch = Mock()
        batch.to.return_value = inputs
        model = Mock(device="cpu")
        model.generate.return_value = torch.tensor([[0, 0, 1, 2, 11], [1, 2, 3, 4, 12]])
        processor = Mock()
        processor.batch_decode.return_value = ["present", "absent"]
        with patch.object(backends, "prepare_inputs", return_value=batch):
            result = backends.infer_batch("gemma", model, processor, [Mock(), Mock()])
        self.assertEqual(result, ["present", "absent"])
        decoded_ids = processor.batch_decode.call_args.args[0]
        self.assertEqual(decoded_ids.tolist(), [[11], [12]])
        self.assertFalse(model.generate.call_args.kwargs["do_sample"])


if __name__ == "__main__":
    unittest.main()
