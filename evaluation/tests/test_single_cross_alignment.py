"""Regression checks for single-image/reference judges and preservation scoring."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluation_esr_single_cross as single
import evaluation_selective_alignment as alignment


class SingleCrossTests(unittest.TestCase):
    def test_both_backends_keep_reference_and_single_image_protocols(self):
        for backend in ("qwen", "gemma"):
            for reference in (False, True):
                with self.subTest(backend=backend, reference=reference):
                    images = [Image.new("RGB", (4, 4)) for _ in range(4 if reference else 1)]
                    inputs = {"input_ids": torch.tensor([[0, 1, 2, 3]])}
                    batch = Mock()
                    batch.to.return_value = inputs
                    processor = Mock()
                    processor.apply_chat_template.return_value = batch if backend == "gemma" else "prompt"
                    processor.return_value = batch
                    processor.decode.return_value = "absent"
                    model = Mock(device="cpu")
                    model.generate.return_value = torch.tensor([[0, 1, 2, 3, 11]])
                    kwargs = dict(model=model, processor=processor, target="Cat",
                                  prompt_text="Cat and Fox", backend=backend)
                    if reference:
                        verdict = single.eval_target_erasure_chat(images=images, **kwargs)
                    else:
                        verdict = single.eval_target_presence_chat(image=images[0], **kwargs)
                    self.assertEqual(verdict, "absent")
                    self.assertEqual(processor.decode.call_args.args[0].tolist(), [11])
                    messages = processor.apply_chat_template.call_args.args[0]
                    if backend == "gemma":
                        self.assertFalse(processor.apply_chat_template.call_args.kwargs["enable_thinking"])
                        messages = messages[0]
                    self.assertEqual([part["image"] for part in messages[0]["content"]
                                      if part["type"] == "image"], images)
                    self.assertFalse(model.generate.call_args.kwargs["do_sample"])

    def test_gemma_defaults_do_not_share_qwen_outputs(self):
        qwen = single.parse_args(["--json_path", "prompts.json"])
        gemma = single.parse_args(["--json_path", "prompts.json", "--backend", "gemma"])
        self.assertEqual(qwen.model_id, "Qwen/Qwen3-VL-8B-Instruct")
        self.assertEqual(gemma.model_id, "google/gemma-4-12B-it")
        self.assertNotEqual(qwen.out_csv, gemma.out_csv)
        self.assertNotEqual(qwen.out_detail_csv, gemma.out_detail_csv)


class AlignmentTests(unittest.TestCase):
    def test_target_head_nouns_excluded_without_substring_matches(self):
        self.assertTrue(alignment.is_target_element("dog", {"siberian husky dog"}))
        self.assertFalse(alignment.is_target_element("cat", {"caterpillar"}))

    def test_omitted_entities_remain_in_denominator(self):
        elements = ["Tree", "road", "cloud"]
        verdicts = alignment.align_entity_verdict({"tree": "present", "road": "unknown"}, elements)
        self.assertEqual(verdicts, {"Tree": "PRESENT", "road": "UNCERTAIN", "cloud": "UNCERTAIN"})

    def test_resume_updates_summary_and_preserves_other_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            row = dict(method="mosaic", category="intra_2_character", seed=42,
                       mean_present_ratio=0.2, evaluated=1)
            alignment.append_seed_summary_csv(str(path), row)
            alignment.append_seed_summary_csv(str(path), dict(row, seed=43))
            alignment.append_seed_summary_csv(str(path), dict(row, mean_present_ratio=0.7, evaluated=3))
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({r["seed"]: r["evaluated"] for r in rows}, {"42": "3", "43": "1"})


if __name__ == "__main__":
    unittest.main()
