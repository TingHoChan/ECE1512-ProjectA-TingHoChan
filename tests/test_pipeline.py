"""Software correctness tests; synthetic fixtures are not benchmark results."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import (  # noqa: E402
    BRACS_2CLASS,
    BRACS_3CLASS,
    HDF5FeatureDataset,
    build_HDF5_feat_dataset,
    mil_collate_fn,
)
from models import ABMIL  # noqa: E402
from training import FocalLoss, evaluate, train_one_epoch  # noqa: E402
from utils import classification_metrics  # noqa: E402


def model_config(n_class=3):
    return SimpleNamespace(
        D_feat=8, D_inner=12, D_attn=6, n_class=n_class, droprate=0.0
    )


class ModelCorrectnessTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.conf = model_config()
        self.model = ABMIL(self.conf).eval()

    def test_mask_normalization_and_padding_value_invariance(self):
        x = torch.randn(2, 7, self.conf.D_feat)
        mask = torch.tensor([[True] * 3 + [False] * 4, [True] * 7])
        with torch.no_grad():
            logits, attention = self.model(x, mask=mask, return_attention=True)
            altered = x.clone()
            altered[~mask] = 1e4
            altered_logits, altered_attention = self.model(
                altered, mask=mask, return_attention=True
            )
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(attention.shape), (2, 1, 7))
        torch.testing.assert_close(attention.sum(-1), torch.ones(2, 1))
        self.assertTrue(torch.equal(attention[:, 0][~mask], torch.zeros(4)))
        self.assertTrue((attention >= 0).all().item())
        torch.testing.assert_close(logits, altered_logits)
        torch.testing.assert_close(attention, altered_attention)

    def test_single_bag_matches_padded_batch(self):
        first = torch.randn(3, self.conf.D_feat)
        second = torch.randn(9, self.conf.D_feat)
        data = mil_collate_fn([
            {"input": first, "label": 0, "slide_id": "short"},
            {"input": second, "label": 1, "slide_id": "long"},
        ])
        with torch.no_grad():
            alone = self.model(first.unsqueeze(0))
            batched = self.model(data["input"], mask=data["mask"])
        torch.testing.assert_close(alone[0], batched[0], atol=1e-6, rtol=1e-5)

    def test_permutation_invariance_and_attention_equivariance(self):
        x = torch.randn(1, 9, self.conf.D_feat)
        order = torch.tensor([8, 1, 5, 0, 3, 7, 2, 6, 4])
        with torch.no_grad():
            logits, attention = self.model(x, return_attention=True)
            permuted_logits, permuted_attention = self.model(
                x[:, order], return_attention=True
            )
        torch.testing.assert_close(logits, permuted_logits, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(attention[:, :, order], permuted_attention)

    def test_all_masked_bag_rejected(self):
        with self.assertRaises((ValueError, RuntimeError)):
            self.model(torch.randn(1, 3, 8), mask=torch.zeros(1, 3, dtype=torch.bool))

    def test_gradients_reach_all_trainable_components(self):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        logits = self.model(torch.randn(4, 6, 8))
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 2, 0]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        component_gradient = {}
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
            component = name.split(".")[0]
            component_gradient[component] = component_gradient.get(component, 0.0) + float(
                parameter.grad.abs().sum()
            )
        self.assertTrue(all(value > 0 for value in component_gradient.values()))
        optimizer.step()
        self.assertTrue(any(
            not torch.equal(before[name], parameter.detach())
            for name, parameter in self.model.named_parameters()
        ))

    def test_focal_gamma_zero_is_cross_entropy(self):
        logits = torch.randn(7, 3, requires_grad=True)
        target = torch.tensor([0, 1, 2, 0, 1, 2, 0])
        torch.testing.assert_close(
            FocalLoss(gamma=0.0)(logits, target), nn.CrossEntropyLoss()(logits, target)
        )
        focal = FocalLoss(gamma=2.0)(logits, target)
        self.assertTrue(torch.isfinite(focal).item())
        focal.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_training_updates_and_evaluation_freezes_parameters(self):
        samples = [
            {"input": torch.randn(3 + i, 8), "label": i % 3, "slide_id": str(i)}
            for i in range(9)
        ]
        loader = DataLoader(samples, batch_size=3, collate_fn=mil_collate_fn)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        before = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        training = train_one_epoch(self.model, criterion, loader, optimizer, torch.device("cpu"))
        self.assertTrue(np.isfinite(training["loss"]))
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in self.model.state_dict().items()))
        selected = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        metrics = evaluate(self.model, criterion, loader, torch.device("cpu"), self.conf)
        self.assertFalse(self.model.training)
        self.assertTrue(all(torch.equal(selected[k], v) for k, v in self.model.state_dict().items()))
        for metric in ("loss", "acc", "auc", "f1"):
            self.assertTrue(np.isfinite(metrics[metric]), metric)
        self.assertGreaterEqual(metrics["auc"], 0)
        self.assertLessEqual(metrics["auc"], 1)
        self.assertGreaterEqual(metrics["acc"], 0)
        self.assertLessEqual(metrics["acc"], 100)


class ClassificationMetricTests(unittest.TestCase):
    def test_macro_metrics_match_hand_counted_example(self):
        probabilities = torch.tensor([
            [0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.3, 0.6, 0.1],
            [0.2, 0.7, 0.1], [0.1, 0.3, 0.6], [0.1, 0.2, 0.7],
        ])
        labels = torch.tensor([0, 0, 0, 1, 1, 2])
        metrics = classification_metrics(probabilities, labels, 3)
        # Class F1s: 4/5, 1/2, 2/3. OVR AUCs: 1, 13/16, 1.
        self.assertAlmostEqual(metrics["f1"], 59 / 90, places=7)
        self.assertAlmostEqual(metrics["auc"], 15 / 16, places=7)
        self.assertAlmostEqual(metrics["acc"], 100 * 4 / 6, places=4)

    def test_missing_class_makes_full_macro_auc_undefined(self):
        probabilities = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
        metrics = classification_metrics(probabilities, torch.tensor([0, 1]), 3)
        self.assertTrue(np.isnan(metrics["auc"]))
        self.assertAlmostEqual(metrics["f1"], 2 / 3, places=7)
        self.assertEqual(metrics["acc"], 100.0)

    def test_invalid_probability_and_label_values_rejected(self):
        with self.assertRaises(ValueError):
            classification_metrics(torch.tensor([[float("nan"), 0.5]]), torch.tensor([0]), 2)
        with self.assertRaises(ValueError):
            classification_metrics(torch.tensor([[0.2, 0.8]]), torch.tensor([2]), 2)


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        # Keep generated fixtures in the explicitly writable project directory.
        # Some managed Windows environments expose a system temp directory that
        # Python can create but cannot subsequently write into or remove.
        temporary_root = PROJECT_ROOT / "tests" / ".tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_", dir=temporary_root)
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_h5(self, filename, labels):
        path = self.directory / filename
        with h5py.File(path, "w") as h5:
            for index, (slide_id, label) in enumerate(labels):
                group = h5.create_group(slide_id)
                group.create_dataset("feat", data=np.ones((2 + index % 3, 8), dtype=np.float32) * index)
                group.create_dataset("coords", data=np.zeros((2 + index % 3, 2), dtype=np.int64))
                group.attrs["label"] = label
        return path

    def write_csv(self, filename, fields, rows):
        with (self.directory / filename).open("w", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def config(self, dataset, n_class):
        return SimpleNamespace(
            dataset=dataset, n_class=n_class, dataset_csv_dir=str(self.directory),
            n_shot=0, val_fraction=0.25, seed=42, cam17_test_center_min=3,
        )

    def assert_roundtrip(self, path, conf, expected_total):
        splits = build_HDF5_feat_dataset(str(path), conf)
        names = [set(split.data_names) for split in splits]
        self.assertTrue(all(len(split) > 0 for split in splits))
        self.assertTrue(names[0].isdisjoint(names[1]))
        self.assertTrue(names[0].isdisjoint(names[2]))
        self.assertTrue(names[1].isdisjoint(names[2]))
        self.assertEqual(len(set.union(*names)), expected_total)
        for split in splits:
            example = split[0]
            self.assertEqual(example["input"].dtype, torch.float32)
            self.assertEqual(example["input"].shape[1], 8)
            self.assertEqual(tuple(example["coords"].shape), (len(example["input"]), 2))
            batch = mil_collate_fn([split[i] for i in range(min(3, len(split)))])
            self.assertEqual(batch["label"].dtype, torch.long)
            self.assertEqual(batch["mask"].dtype, torch.bool)
            self.assertTrue((batch["label"] < conf.n_class).all().item())
            self.assertEqual(int(batch["mask"].sum()), sum(len(split[i]["input"]) for i in range(min(3, len(split)))))
        repeated = build_HDF5_feat_dataset(str(path), conf)
        self.assertEqual([s.data_names for s in splits], [s.data_names for s in repeated])
        return splits

    def test_camelyon16_loader_preserves_test_ids(self):
        labels = [(f"normal_{i:03d}", i % 2) for i in range(24)]
        labels += [(f"test_{i:03d}", i % 2) for i in range(8)]
        path = self.write_h5("c16.h5", labels)
        splits = self.assert_roundtrip(path, self.config("CAMELYON16", 2), 32)
        self.assertEqual(len(splits[2]), 8)
        self.assertTrue(all("test" in name for name in splits[2].data_names))

    def test_camelyon17_loader_preserves_center_holdout(self):
        labels, rows = [], []
        for index in range(40):
            slide_id = f"patient_{index:03d}_node_0"
            center = 0 if index < 28 else 3
            labels.append((slide_id, index % 4))
            rows.append(dict(slide_id=slide_id, center=center, case_id=f"patient_{index:03d}", label=index % 4))
        self.write_csv("camelyon17.csv", ["slide_id", "center", "case_id", "label"], rows)
        path = self.write_h5("c17.h5", labels)
        splits = self.assert_roundtrip(path, self.config("CAMELYON17", 4), 40)
        self.assertEqual(len(splits[2]), 12)
        test_ids = {r["slide_id"] for r in rows if r["center"] >= 3}
        self.assertEqual(set(splits[2].data_names), test_ids)

    def test_bracs_loader_preserves_fixed_splits_and_label_mapping(self):
        labels, rows = [], []
        for split in ("train", "val", "test"):
            for label in range(7):
                slide_id = f"BRACS_{split}_{label}"
                labels.append((slide_id, label))
                rows.append(dict(slide_id=slide_id, split_info=split, label=label))
        self.write_csv("bracs.csv", ["slide_id", "split_info", "label"], rows)
        path = self.write_h5("bracs.h5", labels)
        splits = self.assert_roundtrip(path, self.config("BRACS", 7), 21)
        self.assertEqual([len(split) for split in splits], [7, 7, 7])
        for classes, mapping in ((2, BRACS_2CLASS), (3, BRACS_3CLASS)):
            data = HDF5FeatureDataset(str(path), [name for name, _ in labels], self.config("BRACS", classes))
            self.assertEqual([data[i]["label"] for i in range(len(data))], [mapping[y] for _, y in labels])

    def test_fewshot_selection_is_reproducible_and_per_class(self):
        labels = [(f"normal_{i:03d}", i % 2) for i in range(24)]
        labels += [(f"test_{i:03d}", i % 2) for i in range(8)]
        path = self.write_h5("fewshot.h5", labels)
        conf = self.config("CAMELYON16", 2)
        conf.n_shot = 2
        first = build_HDF5_feat_dataset(str(path), conf)[0]
        second = build_HDF5_feat_dataset(str(path), conf)[0]
        self.assertEqual(first.data_names, second.data_names)
        self.assertEqual(len(first), 4)
        self.assertEqual(sorted(first[i]["label"] for i in range(4)), [0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
