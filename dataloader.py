"""Dataset utilities for pre-extracted WSI features stored in HDF5."""

from __future__ import annotations

import os
import random
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def _cfg(conf, name, default=None):
    return getattr(conf, name, default)


def _resolve_csv(conf, filename: str) -> str:
    """Resolve a dataset metadata CSV from config or common project locations."""
    explicit = _cfg(conf, f"{os.path.splitext(filename)[0]}_csv", None)
    if explicit:
        return explicit
    csv_dir = _cfg(conf, "dataset_csv_dir", "./dataset_csv")
    return os.path.join(csv_dir, filename)


def _read_labels(file_path: str, names: Sequence[str]) -> List[int]:
    with h5py.File(file_path, "r") as h5:
        return [int(h5[name].attrs["label"]) for name in names]


def _safe_train_val_split(
    names: Sequence[str], labels: Sequence[int], val_fraction: float, seed: int
) -> Tuple[List[str], List[str]]:
    names = list(names)
    labels = list(labels)
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
    if len(names) < 2:
        raise ValueError("At least two training/validation slides are required.")

    # Stratification is preferred, but sklearn requires >=2 examples/class and
    # a validation set large enough to include every class.
    counts = pd.Series(labels).value_counts()
    # Match sklearn's rounding so the stratification check uses the actual sizes.
    n_val = int(math.ceil(len(names) * val_fraction))
    can_stratify = (
        len(counts) > 1
        and counts.min() >= 2
        and n_val >= len(counts)
        and (len(names) - n_val) >= len(counts)
    )
    stratify = labels if can_stratify else None
    train_names, val_names = train_test_split(
        names,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return list(train_names), list(val_names)


def split_dataset_camelyon16(file_path, conf):
    with h5py.File(file_path, "r") as h5:
        slide_names = sorted(h5.keys())

    test_token = str(_cfg(conf, "cam16_test_token", "test")).lower()
    test_names = [n for n in slide_names if test_token in n.lower()]
    train_val_names = [n for n in slide_names if n not in set(test_names)]
    if not test_names:
        raise ValueError(
            "No CAMELYON16 test slides were found. By default, slide IDs must "
            "contain 'test'. Set conf.cam16_test_token if your naming differs."
        )

    labels = _read_labels(file_path, train_val_names)
    train_names, val_names = _safe_train_val_split(
        train_val_names,
        labels,
        float(_cfg(conf, "val_fraction", 0.1)),
        int(_cfg(conf, "seed", 42)),
    )
    return train_names, val_names, test_names


def split_dataset_camelyon17(file_path, conf):
    csv_path = _resolve_csv(conf, "camelyon17.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CAMELYON17 metadata not found: {csv_path}. Set dataset_csv_dir or camelyon17_csv."
        )
    slide_info = pd.read_csv(csv_path).set_index("slide_id")
    with h5py.File(file_path, "r") as h5:
        slide_names = sorted(h5.keys())

    missing = [n for n in slide_names if n not in slide_info.index]
    if missing:
        raise KeyError(f"{len(missing)} HDF5 slides are missing from {csv_path}; first: {missing[:5]}")

    test_center_min = int(_cfg(conf, "cam17_test_center_min", 3))
    test_names = [n for n in slide_names if int(slide_info.loc[n, "center"]) >= test_center_min]
    train_val_names = [n for n in slide_names if n not in set(test_names)]
    labels = _read_labels(file_path, train_val_names)
    train_names, val_names = _safe_train_val_split(
        train_val_names,
        labels,
        float(_cfg(conf, "val_fraction", 0.1)),
        int(_cfg(conf, "seed", 42)),
    )
    return train_names, val_names, test_names


def split_dataset_bracs(file_path, conf):
    csv_path = _resolve_csv(conf, "bracs.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"BRACS metadata not found: {csv_path}. Set dataset_csv_dir or bracs_csv."
        )
    slide_info = pd.read_csv(csv_path).set_index("slide_id")
    with h5py.File(file_path, "r") as h5:
        slide_names = sorted(h5.keys())

    missing = [n for n in slide_names if n not in slide_info.index]
    if missing:
        raise KeyError(f"{len(missing)} HDF5 slides are missing from {csv_path}; first: {missing[:5]}")

    train_names, val_names, test_names = [], [], []
    for slide_id in slide_names:
        split_info = str(slide_info.loc[slide_id, "split_info"]).strip().lower()
        if split_info == "train":
            train_names.append(slide_id)
        elif split_info in {"val", "valid", "validation"}:
            val_names.append(slide_id)
        elif split_info == "test":
            test_names.append(slide_id)
        else:
            raise ValueError(f"Unknown BRACS split_info={split_info!r} for slide {slide_id}")
    return train_names, val_names, test_names


BRACS_3CLASS = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
BRACS_2CLASS = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}


def _map_label(label: int, dataset: str, n_class: int) -> int:
    if dataset.upper() != "BRACS":
        return int(label)
    if n_class == 3:
        return BRACS_3CLASS[int(label)]
    if n_class == 2:
        return BRACS_2CLASS[int(label)]
    return int(label)


class HDF5FeatureDataset(Dataset):
    """Lazy HDF5 dataset.

    Features are read only when a slide is requested, avoiding the original
    implementation's potentially very large in-memory copy of the whole HDF5.
    """

    def __init__(self, file_path: str, slide_names: Sequence[str], conf):
        self.file_path = file_path
        self.data_names = list(slide_names)
        self.dataset_name = str(conf.dataset)
        self.n_class = int(conf.n_class)
        # Optional storage cache only. It never changes the slide list, patch
        # count, labels, or float32 values supplied to the model.
        self._cache = {}
        self._cache_bytes = 0

    def __len__(self):
        return len(self.data_names)

    @property
    def feature_dim(self):
        with h5py.File(self.file_path, "r") as h5:
            return int(h5[self.data_names[0]]["feat"].shape[-1])

    @property
    def cache_bytes(self):
        return self._cache_bytes

    @property
    def has_device_cache(self):
        return any(item['input'].device.type != 'cpu' for item in self._cache.values())

    def clear_cache(self):
        """Release this dataset's cached tensors; lazy reading remains available."""
        self._cache.clear()
        self._cache_bytes = 0

    def warm_cache(self, device='cuda', max_bytes=None):
        """Cache complete bags within a deterministic device-memory budget.

        Float16 source features are stored as float16 on the device, then
        converted exactly to float32 by ``mil_collate_fn``. Other source types
        retain the baseline reader's float32 representation. Coordinates remain
        on CPU. Bags that do not fit are still read lazily; no bag or patch is
        removed. Use a single-process DataLoader with pin_memory=False whenever
        ``has_device_cache`` is true.

        ``max_bytes`` counts feature tensors, not model/activation allocations.
        With no explicit limit, at most 55% of currently free CUDA memory is
        used, leaving room for the model and full-bag activations.
        """
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('The optional feature cache requires a CUDA device.')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA feature caching requested, but CUDA is unavailable.')
        self.clear_cache()
        if max_bytes is None:
            max_bytes = int(torch.cuda.mem_get_info(device)[0] * 0.55)
        max_bytes = int(max_bytes)
        if max_bytes < 0:
            raise ValueError('max_bytes must be nonnegative')
        metadata = []
        with h5py.File(self.file_path, 'r') as h5:
            for name in self.data_names:
                feat = h5[name]['feat']
                if len(feat.shape) != 2 or feat.shape[0] == 0:
                    raise ValueError(f'Slide {name}: expected nonempty feat [N,D]')
                dtype = torch.float16 if feat.dtype == np.dtype('float16') else torch.float32
                bytes_per_value = 2 if dtype == torch.float16 else 4
                metadata.append((name, int(np.prod(feat.shape)) * bytes_per_value, dtype))
        for index, (name, nbytes, dtype) in enumerate(metadata):
            if self._cache_bytes + nbytes > max_bytes:
                continue
            # Reuse all normal validation, mapping and coordinate handling.
            item = self[index]
            try:
                item['input'] = item['input'].to(device=device, dtype=dtype)
            except torch.cuda.OutOfMemoryError:
                # Caching is optional: retain the already cached bags and read
                # this and remaining bags lazily rather than changing the task.
                break
            self._cache[name] = item
            self._cache_bytes += nbytes
        return {'cached_slides': len(self._cache), 'total_slides': len(self),
                'cache_bytes': self._cache_bytes, 'max_bytes': max_bytes,
                'device': str(device)}

    def __getitem__(self, index):
        slide_id = self.data_names[index]
        if slide_id in self._cache:
            if torch.utils.data.get_worker_info() is not None:
                raise RuntimeError('CUDA-cached datasets require DataLoader num_workers=0.')
            return dict(self._cache[slide_id])
        with h5py.File(self.file_path, "r") as h5:
            slide = h5[slide_id]
            feat = np.asarray(slide["feat"][:], dtype=np.float32)
            coords = np.asarray(slide["coords"][:]) if "coords" in slide else None
            raw_label = int(slide.attrs["label"])

        if feat.ndim != 2:
            raise ValueError(f"Slide {slide_id}: expected feat [N,D], got {feat.shape}")
        if feat.shape[0] == 0:
            raise ValueError(f"Slide {slide_id} contains zero patches.")
        if not np.isfinite(feat).all():
            raise ValueError(f"Slide {slide_id}: nonfinite features")
        if coords is not None and coords.shape != (len(feat), 2):
            raise ValueError(f"Slide {slide_id}: expected matching coords [N,2]")

        label = _map_label(raw_label, self.dataset_name, self.n_class)
        if not 0 <= label < self.n_class:
            raise ValueError(f"Slide {slide_id}: class {label} outside configured class range")
        return {
            "input": torch.from_numpy(feat),
            "coords": None if coords is None else torch.from_numpy(coords),
            "label": label,
            "slide_id": slide_id,
        }


# Backwards-compatible alias from the starter code.
HDF5_feat_dataset2 = HDF5FeatureDataset


def _labels_for_names(file_path: str, names: Sequence[str], conf) -> List[int]:
    raw = _read_labels(file_path, names)
    return [_map_label(x, str(conf.dataset), int(conf.n_class)) for x in raw]


def generate_fewshot_names(file_path: str, train_names: Sequence[str], num_shots: int, conf):
    """Select ``num_shots`` slides per class reproducibly.

    ``num_shots <= 0`` means use the full training set.
    """
    train_names = list(train_names)
    if num_shots <= 0:
        return train_names

    labels = _labels_for_names(file_path, train_names, conf)
    rng = random.Random(int(_cfg(conf, "seed", 42)))
    selected = []
    for label in sorted(set(labels)):
        class_names = [n for n, y in zip(train_names, labels) if y == label]
        if num_shots >= len(class_names):
            chosen = class_names
        else:
            chosen = rng.sample(class_names, num_shots)
        selected.extend(chosen)
    rng.shuffle(selected)
    return selected


def generate_fewshot_dataset(train_split, train_names, num_shots):
    """Compatibility helper for legacy in-memory dictionaries."""
    if num_shots <= 0:
        return train_split, train_names
    labels = [train_split[name]["label"] for name in train_names]
    selected = []
    for label in sorted(set(labels)):
        class_names = [n for n, y in zip(train_names, labels) if y == label]
        selected.extend(random.sample(class_names, min(num_shots, len(class_names))))
    return {n: train_split[n] for n in selected}, selected


def build_HDF5_feat_dataset(file_path, conf):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feature HDF5 not found: {file_path}")

    dataset = str(conf.dataset).upper()
    if dataset == "CAMELYON16":
        train_names, val_names, test_names = split_dataset_camelyon16(file_path, conf)
    elif dataset == "CAMELYON17":
        train_names, val_names, test_names = split_dataset_camelyon17(file_path, conf)
    elif dataset == "BRACS":
        train_names, val_names, test_names = split_dataset_bracs(file_path, conf)
    else:
        raise ValueError(f"Unsupported dataset: {conf.dataset}. Expected CAMELYON16, CAMELYON17, or BRACS.")

    train_names = generate_fewshot_names(
        file_path, train_names, int(_cfg(conf, "n_shot", 0)), conf
    )
    return (
        HDF5FeatureDataset(file_path, train_names, conf),
        HDF5FeatureDataset(file_path, val_names, conf),
        HDF5FeatureDataset(file_path, test_names, conf),
    )


def mil_collate_fn(batch):
    """Pad variable-size MIL bags and create an attention mask.

    Returns:
      input: ``[B, max_N, D]`` float tensor
      mask: ``[B, max_N]`` bool tensor, True for real patches
      label: ``[B]`` long tensor
      coords: list of per-slide coordinate tensors (or None)
      slide_id: list of strings
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    feature_dims = {int(item["input"].shape[1]) for item in batch}
    if len(feature_dims) != 1:
        raise ValueError(f"All bags must have the same feature dimension; got {sorted(feature_dims)}")
    d = feature_dims.pop()
    max_n = max(int(item["input"].shape[0]) for item in batch)
    b = len(batch)

    devices = {item['input'].device for item in batch if item['input'].device.type != 'cpu'}
    if len(devices) > 1:
        raise ValueError('All device-cached bags in a batch must use the same device.')
    device = next(iter(devices), torch.device('cpu'))
    inputs = torch.zeros((b, max_n, d), dtype=torch.float32, device=device)
    mask = torch.zeros((b, max_n), dtype=torch.bool, device=device)
    labels = torch.empty((b,), dtype=torch.long, device=device)
    coords, slide_ids = [], []

    for i, item in enumerate(batch):
        x = item["input"].to(device=device, dtype=torch.float32)
        n = x.shape[0]
        inputs[i, :n] = x
        mask[i, :n] = True
        labels[i] = int(item["label"])
        coords.append(item.get("coords"))
        slide_ids.append(item.get("slide_id", str(i)))

    return {
        "input": inputs,
        "mask": mask,
        "label": labels,
        "coords": coords,
        "slide_id": slide_ids,
    }
