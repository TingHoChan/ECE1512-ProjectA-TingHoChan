"""Reproducible baseline/focal runs on the supplied, unchanged split protocol."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import build_HDF5_feat_dataset, mil_collate_fn, _labels_for_names
from models import ABMIL
from training import FocalLoss, train_one_epoch, evaluate
from utils import load_config, seed_everything, infer_feature_dim

ROOT = Path(__file__).resolve().parent
CONFIGS = {'C16': 'camelyon16_medical_ssl_config.yml',
           'C17': 'camelyon17_medical_ssl_config.yml',
           'BRACS': 'bracs_medical_ssl_config.yml'}


def save_json(path, data):
    def clean(v):
        if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [clean(x) for x in v]
        if isinstance(v, (float, np.floating)) and not math.isfinite(v): return None
        if isinstance(v, np.integer): return int(v)
        return v
    Path(path).write_text(json.dumps(clean(data), indent=2, allow_nan=False), encoding='utf8')


def load_project_config(dataset, seed=42):
    conf = load_config(ROOT / 'config' / CONFIGS[dataset])
    for attr in ['data_dir', 'dataset_csv_dir']:
        value = Path(getattr(conf, attr))
        setattr(conf, attr, str(value if value.is_absolute() else ROOT / value))
    conf.seed = seed
    return conf


def make_loaders(conf, datasets):
    generator = torch.Generator().manual_seed(conf.seed)
    device_cache = any(getattr(d, 'has_device_cache', False) for d in datasets)
    if device_cache and int(conf.n_worker) != 0:
        raise ValueError('CUDA feature caching requires n_worker=0')
    common = dict(batch_size=int(conf.B), num_workers=int(conf.n_worker),
                  pin_memory=bool(conf.pin_memory) and torch.cuda.is_available() and not device_cache,
                  collate_fn=mil_collate_fn)
    return [DataLoader(d, shuffle=(i == 0), generator=generator if i == 0 else None,
                       **common) for i, d in enumerate(datasets)]


def audit_splits(conf, datasets):
    names = [d.data_names for d in datasets]
    if any(len(n) == 0 for n in names): raise ValueError('All three splits must be nonempty')
    if any(set(names[i]) & set(names[j]) for i,j in [(0,1),(0,2),(1,2)]):
        raise ValueError('Slide leakage between splits')
    result = {}
    for split, ds in zip(['train','val','test'], datasets):
        labels = _labels_for_names(conf.data_dir, ds.data_names, conf)
        if min(labels) < 0 or max(labels) >= conf.n_class:
            raise ValueError('Feature labels incompatible with configured class count')
        result[split] = {'slides': ds.data_names, 'n':len(ds),
                         'class_counts':np.bincount(labels, minlength=conf.n_class).tolist()}
    result['split_sha256'] = hashlib.sha256(json.dumps(names).encode()).hexdigest()
    if conf.dataset == 'CAMELYON17':
        meta = pd.read_csv(Path(conf.dataset_csv_dir) / 'camelyon17.csv').set_index('slide_id')
        cases = [set(meta.loc[n, 'case_id']) for n in names]
        result['patient_overlap_train_val'] = sorted(cases[0] & cases[1])
        result['patient_overlap_train_test'] = sorted(cases[0] & cases[2])
    return result


def run_experiment(dataset, method='baseline', seed=42, epochs=None, output_root=None, gpu_cache=True):
    if method not in ('baseline','focal'): raise ValueError(method)
    conf = load_project_config(dataset, seed)
    if epochs is not None: conf.train_epoch = int(epochs)
    if conf.train_epoch <= 0: raise ValueError('epochs must be positive')
    out = Path(output_root or ROOT / 'results') / f'{dataset}_{method}_seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    # Do not silently reuse a stale scientific result after source/config changes.
    if (out / 'metrics.json').exists():
        raise FileExistsError(f'Finished run already exists: {out}. Use another output root to rerun.')
    seed_everything(seed)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_start = time.perf_counter()
    datasets = build_HDF5_feat_dataset(conf.data_dir, conf)
    conf.D_feat = infer_feature_dim(datasets[0])
    splits = audit_splits(conf, datasets)
    save_json(out / 'splits.json', splits)
    conf.gpu_feature_cache = bool(gpu_cache and device.type == 'cuda')
    cache_info = []
    if conf.gpu_feature_cache:
        torch.cuda.empty_cache()
        # Storage optimization only: every patch is retained at original source
        # precision, then converted to the same float32 model input as before.
        free_bytes, _ = torch.cuda.mem_get_info(device)
        budget = int(free_bytes * 0.55)
        for split, ds in zip(['train', 'val'], datasets[:2]):
            info = ds.warm_cache(device=device, max_bytes=budget)
            budget -= info['cache_bytes']
            cache_info.append(dict(split=split, **info))
            print(f'{dataset}: GPU storage cache {split}: {info}', flush=True)
    save_json(out / 'feature_cache.json', cache_info)
    save_json(out / 'config.json', vars(conf))
    loaders = make_loaders(conf, datasets)
    model = ABMIL(conf).to(device)
    criterion = nn.CrossEntropyLoss() if method == 'baseline' else FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(conf.lr), weight_decay=float(conf.wd))
    best_score, best_epoch, best_state = -float('inf'), -1, None
    history, start_run = [], time.perf_counter()
    setup_seconds = start_run - setup_start
    for epoch in range(int(conf.train_epoch)):
        start = time.perf_counter()
        train = train_one_epoch(model, criterion, loaders[0], optimizer, device)
        val = evaluate(model, criterion, loaders[1], device, conf)
        if not math.isfinite(val['auc']):
            raise ValueError('Validation macro AUC undefined: revise a class-deficient split before training')
        score = val['auc'] + val['f1']
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({'state_dict':best_state, 'config':vars(conf), 'method':method,
                        'epoch':best_epoch, 'val_score':best_score}, out / 'best.pt')
        row = dict(epoch=epoch+1, train_loss=train['loss'], train_acc=train['acc'],
                   **{f'val_{k}':v for k,v in val.items()}, seconds=time.perf_counter()-start)
        history.append(row)
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        print(f'{dataset} {method} {epoch+1:02}/{conf.train_epoch} '
              f'train={train["loss"]:.4f} val_AUC={val["auc"]:.4f} '
              f'F1={val["f1"]:.4f} best={best_epoch} {row["seconds"]:.1f}s', flush=True)
    model.load_state_dict(best_state)
    # The test set is first evaluated here, after all checkpoint selection.
    metrics, predictions = evaluate(model, criterion, loaders[2], device, conf, True)
    frame = pd.DataFrame({'slide_id':predictions['slide_id'], 'label':predictions['labels']})
    for c in range(conf.n_class): frame[f'p{c}'] = predictions['probabilities'][:,c]
    frame.to_csv(out / 'test_predictions.csv', index=False)
    versions = {p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scikit-learn','h5py']}
    source_hashes = {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                     for f in ['models.py','training.py','utils.py','dataloader.py','run_experiments.py']}
    result = dict(dataset=dataset, method=method, seed=seed, best_epoch=best_epoch,
                  validation_score=best_score, test=metrics, config=vars(conf),
                  split_sha256=splits['split_sha256'], versions=versions,
                  source_sha256=source_hashes, parameters=sum(p.numel() for p in model.parameters()),
                  device=str(device), gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                  python=platform.python_version(), total_seconds=time.perf_counter()-start_run,
                  setup_seconds=setup_seconds, feature_cache=cache_info,
                  objective='cross_entropy' if method=='baseline' else 'focal_gamma2_no_alpha',
                  status='complete', training_policy=f'constant_lr_{conf.train_epoch}epochs_no_subsampling')
    save_json(out / 'metrics.json', result)
    print(json.dumps(metrics), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=list(CONFIGS), default=list(CONFIGS))
    p.add_argument('--methods', nargs='+', choices=['baseline','focal'], default=['baseline','focal'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int)
    p.add_argument('--output-root', type=Path)
    p.add_argument('--disable-gpu-cache', action='store_true')
    args = p.parse_args()
    for ds in args.datasets:
        for method in args.methods:
            run_experiment(ds, method, args.seed, args.epochs, args.output_root,
                           gpu_cache=not args.disable_gpu_cache)


if __name__ == '__main__': main()
