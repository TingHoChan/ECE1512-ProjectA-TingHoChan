"""Inspect real feature cohorts and preserve traceable dataset statistics."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import h5py
import numpy as np
import pandas as pd
from dataloader import build_HDF5_feat_dataset
from run_experiments import ROOT, CONFIGS, load_project_config, audit_splits, save_json


def audit(dataset):
    conf=load_project_config(dataset)
    datasets=build_HDF5_feat_dataset(conf.data_dir,conf)
    splits=audit_splits(conf,datasets)
    stats=[]
    with h5py.File(conf.data_dir,'r') as h5:
        for name in sorted(h5):
            g=h5[name]
            shape=g['feat'].shape
            if len(shape)!=2 or min(shape)<=0: raise ValueError(f'Bad features: {name}')
            if 'coords' in g and g['coords'].shape != (shape[0],2):
                raise ValueError(f'Coordinate shape mismatch: {name}')
            stats.append(dict(slide_id=name, patches=shape[0], feature_dim=shape[1],
                              raw_label=int(g.attrs['label']),coords='coords' in g))
    frame=pd.DataFrame(stats)
    if frame.feature_dim.nunique()!=1: raise ValueError('Feature dimensions differ between slides')
    if dataset!='C16':
        csv=ROOT/'dataset_csv'/('camelyon17.csv' if dataset=='C17' else 'bracs.csv')
        meta=pd.read_csv(csv).set_index('slide_id')
        for s in stats:
            if s['raw_label']!=int(meta.loc[s['slide_id'],'label']):
                raise ValueError('HDF5/metadata label mismatch')
    out=ROOT/'output/data_audit'
    out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/f'{dataset}_slides.csv',index=False)
    summary=dict(dataset=dataset,n_slides=len(frame),feature_dim=int(frame.feature_dim.iloc[0]),
                 total_patches=int(frame.patches.sum()),patch_counts=frame.patches.describe().to_dict(),
                 raw_class_counts=frame.raw_label.value_counts().sort_index().to_dict(),
                 all_slides_have_coords=bool(frame.coords.all()),splits=splits)
    save_json(out/f'{dataset}_audit.json',summary)
    print(dataset,len(frame),'slides',summary['total_patches'],'patches',flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--datasets',nargs='+',default=list(CONFIGS),choices=list(CONFIGS))
    for ds in p.parse_args().datasets: audit(ds)
