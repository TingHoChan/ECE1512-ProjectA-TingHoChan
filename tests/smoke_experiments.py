"""Synthetic end-to-end runner check; these are NOT pathology results.

    python tests/smoke_experiments.py
Produces only tests/.tmp/smoke_results, never the scientific results folder.
"""
from pathlib import Path
import json
import sys
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_experiments as runner


def main():
    print('SYNTHETIC ENGINEERING CHECK ONLY: artificial features, not pathology benchmark performance.', flush=True)
    temp = ROOT / 'tests/.tmp/smoke'
    temp.mkdir(parents=True, exist_ok=True)
    original = runner.load_project_config
    rng = np.random.default_rng(7)
    for ds in runner.CONFIGS:
        conf = original(ds)
        if ds=='C16':
            rows=[(f'normal_{i:03}',i%2) for i in range(80)]
            rows += [(f'test_{i:03}',i%2) for i in range(20)]
        else:
            meta=pd.read_csv(ROOT/'dataset_csv'/('camelyon17.csv' if ds=='C17' else 'bracs.csv'))
            rows=list(zip(meta.slide_id,meta.label))
        path=temp/f'{ds}_SYNTHETIC.h5'
        with h5py.File(path,'w') as h5:
            for name,label in rows:
                group=h5.create_group(name)
                group.attrs['label']=label
                n=int(rng.integers(3,10))
                x=rng.normal(0,.1,(n,8)).astype('float32')
                x[:,label]+=2  # Artificial learnable signal, not histopathology.
                group.create_dataset('feat',data=x)
                group.create_dataset('coords',data=np.column_stack([np.arange(n)*512,np.zeros(n)]))
        def load_synthetic(dataset,seed=42):
            c=original(dataset,seed)
            c.data_dir=str(temp/f'{dataset}_SYNTHETIC.h5')
            return c
        runner.load_project_config=load_synthetic
        for method in ('baseline','focal'):
            out=temp/'smoke_results'
            metric_path=out/f'{ds}_{method}_seed42'/'metrics.json'
            if metric_path.exists():
                result=json.loads(metric_path.read_text())
            else:
                result=runner.run_experiment(ds,method,epochs=2,output_root=out)
            assert result['status']=='complete' and result['config']['train_epoch']==2
        a=json.loads((out/f'{ds}_baseline_seed42/splits.json').read_text())
        b=json.loads((out/f'{ds}_focal_seed42/splits.json').read_text())
        assert a==b
    print('SYNTHETIC six-run pipeline check passed; no scientific result was generated.')


if __name__=='__main__': main()
