"""Regenerate figures/tables from measured prediction and training logs."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from run_experiments import ROOT, CONFIGS, load_project_config


def generate_figures():
    out = ROOT / 'output/figures'
    out.mkdir(parents=True, exist_ok=True)
    rows, class_rows = [], []
    # Large source labels stay legible when fitted into the Letter report.
    plt.rcParams.update({'font.size':14, 'figure.dpi':160, 'savefig.facecolor':'white'})
    for ds in CONFIGS:
        histories, predictions = {}, {}
        for method in ('baseline','focal'):
            folder = ROOT / 'results' / f'{ds}_{method}_seed42'
            if not (folder/'metrics.json').exists(): continue
            m = json.loads((folder/'metrics.json').read_text())
            rows.append({'dataset':ds, 'method':method, 'best_epoch':m['best_epoch'],
                         **m['test'], 'seconds':m['total_seconds']})
            histories[method] = pd.read_csv(folder/'history.csv')
            predictions[method] = pd.read_csv(folder/'test_predictions.csv')
            frame = predictions[method]
            columns = [f'p{i}' for i in range(m['config']['n_class'])]
            pred = frame[columns].values.argmax(1)
            details = classification_report(frame.label, pred, labels=list(range(len(columns))),
                                            output_dict=True, zero_division=0)
            for c in range(len(columns)):
                class_rows.append(dict(dataset=ds,method=method, class_index=c, **details[str(c)]))
        if not histories: continue
        fig, ax = plt.subplots(2,2,figsize=(9,6),layout='constrained')
        for method, h in histories.items():
            label = 'ABMIL + CE' if method=='baseline' else 'ABMIL + focal'
            ax[0,0].plot(h.epoch,h.train_loss,label=label)
            ax[0,1].plot(h.epoch,h.val_ce_loss,label=label)
            ax[1,0].plot(h.epoch,h.val_auc,label=label)
            ax[1,1].plot(h.epoch,h.val_f1,label=label)
        for a,title in zip(ax.flat,['Training objective (distinct scales)', 'Validation CE (common scale)',
                                   'Validation macro AUROC','Validation macro F1']):
            a.set_title(title,fontsize=14); a.set_xlabel('Epoch'); a.grid(alpha=.2); a.legend(fontsize=12)
        fig.suptitle(ds+' - full bags, seed 42; checkpoint selection uses validation only')
        fig.savefig(out/f'{ds}_comparison.png'); plt.close(fig)
        with plt.rc_context({'font.size':16}):
            fig, axes = plt.subplots(1,2,figsize=(9,4),layout='constrained')
        for ax,(method,frame) in zip(axes,predictions.items()):
            probs = frame.filter(regex=r'^p\d+$').values
            cm = confusion_matrix(frame.label,probs.argmax(1),labels=list(range(probs.shape[1])))
            ax.imshow(cm,cmap='Blues',vmin=0)
            for (i,j),v in np.ndenumerate(cm):
                ax.text(j,i,str(v),ha='center',va='center',fontsize=15,
                        color='white' if v>cm.max()*.5 else 'black')
            ax.set(title=f'{ds} {method}',xlabel='Predicted class',ylabel='True class',
                   xticks=range(len(cm)),yticks=range(len(cm)))
            ax.set_title(f'{ds} {method}', fontsize=17)
            ax.set_xlabel('Predicted class', fontsize=16)
            ax.set_ylabel('True class', fontsize=16)
            ax.tick_params(labelsize=16)
        fig.savefig(out/f'{ds}_confusion.png'); plt.close(fig)
    pd.DataFrame(rows).to_csv(ROOT/'output/results_summary.csv',index=False)
    pd.DataFrame(class_rows).to_csv(ROOT/'output/per_class_results.csv',index=False)
    if len(rows)==6:
        r = pd.DataFrame(rows).set_index(['dataset','method'])
        changes = []
        for ds in CONFIGS:
            a,b=r.loc[(ds,'baseline')],r.loc[(ds,'focal')]
            changes.append(dict(dataset=ds,delta_accuracy_pp=b.acc-a.acc,
                                delta_auc=b.auc-a.auc,delta_f1=b.f1-a.f1,
                                delta_common_ce=b.ce_loss-a.ce_loss))
        pd.DataFrame(changes).to_csv(ROOT/'output/paired_changes.csv',index=False)
    return rows


def generate_attention():
    import torch
    from models import ABMIL
    from wsi_visualization import visualize_wsi_attention
    folder = ROOT/'results/C16_baseline_seed42'
    checkpoint = torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    conf = load_project_config('C16')
    conf.D_feat = checkpoint['config']['D_feat']
    model = ABMIL(conf)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    visualize_wsi_attention(model,conf,ROOT,checkpoint['state_dict'],
                            output_dir=ROOT/'output/figures',show=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--attention',action='store_true')
    args=p.parse_args()
    print(generate_figures())
    if args.attention: generate_attention()
