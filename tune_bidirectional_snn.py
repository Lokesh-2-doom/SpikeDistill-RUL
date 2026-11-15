"""Validation-only experiment: bidirectional recurrent LIF student."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from train_spike_models import DATA, DEVICE, RecurrentLIF, regression_metrics, seed_all

ROOT = Path(__file__).resolve().parent
TEACHER = ROOT / "results" / "bilstm_teacher_tuned"
OUT = ROOT / "results" / "spike_models" / "bidirectional_snn"
RECIPES = {
    "FD003": {"hidden_per_direction": 96, "lr": .0003, "alpha": .1, "leak": .8},
    "FD004": {"hidden_per_direction": 96, "lr": .0015, "alpha": .2, "leak": .8},
}
BATCH, MAX_EPOCHS, PATIENCE = 128, 55, 10


class BidirectionalLIFStudent(nn.Module):
    def __init__(self, input_size: int, hidden_per_direction: int, leak_init: float):
        super().__init__()
        self.forward_encoder = RecurrentLIF(input_size, hidden_per_direction, leak_init)
        self.backward_encoder = RecurrentLIF(input_size, hidden_per_direction, leak_init)
        width = 2 * hidden_per_direction
        self.temporal_attention = nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, 1, bias=False))
        self.readout = nn.Linear(width, 1)
        self.teacher_projection = nn.Linear(192, width, bias=False)

    def forward(self, x: torch.Tensor):
        mf, sf, tf = self.forward_encoder(x)
        mr, sr, tr = self.backward_encoder(torch.flip(x, dims=[1]))
        mb, sb = torch.flip(mr, dims=[1]), torch.flip(sr, dims=[1])
        membrane = torch.cat((mf, mb), dim=-1)
        spikes = torch.cat((sf, sb), dim=-1)
        threshold = torch.cat((tf, tr), dim=0)
        attention = torch.softmax(self.temporal_attention(membrane).squeeze(-1), dim=1)
        context = torch.sum(membrane * attention.unsqueeze(-1), dim=1)
        pred = self.readout(context).squeeze(-1)
        return pred, membrane, attention, spikes, threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=list(RECIPES), action="append")
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for tag in args.subset or list(RECIPES):
        cfg, seed = RECIPES[tag], 42
        seed_all(seed)
        z = np.load(DATA / f"windows_{tag}.npz")
        x, y = z["X_train"].astype(np.float32), z["y_train"].astype(np.float32)
        xv, yv = z["X_val"].astype(np.float32), z["y_val"].astype(np.float32)
        ty = np.load(TEACHER / f"teacher_signals_{tag}_train.npz")["y_teacher"].astype(np.float32)
        model = BidirectionalLIFStudent(x.shape[2], cfg["hidden_per_direction"], cfg["leak"]).to(DEVICE)
        optimizer = torch.optim.AdamW([p for n,p in model.named_parameters() if n!="teacher_projection.weight"],
                                      lr=cfg["lr"], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=5, min_lr=1e-5)
        sw=np.where(y<30,1.4,np.where(y<80,1.,.85)).astype(np.float64)
        sampler=WeightedRandomSampler(torch.as_tensor(sw),len(y),replacement=True,generator=torch.Generator().manual_seed(seed))
        loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(ty)),batch_size=BATCH,sampler=sampler)
        xvt=torch.from_numpy(xv).to(DEVICE)
        best,state,best_pred,best_epoch,stale=float('inf'),None,None,0,0
        history=[]
        for epoch in range(1,MAX_EPOCHS+1):
            model.train()
            for xb,yb,tb in loader:
                xb,yb,tb=xb.to(DEVICE),yb.to(DEVICE),tb.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                pred=model(xb)[0];loss=(1-cfg['alpha'])*nn.functional.mse_loss(pred,yb)+cfg['alpha']*nn.functional.mse_loss(pred,tb)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            model.eval()
            with torch.no_grad():pred=model(xvt)[0].cpu().numpy().astype(np.float32)
            metrics=regression_metrics(yv,pred);scheduler.step(metrics['MSE']);history.append({'epoch':epoch,**metrics})
            if metrics['RMSE']<best-1e-4:
                best,state,best_pred,best_epoch,stale=metrics['RMSE'],{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},pred.copy(),epoch,0
            else:
                stale+=1
                if stale>=PATIENCE:break
            if epoch==1 or epoch%5==0:print(f"{tag} biLIF epoch={epoch} val_RMSE={metrics['RMSE']:.3f} best={best:.3f}@{best_epoch}",flush=True)
        name=f"{tag}_bidirectional_lif_seed42"
        torch.save({'state_dict':state,'input_size':x.shape[2],'hidden_per_direction':cfg['hidden_per_direction'],
                    'config':cfg,'best_epoch':best_epoch},OUT/f'{name}.pt')
        np.savez_compressed(OUT/f'{name}_validation.npz',y_true=yv,y_pred=best_pred)
        (OUT/f'{name}_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        basefile=ROOT/'results'/'spike_models'/'rmse_refinement'/tag/'output_mse_h192_seed42_validation.npz'
        base=np.load(basefile)
        summary[tag]={'config':cfg,'best_epoch':best_epoch,'bidirectional_validation':regression_metrics(yv,best_pred),
                      'linear_snn_validation':regression_metrics(base['y_true'],base['y_pred'])}
        print(f"{tag}: bidirectional SNN RMSE={best:.3f}@{best_epoch}; linear baseline={summary[tag]['linear_snn_validation']['RMSE']:.3f}",flush=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':main()
