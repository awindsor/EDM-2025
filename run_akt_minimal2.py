# run_akt_minimal2.py — AKT trainer/evaluator (writes payload.json)
# Author: Philip Pavlik – 29 Apr 2025

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from akt import AKT, TransformerLayer   # AKT architecture and layer class
from sklearn.metrics import roc_auc_score

# ─────────────────────────────── CLI ─────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs",   type=int,   default=30,
                    help="Number of epochs to train")
parser.add_argument("--data_dir", type=str,   default="testdir",
                    help="Directory where train/test CSVs are located")
parser.add_argument("--d_model",  type=int,   default=256,
                    help="Embedding dimension (d_model)")
parser.add_argument("--n_heads",  type=int,   default=8,
                    help="Number of multi-head attention heads")
parser.add_argument("--d_ff",     type=int,   default=1024,
                    help="Feedforward network dimension")
parser.add_argument("--n_blocks", type=int,   default=2,
                    help="Number of transformer blocks")
parser.add_argument("--dropout",  type=float, default=0.05,
                    help="Dropout rate")
parser.add_argument("--lr",       type=float, default=1e-3,
                    help="Learning rate for optimizer")
parser.add_argument("--batch_u",  type=int,   default=24,
                    help="Gradient-accumulation size (users)")
args = parser.parse_args()
print("Parsed arguments:", args, file=sys.stderr)

# ─────────────────────────────── I/O ─────────────────────────────────────────
cols = ["uid", "qid", "pid", "correct", "timestamp"]
train_df = (
    pd.read_csv(Path(args.data_dir) / "akt_train.csv", header=None, names=cols)
      .sort_values(["uid", "timestamp"])
      .reset_index(drop=True)
)
test_df = (
    pd.read_csv(Path(args.data_dir) / "akt_test.csv", header=None, names=cols)
      .sort_values(["uid", "timestamp"])
      .reset_index(drop=True)
)
def truncate_sequences(df, maxlen=200):
    truncated_rows = []
    for _, user_df in df.groupby('uid', sort=False):
        n = len(user_df)
        if n <= maxlen:
            truncated_rows.append(user_df)
        else:
            for i in range(0, n, maxlen):
                truncated_rows.append(user_df.iloc[i:i + maxlen])
    return pd.concat(truncated_rows).reset_index(drop=True)
train_df = truncate_sequences(train_df, maxlen=200)
test_df  = truncate_sequences(test_df,  maxlen=200)


print(f"Loaded {len(train_df)} train rows, {len(test_df)} test rows", file=sys.stderr)


# ─────────────────────── Model hyper-parameters ─────────────────────────────
num_q = int(max(train_df.qid.max(), test_df.qid.max())) + 1
num_p = int(max(train_df.pid.max(), test_df.pid.max())) + 1
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"num_q={num_q}, num_p={num_p}, device={device}", file=sys.stderr)

# ─────────────────────────── Determine whether to use PID branch ───────────
# If pid and qid are perfectly matching in every row, use standard AKT (no PID) to avoid redundancy
pid_qid_equal_train = (train_df['pid'] == train_df['qid']).all()
pid_qid_equal_test  = (test_df['pid']  == test_df['qid']).all()

if pid_qid_equal_train and pid_qid_equal_test:
    print("PID and QID are identical across all rows; switching to standard AKT (no PID).", file=sys.stderr)
    use_pid = False
else:
    print("Using AKT-PID variant with skill embeddings.", file=sys.stderr)
    use_pid = True

# ───────────────────── Initialize AKT with appropriate variant ─────────────────
model = AKT(
    n_question=2 * num_q,
    n_pid     =(num_p if use_pid else 0),
    d_model   =args.d_model,
    n_blocks  =args.n_blocks,
    n_heads   =args.n_heads,
    d_ff      =args.d_ff,
    kq_same   =1,
    dropout   =args.dropout,
    model_type=('akt_pid' if use_pid else 'akt')
).to(device)

# ────────── Patch inner Architecture for PID variant only ───────────────────
arch = model.model
if use_pid:
    # define transformer blocks for pid variant
    for name, count in [('blocks_1', args.n_blocks), ('blocks_2', args.n_blocks * 2)]:
        setattr(arch, name, nn.ModuleList([
            TransformerLayer(
                d_model   =args.d_model,
                d_feature =args.d_model // args.n_heads,
                d_ff      =args.d_ff,
                n_heads   =args.n_heads,
                dropout   =args.dropout,
                kq_same   =1
            ) for _ in range(count)
        ]))

print(f"Model initialized ({'PID' if use_pid else 'standard'}) with {sum(p.numel() for p in model.parameters()):,} parameters", file=sys.stderr)
optim = torch.optim.Adam(model.parameters(), lr=args.lr)

# ───────────────────────── Sequence construction ─────────────────────────────
def make_seq(df):
    q  = df.qid.values.astype(int)
    p  = df.pid.values.astype(int) if use_pid else np.zeros_like(q)
    a  = df.correct.values.astype(int)
    qa = q + a * num_q
    tg = np.concatenate(([-1], a[:-1]))
    return (
        torch.tensor(q,  dtype=torch.long,  device=device).unsqueeze(0),
        torch.tensor(qa, dtype=torch.long,  device=device).unsqueeze(0),
        torch.tensor(tg, dtype=torch.float, device=device).unsqueeze(0),
        torch.tensor(p,  dtype=torch.long,  device=device).unsqueeze(0),
    )

train_seqs = [make_seq(g) for _, g in train_df.groupby('uid', sort=False)]
test_seqs  = [make_seq(g) for _, g in test_df.groupby('uid',  sort=False)]

# ───────────────────────────── Training ─────────────────────────────────────
metrics = {'loss': [], 'drop': []}  # Removed 'nodrop' part
for epoch in range(1, args.epochs + 1):
    model.train()
    running_loss, all_t, all_p = [], [], []
    optim.zero_grad()

    for i, (q, qa, tg, pid) in enumerate(train_seqs, 1):
        loss, pred, _ = model(q, qa, tg, pid)
        (loss / args.batch_u).backward()
        running_loss.append(loss.item())
        mask = tg.view(-1) >= 0
        all_t.extend(  tg.view(-1)[mask].cpu().numpy())
        all_p.extend(pred.view(-1)[mask].detach().cpu().numpy())

        if i % args.batch_u == 0 or i == len(train_seqs):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            optim.zero_grad()

    drop_auc = roc_auc_score(all_t, all_p)
    metrics['loss'].append(float(np.mean(running_loss)))
    metrics['drop'].append(drop_auc)

    print(
        f"epoch {epoch}/{args.epochs} | loss {metrics['loss'][-1]:.4f} | "
        f"AUC(drop) {metrics['drop'][-1]:.4f}",
        file=sys.stderr
    )

# ───────────────────── Held-out test evaluation (only at the end) ────────────
model.eval()
test_t, test_p = [], []
with torch.no_grad():
    for q, qa, tg, pid in test_seqs:
        _, pred, _ = model(q, qa, tg, pid)
        m = tg.view(-1) >= 0
        test_t.extend(tg.view(-1)[m].cpu().numpy())
        test_p.extend(pred.view(-1)[m].cpu().numpy())
held_auc = roc_auc_score(test_t, test_p)
print(f"held-out AUC {held_auc:.4f}", file=sys.stderr)

# ─────────────────────────── Write payload ─────────────────────────────────
Path(args.data_dir).mkdir(parents=True, exist_ok=True)
payload = {
    'epoch_auc_drop':    metrics['drop'],
    'train_loss':        metrics['loss'],
    'held_auc':          float(held_auc),
    'test_pred':         [float(x) for x in test_p],
    'test_true':         [int(x)   for x in test_t]
}
with open(Path(args.data_dir) / 'payload.json', 'w') as f:
    json.dump(payload, f)
print("✓ wrote payload.json", file=sys.stderr)
sys.exit(0)
