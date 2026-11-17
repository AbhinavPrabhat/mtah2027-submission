"""A shallow bidirectional GRU over the player features: numpy inference, torch training.

Inference (`predict`) needs numpy only. Training (`train`) needs torch and is run once by the
train_*.py scripts. The network reads a 48 s window of per-frame features at 5 steps per second:

    Linear(d, 32) -> LayerNorm -> GELU -> bidirectional GRU(32) -> Linear(64, 1)    (about 14k-16k weights)

Practices in `train`: dropout and weight decay, input noise, gradient clipping, a validation block
for early stopping and learning-rate drops, blocked cross-fitting on 2-minute blocks with 10 s gaps
(out-of-fold scores for the training match), three seeds averaged. Labels are only the algorithm-made
ones passed in; -1 means unlabelled.
"""
import json
import os
import sys

import numpy as np

STEP = 5                      # rows per step: 5 steps per second
WINDOW, STRIDE = 240, 120     # steps: 48 s
GAP = 50                      # steps: 10 s around a test block
HIDDEN, DROP, NOISE = 32, 0.3, 0.1
EPOCHS, BATCHES, BATCH, PATIENCE = 40, 30, 32, 6
SEEDS = (0, 1, 2)
KEYS = ("inp0_w", "inp0_b", "ln_w", "ln_b", "ih_f", "hh_f", "bi_f", "bh_f", "ih_r", "hh_r", "bi_r", "bh_r", "out_w", "out_b")


# ----------------------------------------------------------------------------- numpy inference

def _gelu(x):
    from math import sqrt
    erf = np.vectorize(__import__("math").erf, otypes=[float])
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _gru_dir(x, ih, hh, bi, bh, reverse):
    """x: (batch, steps, in) -> (batch, steps, hidden). PyTorch gate order r, z, n."""
    b, t, _ = x.shape
    hsz = hh.shape[1]
    out = np.zeros((b, t, hsz))
    h = np.zeros((b, hsz))
    gi = x @ ih.T + bi
    order = range(t - 1, -1, -1) if reverse else range(t)
    for s in order:
        gh = h @ hh.T + bh
        r = _sigmoid(gi[:, s, :hsz] + gh[:, :hsz])
        z = _sigmoid(gi[:, s, hsz:2 * hsz] + gh[:, hsz:2 * hsz])
        n = np.tanh(gi[:, s, 2 * hsz:] + r * gh[:, 2 * hsz:])
        h = (1.0 - z) * n + z * h
        out[:, s] = h
    return out


def _forward(w, x):
    """Logits for a batch of windows x: (batch, steps, d)."""
    h = x @ w["inp0_w"].T + w["inp0_b"]
    mu, var = h.mean(-1, keepdims=True), h.var(-1, keepdims=True)
    h = (h - mu) / np.sqrt(var + 1e-5) * w["ln_w"] + w["ln_b"]
    h = _gelu(h)
    fwd = _gru_dir(h, w["ih_f"], w["hh_f"], w["bi_f"], w["bh_f"], False)
    bwd = _gru_dir(h, w["ih_r"], w["hh_r"], w["bi_r"], w["bh_r"], True)
    return (np.concatenate([fwd, bwd], -1) @ w["out_w"].T + w["out_b"])[..., 0]


def _step_logits(w, xs):
    """Logits for one sequence xs: (steps, d), sliding windows averaged with triangular weights."""
    n = len(xs)
    if n <= WINDOW:
        return _forward(w, xs[None])[0]
    starts = list(range(0, n - WINDOW + 1, STRIDE))
    if starts[-1] != n - WINDOW:
        starts.append(n - WINDOW)
    logits = _forward(w, np.stack([xs[a:a + WINDOW] for a in starts]))
    tri = np.minimum(np.arange(1, WINDOW + 1), np.arange(WINDOW, 0, -1)).astype(float)
    acc, wt = np.zeros(n), np.zeros(n)
    for a, lg in zip(starts, logits):
        acc[a:a + WINDOW] += lg * tri
        wt[a:a + WINDOW] += tri
    return acc / wt


def rows_from_steps(step_values, rows, n):
    """Spread values given every STEP-th row back over all n rows of one half."""
    pos = np.searchsorted(rows, np.arange(rows[0], rows[-1] + 1), side="right") - 1
    out = np.full(n, step_values[-1], float)
    out[:rows[-1] + 1 - rows[0]] = step_values[np.clip(pos, 0, len(rows) - 1)]
    return out


def predict(models, x, half):
    """Mean logit over the stored seeds, per row of x. models: list of weight dicts."""
    out = np.zeros(len(x))
    for h in np.unique(half):
        idx = np.flatnonzero(half == h)
        rows = np.arange(len(idx))[::STEP]
        xs = x[idx][rows].astype(np.float64)
        logit = np.mean([_step_logits(w, xs) for w in models], axis=0)
        out[idx] = rows_from_steps(logit, rows, len(idx))
    return out


def save(path, models):
    np.savez_compressed(path, **{"s%d_%s" % (i, k): w[k] for i, w in enumerate(models) for k in KEYS})


def load(path):
    z = np.load(path)
    seeds = sorted({int(k.split("_")[0][1:]) for k in z.files})
    return [{k: z["s%d_%s" % (i, k)].astype(np.float64) for k in KEYS} for i in seeds]


# ----------------------------------------------------------------------------- torch training

def _extract(net):
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    return {"inp0_w": sd["inp.0.weight"], "inp0_b": sd["inp.0.bias"], "ln_w": sd["inp.1.weight"], "ln_b": sd["inp.1.bias"],
            "ih_f": sd["gru.weight_ih_l0"], "hh_f": sd["gru.weight_hh_l0"], "bi_f": sd["gru.bias_ih_l0"], "bh_f": sd["gru.bias_hh_l0"],
            "ih_r": sd["gru.weight_ih_l0_reverse"], "hh_r": sd["gru.weight_hh_l0_reverse"],
            "bi_r": sd["gru.bias_ih_l0_reverse"], "bh_r": sd["gru.bias_hh_l0_reverse"],
            "out_w": sd["out.1.weight"], "out_b": sd["out.1.bias"]}


def _train(x, y, frames, half, weighted, log=print):
    """(out-of-fold logits per row, [weights per seed fitted on all blocks], report).

    y is 1 / 0 for labelled rows and -1 for the rest. `weighted` balances the classes (dead ball).
    """
    import time
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    torch.set_num_threads(4)

    class Net(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(d, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU(), nn.Dropout(DROP))
            self.gru = nn.GRU(HIDDEN, HIDDEN, batch_first=True, bidirectional=True)
            self.out = nn.Sequential(nn.Dropout(DROP), nn.Linear(2 * HIDDEN, 1))

        def forward(self, z):
            return self.out(self.gru(self.inp(z))[0]).squeeze(-1)

    x = x.astype(np.float32)
    seqs = []
    for h in np.unique(half):
        rows = np.flatnonzero(half == h)[::STEP]
        seqs.append(dict(x=x[rows], y=y[rows], blk=frames[rows] // 3000, rows=rows))
    d = x.shape[1]

    def predict_net(net, which):
        net.eval()
        with torch.no_grad():
            return [_step_logits_torch(net, s["x"]) for s in which]

    def _step_logits_torch(net, xs):
        n = len(xs)
        starts = [0] if n <= WINDOW else list(range(0, n - WINDOW + 1, STRIDE))
        if n > WINDOW and starts[-1] != n - WINDOW:
            starts.append(n - WINDOW)
        w = min(n, WINDOW)
        logits = net(torch.tensor(np.stack([xs[a:a + w] for a in starts]))).numpy()
        tri = np.minimum(np.arange(1, w + 1), np.arange(w, 0, -1)).astype(float)
        acc, wt = np.zeros(n), np.zeros(n)
        for a, lg in zip(starts, logits):
            acc[a:a + w] += lg * tri
            wt[a:a + w] += tri
        return acc / wt

    def val_loss(net, pw):
        tot, cnt = 0.0, 0.0
        for s, p in zip(seqs, predict_net(net, seqs)):
            m = s["val"]
            if m.sum():
                bce = nn.functional.binary_cross_entropy_with_logits(
                    torch.tensor(p), torch.tensor(s["y"]).float(), reduction="none", pos_weight=torch.tensor(pw))
                tot += float((bce * torch.tensor(m).float()).sum())
                cnt += float(m.sum())
        return tot / max(cnt, 1.0)

    def fit(pw, seed):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = Net(d)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
        lens = np.array([len(s["x"]) for s in seqs], float)
        best, state, bad, tloss = 1e9, None, 0, 0.0
        for _ in range(EPOCHS):
            net.train()
            tl = 0.0
            for _ in range(BATCHES):
                xb, yb, mb = [], [], []
                for _ in range(BATCH):
                    s = seqs[rng.choice(len(seqs), p=lens / lens.sum())]
                    a = rng.integers(0, len(s["x"]) - WINDOW)
                    xb.append(s["x"][a:a + WINDOW]); yb.append(s["y"][a:a + WINDOW]); mb.append(s["m"][a:a + WINDOW])
                xt = torch.tensor(np.stack(xb))
                xt = xt + NOISE * torch.randn_like(xt)
                yt, mt = torch.tensor(np.stack(yb)).float(), torch.tensor(np.stack(mb)).float()
                opt.zero_grad()
                bce = nn.functional.binary_cross_entropy_with_logits(net(xt), yt, reduction="none", pos_weight=torch.tensor(pw))
                loss = (bce * mt).sum() / mt.sum().clamp(min=1.0)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tl += loss.item() / BATCHES
            v = val_loss(net, pw)
            sched.step(v)
            if v < best - 1e-4:
                best, bad, tloss = v, 0, tl
                state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        net.load_state_dict(state)
        return net, tloss, best

    def set_masks(test_blocks, val_blocks):
        for s in seqs:
            test = np.isin(s["blk"], list(test_blocks))
            near = np.convolve(test.astype(int), np.ones(2 * GAP + 1, int), "same") > 0
            val = np.isin(s["blk"], list(val_blocks)) & ~near
            lab = s["y"] >= 0
            s["m"] = (lab & ~near & ~val).astype(np.float32)
            s["val"] = (lab & val).astype(np.float32)
            s["test"] = test
        npos = sum(((s["y"] == 1) & (s["m"] > 0)).sum() for s in seqs)
        nneg = sum(((s["y"] == 0) & (s["m"] > 0)).sum() for s in seqs)
        return float(np.clip(nneg / max(npos, 1), 0.3, 10)) if weighted else 1.0

    steps_blk = np.concatenate([s["blk"] for s in seqs])
    oof = [np.full(len(s["x"]), np.nan) for s in seqs]
    gaps, t0 = [], time.time()
    for fold, (_, te) in enumerate(GroupKFold(5).split(np.zeros(len(steps_blk)), groups=steps_blk)):
        test_blocks = set(np.unique(steps_blk[te]))
        tr_blocks = [b for b in np.unique(steps_blk) if b not in test_blocks]
        pw = set_masks(test_blocks, set(b for i, b in enumerate(tr_blocks) if i % 6 == 3))
        acc = [np.zeros(len(s["x"])) for s in seqs]
        for seed in SEEDS:
            net, tl, vl = fit(pw, 100 * fold + seed)
            gaps.append((tl, vl))
            for k, p in enumerate(predict_net(net, seqs)):
                acc[k] += p / len(SEEDS)
        for k, s in enumerate(seqs):
            oof[k][s["test"]] = acc[k][s["test"]]
        log("  fold %d done, %.0f s" % (fold, time.time() - t0))
    logit_rows = np.zeros(len(frames))
    for s, o in zip(seqs, oof):
        idx = np.flatnonzero(half == half[s["rows"][0]])
        logit_rows[idx] = rows_from_steps(o, s["rows"] - idx[0], len(idx))
    labelled = (y >= 0) & np.isfinite(logit_rows)
    auc = float(roc_auc_score(y[labelled], logit_rows[labelled]))

    pw = set_masks(set(), set(b for i, b in enumerate(np.unique(steps_blk)) if i % 6 == 3))
    finals = []
    for seed in SEEDS:
        net, _, _ = fit(pw, 1000 + seed)
        finals.append(_extract(net))
    report = dict(oof_auc=round(auc, 3), train_loss=round(float(np.mean([g[0] for g in gaps])), 3),
                  val_loss=round(float(np.mean([g[1] for g in gaps])), 3),
                  parameters=int(sum(p.numel() for p in Net(d).parameters())))
    return logit_rows, finals, report


def train(x, y, frames, half, weighted, log=print):
    """Run `_train` in a child process: torch and LightGBM each bundle an OpenMP runtime that deadlock
    when both are loaded in one process on macOS, and the parent has LightGBM loaded."""
    import subprocess
    import sys
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")
        np.savez(src, x=x, y=y, frames=frames, half=half, weighted=np.array(weighted))
        subprocess.run([sys.executable, os.path.abspath(__file__), src, dst], check=True)
        z = np.load(dst, allow_pickle=True)
        finals = [{k: z["s%d_%s" % (i, k)] for k in KEYS} for i in range(len(SEEDS))]
        return z["logit_rows"], finals, json.loads(str(z["report"]))


if __name__ == "__main__":
    z = np.load(sys.argv[1])
    logit_rows, finals, report = _train(z["x"], z["y"], z["frames"], z["half"], bool(z["weighted"]),
                                        log=lambda m: print(m, flush=True))
    np.savez(sys.argv[2], logit_rows=logit_rows, report=json.dumps(report),
             **{"s%d_%s" % (i, k): w[k] for i, w in enumerate(finals) for k in KEYS})
