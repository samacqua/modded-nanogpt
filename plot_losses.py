import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = [
    # ("records/track_1_short/2026-04-08_PairedHeadMuon/logs", "Single group (old)", "baseline*-1480.txt"),
    # ("records/track_1_short/2026-04-08_PairedHeadMuon/logs", "Paired QK\nhead groups (old)", "split_qk*-1480.txt"),
    ("logs_og", "Paired QK\nhead groups", "*.txt"),
    ("logs1", "Paired on paired\nlayers only", "*.txt"),
    ("logs2", "Group of 4\n+ group of 2", "*.txt"),
    ("logs3", "Paired on paired\nlayers only,\nper head on rest", "*.txt"),
    ("logs4", "Paired QKV\nhead groups", "*.txt"),
    ("logs5", "Groups of 3", "*.txt"),
    ("logs6", "Misaligned Muon\nvs attn pairs", "*.txt"),
    ("logs7", "Single head\ngroups", "*.txt"),
    ("logs8", "Single group\n(pre-PR code)", "*.txt"),
    ("logs9", "Misaligned Muon\n+ paired QKV", "*.txt"),
]

root = Path(__file__).parent
args = argparse.ArgumentParser()
args.add_argument("--step", type=int, help="Plot val loss at this step instead of final val loss.")
args = args.parse_args()

if args.step is None:
    pattern = re.compile(r"step:(\d+)/\1 val_loss:([\d.]+)")
    loss_group = 2
    ylabel = "Final val loss"
    title = "Muon head grouping ablation"
    output_path = root / "losses.png"
else:
    pattern = re.compile(rf"step:{args.step}/\d+ val_loss:([\d.]+)")
    loss_group = 1
    ylabel = f"Val loss (step {args.step})"
    title = f"Muon head grouping ablation (val loss at step {args.step})"
    output_path = root / f"losses_step{args.step}.png"


def extract_values(lines):
    values = []
    for line in lines:
        m = pattern.search(line)
        if m:
            values.append(float(m.group(loss_group)))
    return values

results = {}
for dirname, label, glob_pat in CONDITIONS:
    d = root / dirname
    if not d.exists():
        continue
    for f in sorted(d.glob(glob_pat)):
        losses = extract_values(f.read_text().splitlines())
        if losses:
            results.setdefault(label, []).extend(losses)

if not results:
    if args.step is None:
        raise SystemExit("No final val_loss lines found.")
    raise SystemExit(f"No val_loss found for step {args.step}.")

fig, ax = plt.subplots(figsize=(9, 5))
x_pos = np.arange(len(results))
colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
rng = np.random.default_rng(42)

for i, (label, losses) in enumerate(results.items()):
    mean = np.mean(losses)
    xs = np.full(len(losses), i) + rng.uniform(-0.12, 0.12, len(losses))
    ax.scatter(xs, losses, color=colors[i], zorder=3, s=60, edgecolors="k", linewidths=0.5)
    ax.hlines(mean, i - 0.25, i + 0.25, color=colors[i], linewidth=2.5, zorder=2)
    ax.text(i, mean + 0.00015, f"{mean:.4f}\n(n={len(losses)})", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x_pos)
ax.set_xticklabels(list(results.keys()), fontsize=8)
ax.set_ylabel(ylabel)
ax.set_title(title)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(output_path, dpi=150)
print(f"Saved to {output_path}")

for label, losses in results.items():
    print(f"  {label:45s}  n={len(losses)}  mean={np.mean(losses):.4f}  vals={losses}")
