import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import json
import os

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

# ── Load Data from JSON ──────────────────────────────────────────────────
json_path = "kapitel/data.json"
with open(json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Combine relevant models for the plot
plot_items = []

# 1. Add Top Open Source (On-Premise)
for m in raw_data["top5_open_source_under_48gb_vram_int4"]:
    label = f"{m['model']}\n({m['total_params_b']}B, {m['vram_int4_gb']}GB)"
    plot_items.append((label, m["arena_score"], m["vram_int4_gb"], True))

# 2. Add Disqualified (too big) for comparison
for m in raw_data["disqualified_open_source"]:
    # Shorten name if too long
    short_name = m["model"].replace("DeepSeek", "DS")
    label = f"{short_name}\n({m['vram_int4_gb']}GB)"
    plot_items.append((label, 1398 if "R1" in m["model"] else 1423, m["vram_int4_gb"], False)) # Scores from script/json context

# Sort by Elo
plot_items.sort(key=lambda x: x[1])

labels = [d[0] for d in plot_items]
scores = [d[1] for d in plot_items]
edge   = [d[3] for d in plot_items]

EDGE_COLOR   = "#2c3e50"     # dark blue-grey
SERVER_COLOR = "#95a5a6"     # lighter grey

colors = [EDGE_COLOR if e else SERVER_COLOR for e in edge]

# ── Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(labels))
bars = ax.bar(x, scores, color=colors, edgecolor="white", width=0.65)

# Value labels on top of each bar
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            str(score), ha="center", va="bottom", fontsize=9.5, fontweight="bold")

# Reference lines from closed source data
closed_top = raw_data["top5_closed_source"][0] # Claude Opus Thinking (1502)
closed_baseline = 1300 # GPT-4 level

ax.axhline(y=1400, color="#c0392b", linestyle="--", linewidth=0.9, alpha=0.6)
ax.text(len(labels) - 0.3, 1403, "GPT-4.5 / Claude 3.5 Niveau (~1400)", ha="right",
        fontsize=9, color="#c0392b", fontstyle="italic")

ax.axhline(y=closed_top["arena_score"], color="#c0392b", linestyle="-", linewidth=1.2, alpha=0.8)
ax.text(len(labels) - 0.3, closed_top["arena_score"] + 3, f"State of the Art: {closed_top['model']} ({closed_top['arena_score']})", ha="right",
        fontsize=9, color="#c0392b", fontweight="bold")

# Axes
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Chatbot Arena Elo Score")
ax.set_ylim(1150, 1550)
ax.set_xlim(-0.6, len(labels) - 0.4)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=EDGE_COLOR,   label="On-Premise tauglich (≤ 48 GB VRAM)"),
    Patch(facecolor=SERVER_COLOR, label="Server-Klasse (> 48 GB VRAM)"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=10,
          framealpha=0.9, edgecolor="#cccccc")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

# Create directory if not exists
os.makedirs("abbildungen", exist_ok=True)

fig.savefig("abbildungen/llm_onpremise_leaderboard.pdf", bbox_inches="tight")
fig.savefig("abbildungen/llm_onpremise_leaderboard.jpg", dpi=200, bbox_inches="tight")
print("Plots saved to abbildungen/llm_onpremise_leaderboard.{pdf,jpg}")
