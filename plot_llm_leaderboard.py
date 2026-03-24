import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from matplotlib.patches import Patch

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

os.makedirs("abbildungen", exist_ok=True)

# ── Colour Palette ───────────────────────────────────────────────────────
EDGE_COLOR   = "#2c3e50"     # dark blue-grey  (on-premise / highlighted)
SERVER_COLOR = "#95a5a6"     # lighter grey     (server-class)

# ── Closed-source benchmark ─────────────────────────────────────────────
closed_top = raw_data["top5_closed_source"][0]  # claude-opus-4-6-thinking (1502)


# =========================================================================
# Figure 1.1 – Top 5 Open-Source (overall, full params + VRAM)
# =========================================================================
models_full = raw_data["top5_overall_open_weights"]

plot1_items = []
for m in models_full:
    label = f"{m['model']}\n({m['total_params_b']}B, {m['vram_int4_gb']}GB)"
    plot1_items.append((label, m["arena_score"]))

# Sort by Elo
plot1_items.sort(key=lambda x: x[1])

labels1 = [d[0] for d in plot1_items]
scores1 = [d[1] for d in plot1_items]

fig1, ax1 = plt.subplots(figsize=(14, 7))
x1 = np.arange(len(labels1))
bars1 = ax1.bar(x1, scores1, color=SERVER_COLOR, edgecolor="white", width=0.65)

for bar, score in zip(bars1, scores1):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             str(score), ha="center", va="bottom", fontsize=9.5, fontweight="bold")

# Benchmark: Claude Opus 4.6 Thinking
ax1.axhline(y=closed_top["arena_score"], color="#c0392b", linestyle="-", linewidth=1.2, alpha=0.8)
ax1.text(len(labels1) - 0.3, closed_top["arena_score"] + 3,
         f"Bestes Closed-Source: {closed_top['model']} ({closed_top['arena_score']})",
         ha="right", fontsize=9, color="#c0392b", fontweight="bold")

ax1.set_xticks(x1)
ax1.set_xticklabels(labels1, rotation=45, ha="right", fontsize=8.5)
ax1.set_ylabel("Chatbot Arena Elo Score")
ax1.set_ylim(1150, 1550)
ax1.set_xlim(-0.6, len(labels1) - 0.4)

legend1 = [
    Patch(facecolor=SERVER_COLOR, label="Server-Klasse (> 48 GB VRAM)"),
]
ax1.legend(handles=legend1, loc="upper left", fontsize=10,
           framealpha=0.9, edgecolor="#cccccc")

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
fig1.tight_layout()

fig1.savefig("abbildungen/llm_leaderboard_overall.pdf", bbox_inches="tight")
fig1.savefig("abbildungen/llm_leaderboard_overall.jpg", dpi=200, bbox_inches="tight")
print("Figure 1.1 saved to abbildungen/llm_leaderboard_overall.{pdf,jpg}")


# =========================================================================
# Figure 1.2 – Top 5 Open-Source under 48 GB VRAM (INT4)
# =========================================================================
models_int4 = raw_data["top5_open_source_under_48gb_vram_int4"]

plot2_items = []
for m in models_int4:
    label = f"{m['model']}\n({m['total_params_b']}B, {m['vram_int4_gb']}GB INT4)"
    plot2_items.append((label, m["arena_score"]))

# Sort by Elo
plot2_items.sort(key=lambda x: x[1])

labels2 = [d[0] for d in plot2_items]
scores2 = [d[1] for d in plot2_items]

fig2, ax2 = plt.subplots(figsize=(14, 7))
x2 = np.arange(len(labels2))
bars2 = ax2.bar(x2, scores2, color=EDGE_COLOR, edgecolor="white", width=0.65)

for bar, score in zip(bars2, scores2):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             str(score), ha="center", va="bottom", fontsize=9.5, fontweight="bold")

# Benchmark: Claude Opus 4.6 Thinking
ax2.axhline(y=closed_top["arena_score"], color="#c0392b", linestyle="-", linewidth=1.2, alpha=0.8)
ax2.text(len(labels2) - 0.3, closed_top["arena_score"] + 3,
         f"Bestes Closed-Source: {closed_top['model']} ({closed_top['arena_score']})",
         ha="right", fontsize=9, color="#c0392b", fontweight="bold")

ax2.set_xticks(x2)
ax2.set_xticklabels(labels2, rotation=45, ha="right", fontsize=8.5)
ax2.set_ylabel("Chatbot Arena Elo Score")
ax2.set_ylim(1150, 1550)
ax2.set_xlim(-0.6, len(labels2) - 0.4)

legend2 = [
    Patch(facecolor=EDGE_COLOR, label="On-Premise tauglich (≤ 48 GB VRAM, INT4)"),
]
ax2.legend(handles=legend2, loc="upper left", fontsize=10,
           framealpha=0.9, edgecolor="#cccccc")

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
fig2.tight_layout()

fig2.savefig("abbildungen/llm_onpremise_leaderboard.pdf", bbox_inches="tight")
fig2.savefig("abbildungen/llm_onpremise_leaderboard.jpg", dpi=200, bbox_inches="tight")
print("Figure 1.2 saved to abbildungen/llm_onpremise_leaderboard.{pdf,jpg}")
