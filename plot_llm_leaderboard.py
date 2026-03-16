"""
Generates a simple bar chart showing Chatbot Arena Elo scores of
self-hostable open-weight LLMs, sorted by score.

Style matches the existing 'vramllm.jpg' figure:
  - simple vertical bars, value labels on top, clean look
  - dashed horizontal reference lines for context

Data source: https://onyx.app/self-hosted-llm-leaderboard  (Stand: März 2026)
"""

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

# ── Selected models (with Chatbot Arena Elo), sorted ascending ───────────
data = [
    # (label,                            Arena Elo,  VRAM INT4 GB, edge?)
    ("Llama 3.1-8B\n(8B, INT4: 5 GB)",      1212,   5,   True),
    ("Phi-4\n(14B, INT4: 9 GB)",             1256,   9,   True),
    ("Mistral Small 3.1\n(24B, INT4: 14 GB)",1304,  14,   True),
    ("Qwen 2.5-72B\n(72B, INT4: 37 GB)",    1303,  37,   True),
    ("Llama 3.3 70B\n(70B, INT4: 38 GB)",   1319,  38,   True),
    ("Gemma 3 12B\n(12B, INT4: 8 GB)",       1342,   8,   True),
    ("Gemma 3 27B\n(27B, INT4: 14 GB)",      1366,  14,   True),
    ("Qwen3-30B-A3B\n(30B, INT4: 16 GB)",   1384,  16,   True),
    ("DeepSeek R1\n(671B, INT4: 351 GB)",    1398, 351,  False),
    ("Qwen3-235B\n(235B, INT4: 120 GB)",     1423, 120,  False),
    ("DeepSeek V3.2\n(685B, INT4: 351 GB)",  1423, 351,  False),
    ("GLM-4.7\n(355B, INT4: 180 GB)",        1441, 180,  False),
    ("Kimi K2.5\n(1T, INT4: 542 GB)",        1438, 542,  False),
    ("Qwen 3.5\n(397B, INT4: 207 GB)",       1450, 207,  False),
]

# Sort by Elo
data.sort(key=lambda x: x[1])

labels = [d[0] for d in data]
scores = [d[1] for d in data]
edge   = [d[3] for d in data]

EDGE_COLOR   = "#2c3e50"     # dark blue-grey  (matches vramllm.jpg bar colour)
SERVER_COLOR = "#95a5a6"     # lighter grey for server-class models

colors = [EDGE_COLOR if e else SERVER_COLOR for e in edge]

# ── Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(labels))
bars = ax.bar(x, scores, color=colors, edgecolor="white", width=0.65)

# Value labels on top of each bar
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            str(score), ha="center", va="bottom", fontsize=9.5, fontweight="bold")

# Reference lines
ax.axhline(y=1300, color="#c0392b", linestyle="--", linewidth=0.9, alpha=0.6)
ax.text(len(labels) - 0.3, 1303, "GPT-4o Niveau (~1300)", ha="right",
        fontsize=9, color="#c0392b", fontstyle="italic")

ax.axhline(y=1400, color="#c0392b", linestyle="--", linewidth=0.9, alpha=0.6)
ax.text(len(labels) - 0.3, 1403, "GPT-4.5 Niveau (~1400)", ha="right",
        fontsize=9, color="#c0392b", fontstyle="italic")

# Axes
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Chatbot Arena Elo Score")
ax.set_ylim(1150, 1500)
ax.set_xlim(-0.6, len(labels) - 0.4)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=EDGE_COLOR,   label="On-Premise tauglich (≤ 48 GB VRAM)"),
    Patch(facecolor=SERVER_COLOR, label="Server-Klasse (> 48 GB VRAM)"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
          framealpha=0.9, edgecolor="#cccccc")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("abbildungen/llm_onpremise_leaderboard.pdf", bbox_inches="tight")
fig.savefig("abbildungen/llm_onpremise_leaderboard.jpg", dpi=200, bbox_inches="tight")
print("Plots saved to abbildungen/llm_onpremise_leaderboard.{pdf,jpg}")
