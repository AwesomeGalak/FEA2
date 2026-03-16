"""
LLaMA 7B Performance-Simulation auf RTX 3080 Ti (12 GB VRAM)
System: Ubuntu 24.04, 16 GB RAM, llama.cpp / ollama
Style: OTH Regensburg Corporate Design
Ausgabe: Einzelne JPG-Dateien für jeden Plot
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ──────────────────────────────────────────────────────────────
# Farbpalette & Styling (OTH Regensburg CD)
# ──────────────────────────────────────────────────────────────
# Primärfarben & Neutrale Töne:
BG       = "#ffffff"       # Weißer Hintergrund (oder #f6f6f6 für leichten Kontrast)
CARD     = "#ffffff"
GRID     = "#d8d8d8"       # Neutral-Color-300
TEXT     = "#191919"       # Primary-Color-900 (Fast Schwarz)
TEXT_MUTED = "#707070"     # Neutral-Color-700

# Akzentfarben:
OTH_RED  = "#e3000b"       # Klassisches OTH Rot (oder #da532c)
OTH_BLUE = "#334155"       # Slate Blue / Anthrazit
OTH_LIGHT= "#94a3b8"       # Helles Blaugrau

plt.rcParams.update({
    "figure.facecolor":   BG,
    "axes.facecolor":     CARD,
    "axes.edgecolor":     TEXT_MUTED,
    "axes.labelcolor":    TEXT,
    "axes.titlecolor":    TEXT,
    "axes.grid":          True,
    "axes.grid.axis":     "y",     # Meistens nur horizontale Gitternetzlinien im CD
    "grid.color":         GRID,
    "grid.alpha":         0.7,
    "grid.linestyle":     "--",
    "text.color":         TEXT,
    "xtick.color":        TEXT_MUTED,
    "ytick.color":        TEXT_MUTED,
    "font.family":        "sans-serif",
    "font.size":          12,
    "axes.titlesize":     14,
    "axes.labelsize":     12,
    "legend.facecolor":   BG,
    "legend.edgecolor":   GRID,
    "figure.dpi":         300,
})

np.random.seed(42)

# ──────────────────────────────────────────────────────────────
# Daten-Simulation
# ──────────────────────────────────────────────────────────────
quant_labels  = ["FP16\n(Offload)", "INT8", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
vram_gb       = [14.0, 7.0, 5.2, 4.2, 3.4, 2.1]               # GB
tok_per_s     = [8.5, 28.3, 42.6, 52.1, 57.4, 62.0]            # Tokens/s
perplexity    = [5.68, 5.71, 5.79, 5.87, 6.12, 7.45]           # Wiki2 PPL

prompt_lens      = np.array([32, 64, 128, 256, 512, 1024, 2048, 4096])
ttft_ms          = np.array([45, 62, 98, 175, 340, 680, 1380, 2850])
ttft_noise       = ttft_ms + np.random.normal(0, ttft_ms * 0.04, len(ttft_ms))

n_tokens   = 256
cum_tokens = np.arange(1, n_tokens + 1)
itl_ms     = 19.2 + 3.5 * np.exp(-cum_tokens / 8) + np.random.normal(0, 0.7, n_tokens)
cum_time_s = np.cumsum(itl_ms) / 1000.0

t_vram = np.linspace(0, 12, 300)
vram_trace = np.piecewise(
    t_vram,
    [t_vram < 0.5, (t_vram >= 0.5) & (t_vram < 1.5), t_vram >= 1.5],
    [lambda t: 0.8 + t * 6.0,                           
     lambda t: 4.1 + 0.3 * (t - 0.5),                   
     lambda t: 4.4 + 0.15 * np.sin(0.8 * t) + np.random.normal(0, 0.05, len(t))]
)
vram_trace = np.clip(vram_trace, 0, 12)

# Hilfsfunktion für einheitliches Speichern
def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, format="jpg", dpi=300, facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"✓ Gespeichert: {filename}")

# ──────────────────────────────────────────────────────────────
# 1. Tokens/s vs Quantisierung
# ──────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))
bars = ax1.bar(quant_labels, tok_per_s, color=OTH_BLUE, width=0.6,
               edgecolor="none", zorder=3)
for bar, val in zip(bars, tok_per_s):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
             f"{val:.1f}", ha="center", va="bottom", fontsize=10, color=TEXT, fontweight='bold')
ax1.set_ylabel("Tokens / s")
ax1.set_title("Generierungsgeschwindigkeit in Abhängigkeit der Quantisierung", pad=15)
ax1.set_ylim(0, 75)
ax1.spines[['top', 'right']].set_visible(False)
save_plot(fig1, "llama7b_tokens_per_s.jpg")

# ──────────────────────────────────────────────────────────────
# 2. VRAM vs Quantisierung
# ──────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
# Farben: Überschrittenes VRAM in OTH Rot, ansonsten OTH Light Blue
bar_colors = [OTH_RED if v > 12 else OTH_LIGHT for v in vram_gb]
bars2 = ax2.bar(quant_labels, vram_gb, color=bar_colors, width=0.6,
                edgecolor="none", zorder=3)
ax2.axhline(12, color=OTH_RED, ls="--", lw=1.5, label="3080 Ti VRAM Limit (12 GB)")
for bar, val in zip(bars2, vram_gb):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{val:.1f}", ha="center", va="bottom", fontsize=10, color=TEXT, fontweight='bold')
ax2.set_ylabel("VRAM (GB)")
ax2.set_title("VRAM-Bedarf pro Quantisierungsstufe", pad=15)
ax2.set_ylim(0, 16)
ax2.spines[['top', 'right']].set_visible(False)
ax2.legend(loc="upper right")
save_plot(fig2, "llama7b_vram.jpg")

# ──────────────────────────────────────────────────────────────
# 3. Perplexity vs Quantisierung
# ──────────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(quant_labels, perplexity, "o-", color=OTH_BLUE, lw=2.5,
         markersize=8, markeredgecolor=BG, markeredgewidth=2, zorder=4)
for i, (lbl, ppl) in enumerate(zip(quant_labels, perplexity)):
    ax3.annotate(f"{ppl:.2f}", (i, ppl), textcoords="offset points",
                 xytext=(0, 12), ha="center", fontsize=10, color=TEXT, fontweight='bold')
ax3.set_ylabel("Perplexity (WikiText-2)")
ax3.set_title("Qualitätsverlust durch Quantisierung", pad=15)
ax3.set_ylim(5.0, 8.0)
ax3.spines[['top', 'right']].set_visible(False)
save_plot(fig3, "llama7b_perplexity.jpg")

# ──────────────────────────────────────────────────────────────
# 4. TTFT vs Prompt-Länge
# ──────────────────────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.plot(prompt_lens, ttft_noise, "s-", color=OTH_RED, lw=2.5,
         markersize=7, markeredgecolor=BG, markeredgewidth=1.5, zorder=4)
ax4.fill_between(prompt_lens, ttft_noise * 0.9, ttft_noise * 1.1,
                 color=OTH_RED, alpha=0.1)
ax4.set_xlabel("Prompt-Länge (Tokens)")
ax4.set_ylabel("Time-to-First-Token (ms)")
ax4.set_title("Prefill-Latenz (Q4_K_M)", pad=15)
ax4.set_xscale("log", base=2)
ax4.set_yscale("log")
ax4.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax4.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax4.spines[['top', 'right']].set_visible(False)
save_plot(fig4, "llama7b_ttft.jpg")

# ──────────────────────────────────────────────────────────────
# 5. Generierte Tokens über Zeit
# ──────────────────────────────────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(8, 5))
ax5.plot(cum_time_s, cum_tokens, color=OTH_BLUE, lw=2.5, zorder=4)
ax5.fill_between(cum_time_s, 0, cum_tokens, color=OTH_BLUE, alpha=0.1)
slope_label = f"Ø {n_tokens / cum_time_s[-1]:.1f} Tokens / s"
ax5.text(cum_time_s[-1] * 0.6, n_tokens * 0.4, slope_label,
         fontsize=12, color=OTH_BLUE, fontweight="bold",
         bbox=dict(boxstyle="square,pad=0.5", fc=BG, ec=OTH_BLUE, lw=1.5))
ax5.set_xlabel("Zeit (s)")
ax5.set_ylabel("Generierte Tokens")
ax5.set_title("Token-Ausgabe über Zeit (Q4_K_M)", pad=15)
ax5.set_xlim(0, max(cum_time_s)*1.05)
ax5.set_ylim(0, n_tokens*1.05)
ax5.spines[['top', 'right']].set_visible(False)
save_plot(fig5, "llama7b_token_gen.jpg")

# ──────────────────────────────────────────────────────────────
# 6. VRAM-Verlauf über Zeit
# ──────────────────────────────────────────────────────────────
fig6, ax6 = plt.subplots(figsize=(8, 5))
ax6.plot(t_vram, vram_trace, color=OTH_LIGHT, lw=2.5, zorder=4)
ax6.fill_between(t_vram, 0, vram_trace, color=OTH_LIGHT, alpha=0.2)
ax6.axhline(12, color=OTH_RED, ls="--", lw=1.5, label="VRAM-Limit (12 GB)")

# Annotate phases
ax6.annotate("Laden", xy=(0.25, 2.5), fontsize=10, color=TEXT,
             fontweight="bold", ha="center")
ax6.annotate("Prefill", xy=(1.0, 4.8), fontsize=10, color=TEXT,
             fontweight="bold", ha="center")
ax6.annotate("Generierung", xy=(7.0, 5.3), fontsize=10, color=TEXT,
             fontweight="bold", ha="center")

ax6.set_xlabel("Zeit (s)")
ax6.set_ylabel("VRAM-Nutzung (GB)")
ax6.set_title("VRAM-Verlauf während Inferenz (Q4_K_M)", pad=15)
ax6.set_xlim(0, 12)
ax6.set_ylim(0, 14)
ax6.spines[['top', 'right']].set_visible(False)
ax6.legend(loc="upper right")
save_plot(fig6, "llama7b_vram_trace.jpg")
