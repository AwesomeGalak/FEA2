#!/usr/bin/env python3
"""
Split the multi-panel benchmark PNGs into individual graph images
for use in the LaTeX thesis.
"""
from PIL import Image
import os

ABBILDUNGEN = "/home/roboadmin/FEA2/FEA2/abbildungen"
os.makedirs(ABBILDUNGEN, exist_ok=True)


def crop_grid(src_path, rows, cols, names, title_crop_px=0):
    """Crop a grid image into individual panels."""
    img = Image.open(src_path)
    w, h = img.size

    # Crop off the main title area at the top
    if title_crop_px > 0:
        img = img.crop((0, title_crop_px, w, h))
        h = h - title_crop_px

    panel_w = w // cols
    panel_h = h // rows

    for idx, name in enumerate(names):
        row = idx // cols
        col = idx % cols
        left = col * panel_w
        top = row * panel_h
        right = left + panel_w
        bottom = top + panel_h
        panel = img.crop((left, top, right, bottom))
        out_path = os.path.join(ABBILDUNGEN, f"{name}.png")
        panel.save(out_path, dpi=(150, 150))
        print(f"  ✅ {name}.png ({panel.size[0]}x{panel.size[1]})")


# === LLM Benchmark (3x2 grid with title) ===
print("📊 Splitting LLM benchmark (benchmark_results.png)...")
llm_panels = [
    "llm_generierungsgeschwindigkeit",  # top-left
    "llm_antwortzeit",                  # top-right
    "llm_gpu_vram",                     # mid-left
    "llm_cpu_ram",                      # mid-right
    "llm_gpu_leistung",                 # bot-left
    "llm_netzwerk_io",                  # bot-right
]
crop_grid(
    "/home/roboadmin/FEA2/GLaDOS/benchmark_results.png",
    rows=3, cols=2, names=llm_panels, title_crop_px=60
)

# === ASR Benchmark (2x2 grid with title) ===
print("\n🎤 Splitting ASR benchmark (asr_benchmark_results.png)...")
asr_panels = [
    "asr_wer_pro_satz",        # top-left
    "asr_fehlerverteilung",    # top-right
    "asr_wer_vs_cer",          # bot-left
    "asr_zusammenfassung",     # bot-right
]
crop_grid(
    "/home/roboadmin/FEA2/GLaDOS/benchmark/asr_benchmark_results.png",
    rows=2, cols=2, names=asr_panels, title_crop_px=50
)

print("\n✅ Alle Grafiken gespeichert in:", ABBILDUNGEN)
