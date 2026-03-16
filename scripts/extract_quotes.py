import os
import glob
import subprocess
import re

pdf_files = glob.glob(r'c:\Users\morit\OneDrive\Desktop\Masterarbeit_2025\stuff\*.pdf')
keywords = ["latency", "on-premise", "local", "edge computing", "privacy", "security", "robotics", "llm"]

results = []
for f in pdf_files:
    try:
        text = subprocess.check_output(['pdftotext', '-f', '1', '-l', '3', f, '-'], stderr=subprocess.DEVNULL).decode('utf-8', 'ignore')
        text = text.replace('\r\n', ' ').replace('\n', ' ')
        sentences = re.split(r'(?<=[.!?]) +', text)
        
        found = []
        for s in sentences:
            if any(k.lower() in s.lower() for k in keywords) and len(s) > 40 and len(s) < 300:
                found.append(s.strip())
        
        if found:
            results.append(f"--- {os.path.basename(f)} ---\n" + "\n".join(found[:5]) + "\n")
    except Exception as e:
        continue

with open(r'c:\Users\morit\OneDrive\Desktop\Masterarbeit_2025\FEA2_Latex\extracted_quotes.txt', 'w', encoding='utf-8') as out:
    out.write("\n\n".join(results))
