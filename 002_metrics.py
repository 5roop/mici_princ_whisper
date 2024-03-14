import pandas as pd
import evaluate

df = pd.read_csv("transcriptions.csv")
print(df.shape)
print(sorted(set(" ".join(df.text.tolist()))))
punctuation = ["\n", "!", ",", ".", ":", "?", "–", "“", "„", "•", "…"]


def normalize(s: str) -> str:
    return "".join([i for i in s if i not in punctuation])


checkpoints = [i for i in df.columns if i not in "text,audio,speaker".split(",")]
candidates = [ 'ȃ', 'ȅ', 'ȋ']
for c in checkpoints:
    text = " ".join(df[c].tolist())
    for i in candidates:
        if i in text:
            print(f"Found {i} in {c}, {text.count(i)} times")
metrics = dict()
for checkpoint in checkpoints:
    metric_w = evaluate.load("wer")
    wer = metric_w.compute(
        predictions=df[checkpoint].apply(normalize).tolist(),
        references=df["text"].apply(normalize).tolist(),
    )
    metric_c = evaluate.load("cer")
    cer = metric_c.compute(
        predictions=df[checkpoint].apply(normalize).tolist(),
        references=df["text"].apply(normalize).tolist(),
    )
    per_speaker = {}
    for speaker in df.speaker.unique():
        subset = df[df.speaker == speaker]
        wers = metric_w.compute(
            predictions=subset[checkpoint].apply(normalize).tolist(),
            references=subset["text"].apply(normalize).tolist(),
        )
        cers = metric_c.compute(
            predictions=subset[checkpoint].apply(normalize).tolist(),
            references=subset["text"].apply(normalize).tolist(),
        )
        per_speaker[speaker] = {
            "cer": cers,
            "wer": wers,
        }
    metrics[checkpoint] = {"cer": cer, "wer": wer, "per_speaker": per_speaker}

import json
from pathlib import Path

Path("metrics.json").write_text(json.dumps(metrics, indent=4, ensure_ascii=False))
import pandas as pd

df = pd.DataFrame(metrics).T
df["checkpoint"] = df.index
df["epoch"] = df.checkpoint.apply(
    lambda s: 0 if "openai" in s else int(s.split("-")[-1]) / 309
)
df = df.sort_values("epoch")
import matplotlib.pyplot as plt

colordict = {"Autor": "k", "Geograf": "red", "Mići Princ": "blue", "Dilavac": "green"}
fig, [ax1, ax2] = plt.subplots(ncols=2)
ax1.plot(df.epoch, df.wer)
ax2.plot(df.epoch, df.cer)
for speaker, color in colordict.items():
    ax1.scatter(
        df.epoch, df.per_speaker.apply(lambda d: d[speaker]["wer"]), label=speaker
    )
    ax2.scatter(
        df.epoch, df.per_speaker.apply(lambda d: d[speaker]["cer"]), label=speaker
    )
ax1.set_title("Wer")
ax2.set_title("Cer")
ax1.legend()
ax2.legend()
fig.tight_layout()

plt.savefig("metrics.png")
print("")