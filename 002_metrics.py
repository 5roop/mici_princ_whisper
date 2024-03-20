import pandas as pd
import evaluate

df = pd.read_csv("transcriptions.csv")
print(df.shape)
df["text"] = df.normalized_text
del df["normalized_text"]

# print(sorted(set(" ".join(df.text.tolist()))))
punctuation = ["!", ",", ".", ":", "?", "–", "“", "„", "•", "…"]


def normalize(s: str) -> str:
    for c in punctuation:
        s = s.replace(c, "")
    return s.casefold()


checkpoints = [
    i for i in df.columns if i not in "text,audio,speaker,normalized_text".split(",")
]

metrics = dict()


def key(c):
    if "openai" in c:
        return 0
    else:
        return int(c.split("-")[-1])


for checkpoint in sorted(checkpoints, key=key):
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
    lambda s: 0 if "openai" in s else int(s.split("-")[-1]) / (277 / 16)
)
df = df.sort_values("epoch")
import matplotlib.pyplot as plt


colordict = {"Autor": "k", "Geograf": "red", "Mići Princ": "blue", "Dilavac": "green"}
fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(10, 5))
ax1.plot(df.epoch, df.wer, label="Overall", linewidth=3, zorder=10)
ax2.plot(df.epoch, df.cer, label="Overall", linewidth=3, zorder=10)
for speaker, color in colordict.items():
    ax1.plot(
        df.epoch,
        df.per_speaker.apply(lambda d: d[speaker]["wer"]),
        label=speaker,
        linestyle="--",
        marker="o",
    )
    ax2.plot(
        df.epoch,
        df.per_speaker.apply(lambda d: d[speaker]["cer"]),
        label=speaker,
        linestyle="--",
        marker="o",
    )
ax1.set_title("WER")
ax2.set_title("CER")
# ax1.set_xticks([i for i in range(11)])
# ax2.set_xticks([i for i in range(11)])

ax1.set_xlabel("Epoch")
ax2.set_xlabel("Epoch")
ax1.set_ylim((0, None))
ax2.set_ylim((0, None))
ax1.legend()
ax2.legend()
fig.tight_layout()

plt.savefig("metrics.png")
plt.savefig("metrics.pdf")
print("")
