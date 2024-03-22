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
    rs = " ".join(df["text"].apply(normalize).tolist())
    ps = " ".join(df[checkpoint].apply(normalize).tolist())
    cdif = len(ps) - len(rs)
    wdif = len(ps.split()) - len(rs.split())
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
        rs = " ".join(subset["text"].apply(normalize).tolist())
        ps = " ".join(subset[checkpoint].apply(normalize).tolist())
        per_speaker[speaker] = {
            "cer": cers,
            "wer": wers,
            "cdif": len(ps) - len(rs),
            "wdif": len(ps.split()) - len(rs.split()),
        }
    metrics[checkpoint] = {
        "cer": cer,
        "wer": wer,
        "cdif": cdif,
        "wdif": wdif,
        "per_speaker": per_speaker,
    }

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
fig, [ax1, ax2, ax3, ax4] = plt.subplots(ncols=4, figsize=(18, 5))
ax1.plot(df.epoch, df.wer, label="Overall", linewidth=3, zorder=2)
ax2.plot(df.epoch, df.cer, label="Overall", linewidth=3, zorder=2)
ax3.plot(df.epoch, df.cdif, label="Overall", linewidth=3, zorder=2)
ax4.plot(df.epoch, df.wdif, label="Overall", linewidth=3, zorder=2)
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
    ax3.plot(
        df.epoch,
        df.per_speaker.apply(lambda d: d[speaker]["cdif"]),
        label=speaker,
        linestyle="--",
        marker="o",
    )
    ax4.plot(
        df.epoch,
        df.per_speaker.apply(lambda d: d[speaker]["wdif"]),
        label=speaker,
        linestyle="--",
        marker="o",
    )
ax1.set_title("WER")
ax2.set_title("CER")
ax3.set_title("Char len diff")
ax4.set_title("Word len diff")
# ax1.set_xticks([i for i in range(11)])
# ax2.set_xticks([i for i in range(11)])

ax1.set_xlabel("Epoch")
ax2.set_xlabel("Epoch")
ax3.set_xlabel("Epoch")
ax4.set_xlabel("Epoch")
ax1.set_ylim((0, 0.5))
ax2.set_ylim((0, 0.5))
ax3.set_ylim((0, 1000))
ax4.set_ylim((0, 100))
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
# ax3.set_yscale("log")
# ax4.set_yscale("log")
fig.tight_layout()

plt.savefig("metrics.png")
plt.savefig("metrics.pdf")
print("")
