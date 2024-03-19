import pandas as pd
import evaluate
import difflib

df = pd.read_csv("transcriptions.csv")
df["text"] = df.normalized_text
del df["normalized_text"]
d = difflib.Differ()
punctuation = ["!", ",", ".", ":", "?", "–", "“", "„", "•", "…"]


def normalize(s: str) -> str:
    for c in punctuation:
        s = s.replace(c, "")
    return s.casefold()


checkpoints = [i for i in df.columns if i not in "text,audio,speaker".split(",")]
from colorama import Fore, Back, Style

for checkpoint in checkpoints:
    original = df.text
    predicted = df[checkpoint]
    speakers = df.speaker
    for o, p, s in zip(original, predicted, speakers):
        diff = d.compare(normalize(o).split(), normalize(p).split())
        output = "\n".join(diff)
        if "+" in output or "-" in output:
            print(
                # Style.BRIGHT
                # + Fore.RED +
                f"**** Evaluating checkpoint {checkpoint} ****"
                # + Fore.RESET
            )
            print(
                # Fore.RED +
                f"Speaker: {s}"
                # + Fore.RESET
            )
            print(
                # Fore.RED +
                f"Original sentence: \n{o}"
                # + Fore.RESET
            )
            print(
                # Fore.RED +
                f"Predicted sentence:\n{p}"
                # + Fore.RESET
            )
            print(
                # Fore.RED +
                "Diff other than punctuation differences:"
                # + Fore.RESET
                # + Back.RESET
                # + Style.RESET_ALL
            )
            print(output, end="\n\n\n")
            2 + 2
