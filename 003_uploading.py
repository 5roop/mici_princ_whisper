import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    WhisperTokenizerFast,
    WhisperForConditionalGeneration,
)
from pathlib import Path

model_id = "output/checkpoint-618"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
try:
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
except:
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=False,
    )

model.push_to_hub("5roop/whisper-large-v3-mici-princ")


from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from datasets import load_dataset, DatasetDict, Dataset, Audio
import pandas as pd
from pathlib import Path


to_remove = [
    # "!",
    # ",",
    # ".",
    # ":",
    # "?",
    "–",
    "“",
    "„",
    "•",
    # "…",
]
to_substitute = {
    "\n": " ",
    "ȅ": "e",
    "é": "e",
    "ȋ": "i",
    "ȉ": "i",
    "ȃ": "a",
    "ȁ": "a",
    "î": "i",
}


def normalize(s: str) -> str:
    for c in to_remove:
        s = s.replace(c, "")
    for c, c_ in to_substitute.items():
        s = s.replace(c, c_)
        s = s.replace(c.upper(), c_.upper())
    s = s.replace("  ", " ")
    return s


jsons = list(Path("data").glob("**/*.asr.json"))

for j in jsons:
    import json

    data = json.loads(j.read_text())
    new_data = [
        {**entry, "normalized_text": normalize(entry["text"])} for entry in data
    ]
    Path(j).write_text(json.dumps(new_data, ensure_ascii=False, indent=4))


test_path = Path("data/test")
test_jsons = test_path.glob("*.asr.json")
test_df = pd.concat([pd.read_json(i) for i in test_jsons])
test_df["audio"] = test_df.audio.apply(lambda s: str(test_path / s))
# test_df["text"] = test_df.normalized_text

train_path = Path("data/train")
train_jsons = train_path.glob("*.asr.json")
train_df = pd.concat([pd.read_json(i) for i in train_jsons])
train_df["audio"] = train_df.audio.apply(lambda s: str(train_path / s))
train_df["speaker"] = ""
# train_df["text"] = train_df.normalized_text

assert train_df.audio.apply(lambda s: Path(s).exists()).all()
assert test_df.audio.apply(lambda s: Path(s).exists()).all()
ds = DatasetDict()
ds["test"] = Dataset.from_pandas(test_df)

ds["train"] = Dataset.from_pandas(train_df)

ds = (
    ds.cast_column("audio", Audio(sampling_rate=16000, mono=True))
    .remove_columns("__index_level_0__")
    .select_columns(["audio", "text", "normalized_text", "speaker"])
)
# ds.push_to_hub("classla/Mici_Princ")
