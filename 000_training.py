from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from datasets import load_dataset, DatasetDict, Dataset
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
test_df["text"] = test_df.normalized_text

train_path = Path("data/train")
train_jsons = train_path.glob("*.asr.json")
train_df = pd.concat([pd.read_json(i) for i in train_jsons])
train_df["audio"] = train_df.audio.apply(lambda s: str(train_path / s))
train_df["text"] = train_df.normalized_text

assert train_df.audio.apply(lambda s: Path(s).exists()).all()
assert test_df.audio.apply(lambda s: Path(s).exists()).all()
ds = DatasetDict()
ds["test"] = Dataset.from_pandas(test_df)
ds["train"] = Dataset.from_pandas(train_df)


feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-large-v3",
    language="croatian",
)

tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small", language="croatian", task="transcribe"
)

from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-large-v3", language="croatian", task="transcribe"
)

input_str = ds["test"][0]["text"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")
from datasets import Audio

ds = ds.cast_column("audio", Audio(sampling_rate=16000, mono=True))

print("")


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


ds = ds.map(
    prepare_dataset,
    # remove_columns=ds.column_names["test"],
    num_proc=4,
)

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


import evaluate

metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.generation_config.language = "croatian"


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./output",  # change to a repo name of your choice
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=309 * 10,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=309,
    eval_steps=309,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=False,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()

kwargs = {
    "dataset": "Mići Princ",  # a 'pretty' name for the training dataset
    "language": "hr",
    "model_name": "Whisper Mići Princ",  # a 'pretty' name for your model
    "finetuned_from": "openai/whisper-large-v3",
    "tasks": "automatic-speech-recognition",
}

trainer.push_to_hub(**kwargs)
