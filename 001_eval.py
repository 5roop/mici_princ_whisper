from datasets import load_dataset, DatasetDict, Dataset, Audio
from transformers.pipelines.pt_utils import KeyDataset
import pandas as pd
from pathlib import Path
from os import environ

test_path = Path("data/test")
test_jsons = test_path.glob("*.asr.json")
test_df = pd.concat([pd.read_json(i) for i in test_jsons])
test_df["audio"] = test_df.audio.apply(lambda s: str(test_path / s))
test_df["text"] = test_df.normalized_text
ds = Dataset.from_pandas(test_df)
ds = ds.cast_column("audio", Audio(sampling_rate=16000, mono=True))


checkpoints = list(Path("output").glob("checkpoint*")) + ["openai/whisper-large-v3"]


def get_transcripts(model_id):
    print(model_id)
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    from pathlib import Path

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # model_id = "openai/whisper-large-v3"
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
    except:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=False,
        )
    model.to(device)
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(
        KeyDataset(ds, "audio"),
        generate_kwargs={"language": "croatian"},
    )
    transcripts = [i.get("text").replace("\n", "") for i in result]
    return transcripts


from tqdm import tqdm


def key(c):
    if "openai" in str(c):
        return 0
    else:
        return int(str(c).split("-")[-1])


for checkpoint in tqdm(sorted(checkpoints, key=key)):
    print(f"Evaluating {checkpoint}...")
    transcription = get_transcripts(checkpoint)
    test_df[str(checkpoint).replace("/", "_")] = transcription
test_df.to_csv("transcriptions.csv", index=False)
