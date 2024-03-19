import torch
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines.pt_utils import KeyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "classla/whisper-large-v3-mici-princ"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
)

model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

ds = load_dataset("classla/Mici_Princ", split="test")
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    device=device,
)

result = pipe(
    KeyDataset(ds, "audio"),
    generate_kwargs={"language": "croatian"},
)

for i in result:
    print(i)

# Output:
# {'text': ' Šesti planet je biv deset put veći. Na njin je bivav niki stari čovik ki je pisav vele knjige.', 'chunks': [{'timestamp': (0.0, 7.18), 'text': ' Šesti planet je biv deset put veći. Na njin je bivav niki stari čovik ki je pisav vele knjige.'}]}
# ...
