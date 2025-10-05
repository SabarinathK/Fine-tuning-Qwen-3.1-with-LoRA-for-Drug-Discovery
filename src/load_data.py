from datasets import load_dataset
from transformers import AutoTokenizer
from load_model import model_name

raw_data = load_dataset("json", data_files="drug_target_discovery_200.json")


raw_data["train"][0]
print(raw_data)

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

def preprocess(sample):
    sample = sample["prompt"] + "\n" + sample["completion"]

    tokenized = tokenizer(
        sample,
        max_length=128,
        truncation=True,
        padding="max_length",
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

data = raw_data.map(preprocess)

print(data["train"][0])
