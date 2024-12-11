import datasets
from torch.utils.data import DataLoader, Dataset
from transformers import MBartTokenizer

class SummarizationDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = texts
        self.summaries = summaries
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        targets = self.tokenizer(
            summary,
            max_length=self.max_length // 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze(),
        }

def load_dataset(config):
    dataset = datasets.load_dataset(config.dataset_name)
    
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    return train_dataset, val_dataset, test_dataset