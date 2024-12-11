from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    model_name: str = "facebook/mbart-large-cc25"
    max_length: int = 512
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    source_lang: str = "en_XX"
    target_lang: str = "sq_AL"  # Albanian
    device: str = "cuda"

@dataclass
class DataConfig:
    dataset_name: str = "abisee/cnn_dailymail"
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    cache_dir: str = "./cache"
