"""
preprocess_data.py
──────────────────
Loads raw train CSV, normalizes text, builds label mappings,
tokenizes with the banking prompt, and returns a HuggingFace Dataset
ready for the Trainer.
"""

import re
import json
import pandas as pd
from datasets import Dataset


BANKING_PROMPT = """Here is a banking query:
{input_text}

Classify this query into one of the following intents:

{categories_list}

SOLUTION
The correct answer is: class """


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"[''´`']", "'", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def build_label_maps(categories: list[str]) -> tuple[dict, dict]:
    label2id = {label: idx for idx, label in enumerate(categories)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def load_and_prepare_dataframe(
    train_path: str,
    categories: list[str],
    label2id: dict,
    random_state: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(train_path)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df["text"] = df["text"].apply(normalize_text)
    df["label"] = df["category"].map(label2id)
    return df


def build_tokenized_dataset(
    df: pd.DataFrame,
    tokenizer,
    label2id: dict,
    number_token_ids: list[int],
    categories: list[str],
    max_input_chars: int = 1500,
) -> Dataset:
    categories_str = "\n".join(
        [f"class {i}: {label}" for i, label in enumerate(categories)]
    )

    def manual_tokenize_and_format(examples):
        input_ids_list, attention_mask_list = [], []

        for input_text, output_text in zip(examples["text"], examples["category"]):
            safe_input = str(input_text)[:max_input_chars]
            class_id_number = label2id[output_text]
            target_token_id = number_token_ids[class_id_number]

            prompt_str = BANKING_PROMPT.format(
                categories_list=categories_str,
                input_text=safe_input,
            )

            tokenized = tokenizer(prompt_str, add_special_tokens=True)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            input_ids.append(target_token_id)
            attention_mask.append(1)

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        return {"input_ids": input_ids_list, "attention_mask": attention_mask_list}

    hf_dataset = Dataset.from_pandas(df[["text", "category"]])
    tokenized_dataset = hf_dataset.map(
        manual_tokenize_and_format,
        batched=True,
        remove_columns=hf_dataset.column_names,
        load_from_cache_file=False,
    )
    return tokenized_dataset