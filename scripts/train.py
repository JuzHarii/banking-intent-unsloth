"""
train.py
Fine-tunes Llama-3.2-1B-Instruct for banking intent classification.
Loads all hyperparameters from configs/train.yaml.

Usage:
    python scripts/train.py --config configs/train.yaml
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from typing import Any, Dict, List, Union

import yaml
from datasets import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from unsloth import FastLanguageModel, is_bfloat16_supported, tokenizer_utils
from torch.utils.data import DataLoader

# Local
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.preprocess_data import (
    normalize_text,
    build_label_maps,
    load_and_prepare_dataframe,
    build_tokenized_dataset,
)


# Silence Unsloth tokenizer warning
def _do_nothing(*args, **kwargs):
    pass

tokenizer_utils.fix_untrained_tokens = _do_nothing


# Custom collator: mask all labels except the last token
class DataCollatorForLastTokenLM(DataCollatorForLanguageModeling):
    def __init__(self, *args, mlm: bool = False, ignore_index: int = -100, **kwargs):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.ignore_index = ignore_index

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["labels"][i]
            non_masked = (labels != self.ignore_index).nonzero(as_tuple=True)[0]
            if len(non_masked) == 0:
                continue
            last_token_idx = non_masked[-1].item()
            original_label = labels[last_token_idx].item()

            batch["labels"][i, :last_token_idx] = self.ignore_index

            if original_label in reverse_map:
                batch["labels"][i, last_token_idx] = reverse_map[original_label]
            else:
                batch["labels"][i, last_token_idx] = self.ignore_index

        return batch


def sanity_check(tokenized_dataset, collator):
    loader = DataLoader(tokenized_dataset, batch_size=2, collate_fn=collator)
    batch = next(iter(loader))
    labels = batch["labels"][0]
    non_masked = labels[labels != -100]
    print("Non-masked labels (should all be 0-76):", non_masked)
    assert all(
        0 <= x <= 76 for x in non_masked.tolist()
    ), "Labels out of range. Check reverse_map."
    print("Labels look correct, safe to train.")


def main(cfg: dict):
    global reverse_map  # needed inside DataCollatorForLastTokenLM

    # 1. Load categories
    with open(cfg["data"]["categories_path"]) as f:
        categories = json.load(f)

    num_classes = cfg["model"]["num_classes"]
    label2id, id2label = build_label_maps(categories)

    os.makedirs("configs", exist_ok=True)
    with open(cfg["data"]["label_mapping_path"], "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=4)
    print(f"Label mapping saved to {cfg['data']['label_mapping_path']}")

    # 2. Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=cfg["model"]["dtype"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )

    # 3. Prune lm_head from 128256 to 77
    number_token_ids = [
        tokenizer.encode(str(i), add_special_tokens=False)[0]
        for i in range(num_classes)
    ]
    pruned_param = torch.nn.Parameter(
        model.lm_head.weight[number_token_ids, :].clone().detach()
    )
    model.lm_head.weight = pruned_param
    reverse_map = {token_id: idx for idx, token_id in enumerate(number_token_ids)}
    print(f"lm_head pruned: {model.lm_head.weight.shape}")

    # 4. Apply LoRA
    lora_cfg = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=lora_cfg["random_state"],
        use_rslora=lora_cfg["use_rslora"],
        loftq_config=None,
    )
    FastLanguageModel.for_training(model)

    # 5. Prepare data
    df = load_and_prepare_dataframe(
        train_path=cfg["data"]["train_path"],
        categories=categories,
        label2id=label2id,
    )
    tokenized_dataset = build_tokenized_dataset(
        df=df,
        tokenizer=tokenizer,
        label2id=label2id,
        number_token_ids=number_token_ids,
        categories=categories,
        max_input_chars=cfg["data"]["max_input_chars"],
    )
    print(f"Dataset tokenized: {len(tokenized_dataset)} samples")

    # 6. Collator and sanity check
    collator = DataCollatorForLastTokenLM(tokenizer=tokenizer)
    sanity_check(tokenized_dataset, collator)

    # 7. Training
    train_cfg = cfg["training"]
    os.makedirs(train_cfg["output_dir"], exist_ok=True)

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        logging_steps=train_cfg["logging_steps"],
        optim=train_cfg["optim"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        seed=train_cfg["seed"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        report_to=train_cfg["report_to"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
    )

    resume = train_cfg.get("resume_from_checkpoint") or None
    trainer_stats = trainer.train(resume_from_checkpoint=resume)
    print(f"Training complete. Stats: {trainer_stats}")

    # 8. Save merged model
    out_cfg = cfg["output"]
    merged_dir = out_cfg["merged_model_dir"]

    model.save_pretrained_merged(
        merged_dir,
        tokenizer,
        save_method=out_cfg["save_method"],
    )

    with open(os.path.join(merged_dir, "label_mapping.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=4)

    print(f"Merged model saved to: {merged_dir}/")
    print("   Upload this folder to Hugging Face, then update configs/inference.yaml.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/train.yaml",
        help="Path to train config YAML"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)