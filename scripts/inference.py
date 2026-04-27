"""
inference.py
Standalone inference class for Banking Intent Classification.

Usage example:
    classifier = IntentClassification(model_path="configs/inference.yaml")
    label = classifier("I lost my card and need a replacement")
    print(label)  # "lost_or_stolen_card"
"""

import re
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from unsloth import FastLanguageModel, tokenizer_utils


BANKING_PROMPT = """Here is a banking query:
{input_text}

Classify this query into one of the following intents:

{categories_list}

SOLUTION
The correct answer is: class """


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"[''´`']", "'", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


class IntentClassification:

    def __init__(self, model_path: str):
        """
        Load config, tokenizer, and model from Hugging Face (public repo).

        Args:
            model_path: Path to a YAML config file containing at minimum:
                        model.hf_repo, model.num_classes, model.max_seq_length,
                        data.label_mapping_path, data.max_input_chars
        """
        # Load config
        with open(model_path) as f:
            cfg = yaml.safe_load(f)

        self._num_classes     = cfg["model"]["num_classes"]
        self._max_input_chars = cfg["data"]["max_input_chars"]

        # Silence Unsloth tokenizer warning
        tokenizer_utils.fix_untrained_tokens = lambda *a, **kw: None

        # Load merged model from Hugging Face repo
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["model"]["hf_repo"],
            max_seq_length=cfg["model"]["max_seq_length"],
            dtype=cfg["model"]["dtype"],
            load_in_4bit=cfg["model"]["load_in_4bit"],
        )

        # Runtime pruning guard
        # If lm_head still has full vocab, prune it down to num_classes.
        if self._model.lm_head.weight.shape[0] != self._num_classes:
            print(
                f"lm_head is {self._model.lm_head.weight.shape[0]} — "
                f"pruning to {self._num_classes}..."
            )
            number_token_ids = [
                self._tokenizer.encode(str(i), add_special_tokens=False)[0]
                for i in range(self._num_classes)
            ]
            hidden_dim    = self._model.lm_head.in_features
            pruned_weight = self._model.lm_head.weight.data[number_token_ids, :].clone().detach()
            new_lm_head   = nn.Linear(hidden_dim, self._num_classes, bias=False, device=self._model.device)
            new_lm_head.weight = nn.Parameter(pruned_weight.to(new_lm_head.weight.dtype))
            self._model.lm_head = new_lm_head

        FastLanguageModel.for_inference(self._model)
        print(f"Model ready - lm_head shape: {self._model.lm_head.weight.shape}")

        # Load label mapping
        with open(cfg["data"]["label_mapping_path"]) as f:
            mapping = json.load(f)
        self._id2label = {int(k): v for k, v in mapping["id2label"].items()}

        # Build categories string used in every prompt
        self._categories_str = "\n".join(
            f"class {i}: {self._id2label[i]}" for i in range(self._num_classes)
        )

    def __call__(self, message: str) -> str:
        """
        Predict the intent label for a single text input.

        Args:
            message: Raw user query string.

        Returns:
            predicted_label: The intent class name (e.g. "lost_or_stolen_card").
        """
        prompt = BANKING_PROMPT.format(
            input_text=_normalize_text(message)[:self._max_input_chars],
            categories_list=self._categories_str,
        )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.inference_mode():
            outputs = self._model(**inputs)

        # Slice to num_classes as an extra safety check.
        last_token_logits = outputs.logits[0, -1, :self._num_classes]
        pred_id = torch.argmax(last_token_logits).item()

        predicted_label = self._id2label[pred_id]
        return predicted_label


# Usage example
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/inference.yaml",
        help="Path to inference config YAML (model_path)"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Single query to classify"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run evaluation on the full test set"
    )
    args = parser.parse_args()

    # Instantiate
    classifier = IntentClassification(model_path=args.config)

    # Single query
    if args.query:
        label = classifier(args.query)
        print(f"Query : {args.query}")
        print(f"Intent: {label}")

    # Evaluation
    elif args.evaluate:
        import os
        import pandas as pd
        from tqdm.auto import tqdm
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
        )

        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        test_df = pd.read_csv(cfg["data"]["test_path"])
        test_df["text"] = test_df["text"].apply(_normalize_text)

        y_true, y_pred = [], []
        for text, label in tqdm(
            zip(test_df["text"], test_df["category"]),
            total=len(test_df),
            desc="Evaluating",
        ):
            y_true.append(label)
            y_pred.append(classifier(text))

        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        report = "\n".join([
            f"Accuracy : {acc * 100:.2f}%",
            f"Precision: {p   * 100:.2f}%",
            f"Recall   : {r   * 100:.2f}%",
            f"F1-Score : {f1  * 100:.2f}%",
        ])
        print(report)

        out_path = cfg["evaluation"]["output_report_path"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(report)
        print(f"Report saved to {out_path}")

    # Interactive mode
    else:
        print("Interactive mode - type 'exit' to quit.\n")
        while True:
            query = input("Query: ").strip()
            if query.lower() in ("exit", "quit", "q"):
                break
            print(f"Intent: {classifier(query)}\n")