# Banking Intent Classifier — Llama 3.2 + Unsloth

Fine-tuned **Llama-3.2-1B-Instruct** for banking intent classification across **77 classes**, using LoRA (via [Unsloth](https://github.com/unslothai/unsloth)) and a custom logit-based inference strategy (no `model.generate()`).

---

## Repository Structure

```
banking-intent-unsloth/
├── scripts/
│   ├── train.py              # Full training pipeline
│   ├── inference.py          # Inference + evaluation from Hugging Face
│   └── preprocess_data.py    # Data loading, normalization, tokenization
├── configs/
│   ├── train.yaml            # All training hyperparameters & paths
│   ├── inference.yaml        # HF repo + inference/evaluation settings
│   └── label_mapping.json    # Generated after training
├── sample_data/
│   ├── categories.json       # List of 77 intent class names
│   ├── train.csv
│   ├── test.csv
│   └── test_demo.csv         # Optional balanced demo subset (1000 rows)
├── train.sh                  # Shell wrapper for train.py
├── inference.sh              # Shell wrapper for inference.py
├── requirements.txt
└── README.md
```

---

## Installation

> **Unsloth must be installed separately** because its installation command depends on your CUDA version.

```bash
# 1. Clone the repo
git clone https://github.com/juzharii/banking-intent-unsloth.git
cd banking-intent-unsloth

# 2. Install Unsloth (see https://github.com/unslothai/unsloth for your CUDA version)
pip install unsloth

# 3. Install remaining dependencies
pip install -r requirements.txt
```

---

## Data Format

Both `train.csv` and `test.csv` must have exactly two columns:

| text                      | category            |
| ------------------------- | ------------------- |
| i lost my card            | lost_or_stolen_card |
| what is the exchange rate | exchange_rate       |

`sample_data/categories.json` must be a JSON array of all 77 class name strings, in the same order used during training:

```json
["lost_or_stolen_card", "exchange_rate", "card_arrival", ...]
```

---

## Option A — Run Inference Only (Pre-trained Model)

If you just want to classify banking queries using the already fine-tuned model on Hugging Face, **you do not need to train anything**.

### 1. Configure `configs/inference.yaml`

```yaml
model:
  hf_repo: "juzharii/banking-intent-llama-3.2"
```

### 2. Single query

```bash
bash inference.sh "I need to reset my PIN"
# Output:
# Query : I need to reset my PIN
# Intent: change_pin
```

### 3. Interactive mode

```bash
bash inference.sh
# Query: my card was declined abroad
# Intent: card_not_working

# Query: what is the exchange rate
# Intent: exchange_rate

# Query: i lost my card
# Intent: lost_or_stolen_card
```

### 4. Evaluate on test set

```bash
bash inference.sh --evaluate
# Accuracy : 91.24%
# Precision: 91.58%
# Recall   : 91.24%
# F1-Score : 91.19%
```

To evaluate on the balanced 1000-sample demo file, set in `configs/inference.yaml`:

```yaml
data:
  test_path: "sample_data/test_demo.csv"
```

---

## Option B — Train From Scratch

Use this option if you want to fine-tune the model yourself on your own data.

### 1. Prepare your data

Place your files at the paths defined in `configs/train.yaml`:

- `sample_data/train.csv`
- `sample_data/categories.json`

### 2. Configure `configs/train.yaml`

Key fields to review:

```yaml
model:
  name: "unsloth/Llama-3.2-1B-Instruct"
  num_classes: 77

data:
  train_path: "sample_data/train.csv"
  categories_path: "sample_data/categories.json"

training:
  num_train_epochs: 1
  learning_rate: 2e-4
  output_dir: "outputs"

output:
  merged_model_dir: "model_finetuned"
  save_method: "merged_16bit"
```

### 3. Hyperparameter Configuration and Rationale

The following choices summarize both the values I used and why I selected them for this project.

1. Batch size
   I used `per_device_train_batch_size: 4` and `gradient_accumulation_steps: 4`, so the effective batch size is 16 samples per optimizer update. I chose this setup to keep memory usage manageable while still getting more stable gradient updates than a small effective batch.

2. Learning rate
   I used `learning_rate: 2e-4` with `lr_scheduler_type: "cosine"` and `warmup_steps: 50`. I chose this because `2e-4` is a practical range for LoRA fine-tuning, and warmup plus cosine decay helps avoid unstable early updates and supports smoother convergence.

3. Optimizer
   I used `optim: "adamw_8bit"`. I chose 8-bit AdamW to reduce optimizer memory overhead while keeping optimization behavior close to AdamW, which is important when training large models on limited hardware.

4. Number of training steps or epochs
   I trained for `num_train_epochs: 1`, and saved checkpoints with `save_strategy: "steps"`, `save_steps: 100`, `save_total_limit: 3`. I chose one epoch as a strong baseline to control training time and reduce overfitting risk, while periodic checkpoints make it easy to resume or compare intermediate states.

5. Maximum sequence length
   I set `max_seq_length: 2048`. I chose this because each input prompt contains both the user query and the full list of 77 intent classes, so a longer context window helps avoid truncating useful prompt information.

6. Regularization or augmentation techniques
   I used `weight_decay: 0.001` and LoRA settings `r: 16`, `lora_alpha: 32`, `lora_dropout: 0`. I did not apply explicit data augmentation. Instead, I controlled input quality with text normalization and `max_input_chars: 1500` truncation. This keeps training simple and reproducible while still limiting noise from very long or inconsistent inputs.

### 4. Run training

```bash
bash train.sh
# or with a custom config:
bash train.sh configs/train.yaml
```

Training will:

1. Load `Llama-3.2-1B-Instruct` and prune `lm_head` from 128 256 → 77 tokens
2. Apply LoRA adapters
3. Tokenize your data with a banking prompt template
4. Run a sanity check to verify labels are in range `[0, 76]`
5. Train with `transformers.Trainer` (not SFTTrainer)
6. Save the **fully merged 16-bit model** to `model_finetuned/`

### 5. Upload to Hugging Face

```bash
huggingface-cli upload your-username/banking-intent-llama3 model_finetuned/ .
```

Or via Python:

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
  folder_path="model_finetuned",
    repo_id="your-username/banking-intent-llama3",
    repo_type="model",
)
```

### 6. Update `configs/inference.yaml` and run inference

```yaml
model:
  hf_repo: "your-username/banking-intent-llama3"
```

```bash
bash inference.sh "I want to top up my account"
```

---

## How Inference Works

This model does **not** use `model.generate()`. Instead:

1. The full banking prompt (query + all 77 class descriptions) is tokenized and fed to the model.
2. The **logits of the last token** are extracted — shape `[77]` because `lm_head` was pruned to 77 during training and saved merged.
3. `torch.argmax` on those 77 logits gives the predicted class index.
4. `label_mapping.json` maps the index back to the intent label name.

This makes inference fast and deterministic.

---

## Resume Training from Checkpoint

Set `resume_from_checkpoint` in `configs/train.yaml`:

```yaml
training:
  resume_from_checkpoint: "outputs/checkpoint-400"
```

Then rerun `bash train.sh`.

---

## Key Design Decisions

| Decision                           | Reason                                                                                                                                      |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `Trainer` instead of `SFTTrainer`  | `SFTTrainer` applies its own internal label masking that conflicts with the custom `DataCollatorForLastTokenLM`, corrupting training labels |
| `lm_head` pruned to 77 before LoRA | Forces the model to output exactly 77 logits, one per intent class, with no wasted capacity                                                 |
| `merged_16bit` save                | Merges LoRA into base weights so inference needs no PEFT/adapter loading — clean and portable                                               |
| Logit argmax, not `generate()`     | Classification tasks need a single discrete output; argmax on logits is faster and more accurate than text generation for this use case     |
