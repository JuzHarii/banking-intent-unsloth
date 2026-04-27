# Banking Intent Classifier With Llama 3.2 + Unsloth

This project fine-tunes **Llama-3.2-1B-Instruct** for banking intent classification across **77 classes**, using LoRA (via [Unsloth](https://github.com/unslothai/unsloth)) and a custom logit-based inference strategy for fast, deterministic predictions.

---

## Demonstration Video
https://drive.google.com/file/d/155NfwXq24ju4GurYDxf-exAbTRqwKERH/view?usp=sharing

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

### 3. Model Configuration

| Parameter           | Value                           |
| :------------------ | :------------------------------ |
| Base Model          | `unsloth/Llama-3.2-1B-Instruct` |
| Max Sequence Length | 2048                            |
| Quantization        | 4-bit                           |

In practice, this configuration gives a good trade-off between model quality and hardware cost. I keep the context length at 2048 because each prompt includes both the user query and all intent labels, so shorter contexts can cut useful information. I also run the model in 4-bit mode to reduce VRAM usage and make experiments easier to run on limited GPUs.

### 4. Training Hyperparameters

| Setting(s)                  | Value        |
| :-------------------------- | :----------- |
| Batch Size (per device)     | `4`          |
| Gradient Accumulation Steps | `4`          |
| Effective Batch Size        | `16`         |
| Learning Rate               | `2e-4`       |
| Learning Rate Scheduler     | `cosine`     |
| Warmup Steps                | `50`         |
| Optimizer                   | `adamw_8bit` |
| Epochs                      | `1`          |
| Weight Decay                | `0.001`      |
| Max Input Characters        | `1500`       |
| Seed                        | `3407`       |
| Checkpoint Strategy         | `steps`      |
| Save Every N Steps          | `100`        |
| Max Saved Checkpoints       | `3`          |

These hyperparameters are stable enough to train, but still fast to iterate. The effective batch size of 16 helps reduce noisy updates, and a learning rate of 2e-4 with cosine scheduling plus warmup makes optimization smoother at the beginning of training. Weight decay is used as lightweight regularization, and frequent checkpoints help recover training.

### 5. LoRA Settings

| LoRA Parameter         | Value                                                                                  |
| :--------------------- | :------------------------------------------------------------------------------------- |
| Rank (`r`)             | `16`                                                                                   |
| Alpha                  | `32`                                                                                   |
| Dropout                | `0`                                                                                    |
| Bias                   | `none`                                                                                 |
| Gradient Checkpointing | `unsloth`                                                                              |
| RSLoRA                 | `false`                                                                                |
| Target Modules         | `lm_head`, `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

For LoRA, I focus on keeping the trainable parameter count low while still adapting the most important attention and feed-forward modules. I keep dropout at 0 in this baseline to avoid adding extra instability in short runs, and I enable Unsloth gradient checkpointing to save memory.

### 6. Run training

```bash
bash train.sh
# or with a custom config:
bash train.sh configs/train.yaml
```

During training, the pipeline does the following:

1. Load `Llama-3.2-1B-Instruct` and prune `lm_head` from 128 256 → 77 tokens
2. Apply LoRA adapters
3. Tokenize your data with a banking prompt template
4. Run a sanity check to verify labels are in range `[0, 76]`
5. Train with `transformers.Trainer`
6. Save the **fully merged 16-bit model** to `model_finetuned/`

### 7. Update `configs/inference.yaml` and run inference

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
