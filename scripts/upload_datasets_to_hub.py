#!/usr/bin/env python3
"""
============================================================
Upload ViMed Datasets to HuggingFace Hub
============================================================
This script uploads SFT and DPO datasets with detailed dataset cards.

Usage:
    python scripts/upload_datasets_to_hub.py

Environment Variables:
    - HF_TOKEN: HuggingFace token with write access
============================================================
"""

import os
import json
import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login

# ============================================================
# Configuration
# ============================================================
HF_USERNAME = "quannguyen204"

# Dataset paths
DATA_DIR = Path(__file__).parent.parent / "data"
SFT_DATA_PATH = DATA_DIR / "final_sft_cleaned.jsonl"
DPO_DATA_PATH = DATA_DIR / "final_dpo_cleaned.jsonl"

# HuggingFace repo names
SFT_REPO_ID = f"{HF_USERNAME}/medical_sft_crawl_vi_10k_v1"
DPO_REPO_ID = f"{HF_USERNAME}/medical_dpo_synthetic_vi_3.6k_v1"


# ============================================================
# Dataset Card Templates
# ============================================================
SFT_DATASET_CARD = """---
license: apache-2.0
language:
  - vi
task_categories:
  - text-generation
tags:
  - medical
  - healthcare
  - vietnamese
  - sft
  - chat
  - instruction-tuning
size_categories:
  - 1K<n<10K
---

# ViMed-SFT: Vietnamese Medical Conversational Dataset

## Dataset Description

**ViMed-SFT** is a high-quality Vietnamese medical conversational dataset designed for Supervised Fine-Tuning (SFT) of Large Language Models. The dataset contains medical Q&A conversations between users and a virtual medical assistant.

### Dataset Summary

| Attribute | Value |
|-----------|-------|
| **Language** | Vietnamese |
| **Domain** | Healthcare / Medical |
| **Task** | Conversational AI, Instruction Tuning |
| **Samples** | {num_samples:,} conversations |
| **Format** | ChatML-compatible messages |

## Dataset Structure

### Data Format

Each sample contains a `messages` field with a list of conversation turns:

```json
{{
  "messages": [
    {{
      "role": "system",
      "content": "Bạn là một trợ lý y tế ảo thông minh..."
    }},
    {{
      "role": "user", 
      "content": "User's medical question"
    }},
    {{
      "role": "assistant",
      "content": "Medical assistant's response"
    }}
  ]
}}
```

### Fields

- `messages`: List of conversation turns
  - `role`: One of "system", "user", or "assistant"
  - `content`: The text content of the message

### System Prompt

All conversations use a consistent system prompt that instructs the model to:
- Act as a professional online medical consultant
- Provide informative, guidance-oriented responses
- Explain possible causes
- Always recommend visiting healthcare facilities for accurate diagnosis
- Never replace professional medical diagnosis

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("{sft_repo_id}")

# Access training data
train_data = dataset["train"]
print(f"Number of samples: {{len(train_data)}}")

# Preview a sample
print(train_data[0])
```

### Training with TRL

```python
from trl import SFTTrainer, SFTConfig

# Use with SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    # ... other config
)
```

## Medical Topics Covered

The dataset covers a wide range of medical topics including:
- General health consultations
- Medication questions
- Lab test interpretations (blood tests, etc.)
- Disease symptoms and conditions
- Nutrition and lifestyle advice
- Pediatric health
- Women's health
- Orthopedic issues
- Cardiovascular health
- Neurological conditions
- And many more...

## Intended Use

This dataset is intended for:
- Fine-tuning LLMs for Vietnamese medical chatbot applications
- Research in medical NLP for Vietnamese language
- Building healthcare AI assistants

## Limitations and Biases

- **Not a replacement for professional medical advice**: Models trained on this data should always recommend users consult healthcare professionals
- **Vietnamese-specific**: Optimized for Vietnamese language and medical context
- **General medical knowledge**: Does not cover specialized or rare conditions comprehensively

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{vimed_sft_2024,
  author = {{Quan Nguyen}},
  title = {{ViMed-SFT: Vietnamese Medical Conversational Dataset}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/datasets/{sft_repo_id}}}
}}
```

## License

This dataset is released under the Apache 2.0 License.

## Contact

For questions or feedback, please open an issue on the dataset repository.
"""


DPO_DATASET_CARD = """---
license: apache-2.0
language:
  - vi
task_categories:
  - text-generation
tags:
  - medical
  - healthcare
  - vietnamese
  - dpo
  - preference-learning
  - rlhf
  - alignment
size_categories:
  - 1K<n<10K
---

# ViMed-DPO: Vietnamese Medical Preference Dataset

## Dataset Description

**ViMed-DPO** is a preference dataset for Direct Preference Optimization (DPO) training of Vietnamese medical AI assistants. Each sample contains a prompt with a chosen (preferred) response and a rejected (less preferred) response.

### Dataset Summary

| Attribute | Value |
|-----------|-------|
| **Language** | Vietnamese |
| **Domain** | Healthcare / Medical |
| **Task** | Preference Learning, RLHF, DPO |
| **Samples** | {num_samples:,} preference pairs |
| **Format** | prompt/chosen/rejected |

## Dataset Structure

### Data Format

Each sample contains three fields:

```json
{{
  "prompt": "User's medical question",
  "chosen": "High-quality, accurate medical response",
  "rejected": "Lower-quality or incorrect response"
}}
```

### Fields

- `prompt`: The user's medical question
- `chosen`: The preferred response - accurate, helpful, and follows medical guidelines
- `rejected`: The less preferred response - may be inaccurate, unhelpful, or violate medical guidelines

### Quality Criteria

**Chosen responses** are characterized by:
- Accurate medical information
- Recommends professional consultation
- Explains reasoning clearly
- Provides actionable guidance
- Appropriate level of detail

**Rejected responses** may have issues such as:
- Incorrect medical information
- Dismissive of symptoms
- Missing recommendation to see a doctor
- Overly simplistic or vague
- Potentially harmful advice

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("{dpo_repo_id}")

# Access training data
train_data = dataset["train"]
print(f"Number of preference pairs: {{len(train_data)}}")

# Preview a sample
sample = train_data[0]
print(f"Prompt: {{sample['prompt']}}")
print(f"Chosen: {{sample['chosen']}}")
print(f"Rejected: {{sample['rejected']}}")
```

### Training with TRL DPOTrainer

```python
from trl import DPOTrainer, DPOConfig

# Prepare dataset with chat template
def format_dpo_sample(example):
    system_prompt = "Bạn là một trợ lý y tế ảo thông minh..."
    messages = [
        {{"role": "system", "content": system_prompt}},
        {{"role": "user", "content": example["prompt"]}}
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {{
        "prompt": formatted_prompt,
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }}

formatted_dataset = dataset.map(format_dpo_sample)

# Train with DPOTrainer
trainer = DPOTrainer(
    model=model,
    train_dataset=formatted_dataset["train"],
    args=DPOConfig(beta=0.1, ...),
)
```

## Example Samples

### Example 1: Vitamin Information

**Prompt**: "Vitamin C có vai trò gì trong cơ thể?"

**Chosen** :
> Vitamin C là một chất chống oxy hóa quan trọng, hỗ trợ hệ miễn dịch, giúp tổng hợp collagen cho da, xương và mạch máu, và tăng cường hấp thu sắt từ thức ăn...

**Rejected** :
> Vitamin C chỉ giúp cơ thể chống cảm lạnh và là vitamin quan trọng, không cần quan tâm nhiều đến liều lượng hay các tác dụng khác.

### Example 2: Disease Recognition

**Prompt**: "Làm sao để nhận biết bệnh xơ cứng rải rác ở giai đoạn đầu?"

**Chosen** :
> Ở giai đoạn đầu, xơ cứng rải rác có thể biểu hiện qua những triệu chứng như mệt mỏi, tê cứng hoặc ngứa ở các chi... Nếu nghi ngờ, bạn nên đến gặp bác sĩ chuyên khoa để được chẩn đoán chính xác.

**Rejected** :
> Xơ cứng rải rác thường khó nhận biết ở giai đoạn đầu... Kiểm tra máu hay đi khám cũng không cần thiết ngay...

## Medical Topics Covered

- Vitamins and supplements
- Medical tests and procedures
- Disease diagnosis and symptoms
- Pediatric health
- Women's health (pregnancy, etc.)
- Dental health
- Cardiovascular conditions
- Nutrition and diet
- And more...

## Intended Use

This dataset is intended for:
- DPO/RLHF training of Vietnamese medical AI
- Preference learning for healthcare chatbots
- Research in medical AI alignment

## Limitations

- **Supplement to SFT**: Best used after SFT training
- **Vietnamese-specific**: Optimized for Vietnamese medical context
- **Not exhaustive**: May not cover all medical edge cases

## Citation

```bibtex
@dataset{{vimed_dpo_2025,
  author = {{Quan Nguyen}},
  title = {{ViMed-DPO: Vietnamese Medical Preference Dataset}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/datasets/{dpo_repo_id}}}
}}
```

## License

Apache 2.0 License

## Related Datasets

- [ViMed-SFT]({sft_repo_link}) - SFT dataset for initial fine-tuning

## Contact

For questions or feedback, please open an issue on the dataset repository.
"""


# ============================================================
# Helper Functions
# ============================================================
def load_jsonl(file_path: Path) -> list[dict]:
    """Load JSONL file into list of dicts."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def upload_dataset(
    data: list[dict],
    repo_id: str,
    readme_content: str,
    token: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.05,
):
    """
    Upload dataset to HuggingFace Hub with train/validation/test split.

    Args:
        data: List of data samples
        repo_id: HuggingFace repository ID
        readme_content: Dataset card content
        token: HuggingFace token
        val_ratio: Validation set ratio (default 10%)
        test_ratio: Test set ratio (default 5%)

    Split strategy: Train (85%) / Validation (10%) / Test (5%)
    - Validation: Used for early stopping & hyperparameter tuning during training
    - Test: Used for comprehensive final evaluation on unseen data (generalization check)
    - Smaller test set ensures full evaluation during inference
    """
    print(f"\n{'='*60}")
    print(f"Uploading to: {repo_id}")
    print(f"Total samples: {len(data)}")

    # Create Dataset
    dataset = Dataset.from_list(data)

    # First split: separate test set
    train_val_test = dataset.train_test_split(test_size=test_ratio, seed=42)
    test_dataset = train_val_test["test"]

    # Second split: separate validation from remaining train
    # Adjust ratio: val_ratio / (1 - test_ratio) to get correct proportion
    adjusted_val_ratio = val_ratio / (1 - test_ratio)
    train_val = train_val_test["train"].train_test_split(
        test_size=adjusted_val_ratio, seed=42
    )

    # Create final DatasetDict with 3 splits
    dataset_dict = DatasetDict(
        {
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": test_dataset,
        }
    )

    print(
        f"Train samples: {len(dataset_dict['train'])} ({100*(1-val_ratio-test_ratio):.0f}%)"
    )
    print(
        f"Validation samples: {len(dataset_dict['validation'])} ({100*val_ratio:.0f}%)"
    )
    print(f"Test samples: {len(dataset_dict['test'])} ({100*test_ratio:.0f}%)")

    # Push to Hub
    dataset_dict.push_to_hub(
        repo_id,
        token=token,
        private=False,
    )

    # Update README
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f" Successfully uploaded to https://huggingface.co/datasets/{repo_id}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Upload ViMed datasets to HuggingFace Hub"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--sft-only", action="store_true", help="Only upload SFT dataset"
    )
    parser.add_argument(
        "--dpo-only", action="store_true", help="Only upload DPO dataset"
    )
    args = parser.parse_args()

    # Get token
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print(" Error: HF_TOKEN not found. Set HF_TOKEN env var or use --token")
        return

    # Login
    login(token=token)
    print(" Logged in to HuggingFace Hub")

    # Upload SFT dataset
    if not args.dpo_only:
        print("\n" + "=" * 60)
        print("Loading SFT dataset...")
        sft_data = load_jsonl(SFT_DATA_PATH)

        sft_readme = SFT_DATASET_CARD.format(
            num_samples=len(sft_data),
            sft_repo_id=SFT_REPO_ID,
        )

        upload_dataset(
            data=sft_data,
            repo_id=SFT_REPO_ID,
            readme_content=sft_readme,
            token=token,
        )

    # Upload DPO dataset
    if not args.sft_only:
        print("\n" + "=" * 60)
        print("Loading DPO dataset...")
        dpo_data = load_jsonl(DPO_DATA_PATH)

        dpo_readme = DPO_DATASET_CARD.format(
            num_samples=len(dpo_data),
            dpo_repo_id=DPO_REPO_ID,
            sft_repo_link=f"https://huggingface.co/datasets/{SFT_REPO_ID}",
        )

        upload_dataset(
            data=dpo_data,
            repo_id=DPO_REPO_ID,
            readme_content=dpo_readme,
            token=token,
        )

    print("\n" + "=" * 60)
    print("🎉 All uploads complete!")
    print("=" * 60)
    print(f"\nDataset links:")
    if not args.dpo_only:
        print(f"  - SFT: https://huggingface.co/datasets/{SFT_REPO_ID}")
    if not args.sft_only:
        print(f"  - DPO: https://huggingface.co/datasets/{DPO_REPO_ID}")


if __name__ == "__main__":
    main()
