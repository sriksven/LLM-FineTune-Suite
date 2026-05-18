# LLM-FineTune-Suite

Seven specialized Qwen2.5-7B instruction models fine-tuned with **Unsloth + QLoRA** on **RunPod**, published to Hugging Face. Total training cost: ~$10.

## Models

| Model | HF Link | Dataset | Use Case |
|---|---|---|---|
| **krishna-toolcall-7b** | [sriksven/krishna-toolcall-7b](https://huggingface.co/sriksven/krishna-toolcall-7b) | glaiveai/glaive-function-calling-v2 | Agentic tool/function calling |
| **krishna-finance-7b** | [sriksven/krishna-finance-7b](https://huggingface.co/sriksven/krishna-finance-7b) | TheFinAI/flare-finqa + Sujet-Finance-177k | Financial QA and reasoning |
| **krishna-resumatch-7b** | [sriksven/krishna-resumatch-7b](https://huggingface.co/sriksven/krishna-resumatch-7b) | Custom JD-to-resume pairs | Resume tailoring from job descriptions |
| **SQLForge-7B** | [sriksven/SQLForge-7B](https://huggingface.co/sriksven/SQLForge-7B) | gretelai/synthetic_text_to_sql | Natural language to SQL generation |
| **ExtractIQ-7B** | [sriksven/ExtractIQ-7B](https://huggingface.co/sriksven/ExtractIQ-7B) | Universal-NER/Pile-NER-type | Structured data extraction / NER |
| **CodeLens-7B** | [sriksven/CodeLens-7B](https://huggingface.co/sriksven/CodeLens-7B) | sahil2801/CodeAlpaca-20k | Code review, bug detection, programming |
| **MedSage-7B** | [sriksven/MedSage-7B](https://huggingface.co/sriksven/MedSage-7B) | medalpaca flashcards + wikidoc + MedQuad | Medical question answering |

All models share the same base: `Qwen/Qwen2.5-7B-Instruct`, quantized to 4-bit with QLoRA (rank 16, alpha 16), trained with Unsloth + TRL SFTTrainer, and saved as merged 16-bit safetensors.

## Training Stack

| Component | Detail |
|---|---|
| Base model | Qwen2.5-7B-Instruct (4-bit NF4) |
| Fine-tuning | QLoRA via Unsloth 2026.5.2 |
| Trainer | SFTTrainer from TRL |
| LoRA config | rank=16, alpha=16, all attn + MLP projections |
| Hardware | NVIDIA RTX A5000 (24GB) on RunPod |
| Precision | BF16 training |
| Format | ChatML |
| Output | Merged 16-bit safetensors |

## Training Results

| Model | Steps | Final Loss | Training Time | Approx Cost |
|---|---|---|---|---|
| krishna-toolcall-7b | 500 | 0.375 | ~2.75 hrs | ~$0.75 |
| krishna-finance-7b | 500 | — | ~2.75 hrs | ~$0.75 |
| krishna-resumatch-7b | 300 | 0.218 | ~6.5 min | ~$0.05 |
| SQLForge-7B | 500 | 0.414 | ~2.75 hrs | ~$0.75 |
| ExtractIQ-7B | 500 | 0.115 | ~2.25 hrs | ~$0.60 |
| CodeLens-7B | 500 | 0.450 | ~2.65 hrs | ~$0.70 |
| MedSage-7B | 500 | 1.006 | ~2.75 hrs | ~$0.75 |

## Quick Start

### Use a model with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("sriksven/SQLForge-7B")
tokenizer = AutoTokenizer.from_pretrained("sriksven/SQLForge-7B")

messages = [
    {"role": "system", "content": "You are an expert SQL assistant."},
    {"role": "user", "content": "Schema:\nCREATE TABLE sales (id INT, product VARCHAR(50), amount DECIMAL, region VARCHAR(20));\n\nQuestion: Total sales by region?"},
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Use with Unsloth (faster, less VRAM)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="sriksven/SQLForge-7B",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

## Reproduce Training

### RunPod Setup

1. Deploy a GPU pod: RTX A5000 (24GB) or A40 (48GB)
2. Template: RunPod PyTorch 2.4.0
3. Volume disk: 50 GB
4. Container disk: 20 GB

### Install and Train

```bash
cd /workspace
git clone https://github.com/sriksven/LLM-FineTune-Suite.git
cd LLM-FineTune-Suite

pip install -r requirements.txt
hf auth login

# Round 1: Original 3 models
python scripts/train_toolcall.py --config configs/toolcall_config.yaml
python scripts/train_finance.py --config configs/finance_config.yaml
python scripts/train_resume.py --config configs/resume_config.yaml

# Round 2: 4 additional models
python scripts/train_sql.py
python scripts/train_extractor.py
python scripts/train_codereview.py
python scripts/train_medical.py
```

### Push to Hugging Face

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("sriksven/MODEL_NAME", exist_ok=True, token=True)
api.upload_folder(folder_path="./OUTPUT_DIR/merged", repo_id="sriksven/MODEL_NAME", token=True)
```

## Project Layout

```text
LLM-FineTune-Suite/
├── configs/
│   ├── toolcall_config.yaml
│   ├── finance_config.yaml
│   └── resume_config.yaml
├── data/
│   └── resume_dataset.jsonl
├── docs/
│   └── PROJECT_STATUS_AND_NEXT_STEPS.md
├── notebooks/
│   └── RunPod_QuickStart.ipynb
├── scripts/
│   ├── build_resume_dataset.py
│   ├── test_model.py
│   ├── train_toolcall.py
│   ├── train_finance.py
│   ├── train_resume.py
│   ├── train_sql.py
│   ├── train_extractor.py
│   ├── train_codereview.py
│   └── train_medical.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Default Training Settings

| Setting | Value |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Quantized load | 4-bit NF4 |
| Max sequence length | 2048 |
| Max steps | 500 (300 for resume) |
| Optimizer | adamw_8bit |
| Precision | bf16 |
| Packing | Enabled (except resume) |
| Batch size | 16 effective (4 × 4 accumulation) |

## Budget Breakdown

| Item | Cost |
|---|---|
| Round 1: 3 models (toolcall, finance, resume) + push | ~$3.50 |
| Round 2: 4 models (SQL, extractor, code review, medical) + push | ~$6.50 |
| **Total** | **~$10** |

Trained on $25 RunPod budget with ~$15 remaining.

## Cleanup

When finished:

1. Stop the RunPod pod
2. Delete the pod
3. Delete the volume to stop storage charges
4. Regenerate your Hugging Face token if it was used in any shared environment

## License

Apache 2.0 — see [LICENSE](LICENSE)