# LLM Fine-Tune Suite

Fine-tune three Qwen2.5 7B instruction models on RunPod with Unsloth, QLoRA, and TRL.

| Model | Base | Dataset | Use case |
|---|---|---|---|
| `krishna-toolcall-7b` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | `glaiveai/glaive-function-calling-v2` | Agentic tool/function calling |
| `krishna-finance-7b` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | `TheFinAI/flare-finqa` + finance instruction data | Financial QA and reasoning |
| `krishna-resumatch-7b` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | Custom JD-to-resume pairs | Resume tailoring from job descriptions |

The scripts save merged 16-bit models locally and can push merged and GGUF `q4_k_m` versions to Hugging Face once you set your repo IDs in `configs/*.yaml`.

## Project Layout

```text
LLM-FineTune-Suite/
├── configs/
│   ├── toolcall_config.yaml
│   ├── finance_config.yaml
│   └── resume_config.yaml
├── data/
│   └── resume_dataset.jsonl
├── notebooks/
│   └── RunPod_QuickStart.ipynb
├── scripts/
│   ├── build_resume_dataset.py
│   ├── test_model.py
│   ├── train_finance.py
│   ├── train_resume.py
│   └── train_toolcall.py
├── LICENSE
├── README.md
└── requirements.txt
```

For the full handoff checklist, see:

```text
docs/PROJECT_STATUS_AND_NEXT_STEPS.md
```

## RunPod Setup

1. Create or open your RunPod account and add funds.
2. Deploy a GPU pod with an A40 48GB GPU.
3. Use a PyTorch template with JupyterLab and SSH enabled.
4. Use about 50 GB persistent volume storage and 20 GB container disk.
5. Stop the pod whenever you are not actively training.

Expected training cost on an A40 is usually well under a $25 budget if you stop the pod between runs.

## Install

On the pod:

```bash
cd /workspace
git clone <YOUR_REPO_URL> LLM-FineTune-Suite
cd LLM-FineTune-Suite

pip install -r requirements.txt
huggingface-cli login
```

Verify GPU:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

## Configure Hugging Face Repos

Edit these values before training if you want automatic Hub pushes:

```yaml
hf_repo: YOUR_HF_USERNAME/krishna-toolcall-7b
```

Files to edit:

```text
configs/toolcall_config.yaml
configs/finance_config.yaml
configs/resume_config.yaml
```

If the placeholder is left unchanged, the scripts still save the merged model locally but skip Hub upload.

## Train

Tool-calling:

```bash
python scripts/train_toolcall.py --config configs/toolcall_config.yaml
```

Finance:

```bash
python scripts/train_finance.py --config configs/finance_config.yaml
```

Resume tailoring:

```bash
python scripts/build_resume_dataset.py --output data/resume_dataset.jsonl
python scripts/train_resume.py --config configs/resume_config.yaml
```

The resume seed data is intentionally small. Add 200-500 real JD-to-resume examples to `data/resume_dataset.jsonl` before expecting strong behavior.

## Test a Pushed Model

```bash
python scripts/test_model.py \
  --config configs/toolcall_config.yaml \
  --repo YOUR_HF_USERNAME/krishna-toolcall-7b \
  --prompt "What's the weather in Boston today?"
```

## Important Settings

The default configs use:

| Setting | Value |
|---|---|
| LoRA rank | `16` |
| LoRA alpha | `16` |
| Quantized load | `4-bit` |
| Max sequence length | `2048` |
| Tool/finance max steps | `500` |
| Resume max steps | `300-500`, based on dataset size |
| Optimizer | `adamw_8bit` |
| Precision | `bf16: auto` |

## Model Card Template

Use this on Hugging Face and adjust the dataset/tags for each model:

```markdown
---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
  - qlora
  - unsloth
  - qwen2.5
language:
  - en
pipeline_tag: text-generation
---

# krishna-toolcall-7b

Fine-tuned Qwen2.5-7B-Instruct for reliable agentic tool/function calling.

## Training

- Method: QLoRA via Unsloth
- LoRA rank: 16
- Hardware: NVIDIA A40 48GB on RunPod
- Dataset: glaiveai/glaive-function-calling-v2

## Intended Use

Building local AI agents that need structured tool/function calling behavior.
```

## Cleanup

When finished:

1. Stop the RunPod pod.
2. Delete the pod if you no longer need it.
3. Delete the network volume after all models and artifacts are pushed or copied.
