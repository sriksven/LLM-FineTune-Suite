# Project Status And Next Steps

This document explains what has already been built in this repo, what is still left, what you need to provide, and exactly what to do next on RunPod and Hugging Face.

## Goal

Fine-tune and publish three 7B LLMs using Unsloth, QLoRA, TRL, and RunPod:

| Model | Base model | Dataset | Purpose |
|---|---|---|---|
| `krishna-toolcall-7b` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | `glaiveai/glaive-function-calling-v2` | Tool/function calling |
| `krishna-finance-7b` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | `TheFinAI/flare-finqa` plus finance instruction data | Financial QA and reasoning |
| `krishna-resumatch-7b` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | Custom JD-to-resume examples | Resume tailoring |

## What Is Done

The repo has been scaffolded and now contains the main files needed to train the three models.

### Project files

Created:

```text
README.md
LICENSE
requirements.txt
.gitignore
```

### Config files

Created:

```text
configs/toolcall_config.yaml
configs/finance_config.yaml
configs/resume_config.yaml
```

These configs contain:

- Base model name
- Max sequence length
- Output directory
- Hugging Face repo name placeholder
- Dataset settings
- LoRA settings
- Training hyperparameters
- Hub push settings

Current repo placeholders still need your Hugging Face username:

```yaml
hf_repo: YOUR_HF_USERNAME/krishna-toolcall-7b
hf_repo: YOUR_HF_USERNAME/krishna-finance-7b
hf_repo: YOUR_HF_USERNAME/krishna-resumatch-7b
```

### Training scripts

Created:

```text
scripts/train_toolcall.py
scripts/train_finance.py
scripts/train_resume.py
```

These scripts:

- Load Qwen2.5 7B through Unsloth in 4-bit mode
- Add LoRA adapters
- Format datasets into ChatML-style text
- Train with `SFTTrainer`
- Save a merged 16-bit model locally
- Push merged and GGUF versions to Hugging Face if `hf_repo` is configured

Important behavior:

- If `hf_repo` still starts with `YOUR_HF_USERNAME/`, the scripts skip Hugging Face upload and only save locally.
- This prevents accidental failures before you configure your Hub repos.

### Resume dataset builder

Created:

```text
scripts/build_resume_dataset.py
```

This creates:

```text
data/resume_dataset.jsonl
```

Current seed dataset size:

```text
10 examples
```

This is enough to test the pipeline, but not enough for a strong resume model.

### Test script

Created:

```text
scripts/test_model.py
```

This loads a pushed model from Hugging Face and runs a prompt through it.

### Notebook

Created:

```text
notebooks/RunPod_QuickStart.ipynb
```

This notebook contains the basic RunPod flow:

- Install dependencies
- Check GPU
- Run each training script

### Validation already completed

The following local checks passed:

```text
Python syntax compile for all scripts
YAML config parsing
Notebook JSON parsing
Resume JSONL parsing
```

No actual model training has been run locally, because that requires a CUDA GPU such as the RunPod A40.

## What Is Left

These are the remaining tasks before the full project is complete.

### 1. Configure Hugging Face repo names

You need to replace `YOUR_HF_USERNAME` in all three config files.

Example:

```yaml
hf_repo: krishnasrikar/krishna-toolcall-7b
```

Files:

```text
configs/toolcall_config.yaml
configs/finance_config.yaml
configs/resume_config.yaml
```

### 2. Create or confirm Hugging Face access

You need a Hugging Face account and a write token.

Token page:

```text
https://huggingface.co/settings/tokens
```

The token must have permission to create and write model repositories.

You will use it on RunPod with:

```bash
huggingface-cli login
```

### 3. Add more resume training data

The resume model needs more examples.

Current:

```text
10 examples
```

Recommended:

```text
200-500 examples
```

Minimum useful target:

```text
50-100 examples
```

Each example must be one JSON object per line in:

```text
data/resume_dataset.jsonl
```

Required format:

```json
{"instruction":"Given this job description, generate a tailored 1-line resume bio and 6 technical skill headers with relevant skills for each.\n\nJob Description: ...","output":"Bio: ...\n\nSkills:\nHeader 1: ...\nHeader 2: ...\nHeader 3: ...\nHeader 4: ...\nHeader 5: ...\nHeader 6: ..."}
```

Important:

Do not run this command after manually expanding the dataset:

```bash
python scripts/build_resume_dataset.py --output data/resume_dataset.jsonl
```

That command regenerates the seed dataset and will overwrite your expanded file.

### 4. Run training on RunPod

Training should be done on a GPU pod, not locally.

Recommended:

```text
GPU: NVIDIA A40 48GB
Template: RunPod PyTorch template
Persistent volume: 50 GB or more
Container disk: 20 GB or more
```

### 5. Test pushed models

After training and pushing, run:

```bash
python scripts/test_model.py \
  --config configs/toolcall_config.yaml \
  --repo YOUR_HF_USERNAME/krishna-toolcall-7b \
  --prompt "What's the weather in Boston today?"
```

Repeat with the finance and resume repos.

### 6. Write Hugging Face model cards

After each model is pushed, update the model README on Hugging Face.

Each model card should include:

- Base model
- Dataset
- Training method
- Intended use
- Limitations
- Usage example
- License

### 7. Stop or delete RunPod resources

When training is done:

1. Stop the pod if you may use it again.
2. Delete the pod if fully done.
3. Delete the network volume after all artifacts are pushed or copied.

This avoids ongoing storage charges.

## What You Should Give Me

Give me these items if you want me to finish the repo setup further.

### Required

1. Your Hugging Face username.

Example:

```text
krishnasrikar
```

I can then update all three config files.

2. Final model repo names if you want different names.

Current planned names:

```text
krishna-toolcall-7b
krishna-finance-7b
krishna-resumatch-7b
```

If these are fine, only give your username.

### Strongly recommended

3. Resume examples.

Best source:

- Job descriptions you applied to
- Resume bio and skill sections previously generated for those jobs
- Claude or ChatGPT conversations where you generated JD-specific resume bullets, bios, or skills

Send them in any rough format and I can convert them into JSONL.

Good input format:

```text
JD:
<paste job description>

Output:
Bio: ...

Skills:
Machine Learning: ...
Cloud & MLOps: ...
Data Engineering: ...
Programming: ...
...
```

### Optional

4. Whether Hugging Face repos should be public or private.

The current scripts use Unsloth push helpers and assume normal Hub behavior. If you want private repos, create them manually first or adjust the push workflow.

5. Whether to reduce cost and time further.

Options:

- Lower sample size in configs
- Lower `max_steps`
- Save only LoRA adapters instead of merged 16-bit models
- Skip GGUF upload until later

6. Whether the final models should be optimized for a specific runtime.

Examples:

```text
Ollama
llama.cpp
vLLM
Transformers
TGI
Open WebUI
LM Studio
```

This affects whether GGUF, merged safetensors, or adapter-only artifacts matter most.

## What You Should Do Next

Follow this order.

### Step 1: Give your Hugging Face username

Once provided, update:

```text
configs/toolcall_config.yaml
configs/finance_config.yaml
configs/resume_config.yaml
```

Example final values:

```yaml
hf_repo: yourname/krishna-toolcall-7b
hf_repo: yourname/krishna-finance-7b
hf_repo: yourname/krishna-resumatch-7b
```

### Step 2: Add resume examples

If you do not have 200-500 examples yet, start with 50.

Keep every JSONL row valid:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("data/resume_dataset.jsonl")
for line_number, line in enumerate(path.read_text().splitlines(), start=1):
    row = json.loads(line)
    assert "instruction" in row and "output" in row, line_number
print(f"Validated {line_number} examples")
PY
```

### Step 3: Push or upload this repo to RunPod

Option A, from GitHub:

```bash
git clone <YOUR_REPO_URL> LLM-FineTune-Suite
cd LLM-FineTune-Suite
```

Option B, from your local machine:

```bash
scp -r LLM-FineTune-Suite root@YOUR_POD_IP:/workspace/
```

### Step 4: Install dependencies on RunPod

```bash
pip install -r requirements.txt
```

Then login:

```bash
huggingface-cli login
```

Paste your Hugging Face write token.

### Step 5: Verify GPU

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

Expected:

```text
NVIDIA A40
about 48 GB
```

### Step 6: Train tool-calling model

```bash
python scripts/train_toolcall.py --config configs/toolcall_config.yaml
```

Expected:

- Downloads the base model and dataset
- Trains about 500 steps
- Saves merged model to `toolcall-output/merged`
- Pushes to Hugging Face if `hf_repo` is configured
- Pushes GGUF if enabled

### Step 7: Train finance model

```bash
python scripts/train_finance.py --config configs/finance_config.yaml
```

Expected:

- Downloads finance datasets
- Formats examples
- Trains about 500 steps
- Saves merged model to `finance-output/merged`
- Pushes to Hugging Face if configured

### Step 8: Train resume model

Only run this after you have finalized `data/resume_dataset.jsonl`.

```bash
python scripts/train_resume.py --config configs/resume_config.yaml
```

Expected:

- Loads local JSONL data
- Trains 300-500 steps depending on dataset size
- Saves merged model to `resume-output/merged`
- Pushes to Hugging Face if configured

### Step 9: Test models

Tool-calling:

```bash
python scripts/test_model.py \
  --config configs/toolcall_config.yaml \
  --repo YOUR_HF_USERNAME/krishna-toolcall-7b \
  --prompt "What's the weather in Boston today?"
```

Finance:

```bash
python scripts/test_model.py \
  --config configs/finance_config.yaml \
  --repo YOUR_HF_USERNAME/krishna-finance-7b \
  --system "You are a financial analyst." \
  --prompt "A company has revenue of $120M and gross profit of $45M. What is gross margin?"
```

Resume:

```bash
python scripts/test_model.py \
  --config configs/resume_config.yaml \
  --repo YOUR_HF_USERNAME/krishna-resumatch-7b \
  --system "You are a resume optimization expert." \
  --prompt "Given this job description, generate a tailored 1-line resume bio and 6 technical skill headers with relevant skills for each. Job Description: AI Engineer requiring LangChain, RAG, FastAPI, Docker, PostgreSQL, and OpenAI API."
```

### Step 10: Update Hugging Face model cards

Use the template in `README.md` and adjust per model.

### Step 11: Stop billing

After everything is pushed:

```text
Stop pod
Delete pod if done
Delete network volume if done
```

## Recommended Training Order

Train in this order:

1. Tool-calling model
2. Finance model
3. Resume model

Reason:

- Tool-calling and finance use public datasets and validate the training setup.
- Resume depends on your custom data quality, so it should run after the pipeline is confirmed.

## Current Risks And Notes

### Resume dataset quality is the biggest risk

With only 10 examples, the resume model will overfit and may not generalize.

You should add more examples before serious training.

### Dataset schema changes can happen

The finance script is defensive and checks several possible column names, but public dataset schemas can change. If a dataset fails on RunPod, capture the error and inspect columns with:

```python
from datasets import load_dataset
ds = load_dataset("TheFinAI/flare-finqa", split="train")
print(ds.column_names)
print(ds[0])
```

### Full merged model uploads can take time

Merged 16-bit 7B models are large. Upload time can be longer than training.

If upload time or storage becomes a problem, change the config:

```yaml
hub:
  push_merged: false
  push_gguf: true
```

or save adapters only in a future script revision.

### Do not forget to stop RunPod

The easiest way to waste budget is leaving the pod running after training.

## Quick Checklist

Before RunPod:

- [ ] Provide Hugging Face username
- [ ] Confirm final model repo names
- [ ] Add more resume JSONL examples
- [ ] Commit or upload this repo

On RunPod:

- [ ] Clone or upload repo
- [ ] Install requirements
- [ ] Login to Hugging Face
- [ ] Verify GPU
- [ ] Run tool-calling training
- [ ] Run finance training
- [ ] Run resume training
- [ ] Test each pushed model
- [ ] Update model cards
- [ ] Stop or delete RunPod resources

## Exact Info I Need From You Now

Reply with:

```text
HF username:
<your username>

Repo names:
toolcall: krishna-toolcall-7b
finance: krishna-finance-7b
resume: krishna-resumatch-7b

Resume examples:
<paste examples, or say "I will add later">

Target runtime:
<Ollama / llama.cpp / Transformers / vLLM / not sure>

Public or private HF repos:
<public / private>
```

With just the Hugging Face username, the configs can be finalized. With resume examples too, the resume dataset can be made useful before training.
