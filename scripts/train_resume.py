import argparse
import inspect
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_bf16(value):
    if value == "auto":
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return bool(value)


def build_sft_config(config: dict, output_dir: str, max_seq_length: int, steps: int):
    training = dict(config["training"])
    training.pop("min_steps", None)
    training.pop("steps_per_example", None)
    training["max_steps"] = steps
    bf16 = resolve_bf16(training.pop("bf16", "auto"))
    params = {
        **training,
        "bf16": bf16,
        "fp16": not bf16,
        "output_dir": output_dir,
        "report_to": "none",
        "dataset_text_field": "text",
    }

    fields = getattr(SFTConfig, "__dataclass_fields__", {})
    if "max_length" in fields:
        params["max_length"] = max_seq_length
    else:
        params["max_seq_length"] = max_seq_length

    return SFTConfig(**params)


def build_trainer(model, tokenizer, dataset, args):
    kwargs = {"model": model, "train_dataset": dataset, "args": args}
    signature = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return SFTTrainer(**kwargs)


def format_resume(example):
    text = (
        "<|im_start|>system\n"
        "You are a resume optimization expert. Given a job description, generate "
        "a tailored 1-line bio and exactly 6 technical skill headers with relevant "
        "skills. Bio must mention $1.5M USD impact. Skills must be purely technical "
        "with no soft skills.<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    )
    return {"text": text}


def maybe_push(model, tokenizer, config: dict):
    output_dir = Path(config["output_dir"])
    hf_repo = config["hf_repo"]
    hub = config.get("hub", {})

    model.save_pretrained_merged(
        str(output_dir / "merged"),
        tokenizer,
        save_method="merged_16bit",
    )

    if hf_repo.startswith("YOUR_HF_USERNAME/"):
        print("Skipping Hub push because hf_repo still uses YOUR_HF_USERNAME.")
        return

    if hub.get("push_merged", True):
        model.push_to_hub_merged(
            hf_repo,
            tokenizer,
            save_method="merged_16bit",
            token=True,
        )

    if hub.get("push_gguf", True):
        model.push_to_hub_gguf(
            hf_repo,
            tokenizer,
            quantization_method=hub.get("gguf_quantization", "q4_k_m"),
            token=True,
        )

    print(f"Model pushed to https://huggingface.co/{hf_repo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resume_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )

    lora = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        target_modules=TARGET_MODULES,
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=lora["random_state"],
        max_seq_length=config["max_seq_length"],
    )

    dataset = load_dataset("json", data_files=config["dataset_path"], split="train")
    dataset = dataset.map(format_resume, remove_columns=dataset.column_names)

    training = config["training"]
    steps = max(training["min_steps"], len(dataset) * training["steps_per_example"])
    steps = min(steps, training["max_steps"])

    print(f"Training on {len(dataset)} examples for {steps} steps")
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        args=build_sft_config(
            config,
            config["output_dir"],
            config["max_seq_length"],
            steps,
        ),
    )

    print("Starting training...")
    trainer.train()
    maybe_push(model, tokenizer, config)


if __name__ == "__main__":
    main()
