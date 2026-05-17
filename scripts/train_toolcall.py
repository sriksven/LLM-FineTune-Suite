import argparse
import inspect
import json
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


def value_to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def resolve_bf16(value):
    if value == "auto":
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return bool(value)


def build_sft_config(config: dict, output_dir: str, max_seq_length: int) -> SFTConfig:
    training = dict(config["training"])
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


def format_example(example):
    system = value_to_text(
        example.get("system") or "You are a helpful assistant with access to tools."
    )
    chat = value_to_text(example.get("chat"))
    text = f"<|im_start|>system\n{system}<|im_end|>\n{chat}"
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
    parser.add_argument("--config", default="configs/toolcall_config.yaml")
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

    dataset_config = config["dataset"]
    dataset = load_dataset(dataset_config["name"], split=dataset_config["split"])
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    sample_size = dataset_config.get("sample_size")
    if sample_size:
        dataset = dataset.shuffle(seed=dataset_config.get("seed", 42)).select(
            range(min(sample_size, len(dataset)))
        )

    print(f"Training on {len(dataset)} examples")
    print(f"Sample:\n{dataset[0]['text'][:500]}")

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        args=build_sft_config(config, config["output_dir"], config["max_seq_length"]),
    )

    print("Starting training...")
    trainer.train()
    maybe_push(model, tokenizer, config)


if __name__ == "__main__":
    main()
