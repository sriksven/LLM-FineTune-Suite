import argparse

import torch
import yaml
from unsloth import FastLanguageModel


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/toolcall_config.yaml")
    parser.add_argument("--repo", default=None)
    parser.add_argument("--prompt", default="What's the weather in Boston today?")
    parser.add_argument(
        "--system",
        default="You are a helpful assistant with access to tools.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    config = load_config(args.config)
    repo = args.repo or config["hf_repo"]
    if repo.startswith("YOUR_HF_USERNAME/"):
        raise ValueError("Set hf_repo in the config or pass --repo.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=repo,
        max_seq_length=config.get("max_seq_length", 2048),
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    generation_kwargs = {
        "input_ids": inputs,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.temperature > 0,
    }
    if torch.cuda.is_available():
        generation_kwargs["use_cache"] = True

    outputs = model.generate(**generation_kwargs)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
