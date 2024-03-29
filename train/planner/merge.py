from pathlib import Path

import jsonargparse
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    tokenizer.chat_template = None

    model = AutoPeftModelForCausalLM.from_pretrained(args.checkpoint_dir, torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("project-1", "checkpoint-999"))
    parser.add_argument("--output_dir", type=Path, default=Path("project-1", "checkpoint-999_merged"))

    main(parser.parse_args())
