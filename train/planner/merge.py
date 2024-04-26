from pathlib import Path

import jsonargparse
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.chat_template = None

    model = AutoPeftModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, default=Path("gba-planner-7B-v0.1"))    
    parser.add_argument("--output_dir", type=Path, default=Path("gba-planner-7B-v0.1-merged"))

    main(parser.parse_args())
